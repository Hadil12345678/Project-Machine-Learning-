import json
from pathlib import Path

import joblib
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR.parent / "outputs"

FINAL_PIPELINE_PATH = OUTPUTS_DIR / "final_pipeline.pkl"
OLD_MODEL_PATH = OUTPUTS_DIR / "best_model_smote.pkl"

FINAL_METADATA_PATH = OUTPUTS_DIR / "final_metadata.json"
FINAL_FEATURES_PATH = OUTPUTS_DIR / "final_feature_list.pkl"
COLUMNS_PATH = OUTPUTS_DIR / "columns.pkl"


def _load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load metadata
metadata = _load_json(FINAL_METADATA_PATH)
THRESHOLD = float(metadata.get("threshold", 0.5))
RAW_MEDIANS = metadata.get("raw_feature_medians", {})

# Load model
if FINAL_PIPELINE_PATH.exists():
    model = joblib.load(FINAL_PIPELINE_PATH)
    print("✅ Loaded model: final_pipeline.pkl")
elif OLD_MODEL_PATH.exists():
    model = joblib.load(OLD_MODEL_PATH)
    print("✅ Loaded model: best_model_smote.pkl")
else:
    raise FileNotFoundError("No model file found in outputs/")

# Load trained columns / final features
if metadata.get("final_raw_features"):
    trained_columns = metadata["final_raw_features"]
elif FINAL_FEATURES_PATH.exists():
    trained_columns = joblib.load(FINAL_FEATURES_PATH)
elif hasattr(model, "feature_names_in_"):
    trained_columns = list(model.feature_names_in_)
elif COLUMNS_PATH.exists():
    trained_columns = joblib.load(COLUMNS_PATH)
else:
    trained_columns = []

if not trained_columns:
    raise ValueError("No trained columns found. Check outputs/final_metadata.json or columns artifacts.")

print("✅ Columns:", trained_columns)
print("✅ Threshold:", THRESHOLD)


def predict(data: dict):
    try:
        # Step 1: create full row with all expected features
        row = {col: float(RAW_MEDIANS.get(col, 0.0)) for col in trained_columns}

        # Step 2: fill known inputs
        # Supports both:
        # - backend/schema names with underscores
        # - old names with spaces
        mapping = {
            "Age": ["Age"],
            "Number of sexual partners": ["Number_of_sexual_partners", "Number of sexual partners"],
            "First sexual intercourse": ["First_sexual_intercourse", "First sexual intercourse"],
            "Num of pregnancies": ["Num_of_pregnancies", "Num of pregnancies"],
            "Smokes": ["Smokes"],
            "Smokes (years)": ["Smokes_years", "Smokes (years)"],
            "Smokes (packs/year)": ["Smokes_packs_per_year", "Smokes (packs/year)"],
            "Hormonal Contraceptives": ["Hormonal_Contraceptives", "Hormonal Contraceptives"],
            "Hormonal Contraceptives (years)": ["Hormonal_Contraceptives_years", "Hormonal Contraceptives (years)"],
            "IUD": ["IUD"],
            "IUD (years)": ["IUD_years", "IUD (years)"],
            "STDs": ["STDs"],
            "STDs (number)": ["STDs_number", "STDs (number)"],
        }

        for final_col, possible_keys in mapping.items():
            if final_col in row:
                for key in possible_keys:
                    if key in data and data[key] is not None:
                        row[final_col] = float(data[key])
                        break

        # Step 3: engineered features only if they exist in trained columns
        if "age_first_sex_gap" in row:
            row["age_first_sex_gap"] = row.get("Age", 0.0) - row.get("First sexual intercourse", 0.0)

        if "smoke_exposure" in row:
            row["smoke_exposure"] = row.get("Smokes", 0.0) * row.get("Smokes (years)", 0.0)

        if "stds_score" in row:
            row["stds_score"] = row.get("STDs", 0.0) * row.get("STDs (number)", 0.0)

        # Step 4: DataFrame with correct order
        X = pd.DataFrame([row]).reindex(columns=trained_columns, fill_value=0.0)

        print("✅ FINAL INPUT:")
        print(X)

        # Step 5: predict
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
            pred = 1 if proba >= THRESHOLD else 0
        else:
            pred = int(model.predict(X)[0])
            proba = float(pred)

        return {
            "result": "Cancer" if pred == 1 else "No Cancer",
            "confidence": round(proba, 6),
        }

    except Exception as e:
        import traceback
        print("🔥 PREDICT ERROR:")
        traceback.print_exc()
        raise