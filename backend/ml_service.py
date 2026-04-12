import os
import joblib
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "outputs", "best_model_smote.pkl")
columns_path = os.path.join(BASE_DIR, "..", "outputs", "columns.pkl")

# Load model + columns
model = joblib.load(model_path)
trained_columns = joblib.load(columns_path)

print("✅ Model loaded")
print("✅ Columns:", trained_columns)


def predict(data: dict):
    try:
        import pandas as pd
        import joblib
        import os

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model = joblib.load(os.path.join(BASE_DIR, "..", "outputs", "best_model_smote.pkl"))
        columns = joblib.load(os.path.join(BASE_DIR, "..", "outputs", "columns.pkl"))

        print("✅ Using columns:", columns)

        # Step 1: create full row with ALL features
        row = {col: 0 for col in columns}

        # Step 2: fill known inputs
        mapping = {
            "Age": "Age",
            "Number of sexual partners": "Number of sexual partners",
            "First sexual intercourse": "First sexual intercourse",
            "Num of pregnancies": "Num of pregnancies",
            "Smokes": "Smokes",
            "Smokes (years)": "Smokes (years)",
            "Smokes (packs/year)": "Smokes (packs/year)",
            "Hormonal Contraceptives": "Hormonal Contraceptives",
            "Hormonal Contraceptives (years)": "Hormonal Contraceptives (years)",
            "IUD": "IUD",
            "IUD (years)": "IUD (years)",
            "STDs": "STDs",
            "STDs (number)": "STDs (number)",
        }

        for k, v in mapping.items():
            if k in data:
                row[v] = float(data[k])

        # Step 3: compute engineered features
        row["age_first_sex_gap"] = row["Age"] - row["First sexual intercourse"]
        row["smoke_exposure"] = row["Smokes"] * row["Smokes (years)"]
        row["stds_score"] = row["STDs"] * row["STDs (number)"]

        # Step 4: DataFrame with correct order
        X = pd.DataFrame([row])[columns]

        print("✅ FINAL INPUT:", X)

        # Step 5: predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        return {
            "result": "Cancer" if pred == 1 else "No Cancer",
            "confidence": float(proba)
        }

    except Exception as e:
        import traceback
        print("🔥 PREDICT ERROR:")
        traceback.print_exc()
        raise