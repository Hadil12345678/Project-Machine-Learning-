import joblib
import pandas as pd

pipeline = joblib.load("outputs/final_pipeline.pkl")

print("FEATURES:", pipeline.feature_names_in_)

# patient test (valeurs fortes)
data = {col: 0 for col in pipeline.feature_names_in_}

data["Age"] = 50
data["Smokes"] = 1
data["Smokes (years)"] = 20
data["Smokes (packs/year)"] = 10
data["STDs"] = 1
data["STDs (number)"] = 3

df = pd.DataFrame([data])

proba = pipeline.predict_proba(df)[0][1]
print("PROBA:", proba)