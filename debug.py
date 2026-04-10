# debug.py
import pandas as pd
import numpy as np
import joblib

print("=== DIAGNOSTIC COMPLET ===\n")

model  = joblib.load('outputs/best_model_tuned.pkl')
scaler = joblib.load('outputs/scaler.pkl')
X_test = pd.read_csv('outputs/X_test.csv')
y_test = pd.read_csv('outputs/y_test.csv').squeeze()

# ── Test 1 : Le modèle fonctionne-t-il sur le vrai test set ?
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Test set — prob min  : {y_prob.min():.6f}")
print(f"Test set — prob max  : {y_prob.max():.6f}")
print(f"Test set — prob mean : {y_prob.mean():.6f}")
print(f"Prédictions > 0.5    : {(y_prob > 0.5).sum()}")
print(f"Vrais positifs réels : {y_test.sum()}\n")

# ── Test 2 : Vérifier si X_test est normalisé
print(f"X_test valeurs (doit être entre -3 et +3 si normalisé) :")
print(X_test.describe().T[['mean','std','min','max']].head(5).round(2))
print()

# ── Test 3 : Simuler un patient à risque (depuis le vrai test set)
idx_pos = y_test[y_test == 1].index[0]
pos_in_xtest = X_test.index.get_loc(idx_pos) if idx_pos in X_test.index else 0
patient_reel = X_test.iloc[pos_in_xtest]
prob_reel = float(model.predict_proba(patient_reel.values.reshape(1,-1))[0][1])
print(f"Patient réel (Biopsy=1) → prob prédite : {prob_reel:.6f}")

# ── Test 4 : Simuler ce que le dashboard fait (vecteur médiane)
iv = X_test.median()
prob_median = float(model.predict_proba(iv.values.reshape(1,-1))[0][1])
print(f"Vecteur médiane       → prob prédite : {prob_median:.6f}  ← BUG si = 0")

# ── Test 5 : Vecteur avec valeurs brutes non normalisées (bug actuel)
raw_wrong = X_test.median().copy()
raw_wrong['Age'] = 71          # valeur brute dans un vecteur normalisé
raw_wrong['Number of sexual partners'] = 23
prob_wrong = float(model.predict_proba(raw_wrong.values.reshape(1,-1))[0][1])
print(f"Vecteur mixte (bug)   → prob prédite : {prob_wrong:.6f}  ← doit être bizarre")

print("\n=== FIN DIAGNOSTIC ===")