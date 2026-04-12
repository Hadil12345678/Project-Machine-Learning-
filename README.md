# 🔬 Cervical Cancer Risk Prediction

**Hadil Dhaya · 4th Year Data Science · Group 5 · 2026**

---

## 📁 Structure du projet

```
## 🔧 Version
cervical_cancer_ML/
│
├── data/
│   └── risk_factors_cervical_cancer.csv   ← place ton CSV ici
│
├── notebooks/
│   ├── 01_EDA.ipynb                        ← Analyse exploratoire
│   ├── 02_Preprocessing.ipynb             ← Nettoyage + Feature Eng + SMOTE
│   ├── 03_Experiments.ipynb               ← Essayage des 7 modèles
│   └── 04_Final_Notebook.ipynb            ← Version finale + SHAP
│
├── outputs/                               ← Généré automatiquement
│   ├── X_train.csv / X_test.csv
│   ├── y_train.csv / y_test.csv
│   ├── scaler.pkl / imputer.pkl
│   ├── best_model_tuned.pkl
│   ├── best_model_name.pkl
│   ├── all_models.pkl
│   └── *.png  (graphiques)
│
├── app.py                                 ← Dashboard Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 Commandes à exécuter dans l'ordre

### Étape 1 — Installation

```bash
pip install -r requirements.txt
```

### Étape 2 — Placer le dataset

Télécharge le CSV depuis :
https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

Place-le dans : `data/risk_factors_cervical_cancer.csv`

### Étape 3 — Ouvrir VS Code

```bash
code .
```

### Étape 4 — Lancer les notebooks dans l'ordre

Dans VS Code, ouvre chaque notebook et clique **"Run All"** :

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Preprocessing.ipynb`
3. `notebooks/03_Experiments.ipynb`
4. `notebooks/04_Final_Notebook.ipynb`

### Étape 5 — Lancer le dashboard

```bash
streamlit run app.py
```

→ Ouvre automatiquement : http://localhost:8501

### API FastAPI (optionnel)

Depuis la racine du projet :

```bash
py -m uvicorn main:app --reload
```

ou :

```bash
py -m uvicorn backend.main:app --reload
```

→ http://127.0.0.1:8000

La base SQLite est créée automatiquement dans **`backend/cervical.db`** (pas à la racine du projet).

**Si Windows affiche l’erreur 10048** (« le port est déjà utilisé ») : une autre instance tourne encore sur le port 8000. Soit tu fermes l’autre terminal (Ctrl+C), soit tu utilises un autre port :

```bash
py -m uvicorn backend.main:app --reload --port 8001
```

---

## 📊 Pipeline ML

| Étape         | Contenu                                                                        |
| ------------- | ------------------------------------------------------------------------------ |
| EDA           | Distributions, corrélations, valeurs manquantes, déséquilibre                  |
| Preprocessing | Suppression NaN > 80%, KNN Imputer, Feature Engineering, SMOTE, StandardScaler |
| Experiments   | 7 modèles : LR, KNN, RF, SVM, GB, AdaBoost, MLP                                |
| Final         | Meilleur modèle + Hyperparameter Tuning + SHAP complet                         |
| Dashboard     | Interface Streamlit — prédiction temps réel                                    |

---

## 🤖 Modèles testés (notebook 03)

- Logistic Regression
- K-Nearest Neighbors
- **Random Forest** ← souvent le meilleur
- SVM (RBF kernel)
- Gradient Boosting
- AdaBoost
- MLP Neural Network

---

## ⚕️ Note médicale

> Dans le dépistage du cancer, un **faux négatif** (cancer manqué) est bien plus grave
> qu'un faux positif. Le modèle est calibré pour maximiser le **Recall**.
> Usage académique uniquement — ne remplace pas un avis médical.
