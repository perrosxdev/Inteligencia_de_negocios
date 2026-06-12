"""
Analisis CRISP-DM - Arboles de Decision
Dataset: Neurofibromatosis Type 1 - Clinical Symptoms (UCI, ID 1162)
Target: Case Type (0 = Esporadico, 1 = Familiar)
"""
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay, classification_report,
                              roc_curve, auc)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})
sns.set_style("whitegrid")

# Carpeta donde se guardarán las figuras
OUT = Path("figs")
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------------
df = pd.read_excel("dataset-uci.xlsx", sheet_name="Dataset")
df = df.drop(columns=["Unnamed: 0"])
print("Dimensiones:", df.shape)
print(df.columns.tolist())

TARGET = "Case Type"

# ------------------------------------------------------------------
# 2. EDA - Comprension de los datos
# ------------------------------------------------------------------

# 2.1 Valores faltantes
miss = df.isna().sum()
miss = miss[miss > 0]
print("\nValores faltantes:\n", miss)

plt.figure(figsize=(5, 3.2))
miss.sort_values().plot(kind="barh", color="#4C72B0")
plt.xlabel("N° de valores faltantes")
plt.title("Valores faltantes por variable")
plt.tight_layout()
plt.savefig(OUT / "missing_values.png", bbox_inches="tight")
plt.close()

# 2.2 Distribucion de la clase objetivo
plt.figure(figsize=(4, 3.2))
counts = df[TARGET].value_counts().sort_index()
labels = ["Esporadico (0)", "Familiar (1)"]
plt.bar(labels, counts.values, color=["#55A868", "#C44E52"])
for i, v in enumerate(counts.values):
    plt.text(i, v + 2, str(v), ha="center")
plt.ylabel("N° de casos")
plt.title("Distribucion de la variable objetivo (Case Type)")
plt.tight_layout()
plt.savefig(OUT / "target_distribution.png", bbox_inches="tight")
plt.close()

# 2.3 Distribucion variables de edad
edad_cols = ["Age of Mother", "Age of Father", "Age at First Diagnosis"]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, col in zip(axes, edad_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(OUT / "edad_distribuciones.png", bbox_inches="tight")
plt.close()

# 2.4 Prevalencia de sintomas binarios
bin_cols = [c for c in df.columns if c not in edad_cols + [TARGET]]
prevalencia = df[bin_cols].mean().sort_values(ascending=True) * 100

plt.figure(figsize=(6, 5))
prevalencia.plot(kind="barh", color="#4C72B0")
plt.xlabel("Prevalencia (%)")
plt.title("Prevalencia de sintomas y caracteristicas binarias")
plt.tight_layout()
plt.savefig(OUT / "prevalencia_sintomas.png", bbox_inches="tight")
plt.close()

# 2.5 Matriz de correlacion
corr = df.corr(numeric_only=True)
plt.figure(figsize=(9, 7.5))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
            cbar_kws={"shrink": 0.7})
plt.title("Matriz de correlacion entre variables")
plt.tight_layout()
plt.savefig(OUT / "correlacion.png", bbox_inches="tight")
plt.close()

# 2.6 Sintomas vs Case Type (algunas variables de interes)
interes = ["Tumour Case", "Café au lait (CLS)", "Axillary Freckles",
           "Plexiform Neurofibromins", "Optic Glioma", "Scoliosis"]
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
for ax, col in zip(axes.ravel(), interes):
    ct = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
    ct.columns = ["Esporadico", "Familiar"]
    ct.plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"], legend=False)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("% por categoria")
    ax.set_xticklabels(["No (0)", "Si (1)"], rotation=0)
handles, lbls = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "sintomas_vs_target.png", bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 3. PREPARACION DE LOS DATOS
# ------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Imputacion de valores faltantes (mediana) para las variables de edad
imputer = SimpleImputer(strategy="median")
X[edad_cols] = imputer.fit_transform(X[edad_cols])

print("\nValores faltantes tras imputacion:", X.isna().sum().sum())

# Particion entrenamiento / prueba (estratificada)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ------------------------------------------------------------------
# 4. MODELADO - Arbol de Decision
# ------------------------------------------------------------------

# 4.1 Busqueda de hiperparametros (validacion cruzada)
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "criterion": ["gini", "entropy"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                     cv=cv, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nMejores hiperparametros:", grid.best_params_)
print("Mejor score CV (f1_macro):", grid.best_score_)

best_tree = grid.best_estimator_

# Curva de validacion (profundidad vs desempeño)
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores, cv_scores = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    cv_scores.append(cv_acc.mean())

plt.figure(figsize=(5, 3.5))
plt.plot(depths, train_scores, marker="o", label="Entrenamiento")
plt.plot(depths, cv_scores, marker="s", label="Validacion cruzada (5-fold)")
plt.xlabel("Profundidad maxima del arbol")
plt.ylabel("Exactitud (accuracy)")
plt.title("Curva de validacion: profundidad vs exactitud")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "curva_validacion.png", bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------
# 5. EVALUACION
# ------------------------------------------------------------------
y_pred = best_tree.predict(X_test)
y_proba = best_tree.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"\nAccuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {roc_auc:.3f}")
print("\n", classification_report(y_test, y_pred, target_names=["Esporadico", "Familiar"]))

# 5.1 Matriz de confusion
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 3.5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Esporadico", "Familiar"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Matriz de confusion - conjunto de prueba")
plt.tight_layout()
plt.savefig(OUT / "matriz_confusion.png", bbox_inches="tight")
plt.close()

# 5.2 Curva ROC
plt.figure(figsize=(4, 3.5))
plt.plot(fpr, tpr, color="#C44E52", label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Tasa de falsos positivos")
plt.ylabel("Tasa de verdaderos positivos")
plt.title("Curva ROC")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", bbox_inches="tight")
plt.close()

# 5.3 Importancia de variables
importances = pd.Series(best_tree.feature_importances_, index=X.columns)
importances = importances[importances > 0].sort_values()

plt.figure(figsize=(6, 4.5))
importances.plot(kind="barh", color="#4C72B0")
plt.xlabel("Importancia (Gini)")
plt.title("Importancia de variables - Arbol de Decision")
plt.tight_layout()
plt.savefig(OUT / "importancia_variables.png", bbox_inches="tight")
plt.close()

# 5.4 Visualizacion del arbol (version reducida para legibilidad)
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42,
                                     criterion=grid.best_params_["criterion"])
small_tree.fit(X_train, y_train)
plt.figure(figsize=(16, 8))
plot_tree(small_tree, feature_names=X.columns, class_names=["Esporadico", "Familiar"],
          filled=True, rounded=True, fontsize=8, impurity=True)
plt.title("Arbol de Decision (profundidad=3) - visualizacion")
plt.tight_layout()
plt.savefig(OUT / "arbol_decision.png", bbox_inches="tight")
plt.close()

acc_small = small_tree.score(X_test, y_test)
print(f"\nAccuracy arbol depth=3 (visualizacion): {acc_small:.3f}")

# ------------------------------------------------------------------
# 6. RESUMEN FINAL PARA EL INFORME
# ------------------------------------------------------------------
print("\n===== RESUMEN PARA EL INFORME =====")
print("N instancias:", df.shape[0])
print("N variables (sin target):", X.shape[1])
target_dist = df[TARGET].value_counts().to_dict()
print("Distribucion target:", target_dist)
print("Mejor configuracion:", grid.best_params_)
print(f"Accuracy test: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
print("Top 5 variables importantes:")
print(importances.sort_values(ascending=False).head(5))
