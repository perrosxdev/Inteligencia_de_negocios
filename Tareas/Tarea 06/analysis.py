import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay, classification_report,
                              roc_curve, auc)

plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})
sns.set_style("whitegrid")

RANDOM_STATE = 42
OUT = Path("figs")
OUT.mkdir(exist_ok=True)


# CARGA DE DATOS

df = pd.read_excel("dataset-uci.xlsx", sheet_name="Dataset")
df = df.drop(columns=["Unnamed: 0"])
print("Dimensiones:", df.shape)
print(df.columns.tolist())

TARGET = "Case Type"
EDAD_COLS = ["Age of Mother", "Age of Father", "Age at First Diagnosis"]
SYMPTOM_COLS = [c for c in df.columns if c not in EDAD_COLS + [TARGET, "Tumour Case"]]

# Comprension de los datos


# Valores faltantes
miss = df.isna().sum()
miss = miss[miss > 0]
print("\nValores faltantes:\n", miss)

# Distribucion de la clase objetivo
counts = df[TARGET].value_counts().sort_index()
labels = ["Esporadico (0)", "Familiar (1)"]

# Prevalencia de sintomas binarios
bin_cols = [c for c in df.columns if c not in EDAD_COLS + [TARGET]]
prevalencia = df[bin_cols].mean().sort_values(ascending=True) * 100

# Figura compuesta: panorama del EDA (objetivo + faltantes + prevalencia)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2),
                          gridspec_kw={"width_ratios": [1, 1, 1.6]})

axes[0].bar(labels, counts.values, color=["#55A868", "#C44E52"])
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha="center")
axes[0].set_ylabel("N° de casos")
axes[0].set_title("(a) Variable objetivo")

miss.sort_values().plot(kind="barh", color="#4C72B0", ax=axes[1])
axes[1].set_xlabel("N° de valores faltantes")
axes[1].set_title("(b) Valores faltantes")

prevalencia.plot(kind="barh", color="#4C72B0", ax=axes[2])
axes[2].set_xlabel("Prevalencia (%)")
axes[2].set_title("(c) Prevalencia de sintomas")

plt.tight_layout()
plt.savefig(OUT / "eda_overview.png", bbox_inches="tight")
plt.close()

# Matriz de correlacion
corr = df.corr(numeric_only=True)
plt.figure(figsize=(9, 7.5))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
            cbar_kws={"shrink": 0.7})
plt.title("Matriz de correlacion entre variables")
plt.tight_layout()
plt.savefig(OUT / "correlacion.png", bbox_inches="tight")
plt.close()


# 3. PREPARACION DE LOS DATOS

y = df[TARGET].copy()

cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
base_clf = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")

# ---- Comparacion de estrategias de imputacion ----------------
def build_X(strategy, add_features=False):
    """Devuelve X, y segun la estrategia de imputacion."""
    Xc = df.drop(columns=[TARGET]).copy()
    yc = y.copy()

    if strategy == "drop":
        mask = Xc[EDAD_COLS].notna().all(axis=1)
        Xc = Xc[mask].reset_index(drop=True)
        yc = yc[mask].reset_index(drop=True)
    elif strategy == "median":
        imp = SimpleImputer(strategy="median")
        Xc[EDAD_COLS] = imp.fit_transform(Xc[EDAD_COLS])
    elif strategy == "knn":
        imp = KNNImputer(n_neighbors=5)
        Xc[EDAD_COLS] = imp.fit_transform(Xc[EDAD_COLS])
    else:
        raise ValueError(strategy)

    if add_features:
        Xc["Symptom_Count"] = Xc[SYMPTOM_COLS].sum(axis=1)
        Xc["Parents_Age_Diff"] = (Xc["Age of Father"] - Xc["Age of Mother"]).abs()

    return Xc, yc

print("\n--- Comparacion de estrategias de imputacion (f1_macro, CV 5x10) ---")
imput_results = {}
for strat in ["median", "drop", "knn"]:
    Xc, yc = build_X(strat, add_features=False)
    scores = cross_val_score(base_clf, Xc, yc, cv=cv_rep, scoring="f1_macro")
    imput_results[strat] = (scores.mean(), scores.std())
    print(f"  {strat:8s}: f1_macro = {scores.mean():.3f} +/- {scores.std():.3f}  (n={len(yc)})")

best_strategy = max(imput_results, key=lambda k: imput_results[k][0])
print(f"Estrategia de imputacion seleccionada: '{best_strategy}'")

# ---- Ingenieria de caracteristicas ----------------------------
print("\n--- Efecto de variables derivadas (Symptom_Count, Parents_Age_Diff) ---")
feat_results = {}
for add_feat, label in [(False, "sin_features"), (True, "con_features")]:
    Xc, yc = build_X(best_strategy, add_features=add_feat)
    scores = cross_val_score(base_clf, Xc, yc, cv=cv_rep, scoring="f1_macro")
    feat_results[label] = (scores.mean(), scores.std())
    print(f"  {label:13s}: f1_macro = {scores.mean():.3f} +/- {scores.std():.3f}")

use_extra_features = feat_results["con_features"][0] >= feat_results["sin_features"][0]
print(f"Uso de variables derivadas: {use_extra_features}")

# Figura compuesta: comparacion de preparacion de datos
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

strats = list(imput_results.keys())
means = [imput_results[s][0] for s in strats]
stds = [imput_results[s][1] for s in strats]
axes[0].bar(strats, means, yerr=stds, color="#4C72B0", capsize=4)
axes[0].set_ylabel("F1-macro (CV 5x10)")
axes[0].set_title("(a) Estrategias de imputacion")

labels_f = list(feat_results.keys())
means_f = [feat_results[k][0] for k in labels_f]
stds_f = [feat_results[k][1] for k in labels_f]
axes[1].bar(["Sin variables\nderivadas", "Con variables\nderivadas"], means_f,
            yerr=stds_f, color=["#55A868", "#C44E52"], capsize=4)
axes[1].set_ylabel("F1-macro (CV 5x10)")
axes[1].set_title("(b) Variables derivadas")

plt.tight_layout()
plt.savefig(OUT / "comparacion_preparacion.png", bbox_inches="tight")
plt.close()

# Conjunto de datos final
X, y = build_X(best_strategy, add_features=use_extra_features)
print(f"\nDataset final: X={X.shape}, y={y.shape}")
print("Columnas finales:", X.columns.tolist())

# Particion entrenamiento / prueba (estratificada) - solo para
# visualizaciones puntuales (matriz de confusion, ROC, arbol)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)


# MODELADO - Arbol de Decision


# Busqueda de hiperparametros ampliada
param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.005, 0.01, 0.02],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), param_grid,
                     cv=cv5, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nMejores hiperparametros:", grid.best_params_)
print("Mejor score CV (f1_macro, train):", grid.best_score_)

# Curva de validacion (profundidad vs desempeño), demas hiperparametros fijos
fixed_params = {k: v for k, v in grid.best_params_.items() if k != "max_depth"}
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores, cv_scores = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE, **fixed_params)
    cv_acc = cross_val_score(clf, X_train, y_train, cv=cv5, scoring="accuracy")
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


# EVALUACION


# Evaluacion robusta: CV repetida sobre todo el dataset
final_clf = DecisionTreeClassifier(random_state=RANDOM_STATE, **grid.best_params_)

acc_cv = cross_val_score(final_clf, X, y, cv=cv_rep, scoring="accuracy")
f1_cv = cross_val_score(final_clf, X, y, cv=cv_rep, scoring="f1_macro")
prec_cv = cross_val_score(final_clf, X, y, cv=cv_rep, scoring="precision")
rec_cv = cross_val_score(final_clf, X, y, cv=cv_rep, scoring="recall")
auc_cv_scores = cross_val_score(final_clf, X, y, cv=cv_rep, scoring="roc_auc")

print("\n--- Evaluacion robusta (CV repetida 5x10, todo el dataset) ---")
print(f"Accuracy : {acc_cv.mean():.3f} +/- {acc_cv.std():.3f}")
print(f"F1-macro : {f1_cv.mean():.3f} +/- {f1_cv.std():.3f}")
print(f"Precision: {prec_cv.mean():.3f} +/- {prec_cv.std():.3f}")
print(f"Recall   : {rec_cv.mean():.3f} +/- {rec_cv.std():.3f}")
print(f"AUC      : {auc_cv_scores.mean():.3f} +/- {auc_cv_scores.std():.3f}")

# Evaluacion puntual (split de prueba) para matriz de confusion, ROC y arbol
best_tree_eval = DecisionTreeClassifier(random_state=RANDOM_STATE, **grid.best_params_)
best_tree_eval.fit(X_train, y_train)

y_pred = best_tree_eval.predict(X_test)
y_proba = best_tree_eval.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("\n--- Evaluacion puntual (split de prueba, 25%) ---")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC: {roc_auc:.3f}")
print("\n", classification_report(y_test, y_pred, target_names=["Esporadico", "Familiar"]))

# Figura compuesta: matriz de confusion + curva ROC
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

disp = ConfusionMatrixDisplay(cm, display_labels=["Esporadico", "Familiar"])
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("(a) Matriz de confusion (prueba)")

axes[1].plot(fpr, tpr, color="#C44E52", label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("Tasa de falsos positivos")
axes[1].set_ylabel("Tasa de verdaderos positivos")
axes[1].set_title("(b) Curva ROC")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT / "evaluacion_overview.png", bbox_inches="tight")
plt.close()

# Importancia de variables (modelo entrenado en todo X)
final_clf.fit(X, y)
importances = pd.Series(final_clf.feature_importances_, index=X.columns)
importances = importances[importances > 0].sort_values()

plt.figure(figsize=(6, 4.5))
importances.plot(kind="barh", color="#4C72B0")
plt.xlabel("Importancia")
plt.title("Importancia de variables - Arbol de Decision")
plt.tight_layout()
plt.savefig(OUT / "importancia_variables.png", bbox_inches="tight")
plt.close()

# Visualizacion del arbol (version reducida para legibilidad)
small_tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE,
                                     criterion=grid.best_params_["criterion"],
                                     class_weight=grid.best_params_["class_weight"])
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


# 6. RESUMEN FINAL PARA EL INFORME
print("\n===== RESUMEN PARA EL INFORME =====")
print("N instancias (dataset final):", X.shape[0])
print("N variables (sin target):", X.shape[1])
target_dist = y.value_counts().to_dict()
print("Distribucion target:", target_dist)
print("Estrategia de imputacion:", best_strategy)
print("Variables derivadas usadas:", use_extra_features)
print("Mejor configuracion:", grid.best_params_)
print(f"CV repetida -> Accuracy: {acc_cv.mean():.3f} +/- {acc_cv.std():.3f} | "
      f"F1-macro: {f1_cv.mean():.3f} +/- {f1_cv.std():.3f} | "
      f"Precision: {prec_cv.mean():.3f} +/- {prec_cv.std():.3f} | "
      f"Recall: {rec_cv.mean():.3f} +/- {rec_cv.std():.3f}")
print(f"Split prueba -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | "
      f"Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
print("Top variables importantes:")
print(importances.sort_values(ascending=False).head(8))
print("Listo")