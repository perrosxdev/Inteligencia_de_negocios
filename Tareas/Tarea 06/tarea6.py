
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc,
                             accuracy_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE RUTAS 
# ============================================================
import os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r'C:\Users\macka\Desktop\Inteligencia de Negocios\Tarea 06\dataset-uci.xlsx'
OUTPUT_DIR   = r'C:\Users\macka\Desktop\Inteligencia de Negocios\Tarea 06' + os.sep

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

COLORS = {
    'primary': '#2563EB',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'danger': '#EF4444',
    'purple': '#8B5CF6',
    'gray': '#6B7280',
    'bg': '#F8FAFC',
}

# ============================================================
# FASE 2: COMPRENSIÓN DE LOS DATOS
# ============================================================
print("=" * 65)
print("FASE 2: COMPRENSIÓN DE LOS DATOS")
print("=" * 65)

df = pd.read_excel(DATASET_PATH, sheet_name='Dataset')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print(f"\nDimensiones del dataset: {df.shape[0]} instancias × {df.shape[1]} variables")
print(f"\nVariable objetivo: 'Case Type'")
print(f"  0 = Caso Esporádico  → {(df['Case Type'] == 0).sum()} instancias ({(df['Case Type']==0).mean()*100:.1f}%)")
print(f"  1 = Caso Familiar    → {(df['Case Type'] == 1).sum()} instancias ({(df['Case Type']==1).mean()*100:.1f}%)")

print(f"\nValores faltantes por variable:")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, n in missing.items():
    print(f"  {col}: {n} ({n/len(df)*100:.1f}%)")

# ============================================================
# FASE 3: PREPARACIÓN DE LOS DATOS
# ============================================================
print("\n" + "=" * 65)
print("FASE 3: PREPARACIÓN DE LOS DATOS")
print("=" * 65)

target = 'Case Type'
features = [c for c in df.columns if c != target]

X = df[features].copy()
y = df[target].copy()

# Imputación por mediana (variables continuas con NaN)
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=features)

print(f"\nImputación con mediana aplicada a: Age of Mother, Age of Father, Age at First Diagnosis")
print(f"Registros finales para modelado: {len(X_imp)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nDivisión Train/Test: {len(X_train)} / {len(X_test)} (75% / 25%)")

# ============================================================
# FASE 4: MODELADO
# ============================================================
print("\n" + "=" * 65)
print("FASE 4: MODELADO - ÁRBOL DE DECISIÓN")
print("=" * 65)

# Árbol sin restricciones
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

# Árbol podado (max_depth=4 para interpretabilidad)
dt_pruned = DecisionTreeClassifier(max_depth=4, min_samples_split=10,
                                    min_samples_leaf=5, random_state=42)
dt_pruned.fit(X_train, y_train)

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_full   = cross_val_score(dt_full,   X_imp, y, cv=cv, scoring='accuracy')
cv_scores_pruned = cross_val_score(dt_pruned, X_imp, y, cv=cv, scoring='accuracy')

print(f"\nÁrbol completo  - Precisión CV (5-fold): {cv_scores_full.mean():.4f} ± {cv_scores_full.std():.4f}")
print(f"Árbol podado    - Precisión CV (5-fold): {cv_scores_pruned.mean():.4f} ± {cv_scores_pruned.std():.4f}")

acc_test = accuracy_score(y_test, dt_pruned.predict(X_test))
print(f"Árbol podado    - Precisión en Test:     {acc_test:.4f}")

print("\nReporte de clasificación (Árbol podado - Test set):")
print(classification_report(y_test, dt_pruned.predict(X_test),
      target_names=['Esporádico (0)', 'Familiar (1)']))

# ============================================================
# VISUALIZACIONES
# ============================================================
print("\n" + "=" * 65)
print("GENERANDO VISUALIZACIONES")
print("=" * 65)

# ── FIG 1: Distribución de clase + valores faltantes ──────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLORS['bg'])
fig.suptitle('Comprensión del Dataset (Fase 2 CRISP-DM)',
             fontsize=14, fontweight='bold', y=1.01)

# 1a. Distribución de clase
ax = axes[0]
counts = y.value_counts().sort_index()
bars = ax.bar(['Esporádico (0)', 'Familiar (1)'], counts.values,
               color=[COLORS['primary'], COLORS['secondary']],
               edgecolor='white', linewidth=1.5, width=0.55)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val}\n({val/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_title('Distribución de la Variable Objetivo', fontweight='bold')
ax.set_ylabel('Número de casos')
ax.set_ylim(0, max(counts.values) * 1.18)
ax.set_facecolor(COLORS['bg'])
ax.spines[['top', 'right']].set_visible(False)

# 1b. Valores faltantes
ax2 = axes[1]
miss_pct = df.isnull().mean() * 100
miss_pct = miss_pct[miss_pct > 0]
bars2 = ax2.barh(miss_pct.index, miss_pct.values,
                  color=COLORS['accent'], edgecolor='white', linewidth=1.2)
for bar, val in zip(bars2, miss_pct.values):
    ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=10)
ax2.set_title('Porcentaje de Valores Faltantes por Variable', fontweight='bold')
ax2.set_xlabel('% Valores faltantes')
ax2.set_xlim(0, 20)
ax2.set_facecolor(COLORS['bg'])
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'comprension_datos.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 2: Heatmap de correlación ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
fig.suptitle('Mapa de Calor de Correlaciones entre Variables',
             fontsize=14, fontweight='bold')

corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8},
            vmin=-1, vmax=1)
ax.set_title('', pad=10)
plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'correlacion.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 3: Prevalencia síntomas por tipo de caso ──────────────
fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
fig.suptitle('Prevalencia de Síntomas por Tipo de Caso (Fase 2 CRISP-DM)',
             fontsize=14, fontweight='bold')

binary_features = [c for c in features if df[c].nunique() == 2 and c != 'Tumour Case']
prev = df.groupby('Case Type')[binary_features].mean().T
prev.columns = ['Esporádico', 'Familiar']
prev = prev.sort_values('Familiar', ascending=True)

x = np.arange(len(prev))
w = 0.38
ax.barh(x - w/2, prev['Esporádico'], height=w, color=COLORS['primary'],
         label='Esporádico (0)', alpha=0.88)
ax.barh(x + w/2, prev['Familiar'], height=w, color=COLORS['secondary'],
         label='Familiar (1)', alpha=0.88)
ax.set_yticks(x)
ax.set_yticklabels(prev.index, fontsize=9)
ax.set_xlabel('Prevalencia (proporción)')
ax.set_title('', pad=8)
ax.legend(fontsize=10)
ax.set_facecolor(COLORS['bg'])
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'prevalencia_sintomas.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 4: Árbol de decisión podado ───────────────────────────
fig, ax = plt.subplots(figsize=(22, 9), facecolor='white')
fig.suptitle('Árbol de Decisión Podado (max_depth=4)',
             fontsize=15, fontweight='bold')
plot_tree(dt_pruned, feature_names=features,
          class_names=['Esporádico', 'Familiar'],
          filled=True, rounded=True, fontsize=8,
          impurity=True, proportion=False, ax=ax,
          precision=2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'arbol_decision.png', bbox_inches='tight',
            facecolor='white', dpi=130)
plt.close()

# ── FIG 5: Importancia de características ─────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
fig.suptitle('Importancia de Características (Árbol Podado)',
             fontsize=14, fontweight='bold')

importances = pd.Series(dt_pruned.feature_importances_, index=features)
importances = importances[importances > 0].sort_values(ascending=True)
palette = [COLORS['primary'] if v < importances.max() * 0.5 else COLORS['secondary']
           for v in importances.values]
bars = ax.barh(importances.index, importances.values, color=palette,
               edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Importancia (Gini)')
ax.set_facecolor(COLORS['bg'])
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'importancia_features.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 6: Matriz de confusión + Curva ROC ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLORS['bg'])
fig.suptitle('Evaluación del Modelo (Árbol Podado, Test Set)',
             fontsize=14, fontweight='bold')

# Conf matrix
y_pred = dt_pruned.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Esporádico', 'Familiar'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Matriz de Confusión', fontweight='bold')
axes[0].set_facecolor(COLORS['bg'])

# ROC
y_prob = dt_pruned.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color=COLORS['primary'], lw=2.5,
              label=f'AUC = {roc_auc_val:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Clasificador aleatorio')
axes[1].fill_between(fpr, tpr, alpha=0.08, color=COLORS['primary'])
axes[1].set_xlabel('Tasa de Falsos Positivos')
axes[1].set_ylabel('Tasa de Verdaderos Positivos')
axes[1].set_title('Curva ROC', fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_facecolor(COLORS['bg'])
axes[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'evaluacion_modelo.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 7: Profundidad vs Precisión (análisis de sobreajuste) ─
fig, ax = plt.subplots(figsize=(10, 5), facecolor=COLORS['bg'])
fig.suptitle('Análisis de Profundidad del Árbol vs. Precisión',
             fontsize=14, fontweight='bold')

depths = range(1, 15)
train_acc, test_acc, cv_acc = [], [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train)))
    test_acc.append(accuracy_score(y_test, m.predict(X_test)))
    cv_acc.append(cross_val_score(m, X_imp, y, cv=5, scoring='accuracy').mean())

ax.plot(list(depths), train_acc, 'o-', color=COLORS['primary'],
         linewidth=2, markersize=5, label='Train')
ax.plot(list(depths), test_acc,  's-', color=COLORS['secondary'],
         linewidth=2, markersize=5, label='Test')
ax.plot(list(depths), cv_acc,    '^--', color=COLORS['accent'],
         linewidth=2, markersize=5, label='CV 5-fold')
ax.axvline(x=4, color=COLORS['danger'], linestyle='--', lw=1.5,
            label='Profundidad seleccionada (4)')
ax.set_xlabel('Profundidad máxima del árbol')
ax.set_ylabel('Precisión (Accuracy)')
ax.legend(fontsize=9)
ax.set_facecolor(COLORS['bg'])
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'profundidad_vs_precision.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ── FIG 8: Distribución variables continuas por clase ─────────
cont_vars = ['Age of Mother', 'Age of Father', 'Age at First Diagnosis']
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=COLORS['bg'])
fig.suptitle('Distribución de Variables Continuas por Tipo de Caso',
             fontsize=14, fontweight='bold')

for ax, var in zip(axes, cont_vars):
    data0 = df[df['Case Type'] == 0][var].dropna()
    data1 = df[df['Case Type'] == 1][var].dropna()
    ax.hist(data0, bins=18, alpha=0.65, color=COLORS['primary'],
             label=f'Esporádico (μ={data0.mean():.1f})', density=True)
    ax.hist(data1, bins=18, alpha=0.65, color=COLORS['secondary'],
             label=f'Familiar (μ={data1.mean():.1f})', density=True)
    ax.set_title(var, fontweight='bold')
    ax.set_xlabel('Años')
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=8)
    ax.set_facecolor(COLORS['bg'])
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
fig.savefig(OUTPUT_DIR + 'variables_continuas.png', bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()

# ============================================================
# FASE 5: EVALUACIÓN — RESUMEN
# ============================================================
print("\n" + "=" * 65)
print("FASE 5: EVALUACIÓN — RESUMEN DE MÉTRICAS")
print("=" * 65)
print(f"\n  Accuracy en test set      : {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"  AUC-ROC                   : {roc_auc_val:.4f}")
print(f"  Accuracy CV (5-fold)      : {cv_scores_pruned.mean():.4f} ± {cv_scores_pruned.std():.4f}")
print(f"  Nodos del árbol podado    : {dt_pruned.tree_.node_count}")
print(f"  Profundidad real          : {dt_pruned.get_depth()}")

top3 = importances.nlargest(3)
print(f"\n  Top 3 características más relevantes:")
for feat, imp in top3.items():
    print(f"    - {feat}: {imp:.4f}")


