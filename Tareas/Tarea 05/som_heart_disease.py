"""
=============================================================
  Mapas de Kohonen - Heart Disease UCI Dataset
  INFO1184 - Inteligencia de Negocios
=============================================================
  Cubre los 5 ítems del curso:
    9  - Inicialización de pesos de la red
    10 - Estimación de los pesos
    11 - Kernel / función de vecindad
    12 - Visualización
    13 - Densidades y centroides
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# ─── Carpeta de salida: misma carpeta que este script ────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

# ─── Reproducibilidad ────────────────────────────────────────
np.random.seed(42)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# ════════════════════════════════════════════════════════════
#  0. CARGA Y PREPROCESAMIENTO
# ════════════════════════════════════════════════════════════


df = pd.read_csv(os.path.join(OUTPUT_DIR, "heart.csv"))
print(f"[OK] Dataset cargado — {df.shape[0]} filas, {df.shape[1]} columnas")

# Etiquetas interpretables de columnas
col_labels = {
    "age":      "Edad",
    "sex":      "Sexo",
    "cp":       "Tipo dolor pecho",
    "trestbps": "Presión en reposo",
    "chol":     "Colesterol",
    "fbs":      "Glucosa en ayunas",
    "restecg":  "ECG en reposo",
    "thalach":  "FC máxima",
    "exang":    "Angina por ejercicio",
    "oldpeak":  "Depresión ST",
    "slope":    "Pendiente ST",
    "ca":       "Vasos coloreados",
    "thal":     "Thal",
}

# Variables de entrada (sin target)
feature_cols = [c for c in df.columns if c != "target"]
X_raw = df[feature_cols].values
y     = df["target"].values          # 0 = sin enfermedad, 1 = con enfermedad

# Normalización MinMax → rango [0, 1] requerido por SOM
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

print(f"Variables: {feature_cols}")
print(f"Shape entrada al SOM: {X.shape}")
print(f"Distribución target: {np.bincount(y)}\n")


# ════════════════════════════════════════════════════════════
#  9. INICIALIZACIÓN DE PESOS
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("9. INICIALIZACIÓN DE PESOS")
print("=" * 60)

# Tamaño del mapa: regla ~sqrt(5*sqrt(N))
n_samples  = X.shape[0]
map_size   = int(np.ceil(np.sqrt(5 * np.sqrt(n_samples))))
SOM_ROWS   = map_size
SOM_COLS   = map_size
INPUT_DIM  = X.shape[1]
N_EPOCHS   = 500
SIGMA      = 1.5    # radio inicial de vecindad (kernel)
LR         = 0.5    # tasa de aprendizaje inicial

print(f"  Tamaño mapa   : {SOM_ROWS} × {SOM_COLS} = {SOM_ROWS*SOM_COLS} neuronas")
print(f"  Dimensión entrada : {INPUT_DIM} variables")
print(f"  Épocas        : {N_EPOCHS}")
print(f"  Sigma inicial : {SIGMA}")
print(f"  LR inicial    : {LR}\n")

# ── Método A: Inicialización ALEATORIA ──────────────────────
som_random = MiniSom(
    SOM_ROWS, SOM_COLS, INPUT_DIM,
    sigma=SIGMA, learning_rate=LR,
    neighborhood_function="gaussian",
    random_seed=42
)
# Inicializa pesos desde distribución uniforme [0,1]
som_random.random_weights_init(X)
print("  [Aleatorio] Pesos inicializados — rango aprox:",
      np.round(som_random.get_weights().min(), 3), "a",
      np.round(som_random.get_weights().max(), 3))

# ── Método B: Inicialización por PCA ───────────────────────
som_pca = MiniSom(
    SOM_ROWS, SOM_COLS, INPUT_DIM,
    sigma=SIGMA, learning_rate=LR,
    neighborhood_function="gaussian",
    random_seed=42
)
# Los pesos se inicializan sobre los 2 primeros componentes principales
som_pca.pca_weights_init(X)
print("  [PCA]      Pesos inicializados sobre los 2 PCs principales")

# Visualizar diferencia en distribución de pesos iniciales
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Ítem 9 — Inicialización de pesos\nDistribución de pesos iniciales por variable",
             fontsize=13, fontweight="bold")

for ax, (som, label) in zip(axes, [(som_random, "Aleatoria"), (som_pca, "PCA")]):
    w = som.get_weights().reshape(-1, INPUT_DIM)
    ax.boxplot(w, labels=[col_labels.get(c, c) for c in feature_cols],
               patch_artist=True,
               boxprops=dict(facecolor="#cce5ff", color="#004085"))
    ax.set_title(f"Inicialización {label}", fontsize=11)
    ax.set_ylabel("Valor del peso (normalizado)")
    ax.set_xticklabels([col_labels.get(c, c) for c in feature_cols],
                       rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig(out("item9_inicializacion.png"), bbox_inches="tight")
plt.close()
print(f"  [GUARDADO] {out('item9_inicializacion.png')}")


# ════════════════════════════════════════════════════════════
#  10. ESTIMACIÓN DE PESOS (ENTRENAMIENTO)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("10. ESTIMACIÓN DE PESOS (ENTRENAMIENTO)")
print("=" * 60)

# Entrenamos ambos SOM registrando el error cuantización por época
def train_som_track_error(som, X, n_epochs):
    """Entrena el SOM y registra el error de cuantización cada 10 épocas."""
    errors = []
    checkpoints = list(range(0, n_epochs + 1, max(1, n_epochs // 50)))
    for i in range(n_epochs):
        idx = np.random.randint(0, len(X))
        som.update(X[idx], som.winner(X[idx]), i, n_epochs)
        if i in checkpoints:
            errors.append((i, som.quantization_error(X)))
    return errors

print("  Entrenando SOM con init aleatoria...")
errors_random = train_som_track_error(som_random, X, N_EPOCHS)

print("  Entrenando SOM con init PCA...")
errors_pca    = train_som_track_error(som_pca, X, N_EPOCHS)

# Graficar convergencia
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Ítem 10 — Estimación de pesos\nConvergencia del Error de Cuantización",
             fontsize=13, fontweight="bold")

ep_r, err_r = zip(*errors_random)
ep_p, err_p = zip(*errors_pca)

ax.plot(ep_r, err_r, color="#e74c3c", lw=2, label="Init aleatoria")
ax.plot(ep_p, err_p, color="#2980b9", lw=2, label="Init PCA")
ax.set_xlabel("Época", fontsize=11)
ax.set_ylabel("Error de cuantización", fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.35)

# Anotar error final
ax.annotate(f"Final: {err_r[-1]:.4f}", xy=(ep_r[-1], err_r[-1]),
            xytext=(-80, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
            color="#e74c3c", fontsize=9)
ax.annotate(f"Final: {err_p[-1]:.4f}", xy=(ep_p[-1], err_p[-1]),
            xytext=(-80, -25), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#2980b9"),
            color="#2980b9", fontsize=9)

plt.tight_layout()
plt.savefig(out("item10_convergencia.png"), bbox_inches="tight")
plt.close()
print(f"  Error final aleatorio : {err_r[-1]:.4f}")
print(f"  Error final PCA       : {err_p[-1]:.4f}")
print(f"  [GUARDADO] {out('item10_convergencia.png')}")

# Usar el SOM con mejor init (PCA) para los siguientes ítems
som = som_pca


# ════════════════════════════════════════════════════════════
#  11. KERNEL (FUNCIÓN DE VECINDAD)
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("11. KERNEL / FUNCIÓN DE VECINDAD")
print("=" * 60)

kernels = {
    "gaussian": "Gaussiana",
    "bubble":   "Bubble",
    "mexican_hat": "Mexican Hat",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Ítem 11 — Kernel / Función de vecindad\n"
             "Comparación de U-Matrix según tipo de kernel",
             fontsize=13, fontweight="bold")

som_kernels = {}
for ax, (kname, klabel) in zip(axes, kernels.items()):
    s = MiniSom(
        SOM_ROWS, SOM_COLS, INPUT_DIM,
        sigma=SIGMA, learning_rate=LR,
        neighborhood_function=kname,
        random_seed=42
    )
    s.pca_weights_init(X)
    s.train(X, N_EPOCHS, verbose=False)
    som_kernels[kname] = s

    # U-Matrix
    umatrix = s.distance_map()
    im = ax.imshow(umatrix, cmap="bone_r", interpolation="nearest")
    ax.set_title(f"Kernel: {klabel}", fontsize=11)
    ax.set_xlabel("Columna neurona")
    ax.set_ylabel("Fila neurona")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(out("item11_kernels.png"), bbox_inches="tight")
plt.close()
print("  Kernels entrenados: Gaussiana, Bubble, Mexican Hat")
print(f"  [GUARDADO] {out('item11_kernels.png')}")

# Usar el SOM gaussiano (más estable) para ítems 12 y 13
som_gauss = som_kernels["gaussian"]


# ════════════════════════════════════════════════════════════
#  12. VISUALIZACIÓN
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("12. VISUALIZACIÓN")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Ítem 12 — Visualización del Mapa de Kohonen\nHeart Disease UCI Dataset",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── 12a. U-Matrix con etiquetas de pacientes ───────────────
ax_u = fig.add_subplot(gs[0:2, 0:2])
umatrix = som_gauss.distance_map()
ax_u.imshow(umatrix, cmap="bone_r", interpolation="nearest", alpha=0.85)
ax_u.set_title("U-Matrix con pacientes proyectados", fontsize=11, fontweight="bold")
ax_u.set_xlabel("Columna neurona")
ax_u.set_ylabel("Fila neurona")

colors = {0: "#2ecc71", 1: "#e74c3c"}
labels_map = {0: "Sin enfermedad", 1: "Con enfermedad"}
markers_map = {0: "o", 1: "^"}
offset = 0.38

for i, x in enumerate(X):
    w = som_gauss.winner(x)
    jitter = (np.random.rand(2) - 0.5) * offset
    ax_u.plot(
        w[1] + jitter[1], w[0] + jitter[0],
        markers_map[y[i]],
        color=colors[y[i]],
        markersize=5, alpha=0.75, markeredgewidth=0.3,
        markeredgecolor="white"
    )

legend_elements = [
    mpatches.Patch(color=colors[k], label=labels_map[k]) for k in colors
]
ax_u.legend(handles=legend_elements, loc="upper right", fontsize=9)

# ── 12b. Component planes (variables más relevantes) ────────
key_vars = ["age", "chol", "thalach", "oldpeak", "trestbps", "cp"]
key_labels = [col_labels[v] for v in key_vars]
weights = som_gauss.get_weights()

cp_axes = [
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[1, 3]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
]

cmaps = ["YlOrRd", "Blues", "Greens", "Purples", "Oranges", "PuRd"]
for ax_cp, var, lbl, cmap in zip(cp_axes, key_vars, key_labels, cmaps):
    idx = feature_cols.index(var)
    plane = weights[:, :, idx]
    im = ax_cp.imshow(plane, cmap=cmap, interpolation="nearest")
    ax_cp.set_title(lbl, fontsize=9, fontweight="bold")
    ax_cp.set_xticks([])
    ax_cp.set_yticks([])
    plt.colorbar(im, ax=ax_cp, fraction=0.046, pad=0.04)

# ── 12c. Mapa de activación por clase ─────────────────────
ax_act = fig.add_subplot(gs[2, 2:4])
activation_0 = np.zeros((SOM_ROWS, SOM_COLS))
activation_1 = np.zeros((SOM_ROWS, SOM_COLS))

for i, x in enumerate(X):
    w = som_gauss.winner(x)
    if y[i] == 0:
        activation_0[w] += 1
    else:
        activation_1[w] += 1

ratio = np.divide(
    activation_1,
    activation_0 + activation_1 + 1e-9
)
im = ax_act.imshow(ratio, cmap="RdYlGn_r", interpolation="nearest",
                   vmin=0, vmax=1)
ax_act.set_title("Proporción de pacientes CON enfermedad por neurona",
                 fontsize=10, fontweight="bold")
ax_act.set_xlabel("Columna neurona")
ax_act.set_ylabel("Fila neurona")
plt.colorbar(im, ax=ax_act, fraction=0.046, pad=0.04,
             label="0 = sin enfermedad · 1 = con enfermedad")

plt.savefig(out("item12_visualizacion.png"), bbox_inches="tight")
plt.close()
print("  Visualizaciones generadas:")
print("    - U-Matrix con pacientes proyectados")
print("    - Component planes: edad, colesterol, FC máx, depresión ST, presión, dolor")
print("    - Mapa de activación por clase")
print(f"  [GUARDADO] {out('item12_visualizacion.png')}")


# ════════════════════════════════════════════════════════════
#  13. DENSIDADES Y CENTROIDES
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("13. DENSIDADES Y CENTROIDES")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Ítem 13 — Densidades y Centroides\nHeart Disease UCI Dataset",
             fontsize=13, fontweight="bold")

# ── 13a. Hit Map (densidad de muestras por neurona) ─────────
ax_hit = axes[0]
hit_map = np.zeros((SOM_ROWS, SOM_COLS))
for x in X:
    w = som_gauss.winner(x)
    hit_map[w] += 1

im = ax_hit.imshow(hit_map, cmap="YlOrRd", interpolation="nearest")
ax_hit.set_title("Hit Map\n(densidad de pacientes por neurona)",
                 fontsize=11, fontweight="bold")
ax_hit.set_xlabel("Columna")
ax_hit.set_ylabel("Fila")
plt.colorbar(im, ax=ax_hit, fraction=0.046, pad=0.04, label="N° pacientes")

# Anotar número en cada celda
for i in range(SOM_ROWS):
    for j in range(SOM_COLS):
        n = int(hit_map[i, j])
        if n > 0:
            color = "white" if n > hit_map.max() * 0.5 else "black"
            ax_hit.text(j, i, str(n), ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

# ── 13b. Centroides en espacio PCA ──────────────────────────
ax_pca = axes[1]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_

# Proyectar los pesos de las neuronas al espacio PCA
weights_2d = pca.transform(
    som_gauss.get_weights().reshape(-1, INPUT_DIM)
)
centroids_2d = weights_2d.reshape(SOM_ROWS, SOM_COLS, 2)

# Graficar datos
for cls, color, label in [(0, "#2ecc71", "Sin enfermedad"),
                           (1, "#e74c3c", "Con enfermedad")]:
    mask = y == cls
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, alpha=0.45, s=20, label=label,
                   edgecolors="none")

# Graficar centroides de neuronas
for i in range(SOM_ROWS):
    for j in range(SOM_COLS):
        n = hit_map[i, j]
        if n > 0:
            cx, cy = centroids_2d[i, j]
            size = 40 + (n / hit_map.max()) * 180
            ax_pca.scatter(cx, cy, c="navy", s=size, marker="s",
                           alpha=0.6, edgecolors="white", linewidths=0.5,
                           zorder=5)

ax_pca.set_title("Centroides de neuronas en espacio PCA\n"
                 f"(PC1={explained[0]:.1%}, PC2={explained[1]:.1%})",
                 fontsize=11, fontweight="bold")
ax_pca.set_xlabel(f"PC1 ({explained[0]:.1%})")
ax_pca.set_ylabel(f"PC2 ({explained[1]:.1%})")
ax_pca.legend(fontsize=9)
ax_pca.grid(alpha=0.3)

centroid_legend = mpatches.Patch(color="navy",
    label="Centroide neurona\n(tamaño ∝ densidad)")
ax_pca.legend(handles=[
    mpatches.Patch(color="#2ecc71", label="Sin enfermedad"),
    mpatches.Patch(color="#e74c3c", label="Con enfermedad"),
    centroid_legend,
], fontsize=8)

# ── 13c. Perfil de centroides para neuronas más densas ──────
ax_prof = axes[2]

# Tomar las 5 neuronas más pobladas
flat_hits = hit_map.flatten()
top5_idx  = np.argsort(flat_hits)[::-1][:5]
top5_pos  = [(idx // SOM_COLS, idx % SOM_COLS) for idx in top5_idx]

weights_all = som_gauss.get_weights()
profile_data = []
for rank, (r, c) in enumerate(top5_pos):
    w = weights_all[r, c, :]
    w_orig = scaler.inverse_transform(w.reshape(1, -1))[0]
    profile_data.append({
        "Neurona": f"N({r},{c})\nn={int(hit_map[r,c])}",
        **{col_labels.get(f, f): round(float(w_orig[fi]), 1)
           for fi, f in enumerate(feature_cols)}
    })

profile_df = pd.DataFrame(profile_data).set_index("Neurona")

# Mostrar solo variables numéricas continuas para el heatmap
cont_vars = ["Edad", "Presión en reposo", "Colesterol", "FC máxima",
             "Depresión ST"]
cont_vars = [v for v in cont_vars if v in profile_df.columns]

if cont_vars:
    sub = profile_df[cont_vars].T
    # Normalizar por fila para comparar entre neuronas
    sub_norm = sub.apply(lambda row: (row - row.min()) /
                         (row.max() - row.min() + 1e-9), axis=1)
    sns.heatmap(sub_norm, ax=ax_prof, cmap="RdYlGn_r",
                annot=profile_df[cont_vars].T.round(0).astype(int),
                fmt="d", linewidths=0.5, cbar_kws={"label": "Valor relativo"})
    ax_prof.set_title("Perfil de centroides\n(5 neuronas más densas)",
                      fontsize=11, fontweight="bold")
    ax_prof.set_xlabel("Neurona")
    ax_prof.set_ylabel("Variable")
    ax_prof.tick_params(axis="x", rotation=0, labelsize=8)
    ax_prof.tick_params(axis="y", rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig(out("item13_densidades_centroides.png"), bbox_inches="tight")
plt.close()
print("  Visualizaciones generadas:")
print("    - Hit Map: densidad de pacientes por neurona")
print("    - Centroides en espacio PCA con tamaño proporcional a densidad")
print("    - Perfil de centroides de las 5 neuronas más densas")
print(f"  [GUARDADO] {out('item13_densidades_centroides.png')}")

# Imprimir tabla de centroides más densas
print("\n  TABLA — Perfil de las 5 neuronas más densas (valores originales):")
print(profile_df[cont_vars].to_string())

print("\n" + "=" * 60)
print("PIPELINE COMPLETADO — 4 figuras generadas:")
print("  item9_inicializacion.png")
print("  item10_convergencia.png")
print("  item11_kernels.png")
print("  item12_visualizacion.png")
print("  item13_densidades_centroides.png")
print("=" * 60)