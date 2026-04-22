# ── 0. Configuración ─────────────────────────────────────────────────────────

# Buscador de la ruta del script (funciona con Rscript, source() en RStudio, o interactive)
thisfile <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("--file=", args)
  if (length(m) > 0) return(normalizePath(sub("--file=", "", args[m[1]])))
  if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) return(normalizePath(p))
  }
  return(NULL)
}

script_path <- thisfile()
script_dir  <- if (!is.null(script_path)) dirname(script_path) else getwd()

RUTA_DATOS  <- file.path(script_dir, "wine", "wine.data")
if (!file.exists(RUTA_DATOS)) {
  stop(sprintf("No se encontró '%s'. Verifica que la carpeta 'wine' exista en: %s",
               basename(RUTA_DATOS), file.path(script_dir, "wine")))
}
RUTA_SALIDA <- file.path(script_dir, "anexos")
dir.create(RUTA_SALIDA, recursive = TRUE, showWarnings = FALSE)
set.seed(42)

# ── 1. Paquetes ──────────────────────────────────────────────────────────────

paquetes <- c("tidyverse", "cluster", "factoextra",
              "gridExtra", "knitr", "kableExtra", "gt", "gtExtras", "webshot2")

for (p in paquetes) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Instalando: ", p)
    install.packages(p, repos = "https://cran.rstudio.com/")
  }
  library(p, character.only = TRUE)
}

# ── 2. Carga y preprocesamiento ───────────────────────────────────────────────

nombres_vars <- c("Clase", "Alcohol", "AcidoMalico", "Ceniza",
                  "AlcalinidadCeniza", "Magnesio", "FenolesTotales",
                  "Flavanoides", "FenNoFlavanoides", "Proantocianinas",
                  "IntensidadColor", "Matiz", "OD280_315", "Prolina")

abreviaturas <- c("Alc.", "Ác.Mal.", "Cen.", "AlcCen.", "Mg",
                  "Fen.T.", "Flav.", "FNF", "ProAnt",
                  "IntCol.", "Matiz", "OD", "Prol.")

wine_raw <- read.csv(RUTA_DATOS, header = FALSE, col.names = nombres_vars)
X        <- wine_raw[ , -1]           # quitar columna Clase
X_scaled <- scale(X)                  # estandarizar (media 0, sd 1)

cat("\n── Dataset cargado ──\n")
cat("Filas:", nrow(X), " | Columnas:", ncol(X), "\n")
cat("Variables:", paste(names(X), collapse = ", "), "\n")

# ── 3. K-Means k = 2…15 ──────────────────────────────────────────────────────

resultados <- list()

for (k in 2:15) {
  km <- kmeans(X_scaled, centers = k, nstart = 25, iter.max = 300)
  
  # Distancias euclidianas entre todos los pares de centroides (espacio escalado)
  cents  <- km$centers
  pares  <- combn(k, 2)
  dists  <- apply(pares, 2, function(p) sqrt(sum((cents[p[1],] - cents[p[2],])^2)))
  
  resultados[[k]] <- list(
    km         = km,
    labels     = km$cluster,
    centroids  = cents,                          # espacio escalado
    cents_orig = t(t(cents) * attr(X_scaled, "scaled:scale") +
                     attr(X_scaled, "scaled:center")),  # espacio original
    inertia    = km$tot.withinss,
    dist_min   = min(dists),
    dist_media = mean(dists)
  )
}

# ── 4. TABLA 1: Resumen distancias + SSE (k = 2…15) ─────────────────────────

tabla_dist <- tibble(
  k          = 2:15,
  dist_min   = sapply(2:15, \(k) resultados[[k]]$dist_min),
  dist_media = sapply(2:15, \(k) resultados[[k]]$dist_media),
  SSE        = sapply(2:15, \(k) resultados[[k]]$inertia)
) |>
  mutate(across(c(dist_min, dist_media), \(x) round(x, 4)),
         SSE = round(SSE, 2))

# — Mostrar en RStudio —
cat("\n── TABLA 1: Distancias entre centroides y SSE ──\n")
print(tabla_dist, n = 14)

# — Exportar como PNG via gt ─
tab1_gt <- tabla_dist |>
  gt() |>
  cols_label(
    k          = "k",
    dist_min   = "Dist. mín. (escalado)",
    dist_media = "Dist. media (escalado)",
    SSE        = "SSE"
  ) |>
  tab_header(title = "Tabla 1. Distancias entre centroides y SSE para k = 2…15") |>
  tab_style(
    style     = list(cell_text(weight = "bold"),
                     cell_fill(color = "#d4edda")),
    locations = cells_body(rows = k == 3)
  ) |>
  fmt_number(columns = c(dist_min, dist_media), decimals = 4) |>
  fmt_number(columns = SSE, decimals = 2) |>
  opt_table_font(font = "Arial") |>
  tab_options(table.font.size = 11,
              heading.align  = "left")

ruta_tab1 <- file.path(RUTA_SALIDA, "tabla1_distancias.png")
gtsave(tab1_gt, ruta_tab1, expand = 10)
cat("✔ tabla1_distancias.png guardada\n")

# ── 5. TABLA 2: Centroides k=3 en espacio original ──────────────────────────

cents3 <- resultados[[3]]$cents_orig
rownames(cents3) <- paste0("C", 1:3)
colnames(cents3) <- abreviaturas

tabla_cents3 <- as.data.frame(round(cents3, 3))
tabla_cents3 <- rownames_to_column(tabla_cents3, var = "Centroide")

# — Mostrar en RStudio —
cat("\n── TABLA 2: Centroides k=3 (espacio original) ──\n")
print(tabla_cents3)

# — Exportar como PNG via gt —
tab2_gt <- tabla_cents3 |>
  gt() |>
  tab_header(title = "Tabla 2. Centroides k=3 en espacio original") |>
  cols_label(Centroide = "") |>
  tab_style(
    style     = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) |>
  opt_table_font(font = "Arial") |>
  tab_options(table.font.size = 10,
              heading.align  = "left")

ruta_tab2 <- file.path(RUTA_SALIDA, "tabla2_centroides_k3.png")
gtsave(tab2_gt, ruta_tab2, expand = 10)
cat("✔ tabla2_centroides_k3.png guardada\n")

# ── 6. Distancias entre los 3 centroides de k=3 (para ecuación en LaTeX) ────

cents3_scaled <- resultados[[3]]$centroids
d12 <- sqrt(sum((cents3_scaled[1,] - cents3_scaled[2,])^2))
d13 <- sqrt(sum((cents3_scaled[1,] - cents3_scaled[3,])^2))
d23 <- sqrt(sum((cents3_scaled[2,] - cents3_scaled[3,])^2))

cat(sprintf("\n── Distancias entre centroides k=3 (espacio escalado) ──\n"))
cat(sprintf("d(C1,C2) = %.4f\n", d12))
cat(sprintf("d(C1,C3) = %.4f\n", d13))
cat(sprintf("d(C2,C3) = %.4f\n", d23))

# ── 7. FIG 1: Distancia mínima entre centroides vs k ─────────────────────────

ruta_fig1 <- file.path(RUTA_SALIDA, "fig1_distancia_centroides.png")
png(ruta_fig1, width = 1800, height = 900, res = 200)

ggplot(tabla_dist, aes(x = k, y = dist_min)) +
  geom_line(color = "steelblue", linewidth = 1.1) +
  geom_point(aes(color = (k == 3)), size = 3.5, show.legend = FALSE) +
  scale_color_manual(values = c("FALSE" = "steelblue", "TRUE" = "red")) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red", alpha = 0.7) +
  annotate("text", x = 3.3, y = max(tabla_dist$dist_min) * 0.97,
           label = "k = 3", color = "red", size = 3.5, hjust = 0) +
  scale_x_continuous(breaks = 2:15) +
  labs(
    title = "Distancia euclidiana mínima entre centroides vs k",
    x     = "Número de clústeres k",
    y     = "Distancia mínima (espacio estandarizado)"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank())

dev.off()
cat("✔ fig1_distancia_centroides.png guardada\n")

# ── 8. FIG 2: Scatter PCA — k=3 con centroides ───────────────────────────────

pca_res  <- prcomp(X_scaled, scale. = FALSE)
var_exp  <- round(pca_res$sdev^2 / sum(pca_res$sdev^2) * 100, 1)
scores   <- as.data.frame(pca_res$x[ , 1:2])
scores$cluster <- factor(resultados[[3]]$labels)

cents_pca <- as.data.frame(predict(pca_res, resultados[[3]]$centroids)[ , 1:2])
cents_pca$cluster <- factor(1:3)

ruta_fig2 <- file.path(RUTA_SALIDA, "fig2_pca_k3.png")
png(ruta_fig2, width = 1800, height = 1400, res = 200)

paleta3 <- c("1" = "#E63946", "2" = "#2A9D8F", "3" = "#E9A227")

ggplot(scores, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.75, size = 2) +
  geom_point(data = cents_pca, aes(x = PC1, y = PC2),
             shape = 8, size = 5, color = "black", stroke = 1.5) +
  geom_text(data = cents_pca,
            aes(x = PC1, y = PC2, label = paste0("C", cluster)),
            vjust = -1, color = "black", fontface = "bold", size = 3.5) +
  scale_color_manual(values = paleta3, name = "Clúster") +
  labs(
    title = sprintf("Proyección PCA — K-Means k=3  (varianza explicada: %.1f%%)",
                    sum(var_exp[1:2])),
    x = sprintf("PC1 (%.1f%%)", var_exp[1]),
    y = sprintf("PC2 (%.1f%%)", var_exp[2])
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right")

dev.off()
cat("✔ fig2_pca_k3.png guardada\n")

# ── 9. FIG 3: Comparación k=2,3,4,5 en PCA ───────────────────────────────────

paleta6 <- c("#E63946","#2A9D8F","#E9A227","#457B9D","#6A0572","#F4A261")

plots_comp <- lapply(c(2, 3, 4, 5), function(k) {
  sc <- scores
  sc$cluster <- factor(resultados[[k]]$labels)
  cp <- as.data.frame(predict(pca_res, resultados[[k]]$centroids)[ , 1:2])
  cp$cluster <- factor(seq_len(k))
  
  ggplot(sc, aes(x = PC1, y = PC2, color = cluster)) +
    geom_point(alpha = 0.65, size = 1.5) +
    geom_point(data = cp, aes(x = PC1, y = PC2),
               shape = 8, size = 4, color = "black", stroke = 1.3) +
    scale_color_manual(values = paleta6[seq_len(k)]) +
    labs(title = paste0("k = ", k), x = "PC1", y = "PC2") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none",
          plot.title     = element_text(face = "bold"))
})

ruta_fig3 <- file.path(RUTA_SALIDA, "fig3_comparacion_k.png")
png(ruta_fig3, width = 2800, height = 2200, res = 200)
grid.arrange(grobs = plots_comp, ncol = 2,
             top = "Comparación de agrupamientos k=2,3,4,5 (PCA)")
dev.off()
cat("✔ fig3_comparacion_k.png guardada\n")

# ── 10. Resumen final en consola ──────────────────────────────────────────────

cat("\n═══════════════════════════════════════════════════════════════\n")
cat("  RESUMEN — ARCHIVOS GENERADOS\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  TABLAS (PNG para insertar en LaTeX):\n")
cat("    ✔ tabla1_distancias.png\n")
cat("    ✔ tabla2_centroides_k3.png\n")
cat("  FIGURAS (PNG para insertar en LaTeX):\n")
cat("    ✔ fig1_distancia_centroides.png\n")
cat("    ✔ fig2_pca_k3.png\n")
cat("    ✔ fig3_comparacion_k.png\n")
cat("  VALORES CLAVE para el .tex:\n")
cat(sprintf("    k óptimo             : 3\n"))
cat(sprintf("    d_min(k=3)           : %.4f\n", resultados[[3]]$dist_min))
cat(sprintf("    d_min(k=4)           : %.4f\n", resultados[[4]]$dist_min))
cat(sprintf("    Caída k3→k4          : %.4f\n",
            resultados[[3]]$dist_min - resultados[[4]]$dist_min))
cat(sprintf("    d(C1,C2) escalado    : %.4f\n", d12))
cat(sprintf("    d(C1,C3) escalado    : %.4f\n", d13))
cat(sprintf("    d(C2,C3) escalado    : %.4f\n", d23))
cat(sprintf("    Varianza PCA (PC1+PC2): %.1f%%\n", sum(var_exp[1:2])))
cat("═══════════════════════════════════════════════════════════════\n")
cat("  Copia todos los archivos .png a la carpeta del .tex y compila.\n\n")