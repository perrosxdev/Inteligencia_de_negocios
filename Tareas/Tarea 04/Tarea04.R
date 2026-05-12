# ==============================================================================
# INFO1184 - Inteligencia de Negocios | Semestre I-2026
# Tarea 4 - Análisis de Componentes Principales (PCA)
# Dataset: Boston (MASS)
# Metodología: CRISP-DM - Fases 1 a 5
# ==============================================================================

# ---- LIBRERÍAS ---------------------------------------------------------------
# Establecer directorio de trabajo
setwd("d:/Perrosxdev/Inteligencia_de_negocios/Tareas/Tarea 04")

if (!require(MASS))      install.packages("MASS")
if (!require(ggplot2))   install.packages("ggplot2")
if (!require(corrplot))  install.packages("corrplot")
if (!require(factoextra))install.packages("factoextra")
if (!require(dplyr))     install.packages("dplyr")
if (!require(gridExtra)) install.packages("gridExtra")
if (!require(ggcorrplot))install.packages("ggcorrplot")
if (!require(reshape2))  install.packages("reshape2")
if (!require(stargazer)) install.packages("stargazer")
if (!require(xtable))    install.packages("xtable")
library(MASS)
library(ggplot2)
library(corrplot)
library(factoextra)
library(dplyr)
library(gridExtra)
library(ggcorrplot)
library(reshape2)
library(stargazer)
library(xtable)

# Crea carpeta de resultados si no existe
if (!dir.exists("resultados")) dir.create("resultados")

write_text_file <- function(path, lines) {
  writeLines(lines, con = path, useBytes = TRUE)
}

# ==============================================================================
# FASE 1 - COMPRENSIÓN DEL NEGOCIO
# ==============================================================================
cat("=== FASE 1: COMPRENSIÓN DEL NEGOCIO ===\n")
cat("Dataset: Boston Housing (MASS)\n")
cat("Objetivo: Aplicar PCA y responder preguntas de investigación\n\n")

# ==============================================================================
# FASE 2 - COMPRENSIÓN DE LOS DATOS
# ==============================================================================
cat("=== FASE 2: COMPRENSIÓN DE LOS DATOS ===\n\n")

data("Boston")
df <- Boston

# -- 2.1 Estructura del dataset ------------------------------------------------
cat("--- Dimensiones del dataset ---\n")
cat("Filas:", nrow(df), "| Columnas:", ncol(df), "\n\n")

cat("--- Tipos de variables ---\n")
str(df)

cat("\n--- Primeras filas ---\n")
print(head(df))

# -- 2.2 Estadísticas descriptivas --------------------------------------------
cat("\n--- Estadísticas descriptivas ---\n")
summary_table <- summary(df)
print(summary_table)
write.csv(summary_table, file = "resultados/tabla_summary.csv")

# -- 2.3 Visualización de distribuciones (histogramas) ------------------------
pdf("resultados/histogramas_distribuciones.pdf", width = 12, height = 12)
par(mfrow = c(4, 4), mar = c(3, 3, 2, 1))
for (col in names(df)) {
  hist(df[[col]], main = col, xlab = col, col = "steelblue", border = "white")
}
par(mfrow = c(1, 1))
dev.off()

# -- 2.4 Detección de valores atípicos (Boxplots) - Pregunta 1 ----------------
cat("\n--- PREGUNTA 1: Valores atípicos (Boxplots) ---\n")

vars_boxplot <- c("crim", "zn", "rm", "black", "medv", "lstat", "dis", "tax")
df_melt <- melt(df[, vars_boxplot])

p_box <- ggplot(df_melt, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, outlier.size = 2) +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  labs(title = "Boxplots - Detección de Valores Atípicos",
       x = "", y = "Valor")
print(p_box)
ggsave("resultados/boxplots_outliers.pdf", p_box, width = 11, height = 6)

# Contar atípicos por variable usando regla IQR
cat("\nNúmero de valores atípicos por variable (regla IQR):\n")
outlier_count <- sapply(df, function(x) {
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR_val <- Q3 - Q1
  sum(x < (Q1 - 1.5 * IQR_val) | x > (Q3 + 1.5 * IQR_val))
})
print(sort(outlier_count, decreasing = TRUE))
write.csv(outlier_count, file = "resultados/tabla_outliers.csv")

# -- 2.5 Correlaciones ---------------------------------------------------------
cat("\n--- Matriz de correlación ---\n")
corr_matrix <- cor(df)
print(round(corr_matrix, 2))
write.csv(round(corr_matrix,2), file = "resultados/tabla_correlaciones.csv")

# ==============================================================================
# FASE 3 - PREPARACIÓN DE LOS DATOS
# ==============================================================================
cat("\n=== FASE 3: PREPARACIÓN DE LOS DATOS ===\n\n")

# -- 3.1 Verificar valores faltantes ------------------------------------------
cat("--- Valores faltantes por variable ---\n")
na_table <- colSums(is.na(df))
print(na_table)
cat("Total de NA:", sum(is.na(df)), "\n")
write.csv(na_table, file = "resultados/tabla_valores_faltantes.csv")

# -- 3.2 Separar variable 'chas' (dummy) y 'medv' (objetivo) ------------------
df_pca_input <- df %>% select(-chas, -medv)
cat("\nVariables incluidas en PCA:", paste(names(df_pca_input), collapse = ", "), "\n")

# -- 3.3 Estandarización (media=0, sd=1) - OBLIGATORIO para PCA ---------------
df_scaled <- scale(df_pca_input)
cat("\nDatos estandarizados (primeras 5 filas):\n")
scaled_head <- round(head(df_scaled), 3)
print(scaled_head)
write.csv(scaled_head, file = "resultados/datos_estandarizados_head.csv")

# ==============================================================================
# FASE 4 - MODELADO: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# ==============================================================================
cat("\n=== FASE 4: MODELADO - PCA ===\n\n")

# -- 4.1 Aplicar PCA -----------------------------------------------------------
pca_result <- prcomp(df_scaled, center = FALSE, scale. = FALSE)

cat("--- Resumen del PCA ---\n")
pca_summary <- summary(pca_result)
print(pca_summary)
capture.output(pca_summary, file = "resultados/summary_pca.txt")

# Varianza acumulada
var_explicada <- summary(pca_result)$importance[3, ]
cat("\nVarianza acumulada:\n")
print(round(var_explicada, 4))
write.csv(round(var_explicada, 4), file = "resultados/varianza_acumulada.csv")

# ¿Cuántos componentes explican al menos el 80%?
n_comp_80 <- which(var_explicada >= 0.80)[1]
cat("\nComponentes necesarios para explicar ≥80% de varianza:", n_comp_80, "\n")
write(n_comp_80, file = "resultados/n_componentes_80pct.txt")

# -- 4.2 Scree Plot (gráfico de varianza explicada) --------------------------
varianza_df <- data.frame(
  Componente = factor(paste0("PC", seq_along(summary(pca_result)$importance[2, ])),
                      levels = paste0("PC", seq_along(summary(pca_result)$importance[2, ]))),
  Varianza = as.numeric(summary(pca_result)$importance[2, ]) * 100,
  Acumulada = as.numeric(summary(pca_result)$importance[3, ]) * 100
)

p_scree <- ggplot(varianza_df, aes(x = Componente, y = Varianza, group = 1)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_line(aes(y = Acumulada), color = "red", linewidth = 1) +
  geom_point(aes(y = Acumulada), color = "red", size = 2.5) +
  geom_hline(yintercept = 80, linetype = "dashed", color = "gray40", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%", Varianza)), vjust = -0.5, size = 2.8) +
  labs(title = "Scree Plot - Varianza Explicada por Componente Principal",
       x = "Componente Principal",
       y = "Varianza Explicada (%)") +
  ylim(0, 100) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("resultados/scree_plot.pdf", plot = p_scree, width = 9, height = 6)

# -- 4.3 Cargas (Loadings) de los componentes ---------------------------------
cat("\n--- Cargas de PC1 y PC2 ---\n")
# --- Cargas de PC1 y PC2 ---
rot_mat <- pca_result$rotation[, 1:4]
loadings_df <- data.frame(
  Variable = rownames(rot_mat),
  PC1 = rot_mat[, 1],
  PC2 = rot_mat[, 2],
  PC3 = rot_mat[, 3],
  PC4 = rot_mat[, 4],
  stringsAsFactors = FALSE
)
print(round(loadings_df[, -1], 3))
write.csv(data.frame(Variable=loadings_df$Variable, round(loadings_df[,-1], 3)), file = "resultados/tabla_pca_loadings.csv", row.names = FALSE)

p_load <- ggplot(loadings_df, aes(x = reorder(Variable, PC1), y = PC1, fill = PC1 > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("steelblue", "tomato"), guide = "none") +
  labs(title = "Cargas del PC1", x = "Variable", y = "Carga") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

p_load2 <- ggplot(loadings_df, aes(x = reorder(Variable, PC2), y = PC2, fill = PC2 > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("steelblue", "tomato"), guide = "none") +
  labs(title = "Cargas del PC2", x = "Variable", y = "Carga") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("resultados/cargas_PC1.pdf", plot = p_load, width = 7, height = 5)
ggsave("resultados/cargas_PC2.pdf", plot = p_load2, width = 7, height = 5)
pdf("resultados/cargas_barras_PC1_PC2.pdf", width = 12, height = 5)
grid.arrange(p_load, p_load2, ncol = 2)
dev.off()

# -- 4.4 Contribución de variables a PC1 y PC2 ---------------------------------
p_contrib1 <- fviz_contrib(pca_result, choice = "var", axes = 1,
                           title = "Contribución de variables a PC1")
p_contrib2 <- fviz_contrib(pca_result, choice = "var", axes = 2,
                           title = "Contribución de variables a PC2")
g2 <- grid.arrange(p_contrib1, p_contrib2, ncol = 2)
ggsave("resultados/contrib_pc1.pdf", plot = p_contrib1, width = 7, height = 5)
ggsave("resultados/contrib_pc2.pdf", plot = p_contrib2, width = 7, height = 5)
ggsave("resultados/contribucion_PC1_PC2.pdf", plot = g2, width = 12, height = 5)

# -- 4.5 Scores (coordenadas de las observaciones en el espacio PCA) -----------
pca_scores <- as.data.frame(pca_result$x[, 1:4])
pca_scores$medv <- df$medv
pca_scores$chas  <- as.factor(df$chas)
pca_scores$lstat <- df$lstat
pca_scores$crim  <- df$crim
write.csv(head(pca_scores,20), file = "resultados/tabla_scores_pca_head.csv")

# ==============================================================================
# FASE 5 - EVALUACIÓN Y ANÁLISIS DE PREGUNTAS DE INVESTIGACIÓN
# ==============================================================================
cat("\n=== FASE 5: EVALUACIÓN / RESPUESTA A PREGUNTAS ===\n\n")

# ---- PREGUNTA 2: ¿Qué suburbios tienen las casas más baratas? ---------------
cat("--- PREGUNTA 2: Suburbios con casas más baratas ---\n")
df_precio <- df %>%
  mutate(suburbio = row_number()) %>%
  arrange(medv)

precio_table <- head(df_precio %>% select(suburbio, medv, lstat, crim, rm), 10)
cat("Top 10 suburbios con menor medv (precio más bajo):\n")
print(precio_table)
write.csv(precio_table, file = "resultados/tabla_suburbios_bajo_precio.csv")

p_precio <- ggplot(df_precio, aes(x = reorder(factor(suburbio), medv), y = medv)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Suburbios con casas más baratas (medv)",
       x = "Suburbio (índice)", y = "Valor mediano ($1,000)") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 5),
        plot.title = element_text(hjust = 0.5, face = "bold"))
ggsave("resultados/suburbios_bajo_precio.pdf", p_precio, width=7,height=5)

# Histograma medv
p_hist_medv <- ggplot(df, aes(x = medv)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  geom_vline(xintercept = quantile(df$medv, 0.25), color = "red", linetype = "dashed") +
  labs(title = "Distribución del Precio Mediano de Viviendas (medv)",
       x = "medv ($1,000)", y = "Frecuencia") +
  theme_minimal()
ggsave("resultados/histograma_medv.pdf", p_hist_medv, width = 7, height = 5)

# ---- PREGUNTA 3: ¿Cómo influye el tamaño (rm) en el precio? ----------------
cat("\n--- PREGUNTA 3: Influencia del tamaño (rm) en el precio ---\n")
cor_rm_medv <- cor(df$rm, df$medv)
cat("Correlación rm - medv:", round(cor_rm_medv, 4), "\n")
write(round(cor_rm_medv, 4), file = "resultados/correlacion_rm_medv.txt")

p_rm_medv <- ggplot(df, aes(x = rm, y = medv)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Relación entre N° de habitaciones (rm) y precio (medv)",
       x = "Promedio de habitaciones (rm)", y = "Valor mediano ($1,000)") +
  theme_minimal() +
  annotate("text", x = min(df$rm) + 0.5, y = max(df$medv) - 2,
           label = paste("r =", round(cor_rm_medv, 3)), color = "red", size = 5)
ggsave("resultados/relacion_rm_medv.pdf", p_rm_medv, width = 7, height = 5)

modelo_rm <- lm(medv ~ rm, data = df)
cat("\nResumen regresión lineal medv ~ rm:\n")
modelo_rm_summary <- summary(modelo_rm)
print(modelo_rm_summary)
capture.output(modelo_rm_summary, file = "resultados/regresion_medv_rm.txt")

# ---- PREGUNTA 4: ¿Afecta la cercanía al río Charles (chas) al precio? ------
cat("\n--- PREGUNTA 4: Efecto de 'chas' sobre el precio ---\n")

stats_chas <- df %>%
  group_by(chas) %>%
  summarise(
    n         = n(),
    media_medv = mean(medv),
    mediana   = median(medv),
    sd        = sd(medv)
  ) %>%
  arrange(chas)
cat("Estadísticas de medv según cercanía al río Charles:\n")
print(stats_chas)
write.csv(stats_chas, file = "resultados/tabla_stats_chas.csv")

p_box_chas <- ggplot(df, aes(x = factor(chas), y = medv, fill = factor(chas))) +
  geom_boxplot(outlier.colour = "red") +
  scale_fill_manual(values = c("steelblue", "tomato"),
                    labels = c("No limita (0)", "Limita (1)")) +
  labs(title = "Valor de viviendas según cercanía al río Charles",
       x = "¿Limita con el río? (chas)", y = "Valor mediano ($1,000)",
       fill = "chas") +
  theme_minimal()
ggsave("resultados/boxplot_medv_chas.pdf", p_box_chas, width = 7, height = 5)

t_test_chas <- t.test(medv ~ chas, data = df)
cat("\nPrueba t (medv ~ chas):\n")
print(t_test_chas)
capture.output(t_test_chas, file = "resultados/prueba_t_chas.txt")

p_pca_chas <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = chas)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("steelblue", "tomato"),
                     labels = c("No limita río", "Limita río")) +
  labs(title = "Scores PCA coloreados por chas",
       color = "chas") +
  theme_minimal()
ggsave("resultados/pca_chas.pdf", p_pca_chas, width = 7, height = 5)

# ---- PREGUNTA 5: Impacto del estatus socioeconómico (lstat) -----------------
cat("\n--- PREGUNTA 5: Impacto del estatus socioeconómico (lstat) ---\n")

cor_lstat_medv <- cor(df$lstat, df$medv)
cat("Correlación lstat - medv:", round(cor_lstat_medv, 4), "\n")
write(round(cor_lstat_medv, 4), file = "resultados/correlacion_lstat_medv.txt")

p_lstat_medv <- ggplot(df, aes(x = lstat, y = medv)) +
  geom_point(alpha = 0.4, color = "darkorange") +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  labs(title = "Relación entre estatus socioeconómico (lstat) y precio (medv)",
       x = "% Población estatus bajo (lstat)", y = "Valor mediano ($1,000)") +
  theme_minimal() +
  annotate("text", x = max(df$lstat) - 5, y = max(df$medv) - 2,
           label = paste("r =", round(cor_lstat_medv, 3)), color = "red", size = 5)
ggsave("resultados/relacion_lstat_medv.pdf", p_lstat_medv, width = 7, height = 5)

modelo_lstat <- lm(medv ~ lstat, data = df)
cat("\nResumen regresión lineal medv ~ lstat:\n")
modelo_lstat_summary <- summary(modelo_lstat)
print(modelo_lstat_summary)
capture.output(modelo_lstat_summary, file = "resultados/regresion_medv_lstat.txt")

cat("\nCorrelación de lstat con todas las variables:\n")
cor_lstat <- sort(cor(df)[, "lstat"], decreasing = TRUE)
print(round(cor_lstat, 3))
write.csv(round(cor_lstat, 3), file = "resultados/tabla_correlacion_lstat_todas.csv")

p_pca_lstat <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = lstat)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_gradient(low = "green", high = "red") +
  labs(title = "Scores PCA coloreados por lstat",
       color = "lstat (%)") +
  theme_minimal()
ggsave("resultados/pca_lstat.pdf", p_pca_lstat, width = 7, height = 5)

# ---- PREGUNTA 6: ¿Es posible predecir la tasa de criminalidad (crim)? ------
cat("\n--- PREGUNTA 6: Predicción de la tasa de criminalidad (crim) ---\n")

cor_crim <- sort(cor(df)[, "crim"], decreasing = FALSE)
cat("Correlaciones de crim con todas las variables:\n")
print(round(cor_crim, 3))
write.csv(round(cor_crim, 3), file = "resultados/tabla_correlacion_crim_todas.csv")

p_crim1 <- ggplot(df, aes(x = rad, y = crim)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "crim vs rad", x = "Acceso autopistas (rad)", y = "Criminalidad (crim)") +
  theme_minimal()

p_crim2 <- ggplot(df, aes(x = tax, y = crim)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "crim vs tax", x = "Impuesto ($10k)", y = "Criminalidad (crim)") +
  theme_minimal()

p_crim3 <- ggplot(df, aes(x = lstat, y = crim)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "crim vs lstat", x = "Estatus bajo (%)", y = "Criminalidad (crim)") +
  theme_minimal()

p_crim4 <- ggplot(df, aes(x = dis, y = crim)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "crim vs dis", x = "Distancia empleo (dis)", y = "Criminalidad (crim)") +
  theme_minimal()

g3 <- grid.arrange(p_crim1, p_crim2, p_crim3, p_crim4, ncol = 2,
                   top = "Variables más correlacionadas con criminalidad")
ggsave("resultados/crim_correlados_4paneles.pdf", plot = g3, width=12,height=8)

modelo_crim <- lm(crim ~ rad + tax + lstat + dis + medv + indus + nox, data = df)
cat("\nModelo regresión múltiple para predecir crim:\n")
modelo_crim_summary <- summary(modelo_crim)
print(modelo_crim_summary)
capture.output(modelo_crim_summary, file = "resultados/regresion_multiple_crim.txt")

p_pca_crim <- ggplot(pca_scores, aes(x = PC1, y = PC2, color = crim)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_gradient(low = "lightblue", high = "darkred") +
  labs(title = "Scores PCA coloreados por tasa de criminalidad (crim)",
       color = "crim") +
  theme_minimal()
ggsave("resultados/pca_crim.pdf", p_pca_crim, width = 7, height = 5)

# -- Regresión usando scores de PCA para predecir crim -----------------------
cat("\nRegresión de crim sobre scores PCA (PC1 a PC4):\n")
modelo_pca_crim <- lm(crim ~ PC1 + PC2 + PC3 + PC4, data = pca_scores)
modelo_pca_crim_summary <- summary(modelo_pca_crim)
print(modelo_pca_crim_summary)
capture.output(modelo_pca_crim_summary, file = "resultados/regresion_crim_scorespca.txt")

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
cat("\n=== RESUMEN FINAL ===\n")
cat("Varianza explicada por PC1:", round(summary(pca_result)$importance[2,1]*100, 2), "%\n")
cat("Varianza explicada por PC2:", round(summary(pca_result)$importance[2,2]*100, 2), "%\n")
cat("Varianza explicada acumulada PC1+PC2:", round(summary(pca_result)$importance[3,2]*100, 2), "%\n")
cat("Componentes para >= 80% varianza:", n_comp_80, "\n")

cat("\n--- Correlaciones clave con medv ---\n")
cor_medv <- sort(abs(cor(df)[, "medv"]), decreasing = TRUE)
print(round(cor_medv, 3))
write.csv(round(cor_medv, 3), file = "resultados/tabla_correlacion_clave_medv.csv")

outlier_summary <- sort(outlier_count[outlier_count > 0], decreasing = TRUE)
if (length(outlier_summary) == 0) {
  outlier_summary <- sort(outlier_count, decreasing = TRUE)
}

respuestas_preguntas <- c(
  "Resumen automático de resultados - Boston Housing",
  "",
  sprintf("P1. Variables con atípicos detectados: %s.",
          paste(names(outlier_summary), outlier_summary, sep = "=", collapse = ", ")),
  "P2. Las viviendas más baratas se identifican en resultados/tabla_suburbios_bajo_precio.csv.",
  sprintf("P3. Correlación rm-medv = %.4f; pendiente del modelo lineal = %.4f; R2 = %.4f.",
          cor_rm_medv,
          coef(modelo_rm)["rm"],
          summary(modelo_rm)$r.squared),
  sprintf("P4. Medv promedio por chas: 0 = %.2f, 1 = %.2f; p-value t-test = %.4g.",
          stats_chas$media_medv[stats_chas$chas == 0],
          stats_chas$media_medv[stats_chas$chas == 1],
          t_test_chas$p.value),
  sprintf("P5. Correlación lstat-medv = %.4f; pendiente del modelo lineal = %.4f; R2 = %.4f.",
          cor_lstat_medv,
          coef(modelo_lstat)["lstat"],
          summary(modelo_lstat)$r.squared),
  sprintf("P6. Modelo múltiple para crim: R2 = %.4f, R2 ajustado = %.4f; modelo PCA (PC1-PC4) R2 = %.4f.",
          summary(modelo_crim)$r.squared,
          summary(modelo_crim)$adj.r.squared,
          summary(modelo_pca_crim)$r.squared),
  sprintf("PCA. PC1 explica %.2f%%, PC2 %.2f%% y se requieren %s componentes para llegar al 80%% de varianza.",
          summary(pca_result)$importance[2,1] * 100,
          summary(pca_result)$importance[2,2] * 100,
          n_comp_80)
)

write_text_file("resultados/respuestas_preguntas.txt", respuestas_preguntas)

cat("\nAnálisis completado.\n")
cat("\n--- Exportación de resultados completa. Consulta la carpeta 'resultados'.\n")