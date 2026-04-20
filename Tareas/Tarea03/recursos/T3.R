# ================================
# FASE 1: Comprensión del negocio
# ================================
# Objetivo: Segmentar cultivos de arroz para encontrar patrones productivos

# ================================
# FASE 2: Comprensión de datos
# ================================
library(tidyverse)
library(factoextra)
library(cluster)   # necesario para silhouette()

# Cargar dataset
paddy <- read.csv("C:/Users/macka/Desktop/Inteligencia de Negocios/Tarea3/paddy+dataset/paddydataset.csv", 
                  header = TRUE, sep = ",")

# ================================
# FASE 3: Preparación
# ================================
# Limpiar datos
paddy_clean <- na.omit(paddy)

# Seleccionar solo variables numéricas
paddy_num <- paddy_clean %>% select(where(is.numeric))

# Escalar datos
paddy_scaled <- scale(paddy_num)

# ================================
# PCA (REDUCCIÓN DE DIMENSIONALIDAD)
# ================================
pca <- prcomp(paddy_scaled, center = TRUE, scale. = TRUE)

# Scree plot: varianza explicada por cada componente
fviz_eig(pca)

# Revisar las cargas (pesos de cada variable en PC1, PC2, PC3)
print(round(pca$rotation[,1:3], 3))

# Seleccionar 2 o 3 componentes principales
pca_data <- as.data.frame(pca$x[, 1:3])

# ================================
# FASE 4: MODELADO (K-MEANS con Silueta)
# ================================
set.seed(123)

# Método de la silueta (único criterio)
fviz_nbclust(pca_data, kmeans, method = "silhouette") +
  ggtitle("Número óptimo de clusters - Método de la Silueta")

# Aplicar k-means con K=3 (ejemplo, según silueta)
kmeans_pca <- kmeans(pca_data, centers = 3, nstart = 25)

# Visualización de clusters en PCA (PC1 vs PC2)
fviz_cluster(kmeans_pca, data = pca_data,
             geom = "point", ellipse.type = "norm",
             ggtheme = theme_minimal())

# Visualización de clusters en PC1 vs PC3
fviz_cluster(kmeans_pca, data = pca_data[,c("PC1","PC3")],
             geom = "point", ellipse.type = "norm",
             ggtheme = theme_minimal())

# Visualización de clusters en PC2 vs PC3
fviz_cluster(kmeans_pca, data = pca_data[,c("PC2","PC3")],
             geom = "point", ellipse.type = "norm",
             ggtheme = theme_minimal())

# ================================
# INTERPRETACIÓN
# ================================
# Agregar cluster al dataset original
paddy_clustered <- paddy_clean
paddy_clustered$cluster <- kmeans_pca$cluster

# Promedios por cluster
cluster_summary <- paddy_clustered %>%
  group_by(cluster) %>%
  summarise(across(where(is.numeric), mean))

print(cluster_summary)

# ================================
# ANÁLISIS DE SILUETA POR CLUSTER
# ================================
# Calcular silueta para cada observación
sil <- silhouette(kmeans_pca$cluster, dist(pca_data))

# Promedio de silueta por cluster
sil_summary <- aggregate(sil[, "sil_width"], 
                         by = list(cluster = sil[, "cluster"]), 
                         FUN = mean)

print(sil_summary)


