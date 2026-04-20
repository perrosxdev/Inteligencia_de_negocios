# Proyecto de Inteligencia de Negocios - Análisis de Datos de Cultivos de Arroz

Este repositorio contiene los materiales y resultados del análisis de datos según la metodología CRISP-DM, aplicado a cultivos de arroz. Incluye la visualización dinámica en **Power BI**, el código en **R**, así como documentos complementarios generados para el informe.

---

## 📁 Estructura de carpetas / Archivos importantes

- `/recursos/paddy+dataset/paddydataset.csv`  
  Archivo CSV con los datos originales utilizados tanto en R como en Power BI.

- `/recursos/Parte02.pbix`  
  Archivo de Power BI (**.pbix**) con el dashboard generado para el análisis y visualización interactiva.

- `PC1vsPC2.pdf`, `PC1vsPC3.pdf`, `PC2vsPC3.pdf`  
  Visualizaciones de clustering generadas en R.

- `T3.R`  
  Script en R utilizado para procesamiento, clustering y análisis inicial de los datos.

---

## 🚦 Instrucciones para abrir y visualizar el dashboard de Power BI

1. **Asegúrese de contar con Power BI Desktop instalado.**  
   Puede descargarlo gratuitamente desde:  
   https://powerbi.microsoft.com/es-es/desktop/

2. **Descargue los siguientes archivos de este repositorio** (manteniendo la estructura de carpetas):
   - `/recursos/Parte02.pbix`
   - `/recursos/paddy+dataset/paddydataset.csv`

3. **Abra `Parte02.pbix` con Power BI Desktop.**

   > ⚠️ **IMPORTANTE:**  
   La primera vez que abra el archivo, Power BI le solicitará que valide o restablezca la conexión al archivo CSV (`paddydataset.csv`).  
   Si es así:
   - Verifique que el archivo `.csv` esté en la ruta relativa:  
     `/recursos/paddy+dataset/paddydataset.csv`
   - Si el CSV está en otra ubicación, puede **restablecer el origen de datos** desde Power BI:
     - Menú: “Inicio” → “Transformar datos” → “Origen de datos” ��� "Examinar" y seleccione el CSV correcto.

---

## 📊 ¿Qué contiene el dashboard de Power BI?

El archivo **Parte02.pbix** incluye:

- **Página 1:** Resumen general de producción y comparación por localidad y variedad.
- **Página 2:** Análisis de variables productivas e insumos, y su impacto en el rendimiento.
- **Página 3:** Impacto de variables climáticas (precipitación y humedad) sobre el rendimiento de arroz.

Cada página cuenta con segmentadores para filtrar y explorar los resultados por variedad, tipo de suelo y localidad.

---

## 🔗 Recursos adicionales

- **Código fuente R:** Disponible como `T3.R` para replicar el procesamiento y análisis estadístico.
- **Gráficos PDF:** Material de soporte generado desde R para el informe.

---

## 📬 Contacto

- Natalay Huaiquinao — nhuaiquinao2021@alu.uct.cl  
- Carlos Ulloa — carlos.ulloa2020@alu.uct.cl

---
