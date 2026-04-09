# Inteligencia_de_negocios# Tarea 2 — Segmentación Bank Marketing (R + Power BI)

Este repositorio contiene el trabajo de la **Tarea 2 (INFO1184, Semestre I-2026)**: segmentación de clientes del dataset **Bank Marketing** usando **CRISP-DM** y **clustering jerárquico (Average Linkage)** en **R**, además de un **dashboard en Power BI** (vista general + vista por *clusters*).

## Estructura del proyecto

- `tarea02PB.pbix`  
  Reporte Power BI con 2 páginas: exploración general y segmentación por `cluster`.

- `data/`
  - `bank-full.csv` — dataset original (separado por `;`)
  - `bank-names.txt` — descripción/metadata del dataset
  - `bank_sample_with_cluster_powerbi.csv` — muestra utilizada en el clustering con columna `cluster` (exportada desde R)

- `README.md` — este archivo

## Requisitos

- **Power BI Desktop** (Windows) para abrir `tarea02PB.pbix`.
- (Opcional) **R / RStudio** si quieres reproducir el clustering y regenerar el CSV con `cluster`.

## Cómo abrir el dashboard (sin refrescar)

1. Descarga o clona el repositorio.
2. Abre `tarea02PB.pbix` con Power BI Desktop.
3. Puedes interactuar con filtros/segmentaciones y visualizar los resultados.

> Nota: el archivo `.pbix` suele incluir una copia de los datos importados, por lo que normalmente abre y se puede usar sin necesidad de recargar los CSV.

## Cómo refrescar datos (si aparece error por rutas)

Si al presionar **Actualizar** Power BI muestra error, es porque el `.pbix` quedó apuntando a una ruta local del computador donde se creó (por ejemplo `D:\...`). Para corregirlo:

1. En Power BI Desktop, ve a **Inicio → Transformar datos** (Power Query).
2. En el panel izquierdo, selecciona la consulta/tabla que proviene del CSV (por ejemplo `bank-full` y/o `bank_sample_with_cluster_powerbi`).
3. En **Pasos aplicados**, entra al paso **Origen** (Source) y actualiza la ruta para que apunte a los archivos dentro del repositorio:
   - `./data/bank-full.csv`
   - `./data/bank_sample_with_cluster_powerbi.csv`
4. Haz clic en **Cerrar y aplicar**.
5. Ahora sí, presiona **Actualizar**.

## Notas sobre los datos y el clustering

- El dashboard general utiliza el dataset completo (`bank-full.csv`).
- La segmentación por `cluster` se basa en una **muestra de 1500 registros** (por eficiencia computacional), exportada como `bank_sample_with_cluster_powerbi.csv`.
- La variable `y` (suscripción) **no se usó para entrenar el clustering**, pero se incluye para evaluar la tasa de suscripción por segmento.

## Autoría

Trabajo académico para INFO1184 (Semestre I-2026).