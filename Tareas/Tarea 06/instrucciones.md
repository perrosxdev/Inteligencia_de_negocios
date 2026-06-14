# Requisitos
- Python 3.10 o superior
- pip

# Verificar instalación:

- python --version
- pip --version
- Crear entorno virtual

# Desde la carpeta del proyecto:

- python -m venv .venv

# Activar entorno virtual

## Windows (CMD)
- .venv\Scripts\activate
## Windows (PowerShell)
- .venv\Scripts\Activate.ps1
## Linux / macOS
- source .venv/bin/activate

Si la activación fue exitosa, aparecerá:

(.venv)

al comienzo de la línea de comandos.

# Instalar dependencias

Con el entorno virtual activado:

- pip install -r requirements.txt

# Ejecutar el análisis

Desde la carpeta del proyecto:

- python analysis.py

# Resultados generados

Durante la ejecución se mostrará información en consola:

- Dimensiones del dataset.
- Valores faltantes.
- División entrenamiento/prueba.
- Mejores hiperparámetros encontrados.
- Accuracy.
- Precision.
- Recall.
- F1-Score.
- AUC.
- Resumen final para el informe.


# Gráficos generados

Las figuras se guardarán automáticamente en la carpeta:

figs/

Archivos generados:

- arbol_decision.png
- comparacion_preparacion.png
- correlacion.png
- curva_validacion.png
- eda_overview.png
- evaluacion_overview.png
- importancia_variables.png