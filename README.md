# Predicción de Reservas de Hotel

Este proyecto es un pipeline de MLOps de extremo a extremo para predecir cancelaciones de reservas de hotel. Incluye la ingesta de datos desde Google Cloud Storage, el preprocesamiento de datos, el entrenamiento del modelo con ajuste de hiperparámetros y el seguimiento de experimentos con MLflow.

## Estructura del Proyecto

```
.
├── artifacts/
│   ├── models/
│   │   └── lgbm_model.pkl
│   ├── processed/
│   │   ├── processed_test.csv
│   │   └── processed_train.csv
│   └── raw/
│       ├── raw.csv
│       ├── test.csv
│       └── train.csv
├── config/
│   ├── config.yaml
│   ├── model_params.py
│   └── paths_config.py
├── logs/
├── mlruns/
├── notebook/
│   └── notebook.ipynb
├── pipeline/
│   └── training_pipeline.py
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   └── model_training.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Flujo de Trabajo

El flujo de trabajo principal se define en `pipeline/training_pipeline.py`, que orquesta los siguientes pasos:

1.  **Ingesta de Datos:**
    *   El script `src/data_ingestion.py` descarga el conjunto de datos de un bucket de Google Cloud Storage, como se especifica en `config/config.yaml`.
    *   Luego, divide los datos en conjuntos de entrenamiento y prueba y los guarda en el directorio `artifacts/raw/`.

2.  **Preprocesamiento de Datos:**
    *   El script `src/data_preprocessing.py` preprocesa los datos brutos. Esto incluye:
        *   Eliminar columnas innecesarias.
        *   Manejar características categóricas mediante codificación de etiquetas.
        *   Corregir la asimetría de los datos.
        *   Equilibrar el conjunto de datos mediante SMOTE (Técnica de sobremuestreo de minorías sintéticas).
        *   Seleccionar las características principales utilizando un clasificador de bosque aleatorio.
    *   Los datos procesados se guardan en el directorio `artifacts/processed/`.

3.  **Entrenamiento del Modelo:**
    *   El script `src/model_training.py` entrena un clasificador LightGBM con los datos preprocesados.
    *   Utiliza `RandomizedSearchCV` para el ajuste de hiperparámetros para encontrar los mejores parámetros del modelo.
    *   El modelo entrenado se evalúa utilizando métricas como exactitud, precisión, recall y F1-score.
    *   El mejor modelo se guarda en `artifacts/models/lgbm_model.pkl`.
    *   MLflow se utiliza para registrar el modelo, los parámetros, las métricas y los artefactos para el seguimiento de experimentos.

## Dependencias

Las bibliotecas de Python necesarias se enumeran en `requirements.txt`:

*   pandas
*   numpy
*   google-cloud-storage
*   scikit-learn
*   pyyaml
*   imbalanced-learn
*   lightgbm
*   mlflow
*   flask
*   seaborn

## Configuración

El proyecto se puede configurar utilizando los siguientes archivos:

*   **`config/config.yaml`**: Contiene la configuración para la ingesta de datos, como el nombre del bucket de GCS y el nombre del archivo.
*   **`config/model_params.py`**: Define el espacio de búsqueda de hiperparámetros para el modelo LightGBM.
*   **`config/paths_config.py`**: Especifica las rutas para varios artefactos de datos y modelos.

## Uso

Para ejecutar el pipeline de entrenamiento, ejecute el siguiente comando:

```bash
python pipeline/training_pipeline.py
```
