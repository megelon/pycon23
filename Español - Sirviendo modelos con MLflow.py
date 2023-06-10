# Databricks notebook source
# MAGIC %md # Sirviendo modelos con MLflow de cero a hero.
# MAGIC
# MAGIC Este tutorial cubre lo siguiente:
# MAGIC - Cómo importar datos de una máquina local al Databricks File System (DBFS).
# MAGIC - Visualización de datos usando Seaborn y matplotlib.
# MAGIC - Optimización de hiperparámetros paralelizada para entrenamiento de modelos de machine learning.
# MAGIC - Exploración de resultados de optimización de hiperparámetros con MLflow.
# MAGIC - Registro del mejor modelo en el Model Registry de MLflow.
# MAGIC - Uso del modelo registrado para generar predicciones usando otro set de datos con Spark UDF.
# MAGIC - Y finalmente configuración del model serving para entrega de recomendaciones con baja latencia.
# MAGIC
# MAGIC En este ejemplo, trabajaremos en precir la calidad del vino "Vinho Verde" basado en las propiedades fisicoquímicas.
# MAGIC
# MAGIC El ejemplo usa un set de datos del repositorio de Machine Learning de UCI, presentado en [*
# MAGIC Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC
# MAGIC ### Configuración
# MAGIC - Este notebook requiere Databricks Runtime 7.6+ para Machine Learning, el cuál incluye lo último de MLFlow, además de otros frameworks de Machine Learning tales como sklearn, PyTorch, TensorFlow, XGBoost, etc. No hay necesidad de intalarlos.
# MAGIC
# MAGIC - Sí se debe instalar mlflow hyperopt y xgboost

# COMMAND ----------

# MAGIC %pip install mlflow hyperopt xgboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Importación de Datos
# MAGIC   
# MAGIC En esta sección, descargaremos a local un dataset disponible en el archivo de una universidad para cargarlo en el Databricks File System (DBFS).
# MAGIC
# MAGIC 1. Vamos a https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ y descarguemos tanto `winequality-red.csv` como `winequality-white.csv` a nuestra máquina local.
# MAGIC
# MAGIC 1. Desde este libro de Databricks, seleccionemos *File* > *Upload Data*, y arrastraremos estos archivos a la opción que dice drag-and-drop en la consola de Databricks para cargarlos al Databricks File System (DBFS). 
# MAGIC
# MAGIC **Nota**: si no tiene la opción *File* > *Upload Data*, ud puede cargar el dataset usando la carpeta de ejemplos. Descomente las siguientes lineas en las seldas a continuación.
# MAGIC
# MAGIC 1. Seleccione *Next*. Código auto-generado para cargar los datos aparecerá. Seleccione *pandas*, copie el codigo de ejemplo.
# MAGIC
# MAGIC 1. Cree una neva celda, luego péguela en el código de ejemplo. Se verá similar al código en la celda subsiguiente. Haga los siguientes cambios:
# MAGIC   - Convierta `sep=';'` a `pd.read_csv`
# MAGIC   - Cambie los nombres de las variables de `df1` y `df2` a `white_wine` y `red_wine`, como muestra la celda subsiguiente.

# COMMAND ----------

dbutils.fs.cp("databricks-datasets/wine-quality/winequality-white.csv", "file:///tmp/winequality-white.csv")
dbutils.fs.cp("databricks-datasets/wine-quality/winequality-red.csv", "file:///tmp/winequality-red.csv")

# COMMAND ----------

dbutils.fs.ls("file:///tmp/winequality-white.csv")
dbutils.fs.ls("file:///tmp/")

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("file:///tmp/winequality-white.csv", sep=';')
red_wine = pd.read_csv("file:///tmp/winequality-red.csv", sep=';')

# COMMAND ----------

white_wine

# COMMAND ----------

red_wine

# COMMAND ----------

# MAGIC %md Unamos los dos DataFrames en un único set de datos, con una nueva carácterística (feature) binaria "is_red" que indica si el vino es rojo o blanco.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

data.head()

# COMMAND ----------

# MAGIC %md ##Visualización de datos
# MAGIC
# MAGIC Antes de entrenar el modelo, vamos a explorar el set de datos con Seabrorn y Matplotlib.

# COMMAND ----------

# MAGIC %md Histograma de la variable dependiente: Quality

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Parece que los scores de calidad se distibuyen de forma normal entre 3 y 9. 
# MAGIC
# MAGIC El vino lo vamos a definir como de alta calidad si su calidad (quality) está igual o por encima de 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Los "Box Plots" son útiles para identificar correlaciones entre una variable binaria y carácterisiticas (features)

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md Podemos evidenciar lo siguiente:
# MAGIC
# MAGIC - Para el caso del alcohol a mayor grado de alcohol mayor calidad.  
# MAGIC
# MAGIC - En el caso de la densidad, podemos ver que los vinos de baja calidad tienen una densidad mayor.

# COMMAND ----------

# MAGIC %md ## Procesamiento de datos.
# MAGIC
# MAGIC Antes de entrenar el modelo, vamos a revisar los valores faltantes y crearemos los set de datos de entrenamiento del modelo y validación de resultados.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md Podemos ver que no hay valores faltantes.

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

data

# COMMAND ----------

# MAGIC %md ## Vamos con nuestro primer modelo (que usaremos como base para mejoras)
# MAGIC
# MAGIC Usaremos un RANDOM FOREST, dado que trabajamos con un objetivo binario y puede haber interacciones entre múltiples variables.
# MAGIC
# MAGIC En el siguiente code snippet tenemos un clasificador simple en scikit-learn. 
# MAGIC
# MAGIC Además usamos flow para guardar métricas y el modelo como artefacto para usar después. 

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature

# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='pycon23_modelo_preferencias_vino'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  mlflow.pyfunc.log_model("random_forest_model_artifact", python_model=wrappedModel, signature=signature)

# COMMAND ----------

# MAGIC %md Examinemos la importancia de las características del modelo

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md Confirmamos nuestra hipótesis el alcohol y la densidad son importantes prediciendo la calidad del vino.

# COMMAND ----------

# MAGIC %md Podríamos ver en el MLflow las métricas del modelo recién entrenado.
# MAGIC
# MAGIC Tenemos un AUC de 0.89. 
# MAGIC
# MAGIC Un clasificador aleatorio habría tenido un AUC de 0.5 y entre más alto el AUC mejor. 

# COMMAND ----------

# MAGIC %md #### Esto cada vez se pone mejor, vamos a Registrar el modelo con MLflow Model Registry
# MAGIC
# MAGIC Al registrarlo podremos referenciarlo facilmente en cualquier momento al trabajar en Databricks.
# MAGIC
# MAGIC La siguiente sección muestra cómo hacerlo en código pero también lo podríamos hacer todo usando el UI. [Acá el paso a paso alternativo.](https://docs.databricks.com/applications/mlflow/model-registry.html#register-a-model-in-the-model-registry).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "pycon23_modelo_preferencias_vino"').iloc[0].run_id

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "pycon23_preferencias_vino_registrado". Try using a different name.
model_name = "pycon23_modelo_preferencias_vino_registrado"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model_artifact", model_name)

# COMMAND ----------

model_version

# COMMAND ----------

# MAGIC %md Ahora deberíamos poder ver el modelo en la sección "Models". 
# MAGIC
# MAGIC Ahora programáticamente vamos a pasar el modelo a Producción.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md La página de modelos ahora debería mostrar el modelo en el Stage "Production".
# MAGIC
# MAGIC El path del modelo es el siguiente "models:/wine-quality/production".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ## Vamos ahora a experimentar con un XGBOOST un nuevo modelo
# MAGIC
# MAGIC El random forest tuvo un buen rendimiento incluso sin hacer ajuste de hiperparámetros (hyperparameter tuning).
# MAGIC
# MAGIC Acá podríamos exigir a nuestros servidores y hacer el ajuste de hiperparámetros de tal forma que evaluemos múltiples modelos en paralelo, usando Hyperopt y SparkTrials. Para Pycon vamos a hacerlo un par de veces porque queremos que corra rápido. 
# MAGIC
# MAGIC Claramente también usaremos MLflow en esta etapa para guardar métricas de cada modelo y los modelos como artefacto para usar después. 

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb

search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}

def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)

    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}





# COMMAND ----------

# MAGIC %md
# MAGIC Justo acá seleccionamos el número de evaluaciones y la paralelización

# COMMAND ----------

parallelism_ = 2
max_evals_ = 3

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=parallelism_)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='pycon23_modelo_preferencias_vino_xgboost_1'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=max_evals_,
    trials=spark_trials
    #,rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md ## Vamos a MLFlow para analizar los resultados
# MAGIC
# MAGIC Vamos a Experiments Runs a comparar los modelos por 'auc'. El valor mayor es de 0.91. Tenemos un modelo mejor que el inicial!

# COMMAND ----------

# MAGIC %md Podemos hacer gráficas comparativas en el UI de mlflow para entender como los parámetros se correlacionan con el AUC. 
# MAGIC
# MAGIC Tal como vemos en el ejemplo abajo
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC
# MAGIC Vemos que las mejores corridasd tienen valores bajos de reg_lambda y learning_rate.
# MAGIC
# MAGIC Podemos correr otros optimización de hiperparámetros para explorar valores incluso menores para estros parámetros.

# COMMAND ----------

# MAGIC %md #### Actualicemos MLflow Model Registry con el nuevo mejor modelo para predecir la calidad del vino
# MAGIC
# MAGIC Ahora vamos guardar el mejor modelo en el model registry

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md Podemos ver todo esto en el UI también.
# MAGIC
# MAGIC Y ahora usemos el mejor modelo en producción

# COMMAND ----------

# Archive the old model version
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)

# COMMAND ----------


# Promote the new model version to Production
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md Ahora quienes llamen al modelo de calidad van a recibir las recomendaciones del xgboost no del random forest
# MAGIC
# MAGIC Así de fácil fue pasar a producción de nuevo.

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##Batch Inference
# MAGIC
# MAGIC Hay muchos escenarios en los que deberíamos evaluar el modelo en un nuevo set de datos. 
# MAGIC
# MAGIC Tomaremos un nuevo set de datos guardados en una tabla Delta usaremos spark para correr el proceso en paralelo.

# COMMAND ----------

# To simulate a new corpus of data, save the existing X_train data to a Delta table. 
# In the real world, this would be a new batch of data.
spark_df = spark.createDataFrame(X_train)
# Replace <username> with your username before running this cell.
table_path = "/user/<username>/delta/wine_data"

# Delete the contents of this path in case this cell has already been run
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------

# MAGIC %md Cargaremos el modelo en un Spark UDF para poder aplicarlo en la tabla delta.

# COMMAND ----------

import mlflow.pyfunc

apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

# COMMAND ----------

# Read the "new data" from Delta
new_data = spark.read.format("delta").load(table_path)

display(new_data)

# COMMAND ----------

from pyspark.sql.functions import struct

# Apply the model to the new data
udf_inputs = struct(*(X_train.columns.tolist()))

new_data = new_data.withColumn(
  "prediction",
  apply_model_udf(udf_inputs)
)

# COMMAND ----------

# Each row now has an associated prediction. Note that the xgboost function does not output probabilities by default, so the predictions are not limited to the range [0, 1].
display(new_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MODEL SERVING con MLFLOW 
# MAGIC
# MAGIC Paso NO disponible en la [edición community de databricks](https://community.cloud.databricks.com/login.html).
# MAGIC
# MAGIC Ahora vamos a crear un endpoint para servir los resultados del modelo [model serving](https://docs.databricks.com/machine-learning/model-serving/index.html).
# MAGIC
# MAGIC El código a continuación nos muestra como solicitar las resultados del modelo desplegado usando una REST API.

# COMMAND ----------

# MAGIC %md
# MAGIC Lo primero que vamos a hacer es crear nuestro token para hacer las solicitudes a nuestro endpoint. 
# MAGIC
# MAGIC Vamos al User Settings page (en el perfil arriba a la derecha), click en `Access Token` tab, y `Generate Token`. 
# MAGIC
# MAGIC Copiamos el toquen en la celda a continuación.

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapi1541924f1d84e692bd0f455479472696-3"

# COMMAND ----------

# MAGIC %md
# MAGIC Vamos a **Models** El tab de **serving**, y luego click en **Enable Serving**.
# MAGIC
# MAGIC Luego vamos a **Call The Model**, click en el botón de **Python** para tener el code snippet necesario para hacer las peticiones y lo pegamos en la celda a continuación. 
# MAGIC
# MAGIC También lo podemos usar por fuera de Databricks si quisieramos conectar con una aplicación mobile o lo que sea.
# MAGIC

# COMMAND ----------

    import os
    import requests
    import numpy as np
    import pandas as pd
    import json

    def create_tf_serving_json(data):
      return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

    def score_model(dataset):
      url = 'https://adb-2976239899983568.8.azuredatabricks.net/serving-endpoints/pycon_serfeliz_endpoint/invocations'
      headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 
    'Content-Type': 'application/json'}
      ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
      data_json = json.dumps(ds_dict, allow_nan=True)
      response = requests.request(method='POST', headers=headers, url=url, data=data_json)
      if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
      return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC Ahora vamos a evaluar que los resultados del endpoint y los resultados del modelo en batch sean los mismos.

# COMMAND ----------

# Model serving is designed for low-latency predictions on smaller batches of data
num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])

# COMMAND ----------

# Compare the results from the deployed model and the trained model
pd.DataFrame(served_predictions)

# COMMAND ----------

# Compare the results from the deployed model and the trained model
pd.DataFrame({ "Model Prediction":model_evaluations}) 

# COMMAND ----------

# MAGIC %md # Y esta etapa de mi vida se llama FELICIDAD.
# MAGIC
