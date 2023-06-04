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

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

data.head()

# COMMAND ----------

# MAGIC %md ##Data Visualization
# MAGIC
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

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

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocessing Data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["quality"], axis=1)
X_test = test.drop(["quality"], axis=1)
y_train = train.quality
y_test = test.quality

data

# COMMAND ----------

# MAGIC %md ## Building a Baseline Model
# MAGIC
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

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

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md As illustrated by the boxplots shown previously, both alcohol and density are important in predicting quality.

# COMMAND ----------

# MAGIC %md You logged the Area Under the ROC Curve (AUC) to MLflow. Click **Experiment** at the upper right to display the Experiment Runs sidebar. 
# MAGIC
# MAGIC The model achieved an AUC of 0.89. 
# MAGIC
# MAGIC A random classifier would have an AUC of 0.5, and higher AUC values are better. For more information, see [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

# COMMAND ----------

# MAGIC %md #### Registering the model in the MLflow Model Registry
# MAGIC
# MAGIC By registering this model in the Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI by following the steps in [Register a model in the Model Registry
# MAGIC ](https://docs.databricks.com/applications/mlflow/model-registry.html#register-a-model-in-the-model-registry).

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

# MAGIC %md You should now see the wine-quality model in the Models page. To display the Models page, click the Models icon in the left sidebar. 
# MAGIC
# MAGIC Next, transition this model to production and load it into this notebook from the model registry.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md The Models page now shows the model version in stage "Production".
# MAGIC
# MAGIC You can now refer to the model using the path "models:/wine-quality/production".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##Experimenting with a new model
# MAGIC
# MAGIC The random forest model performed well even without hyperparameter tuning.
# MAGIC
# MAGIC The following code uses the xgboost library to train a more accurate model. It runs a parallel hyperparameter sweep to train multiple
# MAGIC models in parallel, using Hyperopt and SparkTrials. As before, the code tracks the performance of each parameter configuration with MLflow.

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

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=2)



# COMMAND ----------

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
with mlflow.start_run(run_name='pycon23_modelo_preferencias_vino_xgboost_1'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=3,
    trials=spark_trials
    #,rstate=np.random.RandomState(123)
  )

# COMMAND ----------

# MAGIC %md  #### Use MLflow to view the results
# MAGIC Open up the Experiment Runs sidebar to see the MLflow runs. Click on Date next to the down arrow to display a menu, and select 'auc' to display the runs sorted by the auc metric. The highest auc value is 0.91. You beat the baseline!
# MAGIC
# MAGIC MLflow tracks the parameters and performance metrics of each run. Click the External Link icon <img src="https://docs.databricks.com/_static/images/external-link.png"/> at the top of the Experiment Runs sidebar to navigate to the MLflow Runs Table.

# COMMAND ----------

# MAGIC %md Now investigate how the hyperparameter choice correlates with AUC. Click the "+" icon to expand the parent run, then select all runs except the parent, and click "Compare". Select the Parallel Coordinates Plot.
# MAGIC
# MAGIC The Parallel Coordinates Plot is useful in understanding the impact of parameters on a metric. You can drag the pink slider bar at the upper right corner of the plot to highlight a subset of AUC values and the corresponding parameter values. The plot below highlights the highest AUC values:
# MAGIC
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/end-to-end-example/parallel-coordinates-plot.png"/>
# MAGIC
# MAGIC Notice that all of the top performing runs have a low value for reg_lambda and learning_rate. 
# MAGIC
# MAGIC You could run another hyperparameter sweep to explore even lower values for these parameters. For simplicity, that step is not included in this example.

# COMMAND ----------

# MAGIC %md 
# MAGIC You used MLflow to log the model produced by each hyperparameter configuration. The following code finds the best performing run and saves the model to the model registry.

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md #### Updating the production wine_quality model in the MLflow Model Registry
# MAGIC
# MAGIC Earlier, you saved the baseline model to the Model Registry under "wine_quality". Now that you have a created a more accurate model, update wine_quality.

# COMMAND ----------

#dbutils.fs.ls('dbfs:/databricks/mlflow-tracking/596682901737901/858737b9e114459eb34796b36f94ac31/artifacts/model')

# COMMAND ----------


new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md Click **Models** in the left sidebar to see that the wine_quality model now has two versions. 
# MAGIC
# MAGIC The following code promotes the new version to production.

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

# MAGIC %md Clients that call load_model now receive the new model.

# COMMAND ----------

# This code is the same as the last block of "Building a Baseline Model". No change is required for clients to get the new model!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md ##Batch Inference
# MAGIC
# MAGIC There are many scenarios where you might want to evaluate a model on a corpus of new data. For example, you may have a fresh batch of data, or may need to compare the performance of two models on the same corpus of data.
# MAGIC
# MAGIC The following code evaluates the model on data stored in a Delta table, using Spark to run the computation in parallel.

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

# MAGIC %md Load the model into a Spark UDF, so it can be applied to the Delta table.

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
# MAGIC ## Model serving (here we will need a free account, with full permisions to create the token)
# MAGIC
# MAGIC To productionize the model for low latency predictions, use MLflow [model serving](https://docs.databricks.com/applications/mlflow/model-serving.html) to deploy the model to an endpoint.
# MAGIC
# MAGIC The following code illustrates how to issue requests using a REST API to get predictions from the deployed model.

# COMMAND ----------

# MAGIC %md
# MAGIC You need a Databricks token to issue requests to your model endpoint. You can generate a token from the User Settings page (under the profile icon on the upper right), click on `Access Token` tab, and `Generate Token`. Copy the token into the next cell.

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapi1541924f1d84e692bd0f455479472696-3"

# COMMAND ----------

# MAGIC %md
# MAGIC Click **Models** in the left sidebar and navigate to the registered wine model. Click the serving tab, and then click **Enable Serving**.
# MAGIC
# MAGIC Then, under **Call The Model**, click the **Python** button to display a Python code snippet to issue requests. Copy the code into this notebook. It should look similar to the code in the next cell. 
# MAGIC
# MAGIC You can use the token to make these requests from outside Databricks notebooks as well.

# COMMAND ----------

    import os
    import requests
    import numpy as np
    import pandas as pd
    import json

    def create_tf_serving_json(data):
      return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

    def score_model(dataset):
      url = 'https://adb-2976239899983568.8.azuredatabricks.net/serving-endpoints/pycon23_cheers/invocations'
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
# MAGIC The model predictions from the endpoint should agree with the results from locally evaluating the model.

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
