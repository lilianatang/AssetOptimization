# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md # Asset optimization and Predictive Maintenance of Progressive Cavity Pumps
# MAGIC 
# MAGIC ## Bringing value with Machine Learning
# MAGIC 
# MAGIC As discussed in our [previous notebook]($./01-Pump-Ingestion-Data-Engineering), our Oil & Gas company has identified that properly tuning our pumps can increase profit and reduce potential environmental impact. 
# MAGIC 
# MAGIC Our requirements are the following:
# MAGIC 
# MAGIC - Predict & reduce pump maintenance
# MAGIC - Forecast pump output in the next hours 
# MAGIC - Adjust pump pressure to ensure the best output (correlated to revenues) while lowering failure risk/maintenance.
# MAGIC 
# MAGIC Our Data Engineering team has produced the dataset for us. Let's now see how our Data Scientist team can leverage this information to build a Predictive Maintenance model.
# MAGIC 
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fmanufacturing%2Fpump_asset_optim%2Fnotebook_ml&dt=MANUFACTURING_PUMP_OPTIMIZATION">

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md ## Implementing Asset Optimization
# MAGIC 
# MAGIC Now that our data is flowing reliably from our sensor devices into an enriched Delta table in Data Lake storage, we can start to build ML models to predict the remaining life of our assets using historical sensor and forecast the future oil output.
# MAGIC 
# MAGIC Therefore, we'll create two models ***for each Progressive Capacity Pump***:
# MAGIC 
# MAGIC 1. Pump Oil Output - using current readings for pump operating parameters (pressure, torque, fluid levels, inflows) and other operational sources, predict the expected oil production 6 hours from now
# MAGIC 2. Pump Remaining Life - predict the remaining life in days until the next maintenance event
# MAGIC 
# MAGIC Ultimately, we'll use these 2 results to define the best pressure to be used per pump, optimizing our revenues while reducing maintenance.
# MAGIC 
# MAGIC *Note that another option would be to create 1 model for a group of similar pump, based on external clustering factors such as geo-location, usage, pipeline etc.*
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-ml-flow-0.png" width="1000px">
# MAGIC 
# MAGIC We will use the XGBoost framework to train regression models. Due to the size of the data and number of pumps, we will use Spark UDFs to distribute training across all the nodes in our cluster.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 1. Preparing and visualizing our dataset 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-ml-flow-1.png" width="700px" style="float: right">
# MAGIC 
# MAGIC In order to predict oil output 6 hours ahead, we need to first time-shift our data to create our label column. We can do this easily using Spark Window partitioning. 
# MAGIC 
# MAGIC In order to predict remaining life, we need to backtrace the remaining life from the maintenance events. We can do this easily using cross joins. The following diagram illustrates the ML Feature Engineering pipeline:

# COMMAND ----------

# DBTITLE 1,Pump age 
# MAGIC %sql
# MAGIC -- Calculate the age of each pump and the remaining life in days
# MAGIC CREATE OR REPLACE VIEW pump_age AS
# MAGIC WITH reading_dates AS (SELECT distinct date, pumpId FROM pump_output),
# MAGIC   maintenance_dates AS (
# MAGIC     SELECT d.*, datediff(nm.date, d.date) as datediff_next, datediff(d.date, lm.date) as datediff_last 
# MAGIC     FROM reading_dates d LEFT JOIN pump_maintenance nm ON (d.pumpId=nm.pumpId AND d.date<=nm.date)
# MAGIC     LEFT JOIN pump_maintenance lm ON (d.pumpId=lm.pumpId AND d.date>=lm.date ))
# MAGIC SELECT date, pumpId, ifnull(min(datediff_last),0) AS age, ifnull(min(datediff_next),0) AS remaining_life
# MAGIC FROM maintenance_dates 
# MAGIC GROUP BY pumpId, date;
# MAGIC 
# MAGIC SELECT * FROM pump_age;

# COMMAND ----------

# DBTITLE 1,Final feature dataset, with a 6h rolling window for our predictions
# MAGIC %sql 
# MAGIC -- Calculate the power 6 hours ahead using Spark Windowing and build a feature_table to feed into our ML models
# MAGIC CREATE OR REPLACE VIEW feature_table AS
# MAGIC SELECT r.*, age, remaining_life,
# MAGIC --72 rows ahead = 72 5 minute windows = 6 hours
# MAGIC   LEAD(oilOutput, 72, oilOutput) OVER (PARTITION BY r.pumpId ORDER BY window) as output_6_hours_ahead
# MAGIC FROM gold_readings r JOIN pump_age a ON (r.date=a.date AND r.pumpId=a.pumpId)
# MAGIC WHERE r.date < CURRENT_DATE();
# MAGIC 
# MAGIC SELECT * FROM feature_table

# COMMAND ----------

# DBTITLE 1,Age & remaining life per pump
# MAGIC %sql
# MAGIC SELECT date, avg(age) as age, avg(remaining_life) as life FROM feature_table WHERE pumpId='19739' GROUP BY date ORDER BY date
# MAGIC -- Area Chart. Keys: date. Values: age, life.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 2 Distributed Model Training per pump
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-ml-flow-2.png" width="700px" style="float: right">
# MAGIC 
# MAGIC We'll leverage [Pandas UDFs](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) to group our dataset per pump and build small individual model for each pump. 
# MAGIC 
# MAGIC We'll then register all the models in MLFlow, tracking the pumpID in mlflow metadata.

# COMMAND ----------

# MAGIC %md ### 2.1 Oil Output prediction
# MAGIC 
# MAGIC The first model will be used to determine the oil output in the next 6 hours.
# MAGIC 
# MAGIC #### Automated Model Tracking in Databricks
# MAGIC 
# MAGIC As you train the models, notice how Databricks-managed MLflow automatically tracks each run in the "Runs" tab of the notebook. 
# MAGIC 
# MAGIC You can open each run and view the parameters, metrics, models and model artifacts that are captured by MLflow Autologging. For XGBoost Regression models, MLflow tracks: 
# MAGIC 
# MAGIC 1. Any model parameters (alpha, colsample, learning rate, etc.) passed to the `params` variable
# MAGIC 2. Metrics specified in `evals` (RMSE by default)
# MAGIC 3. The trained XGBoost model file
# MAGIC 4. Feature importances
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iiot_mlflow_tracking.gif" width=800>

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Create a function to train a XGBoost Regressor on a turbine's data
def train_distributed_xgb(readings_pd, model_type, label_col, prediction_col, run_id, xp_id):
  mlflow.xgboost.autolog(log_input_examples=True, silent=True)
  with mlflow.start_run(run_id=run_id, experiment_id=xp_id):
    with mlflow.start_run(nested = True, experiment_id=xp_id):

      train, test = train_test_split(readings_pd, test_size=0.2)
      # Log the model type and device ID
      mlflow.log_param('pumpId', readings_pd['pumpId'][0])
      mlflow.set_tag('model', model_type)
  
      # Train an XGBRegressor on the data for this Turbine
      alg = xgb.XGBRegressor() 
      train_dmatrix = xgb.DMatrix(data=train[feature_cols].astype('float'), label=train[label_col])
      test_dmatrix = xgb.DMatrix(data=test[feature_cols].astype('float'), label=test[label_col])
      params = {'learning_rate': 0.5, 'alpha':10, 'colsample_bytree': 0.5, 'max_depth': 5}
      model = xgb.train(params=params, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train'), (test_dmatrix, 'test')])

      # Make predictions on the dataset and return the results
      readings_pd[prediction_col] = model.predict(xgb.DMatrix(data=readings_pd[feature_cols].astype('float')))
      return readings_pd
    
with mlflow.start_run(run_name='output_prediction', experiment_id=e.experiment_id) as run:

  # Create a Spark Dataframe that contains the features and labels we need
  non_feature_cols = ['date','window','pumpId','remaining_life']
  feature_cols = ['age','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9','oilOutput']
  label_col = 'output_6_hours_ahead'
  prediction_col = label_col + '_predicted'

  # Read in our feature table and select the columns of interest
  feature_df = spark.table('feature_table').selectExpr(non_feature_cols + feature_cols + [label_col] + [f'0 as {prediction_col}'])

  # Register a Pandas UDF to distribute XGB model training using Spark
  def train_output_models(readings_pd):
    return train_distributed_xgb(readings_pd, 'field_demos_pump_output_prediction', label_col, prediction_col, run.info.run_id, e.experiment_id)

  e = init_experiment_for_batch("manuf_oil_pump", "demos_pump_predictive_maintenance")

  # Run the Pandas UDF against our feature dataset - this will train 1 model per pump and write the predictions to a table
  output_predictions = (
    feature_df.filter('pumpid="19739"') #For the demo filter in 1 single pump for faster model training
     .groupBy('pumpId').applyInPandas(train_output_models, schema=feature_df.schema)
     .write.format("delta").mode("overwrite")
     .partitionBy("date")
     .saveAsTable("pump_output_predictions"))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM pump_output_predictions

# COMMAND ----------

data = spark.sql("""SELECT date, avg(output_6_hours_ahead) as actual,  avg(output_6_hours_ahead_predicted) as predicted 
                      FROM pump_output_predictions  WHERE pumpId = '19739' AND output_6_hours_ahead > 800 GROUP BY date""").toPandas()
sns.scatterplot(data=data, x="predicted", y="actual")

# COMMAND ----------

# MAGIC %md ### 2.2 Predict Remaining Life
# MAGIC 
# MAGIC Our second model predicts the remaining useful life of each pump based on the current operating conditions. 
# MAGIC 
# MAGIC We'll leverage historical maintenance (indicating when a replacement activity occured) to calculate the remaining life as our training label. 
# MAGIC 
# MAGIC Once again, we train an XGBoost model for each pump to predict the remaining life given a set of operating parameters

# COMMAND ----------

with mlflow.start_run(run_name='life_prediction', experiment_id=e.experiment_id) as run:
  # Create a Spark Dataframe that contains the features and labels we need
  non_feature_cols = ['date','window','pumpId','output_6_hours_ahead_predicted']
  feature_cols = ['age','sensor1','sensor2','sensor3','sensor4','sensor5','sensor6','sensor7','sensor8','sensor9','oilOutput']
  label_col = 'remaining_life'
  prediction_col = label_col + '_predicted'

  # Read in our feature table and select the columns of interest
  feature_df = spark.table('pump_output_predictions').selectExpr(non_feature_cols + feature_cols + [label_col] + [f'0 as {prediction_col}']).filter('pumpid="19739"')

  # Register a Pandas UDF to distribute XGB model training using Spark
  def train_life_models(readings_pd):
    return train_distributed_xgb(readings_pd, 'field_demos_pump_life_prediction', label_col, prediction_col, run.info.run_id, e.experiment_id)

  # Run the Pandas UDF against our feature dataset - this will train 1 model per pump and write the predictions to a table
  life_predictions = (
    feature_df.groupBy('pumpId').applyInPandas(train_life_models, schema=feature_df.schema)
      .write.format("delta").mode("overwrite")
      .partitionBy("date")
      .saveAsTable("pump_life_predictions"))


# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT date, avg(remaining_life) as Actual_Life, avg(remaining_life_predicted) as Predicted_Life 
# MAGIC   FROM pump_life_predictions  WHERE pumpId='19739' GROUP BY date ORDER BY date
# MAGIC -- Line chart - Keys: date, Values: Actual_Life, Predicted_Life

# COMMAND ----------

# MAGIC %md ## 3 Model Deployment
# MAGIC 
# MAGIC Our models have been stored in MLFlow. We can leverage it for batch or real-time serving.
# MAGIC 
# MAGIC We can easily deploy our pumps model in MLFlow Registry and then retrieve the model later for our inferences. Let's see how this can be done using the UI: 
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/mlflow_register.gif" width=800>
# MAGIC 
# MAGIC Of course, for real-world usage we would leverage MLFlow API to programatically pilote our entire pump fleet.
# MAGIC 
# MAGIC Let's see how to do that for our pump #19739

# COMMAND ----------

# DBTITLE 1,Get our first model for the pump 19739
best_models = mlflow.search_runs(filter_string='tags.model="field_demos_pump_output_prediction" and params.pumpId=19739 and attributes.status = "FINISHED" and metrics.`test-rmse` > 0', order_by=['metrics.`test-rmse` ASC'], max_results=1)
output_model_registered = mlflow.register_model("runs:/" + best_models.iloc[0].run_id + "/model", "field_demos_pump_19739_output")
#Second model
best_models = mlflow.search_runs(filter_string='tags.model="field_demos_pump_life_prediction" and params.pumpId=19739 and attributes.status = "FINISHED" and metrics.`test-rmse` > 0', order_by=['metrics.`test-rmse` ASC'], max_results=1)
lifetime_model_registered = mlflow.register_model("runs:/" + best_models.iloc[0].run_id + "/model", "field_demos_pump_19739_life")

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(name = f"field_demos_pump_19739_output", version = output_model_registered.version, stage = "Production", archive_existing_versions=True)
client.transition_model_version_stage(name = f"field_demos_pump_19739_life", version = lifetime_model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# DBTITLE 1,Running inferences
output_model = mlflow.pyfunc.load_model(model_uri=f"models:/field_demos_pump_19739_output/Production")
life_model = mlflow.pyfunc.load_model(model_uri=f"models:/field_demos_pump_19739_life/Production")

payload = {
  'age':       20.0,
  'sensor1':   -0.01,
  'sensor2':   300.0,
  'sensor3':   -0.005,
  'sensor4':   -0.1,
  'sensor5':   -0.2,
  'sensor6':   0.01,
  'sensor7':   -0.07,
  'sensor8':   -0.3,
  'sensor9':   0.006,
  'oilOutput': 980.0
}
print(f"Predicted Age: {life_model.predict(pd.DataFrame([payload]))}")
print(f"Predicted Output in 6 hours: {output_model.predict(pd.DataFrame([payload]))}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### 4 - Asset Optimization
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-ml-flow-3.png" width="700px" style="float: right">
# MAGIC 
# MAGIC We can now predict the pump output and the life time. 
# MAGIC 
# MAGIC Based on these 2 models, we can now identify the optimal operating conditions for maximizing output while also maximizing asset useful life. 
# MAGIC 
# MAGIC Let's define our Revenue as a function of the output
# MAGIC 
# MAGIC \\(Revenue = Price\displaystyle\sum_1^{365} Output_t\\)
# MAGIC 
# MAGIC And the operating cost as a function of lifestyle
# MAGIC 
# MAGIC \\(Cost = {365 \over Life_{pressure}} Price \displaystyle\sum_1^{24} Output_t \\)
# MAGIC 
# MAGIC 
# MAGIC \\(Profit = Revenue - Cost\\)
# MAGIC 
# MAGIC \\(Output_t\\) and \\(Life\\) will be calculated by scoring many different Pressure values. The results can be visualized to identify the Pressure that yields the highest profit.

# COMMAND ----------

# Iterate through 50 different pressure configurations and capture the predicted output and remaining life at each pressure
results = []
for pressure in np.arange(-0.015,-0.007,0.001):
  payload['sensor1'] = pressure.item()
  expected_life = life_model.predict(pd.DataFrame([payload]))[0]
  expected_output = life_model.predict(pd.DataFrame([payload]))[0]
  results.append((pressure, expected_output, expected_life))
  
# Calculalte the Revenue, Cost and Profit generated for each RPM configuration
optimization_df = pd.DataFrame(results, columns=['Pressure', 'Expected Output', 'Expected Life'])
optimization_df['Revenue'] = optimization_df['Expected Output'] * 24 * 365
optimization_df['Cost'] = optimization_df['Expected Output'] * 24 * 365 / optimization_df['Expected Life']
optimization_df['Profit'] = optimization_df['Revenue'] + optimization_df['Cost']

optimization_df.plot(kind='line', x='Pressure', y='Profit', template="simple_white")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## 5 - Optimize or pump fleet & track KPIs
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-ml-flow-4.png" width="700px" style="float: right">
# MAGIC 
# MAGIC The optimal operating parameters for **Pump 19739** given the specified operating conditions is **Sensor1 = -0.013** for generating a maximum profit of **$9.15M**! 
# MAGIC 
# MAGIC This information can be send to our control center to dynamically optimize the pumps pressure over its lifetime, using real-time messaging system / REST api.
# MAGIC 
# MAGIC additionally, the global results can be tracked as a Dashboard leveraging DBSQL or any BI tool.
# MAGIC 
# MAGIC *Note: Your results may vary due to the random nature of the sensor readings. *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion: Integrating sustainability into field operations
# MAGIC 
# MAGIC As you apply A.I. to more field operations , you can see the impact to both profitability, and sustainability:
# MAGIC 
# MAGIC <img src="https://westcacollateral.blob.core.windows.net/collateral/Integrating_Sustainability_Final.png" width="1000">
