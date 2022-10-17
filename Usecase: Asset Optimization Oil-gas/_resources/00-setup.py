# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %run ../../../_resources/00-global-setup $reset_all_data=$reset_all_data $db_prefix=manufacturing

# COMMAND ----------

# DBTITLE 1,Package imports
from delta.tables import *
import pyspark.sql.functions as F
import pyspark.sql.window as W
from pyspark.sql.window import Window
import seaborn as sns

# COMMAND ----------

raw_data_location = "/mnt/field-demos/manufacturing/oil_pump" 
cloud_storage_path = cloud_storage_path + "/oil_pump"

reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
if not spark._jsparkSession.catalog().tableExists("pump_enriched_full") or reset_all_data:
  print("setup demo tables for ML")
  spark.sql("drop table if exists pump_enriched_full")
  spark.read.load("/mnt/field-demos/manufacturing/oil_pump/pump_enriched_full").write.mode('overwrite').saveAsTable("pump_enriched_full")
  
  spark.sql("CREATE TABLE IF NOT EXISTS pump (country string, manufacturer string, pumpId int, region string)")
  spark.sql("COPY INTO pump FROM (SELECT country, manufacturer, cast(pumpId as int) as pumpId, region FROM '/mnt/field-demos/manufacturing/oil_pump/pump_data') FILEFORMAT = JSON")

  spark.sql("CREATE TABLE IF NOT EXISTS pump_maintenance (date date, maintenance boolean, pumpId int)")
  spark.sql("COPY INTO pump_maintenance FROM (SELECT cast(date as date), maintenance, cast(pumpId as int) as pumpId FROM '/mnt/field-demos/manufacturing/oil_pump/pump_maintenance') FILEFORMAT = JSON")

  spark.sql("CREATE TABLE IF NOT EXISTS pump_output (date date, window timestamp, pumpId int, oilOutput double)")
  spark.sql("COPY INTO pump_output FROM (SELECT cast(date as date), cast(window as timestamp) as window, cast(pumpId as int) as pumpId, oilOutput FROM '/mnt/field-demos/manufacturing/oil_pump/pump_output') FILEFORMAT = JSON")
  
  spark.sql("""CREATE OR REPLACE VIEW gold_readings AS
                  SELECT r.*, 
                    p.oilOutput, 
                    ifnull(m.maintenance,False) as maintenance
                  FROM pump_enriched_full r 
                    JOIN pump_output p ON (r.date=p.date AND r.window=p.window AND r.pumpId=p.pumpId)
                    LEFT JOIN pump_maintenance m ON (r.date=m.date AND r.pumpId=m.pumpId)""")

# COMMAND ----------

# Pyspark and ML Imports
import os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import xgboost as xgb
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
pd.options.plotting.backend = "plotly"
