# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md
# MAGIC # Progressive Cavity Pumps on Databricks - Asset Optimization
# MAGIC 
# MAGIC Asset-heavy industries like  oil & gas, manufacturing, chemical, utilities, transportation etc. are looking at ways to predictively optimize their assets.
# MAGIC 
# MAGIC We typically aim to increase revenue while reducing operational risks, including predictive maintenance to proactively fix issues before they occur, reducing the costly impact caused by downtime. 
# MAGIC 
# MAGIC A 2016 study of unplanned downtime by Baker-Hughes states unplanned equipment failure costs Energy companies an estimated $38 million in losses per year.
# MAGIC 
# MAGIC In addition, failures can have major environmental impacts. Being able to prevent mechanical issues such as oil/gas leaks before they happen is critical.
# MAGIC 
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Fmanufacturing%2Fpump_asset_optim%2Fnotebook_ingestion&dt=MANUFACTURING_PUMP_OPTIMIZATION">

# COMMAND ----------

# MAGIC %md
# MAGIC ##The ESG Resilience Loop
# MAGIC ### Integrating sustainability into field operations
# MAGIC 
# MAGIC By getting informations from our equiments, we can start integrating sustainability into our data motion and start building a positive loop.
# MAGIC 
# MAGIC <img src="https://westcacollateral.blob.core.windows.net/collateral/Integrating_Sustainability" width="900px">
# MAGIC 
# MAGIC Diagram : Emission reduction for Production Operations
# MAGIC  

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build a loop for Progressive Cavity Pumps Asset Optimization on Databricks
# MAGIC 
# MAGIC 
# MAGIC In this demo, we'll show how we can build a loop to start improving our equipment productivity and reduce failures / environmental impact.
# MAGIC 
# MAGIC As an Oil & Gas company, we operate thousands of Pumps to move oil in our Pipelines. Pump failures have a massive impact in our production and can have potential impact.
# MAGIC 
# MAGIC Databricks Lakehouse is uniquely positioned to implement such use-case. Within one platform and one layer of governance, the entire pipeline can be implemented, providing:
# MAGIC 
# MAGIC - **Project acceleration**: simplify ingestion, ML and Datawarehousing, moving advanced projects in production within weeks.
# MAGIC - **Simplification**: Databricks solves the low value, technological challenges for you, letting you focus on the business requirement.
# MAGIC - **Governance and security**: build once, share with multiple consumers while enforcing security & governance.
# MAGIC 
# MAGIC 
# MAGIC ### Data Flow
# MAGIC 
# MAGIC This is the flow we'll be implementing end to end:
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-0.png" width="1100px" />
# MAGIC 
# MAGIC This first notebook covers data engineering for the Engine Pump data. We'll focus on:
# MAGIC 
# MAGIC 
# MAGIC - **Data Ingestion** - stream real-time raw sensor data into the Delta format
# MAGIC - **Data Processing** - stream process sensor data from raw (Bronze) to silver (aggregated) to gold (enriched) Delta tables

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## About the Data
# MAGIC 
# MAGIC <img style="float: right" src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Progressive_cavity_pump_animation.gif" width="400">
# MAGIC 
# MAGIC Progressing Cavity Pumping (PCP) systems are pumping and moving fluid (in our case oil) across pipes. 
# MAGIC 
# MAGIC We receive from each pump 9 sensor values (speed, temperature, vibration...). 
# MAGIC 
# MAGIC Our business require this data to be aggregated at an hourly basis for analysis.
# MAGIC 
# MAGIC While final aggregations are at the hour level, we need to update them in near-realtime to quickly detect anomalies (we don't want to wait a full hour to trigger alarms).
# MAGIC 
# MAGIC Dataset: https://openei.org/datasets/

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ##Step 1 - Ingest sensor data
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-1.png" width="700px" />
# MAGIC 
# MAGIC 
# MAGIC The first step is to incrementally consume our sensor data. This is typically done consuming IOT hubs or queuing system like Kafka.
# MAGIC 
# MAGIC In this example, we receive our sensor data as file every minute. Let's see how Databricks Autoloader (`cloudFiles`) can incrementally ingest new files and help us with:
# MAGIC 
# MAGIC - Scalability: easily ingest millions of new files.
# MAGIC - Schema inference / evolution: handle schema change without friction.
# MAGIC - Simplicity: no need for extra maintenance, AutoLoader does the hard job for you.
# MAGIC - Stream or batch: decide your refreshness requirements (ex: every seconds or daily)

# COMMAND ----------

# DBTITLE 1,Read streaming data 
sensor_stream = (spark.readStream.format("cloudFiles")
                        .option("cloudFiles.format", "csv")
                        .option("cloudFiles.inferColumnTypes", "true")
                        .option("cloudFiles.maxFilesPerTrigger", "1") #For demo, to simulate streaming
                        .option("cloudFiles.schemaLocation", cloud_storage_path+"/schema_raw_sensor")
                        .load("/mnt/field-demos/manufacturing/oil_pump/raw_sensor_data")
                      .withColumn("date", to_date('eventTime')))
sensor_stream.createOrReplaceTempView("sensor_stream")
# Write our input stream to pump_bronze table
(sensor_stream.writeStream                                      
                .partitionBy('date')                                                          # Partition our data by Date for performance
                .option("checkpointLocation", cloud_storage_path + "/chkpt_bronze_pump")      # Checkpoint so we can restart streams gracefully
                .table("pump_bronze"))

# COMMAND ----------

# DBTITLE 1,Let's analyse data in real-time
# MAGIC %sql 
# MAGIC -- We can query the data directly from storage immediately as soon as it starts streams into Delta 
# MAGIC -- Plot X: eventTime and Y: sensor1, sensor3
# MAGIC SELECT min(sensor1) as sensor1, min(sensor2) as sensor2, date_trunc('hour', eventTime) as eventTime FROM sensor_stream GROUP BY date_trunc('hour', eventTime)

# COMMAND ----------

# DBTITLE 1,Simplify ingestion by removing painful technical issues with Databricks Lakehouse & Delta Lake
# MAGIC %sql
# MAGIC -- Streaming comes with lot of challenges, including small files issues. As example, Databricks handle technical details like compaction & files optimization out of the box!
# MAGIC ALTER TABLE pump_bronze SET TBLPROPERTIES (spark.databricks.delta.optimizeWrite.enabled=true, spark.databricks.delta.autoCompact.enabled=true);
# MAGIC SELECT * FROM pump_bronze;

# COMMAND ----------

# MAGIC %sql SELECT pumpId, count(sensor1) FROM pump_bronze GROUP BY pumpId ORDER BY pumpId;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Step 2 - Creating hourly aggregation in real-time
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-2.png" width="700px" />
# MAGIC 
# MAGIC The next step of our processing pipeline is to clean and aggregate the measurements to 1 hour intervals. 
# MAGIC 
# MAGIC While we aggregate the data for every hour, we don't want to wait 1h to get the status of the current sensors. 
# MAGIC 
# MAGIC Therefore we'll constantly update the current hour aggregation to provide the business with real-time information.
# MAGIC 
# MAGIC This mean that every X seconds, we'll recompute the hourly window and update the results to our Silver table.
# MAGIC 
# MAGIC Since we'll be doing upsert on the current window, we'll leverage the Delta Lake [**MERGE**](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/merge-into?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json) functionality to upsert records. 
# MAGIC 
# MAGIC *Note: while this notebook is in Python, we could greatly simplify the MERGE operation using Delta Live Table `APPLY INTO` logic.*

# COMMAND ----------

# DBTITLE 1,Generic function to UPSERT the hourly window (insert if it's a new window or replace if exists)
# Create function to merge data into target Delta table
def merge_to_delta_table(incremental, target): 
  # If the †arget table does not exist, create one
  if not spark._jsparkSession.catalog().tableExists(target):
    incremental.write.format("delta").partitionBy("date").saveAsTable(target)
    spark.sql(f"ALTER TABLE {target} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
  else:
    #Get the min/max to avoid updating the entire table and filter on the partition
    min_date, max_date = spark.read.table("pump_bronze").agg(F.min("date"), F.max("date")).collect()[0]
    #Delta MERGE in python
    DeltaTable.forName(spark, target).alias('t').merge(
      incremental.alias('u'), f"u.window = t.window AND u.pumpId = t.pumpId and t.date between '{str(min_date)}' and '{str(max_date)}'"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    
def merge_pump_agg(incremental_df, i): 
  incremental_df = incremental_df.dropDuplicates(['window','pumpId'])
  merge_to_delta_table(incremental_df, "pump_agg_silver")

# COMMAND ----------

# DBTITLE 1,Stream data to silver tables, add silver table to metastore
(spark.readStream.table("pump_bronze")
        .withWatermark("eventTime", "6 hours")                                     # Discard messages being late more than 6hours
        .groupBy('pumpId', 'date', F.window('eventTime', '1 minutes'))             # Aggregate readings to hourly intervals
        .agg(*[F.avg(f'sensor{i}').alias(f'sensor{i}') for i in range(1, 10)])
      .writeStream                                                                 
        .foreachBatch(merge_pump_agg)                                              # Pass each micro-batch to a function
        .outputMode("update")                                                      # update mode will output partial results
        .option("checkpointLocation", cloud_storage_path+"/chkpt_pump_agg_silver") # Checkpoint so we can restart streams gracefully
        .start())

# COMMAND ----------

# MAGIC %sql SELECT * FROM pump_agg_silver

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Step 3: enrich the table with pump data
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-3.png" width="700px" />
# MAGIC 
# MAGIC Next we perform a join to create one enriched dataset, adding pump metadata information.
# MAGIC 
# MAGIC We'll have to consume in near real-time the silver tables changes to get all the updates in the hourly window (we'll consume the partial changes and UPSERT the data in the next table). 
# MAGIC 
# MAGIC Delta can easily capture the UPSERT operations from the previous table with Change Data Flow (CDF) and `table_change` (enabled with `delta.enableChangeDataFeed = true`)
# MAGIC 
# MAGIC *Please read the [CDF Documentation](https://docs.databricks.com/delta/delta-change-data-feed.html) or check our CDF demo for more details.*
# MAGIC 
# MAGIC But first, we have to load the metadata `pump` table. For that we'll use a simple COPY INTO command to load our data. 
# MAGIC 
# MAGIC *Note that COPY INTO is idempotent: if you run it twice it'll only process new data.*

# COMMAND ----------

# DBTITLE 1,Create and load our pump metadata table
# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS pump (country string, manufacturer string, pumpId int, region string);
# MAGIC 
# MAGIC COPY INTO pump FROM
# MAGIC   (SELECT country, manufacturer, cast(pumpId as int) as pumpId, region FROM '/mnt/field-demos/manufacturing/oil_pump/pump_data')
# MAGIC FILEFORMAT = JSON;
# MAGIC 
# MAGIC SELECT * FROM pump;

# COMMAND ----------

# DBTITLE 1,Join our hourly aggregates with the pump table (Stream - batch join)
#Note the readChangeData option used to capture all changes from the tables (we'll insert, updates and deletes)
df_pump_agg_stream = spark.readStream.option("readChangeData", "true").option("startingVersion", 1).table('pump_agg_silver')
df_pump_enriched = df_pump_agg_stream.join(spark.table("pump"), "pumpId")

# COMMAND ----------

# DBTITLE 1,Ingest table changes with Delta CDF
from delta.tables import *   
from pyspark.sql.window import Window
from pyspark.sql.functions import dense_rank
    
# Create function to merge pump data into target Delta table
def merge_pump_enriched(incremental, i): 
  #First we need to deduplicate based on the id and take the most recent update (we can consume multiple changes and want to keep the most recent)
  windowSpec = Window.partitionBy(['window','pumpId']).orderBy(col("_commit_version").desc())
  data_deduplicated = incremental.withColumn("rank", dense_rank().over(windowSpec)).where("rank = 1 and _change_type!='update_preimage'").drop("_commit_version", "rank", "_commit_timestamp", "_change_type")
  merge_to_delta_table(data_deduplicated, "pump_enriched_gold")
  
merge_gold_stream = (df_pump_enriched.writeStream 
                                      .foreachBatch(merge_pump_enriched)
                                      .option("checkpointLocation", cloud_storage_path+"/chkpt_pump_enriched_gold") # Checkpoint so we can restart streams gracefully
                                      .start())

# COMMAND ----------

# DBTITLE 1,Enriched table (with country/region data)
# MAGIC %sql SELECT * FROM pump_enriched_gold;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Step 4: Load our alternative datasets
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-4.png" width="700px" />
# MAGIC 
# MAGIC Our last step will be simple: we need to load a few other datasets required by our Data Scientists. This includes:
# MAGIC 
# MAGIC - The pumps status history (healthy/damaged), that we'll use to label our dataset and train the model
# MAGIC - The pump output
# MAGIC 
# MAGIC As this dataset is available as CSV files and not updating quite often, we'll simply load them using a COPY INTO statement.

# COMMAND ----------

# DBTITLE 1,Maintenance history
# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS pump_maintenance (date date, maintenance boolean, pumpId int);
# MAGIC 
# MAGIC COPY INTO pump_maintenance FROM
# MAGIC   (SELECT cast(date as date), maintenance, cast(pumpId as int) as pumpId FROM '/mnt/field-demos/manufacturing/oil_pump/pump_maintenance')
# MAGIC FILEFORMAT = JSON;
# MAGIC 
# MAGIC SELECT * FROM pump_maintenance;

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS pump_output (date date, window timestamp, pumpId int, oilOutput double);
# MAGIC 
# MAGIC COPY INTO pump_output FROM
# MAGIC   (SELECT cast(date as date), cast(window as timestamp) as window, cast(pumpId as int) as pumpId, oilOutput FROM '/mnt/field-demos/manufacturing/oil_pump/pump_output')
# MAGIC FILEFORMAT = JSON;
# MAGIC 
# MAGIC SELECT * FROM pump_output;

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ### Creating the final table (view) for ML analysis
# MAGIC 
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/manufacturing/oil_gas_pump/oil-gas-pump-flow-5.png" width="700px" />
# MAGIC 
# MAGIC We have now loaded all our data.
# MAGIC 
# MAGIC Because it's not worth materializing the data, we'll create a view joining all these tables which will be made available to our Data Scientists team.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create a data science view that joins all 3 tables together
# MAGIC CREATE OR REPLACE VIEW gold_readings AS
# MAGIC SELECT r.*, 
# MAGIC   p.oilOutput, 
# MAGIC   ifnull(m.maintenance,False) as maintenance
# MAGIC FROM pump_enriched_full r 
# MAGIC   JOIN pump_output p ON (r.date=p.date AND r.window=p.window AND r.pumpId=p.pumpId)
# MAGIC   LEFT JOIN pump_maintenance m ON (r.date=m.date AND r.pumpId=m.pumpId);
# MAGIC   
# MAGIC SELECT * FROM gold_readings ORDER BY pumpId, window

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Databricks Governance with Unity Catalog
# MAGIC 
# MAGIC Leveraging Unity Catalog, we can grant a SELECT access to our Data Scientist Team. This is one of the key value of the Lakehouse: single source of truth for all your use-cases, with security & governance.
# MAGIC 
# MAGIC This is done using plain SQL command:

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Note: this won't run as we don't have our data scientist team defined.
# MAGIC GRANT SELECT ON TABLE gold_readings TO `data_scientist_team`

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Optimizing the file layout for future queries
# MAGIC 
# MAGIC As we know this table will be often requested by pumpID, we'll optimize the table by these field. 
# MAGIC 
# MAGIC All future requets will be much faster!

# COMMAND ----------

# DBTITLE 1,Performance optimizations
# MAGIC %sql
# MAGIC -- Optimize all 3 tables for querying and model training performance
# MAGIC OPTIMIZE pump_enriched_gold WHERE date < current_date() ZORDER BY pumpId, window;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Next: Data Science And Analytics
# MAGIC 
# MAGIC That's it! Our tables are now ready for the Data Science team.
# MAGIC 
# MAGIC Open the next notebook to build our [asset optimization models]($./02-Asset-optimization-Predictive-maintenance) and improve our pump efficiency while increasing their lifetime.
