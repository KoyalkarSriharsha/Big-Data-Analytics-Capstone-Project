# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit, input_file_name
from pyspark.sql.types import *
from datetime import datetime

# Initialize Spark Session
spark = SparkSession.builder.appName("Customer360_Bronze_Ingestion").getOrCreate()

# Configuration
RAW_DATA_PATH = "/Volumes/workspace/default/rawdata"
BRONZE_PATH = "/Volumes/workspace/default/bronze"
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"

# Create bronze directory if it doesn't exist
dbutils.fs.mkdirs(BRONZE_PATH)

print("✅ Configuration loaded")
print(f"📂 Raw Data Path: {RAW_DATA_PATH}")
print(f"📂 Bronze Path: {BRONZE_PATH}")

# COMMAND ----------

def ingest_to_bronze(file_name, table_name, schema=None):
    """
    Generic function to ingest CSV files into Bronze Delta tables
    
    Parameters:
    - file_name: Name of the CSV file (e.g., 'customers.csv')
    - table_name: Name of the bronze table (e.g., 'customers_bronze')
    - schema: Optional schema definition for explicit typing
    """
    
    print(f"\n{'='*60}")
    print(f"🔄 Ingesting {file_name} → {table_name}")
    print(f"{'='*60}")
    
    # Read CSV file
    file_path = f"{RAW_DATA_PATH}/{file_name}"
    
    try:
        if schema:
            df = spark.read.format("csv") \
                .option("header", "true") \
                .schema(schema) \
                .load(file_path)
        else:
            df = spark.read.format("csv") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .load(file_path)
        
        # Add metadata columns
        df_with_metadata = df \
            .withColumn("ingestion_timestamp", current_timestamp()) \
            .withColumn("source_file", lit(file_name)) \
            .withColumn("bronze_load_date", lit(datetime.now().date()))
        
        # Show sample data
        print(f"\n📊 Sample data from {file_name}:")
        df_with_metadata.show(5, truncate=False)
        
        # Write to Delta table
        table_path = f"{BRONZE_PATH}/{table_name}"
        table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
        
        # Ensure parent bronze directory exists (best-effort)
        try:
            dbutils.fs.mkdirs(BRONZE_PATH)
        except Exception as _:
            # If volume creation/permission fails, the write will also fail and bubble up
            pass
        
        # Write Delta files to the volume path (overwrite mode to mirror previous behavior)
        df_with_metadata.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .option("path", table_path) \
            .save()
        
        # Register the table in the metastore if possible pointing to the volume path
        try:
            spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
        except Exception:
            # If Unity Catalog create database fails, ignore (may lack privileges)
            pass
        
        try:
            spark.sql(f"""
                CREATE TABLE IF NOT EXISTS {table_full_name}
                USING DELTA
                LOCATION '{table_path}'
            """)
        except Exception:
            # If CREATE TABLE fails (permissions/UC differences), surface a warning but keep delta files
            print(f"⚠️ Warning: Could not register table {table_full_name} in metastore. Delta files are at {table_path}")
        
        # Get record count
        record_count = df_with_metadata.count()
        
        print(f"\n✅ Successfully loaded {record_count:,} records into {table_full_name} (stored at {table_path})")
        
        # Show table info
        print(f"\n📋 Table Schema:")
        df_with_metadata.printSchema()
        
        return df_with_metadata
        
    except Exception as e:
        print(f"❌ Error loading {file_name}: {str(e)}")
        raise e
        

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Ingest Customers Data

# COMMAND ----------

# Define schema for customers (explicit typing for better data quality)
customers_schema = StructType([
    StructField("customer_id", StringType(), False),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("phone", StringType(), True),
    StructField("date_of_birth", DateType(), True),
    StructField("gender", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("zip_code", StringType(), True),
    StructField("registration_date", DateType(), True),
    StructField("customer_segment", StringType(), True),
    StructField("acquisition_channel", StringType(), True)
])

customers_bronze_df = ingest_to_bronze(
    file_name="customers.csv",
    table_name="customers_bronze",
    schema=customers_schema
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Ingest Products Data

# COMMAND ----------

products_schema = StructType([
    StructField("product_id", StringType(), False),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("subcategory", StringType(), True),
    StructField("brand", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("cost", DoubleType(), True),
    StructField("stock_quantity", IntegerType(), True),
    StructField("supplier_id", StringType(), True),
    StructField("launch_date", DateType(), True)
])

products_bronze_df = ingest_to_bronze(
    file_name="products.csv",
    table_name="products_bronze",
    schema=products_schema
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingest Transactions Data

# COMMAND ----------

transactions_schema = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("transaction_date", TimestampType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("discount_percent", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("store_id", StringType(), True),
    StructField("shipping_cost", DoubleType(), True),
    StructField("order_status", StringType(), True)
])

transactions_bronze_df = ingest_to_bronze(
    file_name="transactions.csv",
    table_name="transactions_bronze",
    schema=transactions_schema
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ingest CRM Interactions Data

# COMMAND ----------

crm_schema = StructType([
    StructField("interaction_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("interaction_date", TimestampType(), True),
    StructField("interaction_type", StringType(), True),
    StructField("reason", StringType(), True),
    StructField("resolution_status", StringType(), True),
    StructField("agent_id", StringType(), True),
    StructField("duration_minutes", IntegerType(), True),
    StructField("satisfaction_score", IntegerType(), True),
    StructField("notes", StringType(), True)
])

crm_bronze_df = ingest_to_bronze(
    file_name="crm_interactions.csv",
    table_name="crm_interactions_bronze",
    schema=crm_schema
)


# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ingest Loyalty Events Data

# COMMAND ----------

loyalty_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("event_date", TimestampType(), True),
    StructField("event_type", StringType(), True),
    StructField("points_change", IntegerType(), True),
    StructField("points_balance", IntegerType(), True),
    StructField("tier", StringType(), True),
    StructField("description", StringType(), True)
])

loyalty_bronze_df = ingest_to_bronze(
    file_name="loyalty_events.csv",
    table_name="loyalty_events_bronze",
    schema=loyalty_schema
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Ingest Web Analytics Data

# COMMAND ----------

web_schema = StructType([
    StructField("session_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("session_date", TimestampType(), True),
    StructField("device_type", StringType(), True),
    StructField("browser", StringType(), True),
    StructField("page_views", IntegerType(), True),
    StructField("session_duration_seconds", IntegerType(), True),
    StructField("bounce", BooleanType(), True),
    StructField("conversion", BooleanType(), True),
    StructField("utm_source", StringType(), True),
    StructField("utm_campaign", StringType(), True),
    StructField("landing_page", StringType(), True),
    StructField("exit_page", StringType(), True)
])

web_bronze_df = ingest_to_bronze(
    file_name="web_analytics.csv",
    table_name="web_analytics_bronze",
    schema=web_schema
)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Ingest Support Tickets Data

# COMMAND ----------

tickets_schema = StructType([
    StructField("ticket_id", StringType(), False),
    StructField("customer_id", StringType(), True),
    StructField("created_date", TimestampType(), True),
    StructField("category", StringType(), True),
    StructField("priority", StringType(), True),
    StructField("status", StringType(), True),
    StructField("assigned_agent", StringType(), True),
    StructField("resolved_date", TimestampType(), True),
    StructField("resolution_time_hours", IntegerType(), True),
    StructField("customer_satisfaction", IntegerType(), True)
])

tickets_bronze_df = ingest_to_bronze(
    file_name="support_tickets.csv",
    table_name="support_tickets_bronze",
    schema=tickets_schema
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Layer Summary

# COMMAND ----------

# Get summary of all bronze tables
bronze_tables = [
    "customers_bronze",
    "products_bronze",
    "transactions_bronze",
    "crm_interactions_bronze",
    "loyalty_events_bronze",
    "web_analytics_bronze",
    "support_tickets_bronze"
]

print("\n" + "="*80)
print("📊 BRONZE LAYER INGESTION SUMMARY")
print("="*80 + "\n")

summary_data = []

for table_name in bronze_tables:
    table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    
    try:
        df = spark.table(table_full_name)
        record_count = df.count()
        column_count = len(df.columns)
        
        summary_data.append({
            "Table": table_name,
            "Records": f"{record_count:,}",
            "Columns": column_count,
            "Status": "✅ Success"
        })
        
    except Exception as e:
        summary_data.append({
            "Table": table_name,
            "Records": "N/A",
            "Columns": "N/A",
            "Status": f"❌ Error: {str(e)}"
        })

# Create summary DataFrame
summary_df = spark.createDataFrame(summary_data)
summary_df.show(truncate=False)

print("\n" + "="*80)
print("✅ BRONZE LAYER INGESTION COMPLETE!")
print("="*80)
print("\n🎯 Next Steps:")
print("  1. Verify data quality in Bronze tables")
print("  2. Proceed to Silver layer transformation")
print("  3. Apply data cleaning and business rules")
print("\n")

# COMMAND ----------

from pyspark.sql.utils import AnalysisException
from pyspark.sql import Row
import traceback

bronze_tables = [
    "customers_bronze",
    "products_bronze",
    "transactions_bronze",
    "crm_interactions_bronze",
    "loyalty_events_bronze",
    "web_analytics_bronze",
    "support_tickets_bronze"
]

summary_rows = []

for table_name in bronze_tables:
    table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    table_path = f"/Volumes/workspace/default/bronze/{table_name}"
    status = "❌ Unknown"
    records = "N/A"
    cols = "N/A"

    # Try to read via catalog first
    try:
        df = spark.table(table_full_name)
        records = df.count()
        cols = len(df.columns)
        status = "✅ Registered (catalog)"
    except AnalysisException as ae:
        # Table not found in metastore: fall back to path-based read
        msg = str(ae).lower()
        if "table not found" in msg or "unresolvedrelation" in msg:
            try:
                # Try path-based read from Volume
                df_path = spark.read.format("delta").load(table_path)
                records = df_path.count()
                cols = len(df_path.columns)
                status = "✅ Found at path (unregistered)"
            except Exception as path_err:
                # Path read also failed — capture concise error
                status = f"❌ Missing. Path read failed: {type(path_err).__name__}: {str(path_err).splitlines()[0]}"
        else:
            # Some other AnalysisException
            status = f"❌ Analysis error: {str(ae).splitlines()[0]}"
    except Exception as e:
        # Generic catch — keep message short
        status = f"❌ Error: {type(e).__name__}: {str(e).splitlines()[0]}"

    summary_rows.append(Row(Table=table_name, Records=str(records), Columns=str(cols), Status=status))

# Convert to DataFrame and show
summary_df = spark.createDataFrame(summary_rows)
summary_df.show(truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks
# MAGIC

# COMMAND ----------


print("🔍 Running Data Quality Checks on Bronze Tables...\n")

# Check 1: Duplicate customer IDs
print("1. Checking for duplicate customer IDs...")
customer_dups = customers_bronze_df.groupBy("customer_id").count().filter("count > 1")
dup_count = customer_dups.count()
if dup_count == 0:
    print("   ✅ No duplicate customer IDs found")
else:
    print(f"   ⚠️ Found {dup_count} duplicate customer IDs")
    customer_dups.show()

# Check 2: Null customer IDs in transactions
print("\n2. Checking for null customer IDs in transactions...")
null_customers = transactions_bronze_df.filter("customer_id IS NULL").count()
if null_customers == 0:
    print("   ✅ No null customer IDs in transactions")
else:
    print(f"   ⚠️ Found {null_customers} transactions with null customer IDs")

# Check 3: Date range validation
print("\n3. Checking date ranges...")
from pyspark.sql.functions import min, max

date_ranges = transactions_bronze_df.select(
    min("transaction_date").alias("earliest_transaction"),
    max("transaction_date").alias("latest_transaction")
)
date_ranges.show()

# Check 4: Invalid product prices
print("\n4. Checking for invalid product prices...")
invalid_prices = products_bronze_df.filter("price <= 0 OR price IS NULL").count()
if invalid_prices == 0:
    print("   ✅ All product prices are valid")
else:
    print(f"   ⚠️ Found {invalid_prices} products with invalid prices")

print("\n✅ Data Quality Checks Complete!")

# COMMAND ----------

