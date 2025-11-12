# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime

# Initialize
spark = SparkSession.builder.appName("Customer360_Silver_Transformation").getOrCreate()

# Configuration
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
BRONZE_BASE = "/Volumes/workspace/default/bronze"
SILVER_PATH = "/Volumes/workspace/default/silver"

# Create silver directory
dbutils.fs.mkdirs(SILVER_PATH)

print("✅ Configuration loaded")
print("BRONZE_BASE =", BRONZE_BASE)
print(f"📂 Silver Path: {SILVER_PATH}")

# COMMAND ----------

# Robust safe_read_bronze: prefer path-based delta loads (no unresolved relation)
def safe_read_bronze(table_name):
    """
    Read a bronze table safely:
      1) Try loading directly from the Delta folder path (preferred)
      2) If that fails, attempt catalog read as a last resort
    This avoids returning unresolved logical relations from spark.table(...) calls.
    """
    table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    path = f"{BRONZE_BASE}/{table_name}"
    
    # Try path-based read first (preferred)
    try:
        df_path = spark.read.format("delta").load(path)
        # Force a tiny action to validate the read (but avoid heavy counts)
        # We'll do a .limit(1) to force planning and validate quickly
        _ = df_path.limit(1).collect()
        print(f"safe_read_bronze: Loaded from path {path}")
        return df_path
    except Exception as path_err:
        print(f"safe_read_bronze: Could not load from path {path}: {type(path_err).__name__}: {str(path_err).splitlines()[0]}")
        # Fallback to catalog read as a last resort
        try:
            df_cat = spark.table(table_full_name)
            # validate similarly
            _ = df_cat.limit(1).collect()
            print(f"safe_read_bronze: Loaded from catalog {table_full_name}")
            return df_cat
        except Exception as cat_err:
            # Provide clear guidance in the error
            raise RuntimeError(f"safe_read_bronze: Failed to load bronze source '{table_name}' by path ({path_err}) and catalog ({cat_err}). "
                               f"Ensure Delta files exist at the path or the table is registered in {CATALOG_NAME}.{SCHEMA_NAME}.")

def create_silver_table(df, table_name, partition_columns=None):
    """
    Generic function to create Silver Delta tables at SILVER_PATH/<table_name> and register as
    workspace.default.<table_name> pointing to that path (best-effort).
    
    Parameters:
    - df: Spark DataFrame
    - table_name: name of table (string)
    - partition_columns: optional list/string for partitioning
    """
    print(f"\n{'='*60}")
    print(f"💾 Creating {table_name}")
    print(f"{'='*60}")
    
    table_path = f"{SILVER_PATH}/{table_name}"
    table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    
    # add silver metadata
    df_with_metadata = df.withColumn("silver_load_timestamp", current_timestamp())
    
    # prepare writer
    writer = df_with_metadata.write.format("delta").mode("overwrite").option("overwriteSchema", "true").option("path", table_path)
    
    # partition handling: support single string or list/tuple
    if partition_columns:
        if isinstance(partition_columns, (list, tuple)):
            writer = writer.partitionBy(*partition_columns)
        else:
            writer = writer.partitionBy(partition_columns)
    
    # attempt write to volume path
    try:
        writer.save()
        print(f"✅ Wrote Delta files to: {table_path}")
    except Exception as write_err:
        # if volume write fails, try DBFS fallback path
        dbfs_fallback = f"dbfs:/user/hive/warehouse/silver/{table_name}"
        print(f"⚠️ Could not write to {table_path}: {type(write_err).__name__}: {str(write_err).splitlines()[0]}")
        print(f"➡️ Attempting fallback write to DBFS: {dbfs_fallback}")
        try:
            if partition_columns:
                df_with_metadata.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy(*partition_columns if isinstance(partition_columns, (list,tuple)) else [partition_columns]).save(dbfs_fallback)
            else:
                df_with_metadata.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dbfs_fallback)
            table_path = dbfs_fallback
            print(f"✅ Wrote Delta files to fallback path: {dbfs_fallback}")
        except Exception as fallback_err:
            raise RuntimeError(f"Failed to write Silver table {table_name} to both primary and fallback paths: {fallback_err}")
    
    # Attempt to register in Unity Catalog / metastore pointing to the table_path
    try:
        # create database if possible (best-effort)
        try:
            spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
        except Exception:
            # ignore if not allowed
            pass
        
        # Create or replace table pointing to path
        try:
            # Remove any existing table entry, then create new external table pointing to path
            spark.sql(f"CREATE TABLE IF NOT EXISTS {table_full_name} USING DELTA LOCATION '{table_path}'")
            print(f"✅ Registered table in metastore: {table_full_name} -> {table_path}")
        except Exception as reg_err:
            # Try a more forceful replace if CREATE fails but REPLACE TABLE may require privileges too
            try:
                spark.sql(f"CREATE TABLE IF NOT EXISTS {table_full_name} USING DELTA LOCATION '{table_path}'")
                print(f"✅ Registered table in metastore: {table_full_name} -> {table_path}")
            except Exception:
                print(f"⚠️ Warning: Could not register table {table_full_name} in metastore. Delta files are at {table_path}. Error: {type(reg_err).__name__}: {str(reg_err).splitlines()[0]}")
    except Exception as final_reg_err:
        print(f"⚠️ Registration step encountered an unexpected error: {type(final_reg_err).__name__}: {str(final_reg_err).splitlines()[0]}")
    
    # print record count (best-effort)
    try:
        record_count = spark.read.format("delta").load(table_path).count()
        print(f"📈 Record count for {table_name}: {record_count:,}")
    except Exception:
        print("⚠️ Could not compute record count (large table or permission issue).")
    
    return df_with_metadata

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Transform Customers to Silver

# COMMAND ----------

print("👥 Transforming customers to Silver layer...")

# Read bronze data (catalog first, then path)
customers_bronze = safe_read_bronze("customers_bronze")

# Cleaning and transformations
customers_silver = customers_bronze.select(
    col("customer_id"),
    # Standardize names - trim whitespace, proper case
    initcap(trim(col("first_name"))).alias("first_name"),
    initcap(trim(col("last_name"))).alias("last_name"),
    
    # Email standardization - lowercase, trim
    lower(trim(col("email"))).alias("email"),
    
    # Phone standardization
    regexp_replace(col("phone"), "[^0-9]", "").alias("phone_cleaned"),
    col("phone").alias("phone_original"),
    
    # Date fields
    col("date_of_birth"),
    col("registration_date"),
    
    # Calculate age
    floor(datediff(current_date(), col("date_of_birth")) / 365.25).alias("age"),
    
    # Calculate customer tenure in days
    datediff(current_date(), col("registration_date")).alias("customer_tenure_days"),
    
    # Demographics
    upper(col("gender")).alias("gender"),
    initcap(trim(col("city"))).alias("city"),
    upper(trim(col("state"))).alias("state"),
    col("zip_code"),
    
    # Segmentation
    col("customer_segment"),
    col("acquisition_channel"),
    
    # Metadata from bronze
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates based on customer_id (keep most recent)
window_spec = Window.partitionBy("customer_id").orderBy(col("bronze_ingestion_timestamp").desc())
customers_silver = customers_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# Data quality flags
customers_silver = customers_silver.withColumn(
    "data_quality_flag",
    when(col("email").isNull() | col("phone_cleaned").isNull(), "Missing Contact Info")
    .when(col("age") < 18, "Underage")
    .when(col("age") > 100, "Invalid Age")
    .otherwise("Valid")
)

# Create silver table
customers_silver_df = create_silver_table(customers_silver, "customers_silver")

print("\n📊 Sample Silver Customers:")
customers_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Transform Products to Silver

# COMMAND ----------

print("📦 Transforming products to Silver layer...")

products_bronze = safe_read_bronze("products_bronze")

products_silver = products_bronze.select(
    col("product_id"),
    trim(col("product_name")).alias("product_name"),
    initcap(trim(col("category"))).alias("category"),
    initcap(trim(col("subcategory"))).alias("subcategory"),
    initcap(trim(col("brand"))).alias("brand"),
    
    # Price fields with validation
    col("price"),
    col("cost"),
    
    # Calculate margin
    round((col("price") - col("cost")) / col("price") * 100, 2).alias("profit_margin_pct"),
    
    # Stock status
    col("stock_quantity"),
    when(col("stock_quantity") == 0, "Out of Stock")
        .when(col("stock_quantity") < 10, "Low Stock")
        .when(col("stock_quantity") < 50, "Medium Stock")
        .otherwise("In Stock").alias("stock_status"),
    
    col("supplier_id"),
    col("launch_date"),
    
    # Product age in days
    datediff(current_date(), col("launch_date")).alias("product_age_days"),
    
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates
window_spec = Window.partitionBy("product_id").orderBy(col("bronze_ingestion_timestamp").desc())
products_silver = products_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# Data quality flags
products_silver = products_silver.withColumn(
    "data_quality_flag",
    when(col("price").isNull() | (col("price") <= 0), "Invalid Price")
    .when(col("cost").isNull() | (col("cost") <= 0), "Invalid Cost")
    .when(col("price") < col("cost"), "Price Below Cost")
    .otherwise("Valid")
)

products_silver_df = create_silver_table(products_silver, "products_silver")

print("\n📊 Sample Silver Products:")
products_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Transform Transactions to Silver

# COMMAND ----------

print("💳 Transforming transactions to Silver layer...")

transactions_bronze = safe_read_bronze("transactions_bronze")

# Join with customers and products for enrichment
# Use the silver DataFrames that returned from create_silver_table above (these are DataFrames with metadata)
transactions_silver = transactions_bronze \
    .join(customers_silver_df.select("customer_id", "customer_segment", "state"), 
          "customer_id", "left") \
    .join(products_silver_df.select("product_id", "category", "brand", "cost"), 
          "product_id", "left")

transactions_silver = transactions_silver.select(
    col("transaction_id"),
    col("customer_id"),
    col("product_id"),
    col("transaction_date"),
    
    # Extract date components for analysis
    year(col("transaction_date")).alias("transaction_year"),
    month(col("transaction_date")).alias("transaction_month"),
    dayofweek(col("transaction_date")).alias("transaction_day_of_week"),
    hour(col("transaction_date")).alias("transaction_hour"),
    quarter(col("transaction_date")).alias("transaction_quarter"),
    
    # Transaction details
    col("quantity"),
    col("unit_price"),
    col("discount_percent"),
    col("total_amount"),
    
    # Calculate actual discount amount
    round(col("unit_price") * col("quantity") * col("discount_percent") / 100, 2).alias("discount_amount"),
    
    # Calculate revenue (total_amount includes discount)
    col("total_amount").alias("revenue"),
    
    # Calculate gross profit (using cost from products)
    round(col("total_amount") - (col("cost") * col("quantity")), 2).alias("gross_profit"),
    
    col("payment_method"),
    col("channel"),
    col("store_id"),
    col("shipping_cost"),
    col("order_status"),
    
    # Enriched dimensions
    col("customer_segment"),
    col("state").alias("customer_state"),
    col("category").alias("product_category"),
    col("brand").alias("product_brand"),
    
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates (same transaction_id + product_id combination)
window_spec = Window.partitionBy("transaction_id", "product_id").orderBy(col("bronze_ingestion_timestamp").desc())
transactions_silver = transactions_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

# Data quality flags
transactions_silver = transactions_silver.withColumn(
    "data_quality_flag",
    when(col("total_amount") < 0, "Negative Amount")
    .when(col("quantity") <= 0, "Invalid Quantity")
    .when(col("customer_id").isNull(), "Missing Customer")
    .when(col("product_id").isNull(), "Missing Product")
    .otherwise("Valid")
)

# Only keep valid transactions for silver
transactions_silver = transactions_silver.filter(col("data_quality_flag") == "Valid")

transactions_silver_df = create_silver_table(
    transactions_silver, 
    "transactions_silver",
    partition_columns=["transaction_year", "transaction_month"]
)

print("\n📊 Sample Silver Transactions:")
transactions_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Transform CRM Interactions to Silver

# COMMAND ----------

print("📞 Transforming CRM interactions to Silver layer...")

crm_bronze = safe_read_bronze("crm_interactions_bronze")

crm_silver = crm_bronze.select(
    col("interaction_id"),
    col("customer_id"),
    col("interaction_date"),
    
    # Extract date components
    year(col("interaction_date")).alias("interaction_year"),
    month(col("interaction_date")).alias("interaction_month"),
    
    col("interaction_type"),
    col("reason"),
    col("resolution_status"),
    col("agent_id"),
    col("duration_minutes"),
    
    # Categorize interaction duration
    when(col("duration_minutes") < 5, "Quick")
        .when(col("duration_minutes") < 15, "Standard")
        .when(col("duration_minutes") < 30, "Extended")
        .otherwise("Long").alias("duration_category"),
    
    col("satisfaction_score"),
    
    # Flag unresolved interactions
    when(col("resolution_status").isin("Pending", "Escalated"), True)
        .otherwise(False).alias("is_unresolved"),
    
    col("notes"),
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates
window_spec = Window.partitionBy("interaction_id").orderBy(col("bronze_ingestion_timestamp").desc())
crm_silver = crm_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

crm_silver_df = create_silver_table(crm_silver, "crm_interactions_silver")

print("\n📊 Sample Silver CRM Interactions:")
crm_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Transform Loyalty Events to Silver

# COMMAND ----------

print("🎁 Transforming loyalty events to Silver layer...")

loyalty_bronze = safe_read_bronze("loyalty_events_bronze")

loyalty_silver = loyalty_bronze.select(
    col("event_id"),
    col("customer_id"),
    col("event_date"),
    
    # Extract date components
    year(col("event_date")).alias("event_year"),
    month(col("event_date")).alias("event_month"),
    
    col("event_type"),
    col("points_change"),
    
    # Categorize point changes
    when(col("points_change") > 0, "Earn")
        .when(col("points_change") < 0, "Redeem")
        .otherwise("Neutral").alias("points_transaction_type"),
    
    col("points_balance"),
    col("tier"),
    col("description"),
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates
window_spec = Window.partitionBy("event_id").orderBy(col("bronze_ingestion_timestamp").desc())
loyalty_silver = loyalty_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

loyalty_silver_df = create_silver_table(loyalty_silver, "loyalty_events_silver")

print("\n📊 Sample Silver Loyalty Events:")
loyalty_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Transform Web Analytics to Silver

# COMMAND ----------

print("🌐 Transforming web analytics to Silver layer...")

web_bronze = safe_read_bronze("web_analytics_bronze")

web_silver = web_bronze.select(
    col("session_id"),
    col("customer_id"),
    col("session_date"),
    
    # Extract date components
    year(col("session_date")).alias("session_year"),
    month(col("session_date")).alias("session_month"),
    dayofweek(col("session_date")).alias("session_day_of_week"),
    hour(col("session_date")).alias("session_hour"),
    
    col("device_type"),
    col("browser"),
    col("page_views"),
    col("session_duration_seconds"),
    
    # Calculate engagement metrics
    round(col("session_duration_seconds") / col("page_views"), 2).alias("avg_time_per_page_seconds"),
    
    # Categorize session duration
    when(col("session_duration_seconds") < 60, "Very Short")
        .when(col("session_duration_seconds") < 300, "Short")
        .when(col("session_duration_seconds") < 900, "Medium")
        .otherwise("Long").alias("session_duration_category"),
    
    col("bounce").alias("is_bounce"),
    col("conversion").alias("is_conversion"),
    
    col("utm_source"),
    col("utm_campaign"),
    col("landing_page"),
    col("exit_page"),
    
    # Flag abandoned sessions (no conversion, multiple pages)
    when((col("conversion") == False) & (col("page_views") > 3), True)
        .otherwise(False).alias("is_abandoned_cart"),
    
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates
window_spec = Window.partitionBy("session_id").orderBy(col("bronze_ingestion_timestamp").desc())
web_silver = web_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

web_silver_df = create_silver_table(web_silver, "web_analytics_silver")

print("\n📊 Sample Silver Web Analytics:")
web_silver_df.show(5, truncate=False)

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Transform Support Tickets to Silver

# COMMAND ----------

print("🎫 Transforming support tickets to Silver layer...")

tickets_bronze = safe_read_bronze("support_tickets_bronze")

tickets_silver = tickets_bronze.select(
    col("ticket_id"),
    col("customer_id"),
    col("created_date"),
    
    # Extract date components
    year(col("created_date")).alias("created_year"),
    month(col("created_date")).alias("created_month"),
    
    col("category"),
    col("priority"),
    col("status"),
    col("assigned_agent"),
    col("resolved_date"),
    
    # Calculate resolution time in hours
    col("resolution_time_hours"),
    
    # Categorize resolution time
    when(col("resolution_time_hours").isNull(), "Unresolved")
        .when(col("resolution_time_hours") < 4, "Fast")
        .when(col("resolution_time_hours") < 24, "Standard")
        .when(col("resolution_time_hours") < 72, "Slow")
        .otherwise("Very Slow").alias("resolution_speed"),
    
    # Flag SLA breach (>48 hours for high priority, >72 for others)
    when((col("priority") == "High") & (col("resolution_time_hours") > 48), True)
        .when((col("priority") != "High") & (col("resolution_time_hours") > 72), True)
        .otherwise(False).alias("sla_breached"),
    
    col("customer_satisfaction"),
    
    # Flag dissatisfied customers
    when(col("customer_satisfaction") <= 2, True)
        .otherwise(False).alias("is_dissatisfied"),
    
    col("ingestion_timestamp").alias("bronze_ingestion_timestamp")
)

# Remove duplicates
window_spec = Window.partitionBy("ticket_id").orderBy(col("bronze_ingestion_timestamp").desc())
tickets_silver = tickets_silver.withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .drop("row_num")

tickets_silver_df = create_silver_table(tickets_silver, "support_tickets_silver")

print("\n📊 Sample Silver Support Tickets:")
tickets_silver_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## USECASE 1 - 
# MAGIC ### UNIFIED CUSTOMER 360° VIEW

# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Create Customer 360° Silver View

# COMMAND ----------

print("🎯 Creating Customer 360° unified silver table...")

# Aggregate transaction metrics per customer
transaction_metrics = transactions_silver_df.groupBy("customer_id").agg(
    count("transaction_id").alias("total_transactions"),
    sum("revenue").alias("total_revenue"),
    avg("revenue").alias("avg_transaction_value"),
    max("transaction_date").alias("last_purchase_date"),
    min("transaction_date").alias("first_purchase_date"),
    countDistinct("product_category").alias("distinct_categories_purchased"),
    sum(when(col("order_status") == "Returned", 1).otherwise(0)).alias("return_count")
)

# Calculate recency
transaction_metrics = transaction_metrics.withColumn(
    "recency_days",
    datediff(current_date(), col("last_purchase_date"))
)

# Aggregate loyalty metrics
loyalty_metrics = loyalty_silver_df.groupBy("customer_id").agg(
    max("points_balance").alias("current_loyalty_points"),
    max("tier").alias("current_loyalty_tier"),
    sum(when(col("points_transaction_type") == "Earn", col("points_change")).otherwise(0)).alias("total_points_earned"),
    sum(when(col("points_transaction_type") == "Redeem", -col("points_change")).otherwise(0)).alias("total_points_redeemed")
)

# Aggregate CRM metrics
crm_metrics = crm_silver_df.groupBy("customer_id").agg(
    count("interaction_id").alias("total_crm_interactions"),
    avg("satisfaction_score").alias("avg_satisfaction_score"),
    sum(when(col("is_unresolved") == True, 1).otherwise(0)).alias("unresolved_interactions_count")
)

# Aggregate web analytics metrics
web_metrics = web_silver_df.filter(col("customer_id").isNotNull()).groupBy("customer_id").agg(
    count("session_id").alias("total_web_sessions"),
    sum("page_views").alias("total_page_views"),
    avg("session_duration_seconds").alias("avg_session_duration"),
    sum(when(col("is_conversion") == True, 1).otherwise(0)).alias("web_conversions"),
    countDistinct("device_type").alias("distinct_devices_used")
)

# Aggregate support ticket metrics
ticket_metrics = tickets_silver_df.groupBy("customer_id").agg(
    count("ticket_id").alias("total_support_tickets"),
    sum(when(col("sla_breached") == True, 1).otherwise(0)).alias("sla_breaches_count"),
    avg("customer_satisfaction").alias("avg_ticket_satisfaction")
)

# Join all metrics to create Customer 360
customer360_silver = customers_silver_df \
    .join(transaction_metrics, "customer_id", "left") \
    .join(loyalty_metrics, "customer_id", "left") \
    .join(crm_metrics, "customer_id", "left") \
    .join(web_metrics, "customer_id", "left") \
    .join(ticket_metrics, "customer_id", "left")

# Calculate derived metrics
customer360_silver = customer360_silver.withColumn(
    "customer_lifetime_value",
    coalesce(col("total_revenue"), lit(0))
)

customer360_silver = customer360_silver.withColumn(
    "is_active_customer",
    when(col("recency_days") <= 90, True).otherwise(False)
)

customer360_silver = customer360_silver.withColumn(
    "customer_health_score",
    round(
        (coalesce(col("avg_satisfaction_score"), lit(3)) * 20) +  # Max 100 points
        (least(coalesce(col("total_transactions"), lit(0)), lit(20)) * 2) +  # Max 40 points
        (when(col("is_active_customer") == True, 30).otherwise(0)) +  # Max 30 points
        (least(coalesce(col("current_loyalty_points"), lit(0)) / 100, lit(10)) * 1),  # Max 10 points
        1
    )
)

# Churn risk indicator (simple rule-based for now)
customer360_silver = customer360_silver.withColumn(
    "churn_risk_flag",
    when(col("recency_days") > 180, "High")
        .when(col("recency_days") > 90, "Medium")
        .otherwise("Low")
)

# Fill nulls for numeric columns
numeric_cols = [
    "total_transactions", "total_revenue", "avg_transaction_value", 
    "return_count", "current_loyalty_points", "total_points_earned", 
    "total_points_redeemed", "total_crm_interactions", "avg_satisfaction_score",
    "unresolved_interactions_count", "total_web_sessions", "total_page_views",
    "avg_session_duration", "web_conversions", "distinct_devices_used",
    "total_support_tickets", "sla_breaches_count", "avg_ticket_satisfaction"
]

for col_name in numeric_cols:
    customer360_silver = customer360_silver.withColumn(col_name, coalesce(col(col_name), lit(0)))

customer360_silver_df = create_silver_table(customer360_silver, "customer360_silver")

print("\n📊 Sample Customer 360° Silver:")
customer360_silver_df.select(
    "customer_id", "first_name", "last_name", "customer_segment",
    "total_transactions", "total_revenue", "recency_days",
    "customer_health_score", "churn_risk_flag"
).show(10, truncate=False)


# COMMAND ----------

# Read the Delta files directly from your Silver path
customer360_silver_df = spark.read.format("delta").load("/Volumes/workspace/default/silver/customer360_silver")

# Show a few records
customer360_silver_df.show(10, truncate=False)

# Print schema
customer360_silver_df.printSchema()

# Count records
print(f"Total records in customer360_silver: {customer360_silver_df.count():,}")

# Create or replace a temporary view
customer360_silver_df.createOrReplaceTempView("customer360_silver_view")

# Run SQL queries directly
spark.sql("""
SELECT 
  customer_id,
  total_transactions,
  total_revenue as total_spent,
  avg_transaction_value,
  current_loyalty_points as loyalty_points,
  last_purchase_date as last_interaction_date
FROM customer360_silver_view
ORDER BY total_spent DESC
LIMIT 10
""").show(truncate=False)


# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Layer Summary & Quality Report

# COMMAND ----------

print("\n" + "="*80)
print("📊 SILVER LAYER TRANSFORMATION SUMMARY")
print("="*80 + "\n")

silver_tables = [
    "customers_silver",
    "products_silver",
    "transactions_silver",
    "crm_interactions_silver",
    "loyalty_events_silver",
    "web_analytics_silver",
    "support_tickets_silver",
    "customer360_silver"
]

summary_data = []

for table_name in silver_tables:
    table_full_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    table_path = f"{SILVER_PATH}/{table_name}"
    status = "❌ Unknown"
    records = "N/A"
    cols = "N/A"
    try:
        # prefer catalog table if exists
        try:
            df = spark.table(table_full_name)
            records = df.count()
            cols = len(df.columns)
            status = "✅ Registered (catalog)"
        except Exception:
            # fallback to path-based read
            try:
                dfp = spark.read.format("delta").load(table_path)
                records = dfp.count()
                cols = len(dfp.columns)
                status = "✅ Found at path (unregistered)"
            except Exception as path_err:
                status = f"❌ Missing (path read failed: {type(path_err).__name__}: {str(path_err).splitlines()[0]})"
    except Exception as e:
        status = f"❌ Error: {type(e).__name__}: {str(e).splitlines()[0]}"
    summary_data.append({"Table": table_name, "Records": str(records), "Columns": str(cols), "Status": status})

summary_df = spark.createDataFrame(summary_data)
summary_df.show(truncate=False)



# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Summary

# COMMAND ----------

print("\n🔍 Data Quality Analysis:\n")

# Customer data quality
print("1. Customer Data Quality:")
customer_quality = customers_silver_df.groupBy("data_quality_flag").count()
customer_quality.show()

# Product data quality
print("\n2. Product Data Quality:")
product_quality = products_silver_df.groupBy("data_quality_flag").count()
product_quality.show()

# Transaction summary
print("\n3. Transaction Summary:")
transaction_summary = transactions_silver_df.agg(
    count("*").alias("total_transactions"),
    sum("revenue").alias("total_revenue"),
    avg("revenue").alias("avg_transaction_value"),
    countDistinct("customer_id").alias("unique_customers")
)
transaction_summary.show()

# Customer 360 health distribution
print("\n4. Customer Health Score Distribution:")
health_distribution = customer360_silver_df.selectExpr(
    "CASE WHEN customer_health_score >= 80 THEN 'Excellent' " +
    "WHEN customer_health_score >= 60 THEN 'Good' " +
    "WHEN customer_health_score >= 40 THEN 'Fair' " +
    "ELSE 'Poor' END as health_category"
).groupBy("health_category").count().orderBy("health_category")
health_distribution.show()

# Churn risk distribution
print("\n5. Churn Risk Distribution:")
churn_distribution = customer360_silver_df.groupBy("churn_risk_flag").count()
churn_distribution.show()

print("\n" + "="*80)
print("✅ SILVER LAYER TRANSFORMATION COMPLETE!")
print("="*80)


# COMMAND ----------

