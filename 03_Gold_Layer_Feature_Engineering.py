# Databricks notebook source
# Configuration & imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
from datetime import datetime, timedelta
from pyspark.sql.utils import AnalysisException
import numpy as np
from datetime import datetime, timedelta

spark = SparkSession.builder.appName("Customer360_Gold_Features").getOrCreate()

CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
SILVER_BASE = "/Volumes/workspace/default/silver"
GOLD_PATH = "/Volumes/workspace/default/gold"

# Ensure gold dir exists (best-effort)
try:
    dbutils.fs.mkdirs(GOLD_PATH)
except Exception as e:
    print("⚠️ Could not create GOLD_PATH via dbutils (may be permission/volume). Proceeding; writes may still succeed:", str(e))


# COMMAND ----------

def safe_read_silver(table_name):
    """
    Accepts: 'customer360_silver' or 'default.customer360_silver' or 'workspace.default.customer360_silver'
    Returns a resolved DataFrame loaded from the Delta path.
    """
    short = table_name.split(".")[-1]
    path = f"{SILVER_BASE}/{short}"
    # Try path-based read (preferred)
    try:
        df = spark.read.format("delta").load(path)
        # quick validate (limit 1)
        _ = df.limit(1).collect()
        print(f"safe_read_silver: Loaded from path {path}")
        return df
    except Exception as path_err:
        print(f"safe_read_silver: Path read failed for {path}: {type(path_err).__name__}: {str(path_err).splitlines()[0]}")
        # fallback to catalog
        try:
            full = f"{CATALOG_NAME}.{SCHEMA_NAME}.{short}"
            df2 = spark.table(full)
            _ = df2.limit(1).collect()
            print(f"safe_read_silver: Loaded from catalog {full}")
            return df2
        except Exception as cat_err:
            raise RuntimeError(f"safe_read_silver: Failed to load '{table_name}' by path ({path_err}) and catalog ({cat_err}). Ensure Delta folder exists at {path} or table is registered in {CATALOG_NAME}.{SCHEMA_NAME}.")

# Utility: save DataFrame as Delta under GOLD_PATH/<table> and attempt register in UC (graceful)
def save_and_register_gold(df, table_name, partition_by=None):
    """
    Save df to GOLD_PATH/<table_name> as Delta. Attempt to register workspace.default.<table_name>.
    Returns path where data was written.
    """
    table_path = f"{GOLD_PATH}/{table_name}"
    table_full = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name}"
    df_to_write = df.withColumn("gold_load_timestamp", current_timestamp())
    writer = df_to_write.write.format("delta").mode("overwrite").option("overwriteSchema","true").option("path", table_path)
    if partition_by:
        if isinstance(partition_by, (list,tuple)):
            writer = writer.partitionBy(*partition_by)
        else:
            writer = writer.partitionBy(partition_by)
    try:
        writer.save()
        print(f"✅ Wrote Delta files to: {table_path}")
    except Exception as e:
        # If write fails, raise — we don't fallback to public DBFS by default (you asked no copying to public DBFS)
        raise RuntimeError(f"Failed to write gold table {table_name} to {table_path}: {type(e).__name__}: {str(e).splitlines()[0]}")
    # Attempt registration (best-effort)
    try:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
    except Exception:
        pass
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS {table_full} USING DELTA LOCATION '{table_path}'")
        print(f"✅ Registered table: {table_full} -> {table_path}")
    except Exception as e:
        # Graceful: print short reason (e.g., Missing cloud file system scheme)
        print(f"⚠️ Warning: Could not register table {table_full} in metastore. Delta files are at {table_path}. Error: {type(e).__name__}: {str(e).splitlines()[0]}")
    # quick count (best-effort)
    try:
        cnt = spark.read.format("delta").load(table_path).count()
        print(f"📈 Record count for {table_name}: {cnt:,}")
    except Exception:
        print("⚠️ Could not compute record count (permission/size).")
    return table_path

# COMMAND ----------

# Load Silver datasets (path-based)
print("📥 Loading Silver datasets (path-based reads)...")
customers_silver = safe_read_silver("customers_silver")
products_silver = safe_read_silver("products_silver")
transactions_silver = safe_read_silver("transactions_silver")
crm_silver = safe_read_silver("crm_interactions_silver")
loyalty_silver = safe_read_silver("loyalty_events_silver")
web_silver = safe_read_silver("web_analytics_silver")
tickets_silver = safe_read_silver("support_tickets_silver")
# customer360_silver is optional (may be created earlier); if available read it
try:
    customer360_silver = safe_read_silver("customer360_silver")
except Exception:
    customer360_silver = None
    print("Note: customer360_silver not found in silver path (ok, it will be derived from joins below).")

print("✅ Silver datasets loaded (or noted missing).")

# Set a reference date for recency calculations
REFERENCE_DATE = current_date()

# For functions that expect Python datetime:
REFERENCE_DATE_PY = datetime.now()

# COMMAND ----------

# 1. Customer RFM Analysis (Gold)
print("\n📊 Creating RFM Analysis table...")

rfm_base = transactions_silver.groupBy("customer_id").agg(
    datediff(REFERENCE_DATE, max("transaction_date")).alias("recency_days"),
    countDistinct("transaction_id").alias("frequency"),
    sum("revenue").alias("monetary_value")
)

# Compute quantiles (approx)
rfm_quantiles = rfm_base.approxQuantile(
    ["recency_days", "frequency", "monetary_value"],
    [0.2, 0.4, 0.6, 0.8],
    0.01
)

# Build scores (1-5)
rfm_with_scores = rfm_base.withColumn(
    "recency_score",
    when(col("recency_days") <= rfm_quantiles[0][0], 5)
    .when(col("recency_days") <= rfm_quantiles[0][1], 4)
    .when(col("recency_days") <= rfm_quantiles[0][2], 3)
    .when(col("recency_days") <= rfm_quantiles[0][3], 2)
    .otherwise(1)
)

rfm_with_scores = rfm_with_scores.withColumn(
    "frequency_score",
    when(col("frequency") >= rfm_quantiles[1][3], 5)
    .when(col("frequency") >= rfm_quantiles[1][2], 4)
    .when(col("frequency") >= rfm_quantiles[1][1], 3)
    .when(col("frequency") >= rfm_quantiles[1][0], 2)
    .otherwise(1)
)

rfm_with_scores = rfm_with_scores.withColumn(
    "monetary_score",
    when(col("monetary_value") >= rfm_quantiles[2][3], 5)
    .when(col("monetary_value") >= rfm_quantiles[2][2], 4)
    .when(col("monetary_value") >= rfm_quantiles[2][1], 3)
    .when(col("monetary_value") >= rfm_quantiles[2][0], 2)
    .otherwise(1)
)

rfm_with_scores = rfm_with_scores.withColumn("rfm_score", col("recency_score") + col("frequency_score") + col("monetary_score"))

rfm_gold = rfm_with_scores.withColumn(
    "rfm_segment",
    when(col("rfm_score") >= 13, "Champions")
    .when(col("rfm_score") >= 11, "Loyal Customers")
    .when(col("rfm_score") >= 9, "Potential Loyalists")
    .when(col("rfm_score") >= 7, "At Risk")
    .when(col("rfm_score") >= 5, "Needs Attention")
    .otherwise("Lost")
).withColumn("gold_load_timestamp", current_timestamp())

# Save RFM to gold
save_and_register_gold(rfm_gold, "customer_rfm_gold")

print("📊 RFM segment sample:")
rfm_gold.groupBy("rfm_segment").count().orderBy(desc("count")).show()

# COMMAND ----------

# Customer Features for ML (Gold) — path-based (no Unity Catalog / no DBFS)
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, coalesce, lit, current_timestamp
from pyspark.sql.types import StringType

print("🤖 Creating ML-ready customer feature table...")

# CONFIG - update to your desired gold folder path (local or mounted volume)
gold_path = "/Volumes/workspace/default/gold/customer_features_gold"  # <-- update as needed

# Calculate time-based features (keep original logic: Python datetimes)
current_date_val = datetime.now()
lookback_30d = current_date_val - timedelta(days=30)
lookback_60d = current_date_val - timedelta(days=60)
lookback_90d = current_date_val - timedelta(days=90)

# Note: when comparing Spark timestamp/date columns to Python datetimes, use lit(...) cast to timestamp
lookback_30d_lit = F.lit(lookback_30d).cast("timestamp")
lookback_60d_lit = F.lit(lookback_60d).cast("timestamp")
lookback_90d_lit = F.lit(lookback_90d).cast("timestamp")

# Ensure REFERENCE_DATE exists; default to current_date() if not set externally
try:
    REFERENCE_DATE  # noqa: F821
except NameError:
    REFERENCE_DATE = F.current_date()

# -------------------------
# Transaction features
# -------------------------
transaction_features = transactions_silver.groupBy("customer_id").agg(
    # Overall metrics
    F.count("transaction_id").alias("total_transactions"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_transaction_value"),
    F.stddev("revenue").alias("stddev_transaction_value"),
    
    # Recent activity (last 30/60/90 days) - using lit-casted lookbacks
    F.sum(when(col("transaction_date") >= lookback_30d_lit, 1).otherwise(0)).alias("transactions_last_30d"),
    F.sum(when(col("transaction_date") >= lookback_60d_lit, 1).otherwise(0)).alias("transactions_last_60d"),
    F.sum(when(col("transaction_date") >= lookback_90d_lit, 1).otherwise(0)).alias("transactions_last_90d"),
    
    F.sum(when(col("transaction_date") >= lookback_30d_lit, col("revenue")).otherwise(0)).alias("revenue_last_30d"),
    F.sum(when(col("transaction_date") >= lookback_60d_lit, col("revenue")).otherwise(0)).alias("revenue_last_60d"),
    F.sum(when(col("transaction_date") >= lookback_90d_lit, col("revenue")).otherwise(0)).alias("revenue_last_90d"),
    
    # Temporal patterns
    F.countDistinct("transaction_month").alias("distinct_months_active"),
    F.avg("transaction_hour").alias("avg_purchase_hour"),
    
    # Product diversity
    F.countDistinct("product_id").alias("distinct_products_purchased"),
    F.countDistinct("product_category").alias("distinct_categories_purchased"),
    
    # Channel behavior
    F.sum(when(col("channel") == "Online", 1).otherwise(0)).alias("online_purchases"),
    F.sum(when(col("channel") == "Store", 1).otherwise(0)).alias("store_purchases"),
    F.sum(when(col("channel") == "Mobile App", 1).otherwise(0)).alias("mobile_purchases"),
    
    # Returns and cancellations
    F.sum(when(col("order_status") == "Returned", 1).otherwise(0)).alias("return_count"),
    F.sum(when(col("order_status") == "Cancelled", 1).otherwise(0)).alias("cancellation_count"),
    
    # Recency
    F.datediff(REFERENCE_DATE, F.max("transaction_date")).alias("days_since_last_purchase"),
    F.datediff(F.max("transaction_date"), F.min("transaction_date")).alias("customer_lifespan_days"),
    
    # Discount sensitivity
    F.avg("discount_percent").alias("avg_discount_used"),
    F.sum(when(col("discount_percent") > 0, 1).otherwise(0)).alias("discounted_purchases"),
    
    # Payment preferences
    F.sum(when(col("payment_method") == "Credit Card", 1).otherwise(0)).alias("credit_card_payments"),
    F.sum(when(col("payment_method") == "PayPal", 1).otherwise(0)).alias("paypal_payments")
)

# -------------------------
# Loyalty features
# -------------------------
loyalty_features = loyalty_silver.groupBy("customer_id").agg(
    F.max("points_balance").alias("current_loyalty_points"),
    F.sum(when(col("points_transaction_type") == "Earn", col("points_change")).otherwise(0)).alias("total_points_earned"),
    F.sum(when(col("points_transaction_type") == "Redeem", F.abs(col("points_change"))).otherwise(0)).alias("total_points_redeemed"),
    F.countDistinct("event_type").alias("distinct_loyalty_event_types"),
    F.max(when(col("tier") == "Gold", F.lit(3)).when(col("tier") == "Silver", F.lit(2)).otherwise(F.lit(1))).alias("loyalty_tier_numeric")
)

# -------------------------
# CRM interaction features
# -------------------------
crm_features = crm_silver.groupBy("customer_id").agg(
    F.count("interaction_id").alias("total_crm_interactions"),
    F.sum(when(col("interaction_type") == "Complaint", 1).otherwise(0)).alias("complaint_count"),
    F.avg("satisfaction_score").alias("avg_crm_satisfaction"),
    F.sum(when(col("is_unresolved") == True, 1).otherwise(0)).alias("unresolved_issues"),
    F.datediff(REFERENCE_DATE, F.max("interaction_date")).alias("days_since_last_contact")
)

# -------------------------
# Web analytics features
# -------------------------
web_features = web_silver.filter(col("customer_id").isNotNull()).groupBy("customer_id").agg(
    F.count("session_id").alias("total_web_sessions"),
    F.sum("page_views").alias("total_page_views"),
    F.avg("page_views").alias("avg_pages_per_session"),
    F.sum(when(col("is_bounce") == True, 1).otherwise(0)).alias("bounce_sessions"),
    F.sum(when(col("is_conversion") == True, 1).otherwise(0)).alias("web_conversions"),
    F.sum(when(col("is_abandoned_cart") == True, 1).otherwise(0)).alias("abandoned_cart_count"),
    F.countDistinct("device_type").alias("distinct_devices"),
    F.sum(when(col("device_type") == "Mobile", 1).otherwise(0)).alias("mobile_sessions")
)

# -------------------------
# Support ticket features
# -------------------------
ticket_features = tickets_silver.groupBy("customer_id").agg(
    F.count("ticket_id").alias("total_support_tickets"),
    F.sum(when(col("priority") == "High", 1).otherwise(0)).alias("high_priority_tickets"),
    F.sum(when(col("sla_breached") == True, 1).otherwise(0)).alias("sla_breach_count"),
    F.avg("customer_satisfaction").alias("avg_ticket_satisfaction"),
    F.sum(when(col("is_dissatisfied") == True, 1).otherwise(0)).alias("dissatisfied_ticket_count")
)

# -------------------------
# Join all features with customer base
# -------------------------
customer_features_gold = customers_silver.select(
    "customer_id",
    "age",
    "gender",
    "state",
    "customer_segment",
    "acquisition_channel",
    "customer_tenure_days"
) \
    .join(transaction_features, "customer_id", "left") \
    .join(loyalty_features, "customer_id", "left") \
    .join(crm_features, "customer_id", "left") \
    .join(web_features, "customer_id", "left") \
    .join(ticket_features, "customer_id", "left") \
    .join(
        rfm_gold.select("customer_id", "recency_score", "frequency_score", "monetary_score", "rfm_segment"),
        "customer_id", "left"
    )

# -------------------------
# Fill nulls with 0 for numeric features (preserve existing logic)
# -------------------------
numeric_columns = [
    "total_transactions", "total_revenue", "avg_transaction_value", "stddev_transaction_value",
    "transactions_last_30d", "transactions_last_60d", "transactions_last_90d",
    "revenue_last_30d", "revenue_last_60d", "revenue_last_90d",
    "distinct_months_active", "avg_purchase_hour",
    "distinct_products_purchased", "distinct_categories_purchased",
    "online_purchases", "store_purchases", "mobile_purchases",
    "return_count", "cancellation_count", "days_since_last_purchase", "customer_lifespan_days",
    "avg_discount_used", "discounted_purchases", "credit_card_payments", "paypal_payments",
    "current_loyalty_points", "total_points_earned", "total_points_redeemed",
    "distinct_loyalty_event_types", "loyalty_tier_numeric",
    "total_crm_interactions", "complaint_count", "avg_crm_satisfaction",
    "unresolved_issues", "days_since_last_contact",
    "total_web_sessions", "total_page_views", "avg_pages_per_session",
    "bounce_sessions", "web_conversions", "abandoned_cart_count",
    "distinct_devices", "mobile_sessions",
    "total_support_tickets", "high_priority_tickets", "sla_breach_count",
    "avg_ticket_satisfaction", "dissatisfied_ticket_count"
]

for col_name in numeric_columns:
    if col_name in customer_features_gold.columns:
        customer_features_gold = customer_features_gold.withColumn(
            col_name,
            coalesce(col(col_name), lit(0))
        )

# -------------------------
# Calculate derived features (preserve existing logic)
# -------------------------
customer_features_gold = customer_features_gold.withColumn(
    "purchase_frequency",
    when(col("customer_lifespan_days") > 0,
         col("total_transactions") / (col("customer_lifespan_days") / 30))
    .otherwise(0)
)

customer_features_gold = customer_features_gold.withColumn(
    "return_rate",
    when(col("total_transactions") > 0,
         col("return_count") / col("total_transactions"))
    .otherwise(0)
)

customer_features_gold = customer_features_gold.withColumn(
    "web_conversion_rate",
    when(col("total_web_sessions") > 0,
         col("web_conversions") / col("total_web_sessions"))
    .otherwise(0)
)

customer_features_gold = customer_features_gold.withColumn(
    "loyalty_redemption_rate",
    when(col("total_points_earned") > 0,
         col("total_points_redeemed") / col("total_points_earned"))
    .otherwise(0)
)

# Churn label (churned if no purchase in last 90 days)
customer_features_gold = customer_features_gold.withColumn(
    "is_churned",
    when(col("days_since_last_purchase") > 90, 1).otherwise(0)
)

# Add metadata
customer_features_gold = customer_features_gold.withColumn("gold_load_timestamp", current_timestamp())

# -------------------------
# Write to gold path (Delta) — path-based (no Unity Catalog)
# -------------------------
customer_features_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(gold_path)

# Read back for verification (path-based)
customer_features_back = spark.read.format("delta").load(gold_path)

print(f"✅ Created customer_features_gold at path: {gold_path} with {customer_features_back.count():,} records")
print(f"📊 Total features: {len(customer_features_back.columns)}")

print("\n📊 Churn Distribution:")
customer_features_back.groupBy("is_churned").count().show()


# COMMAND ----------

# Sales Aggregated for Forecasting (Gold) — no Unity Catalog / no DBFS saveAsTable
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import desc

print("📈 Creating time-series sales aggregation for forecasting...")

# CONFIG - update this to your desired gold folder path (local, mounted volume, etc.)
gold_path = "/Volumes/workspace/default/gold/sales_aggregated_gold"  # <-- update as needed

# Daily sales aggregation
daily_sales = transactions_silver.groupBy(
    F.to_date("transaction_date").alias("date")
).agg(
    F.count("transaction_id").alias("transaction_count"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_transaction_value"),
    F.countDistinct("customer_id").alias("unique_customers"),
    F.sum("quantity").alias("total_quantity_sold")
)

# Add day of week and other temporal features
daily_sales = daily_sales.withColumn("day_of_week", F.dayofweek("date")) \
    .withColumn("day_name", F.date_format("date", "EEEE")) \
    .withColumn("week_of_year", F.weekofyear("date")) \
    .withColumn("month", F.month("date")) \
    .withColumn("quarter", F.quarter("date")) \
    .withColumn("year", F.year("date")) \
    .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), 1).otherwise(0))

# Monthly aggregation
monthly_sales = transactions_silver.groupBy(
    F.year("transaction_date").alias("year"),
    F.month("transaction_date").alias("month")
).agg(
    F.count("transaction_id").alias("transaction_count"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_transaction_value"),
    F.countDistinct("customer_id").alias("unique_customers")
).withColumn("month_year", F.concat(F.col("year"), F.lit("-"), F.lpad(F.col("month").cast("string"), 2, "0"))) \
  .orderBy("year", "month")

# Category-wise sales trends
category_sales = transactions_silver.groupBy(
    F.to_date("transaction_date").alias("date"),
    "product_category"
).agg(
    F.sum("revenue").alias("category_revenue"),
    F.count("transaction_id").alias("category_transaction_count")
)

# Combine into single gold table with type indicator (daily level example)
sales_gold = daily_sales.withColumn("aggregation_level", F.lit("daily")) \
    .withColumn("category", F.lit(None).cast(StringType())) \
    .select(
        "date", "aggregation_level", "category",
        "transaction_count", "total_revenue", "avg_transaction_value",
        "unique_customers", "total_quantity_sold",
        "day_of_week", "day_name", "week_of_year", "month", "quarter", "year", "is_weekend"
    )

# Add metadata
sales_gold = sales_gold.withColumn("gold_load_timestamp", F.current_timestamp())

# Write to gold path (Delta) — no Unity Catalog, path-based
sales_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(gold_path)

# Read back for verification (path-based)
sales_gold_back = spark.read.format("delta").load(gold_path)

count_records = sales_gold_back.count()
print(f"✅ Created sales_aggregated_gold at path: {gold_path} with {count_records:,} records")

print("\n📊 Sample Daily Sales:")
sales_gold_back.orderBy(desc("date")).show(10, truncate=False)


# COMMAND ----------

# Product Performance Analytics (Gold) — no Unity Catalog / no DBFS saveAsTable
from pyspark.sql import functions as F
from pyspark.sql.functions import desc

print("📦 Creating product performance analytics table...")

# CONFIG - update to your desired gold folder path (local or mounted volume)
gold_path = "/Volumes/workspace/default/gold/product_performance_gold"  # <-- update as needed

# Ensure REFERENCE_DATE - using current_date() by default; replace with F.lit("YYYY-MM-DD").cast("date") if needed
REFERENCE_DATE = F.current_date()

# Product-level aggregations from transactions_silver
product_performance = transactions_silver.groupBy("product_id").agg(
    # Sales metrics
    F.count("transaction_id").alias("total_orders"),
    F.sum("quantity").alias("total_quantity_sold"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("revenue").alias("avg_order_value"),
    
    # Customer metrics
    F.countDistinct("customer_id").alias("unique_customers"),
    
    # Returns
    F.sum(F.when(F.col("order_status") == "Returned", 1).otherwise(0)).alias("return_count"),
    
    # Temporal
    F.max("transaction_date").alias("last_sold_date"),
    F.min("transaction_date").alias("first_sold_date")
)

# Join with product master data (left join to keep products with no sales)
product_performance = product_performance.join(
    products_silver.select(
        "product_id", "product_name", "category", "subcategory", 
        "brand", "price", "cost", "profit_margin_pct", "stock_quantity"
    ),
    on="product_id",
    how="left"
)

# Calculate derived metrics (safe divisions / null handling)
product_performance = product_performance.withColumn(
    "return_rate",
    F.when(F.col("total_orders") > 0, F.col("return_count") / F.col("total_orders")).otherwise(F.lit(0.0))
)

# days since last sale (using REFERENCE_DATE)
product_performance = product_performance.withColumn(
    "days_since_last_sale",
    F.datediff(REFERENCE_DATE, F.col("last_sold_date"))
)

# product lifespan in days (difference between last and first sale)
product_performance = product_performance.withColumn(
    "product_lifespan_days",
    F.datediff(F.col("last_sold_date"), F.col("first_sold_date"))
)

# avg daily sales (guard against divide-by-zero / null lifespan)
product_performance = product_performance.withColumn(
    "avg_daily_sales",
    F.when(F.col("product_lifespan_days") > 0,
           F.col("total_quantity_sold") / F.col("product_lifespan_days")
    ).otherwise(F.lit(0.0))
)

# Product status rules
product_performance = product_performance.withColumn(
    "product_status",
    F.when(F.col("days_since_last_sale") > 180, F.lit("Dormant"))
     .when(F.col("days_since_last_sale") > 90, F.lit("Slow Moving"))
     .when(F.col("avg_daily_sales") > 1, F.lit("Fast Moving"))
     .otherwise(F.lit("Active"))
)

# Add metadata
product_performance = product_performance.withColumn("gold_load_timestamp", F.current_timestamp())

# Optional: select/reorder columns for final gold table
final_cols = [
    "product_id", "product_name", "category", "subcategory", "brand",
    "price", "cost", "profit_margin_pct", "stock_quantity",
    "total_orders", "total_quantity_sold", "total_revenue", "avg_order_value",
    "unique_customers", "return_count", "return_rate",
    "first_sold_date", "last_sold_date", "product_lifespan_days", "avg_daily_sales", "days_since_last_sale",
    "product_status", "gold_load_timestamp"
]
# Keep only columns that exist (safe)
final_cols = [c for c in final_cols if c in product_performance.columns]
product_performance_final = product_performance.select(*final_cols)

# Write to gold path (Delta) — path-based
product_performance_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(gold_path)

# Read back for verification (path-based)
product_perf_back = spark.read.format("delta").load(gold_path)

count_records = product_perf_back.count()
print(f"✅ Created product_performance_gold at path: {gold_path} with {count_records:,} records")

print("\n📊 Top 10 Products by Revenue:")
product_perf_back.orderBy(desc("total_revenue")) \
    .select("product_name", "category", "total_revenue", "total_quantity_sold") \
    .show(10, truncate=False)


# COMMAND ----------

# Customer Lifetime Value Features (Gold) — path-based (no Unity Catalog / no DBFS)
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, round as spark_round
from pyspark.sql.types import DoubleType

print("💰 Creating CLV prediction features...")

# CONFIG - update to your desired gold folder path (local or mounted volume)
gold_path = "/Volumes/workspace/default/gold/customer_clv_features_gold"  # <-- update as needed

# Ensure REFERENCE_DATE exists; default to today if not set externally
try:
    REFERENCE_DATE  # noqa: F821
except NameError:
    REFERENCE_DATE = F.current_date()

# Build CLV features from transactions_silver
clv_features = transactions_silver.groupBy("customer_id").agg(
    # Revenue metrics
    F.sum("revenue").alias("historical_revenue"),
    F.avg("revenue").alias("avg_transaction_revenue"),
    
    # Frequency metrics
    F.count("transaction_id").alias("purchase_count"),
    F.countDistinct(F.to_date("transaction_date")).alias("purchase_days"),
    
    # Temporal metrics (use dates to avoid timezone issues)
    F.datediff(F.to_date(F.max("transaction_date")), F.to_date(F.min("transaction_date"))).alias("customer_age_days"),
    F.datediff(F.to_date(REFERENCE_DATE), F.to_date(F.max("transaction_date"))).alias("recency_days"),
    
    # Product diversity
    F.countDistinct("product_category").alias("categories_purchased"),
    F.avg("quantity").alias("avg_items_per_transaction")
)

# Calculate purchase frequency (transactions per month) - guard divide by zero
clv_features = clv_features.withColumn(
    "purchase_frequency_monthly",
    when(col("customer_age_days") > 0,
         col("purchase_count") / (col("customer_age_days") / F.lit(30.0))
    ).otherwise(F.lit(0.0))
)

# Calculate average time between purchases
clv_features = clv_features.withColumn(
    "avg_days_between_purchases",
    when(col("purchase_count") > 1,
         col("customer_age_days") / (col("purchase_count") - 1)
    ).otherwise(F.lit(None))
)

# Join with customer demographics (left join to keep customers with transactions)
clv_features = clv_features.join(
    customers_silver.select("customer_id", "age", "customer_segment", "customer_tenure_days"),
    on="customer_id",
    how="left"
)

# Replace possible NULLs in numeric inputs used in CLV formula to avoid surprises
clv_features = clv_features.withColumn("avg_transaction_revenue", col("avg_transaction_revenue").cast(DoubleType()))
clv_features = clv_features.fillna({"avg_transaction_revenue": 0.0, "purchase_frequency_monthly": 0.0})

# Calculate predicted CLV (simple heuristic model)
# CLV = (Avg Transaction Value) × (Purchase Frequency per month) × (Customer Lifespan in months)
# Assuming 3-year customer lifespan -> 36 months
clv_features = clv_features.withColumn(
    "predicted_clv_3yr",
    spark_round(col("avg_transaction_revenue") * col("purchase_frequency_monthly") * F.lit(36.0), 2)
)

# Customer value tier
clv_features = clv_features.withColumn(
    "clv_tier",
    when(col("predicted_clv_3yr") >= 10000, F.lit("Platinum"))
    .when(col("predicted_clv_3yr") >= 5000, F.lit("Gold"))
    .when(col("predicted_clv_3yr") >= 2000, F.lit("Silver"))
    .otherwise(F.lit("Bronze"))
)

# Add metadata
clv_features = clv_features.withColumn("gold_load_timestamp", F.current_timestamp())

# Optional: select/reorder columns for final gold table
final_cols = [
    "customer_id", "historical_revenue", "avg_transaction_revenue", "purchase_count", "purchase_days",
    "customer_age_days", "recency_days", "categories_purchased", "avg_items_per_transaction",
    "purchase_frequency_monthly", "avg_days_between_purchases",
    "age", "customer_segment", "customer_tenure_days",
    "predicted_clv_3yr", "clv_tier", "gold_load_timestamp"
]
final_cols = [c for c in final_cols if c in clv_features.columns]
clv_features_final = clv_features.select(*final_cols)

# Write to gold path (Delta) — path-based
clv_features_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(gold_path)

# Read back for verification (path-based)
clv_back = spark.read.format("delta").load(gold_path)

count_records = clv_back.count()
print(f"✅ Created customer_clv_features_gold at path: {gold_path} with {count_records:,} records")

print("\n📊 CLV Tier Distribution:")
clv_back.groupBy("clv_tier").count().orderBy(F.desc("count")).show(truncate=False)


# COMMAND ----------

# Complete update cell: MiniBatchKMeans local train + serverless-safe Spark assignment + write to Delta
from pyspark.sql import functions as F
from pyspark.sql.functions import col, desc, current_timestamp, array
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window
import numpy as np

print("🎯 Performing K-Means customer segmentation (MiniBatchKMeans locally + Spark assignment)...")

# -------------------------
# CONFIG - update paths
# -------------------------
customer_features_gold_path = "/Volumes/workspace/default/gold/customer_features_gold"   # source features (delta path)
customer_segments_gold_path = "/Volumes/workspace/default/gold/customer_segments_gold"   # destination delta path

# Clustering hyperparams
k = 5
train_sample_max = 20000   # max rows to collect locally for fitting (tune for driver memory)
random_seed = 42

# -------------------------
# Read features
# -------------------------
customer_features_df = spark.read.format("delta").load(customer_features_gold_path)

required_cols = [
    "customer_id",
    "total_transactions",
    "total_revenue",
    "days_since_last_purchase",
    "distinct_categories_purchased",
    "avg_transaction_value",
    "return_rate",
    "current_loyalty_points",
    "web_conversion_rate"
]
available_cols = [c for c in required_cols if c in customer_features_df.columns]
if "customer_id" not in available_cols:
    raise ValueError("customer_id missing from customer_features_gold. Aborting.")

segmentation_features = customer_features_df.select(*available_cols)

# Cast numerics to Double and fill nulls
num_cols = [c for c in available_cols if c != "customer_id"]
for c in num_cols:
    segmentation_features = segmentation_features.withColumn(c, col(c).cast(DoubleType()))
segmentation_features = segmentation_features.na.fill({c: 0.0 for c in num_cols})

total_count = segmentation_features.count()
print(f"Total customers available: {total_count:,}")

# -------------------------
# Determine sample fraction and collect locally for training
# -------------------------
sample_frac = 1.0
if total_count > train_sample_max:
    sample_frac = float(train_sample_max) / float(total_count)
    print(f"Sampling fraction for local training: {sample_frac:.6f} (~{train_sample_max:,} rows)")

train_sample_df = segmentation_features.sample(withReplacement=False, fraction=sample_frac, seed=random_seed)

# Collect numeric columns to pandas for scikit-learn
try:
    train_pdf = train_sample_df.select(*num_cols).toPandas()
except Exception as e:
    raise RuntimeError(f"Failed to collect training sample to Pandas. Reduce train_sample_max. Error: {e}")

if train_pdf.shape[0] == 0:
    raise RuntimeError("Training sample is empty. Aborting.")

X = train_pdf.values.astype(float)
print(f"Local training sample shape: {X.shape}")

# -------------------------
# Fit MiniBatchKMeans locally
# -------------------------
try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as e:
    raise RuntimeError("scikit-learn not available in this environment. Install scikit-learn or use a Spark-only fallback.") from e

mbk = MiniBatchKMeans(n_clusters=k, random_state=random_seed, batch_size=1024, max_iter=200)
mbk.fit(X)
centers = mbk.cluster_centers_  # numpy array shape (k, n_features)
print(f"Fitted MiniBatchKMeans locally. Centers shape: {centers.shape}")

# -------------------------
# Convert centers -> Spark DataFrame (serverless-safe; no broadcast)
# -------------------------
centers_list = [list(map(float, c)) for c in centers]
centers_df = spark.createDataFrame([(int(i), c) for i, c in enumerate(centers_list)],
                                   schema=["center_id", "center_vec"])

# -------------------------
# Serverless-safe nearest-center assignment using crossJoin + higher-order funcs
# -------------------------
# Build features array column (order must match training X)
features_array_col = array(*[col(c) for c in num_cols])
seg_with_vec = segmentation_features.withColumn("features_array", features_array_col)

# crossJoin with centers (k rows, tiny)
crossed = seg_with_vec.crossJoin(centers_df)

# compute squared euclidean distance using Spark SQL higher-order functions
# arrays_zip(features_array, center_vec) -> array<struct<f:double, c:double>>
# aggregate(..., 0D, (acc, x) -> acc + pow(x.f - x.c, 2))
dist_expr = "aggregate(arrays_zip(features_array, center_vec), 0D, (acc, x) -> acc + pow(x.features_array - x.center_vec, 2))"
crossed = crossed.withColumn("squared_dist", F.expr(dist_expr))

# For each customer pick the nearest center (rn = 1)
w = Window.partitionBy("customer_id").orderBy(col("squared_dist").asc())
ranked = crossed.withColumn("rn", F.row_number().over(w)).filter(col("rn") == 1).drop("rn", "squared_dist", "center_vec")

# center_id is the cluster assignment
clustered = ranked.withColumnRenamed("center_id", "cluster")

# -------------------------
# Inspect cluster profiles (counts + means)
# -------------------------
cluster_profiles = clustered.groupBy("cluster").agg(
    F.count("*").alias("customer_count"),
    F.round(F.avg("total_transactions"), 2).alias("avg_transactions"),
    F.round(F.avg("total_revenue"), 2).alias("avg_revenue"),
    F.round(F.avg("days_since_last_purchase"), 2).alias("avg_recency"),
    F.round(F.avg("return_rate"), 4).alias("avg_return_rate")
).orderBy("cluster")

print("\n📊 Cluster Profiles (preview):")
cluster_profiles.show(truncate=False)

# -------------------------
# Map clusters -> segment names (adjust after inspection if needed)
# -------------------------
segment_names = {
    0: "High Value Frequent Buyers",
    1: "Occasional Shoppers",
    2: "At Risk Customers",
    3: "New Customers",
    4: "Discount Seekers"
}
segment_name_expr = None
for k_id, name in segment_names.items():
    if segment_name_expr is None:
        segment_name_expr = F.when(col("cluster") == F.lit(k_id), F.lit(name))
    else:
        segment_name_expr = segment_name_expr.when(col("cluster") == F.lit(k_id), F.lit(name))
segment_name_expr = (segment_name_expr.otherwise(F.concat(F.lit("Segment_"), col("cluster").cast("string")))
                     if segment_name_expr is not None else F.concat(F.lit("Segment_"), col("cluster").cast("string")))

customer_segments_gold = clustered.select("customer_id", "cluster", *num_cols) \
    .withColumn("segment_name", segment_name_expr) \
    .withColumn("gold_load_timestamp", current_timestamp())

# -------------------------
# Write results to Delta (path-based)
# -------------------------
customer_segments_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(customer_segments_gold_path)

# Read back for verification
customer_segments_back = spark.read.format("delta").load(customer_segments_gold_path)
count_records = customer_segments_back.count()
print(f"\n✅ Created customer_segments_gold at path: {customer_segments_gold_path} with {count_records:,} records")

print("\n📊 Segment Distribution:")
customer_segments_back.groupBy("segment_name").count().orderBy(desc("count")).show(truncate=False)

print("\n🔎 Sample segments:")
customer_segments_back.select("customer_id", "cluster", "segment_name", *num_cols).show(10, truncate=False)


# COMMAND ----------

# Gold Layer Summary & Validation (path-based, no Unity Catalog)
import os
from pyspark.sql import functions as F

print("\n" + "="*80)
print("📊 GOLD LAYER FEATURE ENGINEERING SUMMARY")
print("="*80 + "\n")

# -------------------------
# CONFIG - update this base path if your gold deltas are elsewhere
# -------------------------
gold_base_path = "/Volumes/workspace/default/gold"  # <-- update to your gold root

# List of expected gold table names (logical)
gold_tables = [
    "customer_rfm_gold",
    "customer_features_gold",
    "sales_aggregated_gold",
    "product_performance_gold",
    "customer_clv_features_gold",
    "customer_segments_gold"
]

# Optional: explicit mapping if some tables use different folder names/locations
# e.g. {"customer_rfm_gold": "/mnt/gold/rfm_customer", ...}
table_paths = {t: os.path.join(gold_base_path, t) for t in gold_tables}

summary_data = []

def check_delta_path(path):
    """
    Try to read a Delta table from path and return (records, cols).
    Returns (None, None) on failure with printed error.
    """
    try:
        df = spark.read.format("delta").load(path)
        # compute metadata
        cnt = df.count()
        cols = len(df.columns)
        return int(cnt), int(cols)
    except Exception as e:
        # print short error to help debugging
        print(f"  - Error reading path '{path}': {e}")
        return None, None

for table_name in gold_tables:
    path = table_paths.get(table_name, os.path.join(gold_base_path, table_name))
    records, columns = check_delta_path(path)
    if records is None:
        summary_data.append({
            "Table": table_name,
            "Path": path,
            "Records": "N/A",
            "Columns": "N/A",
            "Status": "❌ Missing / Read Error"
        })
    else:
        summary_data.append({
            "Table": table_name,
            "Path": path,
            "Records": f"{records:,}",
            "Columns": columns,
            "Status": "✅ Success"
        })

# Convert to Spark DataFrame for pretty display
summary_df = spark.createDataFrame(summary_data)

print("\nGold layer summary (path-based):")
summary_df.select("Table", "Path", "Records", "Columns", "Status").show(truncate=False)

print("\n" + "="*80)
print("✅ GOLD LAYER FEATURE ENGINEERING COMPLETE! (path-based validation)")
print("="*80)
print("\n🎯 Gold Layer Tables:")
print("  1. customer_rfm_gold           - RFM segmentation (Champions, Loyal, At Risk, etc.)")
print("  2. customer_features_gold      - 60+ ML features for churn & analytics")
print("  3. sales_aggregated_gold       - Time-series data for forecasting")
print("  4. product_performance_gold    - Product analytics & inventory insights")
print("  5. customer_clv_features_gold  - CLV prediction features & tiers")
print("  6. customer_segments_gold      - K-Means clustering results")


# Optional: expose summary_df to use in downstream notebook cells
# e.g. summary_df.createOrReplaceTempView("gold_summary_temp")


# COMMAND ----------

# Feature Statistics & Data Quality (path-based, no Unity Catalog)
import os
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, count, avg, sum as _sum, desc

print("🔍 Gold Layer Data Quality Checks:\n")

# -------------------------
# CONFIG - update this to your gold root folder
# -------------------------
gold_base_path = "/Volumes/workspace/default/gold"  # <-- update if needed

# Map logical table names to delta paths (adjust if your folder names differ)
table_paths = {
    "customer_rfm_gold": os.path.join(gold_base_path, "customer_rfm_gold"),
    "customer_features_gold": os.path.join(gold_base_path, "customer_features_gold"),
    "sales_aggregated_gold": os.path.join(gold_base_path, "sales_aggregated_gold"),
    "product_performance_gold": os.path.join(gold_base_path, "product_performance_gold"),
    "customer_clv_features_gold": os.path.join(gold_base_path, "customer_clv_features_gold"),
    "customer_segments_gold": os.path.join(gold_base_path, "customer_segments_gold")
}

def load_delta_safe(path, logical_name):
    try:
        df = spark.read.format("delta").load(path)
        return df
    except Exception as e:
        print(f"⚠️  Could not read {logical_name} at path: {path}\n   Error: {e}\n")
        return None

# 1. Customer Features - Data Coverage
print("1. Customer Features - Data Coverage:")
features_df = load_delta_safe(table_paths["customer_features_gold"], "customer_features_gold")
if features_df is not None:
    features_stats = features_df.select(
        count("*").alias("total_customers"),
        _sum(when(col("total_transactions") > 0, 1).otherwise(0)).alias("customers_with_purchases"),
        _sum(when(col("is_churned") == 1, 1).otherwise(0)).alias("churned_customers"),
        avg("total_revenue").alias("avg_customer_revenue"),
        avg("customer_tenure_days").alias("avg_tenure_days")
    )
    features_stats.show(truncate=False)
else:
    print("Skipping customer features checks (table missing).\n")

# 2. RFM Distribution
print("\n2. RFM Segment Distribution:")
rfm_df = load_delta_safe(table_paths["customer_rfm_gold"], "customer_rfm_gold")
if rfm_df is not None:
    rfm_dist = rfm_df.groupBy("rfm_segment").agg(
        count("*").alias("customer_count"),
        avg("monetary_value").alias("avg_revenue")
    ).orderBy(desc("customer_count"))
    rfm_dist.show(truncate=False)
else:
    print("Skipping RFM distribution (table missing).\n")

# 3. Sales Trends
print("\n3. Recent Sales Trends (Last 30 Days):")
sales_df = load_delta_safe(table_paths["sales_aggregated_gold"], "sales_aggregated_gold")
if sales_df is not None:
    # If 'date' is a string, try casting; otherwise assume it's date/timestamp
    if "date" in sales_df.columns:
        recent_sales = sales_df.orderBy(desc("date")).limit(30).agg(
            _sum("total_revenue").alias("revenue_last_30d"),
            avg("total_revenue").alias("avg_daily_revenue"),
            _sum("transaction_count").alias("total_transactions")
        )
        recent_sales.show(truncate=False)
    else:
        print("sales_aggregated_gold has no 'date' column; skipping recent sales aggregation.")
else:
    print("Skipping sales trends (table missing).\n")

# 4. Product Performance
print("\n4. Product Performance Overview:")
product_df = load_delta_safe(table_paths["product_performance_gold"], "product_performance_gold")
if product_df is not None:
    if "product_status" in product_df.columns:
        product_stats = product_df.groupBy("product_status").agg(
            count("*").alias("product_count"),
            _sum("total_revenue").alias("total_revenue")
        ).orderBy(desc("total_revenue"))
    else:
        # fallback: aggregate by presence (all products)
        product_stats = product_df.agg(
            count("*").alias("product_count"),
            _sum("total_revenue").alias("total_revenue")
        )
    product_stats.show(truncate=False)
else:
    print("Skipping product performance (table missing).\n")

# 5. CLV Distribution
print("\n5. Customer Lifetime Value Distribution:")
clv_df = load_delta_safe(table_paths["customer_clv_features_gold"], "customer_clv_features_gold")
if clv_df is not None:
    if "clv_tier" in clv_df.columns:
        clv_dist = clv_df.groupBy("clv_tier").agg(
            count("*").alias("customer_count"),
            avg("predicted_clv_3yr").alias("avg_clv"),
            _sum("historical_revenue").alias("total_revenue")
        ).orderBy(desc("avg_clv"))
    else:
        clv_dist = clv_df.agg(
            count("*").alias("customer_count"),
            avg("predicted_clv_3yr").alias("avg_clv"),
            _sum("historical_revenue").alias("total_revenue")
        )
    clv_dist.show(truncate=False)
else:
    print("Skipping CLV distribution (table missing).\n")

# 6. Segment Characteristics
print("\n6. Customer Segment Characteristics:")
segments_df = load_delta_safe(table_paths["customer_segments_gold"], "customer_segments_gold")
if segments_df is not None:
    # join back to features if total_revenue/other fields not present in segments table
    seg_source = segments_df
    # If avg metrics not present in segments table, try to left-join features_df to get them
    needed_cols = {"total_revenue", "total_transactions", "days_since_last_purchase"}
    if not needed_cols.issubset(set(seg_source.columns)) and features_df is not None:
        seg_source = seg_source.join(features_df.select("customer_id", *list(needed_cols)), "customer_id", "left")
    if "segment_name" in seg_source.columns:
        segment_chars = seg_source.groupBy("segment_name").agg(
            count("*").alias("count"),
            avg("total_revenue").alias("avg_revenue"),
            avg("total_transactions").alias("avg_transactions"),
            avg("days_since_last_purchase").alias("avg_recency")
        ).orderBy(desc("avg_revenue"))
    else:
        segment_chars = seg_source.groupBy("cluster").agg(
            count("*").alias("count"),
            avg("total_revenue").alias("avg_revenue"),
            avg("total_transactions").alias("avg_transactions"),
            avg("days_since_last_purchase").alias("avg_recency")
        ).orderBy(desc("avg_revenue"))
    segment_chars.show(truncate=False)
else:
    print("Skipping segment characteristics (table missing).\n")

print("\n✅ All Gold Layer Quality Checks Complete!")


# COMMAND ----------

## Export Feature Importance Summary

# COMMAND ----------

print("📋 Creating Feature Inventory for ML Models...\n")

# Create a comprehensive feature catalog
feature_catalog_data = [
    ("customer_features_gold", "Churn Prediction", "days_since_last_purchase", "Numeric", "Recency metric"),
    ("customer_features_gold", "Churn Prediction", "transactions_last_90d", "Numeric", "Recent activity"),
    ("customer_features_gold", "Churn Prediction", "avg_crm_satisfaction", "Numeric", "Customer sentiment"),
    ("customer_features_gold", "Churn Prediction", "unresolved_issues", "Numeric", "Support quality"),
    ("customer_features_gold", "Churn Prediction", "web_conversion_rate", "Numeric", "Digital engagement"),
    
    ("customer_rfm_gold", "Segmentation", "recency_score", "Numeric", "RFM - Recency"),
    ("customer_rfm_gold", "Segmentation", "frequency_score", "Numeric", "RFM - Frequency"),
    ("customer_rfm_gold", "Segmentation", "monetary_score", "Numeric", "RFM - Monetary"),
    
    ("customer_clv_features_gold", "CLV Prediction", "historical_revenue", "Numeric", "Past spending"),
    ("customer_clv_features_gold", "CLV Prediction", "purchase_frequency_monthly", "Numeric", "Purchase rate"),
    ("customer_clv_features_gold", "CLV Prediction", "avg_transaction_revenue", "Numeric", "Transaction value"),
    ("customer_clv_features_gold", "CLV Prediction", "categories_purchased", "Numeric", "Product diversity"),
    
    ("sales_aggregated_gold", "Demand Forecasting", "total_revenue", "Numeric", "Target variable"),
    ("sales_aggregated_gold", "Demand Forecasting", "day_of_week", "Categorical", "Temporal pattern"),
    ("sales_aggregated_gold", "Demand Forecasting", "is_weekend", "Binary", "Weekend effect"),
    ("sales_aggregated_gold", "Demand Forecasting", "month", "Categorical", "Seasonality"),
    
    ("product_performance_gold", "Inventory Optimization", "avg_daily_sales", "Numeric", "Velocity"),
    ("product_performance_gold", "Inventory Optimization", "return_rate", "Numeric", "Quality indicator"),
    ("product_performance_gold", "Inventory Optimization", "days_since_last_sale", "Numeric", "Obsolescence risk")
]

feature_catalog = spark.createDataFrame(
    feature_catalog_data,
    ["source_table", "ml_use_case", "feature_name", "feature_type", "description"]
)

print("📊 Feature Catalog for ML Models:")
feature_catalog.show(50, truncate=False)

# Write feature catalog
feature_catalog.withColumn("gold_load_timestamp", current_timestamp()) \
    .write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG_NAME}.{SCHEMA_NAME}.ml_feature_catalog")

print("\n✅ Feature catalog saved to ml_feature_catalog table")
print("\n" + "="*80)
print("🎉 GOLD LAYER COMPLETE - READY FOR MACHINE LEARNING!")
print("="*80)

# COMMAND ----------

