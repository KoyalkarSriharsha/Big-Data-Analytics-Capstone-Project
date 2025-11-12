# Databricks notebook source


# Imports & Configuration (NO MLflow)
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Optional local libs used later (scikit-learn, xgboost, prophet) will be imported inside cells where needed.

from datetime import datetime
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Initialize Spark (re-uses existing session if present)
spark = SparkSession.builder.appName("Customer360_ML_Models").getOrCreate()

# -------------------------
# Configuration - update paths to suit your environment
# -------------------------
# Root folder where gold Delta tables are stored (path-based, no Unity Catalog)
gold_base_path = "/Volumes/workspace/default/gold"   # <-- update if different

# Paths to gold tables (Delta paths)
customer_features_path = os.path.join(gold_base_path, "customer_features_gold")
sales_aggregated_path  = os.path.join(gold_base_path, "sales_aggregated_gold")
clv_features_path      = os.path.join(gold_base_path, "customer_clv_features_gold")
customer_segments_path = os.path.join(gold_base_path, "customer_segments_gold")

# Where to save model artifacts (pick a writable path)
artifacts_base_path = "/Volumes/workspace/default/artifacts"  # <-- update as needed
os.makedirs(artifacts_base_path, exist_ok=True)

# Experiment metadata (for your bookkeeping when not using MLflow)
EXPERIMENT_NAME = "customer360_ml_local_no_mlflow"
RUN_TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

print("✅ Configuration loaded (NO MLflow)")
print(f"Gold base path: {gold_base_path}")
print(f"Artifacts base path: {artifacts_base_path}")
print(f"Experiment name (metadata only): {EXPERIMENT_NAME}")
print(f"Run timestamp: {RUN_TIMESTAMP}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn Prediction Model

# COMMAND ----------

# 1. Churn Prediction Model 
import os
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from datetime import datetime

print("🎯 Training Churn Prediction Model ...")


# Load features from Delta path (path-based)
print(f"Loading customer features from: {customer_features_path}")
churn_data = spark.read.format("delta").load(customer_features_path)

# Select relevant features for churn prediction (preserve your list)
feature_columns = [
    "days_since_last_purchase",
    "total_transactions",
    "total_revenue",
    "avg_transaction_value",
    "transactions_last_30d",
    "transactions_last_60d",
    "transactions_last_90d",
    "revenue_last_30d",
    "distinct_categories_purchased",
    "return_count",
    "return_rate",
    "current_loyalty_points",
    "loyalty_tier_numeric",
    "total_crm_interactions",
    "complaint_count",
    "avg_crm_satisfaction",
    "unresolved_issues",
    "total_web_sessions",
    "web_conversions",
    "web_conversion_rate",
    "abandoned_cart_count",
    "total_support_tickets",
    "high_priority_tickets",
    "dissatisfied_ticket_count",
    "purchase_frequency",
    "customer_tenure_days",
    "recency_score",
    "frequency_score",
    "monetary_score"
]

# Keep only available columns to avoid errors if some are missing
available_features = [c for c in feature_columns if c in churn_data.columns]
missing_features = [c for c in feature_columns if c not in churn_data.columns]
if missing_features:
    print(f"⚠️  Warning: the following features are missing and will be skipped: {missing_features}")

# Build dataframe with label (alias to 'label') and fill nulls with 0 (preserve your logic)
churn_df = churn_data.select(
    "customer_id",
    *available_features,
    col("is_churned").alias("label")
).na.fill(0)

# Check class distribution
print("\n📊 Churn Label Distribution:")
churn_df.groupBy("label").count().show(truncate=False)

# Split data
train_churn, test_churn = churn_df.randomSplit([0.8, 0.2], seed=42)
train_count = train_churn.count()
test_count = test_churn.count()
print(f"\nTraining set: {train_count:,} records")
print(f"Test set: {test_count:,} records")

# -------------------------
# Logistic Regression Model
# -------------------------
print("\n🔵 Training Logistic Regression for Churn Prediction...")

# Feature engineering pipeline
assembler = VectorAssembler(inputCols=available_features, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5
)

pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

# Train model
lr_model = pipeline_lr.fit(train_churn)

# Make predictions
lr_predictions = lr_model.transform(test_churn)

# Evaluate
binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = binary_eval.evaluate(lr_predictions)
# areaUnderPR uses same evaluator with different metric name
binary_eval_pr = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
auc_pr = binary_eval_pr.evaluate(lr_predictions)

mc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = mc_evaluator.evaluate(lr_predictions, {mc_evaluator.metricName: "accuracy"})
f1 = mc_evaluator.evaluate(lr_predictions, {mc_evaluator.metricName: "f1"})

print(f"\n✅ Logistic Regression Results:")
print(f"   AUC-ROC: {auc_roc:.4f}")
print(f"   AUC-PR : {auc_pr:.4f}")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1 Score: {f1:.4f}")

# Confusion matrix
print("\n📊 Confusion Matrix (Logistic Regression):")
lr_predictions.groupBy("label", "prediction").count().show(truncate=False)

# Save LR pipeline model to artifacts path
lr_model_path = os.path.join(artifacts_base_path, f"churn_model_lr_{RUN_TIMESTAMP}")
print(f"\nSaving Logistic Regression pipeline model to: {lr_model_path}")
lr_model.write().overwrite().save(lr_model_path)

# -------------------------
# Gradient Boosted Trees Model
# -------------------------
print("\n🌲 Training Gradient Boosted Trees for Churn Prediction...")

# Use assembler directly to produce 'features' column (no scaler in your original GBT cell)
assembler_gbt = VectorAssembler(inputCols=available_features, outputCol="features")

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    maxDepth=5,
    seed=42
)

pipeline_gbt = Pipeline(stages=[assembler_gbt, gbt])

# Train model
gbt_model = pipeline_gbt.fit(train_churn)

# Make predictions
gbt_predictions = gbt_model.transform(test_churn)

# Evaluate
auc_roc_gbt = binary_eval.evaluate(gbt_predictions)
auc_pr_gbt = binary_eval_pr.evaluate(gbt_predictions)
accuracy_gbt = mc_evaluator.evaluate(gbt_predictions, {mc_evaluator.metricName: "accuracy"})
f1_gbt = mc_evaluator.evaluate(gbt_predictions, {mc_evaluator.metricName: "f1"})

print(f"\n✅ Gradient Boosted Trees Results:")
print(f"   AUC-ROC: {auc_roc_gbt:.4f}")
print(f"   AUC-PR : {auc_pr_gbt:.4f}")
print(f"   Accuracy: {accuracy_gbt:.4f}")
print(f"   F1 Score: {f1_gbt:.4f}")

# Feature importance extraction (from trained GBT stage)
try:
    gbt_trained = gbt_model.stages[-1]
    # featureImportances is a SparseVector-like; convert to array if available
    fi = list(zip(available_features, gbt_trained.featureImportances.toArray()))
    fi.sort(key=lambda x: x[1], reverse=True)
    print("\n📊 Top 10 Feature Importances (GBT):")
    for feat, imp in fi[:10]:
        print(f"   {feat}: {imp:.6f}")
except Exception as e:
    print("Could not extract feature importances:", e)

# Confusion matrix
print("\n📊 Confusion Matrix (GBT):")
gbt_predictions.groupBy("label", "prediction").count().show(truncate=False)

# Save GBT pipeline model to artifacts path
gbt_model_path = os.path.join(artifacts_base_path, f"churn_model_gbt_{RUN_TIMESTAMP}")
print(f"\nSaving GBT pipeline model to: {gbt_model_path}")
gbt_model.write().overwrite().save(gbt_model_path)

print("\n✅ Churn models trained and saved .")
print(f" - Logistic Regression model path: {lr_model_path}")
print(f" - GBT model path: {gbt_model_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Lifetime Value (CLV) Prediction

# COMMAND ----------

# 2. Customer Lifetime Value (CLV) Prediction (fixed ambiguous column, path-based, no MLflow)
import os
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from datetime import datetime

print("💰 Training CLV Prediction Model (fixed ambiguous join)...")

# --- CONFIG: fallbacks if not already set in notebook ---
try:
    clv_features_path  # noqa: F821
except NameError:
    clv_features_path = "/Volumes/workspace/default/gold/customer_clv_features_gold"

try:
    customer_features_path  # noqa: F821
except NameError:
    customer_features_path = "/Volumes/workspace/default/gold/customer_features_gold"

try:
    artifacts_base_path  # noqa: F821
except NameError:
    artifacts_base_path = "/Volumes/workspace/default/artifacts"

# timestamp for model folder
try:
    RUN_TIMESTAMP  # noqa: F821
except NameError:
    RUN_TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

os.makedirs(artifacts_base_path, exist_ok=True)

# Load CLV gold (path-based)
print(f"Loading CLV features from: {clv_features_path}")
clv_data = spark.read.format("delta").load(clv_features_path)

# Load customer_features for enrichment (path-based)
print(f"Loading customer features for enrichment from: {customer_features_path}")
cust_feat_df = spark.read.format("delta").load(customer_features_path)

# Base CLV features (keep your original list but guard missing columns)
clv_feature_columns = [
    "historical_revenue",
    "avg_transaction_revenue",
    "purchase_count",
    "purchase_frequency_monthly",
    "avg_days_between_purchases",
    "categories_purchased",
    "avg_items_per_transaction",
    "recency_days",
    "customer_age_days"
]

# Enrichment columns to join from customer_features_gold (we will rename them to avoid ambiguity)
enrichment_cols = [
    "customer_tenure_days",
    "return_rate",
    "current_loyalty_points",
    "web_conversion_rate"
]

# Check availability and warn if missing
missing_clv = [c for c in clv_feature_columns if c not in clv_data.columns]
if missing_clv:
    print(f"⚠️  Warning: missing CLV input columns in {clv_features_path}: {missing_clv}")

missing_enrich = [c for c in enrichment_cols if c not in cust_feat_df.columns]
if missing_enrich:
    print(f"⚠️  Warning: missing enrichment columns in {customer_features_path}: {missing_enrich}")

# --- IMPORTANT: select and rename enrichment columns to avoid ambiguous names ---
# We'll prefix enrichment columns with "cf_" (customer features) to be explicit.
available_enrich = [c for c in enrichment_cols if c in cust_feat_df.columns]
renamed_enrich = [col(c).alias(f"cf_{c}") for c in available_enrich]

# Build a clean CLV base dataframe selecting only needed CLV columns (and customer_id + label if present)
# Use explicit selection to avoid pulling duplicate columns with same name
clv_select_cols = ["customer_id"] + [c for c in clv_feature_columns if c in clv_data.columns]
# include label column choice if present
label_col_name = "predicted_clv_3yr" if "predicted_clv_3yr" in clv_data.columns else "historical_revenue"
if label_col_name in clv_data.columns:
    # ensure label will be included (but don't duplicate)
    if label_col_name not in clv_select_cols:
        clv_select_cols.append(label_col_name)
else:
    raise RuntimeError(f"Neither 'predicted_clv_3yr' nor 'historical_revenue' found in {clv_features_path} - cannot train CLV model.")

clv_base_df = clv_data.select(*clv_select_cols)

# Now select enrichment (customer features) columns renamed
cust_enrich_df = cust_feat_df.select("customer_id", *renamed_enrich)

# Left join clv_base_df with cust_enrich_df on customer_id
clv_enriched = clv_base_df.join(cust_enrich_df, on="customer_id", how="left")

# Build final feature list: CLV base features (that exist) + renamed enrichment columns (cf_*)
final_feature_columns = [c for c in clv_feature_columns if c in clv_enriched.columns] + [f"cf_{c}" for c in available_enrich]
print(f"Using CLV features (final): {final_feature_columns}")

# Prepare DataFrame with label 'label' for supervised regression and fill nulls
clv_df = clv_enriched.select(
    "customer_id",
    *final_feature_columns,
    col(label_col_name).alias("label")
).na.fill(0.0)

# Filter out extreme outliers (CLV > 0 and < 100000) as in your original logic
clv_df = clv_df.filter((col("label") > 0) & (col("label") < 100000))

# Split into train/test
train_clv, test_clv = clv_df.randomSplit([0.8, 0.2], seed=42)
train_count = train_clv.count()
test_count = test_clv.count()
print(f"Training set: {train_count:,} records")
print(f"Test set: {test_count:,} records")

# -------------------------
# Linear Regression pipeline
# -------------------------
print("\n📈 Training Linear Regression for CLV Prediction...")

assembler = VectorAssembler(inputCols=final_feature_columns, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)

lr_clv = LinearRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.5
)

pipeline_clv = Pipeline(stages=[assembler, scaler, lr_clv])

# Fit model
clv_model = pipeline_clv.fit(train_clv)

# Make predictions
clv_predictions = clv_model.transform(test_clv)

# Evaluate
evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
evaluator_r2  = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(clv_predictions)
mae = evaluator_mae.evaluate(clv_predictions)
r2 = evaluator_r2.evaluate(clv_predictions)

print(f"\n✅ CLV Prediction Results (LinearRegression):")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   MAE : ${mae:,.2f}")
print(f"   R²  : {r2:.4f}")

# Show sample predictions
print("\n📊 Sample CLV Predictions:")
clv_predictions.select("customer_id", "label", "prediction").show(10, truncate=False)

# Save CLV model pipeline to artifacts path
clv_model_path = os.path.join(artifacts_base_path, f"clv_model_linear_{RUN_TIMESTAMP}")
print(f"\nSaving CLV pipeline model to: {clv_model_path}")
clv_model.write().overwrite().save(clv_model_path)

print("\n✅ CLV model trained and saved (no MLflow).")
print(f" - CLV Linear model path: {clv_model_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Demand Forecasting

# COMMAND ----------

# Robust Demand Forecasting cell (SARIMAX fallback fixed for index/frequency issues)
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pyspark.sql.functions import lit, current_timestamp

print("📈 Preparing data for Demand Forecasting (fixed SARIMAX fallback)...")

# paths (fall back to previously set vars)
try:
    sales_aggregated_path
except NameError:
    sales_aggregated_path = "/Volumes/workspace/default/gold/sales_aggregated_gold"
try:
    gold_base_path
except NameError:
    gold_base_path = "/Volumes/workspace/default/gold"
forecast_path = os.path.join(gold_base_path, "revenue_forecast_gold")

# load sales
sales_df = spark.read.format("delta").load(sales_aggregated_path)
sales_pd = sales_df.select("date", "total_revenue", "transaction_count").orderBy("date").toPandas()
sales_pd['date'] = pd.to_datetime(sales_pd['date'])
sales_pd = sales_pd.sort_values('date').reset_index(drop=True)

prophet_df = sales_pd[['date', 'total_revenue']].copy().rename(columns={'date':'ds', 'total_revenue':'y'}).dropna()

if len(prophet_df) < 14:
    raise RuntimeError("Not enough historical data for reliable forecasting (need >= 14 days).")

horizon = 30

def safe_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0
    return mae, rmse, mape

forecast_df = None
model_used = None
mae = rmse = mape = float("nan")

# Try Prophet first
try:
    try:
        from prophet import Prophet
    except Exception:
        from fbprophet import Prophet  # fallback package name if legacy

    print("🔮 Fitting Prophet (if available)...")
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(prophet_df.rename(columns={'ds':'ds','y':'y'}))
    future = m.make_future_dataframe(periods=horizon, freq='D')
    forecast = m.predict(future)

    # in-sample metrics
    insample_len = len(prophet_df)
    y_true = prophet_df['y'].values
    y_pred_insample = forecast.loc[:insample_len - 1, 'yhat'].values
    mae, rmse, mape = safe_metrics(y_true, y_pred_insample)

    model_used = "Prophet"
    forecast_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={
        'ds':'forecast_date','yhat':'predicted_revenue','yhat_lower':'lower_bound','yhat_upper':'upper_bound'
    })

except Exception as e_prop:
    print("⚠️ Prophet not available or failed; falling back to SARIMAX. Reason:", e_prop)

    # Prepare a DAILY ts with explicit daily frequency (fill missing dates with 0)
    ts = prophet_df.set_index('ds')['y']
    start_date = ts.index.min().date()
    end_date = ts.index.max().date()
    full_index = pd.date_range(start=start_date, end=end_date, freq='D')
    ts_daily = ts.reindex(full_index).fillna(0.0)
    ts_daily.index.name = 'ds'

    try:
        import statsmodels.api as sm
        print("🔁 Fitting SARIMAX fallback model (robust index handling)...")

        # Fit SARIMAX on ts_daily (no ambiguous index)
        # Use simple seasonal order (weekly)
        sarimax = sm.tsa.SARIMAX(ts_daily, order=(1,1,1), seasonal_order=(0,1,1,7),
                                 enforce_stationarity=False, enforce_invertibility=False)
        res = sarimax.fit(disp=False)

        # In-sample fitted values (aligned to ts_daily index)
        fitted = res.fittedvalues.copy()
        # Ensure fitted has same length as ts_daily
        fitted = fitted.reindex(ts_daily.index).fillna(method='bfill').fillna(method='ffill')

        # Future forecast using get_forecast(steps=horizon)
        pred_future = res.get_forecast(steps=horizon)
        pred_df_future = pred_future.summary_frame(alpha=0.05)
        # The index of pred_df_future are future dates starting after ts_daily.index[-1]
        # Build a combined forecast DataFrame: in-sample + future
        pred_in_sample_df = pd.DataFrame({
            'forecast_date': ts_daily.index,
            'predicted_revenue': fitted.values,
            'lower_bound': np.nan,  # not always provided for in-sample fittedvalues
            'upper_bound': np.nan
        })
        # Future summary_frame includes mean, mean_ci_lower, mean_ci_upper
        pred_future_df = pred_df_future.reset_index().rename(columns={
            'index':'forecast_date', 'mean':'predicted_revenue',
            'mean_ci_lower':'lower_bound', 'mean_ci_upper':'upper_bound'
        })

        # Combine
        forecast_df = pd.concat([pred_in_sample_df, pred_future_df], ignore_index=True)

        # Compute in-sample metrics comparing original ts_daily (y_true) with fitted (only for in-sample range)
        y_true = ts_daily.values
        y_pred_insample = fitted.values
        mae, rmse, mape = safe_metrics(y_true, y_pred_insample)

        model_used = "SARIMAX"

        print(f"SARIMAX fitted. In-sample MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {np.nan if np.isnan(mape) else f'{mape:.2f}%'}")

    except Exception as e_sar:
        print("❌ SARIMAX also failed:", e_sar)
        raise RuntimeError("Both Prophet and SARIMAX failed for forecasting.") from e_sar

# Finalize forecast_df (pandas) and write to Delta path
if forecast_df is None:
    raise RuntimeError("Forecast generation failed; forecast_df is empty.")

# Normalize columns and types
forecast_df['forecast_date'] = pd.to_datetime(forecast_df['forecast_date'])
# Add metadata
forecast_df['model_type'] = model_used
forecast_df['forecast_timestamp'] = pd.Timestamp.utcnow()

# Convert to Spark DataFrame and write as Delta (overwrite)
spark_forecast_df = spark.createDataFrame(forecast_df)
spark_forecast_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(forecast_path)

# Display next 7 days (future)
last_hist_date = pd.to_datetime(sales_pd['date'].max())
future_rows = forecast_df[forecast_df['forecast_date'] > last_hist_date].head(7)
if future_rows.empty:
    future_rows = forecast_df.tail(7)

print("\n📊 Next 7 Days Revenue Forecast (sample):")
print(future_rows[['forecast_date','predicted_revenue','lower_bound','upper_bound']].to_string(index=False))

print("\n📈 Forecasting complete.")
print(f" - Method used: {model_used}")
print(f" - In-sample MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {np.nan if np.isnan(mape) else f'{mape:.2f}%'}")
print(f" - Forecast Delta path: {forecast_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Anomaly Detection

# COMMAND ----------

# 4. Anomaly Detection (Behavioral) — path-based, no MLflow
import os
from datetime import datetime
from pyspark.sql.functions import col, current_timestamp
import pandas as pd
import numpy as np

print("🔍 Training Anomaly Detection Model (no MLflow)...")

# -------------------------
# CONFIG: fallbacks if not already set
# -------------------------
try:
    customer_features_path  # noqa: F821
except NameError:
    customer_features_path = "/Volumes/workspace/default/gold/customer_features_gold"

try:
    artifacts_base_path  # noqa: F821
except NameError:
    artifacts_base_path = "/Volumes/workspace/default/artifacts"

# where to write anomalies delta
try:
    gold_base_path  # noqa: F821
except NameError:
    gold_base_path = "/Volumes/workspace/default/gold"

anomalies_delta_path = os.path.join(gold_base_path, "customer_anomalies_gold")
os.makedirs(artifacts_base_path, exist_ok=True)

# timestamp for model artifacts
RUN_TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

# -------------------------
# Load features (Delta path)
# -------------------------
print(f"Loading customer features from: {customer_features_path}")
features_df = spark.read.format("delta").load(customer_features_path)

# Features to use (as in your original cell)
anomaly_features = [
    "total_transactions",
    "total_revenue",
    "avg_transaction_value",
    "return_rate",
    "days_since_last_purchase",
    "web_conversion_rate",
    "purchase_frequency"
]

# Keep only available columns
available_features = [c for c in anomaly_features if c in features_df.columns]
missing = [c for c in anomaly_features if c not in features_df.columns]
if missing:
    print(f"⚠️ Warning: missing anomaly feature columns and will be skipped: {missing}")

if "customer_id" not in features_df.columns:
    raise RuntimeError("customer_id column is required in customer_features_gold")

anomaly_df = features_df.select("customer_id", *available_features).na.fill(0)

# If dataset large, sample before collect to driver to avoid OOM
max_collect_rows = 20000  # change as needed based on driver memory
total_rows = anomaly_df.count()
sample_fraction = 1.0
if total_rows > max_collect_rows:
    sample_fraction = float(max_collect_rows) / float(total_rows)
    print(f"Large dataset detected: {total_rows:,} rows. Sampling fraction {sample_fraction:.4f} (~{max_collect_rows} rows) for local training.")
    anomaly_sample_df = anomaly_df.sample(withReplacement=False, fraction=sample_fraction, seed=42)
else:
    anomaly_sample_df = anomaly_df

# Collect to pandas
print("Collecting data to driver for sklearn training...")
anomaly_pd = anomaly_sample_df.toPandas().fillna(0.0)

# Prepare X and ids
X = anomaly_pd[available_features].values.astype(float)
customer_ids = anomaly_pd["customer_id"].values

# -------------------------
# Train IsolationForest (scikit-learn)
# -------------------------
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib

    print("\n🌲 Training Isolation Forest for Anomaly Detection (scikit-learn)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=0.05,  # adjust based on expected anomaly rate
        random_state=42,
        n_estimators=100
    )
    preds = iso.fit_predict(X_scaled)  # -1 anomaly, 1 normal
    anomaly_scores = iso.score_samples(X_scaled)  # higher = more normal, lower = more anomalous

    # Build results pandas DataFrame (for the sampled segment)
    results_pd = pd.DataFrame({
        "customer_id": customer_ids,
        "is_anomaly": (preds == -1).astype(int),
        "anomaly_score": anomaly_scores
    })

    anomaly_count = int((results_pd["is_anomaly"] == 1).sum())
    anomaly_pct = float(anomaly_count) / len(results_pd) * 100.0

    print(f"\n✅ Anomaly Detection Results (on training sample):")
    print(f"   Sample size: {len(results_pd):,}")
    print(f"   Anomalies Detected: {anomaly_count:,} ({anomaly_pct:.2f}%)")

    # Convert pandas results back to Spark DataFrame, add detection timestamp
    results_sdf = spark.createDataFrame(results_pd)
    results_sdf = results_sdf.withColumn("detection_timestamp", current_timestamp())

    # If you sampled, you'd typically want to apply the model to the full dataset.
    # We'll attempt to score the full dataset in a distributed manner if possible:
    apply_to_full = True
    if apply_to_full:
        try:
            # Broadcast model artifacts: we will save scaler & model to artifacts and then score by applying them in pandas UDF
            model_artifact_dir = os.path.join(artifacts_base_path, f"isoforest_{RUN_TIMESTAMP}")
            os.makedirs(model_artifact_dir, exist_ok=True)
            scaler_path = os.path.join(model_artifact_dir, "scaler.joblib")
            model_path = os.path.join(model_artifact_dir, "isoforest.joblib")
            joblib.dump(scaler, scaler_path)
            joblib.dump(iso, model_path)
            print(f"Saved scaler & model to: {model_artifact_dir}")

            # Score full dataset in batches using Pandas (avoid huge collect). We'll use a simple approach:
            #  - iterate over partitions, collect partition to pandas, scale using saved scaler, predict, write partition results.
            # NOTE: This is driver-assisted and may be slower but avoids full model serialization issues on serverless.
            from pyspark.sql.functions import pandas_udf, PandasUDFType

            def score_partition(iterator):
                # generator over partition pandas DataFrames
                import joblib
                import pandas as pd
                scaler_local = joblib.load(scaler_path)
                iso_local = joblib.load(model_path)
                for pdf in iterator:
                    ids = pdf["customer_id"].values
                    X_part = pdf[available_features].fillna(0.0).values.astype(float)
                    Xs = scaler_local.transform(X_part)
                    preds_part = iso_local.predict(Xs)
                    scores_part = iso_local.score_samples(Xs)
                    out = pd.DataFrame({
                        "customer_id": ids,
                        "is_anomaly": (preds_part == -1).astype(int),
                        "anomaly_score": scores_part
                    })
                    yield out

            # Apply scoring per partition and union results
            full_score_rdd = anomaly_df.repartition(64).toPandasOnSpark().mapInPandas(lambda df_iter: score_partition(df_iter), schema="customer_id string, is_anomaly int, anomaly_score double")
            # If mapInPandas with schema not available in your runtime, fallback to collect-in-chunks approach below.
            try:
                scored_full_sdf = full_score_rdd.to_spark()
                scored_full_sdf = scored_full_sdf.withColumn("detection_timestamp", current_timestamp())
                scored_full_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(anomalies_delta_path)
                print(f"✅ Full-dataset anomaly results written to Delta path: {anomalies_delta_path}")
            except Exception as e_map:
                # Fallback: chunked collect approach (safer if mapInPandas not available)
                print(" ⚠️ mapInPandas -> to_spark path failed; falling back to chunked collect (may be slower).", e_map)
                chunk_size = 20000
                rows_total = anomaly_df.count()
                scored_parts = []
                offset = 0
                while offset < rows_total:
                    chunk_pdf = anomaly_df.limit(chunk_size).offset(offset).toPandas() if False else anomaly_df.rdd.zipWithIndex().filter(lambda x: offset <= x[1] < offset+chunk_size).map(lambda x: x[0]).toDF().toPandas()
                    # Simpler robust approach: collect by sampling ranges isn't straightforward in Spark; so for safety we will collect full if small
                    raise RuntimeError("Chunked full scoring fallback is environment-specific; please run this notebook on a cluster with mapInPandas support or increase driver memory to collect full dataset.")
        except Exception as e_full:
            print("⚠️ Could not score full dataset in distributed manner. Saving sample results only. Error:", e_full)
            # Save sample results instead
            results_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(anomalies_delta_path)
            print(f"✅ Sample anomaly results written to Delta path: {anomalies_delta_path} (full scoring skipped)")
    else:
        # Save sample results only
        results_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(anomalies_delta_path)
        print(f"✅ Sample anomaly results written to Delta path: {anomalies_delta_path}")

    # Show sample anomalies (join back with features)
    print("\n📊 Sample Anomalous Customers (top by anomaly_score):")
    saved_results = spark.read.format("delta").load(anomalies_delta_path)
    anomalous_customers = saved_results.filter(col("is_anomaly") == 1) \
        .join(anomaly_df, "customer_id", "left") \
        .orderBy(col("anomaly_score").asc()) \
        .select("customer_id", "anomaly_score", *available_features) \
        .limit(10)
    anomalous_customers.show(truncate=False)

    # Save model artifacts info to a small text file for bookkeeping
    try:
        info_txt = os.path.join(model_artifact_dir, "info.txt")
        with open(info_txt, "w") as f:
            f.write(f"trained_on_sample_rows={len(results_pd)}\n")
            f.write(f"contamination=0.05\n")
            f.write(f"artifact_saved_at={model_artifact_dir}\n")
        print("Saved model artifact info:", info_txt)
    except Exception:
        pass

except ImportError as imp_e:
    print("⚠️ scikit-learn not available in this environment. Skipping anomaly detection. Error:", imp_e)


# COMMAND ----------

## ML Models Summary

# COMMAND ----------

print("\n" + "="*80)
print("🎉 MACHINE LEARNING MODELS TRAINING COMPLETE!")
print("="*80 + "\n")

print("✅ Models Trained:")
print("  1. Churn Prediction")
print("     - Logistic Regression")
print("     - Gradient Boosted Trees")
print("  2. Customer Lifetime Value (CLV) Prediction")
print("     - Linear Regression")
print("  3. Demand Forecasting")
print("     - Prophet Time Series Model")
print("  4. Anomaly Detection")
print("     - Isolation Forest")
print("\n📊 Output Tables Created:")
print("  - revenue_forecast_gold: 30-day revenue predictions")
print("  - customer_anomalies_gold: Behavioral anomaly flags")

# COMMAND ----------

