"""Silver yuuniee 01: labeled cases (train_base) -> base table for the other silver scripts.

yuuniee's feature code is not public, so the yuuniee pipeline uses Volkova's
DataProcessor5 (commit d5e228c) as a stand-in. DataProcessor5 has no fallback
(coalesce) columns, so this table is just train_base with date_decision as a date;
02-10 take date_decision from here to turn dates into days.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_5.py

Output: silver_yuuniee.train_case_sources + reports/01_case_sources/case_summary.csv
"""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_yuuniee"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/yuuniee"

OUTPUT_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_case_sources"
# table and volume need separate folders (Unity Catalog does not allow overlapping paths)
VOLUME = f"{SCHEMA}.files"
VOLUME_LOCATION = f"{STORAGE_ROOT}/files"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_yuuniee/files/reports/01_case_sources"

# 1. Base columns (Volkova's BaseDataLoader reads train_base as it is)

cases = spark.table(f"{BRONZE}.train_base").select(
    "case_id",
    F.col("date_decision").cast("date").alias("date_decision"),
    "WEEK_NUM",
    "MONTH",
    "target",
)

# 2. Checks

out = cases.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col("date_decision").isNull(), 1)).alias("null_dates"),
    F.avg("target").alias("target_rate"),
    F.min("WEEK_NUM").alias("first_week"),
    F.max("WEEK_NUM").alias("last_week"),
).first().asDict()
if out["cases"] != out["distinct_cases"]:
    raise ValueError(f"case_id is not unique: {out}")
if out["null_dates"]:
    raise ValueError(f"{out['null_dates']} date_decision values could not be read as dates")

summary = pd.DataFrame(list(out.items()), columns=["metric", "value"])
print(summary.to_string(index=False))

# 3. Save

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"CREATE EXTERNAL VOLUME IF NOT EXISTS {VOLUME} LOCATION '{VOLUME_LOCATION}'")

# CTAS fails if the table exists, so a rerun can't overwrite it.
# To rebuild: DROP TABLE and delete the table folder in storage first.
cases.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
summary.to_csv(f"{REPORT_DIR}/case_summary.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
