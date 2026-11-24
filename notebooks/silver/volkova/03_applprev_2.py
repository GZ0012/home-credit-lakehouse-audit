"""Silver 03: contacts and cards of previous applications (train_applprev_2) -> one row per case.

Feature logic is Volkova's applprev2 processing in DataProcessor2 (commit d5e228c):
_process_applprev2 only renames columns, then FeatureEngineer.transform
aggregates all rows of a case. Aggregations come from features/features_2.csv.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_applprev_2_features + reports/03_applprev_2/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_applprev_2_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_applprev_2_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/03_applprev_2"

SOURCE = "applprev2"
MISSING_CATEGORY = "a55475b1"

# From Volkova's features_2.csv (all three are categorical, include = 1)
FEATURES = {
    "cacccardblochreas_147M": ("nunique", "nuniquetotal", "mode", "first"),
    "conts_type_509L": ("nunique", "nuniquetotal", "mode", "first"),
    "credacc_cards_status_52L": ("nunique", "nuniquetotal", "mode", "first"),
}


def ratio(a, b):
    # polars gives +/-inf (NaN for 0/0) when dividing by zero, Spark gives null
    a, b = a.cast("double"), b.cast("double")
    return (
        F.when(a.isNull() | b.isNull(), F.lit(None).cast("double"))
        .when(b != 0, a / b)
        .when(F.isnan(a) | (a == 0), F.lit(math.nan))
        .when(a > 0, F.lit(math.inf))
        .otherwise(F.lit(-math.inf))
    )


def agg_expr(column, op):
    # Volkova's Utils.agg_process in Spark
    value = F.col(column)
    nunique = F.size(F.collect_set(value)) + F.max(value.isNull().cast("int"))  # polars counts null as a value
    ops = {
        "nunique": nunique,
        "nuniquetotal": ratio(nunique, F.count(value)),
        "mode": F.mode(value, True),
        "first": F.min_by(value, F.when(value.isNotNull(), F.col("_seq"))),
    }
    return ops[op]


cases = spark.table(CASE_TABLE)
history = spark.table(f"{BRONZE}.train_applprev_2")

# 1. Clean: Volkova casts to string and sets the masked category to null
rows = history.select("case_id", "num_group1", "num_group2", *FEATURES) \
    .join(cases.select("case_id"), "case_id", "inner")
for name in FEATURES:
    value = F.col(name).cast("string")
    rows = rows.withColumn(f"{name}_1", F.when(value != MISSING_CATEGORY, value)).drop(name)

# Volkova does not sort applprev2, so "first" uses the file order.
# We assume the file is sorted by (num_group1, num_group2).
order = Window.partitionBy("case_id").orderBy("num_group1", "num_group2")
rows = rows.withColumn("_seq", F.row_number().over(order))

# 2. Aggregate all rows of a case (no approved/refused or date windows here)
exprs = [
    agg_expr(f"{name}_1", op).alias(f"{name}_1_{op}_{SOURCE}")
    for name, ops in FEATURES.items()
    for op in ops
]
exprs.append(F.count("case_id").alias(f"case_id_1_count__{SOURCE}"))
history_features = rows.groupBy("case_id").agg(*exprs).withColumn(f"has_{SOURCE}", F.lit(True))

# 3. One row per case
features = cases.select("case_id").join(history_features, "case_id", "left") \
    .fillna(False, subset=[f"has_{SOURCE}"]) \
    .fillna(0, subset=[f"case_id_1_count__{SOURCE}"])

# Volkova's final step: +/-inf -> null (NaN too)
for name in FEATURES:
    c = f"{name}_1_nuniquetotal_{SOURCE}"
    features = features.withColumn(c, F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)))

# 4. Checks and report
base_count = cases.count()
out = features.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col(f"has_{SOURCE}"), 1)).alias("with_rows"),
    F.sum(f"case_id_1_count__{SOURCE}").alias("rows_used"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = pd.DataFrame([
    ("rows_in_bronze", history.count()),
    ("rows_used", out.rows_used),
    ("cases", out.cases),
    ("cases_with_rows", out.with_rows),
], columns=["metric", "value"])
print(coverage.to_string(index=False))

# 5. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)
features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
