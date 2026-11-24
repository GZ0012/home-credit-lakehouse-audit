"""Silver 09: debit cards, deposits and other accounts -> one row per case.

Three small depth-1 sources (train_debitcard_1, train_deposit_1, train_other_1).
Volkova processes each one on its own (DataProcessor2, commit d5e228c:
_process_debitcard1, _process_deposit1, _process_other1, then the aggregations
from features/features_2.csv); here the three results share one table.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_debitcard_deposit_other_features + reports/09_debitcard_deposit_other/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_debitcard_deposit_other_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_debitcard_deposit_other_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/09_debitcard_deposit_other"

WINDOW_DAYS = 365 * 5   # Volkova's window for date_col = 1 columns

# From Volkova's features_2.csv. For each source:
#   features: column -> aggregations
#   differences: Volkova's formulas (only other1 has them)
#   date_column: the date_col = 1 column used for the 5 year window
SOURCES = {
    "debitcard1": {
        "table": "train_debitcard_1",
        "features": {
            "last180dayaveragebalance_704A": ("min", "max", "sum"),
            "last180dayturnover_1134A": ("min", "max", "sum"),
            "last30dayturnover_651A": ("min", "max", "sum"),
            "openingdate_857D": ("min", "max", "mean", "count"),
        },
        "differences": {},
        "date_column": "openingdate_857D",
    },
    "deposit1": {
        "table": "train_deposit_1",
        "features": {
            "amount_416A": ("min", "max", "sum"),
            "contractenddate_991D": ("min", "max"),
            "openingdate_313D": ("min", "max", "count"),
        },
        "differences": {},
        "date_column": "openingdate_313D",
    },
    "other1": {
        "table": "train_other_1",
        "features": {
            "amtdebitincoming_4809443A": ("min", "max", "sum"),
            "amtdebitoutgoing_4809440A": ("min", "max", "sum"),
            "amtdepositbalance_4809441A": ("min", "max", "sum"),
            "amtdepositincoming_4809444A": ("min", "max", "sum"),
            "amtdepositoutgoing_4809442A": ("min", "max", "sum", "mean"),
        },
        # Volkova's _process_other1: incoming - outgoing
        "differences": {
            "amtdebitincoming_4809443A_2": ("amtdebitincoming_4809443A_1", "amtdebitoutgoing_4809440A_1"),
            "amtdepositincoming_4809444A_2": ("amtdepositincoming_4809444A_1", "amtdepositoutgoing_4809442A_1"),
        },
        "date_column": None,
    },
}


def clean(name):
    # Volkova: dates -> days from date_decision; the other columns here are amounts
    if name.endswith("D"):
        return F.datediff(F.col(name).cast("date"), F.col("date_decision"))
    return F.col(name).cast("double")


def agg_expr(column, op):
    # Volkova's Utils.agg_process in Spark
    value = F.col(column)
    ops = {
        "min": F.min(value),
        "max": F.max(value),
        "mean": F.avg(value),
        "count": F.count(value),
        "sum": F.coalesce(F.sum(value), F.lit(0)),   # polars: sum of only nulls = 0
    }
    return ops[op]


def aggregate(rows, features, columns, suffix):
    exprs = [
        agg_expr(c, op).alias(f"{c}_{op}{suffix}")
        for c in columns
        for op in features[c.rsplit("_", 1)[0]]
    ]
    exprs.append(F.count("case_id").alias(f"case_id_1_count_{suffix}"))
    return rows.groupBy("case_id").agg(*exprs)


cases = spark.table(CASE_TABLE)
dates = cases.select("case_id", "date_decision")

features = cases.select("case_id")
bronze_counts = {}
for name, source in SOURCES.items():
    bronze = spark.table(f"{BRONZE}.{source['table']}")
    bronze_counts[name] = bronze.count()

    # 1. clean + Volkova's formulas
    rows = bronze.select("case_id", *source["features"]).join(dates, "case_id", "inner")
    rows = rows.select("case_id", *[clean(c).alias(f"{c}_1") for c in source["features"]])
    rows = rows.withColumns({new: F.col(a) - F.col(b) for new, (a, b) in source["differences"].items()})

    # 2. aggregate all rows, and the last 5 years if the source has a date column
    columns = [c for c in rows.columns if c != "case_id"]
    agg = aggregate(rows, source["features"], columns, "")
    if source["date_column"]:
        recent = rows.where(F.col(f"{source['date_column']}_1") >= -WINDOW_DAYS)
        agg = agg.join(aggregate(recent, source["features"], columns, f"_{source['date_column']}_5y"),
                       "case_id", "left")

    agg = agg.select(
        "case_id",
        *[F.col(c).alias(f"{c}_{name}") for c in agg.columns if c != "case_id"],
        F.lit(True).alias(f"has_{name}"),
    )
    features = features.join(agg, "case_id", "left")

# 3. One row per case

flags = [f"has_{name}" for name in SOURCES]
count_columns = [c for c in features.columns if "case_id_" in c]
features = features.fillna(False, subset=flags).fillna(0, subset=count_columns)

# Volkova's final step: +/-inf -> null (NaN too)
floats = [f.name for f in features.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
features = features.withColumns({
    c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
})

# 4. Checks and report

base_count = cases.count()
out = features.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    *[F.count(F.when(F.col(f"has_{name}"), 1)).alias(f"cases_{name}") for name in SOURCES],
    *[F.sum(f"case_id_1_count__{name}").alias(f"rows_{name}") for name in SOURCES],
).first().asDict()
if out["cases"] != base_count or out["distinct_cases"] != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = [("cases", out["cases"])]
for name in SOURCES:
    coverage += [
        (f"{name}_rows_in_bronze", bronze_counts[name]),
        (f"{name}_rows_used", out[f"rows_{name}"]),
        (f"cases_with_{name}", out[f"cases_{name}"]),
    ]
coverage = pd.DataFrame(coverage, columns=["metric", "value"])
print(coverage.to_string(index=False))
print(f"{len(features.columns)} columns")

# 5. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
