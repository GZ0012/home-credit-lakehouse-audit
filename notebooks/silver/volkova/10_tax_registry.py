"""Silver 10: tax registries a, b and c -> one row per case.

Three depth-1 sources (train_tax_registry_a_1, _b_1, _c_1). Volkova processes
each one on its own (DataProcessor2, commit d5e228c: _process_taxregistrya1/b1/c1
and FeatureEngineer.transform); here the three results share one table.
The combined features across the three registries are built in gold 01
(her FinalDataProcessor).
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Bug in her code that we keep: for registry c the last-month filter is
pmtamount_36A_1 >= -30 (the amount, not the date), so it keeps almost every
row. Set TAX_C_FILTER_AS_AUTHOR = False to filter on processingdate_168D_1.

Output: silver_volkova.train_tax_registry_features + reports/10_tax_registry/*.csv
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
OUTPUT_TABLE = f"{SCHEMA}.train_tax_registry_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_tax_registry_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/10_tax_registry"

MONTH_DAYS = 30          # Volkova's "_1month" filter
WINDOW_DAYS = 365 * 5    # Volkova's window for date_col = 1 columns
TAX_C_FILTER_AS_AUTHOR = True

# From Volkova's features_2.csv and her code. For each registry:
#   features: column -> aggregations (the employer name is in her cols_drop)
#   sort_column: she sorts by this date before aggregating (for "last")
#   month_filter: column she filters with >= -30 for the "_1month" group
#   date_window: the date_col = 1 column (5 year window); only registry a has one
SOURCES = {
    "taxregistrya1": {
        "table": "train_tax_registry_a_1",
        "features": {"amount_4527230A": ("min", "max", "mean", "last"),
                     "recorddate_4527225D": ("max", "min", "mean")},
        "sort_column": "recorddate_4527225D_1",
        "month_filter": "recorddate_4527225D_1",
        "date_window": "recorddate_4527225D",
    },
    "taxregistryb1": {
        "table": "train_tax_registry_b_1",
        "features": {"amount_4917619A": ("min", "max", "mean", "last"),
                     "deductiondate_4917603D": ("max", "min", "last")},
        "sort_column": "deductiondate_4917603D_1",
        "month_filter": "deductiondate_4917603D_1",
        "date_window": None,
    },
    "taxregistryc1": {
        "table": "train_tax_registry_c_1",
        "features": {"pmtamount_36A": ("min", "max", "mean", "last"),
                     "processingdate_168D": ("max", "min", "last", "mean")},
        "sort_column": "processingdate_168D_1",
        # Volkova's bug: filter on the amount instead of the date
        "month_filter": "pmtamount_36A_1" if TAX_C_FILTER_AS_AUTHOR else "processingdate_168D_1",
        "date_window": None,
    },
}


def clean(name):
    # Volkova: dates -> days from date_decision; the other column is an amount
    if name.endswith("D"):
        return F.datediff(F.col(name).cast("date"), F.col("date_decision"))
    return F.col(name).cast("double")


def agg_expr(column, op, sort_column):
    # Volkova's Utils.agg_process in Spark.
    # "last" = last non-null value after her df.sort(date) (nulls first); ties by num_group1.
    value = F.col(column)
    sort_key = F.struct(F.col(sort_column).isNotNull(), F.col(sort_column), F.col("num_group1"))
    ops = {
        "min": F.min(value),
        "max": F.max(value),
        "mean": F.avg(value),
        "last": F.max_by(value, F.when(value.isNotNull(), sort_key)),
    }
    return ops[op]


def aggregate(rows, source, columns, suffix):
    exprs = [
        agg_expr(c, op, source["sort_column"]).alias(f"{c}_{op}{suffix}")
        for c in columns
        for op in source["features"][c.rsplit("_", 1)[0]]
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

    # 1. clean
    rows = bronze.select("case_id", "num_group1", *source["features"]).join(dates, "case_id", "inner")
    rows = rows.select("case_id", "num_group1", *[clean(c).alias(f"{c}_1") for c in source["features"]])
    columns = [f"{c}_1" for c in source["features"]]

    # 2. Volkova's groups, in her order: last month, all rows, then the 5 year date window
    month = rows.where(F.col(source["month_filter"]) >= -MONTH_DAYS)
    agg = aggregate(rows, source, columns, "")
    agg = agg.join(aggregate(month, source, columns, "_1month"), "case_id", "left")
    if source["date_window"]:
        recent = rows.where(F.col(f"{source['date_window']}_1") >= -WINDOW_DAYS)
        agg = agg.join(aggregate(recent, source, columns, f"_{source['date_window']}_5y"), "case_id", "left")

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
    *[F.sum(f"case_id_1_count__1month_{name}").alias(f"month_rows_{name}") for name in SOURCES],
).first().asDict()
if out["cases"] != base_count or out["distinct_cases"] != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = [("cases", out["cases"])]
for name in SOURCES:
    coverage += [
        (f"{name}_rows_in_bronze", bronze_counts[name]),
        (f"{name}_rows_used", out[f"rows_{name}"]),
        (f"{name}_rows_in_1month_group", out[f"month_rows_{name}"]),
        (f"cases_with_{name}", out[f"cases_{name}"]),
    ]
coverage = pd.DataFrame(coverage, columns=["metric", "value"])
print(coverage.to_string(index=False))
print(f"{len(features.columns)} columns")
if TAX_C_FILTER_AS_AUTHOR:
    print("Note: taxregistryc1 _1month filters on pmtamount_36A_1 >= -30 (Volkova's filter).")

# 5. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
