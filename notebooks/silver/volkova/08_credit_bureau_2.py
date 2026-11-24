"""Silver 08: credit bureau payment history (train_credit_bureau_a_2 + b_2) -> one row per case.

Feature logic is Volkova's creditbureau2 processing in DataProcessor2 (commit d5e228c):
_process_creditbureau2 (row formulas), _handle_contracts_cb (active / closed
views) and FeatureEngineer.transform (aggregations, with 1 year and 1 month
windows for creditbureau2). Aggregations come from features/features_2.csv.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Monthly payment records (num_group2) per bureau contract (num_group1). Like
silver 05, each row has an active and a closed contract side by side, and we
keep Volkova's "_closed = _active" bug (CLOSED_FILTER_AS_AUTHOR).

This is the biggest table, so "last" uses max_by on a struct sort key
instead of a window sort.

Output: silver_volkova.train_credit_bureau_2_features + reports/08_credit_bureau_2/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_credit_bureau_2_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_credit_bureau_2_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/08_credit_bureau_2"

SOURCE = "creditbureau2"
A2_TABLE, B2_TABLE = "credit_bureau_a_2", "credit_bureau_b_2"
MISSING_CATEGORY = "a55475b1"
SORT_COLUMN = "pmts_month_all_1"   # Volkova sorts the views by this
# Volkova's windows for creditbureau2: 30 * 12 days and 30 days
WINDOWS = {"_pmts_month_all_1y": 30 * 12, "_pmts_month_all_1m": 30}
CLOSED_FILTER_AS_AUTHOR = True

# From Volkova's features_2.csv: column -> (aggregations, view).
# view = active / closed / common for raw columns, "all" for merged ones.
CONFIG = {
    "collater_typofvalofguarant_298M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "collater_typofvalofguarant_407M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "collater_typofvalofguarant_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "collater_valueofguarantee_1124L": (('max', 'mean', 'min'), "active"),
    "collater_valueofguarantee_876L": (('max', 'mean', 'min'), "closed"),
    "collater_valueofguarantee_all": (('max', 'mean', 'min'), "all"),
    "collaterals_typeofguarante_359M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "collaterals_typeofguarante_669M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "collaterals_typeofguarante_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "pmts_dpd_1073P": (('max', 'mean', 'sharenonzero', 'min'), "active"),
    "pmts_dpd_303P": (('max', 'mean', 'sharenonzero', 'min'), "closed"),
    "pmts_dpd_all": (('max', 'mean', 'sharenonzero', 'min'), "all"),
    "pmts_month_158T": (('min', 'max', 'mean'), "closed"),
    "pmts_month_706T": (('min', 'max', 'mean'), "active"),
    "pmts_month_all": (('min', 'max', 'mean'), "all"),
    "pmts_overdue_1140A": (('max', 'mean', 'sum', 'std', 'min'), "active"),
    "pmts_overdue_1152A": (('max', 'mean', 'sum', 'std', 'min'), "closed"),
    "pmts_overdue_all": (('max', 'mean', 'sum', 'std', 'min'), "all"),
    "pmts_year_1139T": ((), "active"),
    "pmts_year_507T": ((), "closed"),
    "pmts_year_all": ((), "all"),
    "subjectroles_name_541M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "subjectroles_name_838M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "subjectroles_name_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "pmts_date_1107D": ((), "active"),
    "pmts_dpdvalue_108P": ((), "active"),
    "pmts_pmtsoverdue_635A": ((), "active"),
}

# Volkova renames b2 columns to their a2 name (new_name in features_2.csv)
B2_RENAME = {
    "pmts_dpdvalue_108P": "pmts_overdue_1140A",
    "pmts_pmtsoverdue_635A": "pmts_dpd_1073P",
}
A2_COLUMNS = (
    "collater_typofvalofguarant_298M", "collater_typofvalofguarant_407M",
    "collater_valueofguarantee_1124L", "collater_valueofguarantee_876L",
    "collaterals_typeofguarante_359M", "collaterals_typeofguarante_669M",
    "pmts_dpd_1073P", "pmts_dpd_303P",
    "pmts_month_158T", "pmts_month_706T",
    "pmts_overdue_1140A", "pmts_overdue_1152A",
    "pmts_year_1139T", "pmts_year_507T",
    "subjectroles_name_541M", "subjectroles_name_838M",
)
B2_COLUMNS = ("pmts_date_1107D", "pmts_dpdvalue_108P", "pmts_pmtsoverdue_635A",)

# Volkova's cols_drop in _process_creditbureau2 (year/month parts, used in the formulas first)
DATE_PARTS = ["pmts_year_507T", "pmts_year_1139T", "pmts_month_158T", "pmts_month_706T"]

# Formulas from Volkova's _process_creditbureau2
# new column -> (year column, month column): first day of that month as day offset
MONTH_DATES = {
    "pmts_month_158T_1": ("pmts_year_507T", "pmts_month_158T"),
    "pmts_month_706T_1": ("pmts_year_1139T", "pmts_month_706T"),
}
# new column -> (DPD column, days): 1 if DPD > days else 0
DPD_FLAGS = {
    "pmts_dpd_1073P_2": ("pmts_dpd_1073P_1", 7),
    "pmts_dpd_303P_2": ("pmts_dpd_303P_1", 7),
    "pmts_dpd_1073P_3": ("pmts_dpd_1073P_1", 15),
    "pmts_dpd_303P_3": ("pmts_dpd_303P_1", 15),
}

HELPERS = ("case_id", "num_group1", "num_group2", "_source", "_active")


def clean(name, dtype):
    # Volkova's type handling: dates -> days from date_decision,
    # isbid/isdebit flags -> int, masked category -> null
    value = F.col(name)
    if name.endswith("D"):
        return F.datediff(value.cast("date"), F.col("date_decision"))
    if name[-1] in "AP":
        return value.cast("double")
    if "isbid" in name or "isdebit" in name:
        return value.cast("boolean").cast("tinyint")
    if name.endswith("M") or isinstance(dtype, (StringType, BooleanType)):
        value = value.cast("string")
        return F.when(value != MISSING_CATEGORY, value)
    return value


def ratio(a, b):
    # polars gives +/-inf (NaN for 0/0) when dividing by zero, Spark gives null.
    # Copy polars so the aggregations match; inf and NaN are set to null at the end.
    a, b = a.cast("double"), b.cast("double")
    return (
        F.when(a.isNull() | b.isNull(), F.lit(None).cast("double"))
        .when(b != 0, a / b)
        .when(F.isnan(a) | (a == 0), F.lit(math.nan))
        .when(a > 0, F.lit(math.inf))
        .otherwise(F.lit(-math.inf))
    )


def base_feature(column):
    # pmts_dpd_1073P_2 -> pmts_dpd_1073P (derived columns use the config of their base column)
    return column.rsplit("_", 1)[0]


def chunk_number(column):
    # train_credit_bureau_a_2_3.parquet -> 3 (files without a number -> 0)
    return F.coalesce(F.regexp_extract(F.col(column), r"_(\d+)\.parquet$", 1).try_cast("int"), F.lit(0))


def month_offset(year, month):
    # Volkova: pl.date(year, month, 1) - date_decision in days
    year, month = F.col(year).cast("double").cast("int"), F.col(month).cast("double").cast("int")
    valid = year.isNotNull() & month.between(1, 12)
    return F.when(valid, F.datediff(F.make_date(year, month, F.lit(1)), F.col("date_decision")))


def sort_key():
    # Volkova: df.sort("pmts_month_all_1") (nulls first). Ties: active view first,
    # then num_group1, num_group2. A struct compares its fields left to right.
    return F.struct(
        F.col(SORT_COLUMN).isNotNull(), F.col(SORT_COLUMN),
        ~F.col("_active"), F.col("num_group1"), F.col("num_group2"),
    )


def agg_expr(column, op):
    # Volkova's Utils.agg_process in Spark
    value = F.col(column)
    nunique = F.size(F.collect_set(value)) + F.max(value.isNull().cast("int"))  # polars counts null as a value
    ops = {
        "min": F.min(value),
        "max": F.max(value),
        "mean": F.avg(value),
        "sum": F.coalesce(F.sum(value), F.lit(0)),   # polars: sum of only nulls = 0
        "std": F.stddev_samp(value),
        "last": F.max_by(value, F.when(value.isNotNull(), sort_key())),
        "sharenonzero": ratio(F.count(F.when(value != 0, 1)), F.count(value)),
        "nunique": nunique,
        "nuniquetotal": ratio(nunique, F.count(value)),
        "mode": F.mode(value, True),
    }
    return ops[op]


def aggregate(rows, columns, suffix):
    # Volkova's _process_aggregations_all; for credit bureau she also adds the source file
    exprs = [
        agg_expr(c, op).alias(f"{c}_{op}{suffix}")
        for c in columns
        for op in CONFIG[base_feature(c)][0]
    ]
    exprs.append(F.count("case_id").alias(f"case_id_1_count_{suffix}"))
    exprs.append(F.max("_source").alias(f"source_1_{suffix}"))
    return rows.groupBy("case_id").agg(*exprs)


cases = spark.table(CASE_TABLE)
dates = cases.select("case_id", "date_decision")

# 1. Load a2 and b2 (b2 columns renamed to the a2 names)

a2 = spark.table(f"{BRONZE}.train_{A2_TABLE}")
b2 = spark.table(f"{BRONZE}.train_{B2_TABLE}")
a2_types = {f.name: f.dataType for f in a2.schema.fields}

a2_rows = a2.select(
    "case_id", "num_group1", "num_group2",
    F.lit(A2_TABLE).alias("_source"), chunk_number("_source_file").alias("_chunk"), *A2_COLUMNS,
).join(dates, "case_id", "inner")

b2_columns = []
for raw in B2_COLUMNS:
    name = B2_RENAME.get(raw, raw)
    # cast to the a2 type so the union works
    value = F.col(raw).cast(a2_types[name]) if name in a2_types else F.col(raw)
    b2_columns.append(value.alias(name))
b2_rows = b2.select(
    "case_id", "num_group1", "num_group2", F.lit(B2_TABLE).alias("_source"), F.lit(0).alias("_chunk"),
    *b2_columns,
).join(dates, "case_id", "inner")

# Volkova aggregates each file on its own and keeps the first result per case:
# a case in a2 never uses b2, and a case split over a2 chunk files only uses the first chunk.
first_chunk = a2_rows.groupBy("case_id").agg(F.min("_chunk").alias("_first_chunk"))
a2_used = a2_rows.join(first_chunk, "case_id").where(F.col("_chunk") == F.col("_first_chunk")).drop("_first_chunk")
b2_used = b2_rows.join(a2_rows.select("case_id").distinct(), "case_id", "left_anti")
rows = a2_used.unionByName(b2_used, allowMissingColumns=True).drop("_chunk")

# 2. Clean (year/month parts keep their raw name, Volkova does not rename them)

types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select(
    "case_id", "num_group1", "num_group2", "_source", "date_decision",
    *[clean(c, types[c]).alias(c if c in DATE_PARTS else f"{c}_1") for c in rows.columns if c in CONFIG],
)

# 3. Row formulas from Volkova's _process_creditbureau2

rows = rows.withColumns({
    **{name: month_offset(y, m) for name, (y, m) in MONTH_DATES.items()},
    # pl.when(dpd > n).then(1).otherwise(0): a missing DPD gives 0
    **{name: F.when(F.col(c) > days, 1).otherwise(0) for name, (c, days) in DPD_FLAGS.items()},
}).drop(*DATE_PARTS, "date_decision")

# 4. Volkova's _handle_contracts_cb: active view + closed view, pairs renamed to <name>_all_<n>

columns = [c for c in rows.columns if base_feature(c) in CONFIG]
views = {"active": [], "closed": [], "common": []}
for c in columns:
    for view in views:
        if view in CONFIG[base_feature(c)][1]:
            views[view].append(c)

groups = {}
for c in views["active"] + views["closed"]:
    parts = c.split("_")
    groups.setdefault("_".join(parts[:-2]) + "_all_" + parts[-1], []).append(c)
rename = {c: merged for merged, names in groups.items() if len(names) >= 2 for c in names}

# if the two columns of a pair have different types, use one type for both
types = {f.name: f.dataType for f in rows.schema.fields}
merged_type = {}
for merged, names in groups.items():
    kinds = {types[c] for c in names}
    if len(names) >= 2 and len(kinds) > 1:
        merged_type[merged] = StringType() if any(isinstance(k, StringType) for k in kinds) else DoubleType()


def make_view(names, active):
    renamed = []
    for c in names:
        merged = rename.get(c, c)
        value = F.col(c).cast(merged_type[merged]) if merged in merged_type else F.col(c)
        renamed.append(value.alias(merged))
    return rows.select("case_id", "num_group1", "num_group2", "_source", *views["common"], *renamed,
                       F.lit(active).alias("_active"))


long = make_view(views["active"], True).unionByName(make_view(views["closed"], False), allowMissingColumns=True)

# 5. Aggregations, the groups are from Volkova's FeatureEngineer.transform

types = {f.name: f.dataType for f in long.schema.fields}
columns = [c for c in long.columns if c not in HELPERS]
numeric = [c for c in columns if isinstance(types[c], NumericType)]

active = long.where(F.col("_active"))
# Volkova's bug: the closed group also uses filter(active)
closed = long.where(F.col("_active") if CLOSED_FILTER_AS_AUTHOR else ~F.col("_active"))

active_agg = aggregate(active, numeric, "_active")
bureau = aggregate(long, columns, "")
parts = [active_agg, aggregate(closed, numeric, "_closed")]
parts += [aggregate(long.where(F.col(SORT_COLUMN) >= -days), columns, suffix) for suffix, days in WINDOWS.items()]
for part in parts:
    bureau = bureau.join(part, "case_id", "left")

# Volkova: every numeric _active aggregate divided by its _closed twin
pairs = [(c, c[: -len("active")] + "closed") for c in active_agg.columns
         if c.endswith("_active") and not c.startswith("source_")]
bureau = bureau.withColumns({f"{a}_div_{b}": ratio(F.col(a), F.col(b)) for a, b in pairs})

bureau = bureau.select(
    "case_id",
    *[F.col(c).alias(f"{c}_{SOURCE}") for c in bureau.columns if c != "case_id"],
    F.lit(True).alias(f"has_{SOURCE}"),
)

# 6. One row per case

features = cases.select("case_id").join(bureau, "case_id", "left")
count_columns = [c for c in features.columns if "case_id_" in c]
features = features.fillna(False, subset=[f"has_{SOURCE}"]).fillna(0, subset=count_columns)

# Volkova's final step: +/-inf -> null (NaN too)
floats = [f.name for f in features.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
features = features.withColumns({
    c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
})

# 7. Save first (this table is slow to build), then check the saved table.
# CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild.

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

written = spark.table(OUTPUT_TABLE)
source_column = f"source_1__{SOURCE}"
base_count = cases.count()
out = written.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col(source_column) == A2_TABLE, 1)).alias("cases_a2"),
    F.count(F.when(F.col(source_column) == B2_TABLE, 1)).alias("cases_b2"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

# 8. Report

coverage = pd.DataFrame([
    ("a2_rows_in_bronze", a2.count()),
    ("a2_rows_used", a2_used.count()),
    ("b2_rows_in_bronze", b2.count()),
    ("b2_rows_used", b2_used.count()),
    ("cases", out.cases),
    ("cases_from_a2", out.cases_a2),
    ("cases_from_b2_only", out.cases_b2),
    ("cases_without_payments", out.cases - out.cases_a2 - out.cases_b2),
], columns=["metric", "value"])
print(coverage.to_string(index=False))
print(f"{len(written.columns)} columns")
if CLOSED_FILTER_AS_AUTHOR:
    print("Note: _closed = _active (Volkova's filter), so the _div_ columns are 1 or null.")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
