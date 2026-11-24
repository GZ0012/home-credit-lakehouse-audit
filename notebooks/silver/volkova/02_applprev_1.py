"""Silver 02: previous applications (train_applprev_1) -> one row per case.

Feature logic is Volkova's applprev1 processing in DataProcessor2 (commit d5e228c):
_process_applprev1 (row formulas), _handle_applprev1 (last application row) and
FeatureEngineer.transform (aggregations). Which aggregations each column gets
comes from her features/features_2.csv.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_applprev_1_features + reports/02_applprev_1/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_applprev_1_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_applprev_1_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/02_applprev_1"

SOURCE = "applprev1"
MISSING_CATEGORY = "a55475b1"
SORT_COLUMN = "creationdate_885D_1"   # Volkova sorts applprev1 by this before aggregating
WINDOW_DAYS = 365 * 5                 # Volkova: date window for date_col = 1 columns
WINDOW_SUFFIX = "_creationdate_885D_5y"

# case-level columns, with Volkova's names
INCOME_FALLBACK = "mainoccupationinc_384A_1_person1"
CURRENT_CREDIT = "credamount_770A_1_static0"

# From Volkova's features_2.csv: column -> (aggregations, include).
# include = 0 only removes the column from the last-application row.
FEATURES = {
    "actualdpd_943P": (("min", "max"), 1),
    "annuity_853A": (("min", "max", "rangefirstlast", "mean"), 1),
    "approvaldate_319D": (("min", "max", "range", "count"), 1),
    "byoccupationinc_3656910L": (("min", "max", "mean", "rangefirstlast"), 0),
    "cancelreason_3545846M": (("nunique", "nuniquetotal", "mode"), 1),
    "childnum_21L": (("first", "max"), 1),
    "creationdate_885D": (("min", "max", "mean", "range"), 1),
    "credacc_actualbalance_314A": (("min", "max"), 1),
    "credacc_credlmt_575A": (("min", "max"), 1),
    "credacc_maxhisbal_375A": (("min", "max"), 1),
    "credacc_minhisbal_90A": (("min", "max", "mean"), 1),
    "credacc_status_367L": (("nunique", "nuniquetotal", "mode"), 1),
    "credacc_transactions_402L": (("min", "max"), 1),
    "credamount_590A": (("min", "max", "mean"), 1),
    "credtype_587L": (("nunique", "nuniquetotal", "mode"), 1),
    "currdebt_94A": (("min", "max"), 1),
    "dateactivated_425D": (("min", "max"), 1),
    "district_544M": (("nunique", "nuniquetotal"), 1),
    "downpmt_134A": (("min", "max", "mean"), 1),
    "dtlastpmt_581D": (("min", "max", "mean"), 1),
    "dtlastpmtallstes_3545839D": (("min", "max"), 1),
    "education_1138M": (("mode",), 1),
    "employedfrom_700D": (("min", "max", "mean"), 1),
    "familystate_726L": (("nunique", "nuniquetotal", "mode"), 1),
    "firstnonzeroinstldate_307D": (("min", "max", "mean"), 1),
    "inittransactioncode_279L": (("nunique", "nuniquetotal", "mode"), 1),
    "isbidproduct_390L": (("min", "max"), 1),
    "isdebitcard_527L": (("min", "max"), 1),
    "mainoccupationinc_437A": (("min", "max"), 1),
    "maxdpdtolerance_577P": (("min", "max"), 1),
    "outstandingdebt_522A": (("max", "min", "mean"), 1),
    "pmtnum_8L": (("min", "max"), 1),
    "postype_4733339M": (("mode",), 1),
    "profession_152M": (("nunique", "nuniquetotal"), 1),
    "rejectreason_755M": (("nunique", "nuniquetotal", "mode"), 1),
    "rejectreasonclient_4145042M": (("nunique", "nuniquetotal", "mode"), 1),
    "revolvingaccount_394A": (("max",), 0),
    "status_219L": (("nunique", "nuniquetotal", "mode"), 1),
    "tenor_203L": (("min", "max"), 1),
}


def check_one_row_per_case(df, name):
    counts = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("case_id").alias("cases"),
        F.count(F.when(F.col("case_id").isNull(), 1)).alias("null_ids"),
    ).first()
    if counts.null_ids or counts.rows != counts.cases:
        raise ValueError(f"{name}: expected one row per case_id, got {counts}")
    return counts.rows


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
    # annuity_853A_2 -> annuity_853A (derived columns use the aggregations of their base column)
    return column.rsplit("_", 1)[0]


def agg_expr(column, op):
    # Volkova's Utils.agg_process in Spark. first/last follow the row order in _seq.
    value = F.col(column)
    position = F.when(value.isNotNull(), F.col("_seq"))
    first = F.min_by(value, position)
    last = F.max_by(value, position)
    nunique = F.size(F.collect_set(value)) + F.max(value.isNull().cast("int"))  # polars counts null as a value
    ops = {
        "min": F.min(value),
        "max": F.max(value),
        "mean": F.avg(value),
        "count": F.count(value),
        "range": F.max(value) - F.min(value),
        "first": first,
        "rangefirstlast": first - last,
        "nunique": nunique,
        "nuniquetotal": ratio(nunique, F.count(value)),
        "mode": F.mode(value, True),
    }
    return ops[op]


def aggregate(rows, columns, suffix):
    # Volkova's _process_aggregations_all: aggregations per column + row count
    exprs = [
        agg_expr(c, op).alias(f"{c}_{op}{suffix}")
        for c in columns
        for op in FEATURES[base_feature(c)][0]
    ]
    exprs.append(F.count("case_id").alias(f"case_id_1_count_{suffix}"))
    return rows.groupBy("case_id").agg(*exprs)


def drop_inf_nan(df):
    # Volkova's FinalDataProcessor sets +/-inf to null; we also null NaN
    # (LightGBM treats both as missing anyway)
    floats = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
    return df.withColumns({
        c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
    })


# 1. Load rows

cases = spark.table(CASE_TABLE)
base_count = check_one_row_per_case(cases, CASE_TABLE)
history = spark.table(f"{BRONZE}.train_applprev_1")
bronze_rows = history.count()

# Volkova reads each parquet chunk file on its own and keeps the first result
# per case, so a case split across the two chunk files only uses the first one.
history = history.withColumn("_chunk", F.coalesce(
    F.regexp_extract("_source_file", r"_(\d+)\.parquet$", 1).try_cast("int"), F.lit(0)))
first_chunk = history.groupBy("case_id").agg(F.min("_chunk").alias("_first_chunk"))
history = history.join(first_chunk, "case_id").where(F.col("_chunk") == F.col("_first_chunk"))

context = cases.select(
    "case_id", "date_decision",
    F.col("person1_mainoccupationinc_384A").alias(INCOME_FALLBACK),
    F.col("static0_credamount_770A").alias(CURRENT_CREDIT),
)
rows = history.select("case_id", "num_group1", *FEATURES).join(context, "case_id", "inner")
types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select(
    "case_id", "num_group1", "date_decision", INCOME_FALLBACK, CURRENT_CREDIT,
    *[clean(c, types[c]).alias(f"{c}_1") for c in FEATURES],
)

# 2. Row formulas, from Volkova's _process_applprev1.
# She also coalesces dtlastpmt_581D with dtlastpmtallstes_3545839D into
# dtlastpmt_581D_2, but the next line overwrites it, so it is left out here.

col = F.col
income, credit = col("mainoccupationinc_437A_1"), col("credamount_590A_1")
annuity, tenor = col("annuity_853A_1"), col("tenor_203L_1")
rows = rows.withColumns({
    "dtlastpmt_581D_2": col("dtlastpmt_581D_1") - col("firstnonzeroinstldate_307D_1"),
    "dtlastpmt_581D_3": col("dtlastpmt_581D_1") - col("creationdate_885D_1"),
    "mainoccupationinc_437A_2": F.coalesce(income, col(INCOME_FALLBACK)),
    "credacc_credlmt_575A_2": ratio(col("credacc_credlmt_575A_1"), col("credacc_actualbalance_314A_1")),
    # she divides by the unfilled income (_1), not the filled one (_2)
    "currdebt_94A_2": ratio(col("currdebt_94A_1"), income),
    "annuity_853A_2": ratio(annuity, income),
    "outstandingdebt_522A_2": ratio(col("outstandingdebt_522A_1"), income),
    "credamount_590A_2": ratio(credit, income),
    "annuity_853A_3": ratio(annuity, credit),
    "downpmt_134A_2": ratio(col("downpmt_134A_1"), credit),
    "annuity_853A_4": ratio(ratio(annuity * tenor, credit) - 1, tenor),
    "credamount_590A_3": ratio(credit, col(CURRENT_CREDIT)),
})

# Volkova: df.sort("creationdate_885D_1") (nulls first). Ties: we use num_group1.
order = Window.partitionBy("case_id").orderBy(F.col(SORT_COLUMN).asc_nulls_first(), "num_group1")
rows = rows.withColumn("_seq", F.row_number().over(order))

# 3. Aggregations, the groups are from Volkova's FeatureEngineer.transform

types = {f.name: f.dataType for f in rows.schema.fields}
columns = [c for c in rows.columns if base_feature(c) in FEATURES]
numeric = [c for c in columns if isinstance(types[c], NumericType)]
not_included = {name for name, (_, include) in FEATURES.items() if include == 0}

# last application (num_group1 = 0) kept as is, without include = 0 columns
last = rows.where(F.col("num_group1") == 0) \
    .select("case_id", *[c for c in columns if base_feature(c) not in not_included])
check_one_row_per_case(last, "applprev1 num_group1 = 0")

history_features = aggregate(rows, columns, "")
parts = [
    # approved = approvaldate present, refused = no approvaldate (numeric columns only)
    aggregate(rows.where(F.col("approvaldate_319D_1").isNotNull()), numeric, "_approved"),
    aggregate(rows.where(F.col("approvaldate_319D_1").isNull()), numeric, "_refused"),
    # last 5 years by creationdate
    aggregate(rows.where(F.col(SORT_COLUMN) >= -WINDOW_DAYS), columns, WINDOW_SUFFIX),
    last,
]
for part in parts:
    history_features = history_features.join(part, "case_id", "left")

# Volkova adds the source name to every column
history_features = history_features.select(
    "case_id",
    *[F.col(c).alias(f"{c}_{SOURCE}") for c in history_features.columns if c != "case_id"],
    F.lit(True).alias(f"has_{SOURCE}"),
)

# 4. One row per case: cases without history get null features and count 0

features = cases.select("case_id").join(history_features, "case_id", "left")
count_columns = [c for c in features.columns if c.startswith("case_id_")]
features = features.fillna(False, subset=[f"has_{SOURCE}"]).fillna(0, subset=count_columns)
features = drop_inf_nan(features)

# 5. Reports

row_counts = rows.agg(
    F.count("*").alias("rows"),
    F.countDistinct("case_id").alias("cases"),
    F.count(F.when(F.col("mainoccupationinc_437A_1").isNull(), 1)).alias("income_missing"),
    F.count(F.when(F.col("mainoccupationinc_437A_2").isNull(), 1)).alias("income_missing_after"),
).first()
out = features.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col(f"has_{SOURCE}"), 1)).alias("with_history"),
    F.count(F.when(F.col(f"employedfrom_700D_1_{SOURCE}").isNull(), 1)).alias("employment_missing"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = pd.DataFrame([
    ("history_rows_in_bronze", bronze_rows),
    ("history_rows_used", row_counts.rows),
    ("cases", out.cases),
    ("cases_with_history", out.with_history),
], columns=["metric", "value"])
fallbacks = pd.DataFrame([
    ("historical_income", "mainoccupationinc_437A_2", "history row", row_counts.rows,
     row_counts.income_missing, row_counts.income_missing_after),
    # employed_length_1_final0 is built in gold 01; this is how often it needs the fallback
    ("employment_start", "employed_length_1_final0", "case", out.cases, out.employment_missing, None),
], columns=["derived_field", "author_column", "grain", "rows", "missing_before_fallback",
            "missing_after_fallback"])
print(coverage.to_string(index=False))
print(fallbacks.to_string(index=False))
print(f"{len(features.columns)} columns")

# 6. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
fallbacks.to_csv(f"{REPORT_DIR}/fallback_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
