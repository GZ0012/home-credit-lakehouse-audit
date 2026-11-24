"""Silver 07: external bureau summary (train_static_cb_0) -> one row per case.

staticcb0 already has one row per case, so nothing is aggregated. The six
fallbacks and the one ratio are Volkova's _process_staticcb0 in DataProcessor2 (commit d5e228c):
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_static_cb_0_features + reports/07_static_cb_0/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, StringType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_static_cb_0_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_static_cb_0_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/07_static_cb_0"

SOURCE = "staticcb0"
MISSING_CATEGORY = "a55475b1"
INCOME = "mainoccupationinc_384A_1_person1"   # applicant income from person1

# staticcb0 columns in Volkova's features_2.csv (none are dropped)
FEATURES = (
    "assignmentdate_238D", "assignmentdate_4527235D", "assignmentdate_4955616D",
    "birthdate_574D", "contractssum_5085716L", "dateofbirth_337D",
    "dateofbirth_342D", "days120_123L", "days180_256L",
    "days30_165L", "days360_512L", "days90_310L",
    "description_5085714M", "education_1103M", "education_88M",
    "firstquarter_103L", "for3years_128L", "for3years_504L",
    "for3years_584L", "formonth_118L", "formonth_206L",
    "formonth_535L", "forquarter_1017L", "forquarter_462L",
    "forquarter_634L", "fortoday_1092L", "forweek_1077L",
    "forweek_528L", "forweek_601L", "foryear_618L",
    "foryear_818L", "foryear_850L", "fourthquarter_440L",
    "maritalst_385M", "maritalst_893M", "numberofqueries_373L",
    "pmtaverage_3A", "pmtaverage_4527227A", "pmtaverage_4955615A",
    "pmtcount_4527229L", "pmtcount_4955617L", "pmtcount_693L",
    "pmtscount_423L", "pmtssum_45A", "requesttype_4525192L",
    "responsedate_1012D", "responsedate_4527233D", "responsedate_4917613D",
    "riskassesment_302T", "riskassesment_940T", "secondquarter_766L",
    "thirdquarter_1082L",
)
# Volkova's pl.coalesce in _process_staticcb0: new column -> inputs, primary first
FALLBACKS = {
    "pmtaverage_3A_2": ("pmtaverage_4527227A_1", "pmtaverage_4955615A_1", "pmtaverage_3A_1"),
    "birthdate_574D_2": ("birthdate_574D_1", "dateofbirth_337D_1", "dateofbirth_342D_1"),
    "education_1103M_2": ("education_1103M_1", "education_88M_1"),
    "maritalst_385M_2": ("maritalst_385M_1", "maritalst_893M_1"),
    "pmtcount_4527229L_2": ("pmtcount_4527229L_1", "pmtcount_4955617L_1", "pmtcount_693L_1"),
    "assignmentdate_238D_2": ("assignmentdate_238D_1", "assignmentdate_4527235D_1",
                              "assignmentdate_4955616D_1"),
}
# Volkova's ratio: tax payments sum / applicant income
RATIOS = {
    "pmtssum_45A_2": ("pmtssum_45A_1", INCOME),
}


def clean(name, dtype):
    # Volkova's type handling: dates -> days from date_decision,
    # isbid/isdebit flags -> int, masked category -> null (before the coalesce)
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
    # polars gives +/-inf (NaN for 0/0) when dividing by zero, Spark gives null
    a, b = a.cast("double"), b.cast("double")
    return (
        F.when(a.isNull() | b.isNull(), F.lit(None).cast("double"))
        .when(b != 0, a / b)
        .when(F.isnan(a) | (a == 0), F.lit(math.nan))
        .when(a > 0, F.lit(math.inf))
        .otherwise(F.lit(-math.inf))
    )


cases = spark.table(CASE_TABLE)
static_cb = spark.table(f"{BRONZE}.train_static_cb_0")

# 1. Clean

context = cases.select("case_id", "date_decision", F.col("person1_mainoccupationinc_384A").alias(INCOME))
rows = static_cb.select("case_id", *FEATURES).join(context, "case_id", "inner")
types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select("case_id", INCOME, *[clean(c, types[c]).alias(f"{c}_1") for c in FEATURES])

# 2. Volkova's fallbacks and ratio

rows = rows.withColumns({
    **{name: F.coalesce(*[F.col(c) for c in inputs]) for name, inputs in FALLBACKS.items()},
    **{name: ratio(F.col(a), F.col(b)) for name, (a, b) in RATIOS.items()},
})

# 3. One row per case

source_rows = rows.drop(INCOME)
source_rows = source_rows.select(
    "case_id",
    *[F.col(c).alias(f"{c}_{SOURCE}") for c in source_rows.columns if c != "case_id"],
    F.lit(True).alias(f"has_{SOURCE}"),
)
features = cases.select("case_id").join(source_rows, "case_id", "left").fillna(False, subset=[f"has_{SOURCE}"])

# Volkova's final step: +/-inf -> null (NaN too)
floats = [f.name for f in features.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
features = features.withColumns({
    c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
})

# 4. Checks and reports

base_count = cases.count()
out = features.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col(f"has_{SOURCE}"), 1)).alias("with_source"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = pd.DataFrame([
    ("staticcb0_rows_in_bronze", static_cb.count()),
    ("cases", out.cases),
    ("cases_with_staticcb0", out.with_source),
], columns=["metric", "value"])

metrics = [F.count("*").alias("rows")]
for name, inputs in FALLBACKS.items():
    metrics += [
        F.count(F.when(F.col(inputs[0]).isNull(), 1)).alias(f"{name}_before"),
        F.count(F.when(F.col(name).isNull(), 1)).alias(f"{name}_after"),
    ]
counts = rows.agg(*metrics).first().asDict()
fallbacks = pd.DataFrame(
    [(name, " -> ".join(inputs), counts["rows"], counts[f"{name}_before"], counts[f"{name}_after"])
     for name, inputs in FALLBACKS.items()],
    columns=["derived_field", "source_order", "rows", "missing_before_fallback", "missing_after_fallback"],
)
print(coverage.to_string(index=False))
print(fallbacks.to_string(index=False))
print(f"{len(features.columns)} columns")

# 5. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
fallbacks.to_csv(f"{REPORT_DIR}/fallback_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
