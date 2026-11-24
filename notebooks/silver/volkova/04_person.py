"""Silver 04: applicant and related persons (train_person_1, train_person_2) -> one row per case.

Feature logic is Volkova's person1/person2 processing in DataProcessor2 (commit d5e228c):
_process_person1, _handle_person1 and _process_person2, then the aggregations
from features/features_2.csv.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

person1: num_group1 = 0 is the applicant (kept as one row), other rows are
related persons (aggregated). person2: addresses/employers per person (aggregated).

Output: silver_volkova.train_person_features + reports/04_person/*.csv
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
OUTPUT_TABLE = f"{SCHEMA}.train_person_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_person_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/04_person"

MISSING_CATEGORY = "a55475b1"

# From Volkova's features_2.csv: column -> (aggregations for related persons, include).
# include = 0 only removes the column from the applicant row.
PERSON1_FEATURES = {
    "birth_259D": (("min", "max"), 1),
    "birthdate_87D": (("min", "max"), 1),
    "childnum_185L": (("min", "max"), 0),
    "contaddr_district_15M": ((), 1),
    "contaddr_matchlist_1032L": ((), 1),
    "contaddr_smempladdr_334L": ((), 1),
    "contaddr_zipcode_807M": ((), 0),
    "education_927M": (("mode", "nuniquetotal"), 1),
    "empl_employedfrom_271D": ((), 1),
    "empl_employedtotal_800L": ((), 1),
    "empl_industry_691L": ((), 1),
    "empladdr_district_926M": ((), 1),
    "empladdr_zipcode_114M": ((), 1),
    "familystate_447L": ((), 1),
    "gender_992L": (("mode", "nuniquetotal"), 1),
    "housetype_905L": ((), 1),
    "housingtype_772L": (("mode", "nuniquetotal"), 0),
    "incometype_1044T": ((), 1),
    "isreference_387L": ((), 0),
    "language1_981M": ((), 1),
    "mainoccupationinc_384A": ((), 1),
    "maritalst_703L": (("mode",), 0),
    "personindex_1023L": ((), 0),
    "persontype_1072L": (("mode",), 1),
    "persontype_792L": (("mode",), 0),
    "registaddr_district_1083M": ((), 1),
    "registaddr_zipcode_184M": ((), 1),
    "relationshiptoclient_415T": (("mode",), 0),
    "relationshiptoclient_642T": (("mode",), 0),
    "remitter_829L": ((), 0),
    "role_1084L": ((), 1),
    "role_993L": ((), 1),
    "safeguarantyflag_411L": ((), 1),
    "sex_738L": ((), 1),
    "type_25L": (("mode",), 1),
}
# Volkova's cols_drop in _process_person1
PERSON1_DROPPED = [
    "contaddr_district_15M", "registaddr_district_1083M", "empladdr_district_926M",
    "contaddr_zipcode_807M", "registaddr_zipcode_184M", "empladdr_zipcode_114M",
    "personindex_1023L", "isreference_387L",
    "birthdate_87D", "sex_738L", "relationshiptoclient_642T",
]
# Volkova's coalesce in _process_person1: column <- fallback (on every row)
PERSON1_FALLBACKS = {
    "birth_259D": "birthdate_87D",
    "gender_992L": "sex_738L",
    "relationshiptoclient_415T": "relationshiptoclient_642T",
}

# person2 after Volkova's cols_drop in _process_person2: only these three are left
PERSON2_FEATURES = {
    "addres_zip_823M": ("nunique", "nuniquetotal"),
    "conts_role_79M": ("nunique", "nuniquetotal"),
    "empls_economicalst_849M": ("nunique", "nuniquetotal"),
}


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
        "min": F.min(value),
        "max": F.max(value),
        "mode": F.mode(value, True),
        "nunique": nunique,
        "nuniquetotal": ratio(nunique, F.count(value)),
    }
    return ops[op]


def aggregate(rows, features, source):
    exprs = [
        agg_expr(f"{name}_1", op).alias(f"{name}_1_{op}_{source}")
        for name, ops in features.items()
        for op in ops
    ]
    exprs.append(F.count("case_id").alias(f"case_id_1_count__{source}"))
    return rows.groupBy("case_id").agg(*exprs)


cases = spark.table(CASE_TABLE)
dates = cases.select("case_id", "date_decision")

# 1. person1: clean, then Volkova's coalesce and cols_drop

person = spark.table(f"{BRONZE}.train_person_1")
rows = person.select("case_id", "num_group1", *PERSON1_FEATURES).join(dates, "case_id", "inner")
types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select("case_id", "num_group1", *[clean(c, types[c]).alias(f"{c}_1") for c in PERSON1_FEATURES])

# keep a copy before the coalesce for the fallback report
before_fallback = rows

rows = rows.withColumns({
    f"{name}_1": F.coalesce(F.col(f"{name}_1"), F.col(f"{fallback}_1"))
    for name, fallback in PERSON1_FALLBACKS.items()
}).drop(*[f"{c}_1" for c in PERSON1_DROPPED])
kept = [c for c in PERSON1_FEATURES if c not in PERSON1_DROPPED]

# Volkova's _handle_person1: applicant row as is (without include = 0 columns) ...
applicant = rows.where(F.col("num_group1") == 0).select(
    "case_id",
    *[F.col(f"{c}_1").alias(f"{c}_1_person1") for c in kept if PERSON1_FEATURES[c][1] == 1],
    F.lit(True).alias("has_person1_applicant"),
)
# ... and the other persons aggregated
related = aggregate(
    rows.where(F.col("num_group1") != 0),
    {c: PERSON1_FEATURES[c][0] for c in kept if PERSON1_FEATURES[c][0]},
    "person1",
).withColumn("has_person1_related", F.lit(True))

# 2. person2: clean and aggregate all rows of a case

person2 = spark.table(f"{BRONZE}.train_person_2")
rows2 = person2.select("case_id", *PERSON2_FEATURES).join(dates, "case_id", "inner")
types2 = {f.name: f.dataType for f in rows2.schema.fields}
rows2 = rows2.select("case_id", *[clean(c, types2[c]).alias(f"{c}_1") for c in PERSON2_FEATURES])
person2_features = aggregate(rows2, PERSON2_FEATURES, "person2").withColumn("has_person2", F.lit(True))

# 3. One row per case

features = cases.select("case_id")
for part in (applicant, related, person2_features):
    features = features.join(part, "case_id", "left")
flags = ["has_person1_applicant", "has_person1_related", "has_person2"]
count_columns = [c for c in features.columns if c.startswith("case_id_")]
features = features.fillna(False, subset=flags).fillna(0, subset=count_columns)

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
    F.count(F.when(F.col("has_person1_applicant"), 1)).alias("with_applicant"),
    F.count(F.when(F.col("has_person1_related"), 1)).alias("with_related"),
    F.count(F.when(F.col("has_person2"), 1)).alias("with_person2"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = pd.DataFrame([
    ("person1_rows_in_bronze", person.count()),
    ("person2_rows_in_bronze", person2.count()),
    ("cases", out.cases),
    ("cases_with_applicant_row", out.with_applicant),
    ("cases_with_related_persons", out.with_related),
    ("cases_with_person2", out.with_person2),
], columns=["metric", "value"])

# how often each coalesce fills a value, for the applicant and for related persons
metrics = []
for name, fallback in PERSON1_FALLBACKS.items():
    primary, second = F.col(f"{name}_1"), F.col(f"{fallback}_1")
    metrics += [
        F.count(F.when(primary.isNull(), 1)).alias(f"{name}_before"),
        F.count(F.when(primary.isNull() & second.isNull(), 1)).alias(f"{name}_after"),
    ]
groups = before_fallback.groupBy((F.col("num_group1") == 0).alias("applicant")) \
    .agg(F.count("*").alias("rows"), *metrics).collect()
fallback_rows = []
for group in groups:
    for name, fallback in PERSON1_FALLBACKS.items():
        fallback_rows.append((
            name, f"{name} -> {fallback}", "applicant" if group.applicant else "related person",
            group.rows, group[f"{name}_before"], group[f"{name}_after"],
        ))
fallbacks = pd.DataFrame(fallback_rows, columns=[
    "column", "source_order", "person_group", "rows", "missing_before_fallback", "missing_after_fallback",
])
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
