"""Silver 01: labeled cases + the case-level sources (person1 applicant, static0, staticcb0).

This is the base table for the other silver scripts: one row per case_id from
train_base, with the columns the later steps need and the fallback (coalesce)
columns Volkova builds.

Fallback rules are from Volkova's DataProcessor2 (commit d5e228c):
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_case_sources + reports/01_case_sources/*.csv
"""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

OUTPUT_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_case_sources"
# table and volume need separate folders (Unity Catalog does not allow overlapping paths)
VOLUME = f"{SCHEMA}.files"
VOLUME_LOCATION = f"{STORAGE_ROOT}/files"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/01_case_sources"

# Volkova: this category code means "missing", she sets it to null in all string columns
MISSING_CATEGORY = "a55475b1"

BASE_COLUMNS = ["case_id", "date_decision", "WEEK_NUM", "MONTH", "target"]

# columns we keep from each source (renamed to <prefix>_<column>)
SOURCE_COLUMNS = {
    "person1": [
        "birth_259D", "birthdate_87D", "gender_992L", "sex_738L",
        "relationshiptoclient_415T", "relationshiptoclient_642T",
        "mainoccupationinc_384A", "empl_employedfrom_271D",
    ],
    "static0": [
        "maininc_215A", "sumoutstandtotal_3546847A", "sumoutstandtotalest_4493215A",
        "credamount_770A", "annuity_780A",
    ],
    "staticcb0": [
        "birthdate_574D", "dateofbirth_337D", "dateofbirth_342D",
        "education_1103M", "education_88M", "maritalst_385M", "maritalst_893M",
        "pmtaverage_4527227A", "pmtaverage_4955615A", "pmtaverage_3A",
        "pmtcount_4527229L", "pmtcount_4955617L", "pmtcount_693L",
        "assignmentdate_238D", "assignmentdate_4527235D", "assignmentdate_4955616D",
    ],
}

# Volkova's pl.coalesce rules: (our name, inputs in her order, her column name, is categorical)
FALLBACKS = [
    ("person_birthdate", ["person1_birth_259D", "person1_birthdate_87D"],
     "birth_259D_1_person1", False),
    ("person_gender", ["person1_gender_992L", "person1_sex_738L"],
     "gender_992L_1_person1", True),
    ("person_relationship", ["person1_relationshiptoclient_415T", "person1_relationshiptoclient_642T"],
     "relationshiptoclient_415T_1_person1", True),
    ("external_birthdate", ["staticcb0_birthdate_574D", "staticcb0_dateofbirth_337D",
                            "staticcb0_dateofbirth_342D"],
     "birthdate_574D_2_staticcb0", False),
    ("external_education", ["staticcb0_education_1103M", "staticcb0_education_88M"],
     "education_1103M_2_staticcb0", True),
    ("external_marital_status", ["staticcb0_maritalst_385M", "staticcb0_maritalst_893M"],
     "maritalst_385M_2_staticcb0", True),
    ("external_tax_mean", ["staticcb0_pmtaverage_4527227A", "staticcb0_pmtaverage_4955615A",
                           "staticcb0_pmtaverage_3A"],
     "pmtaverage_3A_2_staticcb0", False),
    ("external_tax_count", ["staticcb0_pmtcount_4527229L", "staticcb0_pmtcount_4955617L",
                            "staticcb0_pmtcount_693L"],
     "pmtcount_4527229L_2_staticcb0", False),
    ("external_tax_assignment_date", ["staticcb0_assignmentdate_238D",
                                      "staticcb0_assignmentdate_4527235D",
                                      "staticcb0_assignmentdate_4955616D"],
     "assignmentdate_238D_2_staticcb0", False),
    ("outstanding_balance", ["static0_sumoutstandtotal_3546847A", "static0_sumoutstandtotalest_4493215A"],
     "sumoutstandtotal_3546847A_2_static0", False),
    # FinalDataProcessor (final step)
    ("applicant_birthdate", ["external_birthdate", "person_birthdate"],
     "birthdate_1_final0", False),
    ("applicant_main_income", ["static0_maininc_215A", "person1_mainoccupationinc_384A"],
     "mainoccupationinc_1_final0", False),
]
# Not used by her model:
# - outstanding_balance is overwritten by a ratio with the same name in _process_static0
# - person_relationship is dropped from the applicant row (include = 0 in features_2.csv)
NOT_USED_BY_MODEL = {"outstanding_balance", "person_relationship"}


def check_one_row_per_case(df, name):
    counts = df.agg(
        F.count("*").alias("rows"),
        F.countDistinct("case_id").alias("cases"),
        F.count(F.when(F.col("case_id").isNull(), 1)).alias("null_ids"),
    ).first()
    if counts.null_ids or counts.rows != counts.cases:
        raise ValueError(f"{name}: expected one row per case_id, got {counts}")
    return counts.rows


def fallback_input(column, categorical):
    # Volkova: masked category -> null before the coalesce
    if categorical:
        value = F.col(column).cast("string")
        return F.when(value != MISSING_CATEGORY, value)
    return F.col(column)


# 1. Join the sources to the labeled cases

base = spark.table(f"{BRONZE}.train_base").select(*BASE_COLUMNS) \
    .withColumn("date_decision", F.col("date_decision").cast("date"))
base_count = check_one_row_per_case(base, "train_base")

sources = {
    # Volkova (_handle_person1): num_group1 = 0 is the applicant
    "person1": spark.table(f"{BRONZE}.train_person_1").where(F.col("num_group1") == 0),
    "static0": spark.table(f"{BRONZE}.train_static_0"),
    "staticcb0": spark.table(f"{BRONZE}.train_static_cb_0"),
}

cases = base
for prefix, df in sources.items():
    check_one_row_per_case(df, prefix)
    columns = [
        (F.col(c).cast("date") if c.endswith("D") else F.col(c)).alias(f"{prefix}_{c}")
        for c in SOURCE_COLUMNS[prefix]
    ]
    df = df.select("case_id", F.lit(True).alias(f"has_{prefix}"), *columns)
    cases = cases.join(df, "case_id", "left").fillna(False, subset=[f"has_{prefix}"])

# 2. Fallback columns (dates stay dates here; she uses day offsets, same order)

for name, inputs, _, categorical in FALLBACKS:
    cases = cases.withColumn(name, F.coalesce(*[fallback_input(c, categorical) for c in inputs]))

# Volkova: date minus date_decision, so past dates are negative
cases = cases.withColumn(
    "birthdate_1_final0", F.datediff(F.col("applicant_birthdate"), F.col("date_decision"))
)

# 3. Reports: source coverage and how much each fallback fills

metrics = [F.count("*").alias("cases")]
for prefix in SOURCE_COLUMNS:
    metrics.append(F.count(F.when(~F.col(f"has_{prefix}"), 1)).alias(f"no_{prefix}"))
for name, inputs, _, categorical in FALLBACKS:
    primary = fallback_input(inputs[0], categorical)
    metrics += [
        F.count(F.when(primary.isNull(), 1)).alias(f"{name}_before"),
        F.count(F.when(F.col(name).isNull(), 1)).alias(f"{name}_after"),
    ]
counts = cases.agg(*metrics).first().asDict()
if counts["cases"] != base_count:
    raise ValueError(f"expected {base_count} cases, got {counts['cases']}")

coverage = pd.DataFrame(
    [(prefix, counts["cases"], counts[f"no_{prefix}"]) for prefix in SOURCE_COLUMNS],
    columns=["source", "cases", "unmatched_cases"],
)
fallbacks = pd.DataFrame(
    [
        (name, author_column, " -> ".join(inputs), name not in NOT_USED_BY_MODEL, counts["cases"],
         counts[f"{name}_before"], counts[f"{name}_before"] - counts[f"{name}_after"], counts[f"{name}_after"])
        for name, inputs, author_column, _ in FALLBACKS
    ],
    columns=["derived_field", "author_column", "source_order", "used_by_model", "cases",
             "missing_before_fallback", "recovered", "missing_after_fallback"],
)
print(coverage.to_string(index=False))
print(fallbacks.to_string(index=False))

# 4. Save

spark.sql(f"CREATE EXTERNAL VOLUME IF NOT EXISTS {VOLUME} LOCATION '{VOLUME_LOCATION}'")

# CTAS fails if the table exists, so a rerun can't overwrite it.
# To rebuild: DROP TABLE and delete the table folder in storage first.
cases.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/source_coverage.csv", index=False)
fallbacks.to_csv(f"{REPORT_DIR}/fallback_missingness.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
