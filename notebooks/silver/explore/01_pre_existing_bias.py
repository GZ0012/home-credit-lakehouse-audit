# Databricks notebook source
"""Explore 01: pre-existing bias in the raw bronze data. Our own audit, not part of the replication.

Is the default rate (target = 1) different across gender, age and family /
marital state before any feature engineering or model? Reads bronze only, so
the result is the same for both pipelines (volkova and yuuniee).

Tables: bronze.train_base (cases, target), train_person_1, train_static_cb_0.
Groups describe the applicant (person_1 row with num_group1 = 0):
- gender        gender_992L, fallback sex_738L (same row)
- age           birthdate: staticcb0 birthdate_574D -> dateofbirth_337D -> dateofbirth_342D,
                then person1 birth_259D -> birthdate_87D; years before date_decision
- family_state  familystate_447L (MARRIED, SINGLE, ...);
                used instead of maritalst_385M, whose values are hashed codes
The masked category a55475b1 counts as missing. Missing values stay a group of their own.

Per group: cases, 1s and 0s, default rate, gap to the overall rate and to a
reference group (the largest non-missing group), chi-square test, weekly rates.

Output: /Volumes/dbw_home_credit_audit/silver_explore/files/reports/01_pre_existing_bias/*.csv
"""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from scipy.stats import chi2_contingency

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_explore"
VOLUME = f"{SCHEMA}.files"   # managed volume, stored in the schema's own location
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_explore/files/reports/01_pre_existing_bias"

MISSING_CATEGORY = "a55475b1"         # this category code means "missing"
AGE_BINS = [18, 25, 35, 45, 55, 65]   # -> <18, 18-24, 25-34, ..., 65+
ATTRIBUTES = ["gender", "age_group", "family_state"]

# COMMAND ----------

# 1. One row per case with the applicant's gender, age and family state


def category(column):
    value = F.col(column).cast("string")
    return F.when(value != MISSING_CATEGORY, value)


def first_date(*columns):
    return F.coalesce(*[F.col(c).cast("date") for c in columns])


cases = spark.table(f"{BRONZE}.train_base").select(
    "case_id", "WEEK_NUM", "target", F.col("date_decision").cast("date").alias("date_decision"),
)
applicant = (
    spark.table(f"{BRONZE}.train_person_1")
    .where(F.col("num_group1") == 0)
    .select(
        "case_id",
        F.coalesce(category("gender_992L"), category("sex_738L")).alias("gender_value"),
        category("familystate_447L").alias("family_value"),
        first_date("birth_259D", "birthdate_87D").alias("person_birthdate"),
    )
)
external = spark.table(f"{BRONZE}.train_static_cb_0").select(
    "case_id",
    first_date("birthdate_574D", "dateofbirth_337D", "dateofbirth_342D").alias("external_birthdate"),
)

people = (
    cases.join(applicant, "case_id", "left")
    .join(external, "case_id", "left")
    .withColumn(
        "birth_days",   # negative: days before date_decision
        F.datediff(F.coalesce("external_birthdate", "person_birthdate"), F.col("date_decision")),
    )
)

age = -F.col("birth_days") / 365.25
age_group = F.when(F.col("birth_days").isNull(), "missing").when(age < AGE_BINS[0], f"<{AGE_BINS[0]}")
for low, high in zip(AGE_BINS, AGE_BINS[1:]):
    age_group = age_group.when(age < high, f"{low}-{high - 1}")
age_group = age_group.otherwise(f"{AGE_BINS[-1]}+")

df = people.select(
    "case_id", "WEEK_NUM", "target",
    F.coalesce("gender_value", F.lit("missing")).alias("gender"),
    age_group.alias("age_group"),
    F.coalesce("family_value", F.lit("missing")).alias("family_state"),
    age.alias("age_years"),
).cache()

# the joins must not add rows (one applicant row and one staticcb0 row per case)
total = df.count()
case_count = cases.count()
if total != case_count or df.select("case_id").distinct().count() != case_count:
    raise ValueError(f"expected one row per case: {case_count} cases, {total} rows after the joins")
overall_rate = df.agg(F.avg("target")).first()[0]
print(f"{total:,} cases, overall default rate {overall_rate:.4f}")

# COMMAND ----------

# 2. Count of 1s and 0s and default rate per group


def group_table(attribute):
    counts = (
        df.groupBy(F.col(attribute).alias("group"))
        .agg(
            F.count("*").alias("cases"),
            F.sum("target").alias("target_1"),
            F.sum(1 - F.col("target")).alias("target_0"),
        )
        .toPandas()
    )
    counts.insert(0, "attribute", attribute)
    counts["share_of_cases"] = counts["cases"] / total
    counts["default_rate"] = counts["target_1"] / counts["cases"]
    counts["diff_vs_overall"] = counts["default_rate"] - overall_rate

    # reference = largest non-missing group
    known = counts[counts["group"] != "missing"]
    reference = known.loc[known["cases"].idxmax()]
    counts["reference_group"] = reference["group"]
    counts["diff_vs_reference"] = counts["default_rate"] - reference["default_rate"]
    counts["default_ratio_vs_reference"] = counts["default_rate"] / reference["default_rate"]
    # favourable outcome = repaid (target 0); below 0.8 fails the four-fifths rule
    counts["repaid_ratio_vs_reference"] = (1 - counts["default_rate"]) / (1 - reference["default_rate"])
    return counts.sort_values("cases", ascending=False)


groups = pd.concat([group_table(a) for a in ATTRIBUTES], ignore_index=True)
print(groups.to_string(index=False))

# COMMAND ----------

# 3. Chi-square test: is target independent of the group?

tests = []
for attribute in ATTRIBUTES:
    for include_missing in [True, False]:
        part = groups[groups["attribute"] == attribute]
        if not include_missing:
            part = part[part["group"] != "missing"]
        if len(part) < 2:
            continue
        chi2, p_value, dof, _ = chi2_contingency(part[["target_1", "target_0"]].values)
        rates = part["default_rate"]
        tests.append({
            "attribute": attribute,
            "include_missing": include_missing,
            "groups": len(part),
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
            "min_default_rate": rates.min(),
            "max_default_rate": rates.max(),
            "max_minus_min": rates.max() - rates.min(),
            "max_over_min": rates.max() / rates.min() if rates.min() > 0 else float("inf"),
        })
tests = pd.DataFrame(tests)
print(tests.to_string(index=False))

# COMMAND ----------

# 4. Default rate per group per week, to see if the gap is stable over time

weekly = (
    df.select("WEEK_NUM", "target", F.explode(F.array(*[
        F.struct(F.lit(a).alias("attribute"), F.col(a).alias("group")) for a in ATTRIBUTES
    ])).alias("g"))
    .groupBy("g.attribute", "g.group", "WEEK_NUM")
    .agg(F.count("*").alias("cases"), F.avg("target").alias("default_rate"))
    .orderBy("attribute", "group", "WEEK_NUM")
    .toPandas()
)

# ages outside a normal range point to bad birthdays
age_check = df.agg(
    F.min("age_years").alias("min_age"),
    F.max("age_years").alias("max_age"),
    F.sum((F.col("age_years") < 18).cast("int")).alias("under_18"),
    F.sum((F.col("age_years") > 100).cast("int")).alias("over_100"),
).toPandas()
print(age_check.to_string(index=False))

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {VOLUME}")
os.makedirs(REPORT_DIR, exist_ok=True)
groups.to_csv(f"{REPORT_DIR}/group_rates.csv", index=False)
tests.to_csv(f"{REPORT_DIR}/chi2_tests.csv", index=False)
weekly.to_csv(f"{REPORT_DIR}/weekly_group_rates.csv", index=False)
age_check.to_csv(f"{REPORT_DIR}/age_check.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")

display(groups)
display(tests)
