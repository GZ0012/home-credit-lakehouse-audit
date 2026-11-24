"""Gold 02: checks for gold_volkova.train_features (our own checks, not from Volkova).

1. one row per case, same cases as train_case_sources
2. date_decision / WEEK_NUM / target unchanged
3. column profile (null rate, min, max); no inf/NaN left, no null in count columns
4. a 1% sample of cases matches the silver tables
5. derived columns: tax min <= max, *_final0 = coalesce(primary, fallback)

Only reads tables; writes reports/02_check_train_features/*.csv and fails at
the end if a check failed.
"""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, NumericType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
SILVER = f"{CATALOG}.silver_volkova"
GOLD = f"{CATALOG}.gold_volkova"
TABLE = f"{GOLD}.train_features"
CASE_TABLE = f"{SILVER}.train_case_sources"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/gold_volkova/files/reports/02_check_train_features"

BASE_COLUMNS = ["date_decision", "WEEK_NUM", "target"]
SAMPLE_FRACTION = 0.01
SAMPLE_SEED = 7

# same lists as in 01_train_features.py
SOURCE_TABLES = [
    "train_applprev_1_features",
    "train_applprev_2_features",
    "train_person_features",
    "train_credit_bureau_1_features",
    "train_static_0_features",
    "train_static_cb_0_features",
    "train_credit_bureau_2_features",
    "train_debitcard_deposit_other_features",
    "train_tax_registry_features",
]
FINAL_FALLBACKS = {
    "birthdate_1_final0": ("birthdate_574D_2_staticcb0", "birth_259D_1_person1"),
    "employed_length_1_final0": ("employedfrom_700D_1_applprev1", "empl_employedfrom_271D_1_person1"),
    "mainoccupationinc_1_final0": ("maininc_215A_1_static0", "mainoccupationinc_384A_1_person1"),
}

results = []


def record(check, value, expected, passed):
    results.append((check, str(value), str(expected), "pass" if passed else "FAIL"))


gold = spark.table(TABLE)
cases = spark.table(CASE_TABLE)
fields = gold.schema.fields
gold_columns = {f.name for f in fields}

# 1. One row per case

grain = gold.agg(
    F.count("*").alias("rows"),
    F.countDistinct("case_id").alias("distinct"),
    F.count(F.when(F.col("case_id").isNull(), 1)).alias("nulls"),
).first()
base_rows = cases.count()
record("rows", grain.rows, base_rows, grain.rows == base_rows)
record("distinct_case_ids", grain.distinct, grain.rows, grain.distinct == grain.rows)
record("null_case_ids", grain.nulls, 0, grain.nulls == 0)
record("columns", len(fields), "info", True)

# 2. Base columns unchanged

left = gold.select("case_id", *BASE_COLUMNS, F.lit(1).alias("_in_gold"))
right = cases.select("case_id", *[F.col(c).alias(f"_base_{c}") for c in BASE_COLUMNS], F.lit(1).alias("_in_base"))
both = F.col("_in_gold").isNotNull() & F.col("_in_base").isNotNull()
base_check = left.join(right, "case_id", "full_outer").agg(
    F.count(F.when(F.col("_in_gold").isNull(), 1)).alias("missing_in_gold"),
    F.count(F.when(F.col("_in_base").isNull(), 1)).alias("extra_in_gold"),
    *[F.count(F.when(both & ~F.col(c).eqNullSafe(F.col(f"_base_{c}")), 1)).alias(f"changed_{c}")
      for c in BASE_COLUMNS],
    F.avg("target").alias("target_rate"),
).first().asDict()
for name in ["missing_in_gold", "extra_in_gold", *[f"changed_{c}" for c in BASE_COLUMNS]]:
    record(name, base_check[name], 0, base_check[name] == 0)
record("target_rate", round(base_check["target_rate"], 5), "about 0.031 (info)", True)

# 3. Column profile (columns are numbered in the aliases because names can be long)

floats = [f.name for f in fields if isinstance(f.dataType, (DoubleType, FloatType))]
exprs = [F.count(F.when(F.col(f.name).isNull(), 1)).alias(f"n__{i}") for i, f in enumerate(fields)]
exprs += [F.min(f.name).cast("string").alias(f"lo__{i}") for i, f in enumerate(fields)]
exprs += [F.max(f.name).cast("string").alias(f"hi__{i}") for i, f in enumerate(fields)]
exprs += [F.count(F.when(F.isnan(c) | (F.abs(c) == float("inf")), 1)).alias(f"bad__{c}") for c in floats]
stats = gold.agg(*exprs).first().asDict()

profile = []
for i, f in enumerate(fields):
    nulls, lo, hi = stats[f"n__{i}"], stats[f"lo__{i}"], stats[f"hi__{i}"]
    profile.append((f.name, f.dataType.simpleString(), nulls, round(nulls / grain.rows, 6), lo, hi,
                    nulls == grain.rows, lo is not None and lo == hi))
profile = pd.DataFrame(profile, columns=["column", "type", "nulls", "null_rate", "min", "max",
                                         "all_null", "single_value"])

bad_floats = [c for c in floats if stats[f"bad__{c}"]]
record("float_columns_with_inf_or_nan", len(bad_floats), 0, not bad_floats)
null_counts = profile[profile["column"].str.contains("case_id_") & (profile["nulls"] > 0)]
record("count_columns_with_null", len(null_counts), 0, len(null_counts) == 0)
record("all_null_columns", int(profile["all_null"].sum()), "info", True)
record("single_value_columns", int(profile["single_value"].sum()), "info", True)
non_numeric = [f.name for f in fields
               if not isinstance(f.dataType, NumericType) and f.name not in ("case_id", "date_decision")]
record("non_numeric_columns", len(non_numeric), "info: category in the model", True)

# 4. Sample of cases: gold value == silver value (null-safe)

sample = gold.sample(fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED)
mismatches = []
for table in SOURCE_TABLES:
    silver = spark.table(f"{SILVER}.{table}")
    columns = [c for c in silver.columns if c != "case_id" and not c.startswith("has_") and c in gold_columns]
    renamed = silver.select("case_id", *[F.col(c).alias(f"s__{i}") for i, c in enumerate(columns)])
    joined = sample.select("case_id", *columns).join(renamed, "case_id", "left")
    diff = joined.agg(
        F.count("*").alias("rows"),
        *[F.count(F.when(~F.col(c).eqNullSafe(F.col(f"s__{i}")), 1)).alias(f"d__{i}")
          for i, c in enumerate(columns)],
    ).first().asDict()
    bad = [(table, c, diff[f"d__{i}"], diff["rows"]) for i, c in enumerate(columns) if diff[f"d__{i}"]]
    mismatches += bad
    record(f"silver_match_{table}", f"{len(bad)} of {len(columns)} columns differ", 0, not bad)

# 5. Derived columns

derived = []
for window in ["", "_1month"]:
    lo, hi = F.col(f"amount_1_min{window}_taxregistry1"), F.col(f"amount_1_max{window}_taxregistry1")
    derived.append(F.count(F.when(lo > hi, 1)).alias(f"tax_min_gt_max{window or '_all'}"))
for name, (primary, fallback) in FINAL_FALLBACKS.items():
    expected = F.coalesce(F.col(primary), F.col(fallback))
    derived.append(F.count(F.when(~F.col(name).eqNullSafe(expected), 1)).alias(f"{name}_wrong"))
for name, value in gold.agg(*derived).first().asDict().items():
    record(name, value, 0, value == 0)

# Reports

checks = pd.DataFrame(results, columns=["check", "value", "expected", "result"])
print(checks.to_string(index=False))

os.makedirs(REPORT_DIR, exist_ok=True)
checks.to_csv(f"{REPORT_DIR}/checks.csv", index=False)
profile.to_csv(f"{REPORT_DIR}/column_profile.csv", index=False)
pd.DataFrame(mismatches, columns=["source_table", "column", "mismatched_rows", "sampled_rows"]) \
    .to_csv(f"{REPORT_DIR}/silver_mismatches.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")

failed = checks.loc[checks["result"] == "FAIL", "check"].tolist()
if failed:
    raise ValueError(f"{len(failed)} checks failed: {failed}")
print("All checks passed.")
