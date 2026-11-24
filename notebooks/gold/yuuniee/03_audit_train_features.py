"""Gold yuuniee 03: audit of gold_yuuniee.train_features (our own audit, same as gold volkova 03).

Run after 02_check_train_features.py passes.

1. Hand checks: for random cases, recompute some features straight from the
   bronze tables (without the silver code) and compare with gold. Fails on a mismatch.
2. Single-feature AUC on a row sample: a column with AUC far from 0.5 is a
   strong feature or a leak, so flagged columns need a look.
3. Drift: PSI between the first and last third of the weeks (row sample),
   and the lowest / highest weekly missing rate per column (full table).

Sections 2 and 3 only flag. SEED = None picks a new seed every run (it is
printed and saved in summary.csv).

Output: reports/03_audit_train_features/*.csv
"""

import math
import os
import random

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
TABLE = f"{CATALOG}.gold_yuuniee.train_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/gold_yuuniee/files/reports/03_audit_train_features"

SEED = None
RANDOM_CASES = 50          # random cases from the whole table
CASES_PER_SOURCE = 20      # plus random cases that have rows in each bronze table
SAMPLE_ROWS = 100_000      # row sample for AUC and PSI
TOLERANCE = 1e-9
AUC_FLAG = 0.3             # flag when |AUC - 0.5| > 0.3
MIN_POSITIVES = 30
PSI_BINS = 10
MISSING_DRIFT_FLAG = 0.3
NOT_FEATURES = ["case_id", "date_decision", "target"]


def days(name):
    return F.datediff(F.col(name).cast("date"), F.col("date_decision"))


def num(name):
    return F.col(name).cast("double")


# (check name, bronze table, gold column, how to compute it from the case's bronze rows)
# DataProcessor5 aggregates all rows of a case (no applicant / last-application split)
HAND_CHECKS = [
    ("tax_a_max_amount", "train_tax_registry_a_1", "amount_4527230A_1_max_taxregistrya1",
     lambda: F.max(num("amount_4527230A"))),
    ("tax_a_mean_amount", "train_tax_registry_a_1", "amount_4527230A_1_mean_taxregistrya1",
     lambda: F.avg(num("amount_4527230A"))),
    ("tax_b_var_amount", "train_tax_registry_b_1", "amount_4917619A_1_var_taxregistryb1",
     lambda: F.var_samp(num("amount_4917619A"))),
    ("tax_c_max_amount", "train_tax_registry_c_1", "pmtamount_36A_1_max_taxregistryc1",
     lambda: F.max(num("pmtamount_36A"))),
    ("tax_c_latest_date", "train_tax_registry_c_1", "processingdate_168D_1_max_taxregistryc1",
     lambda: F.max(days("processingdate_168D"))),
    ("debitcard_mean_opening", "train_debitcard_1", "openingdate_857D_1_mean_debitcard1",
     lambda: F.avg(days("openingdate_857D"))),
    ("deposit_max_amount", "train_deposit_1", "amount_416A_1_max_deposit1",
     lambda: F.max(num("amount_416A"))),
    ("other_max_num_group1", "train_other_1", "num_group1_1_max_other1",
     lambda: F.max("num_group1")),
    ("applprev_max_credit", "train_applprev_1", "credamount_590A_1_max_applprev1",
     lambda: F.max(num("credamount_590A"))),
    ("applprev_max_num_group1", "train_applprev_1", "num_group1_1_max_applprev1",
     lambda: F.max("num_group1")),
    ("person1_max_income", "train_person_1", "mainoccupationinc_384A_1_max_person1",
     lambda: F.max(num("mainoccupationinc_384A"))),
    ("person1_mean_birth", "train_person_1", "birth_259D_1_mean_person1",
     lambda: F.avg(days("birth_259D"))),
    ("static0_income", "train_static_0", "maininc_215A_1_static0",
     lambda: F.max(num("maininc_215A"))),
]

seed = SEED if SEED is not None else random.randrange(1_000_000)
print(f"seed = {seed}")
summary = [("seed", seed)]

gold = spark.table(TABLE)
fields = gold.schema.fields
types = {f.name: f.dataType for f in fields}
id_type = types["case_id"].simpleString()
total_rows = gold.count()
os.makedirs(REPORT_DIR, exist_ok=True)

# 1. Hand checks

tables = sorted({check[1] for check in HAND_CHECKS})
picks = gold.select("case_id").orderBy(F.rand(seed)).limit(RANDOM_CASES)
for i, table in enumerate(tables):
    ids = spark.table(f"{BRONZE}.{table}").select(F.col("case_id").cast(id_type).alias("case_id")).distinct()
    picks = picks.union(ids.orderBy(F.rand(seed + i + 1)).limit(CASES_PER_SOURCE))
case_ids = sorted(row.case_id for row in picks.distinct().collect())
sample_ids = spark.createDataFrame([(c,) for c in case_ids], f"case_id {id_type}")
decisions = spark.table(f"{BRONZE}.train_base").select(
    F.col("case_id").cast(id_type).alias("case_id"), F.col("date_decision").cast("date").alias("date_decision"))

expected = sample_ids
for table in tables:
    rows = spark.table(f"{BRONZE}.{table}").withColumn("case_id", F.col("case_id").cast(id_type)) \
        .join(sample_ids, "case_id").join(decisions, "case_id")
    checks = [check for check in HAND_CHECKS if check[1] == table]
    expected = expected.join(rows.groupBy("case_id").agg(*[build().alias(name) for name, _, _, build in checks]),
                             "case_id", "left")

targets = [(name, column) for name, _, column, _ in HAND_CHECKS]
missing_columns = [column for _, column in targets if column not in types]
targets = [(name, column) for name, column in targets if column in types]
gold_values = gold.select("case_id", *[F.col(column).alias(f"gold__{name}") for name, column in targets])
compared = expected.join(gold_values, "case_id", "left").toPandas()


def same(a, b):
    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)
    return math.isclose(float(a), float(b), rel_tol=TOLERANCE, abs_tol=TOLERANCE)


hand = pd.DataFrame(
    [(name, column, case_id, e, g, same(e, g))
     for name, column in targets
     for case_id, e, g in zip(compared["case_id"], compared[name], compared[f"gold__{name}"])],
    columns=["check", "gold_column", "case_id", "expected", "gold", "match"],
)
hand.to_csv(f"{REPORT_DIR}/hand_checks.csv", index=False)
print(hand.groupby("check")["match"].agg(cases="size", mismatches=lambda s: int((~s).sum())).to_string())
hand_mismatches = int((~hand["match"]).sum())
summary += [("hand_check_cases", len(case_ids)), ("hand_check_mismatches", hand_mismatches),
            ("hand_check_missing_gold_columns", len(missing_columns))]

# Row sample for 2 and 3

sample = gold.sample(fraction=min(1.0, SAMPLE_ROWS / total_rows), seed=seed).toPandas()
numeric = [f.name for f in fields if isinstance(f.dataType, NumericType) and f.name not in NOT_FEATURES]
summary.append(("sample_rows", len(sample)))

# 2. Single-feature AUC (rank formula, only on rows that have a value)

positive = sample["target"].to_numpy() == 1
auc_parts = []
for start in range(0, len(numeric), 200):
    block = sample[numeric[start:start + 200]].astype("float64")
    ranks = block.rank()
    n = block.notna().sum()
    n_pos = block[positive].notna().sum()
    n_neg = n - n_pos
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc_parts.append(pd.DataFrame({"column": block.columns, "auc": auc.values, "coverage": (n / len(block)).values,
                                   "positives": n_pos.values, "negatives": n_neg.values}))
aucs = pd.concat(auc_parts, ignore_index=True)
aucs.loc[(aucs["positives"] < MIN_POSITIVES) | (aucs["negatives"] < MIN_POSITIVES), "auc"] = np.nan
aucs["strength"] = (aucs["auc"] - 0.5).abs()
aucs["flag"] = aucs["strength"] > AUC_FLAG
aucs = aucs.sort_values("strength", ascending=False)
aucs.to_csv(f"{REPORT_DIR}/single_feature_auc.csv", index=False)
summary += [("auc_columns", int(aucs["auc"].notna().sum())), ("auc_flagged", int(aucs["flag"].sum()))]
print(aucs.head(15).to_string(index=False))

# 3a. PSI: first third of the weeks vs last third (numeric columns, missing is its own bin)

weeks = np.sort(sample["WEEK_NUM"].unique())
third = max(1, len(weeks) // 3)
first = sample[sample["WEEK_NUM"].isin(weeks[:third])]
last = sample[sample["WEEK_NUM"].isin(weeks[-third:])]


def shares(values, edges):
    values = values.to_numpy(dtype="float64")
    missing = np.isnan(values)
    counts = np.bincount(np.searchsorted(edges, values[~missing], side="right"), minlength=len(edges) + 1)
    return np.clip(np.append(counts, missing.sum()) / max(len(values), 1), 1e-4, None)


psi_rows = []
for c in numeric:
    if c == "WEEK_NUM":
        continue
    known = first[c].dropna().to_numpy(dtype="float64")
    edges = np.unique(np.quantile(known, np.linspace(0, 1, PSI_BINS + 1)[1:-1])) if len(known) else np.array([])
    old, new = shares(first[c], edges), shares(last[c], edges)
    psi_rows.append((c, float(np.sum((new - old) * np.log(new / old))), first[c].isna().mean(), last[c].isna().mean()))
psis = pd.DataFrame(psi_rows, columns=["column", "psi", "missing_first_third", "missing_last_third"])
psis["level"] = pd.cut(psis["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "shifted"])
psis = psis.sort_values("psi", ascending=False)
psis.to_csv(f"{REPORT_DIR}/psi.csv", index=False)
summary += [("psi_moderate", int((psis["level"] == "moderate").sum())),
            ("psi_shifted", int((psis["level"] == "shifted").sum()))]

# 3b. Missing rate per week (full table)

tracked = [f.name for f in fields if f.name not in ("case_id", "WEEK_NUM", "target")]
weekly = gold.groupBy("WEEK_NUM").agg(
    F.count("*").alias("rows"),
    F.avg("target").alias("target_rate"),
    *[F.avg(F.col(c).isNull().cast("double")).alias(f"m__{i}") for i, c in enumerate(tracked)],
).orderBy("WEEK_NUM").toPandas()
weekly[["WEEK_NUM", "rows", "target_rate"]].to_csv(f"{REPORT_DIR}/target_by_week.csv", index=False)

rates = weekly[[f"m__{i}" for i in range(len(tracked))]]
rates.columns = tracked
drift = pd.DataFrame({"column": tracked, "lowest_weekly_missing": rates.min().values,
                      "highest_weekly_missing": rates.max().values})
drift["range"] = drift["highest_weekly_missing"] - drift["lowest_weekly_missing"]
drift["flag"] = drift["range"] > MISSING_DRIFT_FLAG
drift = drift.sort_values("range", ascending=False)
drift.to_csv(f"{REPORT_DIR}/missing_drift.csv", index=False)
summary.append(("missing_drift_flagged", int(drift["flag"].sum())))

# Summary

summary = pd.DataFrame(summary, columns=["item", "value"]).astype({"value": str})
summary.to_csv(f"{REPORT_DIR}/summary.csv", index=False)
print(summary.to_string(index=False))
print(f"Saved reports to {REPORT_DIR}")

if missing_columns:
    raise ValueError(f"gold is missing hand-check columns: {missing_columns}")
if hand_mismatches:
    failed = sorted(hand.loc[~hand["match"], "check"].unique())
    raise ValueError(f"{hand_mismatches} hand-check values differ from bronze: {failed} (seed {seed})")
print("Hand checks passed.")
