"""Gold 01: join all Volkova silver tables into one training table (one row per case).

After the join this runs Volkova's FinalDataProcessor.transform from
DataProcessor2 (commit d5e228c): combined tax registry features, three final
fallbacks, fill the count columns with 0 and set +/-inf to null.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

She also drops MONTH (left out here), casts strings to categorical and sorts
by WEEK_NUM, case_id. Delta has no categorical type or row order, so those two
steps happen in 04_train_lgbm.py.

Output: gold_volkova.train_features + reports/01_train_features/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
SILVER = f"{CATALOG}.silver_volkova"
GOLD = f"{CATALOG}.gold_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/gold/volkova"

CASE_TABLE = f"{SILVER}.train_case_sources"
OUTPUT_TABLE = f"{GOLD}.train_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_features"
VOLUME = f"{GOLD}.files"
VOLUME_LOCATION = f"{STORAGE_ROOT}/files"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/gold_volkova/files/reports/01_train_features"

BASE_COLUMNS = ["case_id", "date_decision", "WEEK_NUM", "target"]

# silver tables, in the order the silver scripts build them
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

# Volkova's combined tax features: amount_1_<stat><window>_taxregistry1
TAX_STATS = ["mean", "min", "max", "last"]
TAX_WINDOWS = ["", "_1month"]
TAX_COLUMNS = [
    "amount_4527230A_1_{stat}{window}_taxregistrya1",
    "amount_4917619A_1_{stat}{window}_taxregistryb1",
    "pmtamount_36A_1_{stat}{window}_taxregistryc1",
]

# Volkova's final coalesce: new column -> (primary, fallback)
FINAL_FALLBACKS = {
    "birthdate_1_final0": ("birthdate_574D_2_staticcb0", "birth_259D_1_person1"),
    "employed_length_1_final0": ("employedfrom_700D_1_applprev1", "empl_employedfrom_271D_1_person1"),
    "mainoccupationinc_1_final0": ("maininc_215A_1_static0", "mainoccupationinc_384A_1_person1"),
}


# 1. Join every silver table to the labeled cases

cases = spark.table(CASE_TABLE)
features = cases.select(*BASE_COLUMNS)
owner = {c: "train_case_sources" for c in BASE_COLUMNS}   # which table each column comes from
for table in SOURCE_TABLES:
    df = spark.table(f"{SILVER}.{table}")
    # has_* flags are only for the silver audit
    columns = [c for c in df.columns if c != "case_id" and not c.startswith("has_")]
    clashes = [c for c in columns if c in owner]
    if clashes:
        raise ValueError(f"{table} repeats columns from another table: {clashes[:10]}")
    owner.update({c: table for c in columns})
    features = features.join(df.select("case_id", *columns), "case_id", "left")

# 2. Volkova's FinalDataProcessor.transform

tax = {}
for window in TAX_WINDOWS:
    for stat in TAX_STATS:
        inputs = [F.col(pattern.format(stat=stat, window=window)) for pattern in TAX_COLUMNS]
        name = f"amount_1_{stat}{window}_taxregistry1"
        if stat == "mean":
            # pl.sum_horizontal(...) / 3: sum_horizontal skips nulls, so a missing registry counts as 0
            tax[name] = sum(F.coalesce(c, F.lit(0.0)) for c in inputs) / 3
        elif stat == "min":
            tax[name] = F.least(*inputs)       # pl.min_horizontal
        else:
            tax[name] = F.greatest(*inputs)    # pl.max_horizontal (she uses max for "last" too)
features = features.withColumns(tax)

features = features.withColumns({
    name: F.coalesce(F.col(primary), F.col(fallback)) for name, (primary, fallback) in FINAL_FALLBACKS.items()
})

# fill nans: columns with "case_id_" in the name are row counts -> 0
count_columns = [c for c in features.columns if "case_id_" in c]
features = features.fillna(0, subset=count_columns)

# +/-inf -> null (NaN too)
floats = [f.name for f in features.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
features = features.withColumns({
    c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
})

# 3. Save first, then check the saved table.
# CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild.

spark.sql(f"CREATE EXTERNAL VOLUME IF NOT EXISTS {VOLUME} LOCATION '{VOLUME_LOCATION}'")
features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

written = spark.table(OUTPUT_TABLE)
out = written.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    *[F.count(F.col(name)).alias(name) for name in FINAL_FALLBACKS],
    *[F.count(F.col(primary)).alias(f"{name}_primary") for name, (primary, _) in FINAL_FALLBACKS.items()],
).first().asDict()
base_count = cases.count()
if out["cases"] != base_count or out["distinct_cases"] != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out['cases']} rows")

# 4. Reports

owner.update({c: "gold_01_train_features" for c in written.columns if c not in owner})
inventory = pd.DataFrame([(c, owner[c]) for c in written.columns], columns=["column", "source_table"])
per_table = inventory.groupby("source_table").size().reset_index(name="columns")
fallbacks = pd.DataFrame(
    [(name, f"{primary} -> {fallback}", out["cases"], out["cases"] - out[f"{name}_primary"],
      out["cases"] - out[name])
     for name, (primary, fallback) in FINAL_FALLBACKS.items()],
    columns=["derived_field", "source_order", "cases", "missing_before_fallback", "missing_after_fallback"],
)
print(per_table.to_string(index=False))
print(fallbacks.to_string(index=False))
print(f"{len(written.columns)} columns, {out['cases']} cases")

os.makedirs(REPORT_DIR, exist_ok=True)
inventory.to_csv(f"{REPORT_DIR}/column_inventory.csv", index=False)
per_table.to_csv(f"{REPORT_DIR}/columns_per_source.csv", index=False)
fallbacks.to_csv(f"{REPORT_DIR}/final_fallbacks.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
