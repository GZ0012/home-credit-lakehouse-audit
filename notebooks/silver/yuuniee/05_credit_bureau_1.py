"""Silver yuuniee 05: credit bureau contracts (train_credit_bureau_a_1, _b_1) -> one row per case.

Sources a and b are processed on their own (no active / closed split, no a + b merge).
No b1 column is in features5.txt, so b1 only gives the num_group1 max.

yuuniee's feature code is not public, so we use Volkova's DataProcessor5 (commit
d5e228c) as a stand-in: mean / max / var per column, no hand-made features.
Column lists are from her features/features_5.csv, filtered by features/features5.txt.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_5.py

Output: silver_yuuniee.train_credit_bureau_1_features + reports/05_credit_bureau_1/row_coverage.csv
"""

import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_yuuniee"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/yuuniee"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_credit_bureau_1_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_credit_bureau_1_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_yuuniee/files/reports/05_credit_bureau_1"

# From Volkova's features_5.csv / features5.txt. For each source:
#   table: bronze table; chunked: split into several parquet files
#   features: column -> aggregations (depth 1/2), or the columns kept as they are (depth 0)
SOURCES = {
    "creditbureaua1": {
        "table": "train_credit_bureau_a_1",
        "chunked": True,
        "depth": 1,
        "features": {
            "annualeffectiverate_199L": ("max",),
            "annualeffectiverate_63L": ("max",),
            "classificationofcontr_13M": ("max",),
            "classificationofcontr_400M": ("max",),
            "contractst_545M": ("max",),
            "contractst_964M": ("max",),
            "contractsum_5085717L": ("max",),
            "credlmt_230A": ("mean", "max", "var"),
            "credlmt_935A": ("mean", "max", "var"),
            "dateofcredend_289D": ("mean", "max"),
            "dateofcredend_353D": ("mean", "max"),
            "dateofcredstart_181D": ("mean", "max"),
            "dateofcredstart_739D": ("mean", "max"),
            "dateofrealrepmt_138D": ("mean", "max"),
            "debtoutstand_525A": ("mean", "max"),
            "debtoverdue_47A": ("mean", "max"),
            "description_351M": ("max",),
            "dpdmax_139P": ("mean", "max", "var"),
            "dpdmax_757P": ("mean", "max", "var"),
            "dpdmaxdatemonth_442T": ("max",),
            "dpdmaxdatemonth_89T": ("max",),
            "dpdmaxdateyear_596T": ("max",),
            "dpdmaxdateyear_896T": ("max",),
            "financialinstitution_382M": ("max",),
            "financialinstitution_591M": ("max",),
            "instlamount_768A": ("mean", "max", "var"),
            "instlamount_852A": ("mean", "max", "var"),
            "lastupdate_1112D": ("mean", "max"),
            "lastupdate_388D": ("mean", "max"),
            "monthlyinstlamount_332A": ("mean", "max", "var"),
            "monthlyinstlamount_674A": ("mean", "max", "var"),
            "nominalrate_281L": ("max",),
            "nominalrate_498L": ("max",),
            "numberofcontrsvalue_258L": ("max",),
            "numberofcontrsvalue_358L": ("max",),
            "numberofinstls_229L": ("max",),
            "numberofinstls_320L": ("max",),
            "numberofoutstandinstls_520L": ("max",),
            "numberofoutstandinstls_59L": ("max",),
            "numberofoverdueinstlmax_1039L": ("max",),
            "numberofoverdueinstlmax_1151L": ("max",),
            "numberofoverdueinstlmaxdat_148D": ("mean", "max"),
            "numberofoverdueinstlmaxdat_641D": ("mean", "max"),
            "numberofoverdueinstls_725L": ("max",),
            "numberofoverdueinstls_834L": ("max",),
            "outstandingamount_354A": ("mean", "max", "var"),
            "outstandingamount_362A": ("mean", "max", "var"),
            "overdueamount_31A": ("mean", "max", "var"),
            "overdueamount_659A": ("mean", "max", "var"),
            "overdueamountmax_155A": ("mean", "max", "var"),
            "overdueamountmax_35A": ("mean", "max", "var"),
            "overdueamountmax2_14A": ("mean", "max", "var"),
            "overdueamountmax2_398A": ("mean", "max", "var"),
            "overdueamountmax2date_1002D": ("mean", "max"),
            "overdueamountmax2date_1142D": ("mean", "max"),
            "overdueamountmaxdatemonth_284T": ("max",),
            "overdueamountmaxdatemonth_365T": ("max",),
            "overdueamountmaxdateyear_2T": ("max",),
            "overdueamountmaxdateyear_994T": ("max",),
            "periodicityofpmts_1102L": ("max",),
            "periodicityofpmts_837L": ("max",),
            "prolongationcount_1120L": ("max",),
            "purposeofcred_426M": ("max",),
            "purposeofcred_874M": ("max",),
            "refreshdate_3813885D": ("mean", "max"),
            "residualamount_488A": ("mean", "max", "var"),
            "residualamount_856A": ("mean", "max", "var"),
            "subjectrole_182M": ("max",),
            "subjectrole_93M": ("max",),
            "totalamount_6A": ("mean", "max", "var"),
            "totalamount_996A": ("mean", "max", "var"),
            "totaldebtoverduevalue_178A": ("mean", "max"),
            "totaldebtoverduevalue_718A": ("mean", "max"),
            "totaloutstanddebtvalue_39A": ("mean", "max"),
            "totaloutstanddebtvalue_668A": ("mean", "max"),
        },
    },
    "creditbureaub1": {
        "table": "train_credit_bureau_b_1",
        "chunked": False,
        "depth": 1,
        "features": {},   # nothing from this source is in features5.txt
    },
}


def clean(name, dtype):
    # Volkova's utils.set_dtypes, then dates -> days from date_decision (_handle_dates)
    column = F.col(name)
    if name.endswith(("P", "A")):
        return column.cast("double")
    if name.endswith("M"):
        return column.cast("string")
    if name.endswith("D"):
        return F.datediff(column.cast("date"), F.col("date_decision"))
    if "isbid" in name or "isdebit" in name:
        return column.cast("boolean").cast("tinyint")
    if isinstance(dtype, BooleanType):
        return column.cast("string")   # her change_dtypes: Boolean -> String ("true" / "false")
    return column


def agg_expr(column, op):
    # Volkova's Utils.agg_process (mean / var are only used on numeric and date columns here)
    ops = {"mean": F.avg, "max": F.max, "var": F.var_samp}   # polars var uses ddof=1, like var_samp
    return ops[op](column)


def first_chunk_only(rows):
    # she reads the chunk files in order and keeps the first result per case (unique(keep="first"))
    rows = rows.withColumn("_chunk", F.regexp_extract("_source_file", r"_(\d+)\.parquet$", 1).cast("int"))
    first = rows.groupBy("case_id").agg(F.min("_chunk").alias("_first_chunk"))
    return rows.join(first, "case_id").where(F.col("_chunk") == F.col("_first_chunk"))


cases = spark.table(CASE_TABLE)
dates = cases.select("case_id", "date_decision")

features = cases.select("case_id")
row_counts = []
for name, source in SOURCES.items():
    bronze = spark.table(f"{BRONZE}.{source['table']}")
    rows = first_chunk_only(bronze) if source["chunked"] else bronze
    groups = [c for c in bronze.columns if c.startswith("num_group")]
    columns = list(source["features"])

    # 1. clean
    rows = rows.select("case_id", *groups, *columns).join(dates, "case_id", "inner")
    types = {f.name: f.dataType for f in rows.schema.fields}
    rows = rows.select("case_id", *groups, *[clean(c, types[c]).alias(f"{c}_1") for c in columns])

    # 2. aggregate (depth 0 sources already have one row per case)
    if source["depth"] == 0:
        agg = rows.select("case_id", *[f"{c}_1" for c in columns])
    else:
        exprs = [agg_expr(f"{c}_1", op).alias(f"{c}_1_{op}") for c, ops in source["features"].items() for op in ops]
        exprs += [F.max(g).alias(f"{g}_1_max") for g in groups]   # Volkova: max of num_group1 / num_group2
        agg = rows.groupBy("case_id").agg(*exprs)

    agg = agg.select(
        "case_id",
        *[F.col(c).alias(f"{c}_{name}") for c in agg.columns if c != "case_id"],
        F.lit(True).alias(f"has_{name}"),
    )
    features = features.join(agg, "case_id", "left")
    row_counts += [(f"{name}_rows_in_bronze", bronze.count()), (f"{name}_rows_used", rows.count())]

# 3. One row per case

flags = [f"has_{name}" for name in SOURCES]
features = features.fillna(False, subset=flags)

# 4. Checks and report

base_count = cases.count()
out = features.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    *[F.count(F.when(F.col(f"has_{name}"), 1)).alias(f"cases_{name}") for name in SOURCES],
).first().asDict()
if out["cases"] != base_count or out["distinct_cases"] != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = [("cases", out["cases"])] + row_counts + [(f"cases_with_{name}", out[f"cases_{name}"]) for name in SOURCES]
coverage = pd.DataFrame(coverage, columns=["metric", "value"])
print(coverage.to_string(index=False))
print(f"{len(features.columns)} columns")

# 5. Save (CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild)

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
