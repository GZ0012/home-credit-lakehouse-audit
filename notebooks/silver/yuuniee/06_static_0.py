"""Silver yuuniee 06: current application (train_static_0) -> one row per case.

Depth 0: one row per case, columns kept as they are (no formulas in DataProcessor5).

yuuniee's feature code is not public, so we use Volkova's DataProcessor5 (commit
d5e228c) as a stand-in: mean / max / var per column, no hand-made features.
Column lists are from her features/features_5.csv, filtered by features/features5.txt.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_5.py

Output: silver_yuuniee.train_static_0_features + reports/06_static_0/row_coverage.csv
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
OUTPUT_TABLE = f"{SCHEMA}.train_static_0_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_static_0_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_yuuniee/files/reports/06_static_0"

# From Volkova's features_5.csv / features5.txt. For each source:
#   table: bronze table; chunked: split into several parquet files
#   features: column -> aggregations (depth 1/2), or the columns kept as they are (depth 0)
SOURCES = {
    "static0": {
        "table": "train_static_0",
        "chunked": True,
        "depth": 0,
        "features": (
            "actualdpdtolerance_344P", "amtinstpaidbefduel24m_4187115A", "annuity_780A",
            "annuitynextmonth_57A", "applicationcnt_361L", "applications30d_658L", "applicationscnt_1086L",
            "applicationscnt_464L", "applicationscnt_629L", "applicationscnt_867L",
            "avgdbddpdlast24m_3658932P", "avgdbddpdlast3m_4187120P", "avgdbdtollast24m_4525197P",
            "avgdpdtolclosure24_3658938P", "avginstallast24m_3658937A", "avglnamtstart24m_4525187A",
            "avgmaxdpdlast9m_3716943P", "avgoutstandbalancel6m_4187114A", "avgpmtlast12m_4525200A",
            "bankacctype_710L", "cardtype_51L", "clientscnt_100L", "clientscnt_1022L", "clientscnt_1071L",
            "clientscnt_1130L", "clientscnt_157L", "clientscnt_257L", "clientscnt_304L", "clientscnt_360L",
            "clientscnt_493L", "clientscnt_533L", "clientscnt_887L", "clientscnt_946L",
            "clientscnt12m_3712952L", "clientscnt3m_3712950L", "clientscnt6m_3712949L",
            "cntincpaycont9m_3716944L", "cntpmts24_3658933L", "commnoinclast6m_3546845L", "credamount_770A",
            "credtype_322L", "currdebt_22A", "currdebtcredtyperange_828A", "datefirstoffer_1144D",
            "datelastinstal40dpd_247D", "datelastunpaid_3546854D", "daysoverduetolerancedd_3976961L",
            "deferredmnthsnum_166L", "disbursedcredamount_1113A", "disbursementtype_67L", "downpmt_116A",
            "dtlastpmtallstes_4499206D", "eir_270L", "equalitydataagreement_891L", "firstclxcampaign_1125D",
            "firstdatedue_489D", "homephncnt_628L", "inittransactionamount_650A",
            "inittransactioncode_186L", "interestrate_311L", "isbidproduct_1095L", "isdebitcard_729L",
            "lastactivateddate_801D", "lastapplicationdate_877D", "lastapprcommoditycat_1041M",
            "lastapprcredamount_781A", "lastapprdate_640D", "lastcancelreason_561M", "lastdelinqdate_224D",
            "lastrejectcommoditycat_161M", "lastrejectcommodtypec_5251769M", "lastrejectcredamount_222A",
            "lastrejectdate_50D", "lastrejectreason_759M", "lastrejectreasonclient_4145040M", "lastst_736L",
            "maininc_215A", "mastercontrelectronic_519L", "mastercontrexist_109L", "maxannuity_159A",
            "maxdbddpdlast1m_3658939P", "maxdbddpdtollast12m_3658940P", "maxdbddpdtollast6m_4187119P",
            "maxdebt4_972A", "maxdpdfrom6mto36m_3546853P", "maxdpdinstldate_3546855D",
            "maxdpdinstlnum_3546846P", "maxdpdlast12m_727P", "maxdpdlast24m_143P", "maxdpdlast3m_392P",
            "maxdpdlast6m_474P", "maxdpdlast9m_1059P", "maxdpdtolerance_374P", "maxinstallast24m_3658928A",
            "maxlnamtstart6m_4525199A", "maxoutstandbalancel12m_4187113A", "maxpmtlast3m_4525190A",
            "mindbddpdlast24m_3658935P", "mindbdtollast24m_4525191P", "mobilephncnt_593L",
            "monthsannuity_845L", "numactivecreds_622L", "numactivecredschannel_414L",
            "numactiverelcontr_750L", "numcontrs3months_479L", "numincomingpmts_3546848L",
            "numinstlallpaidearly3d_817L", "numinstls_657L", "numinstlsallpaid_934L",
            "numinstlswithdpd10_728L", "numinstlswithdpd5_4187116L", "numinstlswithoutdpd_562L",
            "numinstmatpaidtearly2d_4499204L", "numinstpaid_4499208L", "numinstpaidearly_338L",
            "numinstpaidearly3d_3546850L", "numinstpaidearly3dest_4493216L", "numinstpaidearly5d_1087L",
            "numinstpaidearly5dest_4493211L", "numinstpaidearly5dobd_4499205L",
            "numinstpaidearlyest_4493214L", "numinstpaidlastcontr_4325080L", "numinstpaidlate1d_3546852L",
            "numinstregularpaid_973L", "numinstregularpaidest_4493210L", "numinsttopaygr_769L",
            "numinsttopaygrest_4493213L", "numinstunpaidmax_3546851L", "numinstunpaidmaxest_4493212L",
            "numnotactivated_1143L", "numpmtchanneldd_318L", "numrejects9m_859L", "opencred_647L",
            "paytype_783L", "paytype1st_925L", "pctinstlsallpaidearl3d_427L", "pctinstlsallpaidlat10d_839L",
            "pctinstlsallpaidlate1d_3546856L", "pctinstlsallpaidlate4d_3546849L",
            "pctinstlsallpaidlate6d_3546844L", "pmtnum_254L", "posfpd10lastmonth_333P",
            "posfpd30lastmonth_3976960P", "posfstqpd30lastmonth_3976962P", "price_1097A",
            "sellerplacecnt_915L", "sellerplacescnt_216L", "sumoutstandtotal_3546847A",
            "sumoutstandtotalest_4493215A", "totaldebt_9A", "totalsettled_863A", "totinstallast1m_4525188A",
            "twobodfilling_608L", "typesuite_864L", "validfrom_1069D"
        ),
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
