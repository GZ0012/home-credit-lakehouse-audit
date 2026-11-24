"""Silver 06: current application (train_static_0) -> one row per case.

static0 already has one row per case, so nothing is aggregated. The fallbacks
and the 36 formulas are Volkova's _process_static0 in DataProcessor2 (commit d5e228c):
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Output: silver_volkova.train_static_0_features + reports/06_static_0/*.csv
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
OUTPUT_TABLE = f"{SCHEMA}.train_static_0_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_static_0_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/06_static_0"

SOURCE = "static0"
MISSING_CATEGORY = "a55475b1"
INCOME_FALLBACK = "mainoccupationinc_384A_1_person1"   # applicant income from person1

# static0 columns in Volkova's features_2.csv
FEATURES = (
    "actualdpdtolerance_344P", "amtinstpaidbefduel24m_4187115A", "annuity_780A",
    "annuitynextmonth_57A", "applicationcnt_361L", "applications30d_658L",
    "applicationscnt_1086L", "applicationscnt_464L", "applicationscnt_629L",
    "applicationscnt_867L", "avgdbddpdlast24m_3658932P", "avgdbddpdlast3m_4187120P",
    "avgdbdtollast24m_4525197P", "avgdpdtolclosure24_3658938P", "avginstallast24m_3658937A",
    "avglnamtstart24m_4525187A", "avgmaxdpdlast9m_3716943P", "avgoutstandbalancel6m_4187114A",
    "avgpmtlast12m_4525200A", "bankacctype_710L", "cardtype_51L",
    "clientscnt_100L", "clientscnt_1022L", "clientscnt_1071L",
    "clientscnt_1130L", "clientscnt_136L", "clientscnt_157L",
    "clientscnt_257L", "clientscnt_304L", "clientscnt_360L",
    "clientscnt_493L", "clientscnt_533L", "clientscnt_887L",
    "clientscnt_946L", "clientscnt12m_3712952L", "clientscnt3m_3712950L",
    "clientscnt6m_3712949L", "cntincpaycont9m_3716944L", "cntpmts24_3658933L",
    "commnoinclast6m_3546845L", "credamount_770A", "credtype_322L",
    "currdebt_22A", "currdebtcredtyperange_828A", "datefirstoffer_1144D",
    "datelastinstal40dpd_247D", "datelastunpaid_3546854D", "daysoverduetolerancedd_3976961L",
    "deferredmnthsnum_166L", "disbursedcredamount_1113A", "disbursementtype_67L",
    "downpmt_116A", "dtlastpmtallstes_4499206D", "eir_270L",
    "equalitydataagreement_891L", "equalityempfrom_62L", "firstclxcampaign_1125D",
    "firstdatedue_489D", "homephncnt_628L", "inittransactionamount_650A",
    "inittransactioncode_186L", "interestrate_311L", "interestrategrace_34L",
    "isbidproduct_1095L", "isbidproductrequest_292L", "isdebitcard_729L",
    "lastactivateddate_801D", "lastapplicationdate_877D", "lastapprcommoditycat_1041M",
    "lastapprcommoditytypec_5251766M", "lastapprcredamount_781A", "lastapprdate_640D",
    "lastcancelreason_561M", "lastdelinqdate_224D", "lastdependentsnum_448L",
    "lastotherinc_902A", "lastotherlnsexpense_631A", "lastrejectcommoditycat_161M",
    "lastrejectcommodtypec_5251769M", "lastrejectcredamount_222A", "lastrejectdate_50D",
    "lastrejectreason_759M", "lastrejectreasonclient_4145040M", "lastrepayingdate_696D",
    "lastst_736L", "maininc_215A", "mastercontrelectronic_519L",
    "mastercontrexist_109L", "maxannuity_159A", "maxannuity_4075009A",
    "maxdbddpdlast1m_3658939P", "maxdbddpdtollast12m_3658940P", "maxdbddpdtollast6m_4187119P",
    "maxdebt4_972A", "maxdpdfrom6mto36m_3546853P", "maxdpdinstldate_3546855D",
    "maxdpdinstlnum_3546846P", "maxdpdlast12m_727P", "maxdpdlast24m_143P",
    "maxdpdlast3m_392P", "maxdpdlast6m_474P", "maxdpdlast9m_1059P",
    "maxdpdtolerance_374P", "maxinstallast24m_3658928A", "maxlnamtstart6m_4525199A",
    "maxoutstandbalancel12m_4187113A", "maxpmtlast3m_4525190A", "mindbddpdlast24m_3658935P",
    "mindbdtollast24m_4525191P", "mobilephncnt_593L", "monthsannuity_845L",
    "numactivecreds_622L", "numactivecredschannel_414L", "numactiverelcontr_750L",
    "numcontrs3months_479L", "numincomingpmts_3546848L", "numinstlallpaidearly3d_817L",
    "numinstls_657L", "numinstlsallpaid_934L", "numinstlswithdpd10_728L",
    "numinstlswithdpd5_4187116L", "numinstlswithoutdpd_562L", "numinstmatpaidtearly2d_4499204L",
    "numinstpaid_4499208L", "numinstpaidearly_338L", "numinstpaidearly3d_3546850L",
    "numinstpaidearly3dest_4493216L", "numinstpaidearly5d_1087L", "numinstpaidearly5dest_4493211L",
    "numinstpaidearly5dobd_4499205L", "numinstpaidearlyest_4493214L", "numinstpaidlastcontr_4325080L",
    "numinstpaidlate1d_3546852L", "numinstregularpaid_973L", "numinstregularpaidest_4493210L",
    "numinsttopaygr_769L", "numinsttopaygrest_4493213L", "numinstunpaidmax_3546851L",
    "numinstunpaidmaxest_4493212L", "numnotactivated_1143L", "numpmtchanneldd_318L",
    "numrejects9m_859L", "opencred_647L", "paytype_783L",
    "paytype1st_925L", "payvacationpostpone_4187118D", "pctinstlsallpaidearl3d_427L",
    "pctinstlsallpaidlat10d_839L", "pctinstlsallpaidlate1d_3546856L", "pctinstlsallpaidlate4d_3546849L",
    "pctinstlsallpaidlate6d_3546844L", "pmtnum_254L", "posfpd10lastmonth_333P",
    "posfpd30lastmonth_3976960P", "posfstqpd30lastmonth_3976962P", "previouscontdistrict_112M",
    "price_1097A", "sellerplacecnt_915L", "sellerplacescnt_216L",
    "sumoutstandtotal_3546847A", "sumoutstandtotalest_4493215A", "totaldebt_9A",
    "totalsettled_863A", "totinstallast1m_4525188A", "twobodfilling_608L",
    "typesuite_864L", "validfrom_1069D",
)

# Volkova's cols_drop in _process_static0 (also include = 0 in features_2.csv)
DROPPED = ["previouscontdistrict_112M", "isdebitcard_729L", "paytype1st_925L", "paytype_783L"]

# Volkova's first with_columns: new column -> (primary, fallback)
FALLBACKS = {
    "sumoutstandtotal_3546847A_2": ("sumoutstandtotal_3546847A_1", "sumoutstandtotalest_4493215A_1"),
    "maininc_215A_2": ("maininc_215A_1", INCOME_FALLBACK),
}
# Volkova's second with_columns, same order: new column -> (kind, left, right)
# "ratio" = left / right, "diff" = left - right.
# Note: sumoutstandtotal_3546847A_2 is written again here, so the fallback above
# never reaches the model.
FORMULAS = {
    "totalsettled_863A_2": ("ratio", "totalsettled_863A_1", "totaldebt_9A_1"),
    "annuity_780A_2": ("ratio", "annuity_780A_1", "maininc_215A_2"),
    "annuitynextmonth_57A_2": ("ratio", "annuitynextmonth_57A_1", INCOME_FALLBACK),
    "annuity_780A_3": ("ratio", "annuity_780A_1", "credamount_770A_1"),
    "annuitynextmonth_57A_3": ("ratio", "annuitynextmonth_57A_1", "credamount_770A_1"),
    "annuity_780A_4": ("ratio", "annuity_780A_1", "avginstallast24m_3658937A_1"),
    "avglnamtstart24m_4525187A_2": ("ratio", "avglnamtstart24m_4525187A_1", "credamount_770A_1"),
    "avgoutstandbalancel6m_4187114A_2": ("ratio", "avgoutstandbalancel6m_4187114A_1", "credamount_770A_1"),
    "annuity_780A_5": ("ratio", "annuity_780A_1", "avgpmtlast12m_4525200A_1"),
    "credamount_770A_2": ("ratio", "credamount_770A_1", "maininc_215A_2"),
    "currdebt_22A_2": ("ratio", "currdebt_22A_1", "maininc_215A_2"),
    "datelastunpaid_3546854D_2": ("diff", "datelastunpaid_3546854D_1", "datefirstoffer_1144D_1"),
    "disbursedcredamount_1113A_2": ("ratio", "disbursedcredamount_1113A_1", "credamount_770A_1"),
    "downpmt_116A_2": ("ratio", "downpmt_116A_1", "credamount_770A_1"),
    "dtlastpmtallstes_4499206D_2": ("diff", "dtlastpmtallstes_4499206D_1", "datelastunpaid_3546854D_1"),
    "inittransactionamount_650A_2": ("ratio", "inittransactionamount_650A_1", "annuity_780A_1"),
    "eir_270L_2": ("diff", "eir_270L_1", "interestrate_311L_1"),
    "interestrategrace_34L_2": ("ratio", "interestrategrace_34L_1", "interestrate_311L_1"),
    "credamount_770A_3": ("ratio", "credamount_770A_1", "lastapprcredamount_781A_1"),
    "lastapprdate_640D_2": ("diff", "lastapprdate_640D_1", "lastapplicationdate_877D_1"),
    "lastrejectdate_50D_2": ("diff", "lastrejectdate_50D_1", "lastapplicationdate_877D_1"),
    "lastrepayingdate_696D_2": ("ratio", "lastrepayingdate_696D_1", "datelastunpaid_3546854D_1"),
    "maininc_215A_3": ("ratio", "maininc_215A_1", INCOME_FALLBACK),
    "annuity_780A_6": ("ratio", "annuity_780A_1", "maxannuity_159A_1"),
    "currdebt_22A_3": ("ratio", "currdebt_22A_1", "maxdebt4_972A_1"),
    "annuity_780A_7": ("ratio", "annuity_780A_1", "maxannuity_4075009A_1"),
    "annuity_780A_8": ("ratio", "annuity_780A_1", "maxinstallast24m_3658928A_1"),
    "credamount_770A_4": ("ratio", "credamount_770A_1", "maxlnamtstart6m_4525199A_1"),
    "maxoutstandbalancel12m_4187113A_2": ("ratio", "maxoutstandbalancel12m_4187113A_1", "credamount_770A_1"),
    "annuity_780A_9": ("ratio", "annuity_780A_1", "maxpmtlast3m_4525190A_1"),
    "monthsannuity_845L_2": ("ratio", "monthsannuity_845L_1", "annuity_780A_1"),
    "price_1097A_2": ("ratio", "price_1097A_1", "maininc_215A_2"),
    "sumoutstandtotal_3546847A_2": ("ratio", "sumoutstandtotal_3546847A_1", "maininc_215A_2"),
    "sumoutstandtotal_3546847A_3": ("ratio", "sumoutstandtotal_3546847A_1", "credamount_770A_1"),
    "totinstallast1m_4525188A_2": ("ratio", "totinstallast1m_4525188A_1", "credamount_770A_1"),
    "totinstallast1m_4525188A_3": ("ratio", "totinstallast1m_4525188A_1", "maininc_215A_2"),
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


cases = spark.table(CASE_TABLE)
static = spark.table(f"{BRONZE}.train_static_0")

# 1. Clean

kept = [c for c in FEATURES if c not in DROPPED]
context = cases.select(
    "case_id", "date_decision", F.col("person1_mainoccupationinc_384A").alias(INCOME_FALLBACK)
)
rows = static.select("case_id", *kept).join(context, "case_id", "inner")
types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select("case_id", INCOME_FALLBACK, *[clean(c, types[c]).alias(f"{c}_1") for c in kept])

# 2. Volkova's formulas (two with_columns steps, like her code)

rows = rows.withColumns({name: F.coalesce(F.col(a), F.col(b)) for name, (a, b) in FALLBACKS.items()})
before_formulas = rows   # for the fallback report
rows = rows.withColumns({
    name: ratio(F.col(a), F.col(b)) if kind == "ratio" else F.col(a) - F.col(b)
    for name, (kind, a, b) in FORMULAS.items()
})

# 3. One row per case

rows = rows.drop(INCOME_FALLBACK)
rows = rows.select(
    "case_id",
    *[F.col(c).alias(f"{c}_{SOURCE}") for c in rows.columns if c != "case_id"],
    F.lit(True).alias(f"has_{SOURCE}"),
)
features = cases.select("case_id").join(rows, "case_id", "left").fillna(False, subset=[f"has_{SOURCE}"])

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
    F.count(F.when(F.col(f"has_{SOURCE}"), 1)).alias("with_static0"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

coverage = pd.DataFrame([
    ("static0_rows_in_bronze", static.count()),
    ("cases", out.cases),
    ("cases_with_static0", out.with_static0),
], columns=["metric", "value"])

metrics = [F.count("*").alias("rows")]
for name, (a, b) in FALLBACKS.items():
    metrics += [
        F.count(F.when(F.col(a).isNull(), 1)).alias(f"{name}_before"),
        F.count(F.when(F.col(name).isNull(), 1)).alias(f"{name}_after"),
    ]
counts = before_formulas.agg(*metrics).first().asDict()
fallbacks = pd.DataFrame(
    [(name, f"{a} -> {b}", name != "sumoutstandtotal_3546847A_2", counts["rows"],
      counts[f"{name}_before"], counts[f"{name}_after"])
     for name, (a, b) in FALLBACKS.items()],
    columns=["derived_field", "source_order", "used_by_model", "rows",
             "missing_before_fallback", "missing_after_fallback"],
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
