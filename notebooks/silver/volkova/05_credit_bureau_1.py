"""Silver 05: credit bureau contracts (train_credit_bureau_a_1 + b_1) -> one row per case.

Feature logic is Volkova's creditbureau1 processing in DataProcessor2 (commit d5e228c):
_process_creditbureau1 (row formulas), _handle_contracts_cb (active / closed
views) and FeatureEngineer.transform (aggregations). Aggregations and the
active/closed/common role of each column come from features/features_2.csv.
https://github.com/evgeniavolkova/kagglehomecredit2024/blob/d5e228cdf7e9bf81d10805cff205b473a94c2063/homecredit/data/data_processor_2.py

Each bureau row has an active contract and a closed contract side by side.
Volkova stacks them into two "views" and aggregates per case.

Bug in her code that we keep: the "_closed" group filters on active rows too
(df.filter(pl.col("active")) twice), so "_closed" = "_active" and the "_div_"
columns are 1 or null. Set CLOSED_FILTER_AS_AUTHOR = False to use closed rows.

Output: silver_volkova.train_credit_bureau_1_features + reports/05_credit_bureau_1/*.csv
"""

import math
import os

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, DoubleType, FloatType, NumericType, StringType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
BRONZE = f"{CATALOG}.bronze"
SCHEMA = f"{CATALOG}.silver_volkova"
STORAGE_ROOT = "abfss://homecredit@sthomecreditds202.dfs.core.windows.net/silver/volkova"

CASE_TABLE = f"{SCHEMA}.train_case_sources"
OUTPUT_TABLE = f"{SCHEMA}.train_credit_bureau_1_features"
OUTPUT_PATH = f"{STORAGE_ROOT}/tables/train_credit_bureau_1_features"
REPORT_DIR = "/Volumes/dbw_home_credit_audit/silver_volkova/files/reports/05_credit_bureau_1"

SOURCE = "creditbureau1"
A1_TABLE, B1_TABLE = "credit_bureau_a_1", "credit_bureau_b_1"
MISSING_CATEGORY = "a55475b1"
SORT_COLUMN = "dateofcredend_all_1"   # Volkova sorts the views by this
WINDOW_DAYS = 365 * 5
WINDOW_SUFFIX = "_dateofcredend_all_5y"
CLOSED_FILTER_AS_AUTHOR = True

# From Volkova's features_2.csv: column -> (aggregations, view).
# view = active / closed / common for raw columns, "all" for merged ones.
CONFIG = {
    "annualeffectiverate_199L": (('min', 'max', 'mean', 'last'), "closed"),
    "annualeffectiverate_63L": (('min', 'max', 'mean', 'last'), "active"),
    "annualeffectiverate_all": (('min', 'max', 'mean', 'last'), "all"),
    "classificationofcontr_13M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "classificationofcontr_400M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "classificationofcontr_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "contractst_545M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "contractst_964M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "contractst_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "contractsum_5085717L": (('min', 'max', 'range'), "common"),
    "credlmt_230A": (('min', 'max', 'mean', 'sum'), "closed"),
    "credlmt_935A": (('min', 'max', 'mean', 'sum', 'rangelastfirst'), "active"),
    "credlmt_all": (('min', 'max', 'mean', 'sum', 'rangelastfirst'), "all"),
    "dateofcredend_289D": (('min', 'max', 'mean'), "active"),
    "dateofcredend_353D": (('min', 'max', 'mean'), "closed"),
    "dateofcredend_all": (('min', 'max', 'mean'), "all"),
    "dateofcredstart_181D": (('min', 'max', 'mean'), "active"),
    "dateofcredstart_739D": (('min', 'max', 'mean'), "closed"),
    "dateofcredstart_all": (('min', 'max', 'mean'), "all"),
    "dateofrealrepmt_138D": (('min', 'max'), "closed"),
    "debtoutstand_525A": (('min', 'max', 'mean', 'sum'), "active"),
    "debtoutstand_all": (('min', 'max', 'mean', 'sum'), "all"),
    "debtoverdue_47A": (('max', 'mean', 'sum', 'meannonzero'), "active"),
    "debtoverdue_all": (('max', 'mean', 'sum', 'meannonzero'), "all"),
    "description_351M": (('nunique', 'nuniquetotal', 'mode', 'last'), "common"),
    "dpdmax_139P": (('min', 'max'), "active"),
    "dpdmax_757P": (('min', 'max'), "closed"),
    "dpdmax_all": (('min', 'max', 'mean'), "all"),
    "dpdmaxdatemonth_442T": (('min', 'max', 'mean'), "closed"),
    "dpdmaxdatemonth_89T": (('min', 'max', 'mean'), "active"),
    "dpdmaxdatemonth_all": (('min', 'max', 'mean'), "all"),
    "dpdmaxdateyear_596T": ((), "active"),
    "dpdmaxdateyear_896T": ((), "closed"),
    "dpdmaxdateyear_all": ((), "all"),
    "financialinstitution_382M": (('nunique', 'nuniquetotal'), "closed"),
    "financialinstitution_591M": (('nunique', 'nuniquetotal'), "active"),
    "financialinstitution_all": (('nunique', 'nuniquetotal'), "all"),
    "instlamount_768A": (('min', 'max'), "active"),
    "instlamount_852A": (('min', 'max'), "closed"),
    "instlamount_all": (('min', 'max'), "all"),
    "interestrate_508L": (('min', 'max', 'last'), "closed"),
    "lastupdate_1112D": (('min', 'max', 'mean'), "active"),
    "lastupdate_388D": (('min', 'max', 'mean'), "closed"),
    "lastupdate_all": (('min', 'max', 'mean'), "all"),
    "monthlyinstlamount_332A": (('min', 'max'), "active"),
    "monthlyinstlamount_674A": (('min', 'max'), "closed"),
    "monthlyinstlamount_all": (('min', 'max'), "all"),
    "nominalrate_281L": (('min', 'max', 'last', 'mean'), "active"),
    "nominalrate_498L": (('min', 'max', 'last', 'mean'), "closed"),
    "nominalrate_all": (('min', 'max', 'last', 'mean'), "all"),
    "numberofcontrsvalue_258L": (('min', 'max', 'last', 'mean'), "active"),
    "numberofcontrsvalue_358L": (('min', 'max', 'last', 'mean'), "closed"),
    "numberofcontrsvalue_all": (('min', 'max', 'last', 'mean'), "all"),
    "numberofinstls_229L": (('min', 'max', 'last', 'mean'), "closed"),
    "numberofinstls_320L": (('min', 'max', 'last', 'mean'), "active"),
    "numberofinstls_all": (('min', 'max', 'last', 'mean'), "all"),
    "numberofoutstandinstls_520L": (('min', 'max', 'last'), "closed"),
    "numberofoutstandinstls_59L": (('min', 'max', 'last'), "active"),
    "numberofoutstandinstls_all": (('min', 'max', 'last'), "all"),
    "numberofoverdueinstlmax_1039L": (('min', 'max', 'last', 'mean'), "active"),
    "numberofoverdueinstlmax_1151L": (('min', 'max', 'last', 'mean'), "closed"),
    "numberofoverdueinstlmax_all": (('min', 'max', 'last', 'mean'), "all"),
    "numberofoverdueinstlmaxdat_148D": (('min', 'max'), "closed"),
    "numberofoverdueinstlmaxdat_641D": (('min', 'max'), "active"),
    "numberofoverdueinstlmaxdat_all": (('min', 'max'), "all"),
    "numberofoverdueinstls_725L": (('min', 'max', 'last'), "active"),
    "numberofoverdueinstls_834L": (('min', 'max', 'last'), "closed"),
    "numberofoverdueinstls_all": (('min', 'max', 'last'), "all"),
    "outstandingamount_354A": (('min', 'max', 'mean'), "closed"),
    "outstandingamount_362A": (('min', 'max', 'mean'), "active"),
    "outstandingamount_all": (('min', 'max', 'mean'), "all"),
    "overdueamount_31A": (('mean', 'max'), "closed"),
    "overdueamount_659A": (('mean', 'max'), "active"),
    "overdueamount_all": (('mean', 'max'), "all"),
    "overdueamountmax_155A": (('min', 'max'), "active"),
    "overdueamountmax_35A": (('min', 'max'), "closed"),
    "overdueamountmax_all": (('min', 'max'), "all"),
    "overdueamountmax2_14A": (('min', 'max', 'mean'), "active"),
    "overdueamountmax2_398A": (('min', 'max', 'mean'), "closed"),
    "overdueamountmax2_all": (('min', 'max', 'mean'), "all"),
    "overdueamountmax2date_1002D": (('min', 'max', 'mean'), "closed"),
    "overdueamountmax2date_1142D": (('min', 'max', 'mean'), "active"),
    "overdueamountmax2date_all": (('min', 'max', 'mean'), "all"),
    "overdueamountmaxdatemonth_284T": (('min', 'max'), "closed"),
    "overdueamountmaxdatemonth_365T": (('min', 'max'), "active"),
    "overdueamountmaxdatemonth_all": (('min', 'max'), "all"),
    "overdueamountmaxdateyear_2T": ((), "active"),
    "overdueamountmaxdateyear_994T": ((), "closed"),
    "overdueamountmaxdateyear_all": ((), "all"),
    "periodicityofpmts_1102L": (('min', 'max', 'last'), "closed"),
    "periodicityofpmts_837L": (('min', 'max', 'last'), "active"),
    "periodicityofpmts_all": (('min', 'max', 'last'), "all"),
    "prolongationcount_1120L": (('min', 'max', 'mean'), "closed"),
    "prolongationcount_599L": (('min', 'max', 'mean'), "active"),
    "prolongationcount_all": (('min', 'max', 'mean'), "all"),
    "purposeofcred_426M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "purposeofcred_874M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "purposeofcred_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "refreshdate_3813885D": (('min', 'max'), "common"),
    "residualamount_488A": (('min', 'max', 'mean'), "closed"),
    "residualamount_856A": (('min', 'max', 'mean'), "active"),
    "residualamount_all": (('min', 'max', 'mean'), "all"),
    "subjectrole_182M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "subjectrole_93M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "subjectrole_all": (('nunique', 'nuniquetotal', 'mode', 'last'), "all"),
    "totalamount_6A": (('min', 'max', 'mean'), "closed"),
    "totalamount_996A": (('min', 'max', 'mean'), "active"),
    "totalamount_all": (('min', 'max', 'mean'), "all"),
    "totaldebtoverduevalue_178A": (('min', 'max', 'mean'), "active"),
    "totaldebtoverduevalue_718A": (('min', 'max', 'mean'), "closed"),
    "totaldebtoverduevalue_all": (('min', 'max', 'mean'), "all"),
    "totaloutstanddebtvalue_39A": (('min', 'max', 'sum', 'mean'), "active"),
    "totaloutstanddebtvalue_668A": (('min', 'max', 'sum', 'mean'), "closed"),
    "totaloutstanddebtvalue_all": (('min', 'max', 'sum', 'mean'), "all"),
    "amount_1115A": ((), "active"),
    "classificationofcontr_1114M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "contractdate_551D": ((), "active"),
    "contractmaturitydate_151D": ((), "active"),
    "contractst_516M": (('nunique', 'nuniquetotal', 'mode', 'last'), "common"),
    "contracttype_653M": (('nunique', 'nuniquetotal'), "common"),
    "credlmt_1052A": ((), "active"),
    "credlmt_228A": ((), "closed"),
    "credlmt_3940954A": ((), "active"),
    "credor_3940957M": (('nunique', 'nuniquetotal'), "common"),
    "credquantity_1099L": ((), "common"),
    "credquantity_984L": ((), "closed"),
    "debtpastduevalue_732A": ((), "active"),
    "debtvalue_227A": ((), "active"),
    "dpd_550P": ((), "active"),
    "dpd_733P": ((), "closed"),
    "dpd_all": ((), "active"),
    "dpdmax_851P": ((), "active"),
    "dpdmaxdatemonth_804T": (('min', 'max', 'mean'), "active"),
    "dpdmaxdateyear_742T": ((), "active"),
    "installmentamount_644A": ((), "closed"),
    "installmentamount_833A": ((), "active"),
    "instlamount_892A": ((), "active"),
    "interesteffectiverate_369L": ((), "active"),
    "interestrateyearly_538L": ((), "active"),
    "lastupdate_260D": ((), "active"),
    "maxdebtpduevalodued_3940955A": ((), "common"),
    "numberofinstls_810L": ((), "active"),
    "overdueamountmax_950A": ((), "active"),
    "overdueamountmaxdatemonth_494T": (('min', 'max'), "active"),
    "overdueamountmaxdateyear_432T": ((), "active"),
    "periodicityofpmts_997L": ((), "active"),
    "periodicityofpmts_997M": ((), "active"),
    "pmtdaysoverdue_1135P": ((), "active"),
    "pmtmethod_731M": (('nunique', 'nuniquetotal'), "active"),
    "pmtnumpending_403L": ((), "active"),
    "purposeofcred_722M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "residualamount_1093A": ((), "closed"),
    "residualamount_127A": ((), "active"),
    "residualamount_3940956A": ((), "active"),
    "subjectrole_326M": (('nunique', 'nuniquetotal', 'mode', 'last'), "active"),
    "subjectrole_43M": (('nunique', 'nuniquetotal', 'mode', 'last'), "closed"),
    "totalamount_503A": ((), "active"),
    "totalamount_881A": ((), "closed"),
}

# Volkova renames b1 columns to their a1 name (new_name in features_2.csv)
B1_RENAME = {
    "classificationofcontr_1114M": "classificationofcontr_13M",
    "contractdate_551D": "dateofcredstart_181D",
    "contractmaturitydate_151D": "dateofcredend_289D",
    "contractst_516M": "contractst_545M",
    "credlmt_228A": "credlmt_230A",
    "credlmt_3940954A": "credlmt_935A",
    "credquantity_1099L": "numberofcontrsvalue_258L",
    "credquantity_984L": "numberofcontrsvalue_358L",
    "debtpastduevalue_732A": "overdueamount_659A",
    "debtvalue_227A": "outstandingamount_362A",
    "dpdmax_851P": "dpdmax_139P",
    "dpdmaxdatemonth_804T": "dpdmaxdatemonth_89T",
    "dpdmaxdateyear_742T": "dpdmaxdateyear_596T",
    "installmentamount_644A": "instlamount_852A",
    "installmentamount_833A": "instlamount_768A",
    "interesteffectiverate_369L": "nominalrate_281L",
    "interestrateyearly_538L": "annualeffectiverate_63L",
    "lastupdate_260D": "lastupdate_1112D",
    "numberofinstls_810L": "numberofinstls_320L",
    "overdueamountmax_950A": "overdueamountmax_155A",
    "overdueamountmaxdatemonth_494T": "overdueamountmaxdatemonth_284T",
    "overdueamountmaxdateyear_432T": "overdueamountmaxdateyear_2T",
    "purposeofcred_722M": "purposeofcred_426M",
    "residualamount_1093A": "residualamount_488A",
    "residualamount_127A": "residualamount_856A",
    "subjectrole_326M": "subjectrole_182M",
    "subjectrole_43M": "subjectrole_93M",
    "totalamount_503A": "totalamount_996A",
    "totalamount_881A": "totalamount_6A",
}
A1_COLUMNS = (
    "annualeffectiverate_199L", "annualeffectiverate_63L", "classificationofcontr_13M",
    "classificationofcontr_400M", "contractst_545M", "contractst_964M",
    "contractsum_5085717L", "credlmt_230A", "credlmt_935A",
    "dateofcredend_289D", "dateofcredend_353D", "dateofcredstart_181D",
    "dateofcredstart_739D", "dateofrealrepmt_138D", "debtoutstand_525A",
    "debtoverdue_47A", "description_351M", "dpdmax_139P",
    "dpdmax_757P", "dpdmaxdatemonth_442T", "dpdmaxdatemonth_89T",
    "dpdmaxdateyear_596T", "dpdmaxdateyear_896T", "financialinstitution_382M",
    "financialinstitution_591M", "instlamount_768A", "instlamount_852A",
    "interestrate_508L", "lastupdate_1112D", "lastupdate_388D",
    "monthlyinstlamount_332A", "monthlyinstlamount_674A", "nominalrate_281L",
    "nominalrate_498L", "numberofcontrsvalue_258L", "numberofcontrsvalue_358L",
    "numberofinstls_229L", "numberofinstls_320L", "numberofoutstandinstls_520L",
    "numberofoutstandinstls_59L", "numberofoverdueinstlmax_1039L", "numberofoverdueinstlmax_1151L",
    "numberofoverdueinstlmaxdat_148D", "numberofoverdueinstlmaxdat_641D", "numberofoverdueinstls_725L",
    "numberofoverdueinstls_834L", "outstandingamount_354A", "outstandingamount_362A",
    "overdueamount_31A", "overdueamount_659A", "overdueamountmax_155A",
    "overdueamountmax_35A", "overdueamountmax2_14A", "overdueamountmax2_398A",
    "overdueamountmax2date_1002D", "overdueamountmax2date_1142D", "overdueamountmaxdatemonth_284T",
    "overdueamountmaxdatemonth_365T", "overdueamountmaxdateyear_2T", "overdueamountmaxdateyear_994T",
    "periodicityofpmts_1102L", "periodicityofpmts_837L", "prolongationcount_1120L",
    "prolongationcount_599L", "purposeofcred_426M", "purposeofcred_874M",
    "refreshdate_3813885D", "residualamount_488A", "residualamount_856A",
    "subjectrole_182M", "subjectrole_93M", "totalamount_6A",
    "totalamount_996A", "totaldebtoverduevalue_178A", "totaldebtoverduevalue_718A",
    "totaloutstanddebtvalue_39A", "totaloutstanddebtvalue_668A",
)
B1_COLUMNS = (
    "amount_1115A", "classificationofcontr_1114M", "contractdate_551D",
    "contractmaturitydate_151D", "contractst_516M", "contracttype_653M",
    "credlmt_1052A", "credlmt_228A", "credlmt_3940954A",
    "credor_3940957M", "credquantity_1099L", "credquantity_984L",
    "debtpastduevalue_732A", "debtvalue_227A", "dpd_550P",
    "dpd_733P", "dpdmax_851P", "dpdmaxdatemonth_804T",
    "dpdmaxdateyear_742T", "installmentamount_644A", "installmentamount_833A",
    "instlamount_892A", "interesteffectiverate_369L", "interestrateyearly_538L",
    "lastupdate_260D", "maxdebtpduevalodued_3940955A", "numberofinstls_810L",
    "overdueamountmax_950A", "overdueamountmaxdatemonth_494T", "overdueamountmaxdateyear_432T",
    "periodicityofpmts_997L", "periodicityofpmts_997M", "pmtdaysoverdue_1135P",
    "pmtmethod_731M", "pmtnumpending_403L", "purposeofcred_722M",
    "residualamount_1093A", "residualamount_127A", "residualamount_3940956A",
    "subjectrole_326M", "subjectrole_43M", "totalamount_503A",
    "totalamount_881A",
)

# Volkova's cols_drop in _process_creditbureau1.
# The year/month parts are used in the date formulas first, then dropped.
DATE_PARTS = [
    "dpdmaxdateyear_896T", "dpdmaxdateyear_596T", "overdueamountmaxdateyear_994T",
    "overdueamountmaxdateyear_2T", "dpdmaxdatemonth_442T", "dpdmaxdatemonth_89T",
    "overdueamountmaxdatemonth_365T", "overdueamountmaxdatemonth_284T",
]
UNUSED = [
    "credlmt_1052A", "instlamount_892A", "periodicityofpmts_997M",
    "periodicityofpmts_997L", "residualamount_3940956A",
]

# Formulas from Volkova's _process_creditbureau1
# new column -> (year column, month column): first day of that month as day offset
MONTH_DATES = {
    "dpdmaxdatemonth_442T_1": ("dpdmaxdateyear_896T", "dpdmaxdatemonth_442T"),
    "dpdmaxdatemonth_89T_1": ("dpdmaxdateyear_596T", "dpdmaxdatemonth_89T"),
    "overdueamountmaxdatemonth_365T_1": ("overdueamountmaxdateyear_2T", "overdueamountmaxdatemonth_365T"),
    "overdueamountmaxdatemonth_284T_1": ("overdueamountmaxdateyear_994T", "overdueamountmaxdatemonth_284T"),
}
# new column -> (primary, fallback)
TOTAL_FALLBACKS = {
    "totalamount_996A_2": ("totalamount_996A_1", "credlmt_935A_1"),
    "totalamount_6A_2": ("totalamount_6A_1", "credlmt_230A_1"),
}
# new column -> (numerator, denominator)
RATIOS = {
    "numberofoverdueinstlmax_1039L_2": ("numberofoverdueinstlmax_1039L_1", "numberofinstls_320L_1"),
    "numberofoverdueinstlmax_1151L_2": ("numberofoverdueinstlmax_1151L_1", "numberofinstls_229L_1"),
    "numberofoutstandinstls_520L_2": ("numberofoutstandinstls_520L_1", "numberofinstls_229L_1"),
    "numberofoutstandinstls_59L_2": ("numberofoutstandinstls_59L_1", "numberofinstls_320L_1"),
    "dpdmax_139P_2": ("dpdmax_139P_1", "numberofoverdueinstls_725L_1"),
    "dpdmax_757P_2": ("dpdmax_757P_1", "numberofoverdueinstls_834L_1"),
    "dateofcredend_289D_3": ("dateofcredend_289D_1", "numberofcontrsvalue_258L_1"),
    "dateofcredend_353D_3": ("dateofcredend_353D_1", "numberofcontrsvalue_358L_1"),
    "debtoverdue_47A_2": ("debtoverdue_47A_1", "debtoutstand_525A_1"),
    "outstandingamount_362A_2": ("outstandingamount_362A_1", "totalamount_996A_2"),
    "overdueamount_31A_2": ("overdueamount_31A_1", "totalamount_6A_2"),
    "overdueamount_659A_2": ("overdueamount_659A_1", "totalamount_996A_2"),
}
# new column -> (left, right): left - right
DIFFERENCES = {
    "dateofcredend_289D_2": ("dateofcredend_289D_1", "dateofcredstart_181D_1"),
    "dateofcredend_353D_2": ("dateofcredend_353D_1", "dateofcredstart_739D_1"),
}

HELPERS = ("case_id", "num_group1", "_source", "_active", "_seq")


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
    # polars gives +/-inf (NaN for 0/0) when dividing by zero, Spark gives null.
    # Copy polars so the aggregations match; inf and NaN are set to null at the end.
    a, b = a.cast("double"), b.cast("double")
    return (
        F.when(a.isNull() | b.isNull(), F.lit(None).cast("double"))
        .when(b != 0, a / b)
        .when(F.isnan(a) | (a == 0), F.lit(math.nan))
        .when(a > 0, F.lit(math.inf))
        .otherwise(F.lit(-math.inf))
    )


def base_feature(column):
    # dpdmax_139P_2 -> dpdmax_139P (derived columns use the config of their base column)
    return column.rsplit("_", 1)[0]


def chunk_number(column):
    # train_credit_bureau_a_1_3.parquet -> 3 (files without a number -> 0)
    return F.coalesce(F.regexp_extract(F.col(column), r"_(\d+)\.parquet$", 1).try_cast("int"), F.lit(0))


def month_offset(year, month):
    # Volkova: pl.date(year, month, 1) - date_decision in days
    year, month = F.col(year).cast("double").cast("int"), F.col(month).cast("double").cast("int")
    valid = year.isNotNull() & month.between(1, 12)
    return F.when(valid, F.datediff(F.make_date(year, month, F.lit(1)), F.col("date_decision")))


def agg_expr(column, op):
    # Volkova's Utils.agg_process in Spark. last/first follow the row order in _seq.
    value = F.col(column)
    position = F.when(value.isNotNull(), F.col("_seq"))
    first = F.min_by(value, position)
    last = F.max_by(value, position)
    nunique = F.size(F.collect_set(value)) + F.max(value.isNull().cast("int"))  # polars counts null as a value
    ops = {
        "min": F.min(value),
        "max": F.max(value),
        "mean": F.avg(value),
        "sum": F.coalesce(F.sum(value), F.lit(0)),   # polars: sum of only nulls = 0
        "range": F.max(value) - F.min(value),
        "last": last,
        "rangelastfirst": last - first,
        "meannonzero": F.avg(F.when(value != 0, value)),
        "nunique": nunique,
        "nuniquetotal": ratio(nunique, F.count(value)),
        "mode": F.mode(value, True),
    }
    return ops[op]


def aggregate(rows, columns, suffix):
    # Volkova's _process_aggregations_all; for credit bureau she also adds the source file
    exprs = [
        agg_expr(c, op).alias(f"{c}_{op}{suffix}")
        for c in columns
        for op in CONFIG[base_feature(c)][0]
    ]
    exprs.append(F.count("case_id").alias(f"case_id_1_count_{suffix}"))
    exprs.append(F.max("_source").alias(f"source_1_{suffix}"))
    return rows.groupBy("case_id").agg(*exprs)


cases = spark.table(CASE_TABLE)
dates = cases.select("case_id", "date_decision")

# 1. Load a1 and b1 (b1 columns renamed to the a1 names)

a1 = spark.table(f"{BRONZE}.train_{A1_TABLE}")
b1 = spark.table(f"{BRONZE}.train_{B1_TABLE}")
a1_types = {f.name: f.dataType for f in a1.schema.fields}

a1_rows = a1.select(
    "case_id", "num_group1", F.lit(A1_TABLE).alias("_source"), chunk_number("_source_file").alias("_chunk"),
    *[c for c in A1_COLUMNS if c not in UNUSED],
).join(dates, "case_id", "inner")

b1_columns = []
for raw in B1_COLUMNS:
    name = B1_RENAME.get(raw, raw)
    if name in UNUSED:
        continue
    # cast to the a1 type so the union works
    value = F.col(raw).cast(a1_types[name]) if name in a1_types else F.col(raw)
    b1_columns.append(value.alias(name))
b1_rows = b1.select(
    "case_id", "num_group1", F.lit(B1_TABLE).alias("_source"), F.lit(0).alias("_chunk"), *b1_columns,
).join(dates, "case_id", "inner")

# Volkova aggregates each file on its own and keeps the first result per case:
# a case in a1 never uses b1, and a case split over a1 chunk files only uses the first chunk.
first_chunk = a1_rows.groupBy("case_id").agg(F.min("_chunk").alias("_first_chunk"))
a1_used = a1_rows.join(first_chunk, "case_id").where(F.col("_chunk") == F.col("_first_chunk")).drop("_first_chunk")
b1_used = b1_rows.join(a1_rows.select("case_id").distinct(), "case_id", "left_anti")
rows = a1_used.unionByName(b1_used, allowMissingColumns=True).drop("_chunk")

# 2. Clean (year/month parts keep their raw name, Volkova does not rename them)

types = {f.name: f.dataType for f in rows.schema.fields}
rows = rows.select(
    "case_id", "num_group1", "_source", "date_decision",
    *[clean(c, types[c]).alias(c if c in DATE_PARTS else f"{c}_1") for c in rows.columns if c in CONFIG],
)

# 3. Row formulas from Volkova's _process_creditbureau1

rows = rows.withColumns({
    **{name: F.coalesce(F.col(a), F.col(b)) for name, (a, b) in TOTAL_FALLBACKS.items()},
    **{name: month_offset(y, m) for name, (y, m) in MONTH_DATES.items()},
})
before_formulas = rows   # for the fallback report
rows = rows.withColumns({
    **{name: ratio(F.col(a), F.col(b)) for name, (a, b) in RATIOS.items()},
    **{name: F.col(a) - F.col(b) for name, (a, b) in DIFFERENCES.items()},
}).drop(*DATE_PARTS, "date_decision")

# 4. Volkova's _handle_contracts_cb: one view with the active columns, one with
# the closed columns. An active/closed pair is renamed to <name>_all_<n>.

columns = [c for c in rows.columns if base_feature(c) in CONFIG]
views = {"active": [], "closed": [], "common": []}
for c in columns:
    for view in views:
        if view in CONFIG[base_feature(c)][1]:
            views[view].append(c)

groups = {}
for c in views["active"] + views["closed"]:
    parts = c.split("_")
    groups.setdefault("_".join(parts[:-2]) + "_all_" + parts[-1], []).append(c)
rename = {c: merged for merged, names in groups.items() if len(names) >= 2 for c in names}

# if the two columns of a pair have different types, use one type for both
types = {f.name: f.dataType for f in rows.schema.fields}
merged_type = {}
for merged, names in groups.items():
    kinds = {types[c] for c in names}
    if len(names) >= 2 and len(kinds) > 1:
        merged_type[merged] = StringType() if any(isinstance(k, StringType) for k in kinds) else DoubleType()


def make_view(names, active):
    renamed = []
    for c in names:
        merged = rename.get(c, c)
        value = F.col(c).cast(merged_type[merged]) if merged in merged_type else F.col(c)
        renamed.append(value.alias(merged))
    return rows.select("case_id", "num_group1", "_source", *views["common"], *renamed,
                       F.lit(active).alias("_active"))


long = make_view(views["active"], True).unionByName(make_view(views["closed"], False), allowMissingColumns=True)
# Volkova: df.sort("dateofcredend_all_1") (nulls first). Ties: active view first, then num_group1.
order = Window.partitionBy("case_id").orderBy(
    F.col(SORT_COLUMN).asc_nulls_first(), F.col("_active").desc(), F.col("num_group1"))
long = long.withColumn("_seq", F.row_number().over(order))

# 5. Aggregations, the groups are from Volkova's FeatureEngineer.transform

types = {f.name: f.dataType for f in long.schema.fields}
columns = [c for c in long.columns if c not in HELPERS]
numeric = [c for c in columns if isinstance(types[c], NumericType)]

active = long.where(F.col("_active"))
# Volkova's bug: the closed group also uses filter(active)
closed = long.where(F.col("_active") if CLOSED_FILTER_AS_AUTHOR else ~F.col("_active"))
recent = long.where(F.col(SORT_COLUMN) >= -WINDOW_DAYS)

active_agg = aggregate(active, numeric, "_active")
bureau = aggregate(long, columns, "")
for part in (active_agg, aggregate(closed, numeric, "_closed"), aggregate(recent, columns, WINDOW_SUFFIX)):
    bureau = bureau.join(part, "case_id", "left")

# Volkova: every numeric _active aggregate divided by its _closed twin
pairs = [(c, c[: -len("active")] + "closed") for c in active_agg.columns
         if c.endswith("_active") and not c.startswith("source_")]
bureau = bureau.withColumns({f"{a}_div_{b}": ratio(F.col(a), F.col(b)) for a, b in pairs})

bureau = bureau.select(
    "case_id",
    *[F.col(c).alias(f"{c}_{SOURCE}") for c in bureau.columns if c != "case_id"],
    F.lit(True).alias(f"has_{SOURCE}"),
)

# 6. One row per case

features = cases.select("case_id").join(bureau, "case_id", "left")
count_columns = [c for c in features.columns if "case_id_" in c]
features = features.fillna(False, subset=[f"has_{SOURCE}"]).fillna(0, subset=count_columns)

# Volkova's final step: +/-inf -> null (NaN too)
floats = [f.name for f in features.schema.fields if isinstance(f.dataType, (DoubleType, FloatType))]
features = features.withColumns({
    c: F.when(~(F.isnan(c) | (F.abs(c) == math.inf)), F.col(c)) for c in floats
})

# 7. Save first (this table is slow to build), then check the saved table.
# CTAS fails if the table exists: DROP TABLE and delete the folder to rebuild.

features.createOrReplaceTempView("output_preview")
spark.sql(f"CREATE TABLE {OUTPUT_TABLE} USING DELTA LOCATION '{OUTPUT_PATH}' AS SELECT * FROM output_preview")
print(f"Created {OUTPUT_TABLE}")

written = spark.table(OUTPUT_TABLE)
source_column = f"source_1__{SOURCE}"
base_count = cases.count()
out = written.agg(
    F.count("*").alias("cases"),
    F.countDistinct("case_id").alias("distinct_cases"),
    F.count(F.when(F.col(source_column) == A1_TABLE, 1)).alias("cases_a1"),
    F.count(F.when(F.col(source_column) == B1_TABLE, 1)).alias("cases_b1"),
).first()
if out.cases != base_count or out.distinct_cases != base_count:
    raise ValueError(f"expected {base_count} unique cases, got {out}")

# 8. Reports

coverage = pd.DataFrame([
    ("a1_rows_in_bronze", a1.count()),
    ("a1_rows_used", a1_used.count()),
    ("b1_rows_in_bronze", b1.count()),
    ("b1_rows_used", b1_used.count()),
    ("cases", out.cases),
    ("cases_from_a1", out.cases_a1),
    ("cases_from_b1_only", out.cases_b1),
    ("cases_without_bureau", out.cases - out.cases_a1 - out.cases_b1),
], columns=["metric", "value"])

metrics = [F.count("*").alias("rows")]
for name, (a, b) in TOTAL_FALLBACKS.items():
    metrics += [
        F.count(F.when(F.col(a).isNull(), 1)).alias(f"{name}_before"),
        F.count(F.when(F.col(name).isNull(), 1)).alias(f"{name}_after"),
    ]
counts = before_formulas.agg(*metrics).first().asDict()
fallbacks = pd.DataFrame(
    [(name, f"{a} -> {b}", counts["rows"], counts[f"{name}_before"], counts[f"{name}_after"])
     for name, (a, b) in TOTAL_FALLBACKS.items()],
    columns=["derived_field", "source_order", "rows", "missing_before_fallback", "missing_after_fallback"],
)
print(coverage.to_string(index=False))
print(fallbacks.to_string(index=False))
print(f"{len(written.columns)} columns")
if CLOSED_FILTER_AS_AUTHOR:
    print("Note: _closed = _active (Volkova's filter), so the _div_ columns are 1 or null.")

os.makedirs(REPORT_DIR, exist_ok=True)
coverage.to_csv(f"{REPORT_DIR}/row_coverage.csv", index=False)
fallbacks.to_csv(f"{REPORT_DIR}/fallback_coverage.csv", index=False)
print(f"Saved reports to {REPORT_DIR}")
