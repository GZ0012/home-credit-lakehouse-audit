"""Quick look at bronze.train_base: schema, first rows, and case_id counts."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

SOURCE = "dbw_home_credit_audit.bronze.train_base"

df = spark.table(SOURCE)
df.printSchema()
df.show(5, truncate=False)

# total rows vs unique case_ids (should be the same) and null ids
stats = df.agg(
    F.count("*").alias("total_rows"),
    F.countDistinct("case_id").alias("unique_cases"),
    F.sum(F.col("case_id").isNull().cast("long")).alias("null_case_ids"),
).first()
print(stats)
