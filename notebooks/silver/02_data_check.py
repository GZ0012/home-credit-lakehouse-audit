# Databricks notebook source
# Null count and null ratio for every column of every bronze train table.

from pyspark.sql import functions as F

CATALOG = "dbw_home_credit_audit"
SCHEMA = "bronze"

tables = [
    r.tableName
    for r in spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}").collect()
    if r.tableName.startswith("train_")
]
print(f"Found {len(tables)} train tables: {tables}")

rows = []
for table in tables:
    print(f"Checking {table} ...")
    df = spark.table(f"{CATALOG}.{SCHEMA}.{table}")

    # one pass per table: row count + null count of each column
    exprs = [F.count("*").alias("__total_rows")] + [
        F.sum(F.col(f"`{c}`").isNull().cast("long")).alias(c) for c in df.columns
    ]
    result = df.agg(*exprs).first().asDict()
    total = result.pop("__total_rows")

    for column, null_count in result.items():
        null_count = null_count or 0
        rows.append((table, column, total, null_count, round(null_count / total, 4) if total else None))

null_report = spark.createDataFrame(
    rows, "table string, column string, total_rows long, null_count long, null_ratio double"
)
null_columns = null_report.filter("null_count > 0").orderBy(F.desc("null_ratio"))
print(f"{null_report.count()} columns, {null_columns.count()} of them have nulls")

null_columns.show(500, truncate=False)
display(null_columns)
