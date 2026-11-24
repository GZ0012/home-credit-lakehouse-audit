"""Bronze 03: load the raw parquet files and the two metadata CSVs into Delta tables.

Kaggle splits big tables into several files (train_static_0_0, train_static_0_1, ...).
Each group of files becomes one table, with two extra columns:
_source_file (which parquet file the row came from) and _ingested_at.
"""

import os
import re
from collections import defaultdict

import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, NullType, NumericType, StringType

spark = SparkSession.builder.getOrCreate()

CATALOG = "dbw_home_credit_audit"
SCHEMA = "bronze"
RAW_ROOT = "/Volumes/dbw_home_credit_audit/bronze/raw_files"
PARQUET_ROOT = f"{RAW_ROOT}/parquet_files"
METADATA_ROOT = f"{RAW_ROOT}/metadata"

ONLY_TABLES = None   # e.g. {"train_base"} to build only some tables; None = all

# train_credit_bureau_a_2_0 ... _10 -> train_credit_bureau_a_2
# train_static_0_0, _1             -> train_static_0
# files without a chunk number keep their name (train_applprev_2, train_static_cb_0)
CHUNK_PATTERN = re.compile(r"^(.*_[012])_(\d+)$")


def group_parquet_files():
    groups = defaultdict(list)
    for split in ["train", "test"]:
        for filename in sorted(os.listdir(f"{PARQUET_ROOT}/{split}")):
            if not filename.endswith(".parquet"):
                continue
            stem = filename[: -len(".parquet")]
            match = CHUNK_PATTERN.match(stem)
            table = match.group(1) if match else stem
            groups[table].append(f"{PARQUET_ROOT}/{split}/{filename}")
    return groups


def resolve_types(dataframes, reference=None):
    # The chunks of one table do not always have the same column types:
    # a column can be all null (void type) in one chunk, or int in one and double in another.
    resolved = {}
    for df in dataframes:
        for field in df.schema.fields:
            current = resolved.get(field.name)
            if isinstance(field.dataType, NullType):
                resolved.setdefault(field.name, None)
                continue
            if current is None:
                resolved[field.name] = field.dataType
            elif current != field.dataType:
                both_numeric = isinstance(current, NumericType) and isinstance(field.dataType, NumericType)
                resolved[field.name] = DoubleType() if both_numeric else StringType()

    # test tables use the train types so both sides match
    if reference:
        for name in resolved:
            if name in reference:
                resolved[name] = reference[name]

    # Delta has no void type: columns that are null in every chunk become string
    return {name: dtype if dtype is not None else StringType() for name, dtype in resolved.items()}


def read_and_union(files, reference=None):
    parts = [spark.read.parquet(path).withColumn("_source_file", F.col("_metadata.file_name")) for path in files]
    data_columns = [c for c in parts[0].columns if c != "_source_file"]
    types = resolve_types([p.drop("_source_file") for p in parts], reference)

    aligned = [
        p.select(*[F.col(c).cast(types[c]).alias(c) for c in data_columns], "_source_file")
        for p in parts
    ]
    combined = aligned[0]
    for part in aligned[1:]:
        combined = combined.unionByName(part)
    return combined.withColumn("_ingested_at", F.current_timestamp()), types


def write_table(df, table, expected_rows=None):
    full_name = f"{CATALOG}.{SCHEMA}.{table}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(full_name)

    actual_rows = spark.table(full_name).count()
    if expected_rows is not None and actual_rows != expected_rows:
        raise ValueError(f"Row count mismatch for {full_name}: expected {expected_rows}, got {actual_rows}")
    print(f"  OK  {full_name}: {actual_rows:,} rows", flush=True)


def main():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    groups = group_parquet_files()

    # build train tables first, so test tables can reuse their types
    ordered = sorted(groups, key=lambda t: (not t.startswith("train_"), t))
    if ONLY_TABLES is not None:
        ordered = [t for t in ordered if t in ONLY_TABLES]
    print(f"Tables: {len(ordered)}", flush=True)

    train_types = {}
    for table in ordered:
        files = groups[table]
        base_name = table.split("_", 1)[1]
        reference = train_types.get(base_name) if table.startswith("test_") else None

        # row count from the parquet footers, to check the written table
        expected_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"\n[{table}] {len(files)} file(s), {expected_rows:,} rows", flush=True)

        df, types = read_and_union(files, reference)
        write_table(df, table, expected_rows)
        if table.startswith("train_"):
            train_types[base_name] = types

    print("\n[metadata]", flush=True)
    feature_definitions = spark.read.csv(
        f"{METADATA_ROOT}/feature_definitions.csv", header=True, multiLine=True, escape='"'
    )
    write_table(feature_definitions, "feature_definitions")
    sample_submission = spark.read.csv(f"{METADATA_ROOT}/sample_submission.csv", header=True, inferSchema=True)
    write_table(sample_submission, "sample_submission")
    print("\nBronze tables finished")


if __name__ == "__main__":
    main()
