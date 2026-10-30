import os
import re
from collections import defaultdict

import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, NullType, NumericType, StringType

spark = SparkSession.builder.getOrCreate()


# ==================================================
# 1. Configuration
# ==================================================

CATALOG = "dbw_home_credit_audit"
SCHEMA = "bronze"

RAW_ROOT = "/Volumes/dbw_home_credit_audit/bronze/raw_files"
PARQUET_ROOT = f"{RAW_ROOT}/parquet_files"
METADATA_ROOT = f"{RAW_ROOT}/metadata"

# 只建部分表时填写，例如 {"train_base"}；None 表示全部
ONLY_TABLES = None


# ==================================================
# 2. 将分片文件归入逻辑表
# ==================================================

# Kaggle 将大表拆成多个分片：
#   train_credit_bureau_a_2_0 ... _10  ->  train_credit_bureau_a_2
#   train_static_0_0, _1               ->  train_static_0
# 不带分片编号的文件保持原名，例如 train_applprev_2、train_static_cb_0
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


# ==================================================
# 3. 统一各分片的列类型
# ==================================================

# 分片之间可能存在类型冲突：
#   - 某个分片里整列为空 -> null 类型，其他分片是 string/bool
#   - num_group1 在不同分片里分别是 int64 和 double
def resolve_types(dataframes, reference=None):
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
                both_numeric = isinstance(current, NumericType) and isinstance(
                    field.dataType, NumericType
                )
                resolved[field.name] = DoubleType() if both_numeric else StringType()

    # test 表以对应 train 表的类型为准，保证两边一致
    if reference:
        for name in resolved:
            if name in reference:
                resolved[name] = reference[name]

    # 所有分片都为空的列：Delta 不支持 void 类型，存为 string
    return {
        name: dtype if dtype is not None else StringType()
        for name, dtype in resolved.items()
    }


def read_and_union(files, reference=None):
    parts = [
        spark.read.parquet(path).withColumn(
            "_source_file", F.col("_metadata.file_name")
        )
        for path in files
    ]

    data_columns = [c for c in parts[0].columns if c != "_source_file"]
    types = resolve_types([p.drop("_source_file") for p in parts], reference)

    aligned = [
        p.select(
            *[F.col(c).cast(types[c]).alias(c) for c in data_columns],
            "_source_file",
        )
        for p in parts
    ]

    combined = aligned[0]
    for part in aligned[1:]:
        combined = combined.unionByName(part)

    return combined.withColumn("_ingested_at", F.current_timestamp()), types


# ==================================================
# 4. 写入 Delta 表
# ==================================================

def write_table(df, table, expected_rows=None):
    full_name = f"{CATALOG}.{SCHEMA}.{table}"

    (
        df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(full_name)
    )

    actual_rows = spark.table(full_name).count()

    if expected_rows is not None and actual_rows != expected_rows:
        raise ValueError(
            f"Row count mismatch for {full_name}: "
            f"expected {expected_rows}, got {actual_rows}"
        )

    print(f"  OK  {full_name}: {actual_rows:,} rows", flush=True)


def main():

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    groups = group_parquet_files()

    # train 先建，test 复用 train 的列类型
    ordered = sorted(groups, key=lambda t: (not t.startswith("train_"), t))

    if ONLY_TABLES is not None:
        ordered = [t for t in ordered if t in ONLY_TABLES]

    print("=" * 60)
    print("HOME CREDIT BRONZE TABLES")
    print(f"Tables: {len(ordered)}")
    print("=" * 60, flush=True)

    train_types = {}

    for table in ordered:

        files = groups[table]
        base_name = table.split("_", 1)[1]
        reference = train_types.get(base_name) if table.startswith("test_") else None

        # parquet footer 里的行数，用于校验
        expected_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

        print(f"\n[{table}] {len(files)} file(s), {expected_rows:,} rows", flush=True)

        df, types = read_and_union(files, reference)
        write_table(df, table, expected_rows)

        if table.startswith("train_"):
            train_types[base_name] = types

    # 两个元数据 CSV
    print("\n[metadata]", flush=True)

    feature_definitions = spark.read.csv(
        f"{METADATA_ROOT}/feature_definitions.csv",
        header=True,
        multiLine=True,
        escape='"',
    )
    write_table(feature_definitions, "feature_definitions")

    sample_submission = spark.read.csv(
        f"{METADATA_ROOT}/sample_submission.csv",
        header=True,
        inferSchema=True,
    )
    write_table(sample_submission, "sample_submission")

    print("\n" + "=" * 60)
    print("BRONZE TABLES FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
