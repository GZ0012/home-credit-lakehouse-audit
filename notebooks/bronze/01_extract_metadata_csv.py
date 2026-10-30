from pathlib import Path
from zipfile import ZipFile
import shutil
import pandas as pd


# ==========================================
# 1. Unity Catalog Volume paths
# ==========================================

ZIP_PATH = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_uploads/"
    "home-credit-credit-risk-model-stability.zip"
)

OUTPUT_DIR = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_files/metadata"
)

# 只提取这两个 CSV
TARGET_FILES = {
    "feature_definitions.csv",
    "sample_submission.csv",
}


# ==========================================
# 2. Check paths
# ==========================================

if not ZIP_PATH.is_file():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

if not OUTPUT_DIR.parent.is_dir():
    raise FileNotFoundError(
        f"Target Volume not found: {OUTPUT_DIR.parent}"
    )

# metadata 不存在时自动创建
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. Extract selected CSV files
# ==========================================

with ZipFile(ZIP_PATH, "r") as archive:

    found = set()

    for info in archive.infolist():

        if info.is_dir():
            continue

        filename = Path(
            info.filename.replace("\\", "/")
        ).name

        if filename not in TARGET_FILES:
            continue

        if filename in found:
            raise ValueError(
                f"Duplicate CSV filename: {filename}"
            )

    
        if info.file_size > 100 * 1024 * 1024:
            raise ValueError(
                f"Unexpectedly large CSV: {filename}"
            )

        output_path = OUTPUT_DIR / filename

        print(f"Extracting: {filename}")

        with archive.open(info, "r") as source:
            with open(output_path, "wb") as target:
                shutil.copyfileobj(
                    source,
                    target,
                    length=1024 * 1024
                )

        found.add(filename)

    
        df = pd.read_csv(output_path)

        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(5).to_string(index=False))
        print(f"Saved to: {output_path}")
        print("-" * 50)

    missing = TARGET_FILES - found

    if missing:
        print("Files not found:", missing)

print("CSV extraction finished!")




