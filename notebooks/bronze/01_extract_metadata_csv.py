"""Bronze 01: extract the two metadata CSVs from the Kaggle zip into the raw_files volume."""

import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ZIP_PATH = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_uploads/"
    "home-credit-credit-risk-model-stability.zip"
)
OUTPUT_DIR = Path("/Volumes/dbw_home_credit_audit/bronze/raw_files/metadata")
TARGET_FILES = {"feature_definitions.csv", "sample_submission.csv"}

if not ZIP_PATH.is_file():
    raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")
if not OUTPUT_DIR.parent.is_dir():
    raise FileNotFoundError(f"Target volume not found: {OUTPUT_DIR.parent}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

found = set()
with ZipFile(ZIP_PATH, "r") as archive:
    for info in archive.infolist():
        if info.is_dir():
            continue
        filename = Path(info.filename.replace("\\", "/")).name
        if filename not in TARGET_FILES:
            continue
        if filename in found:
            raise ValueError(f"Duplicate CSV filename: {filename}")
        # these two files are small, anything big means something is wrong
        if info.file_size > 100 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large CSV: {filename}")

        output_path = OUTPUT_DIR / filename
        print(f"Extracting: {filename}")
        with archive.open(info, "r") as source, open(output_path, "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        found.add(filename)

        # quick look at the file
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
