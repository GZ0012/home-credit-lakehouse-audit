"""Bronze 02: stream all parquet files from the Kaggle zip into the raw_files volume.

Keeps the parquet_files/train/... and parquet_files/test/... folders. Files that
already exist with the right size are skipped, so the script can be rerun
after a failure.
"""

import os
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ZIP_PATH = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_uploads/"
    "home-credit-credit-risk-model-stability.zip"
)
OUTPUT_ROOT = Path("/Volumes/dbw_home_credit_audit/bronze/raw_files")
CHUNK_SIZE = 8 * 1024 * 1024   # copy 8 MB at a time instead of reading whole files into memory
DRY_RUN = False                # True = only list the files


def get_relative_path(filename):
    # parquet_files/<split>/<file>.parquet, or None for other files in the zip
    path = PurePosixPath(filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP path: {filename}")
    if "parquet_files" not in path.parts or path.suffix.lower() != ".parquet":
        return None
    start = path.parts.index("parquet_files")
    return Path(*path.parts[start:])


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")
    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Output volume not found: {OUTPUT_ROOT}")

    start_time = time.time()
    with ZipFile(ZIP_PATH, "r") as archive:
        selected = []
        seen = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative_path = get_relative_path(info.filename)
            if relative_path is None:
                continue
            if relative_path in seen:
                raise ValueError(f"Duplicate destination: {relative_path}")
            seen.add(relative_path)
            selected.append((info, relative_path))

        if not selected:
            raise RuntimeError("No parquet files found in ZIP")

        total_size = sum(info.file_size for info, _ in selected)
        print(f"Files selected: {len(selected)}, uncompressed: {total_size / 1024**3:.2f} GiB", flush=True)

        if DRY_RUN:
            for info, relative_path in selected:
                print(relative_path, f"{info.file_size / 1024**2:.1f} MiB")
            return

        completed = skipped = 0
        for index, (info, relative_path) in enumerate(selected, start=1):
            target = OUTPUT_ROOT / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)

            # skip files that were already extracted completely
            if target.exists():
                if target.stat().st_size == info.file_size:
                    skipped += 1
                    print(f"[{index}/{len(selected)}] SKIP: {relative_path}", flush=True)
                    continue
                raise FileExistsError(f"Existing file has wrong size: {target}")

            # write to a .partial file first, rename when it is complete
            temp = target.with_name(target.name + ".partial")
            if temp.exists():
                temp.unlink()
            print(f"\n[{index}/{len(selected)}] EXTRACT: {relative_path}", flush=True)

            written = last_reported = 0
            try:
                with archive.open(info, "r") as source, open(temp, "wb") as destination:
                    while True:
                        chunk = source.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        destination.write(chunk)
                        written += len(chunk)
                        if written - last_reported >= 256 * 1024**2:   # progress every 256 MiB
                            print(f"    {written / 1024**2:.0f} / {info.file_size / 1024**2:.0f} MiB", flush=True)
                            last_reported = written
                # zipfile checks the CRC while reading; also check the size
                if written != info.file_size:
                    raise IOError(f"Size mismatch: {relative_path}")
                os.replace(temp, target)
            except Exception:
                print(f"FAILED: {relative_path}. Partial file: {temp}", flush=True)
                raise

            completed += 1
            print(f"    DONE: {written / 1024**2:.1f} MiB", flush=True)

    print(f"\nNewly extracted: {completed}, already there: {skipped}, total: {len(selected)}")
    print(f"Elapsed: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
