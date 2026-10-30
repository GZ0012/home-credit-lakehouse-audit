
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import os
import time

# ==================================================
# 1. Configuration
# ==================================================

ZIP_PATH = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_uploads/"
    "home-credit-credit-risk-model-stability.zip"
)

OUTPUT_ROOT = Path(
    "/Volumes/dbw_home_credit_audit/bronze/raw_files"
)

# 8 MB per chunk, avoiding loading entire files into RAM
CHUNK_SIZE = 8 * 1024 * 1024

# Set True to inspect files without extracting
DRY_RUN = False


# ==================================================
# 2. Identify Parquet files inside the ZIP
# ==================================================

def get_relative_path(filename):
    path = PurePosixPath(filename.replace("\\", "/"))

    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP path: {filename}")

    parts = path.parts

    if "parquet_files" not in parts:
        return None

    if path.suffix.lower() != ".parquet":
        return None

    start = parts.index("parquet_files")

    # Preserve parquet_files/train/... or test/...
    return Path(*parts[start:])


# ==================================================
# 3. Main extraction
# ==================================================

def main():

    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"ZIP not found: {ZIP_PATH}")

    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(
            f"Output volume not found: {OUTPUT_ROOT}"
        )

    start_time = time.time()

    with ZipFile(ZIP_PATH, "r") as archive:

        selected = []
        seen_paths = set()

        for info in archive.infolist():

            if info.is_dir():
                continue

            relative_path = get_relative_path(info.filename)

            if relative_path is None:
                continue

            if relative_path in seen_paths:
                raise ValueError(
                    f"Duplicate destination: {relative_path}"
                )

            seen_paths.add(relative_path)
            selected.append((info, relative_path))

        if not selected:
            raise RuntimeError("No Parquet files found in ZIP")

        total_size = sum(
            info.file_size for info, _ in selected
        )

        print("=" * 60)
        print("HOME CREDIT PARQUET EXTRACTION")
        print(f"Files selected: {len(selected)}")
        print(f"Uncompressed: {total_size / 1024**3:.2f} GiB")
        print(f"Source: {ZIP_PATH}")
        print(f"Destination: {OUTPUT_ROOT}")
        print("=" * 60, flush=True)

        if DRY_RUN:
            for info, relative_path in selected:
                print(
                    relative_path,
                    f"{info.file_size / 1024**2:.1f} MiB"
                )
            print("Dry run finished. No files written.")
            return

        completed = 0
        skipped = 0

        for index, (info, relative_path) in enumerate(
            selected, start=1
        ):

            target = OUTPUT_ROOT / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)

            # Resume support: skip complete existing files
            if target.exists():

                if target.stat().st_size == info.file_size:
                    skipped += 1
                    print(
                        f"[{index}/{len(selected)}] "
                        f"SKIP: {relative_path}",
                        flush=True
                    )
                    continue

                raise FileExistsError(
                    f"Existing file has wrong size: {target}"
                )

            # Write to a temporary filename first
            temp = target.with_name(target.name + ".partial")

            if temp.exists():
                temp.unlink()

            print(
                f"\n[{index}/{len(selected)}] "
                f"EXTRACT: {relative_path}",
                flush=True
            )

            written = 0
            last_reported = 0

            # Stream extraction directly into ADLS Volume
            try:
                with archive.open(info, "r") as source:
                    with open(temp, "wb") as destination:

                        while True:
                            chunk = source.read(CHUNK_SIZE)

                            if not chunk:
                                break

                            destination.write(chunk)
                            written += len(chunk)

                            # Progress log every 256 MiB
                            if written - last_reported >= 256 * 1024**2:
                                print(
                                    f"    {written / 1024**2:.0f} "
                                    f"/ {info.file_size / 1024**2:.0f} MiB",
                                    flush=True
                                )
                                last_reported = written

                # ZIP library verifies CRC while reading
                if written != info.file_size:
                    raise IOError(
                        f"Size mismatch: {relative_path}"
                    )

                # Rename within the same external volume
                os.replace(temp, target)

            except Exception:
                print(
                    f"FAILED: {relative_path}. "
                    f"Partial file: {temp}",
                    flush=True
                )
                raise

            completed += 1

            print(
                f"    DONE: {written / 1024**2:.1f} MiB",
                flush=True
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("EXTRACTION FINISHED")
    print(f"Newly extracted: {completed}")
    print(f"Already existed: {skipped}")
    print(f"Total selected: {len(selected)}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
