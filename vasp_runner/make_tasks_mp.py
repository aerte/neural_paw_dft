#!/usr/bin/env python3
import os
import csv
import json
import argparse
import random
import re

# Defaults for your MP FULL data
DEFAULT_ROOT_DIR = "/path/to/MP_FULL_2025"
DEFAULT_OUTPUT_CSV = "MP_TASKS.csv"
SPLIT_FILE = "/path/to/datasplits_mpfull2025.json"


def find_chgcars(root_dir):
    """
    Recursively find all CHGCAR-like files for MP:

    Typical filenames:
      - mp-653005.chgcar.lz4
      - mp-653005.chgcar
      - mp-653005.CHGCAR.lz4
      - mp-653005.CHGCAR

    We accept anything that:
      - contains 'mp-' in the basename, and
      - ends with one of the CHGCAR-ish suffixes.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            if "mp-" not in lower:
                continue
            if (
                lower.endswith("chgcar")
                or lower.endswith("chgcar.lz4")
                or lower.endswith("chgcar.gz")
            ):
                yield os.path.join(dirpath, fn)


def extract_mpid_from_path(path):
    """
    Extract an mp-id from a CHGCAR path.

    Assumes filenames like:
      - 'mp-653005.chgcar.lz4'
      - 'mp-653005.chgcar'
      - 'mp-653005.CHGCAR.lz4'

    Returns:
        'mp-653005' or None
    """
    base = os.path.basename(path)
    # Match 'mp-<digits>' at the start, before the first dot.
    m = re.match(r"^(mp-\d+)\.", base, flags=re.IGNORECASE)
    if not m:
        return None
    # Normalize to lowercase mp-id
    return m.group(1).lower()


def filter_by_json_test(chgcar_paths, json_path):
    """
    From a list of CHGCAR paths, keep only those whose extracted MP_ID
    is in the 'test' list of the JSON file.

    Expected JSON format (your case):
    {
      "train": [
        "mp-1105173",
        "mp-756183",
        ...
      ],
      "validation": [...],
      "test": [
        "mp-123456",
        "mp-653005",
        ...
      ]
    }

    We accept both 'mp-12345' strings and bare integers (converted to 'mp-<int>').
    """
    with open(json_path, "r") as f:
        splits = json.load(f)

    raw_test_ids = splits.get("test", [])
    if not raw_test_ids:
        print(f"⚠️  No 'test' entries found in {json_path}; returning empty list.")
        return []

    test_ids = set()
    for x in raw_test_ids:
        if isinstance(x, int):
            test_ids.add(f"mp-{x}")
        else:
            test_ids.add(str(x).lower())

    filtered = []
    for p in chgcar_paths:
        mpid = extract_mpid_from_path(p)
        if mpid is not None and mpid in test_ids:
            filtered.append(p)

    return filtered


def write_tasks_csv(chgcar_paths, output_csv):
    """
    Write the CSV with headers, including MP_ID and performed_* flags.
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CHGCAR_PATH",
            "MP_ID",
            "performed_default",
            "performed_true_init",
            "performed_ml_init",
        ])
        for path in chgcar_paths:
            mpid = extract_mpid_from_path(path)
            if mpid is None:
                # Skip anything we can't parse an mp-id from
                print(f"⚠️ Skipping {path}: could not parse MP_ID")
                continue
            writer.writerow([path, mpid, "False", "False", "False"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create MP_TASKS.csv for MP CHGCAR(/.lz4) files."
    )
    parser.add_argument(
        "--root",
        dest="root_dir",
        default=DEFAULT_ROOT_DIR,
        help=f"Root directory to search for CHGCAR files (default: {DEFAULT_ROOT_DIR})",
    )
    parser.add_argument(
        "--out",
        dest="output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="Number of random CHGCARs to select. "
             "Use -1 to use all matching files after filtering to 'test'.",
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=SPLIT_FILE,
        help=f"JSON with 'train'/'validation'/'test' MP IDs "
             f"(default: {SPLIT_FILE}). Only 'test' IDs are used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )

    args = parser.parse_args()

    # 1) Find all CHGCAR-like files under root_dir
    all_paths = sorted(find_chgcars(args.root_dir))
    if not all_paths:
        print(f"❌ No mp-*.chgcar* files found under {args.root_dir}")
        raise SystemExit(1)

    print(f"Found {len(all_paths)} MP CHGCAR-like files under {args.root_dir}")

    # 2) Always filter by 'test' split from the JSON (default is SPLIT_FILE)
    before = len(all_paths)
    all_paths = filter_by_json_test(all_paths, args.splits_json)
    print(
        f"Filtered by test split in {args.splits_json}: "
        f"{before} → {len(all_paths)} files"
    )
    if not all_paths:
        print("❌ No files matched the 'test' IDs from the JSON.")
        raise SystemExit(1)

    # 3) Sample N or use all (within 'test')
    if args.num == -1 or args.num >= len(all_paths):
        selected = all_paths
        print(f"Using all {len(selected)} test files.")
    else:
        random.seed(args.seed)
        selected = random.sample(all_paths, args.num)
        print(f"Randomly selected {len(selected)} files out of {len(all_paths)} test files.")

    # 4) Write CSV
    write_tasks_csv(selected, args.output_csv)
    print(f"✅ Wrote {len(selected)} entries to {args.output_csv}")
