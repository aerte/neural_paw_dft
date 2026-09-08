#!/usr/bin/env python3
import os, sys, time, json, random
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mp_api.client import MPRester
from emmet.core.summary import HasProps

# =====================
# Config (env overrides)
# =====================
OUT_DIR         = os.getenv("OUT_DIR", "/MP_FULL_2025")
SPLIT_DIR = os.getenv("SPLIT_DIR", "/ELECTRAFI/data_splits")
API_KEY         = os.getenv("MP_API_KEY", "")
MAX_WORKERS_CAP = int(os.getenv("WORKERS_CAP", "24"))   # be nice to the API
EXCEPT_LOG      = os.getenv("EXCEPT_LOG", "mp-ids-except-full.txt")
SPLIT_JSON_PATH = os.getenv("SPLIT_JSON_PATH", os.path.join(SPLIT_DIR, "datasplits_mpfull2025.json"))

# Split params
RNG_SEED   = int(os.getenv("SPLIT_SEED", "2025"))
VAL_COUNT  = 512
TEST_COUNT = 2000

def detect_cpus(default=1) -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        pass
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return os.cpu_count() or default

WORKERS = min(detect_cpus(), MAX_WORKERS_CAP)

# =====================
# Helpers
# =====================
def list_all_ids_with_charge_density(mpr: MPRester, chunk_size: int = 1000) -> List[str]:
    """Fetch ALL material_ids that advertise charge_density."""
    ids: List[str] = []
    docs = mpr.materials.summary.search(
        has_props=[HasProps.charge_density],
        fields=["material_id"],
        chunk_size=chunk_size,
    )
    for d in tqdm(docs, desc="Listing CD-capable IDs", dynamic_ncols=True):
        ids.append(d.material_id)
    return ids

def make_splits(all_ids: List[str], seed: int, val_n: int, test_n: int):
    """Deterministic shuffle; val first N, test next M, rest train."""
    rng = random.Random(seed)
    ids = all_ids[:]
    rng.shuffle(ids)
    val  = ids[:val_n]
    test = ids[val_n:val_n + test_n]
    train = ids[val_n + test_n:]
    return {"train": train, "validation": val, "test": test}

def download_one(mp_id: str, out_dir: str, api_key: str) -> Tuple[str, str | None]:
    out_path = os.path.join(out_dir, f"{mp_id}.chgcar")
    if os.path.isfile(out_path):
        return mp_id, None
    delays = [1.0, 2.0, 4.0, 8.0]
    for attempt, delay in enumerate([0.0, *delays], start=1):
        try:
            if delay:
                time.sleep(delay)
            with MPRester(api_key) as mpr:
                chgcar = mpr.get_charge_density_from_material_id(mp_id)
            if chgcar is None:
                return mp_id, "No CHGCAR returned (None)"
            chgcar.write_file(out_path)
            return mp_id, None
        except Exception as e:
            if attempt == len(delays) + 1:
                return mp_id, str(e)
            continue

def main():
    if not API_KEY:
        print("ERROR: Set MP_API_KEY env var or fill API_KEY.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[I] OUT_DIR={OUT_DIR}")
    print(f"[I] Using {WORKERS} workers (detected CPUs={detect_cpus()}; cap={MAX_WORKERS_CAP})")

    existing = {fn.split(".", 1)[0] for fn in os.listdir(OUT_DIR) if fn.endswith(".chgcar")}

    with MPRester(API_KEY) as mpr:
        all_ids = list_all_ids_with_charge_density(mpr)

    print(f"[I] Total MP IDs with charge_density: {len(all_ids)}")

    # ---- Create and save splits (val=512, test=2000, rest=train) ----
    splits = make_splits(all_ids, RNG_SEED, VAL_COUNT, TEST_COUNT)
    print(f"[I] Split sizes -> train: {len(splits['train'])}, val: {len(splits['validation'])}, test: {len(splits['test'])}")
    os.makedirs(os.path.dirname(SPLIT_JSON_PATH), exist_ok=True)
    with open(SPLIT_JSON_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[I] Wrote splits JSON: {SPLIT_JSON_PATH}")

    # IDs to download = union of all splits minus existing (everything goes in same OUT_DIR)
    target_ids = set().union(*splits.values())
    ids = [i for i in target_ids if i not in existing]
    print(f"[I] Remaining to download: {len(ids)}")

    if not ids:
        print("[I] Nothing to do.")
        return

    errors: List[str] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(download_one, mp_id, OUT_DIR, API_KEY) for mp_id in ids]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", dynamic_ncols=True):
            mp_id, err = fut.result()
            if err:
                errors.append(f"{mp_id} {err}")

    if errors:
        with open(EXCEPT_LOG, "a") as fh:
            fh.write("\n".join(errors) + "\n")
        print(f"[I] Wrote {len(errors)} errors to {EXCEPT_LOG}")

    print("[I] Done.")

if __name__ == "__main__":
    # keep BLAS from oversubscribing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
