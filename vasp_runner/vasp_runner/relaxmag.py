"""MP relaxed-magmom initial guesses (harvested by exps/harvest_relaxed_magmoms.py).

Shared by the run_default/true_init/ml_seed MP runners: load the harvest CSV and decide, per mp-id,
whether a usable per-site relaxed MAGMOM exists. When it does not, runners
keep MP's own INCAR MAGMOM and record magmom_source="default_fallback".
"""
PBE_FLAVORS = ("GGA", "GGA+U")


def load_relaxed_magmoms(csv_path: str, pbe_only: bool = True) -> dict:
    """Map id -> per-site magmom list (floats, POSCAR site order).

    Two CSV layouts are accepted:
      * harvest CSVs (exps/harvest_relaxed_magmoms.py): rows with a non-ok
        status are dropped; with `pbe_only`, rows whose relax task is not
        GGA/GGA+U are dropped too (those mp-ids then fall back);
      * CHGNet prediction CSVs (chgnet_preds/): recognized by the
        `magmom_chgnet` column, keyed by MP_ID or GNOME_ID; every row is
        usable, so `pbe_only` does not apply. CHGNet moments are unsigned.
      * oracle CSVs (exps/harvest_oracle_magmoms.py): `magmom_oracle` column,
        the converged run's own signed OUTCAR site moments; every row usable.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "magmom_chgnet" in df.columns:
        rows, col = df, "magmom_chgnet"
    elif "magmom_oracle" in df.columns:
        rows, col = df, "magmom_oracle"
        if "status" in df.columns:
            rows = df[df["status"] == "ok"]
    else:
        rows = df[df["status"] == "ok"]
        if pbe_only:
            rows = rows[rows["relax_flavor"].astype(str).isin(PBE_FLAVORS)]
        col = "magmom_relaxed"
    id_col = "MP_ID" if "MP_ID" in df.columns else "GNOME_ID"
    moments = {}
    for _, r in rows.iterrows():
        try:
            moments[str(r[id_col]).lower()] = [
                float(x) for x in str(r[col]).split()]
        except ValueError:
            continue
    return moments


def apply_relaxed_magmom(incar, relax_magmom):
    """Set INCAR MAGMOM to the per-site list, guarding against a site-count
    mismatch with the INCAR's existing MAGMOM (same taskdoc structure, so a
    mismatch is pathological). Returns True if applied."""
    if relax_magmom is None:
        return False
    existing = incar.get("MAGMOM")
    if isinstance(existing, (list, tuple)) and len(existing) != len(relax_magmom):
        print(f"⚠️ relaxed MAGMOM length {len(relax_magmom)} != INCAR MAGMOM "
              f"length {len(existing)}; keeping MP's own MAGMOM.")
        return False
    incar["MAGMOM"] = list(relax_magmom)
    return True
