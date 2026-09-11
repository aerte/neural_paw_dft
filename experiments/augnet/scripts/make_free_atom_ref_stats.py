#!/usr/bin/env python3
"""
make_free_atom_ref_stats.py

Turn the packed free-atom PAW references (reference_calcs/free_atom_refs.npz)
into a PAWStats file, so the model trains on the DELTA from the free atom
instead of on the raw occupancy, with no standardization.

The trick is that PAWStats already applies exactly the transform we want:

    target_std = (target - shift[z]) / scale[z]

Setting shift[z] = D_free(z) and scale[z] = 1 makes the "standardized" target
the physical delta in electrons, and `de_standardize` (already called before
every physical metric, moment term, Sanvito conversion and npz export) adds the
reference back. So nothing in train_augnet.py has to change: point
`data.stats_file` / `--stats-file` at the file this script writes.

Two things are enforced on the reference:

  * only L == 0 blocks are kept. A free atom is spherically symmetric, so every
    L > 0 component is SCF residue. Keeping it would also break equivariance:
    the target y rotates with the structure, a fixed nonzero L>0 reference does
    not, so y - D_free would not be an equivariant quantity. The largest value
    thrown away is reported.
  * the reference is placed in e3nn order via physical_to_canonical_blocks, the
    same map the training targets go through. The npz stores raw CHGCAR
    (= Sanvito, per-element POTCAR projector) order.

The reference is POTCAR-specific: D is defined by the projectors, so this file is
only valid for datasets using the MP POTCARs the free-atom runs used (the TITELs
are checked against mp_potcar_map). Do NOT pair it with a run that sets
data.potcar_l_channels for a different POTCAR set.

The mag channel gets a zero reference by default (--mag zero). A free atom is
maximally spin-polarized by Hund's rule while most atoms in the corpus are
non-magnetic, so subtracting the free-atom magnetization would hand every
non-magnetic atom a large delta. --mag ref opts into it anyway.

Usage:
    python make_free_atom_ref_stats.py --out stats/paw_ref_freeatom.pt \
        --compare stats/paw_stats_mpfull2025_10k.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Bootstrap: repo root (parent of this script's dir) must be importable for
# the src.* / scripts.* imports below when this file is run directly.
import os as _os, sys as _sys
from neural_init.augnet.mp_potcar_map import MP_POTCAR, MP_POTCAR_BY_Z
from neural_init.augnet.paw_basis_transform import (
    Z_TO_SCHEMA,
    physical_to_canonical_blocks,
    sanvito_to_e3nn_with_basis_padded,
    set_l_channel_overrides,
)
from neural_init.augnet.paw_moments import moments_from_coeffs
from neural_init.augnet.paw_stats import CHANNELS, MAX_DIM, PAWStats

SYMBOL_BY_Z = {z: e["element"] for z, e in MP_POTCAR_BY_Z.items()}


def load_refs(npz_path: Path, variant: str):
    """One row per element, as {Z: dict}. Fails loudly on a duplicated element."""
    d = np.load(npz_path)
    sel = d["variant"] == variant
    if not sel.any():
        sys.exit(f"error: no rows with variant={variant!r} in {npz_path}")

    refs = {}
    for i in np.flatnonzero(sel):
        z = int(d["Z"][i])
        if z in refs:
            sys.exit(f"error: duplicate reference for Z={z} in {npz_path}")
        refs[z] = dict(
            element=str(d["element"][i]),
            schema=int(d["schema"][i]),
            total=d["aug_total"][i].astype(np.float64),
            mag=d["aug_mag"][i].astype(np.float64),
            titel=str(d["potcar_titel"][i]),
        )
    return refs


def check_potcar(z: int, ref: dict, overridden: set) -> str | None:
    """None if the reference's POTCAR and schema match what the pipeline assumes.

    Elements in `overridden` (--l-channels) use a non-MP POTCAR on purpose, so
    the variant-name check is skipped for them; the schema check still applies
    (an l-channel override is a permutation, so the schema is unchanged).
    """
    expected_schema = Z_TO_SCHEMA.get(z)
    if expected_schema is None:
        return "not in Z_TO_SCHEMA (element is not trainable)"
    if ref["schema"] != expected_schema:
        return f"schema {ref['schema']} != Z_TO_SCHEMA {expected_schema}"

    entry = MP_POTCAR.get(ref["element"])
    if entry is None:
        return "element not in MP_POTCAR"
    # "TITEL  = PAW_PBE Fe_pv 06Sep2000" -> "Fe_pv"
    try:
        variant = ref["titel"].split("PAW_PBE")[1].split()[0]
    except IndexError:
        return f"unparseable TITEL {ref['titel']!r}"
    if ref["element"] in overridden:
        print(f"  [override] Z={z} {ref['element']}: using {variant} "
              f"(mp_potcar_map has {entry['variant']})")
        return None
    if variant != entry["variant"]:
        return f"POTCAR {variant} != MP_POTCAR {entry['variant']}"
    return None


def reference_to_e3nn(z: int, y_sanvito: np.ndarray):
    """(e3nn-order reference [MAX_DIM], largest discarded |L>0| component)."""
    y = torch.zeros(1, MAX_DIM, dtype=torch.float64)
    y[0, : y_sanvito.shape[0]] = torch.from_numpy(y_sanvito)

    dropped = 0.0
    for san, _e3, L in physical_to_canonical_blocks(z):
        if L > 0:
            dropped = max(dropped, float(y[0, san].abs().max()))
            y[0, san] = 0.0

    mask = torch.zeros(1, MAX_DIM, dtype=torch.bool)
    mask[0, : Z_TO_SCHEMA[z]] = True
    y_e3nn, _ = sanvito_to_e3nn_with_basis_padded(y, mask, torch.tensor([z]))
    return y_e3nn[0], dropped


def build(refs: dict, mag_mode: str, scale_value: float, overridden: set):
    shift = {c: torch.zeros(119, MAX_DIM, dtype=torch.float64) for c in CHANNELS}
    scale = {c: torch.full((119, MAX_DIM), scale_value, dtype=torch.float64)
             for c in CHANNELS}

    covered, skipped, worst = [], [], (0.0, None)
    for z in sorted(Z_TO_SCHEMA):
        ref = refs.get(z)
        if ref is None:
            skipped.append(z)
            continue
        problem = check_potcar(z, ref, overridden)
        if problem is not None:
            print(f"  [skip] Z={z} {ref['element']}: {problem}", file=sys.stderr)
            skipped.append(z)
            continue

        y_e3nn, dropped = reference_to_e3nn(z, ref["total"])
        shift["total"][z] = y_e3nn
        if dropped > worst[0]:
            worst = (dropped, ref["element"])

        if mag_mode == "ref":
            shift["mag"][z], _ = reference_to_e3nn(z, ref["mag"])
        covered.append(z)

    return shift, scale, covered, skipped, worst


def report(shift, covered, refs, compare: Path | None):
    """Tr(D_free) per element, against ZVAL and (optionally) the fitted shift."""
    z_list = torch.tensor(covered)
    tr_free, valid = moments_from_coeffs(shift["total"][z_list], z_list, [1])
    tr_free = tr_free[:, 0]

    zvals = torch.tensor(
        [float(MP_POTCAR[refs[int(z)]["element"]]["zval"]) for z in covered],
        dtype=torch.float64,
    )
    print(f"\n  Tr(D_free): min {tr_free.min():.3f}  median {tr_free.median():.3f}  "
          f"max {tr_free.max():.3f}   (all {int(valid.sum())}/{len(covered)} reconstructed)")
    print(f"  corr(Tr(D_free), ZVAL) = {np.corrcoef(tr_free.numpy(), zvals.numpy())[0, 1]:.4f}")

    if compare is None:
        return
    if not compare.exists():
        print(f"  [warn] --compare file not found: {compare}", file=sys.stderr)
        return

    fitted = PAWStats.load(str(compare))
    fit_shift = fitted.shift["total"][z_list]
    tr_fit, _ = moments_from_coeffs(fit_shift, z_list, [1])
    tr_fit = tr_fit[:, 0]

    # Elements that fell back to schema-pooled stats have a shift that is not
    # really theirs; exclude them from the agreement number.
    fallback = set(fitted.meta.get("used_fallback", {}).get("total", []))
    keep = torch.tensor([int(z) not in fallback for z in covered])
    a, b = tr_free[keep].numpy(), tr_fit[keep].numpy()

    print(f"\n  vs fitted per-Z shift in {compare.name} "
          f"({int(keep.sum())} elements with per-Z stats, "
          f"{len(covered) - int(keep.sum())} on the schema-pooled fallback):")
    print(f"    corr(Tr(D_free), Tr(shift_fitted)) = {np.corrcoef(a, b)[0, 1]:.4f}")
    print(f"    mean |Tr(D_free) - Tr(shift_fitted)| = {np.abs(a - b).mean():.4f} e")
    print(f"    RMS over L=0 coefficients: free {np.sqrt((shift['total'][z_list][keep] ** 2).mean()):.4f}"
          f"  fitted {np.sqrt((fit_shift[keep] ** 2).mean()):.4f}")

    order = np.argsort(-np.abs(a - b))[:8]
    els = [refs[int(z)]["element"] for z, k in zip(covered, keep.tolist()) if k]
    print("    largest disagreements (element, Tr free, Tr fitted):")
    for i in order:
        print(f"      {els[i]:>3}  {a[i]:8.3f}  {b[i]:8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--refs", nargs="+", default=["reference_calcs/free_atom_refs.npz"],
                    help="packed reference npz file(s); a later file overrides "
                         "earlier ones per element (e.g. the main pack plus "
                         "free_atom_refs_mo_sv.npz)")
    ap.add_argument("--l-channels", default=None,
                    help='JSON projector-order override for non-MP POTCARs, same as '
                         'the run config\'s data.potcar_l_channels, e.g. '
                         '\'{"Mo": [0, 0, 1, 1, 2, 2]}\' for Mo_sv. Also relaxes '
                         "the POTCAR variant-name check for those elements. The "
                         "resulting stats file is only valid for runs setting the "
                         "same data.potcar_l_channels.")
    ap.add_argument("--out", default="stats/paw_ref_freeatom.pt")
    ap.add_argument("--variant", default="plain",
                    help="which free-atom run to use (plain | ldau_convention); "
                         "they are identical for this element set")
    ap.add_argument("--mag", choices=["zero", "ref"], default="zero",
                    help="mag-channel reference: zero (default) or the free atom's "
                         "own magnetization")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="single global divisor applied to the delta (default 1.0 = "
                         "raw electrons). Not standardization: one number for every "
                         "element and coefficient, so it only rescales the loss.")
    ap.add_argument("--compare", default=None,
                    help="fitted paw_stats_*.pt to sanity-check the reference against")
    args = ap.parse_args()

    overrides = json.loads(args.l_channels) if args.l_channels else {}
    if overrides:
        set_l_channel_overrides(overrides)

    refs = {}
    for p in args.refs:
        refs_path = Path(p)
        if not refs_path.exists():
            sys.exit(f"error: {refs_path} does not exist")
        file_refs = load_refs(refs_path, args.variant)
        for z, r in file_refs.items():
            if z in refs:
                print(f"  [override] Z={z} {r['element']}: {p} replaces the "
                      f"earlier reference")
        refs.update(file_refs)
        print(f"Loaded {len(file_refs)} free-atom references "
              f"(variant={args.variant}) from {refs_path}")

    shift, scale, covered, skipped, worst = build(refs, args.mag, args.scale,
                                                  set(overrides))

    print(f"  reference set on {len(covered)}/{len(Z_TO_SCHEMA)} trainable elements")
    if skipped:
        names = [SYMBOL_BY_Z.get(z, str(z)) for z in skipped]
        print(f"  [warn] {len(skipped)} element(s) left at a ZERO reference "
              f"(they train on the raw occupancy): {names}", file=sys.stderr)
    print(f"  largest discarded L>0 component: {worst[0]:.2e} ({worst[1]})")
    if args.mag == "zero":
        print("  mag channel: zero reference (raw occupancy), scale "
              f"{args.scale}")

    report(shift, covered, refs, Path(args.compare) if args.compare else None)

    meta = {
        "kind": "free_atom_reference",
        "source": list(args.refs),
        "variant": args.variant,
        "l_channel_overrides": overrides,
        "mag_reference": args.mag,
        "scale_value": args.scale,
        "n_elements": len(covered),
        "elements_without_reference": skipped,
        "note": "shift = free-atom D (L=0 only, e3nn order); scale is a single "
                "global constant. Targets are physical deltas in electrons, NOT "
                "per-element standardized.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    PAWStats(shift=shift, scale=scale, meta=meta).save(str(out))
    print(f"\n[done] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
