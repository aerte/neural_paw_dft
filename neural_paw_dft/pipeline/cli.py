"""``ndi`` console script."""
from __future__ import annotations

import argparse

from neural_paw_dft import __version__
import dataclasses
import sys

from .config import config_template, load_config


def _parse_incar_overrides(items):
    from pymatgen.io.vasp.inputs import Incar

    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--incar expects KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        key = key.strip().upper()
        out[key] = Incar.proc_val(key, val.strip())
    return out


def _apply_flags(cfg, args):
    if args.device:
        cfg = dataclasses.replace(cfg, device=args.device)
    if args.grid:
        cfg = dataclasses.replace(cfg, grid_dims=tuple(args.grid))
    if args.weights_dir:
        cfg = dataclasses.replace(cfg, weights_dir=args.weights_dir)
    if args.no_spin:
        cfg = dataclasses.replace(cfg, electrafi=dataclasses.replace(cfg.electrafi, spin=False))
    if args.no_chgnet:
        cfg = dataclasses.replace(cfg, chgnet=dataclasses.replace(cfg.chgnet, enabled=False))
    incar = _parse_incar_overrides(getattr(args, "incar", None))
    if incar:
        cfg = dataclasses.replace(cfg, vasp=dataclasses.replace(cfg.vasp, incar_overrides={**cfg.vasp.incar_overrides, **incar}))
    return cfg


def _add_common(p):
    p.add_argument("input", help="POSCAR/CIF/any pymatgen- or ASE-readable structure, or a CHGCAR(.lz4)")
    p.add_argument("--config", "-c", help="pipeline YAML (see `ndi config-template`)")
    p.add_argument("--out", "-o", help="output directory (default: out_dir from the config)")
    p.add_argument("--grid", nargs=3, type=int, metavar=("NGX", "NGY", "NGZ"), help="FFT grid dims")
    p.add_argument("--device", help="cpu | cuda | cuda:N")
    p.add_argument("--weights-dir", help="directory holding the model weights")
    p.add_argument("--no-spin", action="store_true", help="charge-only seed: the spin grid is not written (the configured ELECTRAFI checkpoint still runs)")
    p.add_argument("--no-chgnet", action="store_true", help="skip CHGNet (no MAGMOM override / spin constraint)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ndi", description="Neural electronic initialization for VASP")
    ap.add_argument("--version", "-V", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="predict and write CHGCAR + INCAR/POSCAR/KPOINTS and POTCAR (or POTCAR.spec without a POTCAR library)")
    _add_common(b)
    b.add_argument("--incar", nargs="*", metavar="KEY=VAL", help="INCAR overrides, applied last")

    p = sub.add_parser("predict", help="predict only; write grids (.npy), augmentation (.npz), CHGNet MAGMOM and ndi_prediction.json")
    _add_common(p)

    sub.add_parser("config-template", help="print an example YAML config")

    args = ap.parse_args(argv)
    if args.cmd == "config-template":
        sys.stdout.write(config_template())
        return 0

    from .pipeline import Pipeline

    cfg = _apply_flags(load_config(args.config), args)
    pipe = Pipeline(cfg)
    if args.cmd == "build":
        out = pipe.build(args.input, args.out)
        print(f"wrote VASP inputs to {out}")
    else:
        pred = pipe.predict(args.input)
        out = pipe.save_prediction(pred, args.out or cfg.out_dir)
        print(f"wrote predictions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
