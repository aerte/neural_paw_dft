"""Smoke test for the spin (charge-difference) loss pipeline.

Runs one train + val + test step of ELECTRAFI (via `local_conf.yaml`) on two
CHGCARs and checks the spin-loss gating in both directions:

- `data/mp-8648.chgcar.lz4` — spin-polarized. int|m| dv clears spin_mag_min, so
  the loss takes the NMAE branch and `<split>/spin_nmae` is logged.
- `data/mp-8648_nospin.chgcar.lz4` — the same structure with the magnetization
  block dropped. `chgcar_to_cd` falls back to a zero `cd_diff` grid, so the loss
  takes the absolute-L1-toward-zero branch: `spin_loss` is still logged and still
  trained on (that is the point — the head must learn to emit nothing here), but
  `spin_nmae` is not, because it is not meaningful against a zero target.

Run from the repo root:

    python test_spin_pipeline.py
"""

import os
import tempfile

import lightning as L
import torch
import yaml
from torch.utils.data import DataLoader

from neural_paw_dft._resources import resolve_data_path
from neural_paw_dft.spin_electrafi.model.data_utils import DensityDataset, collate_fn
from neural_paw_dft.spin_electrafi.model.ELECTRAFI import ELECTRAFI
from neural_paw_dft.spin_electrafi.tools.density_conversions import get_density
from neural_paw_dft.spin_electrafi.utils.train_helper_funcs import set_all_paths, set_all_seeds

CONFIG_PATH = "local_conf.yaml"
DATA_DIR = "data"
SPIN_FILE = "mp-8648.chgcar.lz4"          # spin-polarized: cd_diff has a real grid
NOSPIN_FILE = "mp-8648_nospin.chgcar.lz4"  # magnetization dropped: cd_diff falls back to zeros
SPIN_MAG_MIN = 0.1                         # electrons; must match spin_mag_min


def build_config() -> dict:
    """Assemble the config the same way train.run() does, then trim it for a
    fast, side-effect-free local smoke test."""
    config = yaml.safe_load(open(CONFIG_PATH))

    # EScAIP backbone config (train.run() loads this into config['escaip_config']).
    with open(resolve_data_path(config["escaip_cfg_path"], "spin_electrafi")) as f:
        config["escaip_config"] = yaml.safe_load(f)

    # Point the single sample and every output at a throwaway temp dir.
    config["dens_path"] = DATA_DIR
    out_dir = tempfile.mkdtemp(prefix="spin_smoke_")
    config["output_base_path"] = out_dir
    config["model_dir"] = os.path.join(out_dir, "trained_models")
    print(f"[smoke] outputs -> {out_dir}")

    # Keep it lean: no wandb, no checkpoints, no CHGCAR/plot writing, no OOD.
    config["wandb"] = False
    config["save_model"] = False
    config["ood_eval"] = False
    config["construct_val_cd"] = False
    config["construct_test_cd"] = False
    return config


def make_loader(files, dens_path, config):
    return DataLoader(
        DensityDataset(files, dens_path, config, None),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )


def run_case(config, data_file, expect_spin_loss):
    """Run one train+val+test step on `data_file` and return the logged spin
    metrics. Fresh model + trainer per case so callback_metrics don't leak."""
    files = [data_file]
    model = ELECTRAFI(
        train_files=files,
        validation_files=files,
        test_files=files,
        model_handler=None,
        config=config,
    )
    assert model.spin_enabled, "spin path is off — check spin_type in the config"

    # Show what target the loader hands the steps (drives the gate).
    _, _, _, _, cd_diff = get_density(os.path.join(DATA_DIR, data_file))
    abs_sum = float(cd_diff.abs().sum())
    print(
        f"[smoke] {data_file}: cd_diff shape={tuple(cd_diff.shape)}  "
        f"abs-sum={abs_sum:.4f}  net(sum)={float(cd_diff.sum()):.4f}  "
        f"expect_spin_loss={expect_spin_loss}"
    )

    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )

    train_loader = make_loader(files, DATA_DIR, config)
    val_loader = make_loader(files, DATA_DIR, config)
    test_loader = make_loader(files, DATA_DIR, config)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    # The steps log "<split>/spin_loss" only when the spin loss was actually
    # computed and added — so its presence/absence proves the gate's behavior.
    spin_metrics = {
        k: float(v)
        for k, v in trainer.callback_metrics.items()
        if "/spin_" in k
    }
    print(f"[smoke] {data_file}: logged spin metrics: {spin_metrics}")
    return spin_metrics


def main():
    config = build_config()
    set_all_seeds(config["seed"])
    config = set_all_paths(config, None)  # creates output dirs, fills path keys
    print(f"[smoke] spin_type={config['spin_type']}")

    # Spin-polarized: NMAE branch.
    spin_metrics = run_case(config, SPIN_FILE, expect_spin_loss=True)
    assert spin_metrics, "no spin metrics logged — the spin loss did not run"
    assert any("spin_nmae" in k for k in spin_metrics), (
        f"expected the NMAE branch on a magnetic system, got {spin_metrics}"
    )
    assert all(
        torch.isfinite(torch.tensor(v)) for v in spin_metrics.values()
    ), f"non-finite spin metrics: {spin_metrics}"

    # Zero cd_diff: absolute-L1 branch. Loss is logged (and trained on), NMAE is not.
    nospin_metrics = run_case(config, NOSPIN_FILE, expect_spin_loss=True)
    assert any("spin_loss" in k for k in nospin_metrics), (
        f"spin_loss must still be logged on a non-magnetic system: {nospin_metrics}"
    )
    assert not any("spin_nmae" in k for k in nospin_metrics), (
        f"spin_nmae is meaningless against a zero target but was logged: {nospin_metrics}"
    )

    print("[smoke] PASS: NMAE branch on the magnetic file, absolute-L1 branch on the zero-grid file.")


if __name__ == "__main__":
    main()
