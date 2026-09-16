"""Make a fresh-head twin of a sharedref checkpoint.

Loads a Lightning checkpoint (head: shared), re-initializes the SharedPAWHead
parameters with repo-code init, keeps the backbone untouched, and writes the
result next to it as <stem>_freshhead.ckpt. This is how
trained_models/full_freshhead.ckpt was made; regenerate it whenever the
base checkpoint changes (the final model weights).

The head is rebuilt from the config stored in the checkpoint's
hyper_parameters, so it works for any width/max_ell/rank without edits. Only
the head's nn.Parameters are replaced (3 for the hybrid-by-L head:
to_coeffs.weight, coupling_weight, linear_head.weight); buffers and everything
under model.backbone.* are copied through bit-identically.

    .venv/bin/python scripts/make_freshhead_ckpt.py trained_models/full.ckpt
"""

import argparse
import sys
from pathlib import Path

import torch


from neural_paw_dft.augnet.paw_head_shared import SharedPAWHead
from neural_paw_dft.augnet.train_augnet import hidden_irreps_from_width, seed_everything

from e3nn import o3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="Base checkpoint (head: shared).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: <ckpt stem>_freshhead.ckpt).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the fresh head init.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt_path.with_name(
        ckpt_path.stem + "_freshhead.ckpt")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["hyper_parameters"]["config"]["model"]
    if mcfg["head"] != "shared":
        raise ValueError(f"checkpoint head is {mcfg['head']!r}, expected 'shared'")

    # Same irreps the training path hands to SharedPAWHead:
    # MACEBackbone.node_feats_irreps = (hidden_irreps * num_interactions).simplify()
    hidden = hidden_irreps_from_width(mcfg["hidden_width"], mcfg["max_ell"])
    node_feats_irreps = (o3.Irreps(hidden) * mcfg["num_interactions"]).simplify()

    seed_everything(args.seed)
    head = SharedPAWHead(
        hidden_irreps=str(node_feats_irreps),
        rank=mcfg["head_rank"],
        block_mixing=mcfg["head_block_mixing"],
        linear_max_l=mcfg["head_linear_max_l"],
    )

    state = ckpt["state_dict"]
    replaced = []
    for name, param in head.named_parameters():
        key = f"model.paw_head.{name}"
        if key not in state:
            raise KeyError(f"{key} not in checkpoint state_dict")
        if state[key].shape != param.shape:
            raise ValueError(f"{key}: checkpoint {tuple(state[key].shape)} vs "
                             f"fresh head {tuple(param.shape)}")
        state[key] = param.detach().clone()
        replaced.append(key)

    if not replaced:
        raise RuntimeError("no head parameters replaced")
    torch.save(ckpt, out_path)
    print(f"replaced {len(replaced)} head parameters: {', '.join(replaced)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
