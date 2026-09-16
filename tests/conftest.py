"""Shared helpers. Weights-gated tests skip unless the files are present or
NDI_TEST_DOWNLOAD_WEIGHTS=1 lets them be fetched from the Hub (needs access to the weights repo)."""
import os

from neural_paw_dft.models import REGISTRY, resolve_weights, weights_dir


def weights_available(*names: str) -> bool:
    root = weights_dir()
    if all((root / REGISTRY[n]).is_file() for n in names):
        return True
    if os.environ.get("NDI_TEST_DOWNLOAD_WEIGHTS") == "1":
        for n in names:
            resolve_weights(n)
        return True
    return False
