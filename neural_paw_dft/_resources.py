"""Access to data files shipped inside the package (configs, stats, split lists)."""
from importlib.resources import files
from pathlib import Path


def resource_path(subpkg: str, rel: str) -> Path:
    """Absolute path of ``rel`` inside ``neural_paw_dft.<subpkg>``."""
    return Path(str(files(f"neural_paw_dft.{subpkg}").joinpath(rel)))


def resolve_data_path(value, subpkg: str) -> Path:
    """Resolve a config path the way the original scripts did, with a packaged fallback.

    A CWD-relative or absolute path that exists wins (old behaviour). Otherwise the
    same relative path is looked up inside ``neural_paw_dft.<subpkg>``, so shipped
    YAML values such as ``./model/escaip/escaip_config.yaml`` work from any CWD.
    """
    p = Path(value).expanduser()
    if p.is_file():
        return p
    rel = str(p)
    if rel.startswith("./"):
        rel = rel[2:]
    cand = resource_path(subpkg, rel)
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"{value!r} not found in CWD or inside neural_paw_dft.{subpkg}")
