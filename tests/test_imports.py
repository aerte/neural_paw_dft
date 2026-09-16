"""Package imports work from any CWD and the removed dependencies are really gone."""
import importlib
import subprocess
import sys

import pytest

MODULES = [
    "neural_paw_dft",
    "neural_paw_dft.models",
    "neural_paw_dft._resources",
    "neural_paw_dft.vasp_runner.chgcar",
    "neural_paw_dft.vasp_runner.sources.mp",
    "neural_paw_dft.vasp_runner.relaxmag",
    "neural_paw_dft.augnet.augnet_model",
    "neural_paw_dft.augnet.paw_basis_transform",
    "neural_paw_dft.augnet.model",
    "neural_paw_dft.augnet.train_augnet",
    "neural_paw_dft.spin_electrafi.model.escaip.utils.fairchem_graph",
    "neural_paw_dft.spin_electrafi.model.escaip.utils.smearing",
    "neural_paw_dft.spin_electrafi.model.ELECTRAFI",
    "neural_paw_dft.pipeline",
    "neural_paw_dft.pipeline.cli",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_no_fairchem_or_torch_scatter_needed():
    code = (
        "import sys, neural_paw_dft.spin_electrafi.model.ELECTRAFI, neural_paw_dft.pipeline.augnet;"
        "assert 'fairchem' not in sys.modules and 'torch_scatter' not in sys.modules;"
        "assert 'neural_paw_dft.augnet.train_augnet' not in sys.modules  # inference must not import the trainer"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd="/")


def test_packaged_data_present():
    from neural_paw_dft._resources import resource_path, resolve_data_path

    assert resource_path("augnet", "stats/paw_ref_freeatom.pt").stat().st_size > 1_000_000
    assert resource_path("spin_electrafi", "model/escaip/escaip_config.yaml").is_file()
    # the shipped yaml value resolves from any CWD
    assert resolve_data_path("./model/escaip/escaip_config.yaml", "spin_electrafi").is_file()
    assert resolve_data_path("./data_splits/datasplits_mpfull2025.json", "spin_electrafi").is_file()
    assert resolve_data_path("stats/paw_ref_freeatom.pt", "augnet").is_file()
