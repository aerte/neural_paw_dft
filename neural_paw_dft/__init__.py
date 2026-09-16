"""Neural electronic initialization for VASP.

Subpackages:
  augnet          PAW augmentation-occupancy model (MACE backbone)
  spin_electrafi  total + spin charge-density model (EScAIP backbone, Gaussian readout)
  vasp_runner     CHGCAR channel surgery, INCAR/OSZICAR helpers, VASP experiment plumbing
  pipeline        structure -> ML-seeded VASP inputs (CHGCAR, INCAR, POSCAR, POTCAR, KPOINTS)
"""
import os as _os

# e3nn 0.4.x torch.load()s cached constants at import; torch>=2.6 defaults to
# weights_only=True and rejects them. Every subpackage import passes through here.
_os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

__version__ = "0.1.0"
