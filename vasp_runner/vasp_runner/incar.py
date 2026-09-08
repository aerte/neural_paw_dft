"""INCAR helpers and the spglib monkey-patch used across all runners."""
import os


def install_spglib_patch():
    """Patch Structure.get_space_group_info so MPStaticSet/Kpoints don't
    explode when spglib's symmetry path is broken.

    MPStaticSet only checks ``structure.get_space_group_info()[0][0] == "F"``
    to detect face-centered lattices, so returning ("P1", 1) is enough.
    """
    import pymatgen.core.structure as pmg_structure

    def _fake_get_space_group_info(self, symprec=0.01, angle_tolerance=5.0):
        return ("P1", 1)

    pmg_structure.Structure.get_space_group_info = _fake_get_space_group_info


def patch_lmaxmix_for_dftu(incar_path):
    """ChargE3Net-style LMAXMIX:
      - LDAUL contains 3 (f) → LMAXMIX = 6
      - else LDAUL contains 2 (d) → LMAXMIX = 4
      - only when DFT+U is actually on (LDAU = .TRUE.)
    """
    if not os.path.exists(incar_path):
        return

    with open(incar_path, "r") as f:
        lines = f.readlines()

    ldaul_line = None
    ldau_on = False
    for line in lines:
        key = line.split("=", 1)[0].strip().upper() if "=" in line else ""
        if key == "LDAU":
            if "T" in line.upper():
                ldau_on = True
        elif key == "LDAUL":
            ldaul_line = line

    if not ldau_on or ldaul_line is None:
        return

    try:
        ldaul_vals = [
            int(x) for x in ldaul_line.split("=", 1)[1].split()
            if x.strip().lstrip("+-").isdigit()
        ]
    except Exception:
        return

    has_f = any(l == 3 for l in ldaul_vals)
    has_d = any(l == 2 for l in ldaul_vals)
    if not (has_f or has_d):
        return

    desired = 6 if has_f else 4

    new_lines = []
    saw_lmaxmix = False
    for line in lines:
        key = line.split("=", 1)[0].strip().upper() if "=" in line else ""
        if key == "LMAXMIX":
            new_lines.append(f"LMAXMIX = {desired}\n")
            saw_lmaxmix = True
        else:
            new_lines.append(line)

    if not saw_lmaxmix:
        new_lines.append(f"LMAXMIX = {desired}\n")

    with open(incar_path, "w") as f:
        f.writelines(new_lines)


def patch_incar_ngxf(incar_path, grid_dims):
    """Overwrite/add NGXF/NGYF/NGZF in an INCAR to match `grid_dims`.

    Kept for parity with the legacy scripts; not currently called.
    """
    if grid_dims is None:
        return

    nx, ny, nz = grid_dims
    try:
        with open(incar_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return

    new_lines = []
    for line in lines:
        if "=" not in line:
            new_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip().upper()
        if key in {"NGXF", "NGYF", "NGZF"}:
            continue
        new_lines.append(line)

    new_lines.append(f"NGXF = {nx}\n")
    new_lines.append(f"NGYF = {ny}\n")
    new_lines.append(f"NGZF = {nz}\n")

    with open(incar_path, "w") as f:
        f.writelines(new_lines)
