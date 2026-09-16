"""OSZICAR SCF-step parsing."""
import re


def count_scf_breakdown_from_oszicar(osz_path: str):
    """Count DAV, RMM, and other tagged iterations in an OSZICAR.

    Returns:
        (dav, rmm, total, other_counts)
        - other_counts: dict like {"HMM": 3, "CG": 1}
        - total = dav + rmm + sum(other_counts.values())
    If the file is missing, returns (0, 0, 0, {}).
    """
    dav = 0
    rmm = 0
    other_counts: dict[str, int] = {}

    try:
        with open(osz_path, "r") as f:
            for line in f:
                ls = line.lstrip()
                m = re.match(r"([A-Z]+):", ls)
                if not m:
                    continue
                tag = m.group(1)
                if tag == "DAV":
                    dav += 1
                elif tag == "RMM":
                    rmm += 1
                else:
                    other_counts[tag] = other_counts.get(tag, 0) + 1
    except FileNotFoundError:
        pass

    total = dav + rmm + sum(other_counts.values())
    return dav, rmm, total, other_counts
