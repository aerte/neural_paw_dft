# atom_tools_cubic.py
import torch
from collections import defaultdict

# --- Initial valence guesses for the Cubic dataset (MP-like POTCARs) ---
# Heuristic: early TMs and many s-block elements use _pv (include n p^6 semicore),
# but NOT the extra n s^2 of _sv. Late TMs and main-group use the common base sets.
# We will refine case-by-case from logs.

valence_dict = {
    # Period 1
    "H": 1, "He": 2,

    # Period 2
    "Li": 3,  # Li_sv commonly used in MP workflows
    "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,

    # Period 3
    "Na": 7,  # Na_pv
    "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,

    # Period 4 (3d + p-block)
    # 3d (use _pv: add 3p^6)
    "K": 9,    # K_sv is common in MP (pv would be 7; we’ll adjust if logs disagree)
    "Ca": 10,  # Ca_pv
    "Sc": 9,   # base 3 + 3p^6
    "Ti": 10,  # Ti_pv  (logs often align with 10 rather than 12)
    "V": 11,   # V_pv
    "Cr": 12,  # Cr_pv
    "Mn": 13,  # Mn_pv
    "Fe": 14,  # Fe_pv  (your Fe6S8 log strongly indicates 14)
    "Co": 15,  # Co_pv
    "Ni": 16,  # Ni_pv
    "Cu": 17,  # Cu_pv
    "Zn": 12,  # standard

    "Ga": 13, "Ge": 14, "As": 5, "Se": 6, "Br": 7, "Kr": 8,

    # Period 5 (4d + p-block)
    "Rb": 9,  # Rb_sv is common in MP (pv would be 7)
    "Sr": 10, # Sr_pv
    "Y": 11,  # often includes 4s^2 as well (sv-like); adjust if needed
    "Zr": 10, # Zr_pv (12 if _sv shows up in logs)
    "Nb": 11, # Nb_pv
    "Mo": 12, # Mo_pv (14 if _sv pops up)
    "Tc": 13, # Tc_pv
    "Ru": 14, # Ru_pv
    "Rh": 15, "Pd": 10, "Ag": 11, "Cd": 12,

    "In": 13, "Sn": 14, "Sb": 5, "Te": 6, "I": 7, "Xe": 8,

    # Period 6 (5d + lanthanides + p-block)
    "Cs": 9,  # Cs_sv typical; adjust if logs indicate 7
    "Ba": 10, # Ba_pv

    # Lanthanides: start trivalent-like defaults; adjust from logs as needed
    "La": 11, "Ce": 12, "Pr": 11, "Nd": 11, "Pm": 11, "Sm": 11,
    "Eu": 8,  # MP often uses ~8; if you see 17 in logs, we’ll add an option later
    "Gd": 9, "Tb": 9, "Dy": 9, "Ho": 9, "Er": 9, "Tm": 9, "Yb": 8, "Lu": 9,

    # 5d (prefer _pv: add 5p^6; NOT +5s^2 unless logs say so)
    "Hf": 10, "Ta": 11, "W": 12,  # W_pv (=12); if logs show 14, we’ll flip to _sv
    "Re": 13, "Os": 14, "Ir": 15, "Pt": 10, "Au": 11, "Hg": 12,

    "Tl": 13, "Pb": 14, "Bi": 15, "Po": 16, "At": 7, "Rn": 8,

    # Period 7 (actinides): placeholders close to common MP choices; refine from logs
    "Fr": 9, "Ra": 10,
    "Ac": 11, "Th": 12, "Pa": 13, "U": 14, "Np": 15, "Pu": 16, "Am": 17, "Cm": 18,
    "Bk": 3, "Cf": 3, "Es": 3, "Fm": 3, "Md": 3, "No": 3, "Lr": 3,

    # Superheavies (rare; placeholders)
    "Rf": 4, "Db": 5, "Sg": 6, "Bh": 7, "Hs": 8, "Mt": 9,
    "Ds": 10, "Rg": 11, "Cn": 12, "Nh": 3, "Fl": 4, "Mc": 5, "Lv": 6, "Ts": 7, "Og": 8,
}

# Build single-option lists (default only). We’ll keep k,opts empty for now.
VALENCE_OPTIONS = defaultdict(set)
for k, v in valence_dict.items():
    VALENCE_OPTIONS[k].add(int(v))

# Add right before you "freeze to sorted lists"
for k, opts in {
    "Bi": {5, 15},
    "Mg": {2, 8},     # std vs Mg_pv
    "Eu": {8, 17},    # lighter vs heavier Eu PAW
    "Co": {9, 15},    # std vs Co_pv
    "Zr": {10, 12},   # Zr_pv vs Zr_sv
    "Gd": {9, 18},    # lighter vs heavier Gd PAW
    "Ir": {9, 15},    # std vs Ir_pv-like
    "Be": {2, 4},     # std vs Be_sv
    "Sc": {9, 11},    # Sc_pv vs Sc_sv
}.items():
    VALENCE_OPTIONS[k].update(opts)


# Freeze to sorted lists (put default first for stable slot mapping)
VALENCE_OPTIONS = {
    el: [valence_dict.get(el, next(iter(vals)))] + sorted([x for x in vals if x != valence_dict.get(el, None)])
    for el, vals in VALENCE_OPTIONS.items()
}

# --------------- Helper utilities (same interface as your main tools) ---------------

def element_counts(atoms) -> dict[str, int]:
    from ase.data import chemical_symbols
    counts: dict[str, int] = {}
    for z in atoms.numbers:
        s = chemical_symbols[int(z)]
        counts[s] = counts.get(s, 0) + 1
    return counts

def _total_electrons_from_assignment(counts: dict[str, int], assign: dict[str, int]) -> int:
    return sum(counts[e] * assign[e] for e in counts)

def _solve_valence_assignment(counts: dict[str, int], Ne_target: float, tol: float = 1e-6) -> tuple[dict[str, int], bool]:
    """
    With only a single default per element, this either hits exactly or returns the closest sum.
    We keep the DFS structure to stay drop-in compatible with your existing code.
    """
    elems = list(counts.keys())
    best_assign = {}
    best_delta = float("inf")
    exact = False

    def dfs(idx, partial_assign, partial_sum):
        nonlocal best_assign, best_delta, exact
        if idx == len(elems):
            delta = abs(partial_sum - Ne_target)
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_assign = dict(partial_assign)
                exact = delta <= tol
            return

        e = elems[idx]
        opts = VALENCE_OPTIONS.get(e, [valence_dict.get(e, 0)])
        for val in opts:  # currently just one
            s = partial_sum + counts[e] * val
            partial_assign[e] = val
            dfs(idx + 1, partial_assign, s)
            del partial_assign[e]

    dfs(0, {}, 0.0)
    return best_assign, exact

def _slot_for_choice(element: str, chosen_valence: int) -> int:
    opts = VALENCE_OPTIONS.get(element, [valence_dict.get(element, 0)])
    try:
        return opts.index(chosen_valence)
    except ValueError:
        return 0
