# tools/valence_picker.py
# Reuse your canonical tables (single source of truth)
from __future__ import annotations
from typing import Dict, Tuple
import torch

def required_valence_slots(VALENCE_OPTIONS: Dict[str, list[int]]) -> int:
    # how many distinct slots you need overall (max options length)
    return max(len(v) for v in VALENCE_OPTIONS.values()) if VALENCE_OPTIONS else 1

def _slot_for_choice(element: str, chosen_valence: int, VALENCE_OPTIONS, valence_dict) -> int:
    opts = VALENCE_OPTIONS.get(element, [valence_dict.get(element, 0)])
    try:
        return opts.index(chosen_valence)
    except ValueError:
        return 0

@torch.no_grad()
def pick_valence_slots(
    atoms,
    *,
    Ne_target: float | None,
    tol: float = 1e-6,
    device: torch.device | None = None,
    VALENCE_OPTIONS=None,
    valence_dict=None,
) -> Tuple[torch.Tensor | None, Dict[str, int], bool, float]:
    assert VALENCE_OPTIONS is not None and valence_dict is not None

    # element_counts inline (or pass it too)
    from ase.data import chemical_symbols
    counts: Dict[str,int] = {}
    for z in atoms.numbers:
        s = chemical_symbols[int(z)]
        counts[s] = counts.get(s, 0) + 1

    # ---- bounded DFS over per-element canonical options ----
    elems = list(counts.keys())

    def opts_for(e: str):
        return VALENCE_OPTIONS.get(e, [valence_dict.get(e, 0)])

    # sort by pruning power (count * range)
    elems.sort(key=lambda e: counts[e] * (max(opts_for(e)) - min(opts_for(e))), reverse=True)

    best_assign: Dict[str, int] = {}
    best_delta = float("inf")
    exact = False

    # precompute min/max for fast pruning
    rem_min = {}
    rem_max = {}
    def min_max_sum(from_idx: int, partial: float) -> tuple[float,float]:
        # compute min/max of remaining contribution
        rem = elems[from_idx:]
        mn = sum(counts[r] * min(opts_for(r)) for r in rem)
        mx = sum(counts[r] * max(opts_for(r)) for r in rem)
        return partial + mn, partial + mx

    def dfs(i: int, s: float, assign: Dict[str,int]):
        nonlocal best_assign, best_delta, exact
        if i == len(elems):
            delta = abs(s - Ne_target)
            if delta + 1e-12 < best_delta:
                best_delta = delta
                best_assign = dict(assign)
                exact = (delta <= tol)
            return

        e = elems[i]
        options = opts_for(e)
        # try default first
        for val in options:
            s2 = s + counts[e] * val
            mn, mx = min_max_sum(i+1, s2)
            # prune if impossible to reach Ne_target
            if (s2 <= Ne_target <= mx) or (mn <= Ne_target <= s2) or (mn <= Ne_target <= mx):
                assign[e] = val
                dfs(i+1, s2, assign)
                del assign[e]
            elif not exact:
                # looser exploration only until an exact is found
                assign[e] = val
                dfs(i+1, s2, assign)
                del assign[e]

    dfs(0, 0.0, {})
    chosen_total = sum(counts[e]*best_assign[e] for e in best_assign) if best_assign else 0.0
    delta = float(chosen_total - Ne_target)

    # ---- build per-atom slot tensor ----
    slots = []
    for z in atoms.numbers:
        sym = chemical_symbols[int(z)]
        v = best_assign.get(sym, valence_dict.get(sym, 0))
        slots.append(_slot_for_choice(sym, int(v), VALENCE_OPTIONS, valence_dict))
    slots_t = torch.tensor(slots, dtype=torch.long, device=device)

    return slots_t, best_assign, exact, delta


def total_from_assignment(counts: Dict[str,int], assignment: Dict[str,int]) -> int:
    return sum(counts[e] * assignment[e] for e in counts)