from neural_init.spin_electrafi.utils.custom_vasp_loader import CustomVaspChargeDensity
from neural_init.spin_electrafi.tools.density_conversions import cd_to_chgcar_mat
import lz4
import tempfile
import os
import numpy as np
import torch
from typing import Optional, Tuple
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
try:  # plotting only; not needed for training or inference
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None
import colorsys


def create_chg_delta(pred_dens_file: str,
                     true_dens_file: str,
                     delta_folder: str,
                     name_iter_str: str):
    vcd_pred = CustomVaspChargeDensity(pred_dens_file)
    with lz4.frame.open(true_dens_file, mode='rb') as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    tmpfile = os.fdopen(tmpfd, "wb")
    tmpfile.write(filecontent)
    tmpfile.close()
    vcd_true = CustomVaspChargeDensity(tmppath)
    os.remove(tmppath)

    cd_pred = np.array(vcd_pred.chg, dtype=np.float64).squeeze(axis=0)
    cd_true = np.array(vcd_true.chg, dtype=np.float64).squeeze(axis=0)
    atoms = vcd_true.atoms[0]

    delta = cd_true - cd_pred
    filename = f'{delta_folder}/{name_iter_str}_DELTA.CHGCAR'
    cd_to_chgcar_mat(original_file=true_dens_file, atoms=atoms, cd=delta, filename=filename)

# ----------------------- helpers -----------------------

def _to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _repeat_parent_indices(n_multiples: np.ndarray) -> np.ndarray:
    idx = [np.full(int(n), i, dtype=np.int64) for i, n in enumerate(n_multiples.tolist())]
    return np.concatenate(idx, axis=0) if len(idx) else np.empty((0,), dtype=np.int64)

def _cell_segments(cell: np.ndarray):
    """Return x,y,z lists for 12 box edges."""
    corners_f = np.array([
        [0,0,0],[1,0,0],[0,1,0],[0,0,1],
        [1,1,0],[1,0,1],[0,1,1],[1,1,1]
    ], dtype=float)
    A = np.array(cell, dtype=float)  # (3,3)
    pts = corners_f @ A
    edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [pts[i,0], pts[j,0], None]
        ys += [pts[i,1], pts[j,1], None]
        zs += [pts[i,2], pts[j,2], None]
    return xs, ys, zs

def _make_layout(title: str, all_pts: np.ndarray):
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    span = float(max((maxs - mins).max(), 1e-9))
    center = (maxs + mins) / 2.0
    ranges = np.column_stack([center - span*0.55, center + span*0.55])

    return go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x (Å)", range=ranges[0].tolist(), showgrid=False, zeroline=False),
            yaxis=dict(title="y (Å)", range=ranges[1].tolist(), showgrid=False, zeroline=False),
            zaxis=dict(title="z (Å)", range=ranges[2].tolist(), showgrid=False, zeroline=False),
            aspectmode="cube",
            camera=dict(projection=dict(type="orthographic")),  # <— add this
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )

def _save_html(fig: go.Figure, atoms: Atoms, save_dir: Optional[str], base_name: Optional[str],
               err_value: Optional[float], suffix: str) -> Optional[str]:
    if not save_dir:
        return None
    os.makedirs(save_dir, exist_ok=True)
    chem = atoms.get_chemical_formula(mode="reduce")
    n = len(atoms)
    err_tag = "" if err_value is None else f"_err{err_value*100:.2f}pct"
    base = base_name or chem
    out_path = os.path.join(save_dir, f"{base}_N{n}{err_tag}_{suffix}.html")
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path

def _title_with_error(base_title: str, err_value: Optional[float]) -> str:
    if err_value is None:
        return base_title
    return f"{base_title}<br><sup>error={err_value:.4f} ({err_value*100:.2f}%)</sup>"

# -------- element styles (colors & sizes) --------

def _build_atom_style_dicts() -> Tuple[dict, dict]:
    """
    Per-element color and size dicts for Z in [1..119].
    - Color: HSV wheel for diversity (kept).
    - Size: proportional to ASE covalent radii (in Å), mapped to px.
    """
    Zmax = 119
    atom_colors, atom_sizes = {}, {}
    # scale radii (Å) → pixels (bigger overall per your request)
    base_px = 8.0   # additive baseline
    scale_px = 22.0 # multiplier for radius
    for Z in range(1, Zmax + 1):
        # color via HSV wheel
        hue = (Z - 1) / Zmax
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.90)
        atom_colors[Z] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        # size from covalent radius (fallback if missing/zero)
        rad = float(covalent_radii[Z]) if Z < len(covalent_radii) else 0.7
        if not np.isfinite(rad) or rad <= 0:
            rad = 0.7
        px = base_px + scale_px * rad
        atom_sizes[Z] = px*3
    return atom_colors, atom_sizes

ATOM_COLORS, ATOM_SIZES = _build_atom_style_dicts()

# ------------------ public functions -------------------

def plot_atoms_with_displacements(
    atoms: Atoms,
    mu,
    n_multiples,
    *,
    cell: Optional[np.ndarray] = None,
    ax=None,                      # kept for compatibility; ignored
    arrow_scale: float = 1.0,     # absolute total arrow length in Å (all arrows same)
    atom_size: int = 40,          # fallback px if Z missing
    max_arrows: Optional[int] = None,
    # saving
    save_dir: Optional[str] = None,
    base_name: Optional[str] = None,
    structure_error: Optional[float] = None,
    dpi: int = 150,               # kept for compatibility; ignored
):
    """
    HTML (Plotly) visualization of atoms and displacement arrows (mu - atom_center).
    Arrows: straight shaft (line) + small cone tip; fixed total length; dark blue.
    Returns (None, saved_path_or_None).
    """
    mu = _to_np(mu)
    n_multiples = _to_np(n_multiples).astype(np.int64)
    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    cell = np.array(cell.cpu() if cell is not None else atoms.cell.array, dtype=float)

    parent_idx = _repeat_parent_indices(n_multiples)
    vec = mu - pos[parent_idx]

    if max_arrows is not None and len(vec) > max_arrows:
        sel = np.linspace(0, len(vec)-1, max_arrows, dtype=int)
        mu = mu[sel]; parent_idx = parent_idx[sel]; vec = vec[sel]

    # normalize → fixed-length vectors
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vec / norms
    L = float(max(1e-9, arrow_scale))
    tip_frac = 0.18                        # fraction of L devoted to the cone tip
    tip_len = L * tip_frac
    shaft_len = L - tip_len

    tails = pos[parent_idx]
    shaft_ends = tails + unit * shaft_len
    heads = tails + unit * L               # full length head (for completeness)

    # cell box
    cell_x, cell_y, cell_z = _cell_segments(cell)
    box = go.Scatter3d(x=cell_x, y=cell_y, z=cell_z, mode="lines", line=dict(color="gray", width=3), hoverinfo="skip")

    # atoms with per-element color & size (larger than before via covalent radii scaling)
    atom_colors = [ATOM_COLORS.get(int(z), "black") for z in Z]
    atom_sizes = [ATOM_SIZES.get(int(z), atom_size) for z in Z]
    atoms_sc = go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2],
        mode="markers",
        marker=dict(size=atom_sizes, color=atom_colors),
        text=[chemical_symbols[int(z)] if int(z) < len(chemical_symbols) else f"Z{int(z)}" for z in Z],
        hoverinfo="text+x+y+z",
        name="atoms"
    )

    # arrow shafts as lines
    DARK_BLUE = "rgb(0, 70, 140)"  # darker than before
    xs, ys, zs = [], [], []
    for p0, p1 in zip(tails, shaft_ends):
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]
    shafts = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=DARK_BLUE, width=6),
        hoverinfo="skip",
        name="displacement shafts",
    )

    DARK_BLUE = "rgb(0, 70, 140)"

    cones = go.Cone(
        x=shaft_ends[:, 0], y=shaft_ends[:, 1], z=shaft_ends[:, 2],
        # direction only; magnitude is ignored in absolute mode
        u=unit[:, 0], v=unit[:, 1], w=unit[:, 2],
        anchor="tail",  # place cone base at the end of the shaft
        sizemode="absolute",
        sizeref=float(tip_len),  # <- actual cone length in Å
        showscale=False,
        colorscale=[[0.0, DARK_BLUE], [1.0, DARK_BLUE]],
        opacity=0.95,
        name="displacement tips",
    )

    all_pts = np.vstack([pos, mu])
    fig = go.Figure(
        data=[box, atoms_sc, shafts, cones],
        layout=_make_layout(_title_with_error("Atoms + (fixed-length) displacement arrows", structure_error), all_pts)
    )

    out_path = _save_html(fig, atoms, save_dir, base_name, structure_error, suffix="arrows")
    return None, out_path


# --- NEW unified function ---

def plot_atoms_with_gaussian_shapes(
    atoms: Atoms,
    mu,
    weights,
    *,
    Sigma=None,
    mode: str = "ellipsoids",     # "ellipsoids" | "spheres"
    iso_sigma: float = 1.5,       # contour scale for ellipsoids: axes = iso_sigma * sqrt(eigs)
    mesh_nu: int = 20,
    mesh_nv: int = 12,
    alpha: float = 0.5,
    atom_size: int = 40,
    sphere_scale: float = 1.0,    # used only in mode="spheres"
    max_shapes: Optional[int] = 400,  # cap to avoid too many traces
    # saving
    save_dir: Optional[str] = None,
    base_name: Optional[str] = None,
    structure_error: Optional[float] = None,
    dpi: int = 150,               # compatibility only
):
    """
    Visualize atoms + Gaussians. If mode='ellipsoids' and Sigma is provided,
    draw an ellipsoid isosurface per Gaussian using eigen-decomposition of Sigma.
    Otherwise fall back to spheres.

    Returns (None, saved_path_or_None).
    """
    mu = _to_np(mu)
    w  = _to_np(weights).reshape(-1)
    w = 100*w/abs(w.mean())
    pos = atoms.get_positions()
    Z   = atoms.get_atomic_numbers()
    cell = np.array(Sigma.device if (hasattr(Sigma, "device") and Sigma is not None) else atoms.cell.array)  # noqa
    cell = np.array(atoms.cell.array, dtype=float)

    # ---- common: cell + atoms ----
    cell_x, cell_y, cell_z = _cell_segments(cell)
    box = go.Scatter3d(x=cell_x, y=cell_y, z=cell_z, mode="lines",
                       line=dict(color="gray", width=3), hoverinfo="skip")
    atom_colors = [ATOM_COLORS.get(int(z), "black") for z in Z]
    atom_sizes  = [ATOM_SIZES.get(int(z), atom_size) for z in Z]
    atoms_sc = go.Scatter3d(
        x=pos[:,0], y=pos[:,1], z=pos[:,2],
        mode="markers",
        marker=dict(size=atom_sizes, color=atom_colors),
        text=[chemical_symbols[int(z)] if int(z) < len(chemical_symbols) else f"Z{int(z)}" for z in Z],
        hoverinfo="text+x+y+z",
        name="atoms"
    )

    data_traces = [box, atoms_sc]

    # ---- ellipsoids path ----
    if (mode.lower() == "ellipsoids") and (Sigma is not None):
        Sig = _to_np(Sigma).reshape(-1, 3, 3)

        # optional cap (sample evenly)
        idx_all = np.arange(len(mu))
        if (max_shapes is not None) and (len(idx_all) > max_shapes):
            idx_all = np.linspace(0, len(mu)-1, max_shapes, dtype=int)

        mu = mu[idx_all]
        w  = w[idx_all]
        Sig = Sig[idx_all]

        # eigen-decomp → axes
        eigs, vecs = _eigsorted_sym3(Sig, eps=1e-12)      # eigs asc
        sigmas = np.sqrt(eigs)                             # (M,3), Å
        axes = iso_sigma * sigmas                          # isosurface scale

        # unit-sphere template
        xs_u, ys_u, zs_u, I, J, K = _uv_sphere(nu=mesh_nu, nv=mesh_nv)

        # color map by sign
        col_pos = "rgb(31,119,180)"
        col_neg = "rgb(214,39,40)"

        # build one Mesh3d per Gaussian (keeps hover info simple & correct)
        for c, ax, R, wi in zip(mu, axes, vecs, w):
            X, Y, Zz = _ellipsoid_vertices(c.astype(float), ax.astype(float), R.astype(float), xs_u, ys_u, zs_u)
            color = col_pos if wi >= 0 else col_neg
            # informative hover: weight and axis lengths
            txt = f"w={wi:.4g}<br>axes(Å)={tuple(ax.tolist())}<br>cond={float((ax.max()/(ax.min()+1e-12))):.3g}"
            m = go.Mesh3d(
                x=X, y=Y, z=Zz, i=I, j=J, k=K,
                color=color, opacity=float(alpha), flatshading=True,
                hoverinfo="text+x+y+z", text=txt, name="gaussian"
            )
            data_traces.append(m)

        title = _title_with_error("Atoms + Gaussian ellipsoids (blue=+, red=−)", structure_error)
        all_pts = np.vstack([pos, mu])
        fig = go.Figure(data=data_traces, layout=_make_layout(title, all_pts))
        out_path = _save_html(fig, atoms, save_dir, base_name, structure_error, suffix="ellipsoids")
        return None, out_path

    # ---- fallback spheres (existing look) ----
    GAUSS_PX = 38.0 * float(sphere_scale)
    size_px = np.full(len(mu), GAUSS_PX)
    colors = np.where(w >= 0.0, "rgb(31,119,180)", "rgb(214,39,40)")
    gauss_sc = go.Scatter3d(
        x=mu[:,0], y=mu[:,1], z=mu[:,2],
        mode="markers",
        marker=dict(size=size_px, color=colors, opacity=float(alpha)),
        name="gaussians",
        text=[f"w={wi:.4g}" for wi in w],
        hoverinfo="text+x+y+z"
    )
    data_traces.append(gauss_sc)
    title = _title_with_error("Atoms + Gaussian spheres (blue=+, red=−)", structure_error)
    all_pts = np.vstack([pos, mu])
    fig = go.Figure(data=data_traces, layout=_make_layout(title, all_pts))
    out_path = _save_html(fig, atoms, save_dir, base_name, structure_error, suffix="spheres")
    return None, out_path

# --- add these helpers near the top with your other helpers ---

def _eigsorted_sym3(S: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Symmetrize then eigen-decompose a stack of 3x3 SPD-ish matrices.
    Returns (eigs_sorted, vecs_sorted) with eigs ascending.
    Shapes:
      S: (M,3,3)
      eigs: (M,3)
      vecs: (M,3,3) columns are eigenvectors matching eigs
    """
    M = S.shape[0]
    Ssym = 0.5 * (S + np.transpose(S, (0,2,1)))
    eigs = np.empty((M, 3), dtype=float)
    vecs = np.empty((M, 3, 3), dtype=float)
    for i in range(M):
        w, V = np.linalg.eigh(Ssym[i])
        w = np.clip(w, eps, None)  # clamp tiny/negative
        # sort ascending
        idx = np.argsort(w)
        eigs[i] = w[idx]
        vecs[i] = V[:, idx]
    return eigs, vecs  # eigs asc; vecs columns aligned

def _uv_sphere(nu: int = 20, nv: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a unit sphere triangulation using (u,v) grid.
    Returns vertices (x,y,z) and Mesh3d triangles (i,j,k) as 1D arrays.
    """
    u = np.linspace(0.0, 2.0 * np.pi, num=nu, endpoint=False)
    v = np.linspace(0.0, np.pi, num=nv)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)

    # build triangles on the grid
    def idx(a, b):  # wrap in u (columns), clamp in v (rows)
        return (a % nu) + b * nu

    I = []
    J = []
    K = []
    for b in range(nv - 1):
        for a in range(nu):
            a1 = (a + 1) % nu
            i0 = idx(a, b);  i1 = idx(a1, b)
            j0 = idx(a, b+1); j1 = idx(a1, b+1)
            # two triangles (quad)
            I += [i0, i1]
            J += [j0, j1]
            K += [j1, j0]
    return x.ravel(), y.ravel(), z.ravel(), np.array(I), np.array(J), np.array(K)

def _ellipsoid_vertices(center: np.ndarray, axes: np.ndarray, R: np.ndarray,
                        xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map unit-sphere vertices (xs,ys,zs) to an ellipsoid:
      center + R @ diag(axes) @ [x,y,z]
    Inputs:
      center: (3,) in Å
      axes:   (3,) lengths along principal axes (Å), >=0
      R:      (3,3) rotation whose columns are eigenvectors
    Returns:
      X,Y,Z flattened arrays
    """
    U = np.vstack((xs, ys, zs))              # (3, P)
    T = (R @ (axes[:, None] * U)) + center[:, None]  # (3, P)
    return T[0], T[1], T[2]
