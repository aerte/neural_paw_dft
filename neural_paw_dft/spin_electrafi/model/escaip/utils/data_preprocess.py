from functools import partial

import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
from .smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

from ..custom_types import GraphAttentionData
from ..configs import (
    GlobalConfigs,
    MolecularGraphConfigs,
    GraphNeuralNetworksConfigs,
)
from .graph_utils import (
    get_node_direction_expansion,
    convert_neighbor_list,
    map_neighbor_list,
    get_attn_mask,
    patch_singleton_atom,
    pad_batch,
)


def data_preprocess(
    data,
    generate_graph_fn: callable,
    global_cfg: GlobalConfigs,
    gnn_cfg: GraphNeuralNetworksConfigs,
    molecular_graph_cfg: MolecularGraphConfigs,
) -> GraphAttentionData:
    # atomic numbers
    atomic_numbers = data.atomic_numbers.long()

    # NEW — prefer species_ids if present (Z + slot*MAX_Z), else fall back to Z
    species_ids_input = getattr(data, "species_ids", None)
    if species_ids_input is None:
        species_ids = atomic_numbers
    else:
        species_ids = species_ids_input.long()

    # edge distance expansion
    expansion_func = {
        "gaussian": GaussianSmearing,
        "gaussian_rbf": GaussianRBF,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[molecular_graph_cfg.distance_function]

    edge_distance_expansion_func = expansion_func(
        0.0,
        molecular_graph_cfg.max_radius,
        gnn_cfg.edge_distance_expansion_size,
        basis_width_scalar=2.0,
    ).to(data.pos.device)

    # generate graph
    graph = generate_graph_fn(data)

    # sort edge index according to receiver node
    edge_index, edge_attr = torch_geometric.utils.sort_edge_index(
        graph.edge_index,
        [graph.edge_distance, graph.edge_distance_vec],
        sort_by_row=False,
    )
    edge_distance, edge_distance_vec = edge_attr[0], edge_attr[1]
    edge_distances_raw = edge_distance

    # edge directions (for direct force prediction, ref: gemnet)
    edge_direction = -edge_distance_vec / edge_distance[:, None]

    # edge distance expansion (ref: scn)
    edge_distance_expansion = edge_distance_expansion_func(edge_distance)

    # node direction expansion
    node_direction_expansion = get_node_direction_expansion(
        distance_vec=edge_distance_vec,
        edge_index=edge_index,
        lmax=gnn_cfg.node_direction_expansion_size - 1,
        num_nodes=data.num_nodes,
    )

    # convert to neighbor list
    neighbor_list, neighbor_mask, index_mapping = convert_neighbor_list(
        edge_index, molecular_graph_cfg.max_neighbors, data.num_nodes
    )

    # map neighbor list
    map_neighbor_list_ = partial(
        map_neighbor_list,
        index_mapping=index_mapping,
        max_neighbors=molecular_graph_cfg.max_neighbors,
        num_nodes=data.num_nodes,
    )
    edge_direction = map_neighbor_list_(edge_direction)
    edge_distance_expansion = map_neighbor_list_(edge_distance_expansion)
    edge_distance_nm = map_neighbor_list_(edge_distance.unsqueeze(-1)).squeeze(-1)  # (N, M)

    # pad batch
    if global_cfg.use_padding:
        (
            atomic_numbers,
            node_direction_expansion,
            edge_distance_expansion,
            edge_direction,
            edge_distance_nm,
            neighbor_list,
            neighbor_mask,
            node_batch,
            node_padding_mask,
            graph_padding_mask,
        ) = pad_batch(
            max_num_nodes_per_batch=molecular_graph_cfg.max_num_nodes_per_batch,
            atomic_numbers=atomic_numbers,
            node_direction_expansion=node_direction_expansion,
            edge_distance_expansion=edge_distance_expansion,
            edge_direction=edge_direction,
            edge_distance=edge_distance_nm,
            neighbor_list=neighbor_list,
            neighbor_mask=neighbor_mask,
            node_batch=data.batch,
            num_graphs=data.num_graphs,
            batch_size=global_cfg.batch_size,
        )
        # NEW: pad positions to match node_padding_mask length
        total_nodes = node_padding_mask.shape[0]
        need = total_nodes - data.pos.shape[0]
        assert need >= 0, "pos longer than padded node length"
        pos_padded = F.pad(data.pos, (0, 0, 0, need), value=0.0)  # (total_nodes, 3)
        # NEW — pad species_ids the same way atomic_numbers were padded
        species_ids = F.pad(species_ids, (0, need), value=0).long()
    else:
        node_padding_mask = torch.ones_like(atomic_numbers, dtype=torch.bool)
        graph_padding_mask = torch.ones(
            data.num_graphs, dtype=torch.bool, device=data.batch.device
        )
        node_batch = data.batch
        pos_padded = data.pos  # no padding used

    # patch singleton atom
    edge_direction, neighbor_list, neighbor_mask = patch_singleton_atom(
        edge_direction, neighbor_list, neighbor_mask
    )

    # get attention mask
    attn_mask, angle_embedding = get_attn_mask(
        edge_direction=edge_direction,
        neighbor_mask=neighbor_mask,
        num_heads=gnn_cfg.atten_num_heads,
        use_angle_embedding=gnn_cfg.use_angle_embedding,
    )

    if gnn_cfg.atten_name in ["memory_efficient", "flash", "math"]:
        # The fused kernels are CUDA-only; on CPU keep the math kernel or SDPA has no backend.
        on_cuda = edge_direction.is_cuda
        torch.backends.cuda.enable_flash_sdp(on_cuda and gnn_cfg.atten_name == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(
            on_cuda and gnn_cfg.atten_name == "memory_efficient"
        )
        torch.backends.cuda.enable_math_sdp((not on_cuda) or gnn_cfg.atten_name == "math")
    else:
        raise NotImplementedError(
            f"Attention name {gnn_cfg.atten_name} not implemented"
        )

    # construct input data
    x = GraphAttentionData(
        atomic_numbers=atomic_numbers,
        species_ids=species_ids,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction=edge_direction,
        edge_distance=edge_distance_nm,  # (N, M)  <-- NEW
        attn_mask=attn_mask,
        angle_embedding=angle_embedding,
        neighbor_list=neighbor_list,
        neighbor_mask=neighbor_mask,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        graph_padding_mask=graph_padding_mask,
        pos=pos_padded,  # (N, 3)
        cell=getattr(data, "cell", None),  # (1, 3, 3) in your atoms_to_pyg
        pbc=getattr(data, "pbc", None),  # (1, 3)
    )
    return x


class GaussianRBF(nn.Module):
    """
    Log-spaced centers in [r_min, r_cut], optional learnable widths,
    and a smooth C2 cutoff envelope so features/gradients vanish at r_cut.
    """
    def __init__(self, start: float, stop: float, num_gaussians: int,
                 basis_width_scalar: float = 1.0, r_min: float = 1e-3,
                 learnable_width: bool = True) -> None:
        super().__init__()
        assert start <= 0.0 and stop > 0.0, "pass start=0.0, stop=r_cut"
        r_cut = float(stop)
        # log-spaced centers
        mu = torch.logspace(torch.log10(torch.tensor(max(r_min, 1e-3))),
                            torch.log10(torch.tensor(r_cut)), num_gaussians)
        self.register_buffer("mu", mu)  # (K,)

        # init widths ~ spacing; make them learnable if requested
        delta = torch.diff(mu, prepend=mu[:1])
        inv_sigma2 = (basis_width_scalar * 1.5 / (delta + 1e-8))**2  # (K,)
        if learnable_width:
            self.log_inv_sigma2 = nn.Parameter(inv_sigma2.log())
            self.register_buffer("inv_sigma2_fixed", torch.zeros(1))
        else:
            self.log_inv_sigma2 = None
            self.register_buffer("inv_sigma2_fixed", inv_sigma2)

        self.r_cut = r_cut
        self.num_output = num_gaussians

    def _envelope(self, r):
        # C2 polynomial cutoff: 1 - (10x^3 - 15x^4 + 6x^5)
        x = (r / self.r_cut).clamp(0, 1)
        return 1.0 - (10*x**3 - 15*x**4 + 6*x**5)

    def forward(self, dist) -> torch.Tensor:
        r = dist.view(-1, 1)                              # (N,1)
        inv_sigma2 = (self.log_inv_sigma2.exp()
                      if self.log_inv_sigma2 is not None else self.inv_sigma2_fixed)  # (K,)
        diff2 = (r - self.mu.view(1, -1))**2
        phi = torch.exp(-0.5 * inv_sigma2.view(1, -1) * diff2)  # (N,K)
        return phi * self._envelope(r)                           # (N,K)

