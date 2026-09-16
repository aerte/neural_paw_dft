import torch
from torch import nn

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..utils.graph_utils import map_sender_receiver_feature
from ..utils.nn_utils import get_linear
from ..custom_types import GraphAttentionData

class BaseGraphNeuralNetworkLayer(nn.Module):
    """
    Base class for Graph Neural Network layers.
    Used in InputLayer and EfficientGraphAttention.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Atomic number embeddings
        # ref: escn https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/escn/escn.py#L823
        # --- new: allow valence slots ---
        # Atomic embeddings (hybrid prior or fallback to table)
        VAL_SLOTS = getattr(molecular_graph_cfg, "valence_slots", 3)
        MAX_Z = molecular_graph_cfg.max_num_elements  # "max atomic number + 1"
        TABLE_LEN = VAL_SLOTS * MAX_Z

        # Fallback: original tables
        self.source_atomic_embedding = nn.Embedding(TABLE_LEN, gnn_cfg.atom_embedding_size)
        self.target_atomic_embedding = nn.Embedding(TABLE_LEN, gnn_cfg.atom_embedding_size)
        nn.init.uniform_(self.source_atomic_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_atomic_embedding.weight.data, -0.001, 0.001)


        # Node direction embedding
        self.source_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=gnn_cfg.node_direction_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.target_direction_embedding = get_linear(
            in_features=gnn_cfg.node_direction_expansion_size,
            out_features=gnn_cfg.node_direction_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

        # Edge distance embedding
        self.edge_distance_embedding = get_linear(
            in_features=gnn_cfg.edge_distance_expansion_size,
            out_features=gnn_cfg.edge_distance_embedding_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        # --- NEW: optional extra narrow RBF distance bank ---
        # enable with gnn_cfg.edge_rbf_extra = True
        self.edge_rbf_extra = bool(getattr(gnn_cfg, "edge_rbf_extra", True))
        if self.edge_rbf_extra:
            self._edge_rbf_K_extra = int(getattr(gnn_cfg, "edge_rbf_K_extra", 64))
            self._edge_rbf_sigma_scale = float(getattr(gnn_cfg, "edge_rbf_sigma_scale", 0.5))
            # choose out dim: match half of main distance emb (or same; your call)
            extra_out = max(1, gnn_cfg.edge_distance_embedding_size // 2)
            self.edge_distance_embedding_extra = get_linear(
                in_features=self._edge_rbf_K_extra,
                out_features=extra_out,
                activation=global_cfg.activation,
                bias=True,
                dropout=reg_cfg.mlp_dropout,
            )
            self._edge_rbf_extra_out_dim = extra_out
        else:
            self.edge_distance_embedding_extra = None
            self._edge_rbf_extra_out_dim = 0

        # keep a cutoff reference for extra RBF centers
        self._cutoff = float(getattr(molecular_graph_cfg, "max_radius", 6.0))

    def get_edge_linear(
        self,
        gnn_cfg: GraphNeuralNetworksConfigs,
        global_cfg: GlobalConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        base = (
            gnn_cfg.edge_distance_embedding_size
            + self._edge_rbf_extra_out_dim                  # NEW (may be 0)
            + 2 * gnn_cfg.node_direction_embedding_size
            + 2 * gnn_cfg.atom_embedding_size
        )
        return get_linear(
            in_features=base,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_node_linear(
        self, global_cfg: GlobalConfigs, reg_cfg: RegularizationConfigs
    ):
        return get_linear(
            in_features=2 * global_cfg.hidden_size,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def get_edge_features(self, x: GraphAttentionData) -> torch.Tensor:
        # ----- atomic embeddings -----
        species_ids = getattr(x, "species_ids", None)
        if species_ids is None:
            species_ids = x.atomic_numbers  # slot 0 path

        source_atomic_embedding = self.source_atomic_embedding(species_ids)
        target_atomic_embedding = self.target_atomic_embedding(species_ids)
        source_atomic_embedding, target_atomic_embedding = map_sender_receiver_feature(
            source_atomic_embedding, target_atomic_embedding, x.neighbor_list
        )

        # ----- node direction embeddings -----
        source_direction_embedding = self.source_direction_embedding(x.node_direction_expansion)
        target_direction_embedding = self.target_direction_embedding(x.node_direction_expansion)
        source_direction_embedding, target_direction_embedding = map_sender_receiver_feature(
            source_direction_embedding, target_direction_embedding, x.neighbor_list
        )

        # ----- distance embedding (existing) -----
        edge_distance_embedding = self.edge_distance_embedding(x.edge_distance_expansion)

        # ----- NEW: extra narrow RBF bank on raw distances (if available) -----
        # Try typical names; skip if not present.
        edge_distance_raw = getattr(x, "edge_distance", None)
        if edge_distance_raw is None:
            edge_distance_raw = getattr(x, "edge_distance_raw", None)

        if (self.edge_rbf_extra is True) and (edge_distance_raw is not None):
            # edge_distance_raw: (N, M) in Å
            N, M = edge_distance_raw.shape
            Kx = self._edge_rbf_K_extra
            # centers in [0, cutoff]
            centers = torch.linspace(
                0.0, self._cutoff, Kx, device=edge_distance_raw.device, dtype=edge_distance_raw.dtype
            ).view(1, 1, Kx)                                         # (1,1,Kx)
            # narrower gaussians than main bank
            sigma = (self._cutoff / max(1, Kx)) * self._edge_rbf_sigma_scale
            rbf = torch.exp(-0.5 * ((edge_distance_raw.unsqueeze(-1) - centers) / (sigma + 1e-12)) ** 2)  # (N,M,Kx)
            edge_distance_embedding_extra = self.edge_distance_embedding_extra(rbf)  # (N,M,extra_out)
        else:
            edge_distance_embedding_extra = None


        # ----- concat all edge features -----
        parts = [edge_distance_embedding]
        if edge_distance_embedding_extra is not None:
            parts.append(edge_distance_embedding_extra)
        parts.extend([source_direction_embedding, source_atomic_embedding,
                      target_direction_embedding, target_atomic_embedding])

        return torch.cat(parts, dim=-1)

    def get_node_features(
        self, node_features: torch.Tensor, neighbor_list: torch.Tensor
    ) -> torch.Tensor:
        sender_feature, receiver_feature = map_sender_receiver_feature(
            node_features, node_features, neighbor_list
        )
        return torch.cat([sender_feature, receiver_feature], dim=-1)

    def aggregate(self, edge_features, neighbor_mask):
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True) + 1e-5
        return (edge_features * neighbor_mask.unsqueeze(-1)).sum(dim=1) / neighbor_count

    def forward(self):
        raise NotImplementedError

