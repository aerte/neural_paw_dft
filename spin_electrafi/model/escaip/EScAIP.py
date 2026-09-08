from functools import partial

import torch
import torch.nn as nn
import torch_geometric

from fairchem.core.models.base import GraphModelMixin, HeadInterface

from .configs import EScAIPConfigs, init_configs
from .custom_types import GraphAttentionData
from .modules import (
    EfficientGraphAttentionBlock,
    InputBlock,
    ReadoutBlock,
    OutputProjection,
    OutputLayer,
)
from .utils.data_preprocess import data_preprocess
from .utils.nn_utils import no_weight_decay, init_linear_weights
from .utils.graph_utils import unpad_results, compilable_scatter
import math

class EScAIPBackbone(nn.Module, GraphModelMixin):
    """
    Efficiently Scaled Attention Interactomic Potential (EScAIP) backbone model.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        # load configs
        cfg = init_configs(EScAIPConfigs, kwargs)
        self.global_cfg = cfg.global_cfg
        self.molecular_graph_cfg = cfg.molecular_graph_cfg
        self.gnn_cfg = cfg.gnn_cfg
        self.reg_cfg = cfg.reg_cfg

        # for trainer
        self.regress_forces = cfg.global_cfg.regress_forces
        self.use_pbc = cfg.molecular_graph_cfg.use_pbc
        self.otf_graph = cfg.molecular_graph_cfg.otf_graph

        # graph generation
        self.use_pbc_single = (
            self.molecular_graph_cfg.use_pbc_single
        )  # TODO: remove this when FairChem fixes the bug
        generate_graph_fn = partial(
            self.generate_graph,
            cutoff=self.molecular_graph_cfg.max_radius,
            max_neighbors=self.molecular_graph_cfg.max_neighbors,
            use_pbc=self.molecular_graph_cfg.use_pbc,
            otf_graph=self.molecular_graph_cfg.otf_graph,
            enforce_max_neighbors_strictly=self.molecular_graph_cfg.enforce_max_neighbors_strictly,
            use_pbc_single=self.molecular_graph_cfg.use_pbc_single,
        )

        # data preprocess
        self.data_preprocess = partial(
            data_preprocess,
            generate_graph_fn=generate_graph_fn,
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
        )

        ## Model Components

        # Input Block
        self.input_block = InputBlock(
            global_cfg=self.global_cfg,
            molecular_graph_cfg=self.molecular_graph_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
        )

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EfficientGraphAttentionBlock(
                    global_cfg=self.global_cfg,
                    molecular_graph_cfg=self.molecular_graph_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers)
            ]
        )

        # Readout Layer
        self.readout_layers = nn.ModuleList(
            [
                ReadoutBlock(
                    global_cfg=self.global_cfg,
                    gnn_cfg=self.gnn_cfg,
                    reg_cfg=self.reg_cfg,
                )
                for _ in range(self.gnn_cfg.num_layers + 1)
            ]
        )
        self.rawirrep_blocks = nn.ModuleList([
            RawIrrepReadoutBlock(
                global_cfg=self.global_cfg,
                gnn_cfg=self.gnn_cfg,
                reg_cfg=self.reg_cfg,
                k_vec=self.gnn_cfg.k_vec, k_t2=self.gnn_cfg.k_t2,
                neighbor_norm="mean",
                neighbor_dropout=0.0,
            )
            for _ in range(self.gnn_cfg.num_layers + 1)
        ])

        # Aggregation mode (use config if present, else default)
        agg_mode = getattr(self.gnn_cfg, "rawirrep_agg_mode", "mean")
        self.apply(init_linear_weights)
        self.rawirrep_aggregator = RawIrrepAggregator(self.global_cfg, self.gnn_cfg, mode=agg_mode)
        # init weights

        # enable torch.set_float32_matmul_precision('high')
        torch.set_float32_matmul_precision("high")

        # log recompiles
        torch._logging.set_logs(recompiles=True)

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

        # Padding buckets: pad to the smallest bucket that fits so torch.compile sees few static shapes.
        top = self.molecular_graph_cfg.max_num_nodes_per_batch
        self.padding_buckets = sorted({b for b in (16, 32, 64) if b < top} | {top})

    def compiled_forward(self, data: GraphAttentionData):
        node_features, edge_features = self.input_block(data)

        scalars_per_layer, vectors_per_layer, tensors_per_layer = [], [], []
        node_readouts_list, edge_readouts_list = [], []

        r0_node, r0_edge = self.readout_layers[0](node_features, edge_features)
        node_readouts_list.append(r0_node)
        edge_readouts_list.append(r0_edge)

        s0, v0, t0 = self.rawirrep_blocks[0](r0_node, r0_edge, data)
        scalars_per_layer.append(s0);
        vectors_per_layer.append(v0);
        tensors_per_layer.append(t0)

        for idx in range(self.gnn_cfg.num_layers):
            node_features, edge_features = self.transformer_blocks[idx](data, node_features, edge_features)
            r_node, r_edge = self.readout_layers[idx + 1](node_features, edge_features)
            node_readouts_list.append(r_node);
            edge_readouts_list.append(r_edge)

            s_i, v_i, t_i = self.rawirrep_blocks[idx + 1](r_node, r_edge, data)
            scalars_per_layer.append(s_i);
            vectors_per_layer.append(v_i);
            tensors_per_layer.append(t_i)

        scalars_stack = torch.stack(scalars_per_layer, dim=1)  # (N,L+1,C)
        vectors_stack = torch.stack(vectors_per_layer, dim=1)  # (N,L+1,C,3)
        tensors_stack = torch.stack(tensors_per_layer, dim=1)  # (N,L+1,C,3,3)
        node_readouts_stack = torch.stack(node_readouts_list, dim=1)  # (N,L+1,H)

        # existing S/V/T aggregation
        rawS, rawV, rawT = self.rawirrep_aggregator(node_readouts_stack, scalars_stack, vectors_stack, tensors_stack)

        return {
            "data": data,
            "rawirrep_scalars": rawS,
            "rawirrep_vectors": rawV,
            "rawirrep_tensors": rawT,
            "edge_features": r_edge,
        }

    def forward(self, data: torch_geometric.data.Batch):
        # gradient force
        if self.regress_forces and not self.global_cfg.direct_force:
            data.pos.requires_grad_(True)

        # smallest padding bucket that fits (oversized structures keep the top bucket).
        if self.global_cfg.use_padding:
            capacity = self.global_cfg.batch_size
            self.molecular_graph_cfg.max_num_nodes_per_batch = next(
                (b for b in self.padding_buckets if b * capacity >= data.num_nodes),
                self.padding_buckets[-1],
            )

        # preprocess data
        x = self.data_preprocess(data)

        return self.forward_fn(x)

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)



class EScAIPHeadBase(nn.Module):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__()
        self.global_cfg = backbone.global_cfg
        self.molecular_graph_cfg = backbone.molecular_graph_cfg
        self.gnn_cfg = backbone.gnn_cfg
        self.reg_cfg = backbone.reg_cfg

    def post_init(self, gain=1.0):
        # init weights
        self.apply(partial(init_linear_weights, gain=gain))

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if self.global_cfg.use_compile
            else self.compiled_forward
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return no_weight_decay(self)


class EScAIPDirectForceHead(EScAIPHeadBase):
    def __init__(self, backbone: EScAIPBackbone):
        super().__init__(backbone)
        self.force_direction_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Vector",
        )
        self.force_magnitude_layer = OutputLayer(
            global_cfg=self.global_cfg,
            gnn_cfg=self.gnn_cfg,
            reg_cfg=self.reg_cfg,
            output_type="Scalar",
        )

        self.post_init()

    def compiled_forward(self, edge_features, node_features, data: GraphAttentionData):
        # get force direction from edge features
        force_direction = self.force_direction_layer(
            edge_features
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (
            force_direction * data.edge_direction
        )  # (num_nodes, max_neighbor, 3)
        force_direction = (force_direction * data.neighbor_mask.unsqueeze(-1)).sum(
            dim=1
        )  # (num_nodes, 3)
        # get force magnitude from node readouts
        force_magnitude = self.force_magnitude_layer(node_features)  # (num_nodes, 1)
        # get output force
        return force_direction * force_magnitude  # (num_nodes, 3)

    def forward(self, data, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        force_output = self.forward_fn(
            edge_features=emb["edge_features"],
            node_features=emb["node_features"],
            data=emb["data"],
        )

        return unpad_results(
            results={"forces": force_output},
            node_padding_mask=emb["data"].node_padding_mask,
            graph_padding_mask=emb["data"].graph_padding_mask,
        )


class RawIrrepReadoutBlock(nn.Module):
    """
    Per-layer raw-irrep readout using node/edge readouts (N,H) / (N,M,H).
    Outputs per layer:
      scalars: (N, C)
      vectors: (N, C, 3)
      tensors: (N, C, 3, 3)

    Simplified for maximal non-redundant expressivity:
      - keep multi-head dyadic vector branch (∑ w_e n_e) with per-node mixing + magnitude
      - keep single S@u gated branch (optional coupling of second moment to vector)
      - keep tensor branch as weighted sum of Q_e with head mixing + isotropic channel
      - drop: second S@u, cross(V1,V2), COM radial, low-rank tensor add-on
    """
    def __init__(
        self,
        global_cfg,
        gnn_cfg,
        reg_cfg,
        k_vec: int = 2,
        k_t2: int = 2,
        neighbor_norm: str = "mean",    # "mean" | "sum"
        neighbor_dropout: float = 0.0,  # optional
    ):
        super().__init__()
        self.global_cfg = global_cfg
        self.gnn_cfg = gnn_cfg
        self.reg_cfg = reg_cfg

        self.k_vec = int(k_vec)
        self.k_t2  = int(k_t2)
        self.neighbor_dropout = float(neighbor_dropout)
        self.mean_norm = (str(neighbor_norm).lower() == "mean")

        C = self.gnn_cfg.atom_embedding_size
        D = self.global_cfg.hidden_size

        # l=0: scalar channel per node/channel
        self.scalar_layer   = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C)

        # l=1: vectors — multi-head dyadic + mixing + magnitude
        self.vec_w_layer    = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C * self.k_vec)
        self.vec_mix_node   = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C * self.k_vec)
        self.vec_mag_layer  = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C)

        self.edge_vec_resid = OutputLayer(
            global_cfg, gnn_cfg, reg_cfg, output_type="Vector", num_channels=1
        )
        self.edge_tens_resid = OutputLayer(
            global_cfg, gnn_cfg, reg_cfg, output_type="Vector", num_channels=1
        )
        self.vec_resid_alpha_net = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, 1),
            nn.Tanh()
        )
        self.tens_resid_alpha_net = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, 1),
            nn.Tanh()
        )

        _init_alpha_random(self.vec_resid_alpha_net, last_bias_std=2.0)
        _init_alpha_random(self.tens_resid_alpha_net, last_bias_std=2.0)

        self._rand_init_vector_head(self.edge_vec_resid, scale_w=2.0, scale_b=0.6)
        self._rand_init_vector_head(self.edge_tens_resid, scale_w=2.0, scale_b=0.6)

        # l=2: tensors — anisotropic (Q_e) multi-head + mixing + isotropic
        self.rank2_iso_layer = OutputLayer(global_cfg, gnn_cfg, reg_cfg,output_type="Scalar", num_channels=C)
        self.iso_gain = nn.Parameter(torch.tensor(0.1))  # start at 0 → iso off at init
        self.aniso_gain = nn.Parameter(torch.tensor(1.5))  # or 2.0

        self.rank2_w_layer   = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C * self.k_t2)
        self.rank2_mix_node  = OutputLayer(global_cfg, gnn_cfg, reg_cfg, output_type="Scalar", num_channels=C * self.k_t2)

        # normalization (keep for stability)
        for m in self.rank2_iso_layer.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # identity (buffer so it moves with module)
        self.register_buffer("I3", torch.eye(3))

    def _rand_init_vector_head(self, head: nn.Module, *, scale_w: float = 2.0, scale_b: float = 0.6):
        """
        Make output directions very random at init:
          - normal weights with boosted std for broad, isotropic spread
          - small-to-moderate random bias so outputs aren't zero even if upstream features start small
          - extra boost on the *final* linear that emits 3-vectors
        """
        for m in head.modules():
            if isinstance(m, nn.Linear):
                # base std ~ gain / sqrt(fan_in), then scaled up
                base_std = _fan_in_std(m, gain=1.0)
                w_std = scale_w * base_std
                nn.init.normal_(m.weight, mean=0.0, std=w_std)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=scale_b)

        # OPTIONAL: if you can reliably find the final linear, punch it further
        last_linear = None
        for m in head.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None and _maybe_name_is_final(last_linear):
            base_std = _fan_in_std(last_linear, gain=1.0)
            nn.init.normal_(last_linear.weight, mean=0.0, std=3.0 * base_std)
            if last_linear.bias is not None:
                nn.init.normal_(last_linear.bias, mean=0.0, std=1.0)

    def _apply_neighbor_dropout(self, mask_f: torch.Tensor) -> torch.Tensor:
        if self.training and self.neighbor_dropout > 0.0:
            keep = (torch.rand_like(mask_f) > self.neighbor_dropout).to(mask_f.dtype)
            # ensure at least one neighbor per node
            dropped = (keep * mask_f).sum(dim=1, keepdim=True) == 0
            keep = torch.where(dropped, torch.ones_like(keep), keep)
            mask_f = mask_f * keep
        return mask_f

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, data: GraphAttentionData):
        """
        node_features: (N, H)
        edge_features: (N, M, H)
        Returns:
          scalars: (N, C)
          vectors: (N, C, 3)
          tensors: (N, C, 3, 3)
        """
        N, M, F = edge_features.shape
        C = self.gnn_cfg.atom_embedding_size
        dtype = edge_features.dtype
        device = edge_features.device



        # -------- masks / normalization --------
        mask = data.neighbor_mask  # (N, M) bool
        mask_f = mask.to(dtype) if mask.dtype not in (torch.float32, torch.float64) else mask
        mask_f = self._apply_neighbor_dropout(mask_f)  # (N, M)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0) if self.mean_norm else torch.ones_like(
            mask_f.sum(dim=1, keepdim=True)
        )
        # shapes for broadcasting
        den_vec_heads = denom.view(N, 1, 1, 1)           # (N,1,1,1)
        den_t         = denom.view(N, 1, 1, 1, 1)        # (N,1,1,1,1)

        mask_vm = mask_f[:, :, None, None, None]         # (N,M,1,1,1)
        mask_e6 = mask_f[:, :, None, None, None, None]   # (N,M,1,1,1,1)

        # -------- l=0 scalars --------
        scalars = self.scalar_layer(node_features)  # (N, C)

        eps = 1e-12

        mask = data.neighbor_mask  # (N,M) bool
        mask3 = mask.unsqueeze(-1)  # (N,M,1)

        # --- base direction (safe) ---
        nb = data.edge_direction.to(dtype)  # (N,M,3)
        nb_norm = nb.norm(dim=-1, keepdim=True)
        n_base_u = nb / (nb_norm + eps)

        # fill masked edges with a benign unit vector to avoid 0/0 upstream
        safe_axis = torch.tensor([1., 0., 0.], device=nb.device, dtype=nb.dtype).view(1, 1, 3).expand_as(n_base_u)
        n_base_u = torch.where(mask3, n_base_u, safe_axis)

        # --- residuals (safe unit) ---
        ef = edge_features.reshape(N * M, F)

        dv = self.edge_vec_resid(ef).reshape(N, M, 3)
        dt = self.edge_tens_resid(ef).reshape(N, M, 3)

        dv_u = dv / (dv.norm(dim=-1, keepdim=True) + eps)
        dt_u = dt / (dt.norm(dim=-1, keepdim=True) + eps)
        resid_project_perp = True
        if resid_project_perp:
            # project out parallel to n_base_u
            dv_par = (dv_u * n_base_u).sum(dim=-1, keepdim=True) * n_base_u
            dt_par = (dt_u * n_base_u).sum(dim=-1, keepdim=True) * n_base_u
            dv_perp = dv_u - dv_par
            dt_perp = dt_u - dt_par

            dv_perp_norm = dv_perp.norm(dim=-1, keepdim=True)
            dt_perp_norm = dt_perp.norm(dim=-1, keepdim=True)

            dv_u = dv_perp / (dv_perp_norm + eps)
            dt_u = dt_perp / (dt_perp_norm + eps)

        # alphas
        vec_alpha = self.vec_resid_alpha_net(ef).reshape(N, M, 1)  # in (0,1)
        tens_alpha = self.tens_resid_alpha_net(ef).reshape(N, M, 1)

        # mix
        n_vec = (1. - vec_alpha) * n_base_u + vec_alpha * dv_u
        n_tens = (1. - tens_alpha) * n_base_u + tens_alpha * dt_u

        # protect re-normalization: if ||n|| ~ 0, fall back to n_base_u (finite)
        n_vec_norm = n_vec.norm(dim=-1, keepdim=True)
        n_tens_norm = n_tens.norm(dim=-1, keepdim=True)

        n_vec = torch.where(n_vec_norm > 1e-8, n_vec / (n_vec_norm + eps), n_base_u)
        n_tens = torch.where(n_tens_norm > 1e-8, n_tens / (n_tens_norm + eps), n_base_u)

        # Q as before (no extra normalization step here)
        I3 = self.I3.to(dtype=dtype, device=device)
        nnT = n_tens.unsqueeze(-1) @ n_tens.unsqueeze(-2)
        Q = nnT - I3 / 3.0
        # also sanitize Q on masked edges to be safe
        Q = torch.where(mask3.unsqueeze(-1), Q, torch.zeros_like(Q))

        # -------- l=1: main multi-head dyadic (∑_e w_e * n_e) --------
        w_vec = self.vec_w_layer(edge_features.reshape(N * M, F)).reshape(N, M, C, self.k_vec)
        n_car = n_vec.unsqueeze(2).unsqueeze(-2)  # (N, M, 1, 1, 3)
        v_heads = (w_vec.unsqueeze(-1) * n_car * mask_vm).sum(dim=1)       # (N, C, k, 3)
        if self.mean_norm:
            v_heads = v_heads / den_vec_heads

        mix_logits = self.vec_mix_node(node_features).reshape(N, C, self.k_vec)
        mix = torch.softmax(mix_logits, dim=-1)                            # (N, C, k)
        vec_main = (mix.unsqueeze(-1) * v_heads).sum(dim=2)                # (N, C, 3)
        vec_mag = self.vec_mag_layer(node_features).unsqueeze(-1)          # (N, C, 1)
        vec_main = vec_main / (vec_main.norm(dim=-1, keepdim=True) + eps)  # normalize direction

        # -------- l=2: anisotropic dyadic tensors (∑_e w_e * Q_e) with head mixing --------
        w_t2 = self.rank2_w_layer(edge_features.reshape(N * M, F)).reshape(N, M, C, self.k_t2, 1, 1)
        T_heads = (w_t2 * Q.unsqueeze(2).unsqueeze(3) * mask_e6).sum(dim=1)  # (N, C, k, 3, 3)
        if self.mean_norm:
            T_heads = T_heads / den_t

        mix_t2_logits = self.rank2_mix_node(node_features).reshape(N, C, self.k_t2)
        mix_t2 = torch.softmax(mix_t2_logits, dim=-1)  # (N, C, k)
        T_aniso = (mix_t2.unsqueeze(-1).unsqueeze(-1) * T_heads).sum(dim=2)  # (N, C, 3, 3)

        # add isotropic part and symmetrize
        iso_raw = self.rank2_iso_layer(node_features)  # (N, C)
        T_iso = iso_raw.unsqueeze(-1).unsqueeze(-1) * self.I3
        T_total = torch.tanh(self.iso_gain) * T_iso + torch.nn.functional.softplus(self.aniso_gain) * T_aniso
        T_total = 0.5 * (T_total + T_total.transpose(-1, -2))

        vectors = vec_main * vec_mag

        return scalars, vectors, T_total


class RawIrrepAggregator(nn.Module):
    """
    Aggregate per-layer raw-irrep outputs along the layer axis (L+1).
    Modes:
      - "mean":  average across layers
      - "sum":   sum across layers
      - "last":  take the last layer
      - "softmax": learned softmax weights from node readouts (H -> C), normalized over layers
    """
    def __init__(self, global_cfg, gnn_cfg, mode: str = "mean"):
        super().__init__()
        self.mode = mode.lower()
        self.hidden_size = global_cfg.hidden_size
        self.C = gnn_cfg.atom_embedding_size

        if self.mode == "softmax":
            # applies to (..., H) -> (..., C), works on (N, L+1, H) directly
            self.weight_proj = nn.Linear(self.hidden_size, self.C, bias=True)

    def forward(
        self,
        node_readouts_stack: torch.Tensor,  # (N, L+1, H)
        scalars_per_layer: torch.Tensor,    # (N, L+1, C)
        vectors_per_layer: torch.Tensor,    # (N, L+1, C, 3)
        tensors_per_layer: torch.Tensor,    # (N, L+1, C, 3, 3)
    ):
        if self.mode == "mean":
            S = scalars_per_layer.mean(dim=1)
            V = vectors_per_layer.mean(dim=1)
            T = tensors_per_layer.mean(dim=1)
            return S, V, T

        if self.mode == "sum":
            S = scalars_per_layer.sum(dim=1)
            V = vectors_per_layer.sum(dim=1)
            T = tensors_per_layer.sum(dim=1)
            return S, V, T

        if self.mode == "last":
            S = scalars_per_layer[:, -1]
            V = vectors_per_layer[:, -1]
            T = tensors_per_layer[:, -1]
            return S, V, T

        if self.mode == "softmax":
            # logits: (N, L+1, C), softmax over layers
            logits = self.weight_proj(node_readouts_stack)            # (N, L+1, C)
            w = torch.softmax(logits, dim=1)                          # (N, L+1, C)

            S = (w * scalars_per_layer).sum(dim=1)                    # (N, C)
            V = (w.unsqueeze(-1) * vectors_per_layer).sum(dim=1)      # (N, C, 3)
            T = (w.unsqueeze(-1).unsqueeze(-1) * tensors_per_layer).sum(dim=1)  # (N, C, 3, 3)
            return S, V, T

        raise ValueError(f"Unknown aggregation mode: {self.mode}")

def _maybe_name_is_final(m: nn.Linear) -> bool:
    # heuristic: 3 or multiples of 3 outputs in a Vector head
    return m.out_features % 3 == 0

def _fan_in_std(m: nn.Linear, gain: float = 1.0) -> float:
    fan_in = m.weight.shape[1]
    return gain / math.sqrt(fan_in)

def _init_alpha_random(seq: nn.Sequential, *, hidden_gain: float = 1.0, last_w_gain: float = 1.0,
                       last_bias_mean: float = 0.0, last_bias_std: float = 2.0):
    """
    Randomize alpha nets so sigmoid outputs cover a wide range initially.
      - Hidden linears: Kaiming normal (fan_in)
      - Last linear: scaled normal weights, bias ~ N(mean, std)
    Sigmoid(last) with bias std≈2 -> outputs span ~[0.05, 0.95].
    """
    last_linear = None
    for m in seq:
        if isinstance(m, nn.Linear):
            last_linear = m

    # init all linears
    for m in seq:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='linear')  # safe default for SiLU too
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # punch the last layer for spread
    assert isinstance(last_linear, nn.Linear), "alpha net missing final Linear"
    with torch.no_grad():
        nn.init.normal_(last_linear.weight, mean=0.0,
                        std=last_w_gain / math.sqrt(last_linear.weight.shape[1]))
        if last_linear.bias is not None:
            nn.init.normal_(last_linear.bias, mean=last_bias_mean, std=last_bias_std)
