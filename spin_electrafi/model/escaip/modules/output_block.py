from typing import Literal

import torch
import torch.nn as nn

from ..utils.nn_utils import get_linear, get_feedforward, get_normalization_layer

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    RegularizationConfigs,
)


class OutputProjection(nn.Module):
    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        # map concatenated readout features to hidden size
        self.node_projection = get_linear(
            in_features=global_cfg.hidden_size * (gnn_cfg.num_layers + 1),
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )
        self.edge_projection = get_linear(
            in_features=global_cfg.hidden_size * (gnn_cfg.num_layers + 1),
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
        )
        self.readout_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=True
        )(global_cfg.hidden_size * (gnn_cfg.num_layers + 1))
        self.output_norm = get_normalization_layer(
            reg_cfg.normalization, is_graph=True
        )(global_cfg.hidden_size)

    def forward(self, node_readouts, edge_readouts):
        node_readouts, edge_readouts = self.readout_norm(node_readouts, edge_readouts)
        node_features = self.node_projection(node_readouts)
        edge_features = self.edge_projection(edge_readouts)
        node_features, edge_features = self.output_norm(node_features, edge_features)
        return node_features, edge_features

class OutputLayer(nn.Module):
    def __init__(self, global_cfg, gnn_cfg, reg_cfg,
                 output_type: Literal["Vector", "Scalar", "Tensor"],
                 num_channels: int = 1):
        super().__init__()
        self.output_type = output_type
        self.num_channels = num_channels

        if output_type == "Scalar":
            out_dim = num_channels
        elif output_type == "Vector":
            out_dim = num_channels * 3
        elif output_type == "Tensor":
            # per-spherical-component mapping: (..., 5, H) -> (..., 5, C) then transpose -> (..., C, 5)
            out_dim = num_channels
        else:
            raise ValueError(f"Invalid output_type {output_type}")

        self.ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.output_hidden_layer_multiplier,
            dropout=reg_cfg.mlp_dropout,
            bias=True,
        )
        self.final_output = get_linear(
            in_features=global_cfg.hidden_size,
            out_features=out_dim,
            activation=None,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # works for (..., H) or (..., 5, H) because Linear applies on the last dim
        x = features + self.ffn(features)
        out = self.final_output(x)  # shapes: Scalar (..., C), Vector (..., 3C), Tensor (..., 5, C)

        # Helper to preserve arbitrary leading batch dims
        batch_shape = out.shape[:-1]

        if self.output_type == "Scalar":
            return out.view(*batch_shape, self.num_channels)

        if self.output_type == "Vector":
            return out.view(*batch_shape, self.num_channels, 3)

        # Tensor case: expect input (..., 5, H) -> out (..., 5, C) -> transpose last two dims
        if out.dim() < 3 or features.shape[-2] != 5:
            raise RuntimeError(
                "Tensor OutputLayer expects input with shape (..., 5, hidden_size). "
                f"Got {features.shape}."
            )
        return out.transpose(-2, -1)  # (..., C, 5)

