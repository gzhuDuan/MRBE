from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

from .init_curvature import initialize_edge_curvature
from .rewiring import LearnableEdgeWeights, RewiringState
from .surrogate import CurvatureState


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    embeddings: torch.Tensor
    active_edge_weight: torch.Tensor
    edge_index: torch.Tensor


class TaskCurvatureGCN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        rewiring_state: RewiringState,
        dropout: float = 0.5,
        threshold: float = 0.05,
        init_curvature_mode: str = "exact",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.init_curvature_mode = init_curvature_mode
        self.rewiring_state = rewiring_state
        self.edge_weights = LearnableEdgeWeights(rewiring_state.edge_weight_init, threshold=threshold)
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

        self.register_buffer(
            "kappa0_edge",
            initialize_edge_curvature(
                rewiring_state.edge_index_base,
                num_nodes=rewiring_state.num_nodes,
                target_edge_index=rewiring_state.edge_index_candidate,
                mode=self.init_curvature_mode,
            ),
        )

    def init_curvature_state(self) -> CurvatureState:
        return CurvatureState(current_kappa_edge=self.kappa0_edge.detach().clone())

    def gnn_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def refresh_curvature(self) -> torch.Tensor:
        refreshed = initialize_edge_curvature(
            self.rewiring_state.edge_index_candidate,
            num_nodes=self.rewiring_state.num_nodes,
            target_edge_index=self.rewiring_state.edge_index_candidate,
            edge_weight=self.edge_weights.active_weights().detach(),
            mode="proxy",
        )
        return refreshed.detach().clone()

    def hard_rebuild(self) -> None:
        keep_mask = self.edge_weights.hard_rebuild_mask()
        if keep_mask.all():
            return

        device = self.edge_weights.edge_weight_param.device
        new_edge_index = self.rewiring_state.edge_index_candidate[:, keep_mask]
        new_weight = self.edge_weights.raw_weights().detach()[keep_mask]

        self.rewiring_state.edge_index_candidate = new_edge_index
        self.rewiring_state.edge_weight_init = new_weight
        self.edge_weights = LearnableEdgeWeights(new_weight.to(device), threshold=self.edge_weights.threshold)
        self.kappa0_edge = self.kappa0_edge[keep_mask].detach().clone()

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        edge_index = self.rewiring_state.edge_index_candidate
        edge_weight = self.edge_weights.active_weights()

        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, edge_index, edge_weight=edge_weight)

        return ForwardOutput(
            logits=logits,
            embeddings=h,
            active_edge_weight=edge_weight,
            edge_index=edge_index,
        )


class TaskCurvatureGAT(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        rewiring_state: RewiringState,
        dropout: float = 0.5,
        threshold: float = 0.05,
        init_curvature_mode: str = "exact",
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.init_curvature_mode = init_curvature_mode
        self.rewiring_state = rewiring_state
        self.edge_weights = LearnableEdgeWeights(rewiring_state.edge_weight_init, threshold=threshold)
        self.conv1 = GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim, num_classes, heads=1, dropout=dropout, concat=False)

        self.register_buffer(
            "kappa0_edge",
            initialize_edge_curvature(
                rewiring_state.edge_index_base,
                num_nodes=rewiring_state.num_nodes,
                target_edge_index=rewiring_state.edge_index_candidate,
                mode=self.init_curvature_mode,
            ),
        )

    def init_curvature_state(self) -> CurvatureState:
        return CurvatureState(current_kappa_edge=self.kappa0_edge.detach().clone())

    def gnn_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def refresh_curvature(self) -> torch.Tensor:
        refreshed = initialize_edge_curvature(
            self.rewiring_state.edge_index_candidate,
            num_nodes=self.rewiring_state.num_nodes,
            target_edge_index=self.rewiring_state.edge_index_candidate,
            edge_weight=self.edge_weights.active_weights().detach(),
            mode="proxy",
        )
        return refreshed.detach().clone()

    def hard_rebuild(self) -> None:
        keep_mask = self.edge_weights.hard_rebuild_mask()
        if keep_mask.all():
            return
        device = self.edge_weights.edge_weight_param.device
        new_edge_index = self.rewiring_state.edge_index_candidate[:, keep_mask]
        new_weight = self.edge_weights.raw_weights().detach()[keep_mask]
        self.rewiring_state.edge_index_candidate = new_edge_index
        self.rewiring_state.edge_weight_init = new_weight
        self.edge_weights = LearnableEdgeWeights(new_weight.to(device), threshold=self.edge_weights.threshold)
        self.kappa0_edge = self.kappa0_edge[keep_mask].detach().clone()

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        edge_index = self.rewiring_state.edge_index_candidate
        edge_weight = self.edge_weights.active_weights()
        active_mask = edge_weight > 0
        active_edge_index = edge_index[:, active_mask]

        h = self.conv1(x, active_edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, active_edge_index)

        return ForwardOutput(
            logits=logits,
            embeddings=h,
            active_edge_weight=edge_weight,
            edge_index=edge_index,
        )


class TaskCurvatureGraphSAGE(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        num_classes: int,
        rewiring_state: RewiringState,
        dropout: float = 0.5,
        threshold: float = 0.05,
        init_curvature_mode: str = "exact",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.init_curvature_mode = init_curvature_mode
        self.rewiring_state = rewiring_state
        self.edge_weights = LearnableEdgeWeights(rewiring_state.edge_weight_init, threshold=threshold)
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)

        self.register_buffer(
            "kappa0_edge",
            initialize_edge_curvature(
                rewiring_state.edge_index_base,
                num_nodes=rewiring_state.num_nodes,
                target_edge_index=rewiring_state.edge_index_candidate,
                mode=self.init_curvature_mode,
            ),
        )

    def init_curvature_state(self) -> CurvatureState:
        return CurvatureState(current_kappa_edge=self.kappa0_edge.detach().clone())

    def gnn_parameters(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters())

    def refresh_curvature(self) -> torch.Tensor:
        refreshed = initialize_edge_curvature(
            self.rewiring_state.edge_index_candidate,
            num_nodes=self.rewiring_state.num_nodes,
            target_edge_index=self.rewiring_state.edge_index_candidate,
            edge_weight=self.edge_weights.active_weights().detach(),
            mode="proxy",
        )
        return refreshed.detach().clone()

    def hard_rebuild(self) -> None:
        keep_mask = self.edge_weights.hard_rebuild_mask()
        if keep_mask.all():
            return
        device = self.edge_weights.edge_weight_param.device
        new_edge_index = self.rewiring_state.edge_index_candidate[:, keep_mask]
        new_weight = self.edge_weights.raw_weights().detach()[keep_mask]
        self.rewiring_state.edge_index_candidate = new_edge_index
        self.rewiring_state.edge_weight_init = new_weight
        self.edge_weights = LearnableEdgeWeights(new_weight.to(device), threshold=self.edge_weights.threshold)
        self.kappa0_edge = self.kappa0_edge[keep_mask].detach().clone()

    def forward(self, x: torch.Tensor) -> ForwardOutput:
        edge_index = self.rewiring_state.edge_index_candidate
        edge_weight = self.edge_weights.active_weights()
        active_mask = edge_weight > 0
        active_edge_index = edge_index[:, active_mask]

        h = self.conv1(x, active_edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.conv2(h, active_edge_index)

        return ForwardOutput(
            logits=logits,
            embeddings=h,
            active_edge_weight=edge_weight,
            edge_index=edge_index,
        )
