from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class GraphState:
    edge_index_base: torch.Tensor
    edge_index_candidate: torch.Tensor
    edge_weight_init: torch.Tensor


def ensure_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    rev = edge_index.flip(0)
    merged = torch.cat([edge_index, rev], dim=1)
    merged = torch.unique(merged.t(), dim=0).t().contiguous()
    return merged


def edge_index_to_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    return edge_index.t().contiguous()


def split_active_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = edge_weight > threshold
    return edge_index[:, active_mask], edge_weight[active_mask], active_mask
