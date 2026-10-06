from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .utils import ensure_undirected_edge_index


@dataclass
class RewiringState:
    num_nodes: int
    edge_index_base: torch.Tensor
    edge_index_candidate: torch.Tensor
    edge_weight_init: torch.Tensor


def _candidate_two_hop_edges(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    k: int,
) -> torch.Tensor:
    if num_nodes == 0 or k <= 0:
        return edge_index.new_empty((2, 0))

    device = x.device
    edge_index = edge_index.to(device)
    x_norm = torch.nn.functional.normalize(x.float(), p=2, dim=-1)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    adj[edge_index[0], edge_index[1]] = True

    two_hop = (adj.to(torch.float32) @ adj.to(torch.float32)) > 0
    two_hop.fill_diagonal_(False)
    two_hop &= ~adj
    if not two_hop.any():
        return edge_index.new_empty((2, 0))

    sim = x_norm @ x_norm.t()
    sim = sim.masked_fill(~two_hop, float("-inf"))
    topk = min(k, max(num_nodes - 1, 1))
    topk_scores, topk_idx = torch.topk(sim, k=topk, dim=1)
    valid = torch.isfinite(topk_scores)
    if not valid.any():
        return edge_index.new_empty((2, 0))

    row_idx = torch.arange(num_nodes, device=device).unsqueeze(1).expand_as(topk_idx)
    candidate_edges = torch.stack([row_idx[valid], topk_idx[valid]], dim=0)
    return ensure_undirected_edge_index(candidate_edges)


def build_candidate_graph(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    k: int = 5,
    candidate_init: float = 0.08,
) -> RewiringState:
    edge_index_base = ensure_undirected_edge_index(edge_index)
    edge_index_knn = _candidate_two_hop_edges(edge_index_base, x, num_nodes, k)
    edge_index_candidate = ensure_undirected_edge_index(
        torch.cat([edge_index_base, edge_index_knn], dim=1)
        if edge_index_knn.numel() > 0
        else edge_index_base
    )

    init_weight = torch.full(
        (edge_index_candidate.size(1),),
        fill_value=candidate_init,
        dtype=torch.float32,
        device=edge_index_candidate.device,
    )
    base_keys = edge_index_base[0].to(torch.int64) * num_nodes + edge_index_base[1].to(torch.int64)
    candidate_keys = edge_index_candidate[0].to(torch.int64) * num_nodes + edge_index_candidate[1].to(torch.int64)
    if hasattr(torch, "isin"):
        base_mask = torch.isin(candidate_keys, base_keys)
    else:
        base_keys_sorted = torch.sort(base_keys).values
        pos = torch.searchsorted(base_keys_sorted, candidate_keys)
        valid = pos < base_keys_sorted.numel()
        base_mask = torch.zeros_like(candidate_keys, dtype=torch.bool)
        base_mask[valid] = base_keys_sorted[pos[valid]] == candidate_keys[valid]
    init_weight[base_mask] = 1.0

    return RewiringState(
        num_nodes=num_nodes,
        edge_index_base=edge_index_base,
        edge_index_candidate=edge_index_candidate,
        edge_weight_init=init_weight,
    )


class LearnableEdgeWeights(nn.Module):
    def __init__(
        self,
        init_weight: torch.Tensor,
        threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.edge_weight_param = nn.Parameter(init_weight.clone())

    def raw_weights(self) -> torch.Tensor:
        return self.edge_weight_param.clamp_min(0.0)

    def active_weights(self) -> torch.Tensor:
        raw = self.raw_weights()
        return torch.where(raw > self.threshold, raw, torch.zeros_like(raw))

    def delta(self, previous_weight: Optional[torch.Tensor]) -> torch.Tensor:
        current = self.active_weights().detach()
        if previous_weight is None:
            return torch.zeros_like(current)
        return current - previous_weight

    def explicit_step(self, lr: float) -> None:
        if self.edge_weight_param.grad is None:
            return
        with torch.no_grad():
            self.edge_weight_param.sub_(lr * self.edge_weight_param.grad)
            self.edge_weight_param.clamp_(min=0.0)
            self.edge_weight_param.grad.zero_()

    def hard_rebuild_mask(self) -> torch.Tensor:
        return self.active_weights().detach() > 0
