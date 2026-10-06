from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TargetCurvatureStats:
    sim_mean: float
    sim_std: float
    gate_mean: float
    gate_std: float
    target_mean: float
    target_std: float


def edge_entropy_gate(
    probs: torch.Tensor,
    edge_index: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    normalizer = max(math.log(max(num_classes, 2)), 1e-12)
    gate = 1.0 - (entropy[edge_index[0]] + entropy[edge_index[1]]) / (2.0 * normalizer)
    return gate.clamp(0.0, 1.0)


def compute_target_curvature(
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    kappa0_edge: torch.Tensor,
    tau: float = 1.0,
    lam: float = 0.5,
    entropy_eps: float = 1e-12,
) -> torch.Tensor:
    z = embeddings.detach()
    p = F.softmax(logits.detach(), dim=-1)
    sim = F.cosine_similarity(z[edge_index[0]], z[edge_index[1]], dim=-1)
    gate = edge_entropy_gate(p, edge_index, logits.size(-1), eps=entropy_eps)
    task_signal = torch.tanh(sim / tau) * gate
    return (1.0 - lam) * kappa0_edge + lam * task_signal


def target_curvature_stats(
    embeddings: torch.Tensor,
    logits: torch.Tensor,
    edge_index: torch.Tensor,
    kappa0_edge: torch.Tensor,
    tau: float = 1.0,
    lam: float = 0.5,
    entropy_eps: float = 1e-12,
) -> TargetCurvatureStats:
    z = embeddings.detach()
    p = F.softmax(logits.detach(), dim=-1)
    sim = F.cosine_similarity(z[edge_index[0]], z[edge_index[1]], dim=-1)
    gate = edge_entropy_gate(p, edge_index, logits.size(-1), eps=entropy_eps)
    target = (1.0 - lam) * kappa0_edge + lam * torch.tanh(sim / tau) * gate
    return TargetCurvatureStats(
        sim_mean=float(sim.mean().item()),
        sim_std=float(sim.std(unbiased=False).item()),
        gate_mean=float(gate.mean().item()),
        gate_std=float(gate.std(unbiased=False).item()),
        target_mean=float(target.mean().item()),
        target_std=float(target.std(unbiased=False).item()),
    )
