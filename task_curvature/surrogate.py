from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CurvatureState:
    current_kappa_edge: torch.Tensor
    last_active_weight: torch.Tensor | None = None


def _edge_to_2hop_mask(
    edge_index: torch.Tensor,
    delta_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    changed_mask = delta_weight.abs() > 1e-12
    if not changed_mask.any():
        return torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    changed_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    changed_edges = edge_index[:, changed_mask]
    changed_nodes[changed_edges[0]] = True
    changed_nodes[changed_edges[1]] = True

    src, dst = edge_index
    one_hop_edges = changed_nodes[src] | changed_nodes[dst]
    one_hop_nodes = changed_nodes.clone()
    one_hop_nodes[src[one_hop_edges]] = True
    one_hop_nodes[dst[one_hop_edges]] = True

    two_hop_edges = one_hop_nodes[src] | one_hop_nodes[dst]
    two_hop_nodes = one_hop_nodes.clone()
    two_hop_nodes[src[two_hop_edges]] = True
    two_hop_nodes[dst[two_hop_edges]] = True
    return two_hop_nodes[src] | two_hop_nodes[dst]


def _transition_delta(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    degree: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row = edge_index[0]

    delta_degree = torch.zeros_like(degree)
    delta_degree.scatter_add_(0, row, delta_weight)

    src = edge_index[0]
    numerator = delta_weight * degree[src] - edge_weight * delta_degree[src]
    denominator = degree[src].pow(2).clamp_min(1e-12)
    delta_transition_weight = numerator / denominator
    return delta_transition_weight, delta_degree


def _edge_two_hop_response(
    edge_index: torch.Tensor,
    transition_weight: torch.Tensor,
    delta_transition_weight: torch.Tensor,
    num_nodes: int,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    src, dst = edge_index
    response = torch.zeros(edge_index.size(1), device=transition_weight.device, dtype=transition_weight.dtype)
    if edge_index.numel() == 0 or not target_mask.any():
        return response

    target_indices = torch.nonzero(target_mask, as_tuple=False).view(-1)
    target_keys = src[target_indices].to(torch.int64) * num_nodes + dst[target_indices].to(torch.int64)
    sorted_order = torch.argsort(target_keys)
    sorted_target_keys = target_keys[sorted_order]
    sorted_target_indices = target_indices[sorted_order]

    incoming_order = torch.argsort(dst)
    outgoing_order = torch.argsort(src)
    incoming_counts = torch.bincount(dst, minlength=num_nodes)
    outgoing_counts = torch.bincount(src, minlength=num_nodes)
    incoming_offsets = torch.cat([incoming_counts.new_zeros(1), incoming_counts.cumsum(dim=0)])
    outgoing_offsets = torch.cat([outgoing_counts.new_zeros(1), outgoing_counts.cumsum(dim=0)])
    active_middle = torch.nonzero((incoming_counts > 0) & (outgoing_counts > 0), as_tuple=False).view(-1)
    max_pair_evals = 262144

    for middle_tensor in active_middle:
        middle = int(middle_tensor.item())
        in_start = int(incoming_offsets[middle].item())
        in_end = int(incoming_offsets[middle + 1].item())
        out_start = int(outgoing_offsets[middle].item())
        out_end = int(outgoing_offsets[middle + 1].item())

        incoming_edges = incoming_order[in_start:in_end]
        outgoing_edges = outgoing_order[out_start:out_end]
        in_src = src[incoming_edges].to(torch.int64)
        if in_src.numel() == 0 or outgoing_edges.numel() == 0:
            continue

        # Bound the outer-product work per chunk so large neighborhoods do not spike GPU memory.
        out_chunk = max(1, max_pair_evals // max(int(in_src.numel()), 1))
        for out_chunk_start in range(0, int(outgoing_edges.numel()), out_chunk):
            out_chunk_edges = outgoing_edges[out_chunk_start:out_chunk_start + out_chunk]
            out_dst = dst[out_chunk_edges].to(torch.int64)
            pair_keys = (in_src.unsqueeze(1) * num_nodes + out_dst.unsqueeze(0)).reshape(-1)

            search_pos = torch.searchsorted(sorted_target_keys, pair_keys)
            valid = search_pos < sorted_target_keys.numel()
            if not valid.any():
                continue

            valid_pos = search_pos[valid]
            valid_keys = pair_keys[valid]
            exact_match = sorted_target_keys[valid_pos] == valid_keys
            if not exact_match.any():
                continue

            contrib = (
                delta_transition_weight[incoming_edges].unsqueeze(1) * transition_weight[out_chunk_edges].unsqueeze(0)
                + transition_weight[incoming_edges].unsqueeze(1) * delta_transition_weight[out_chunk_edges].unsqueeze(0)
            ).reshape(-1)
            matched_targets = sorted_target_indices[valid_pos[exact_match]]
            matched_values = contrib[valid][exact_match]
            response.index_add_(0, matched_targets, matched_values)

    return response


def _edge_linear_response(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    num_nodes: int,
    mix: float,
    affected_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst = edge_index
    degree = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
    degree.scatter_add_(0, src, edge_weight)
    transition_weight = edge_weight / degree[src].clamp_min(1e-12)
    delta_transition_weight, delta_degree = _transition_delta(edge_index, edge_weight, delta_weight, degree)
    if affected_mask is None:
        affected_mask = _edge_to_2hop_mask(edge_index, delta_weight, num_nodes)

    degree_response = delta_degree[src] + delta_degree[dst]
    two_hop_response = _edge_two_hop_response(
        edge_index=edge_index,
        transition_weight=transition_weight,
        delta_transition_weight=delta_transition_weight,
        num_nodes=num_nodes,
        target_mask=affected_mask,
    )
    degree_scale = degree_response[affected_mask].abs().mean().clamp_min(1e-12) if affected_mask.any() else degree_response.new_tensor(1.0)
    two_hop_scale = two_hop_response[affected_mask].abs().mean().clamp_min(1e-12) if affected_mask.any() else two_hop_response.new_tensor(1.0)
    degree_component = degree_response / degree_scale
    two_hop_component = two_hop_response / two_hop_scale
    delta_kappa = (1.0 - mix) * degree_component + mix * two_hop_component
    return delta_kappa, degree_component, two_hop_component


def surrogate_update(
    current_kappa_edge: torch.Tensor,
    edge_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    mix: float = 0.5,
) -> torch.Tensor:
    affected_mask = _edge_to_2hop_mask(edge_index, delta_weight, num_nodes)
    if not affected_mask.any():
        return current_kappa_edge.detach().clone()

    delta_kappa, _, _ = _edge_linear_response(
        edge_index=edge_index,
        edge_weight=edge_weight,
        delta_weight=delta_weight,
        num_nodes=num_nodes,
        mix=mix,
        affected_mask=affected_mask,
    )
    updated = current_kappa_edge.clone()
    updated[affected_mask] = updated[affected_mask] + delta_kappa[affected_mask]
    return updated.detach()


def surrogate_stats(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    num_nodes: int,
    mix: float = 0.5,
) -> dict[str, float]:
    affected_mask = _edge_to_2hop_mask(edge_index, delta_weight, num_nodes)
    changed_edges = int((delta_weight.abs() > 1e-12).sum().item())
    affected_edges = int(affected_mask.sum().item())
    delta_abs = delta_weight.abs()

    if edge_index.numel() == 0:
        degree_mean = 0.0
        two_hop_mean = 0.0
    else:
        _, degree_response, two_hop_response = _edge_linear_response(
            edge_index=edge_index,
            edge_weight=edge_weight,
            delta_weight=delta_weight,
            num_nodes=num_nodes,
            mix=mix,
        )
        degree_mean = float(degree_response[affected_mask].abs().mean().item()) if affected_mask.any() else 0.0
        two_hop_mean = float(two_hop_response[affected_mask].abs().mean().item()) if affected_mask.any() else 0.0

    return {
        "changed_edges": float(changed_edges),
        "affected_edges": float(affected_edges),
        "delta_mean": float(delta_abs.mean().item()) if delta_abs.numel() > 0 else 0.0,
        "delta_max": float(delta_abs.max().item()) if delta_abs.numel() > 0 else 0.0,
        "degree_response_mean": degree_mean,
        "two_hop_response_mean": two_hop_mean,
    }
