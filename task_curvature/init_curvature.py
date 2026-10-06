from __future__ import annotations

import torch


def _build_dense_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    device = edge_index.device
    dtype = edge_weight.dtype if edge_weight is not None else torch.float64
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    if edge_index.numel() == 0:
        return adj
    values = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), dtype=dtype, device=device)
    adj[edge_index[0], edge_index[1]] = values
    return adj


def _gamma_at_center(f: torch.Tensor, adj: torch.Tensor, center_idx: int) -> torch.Tensor:
    diff = f - f[center_idx]
    return 0.5 * torch.sum(adj[center_idx] * diff.pow(2))


def _gamma2_batch(F: torch.Tensor, adj: torch.Tensor, center_idx: int) -> torch.Tensor:
    if F.numel() == 0:
        return F.new_zeros((0,))

    adj_batch = adj.unsqueeze(0)
    f_diff = F.unsqueeze(2) - F.unsqueeze(1)
    delta_f = torch.sum(adj_batch * f_diff, dim=2)
    gamma_f = 0.5 * torch.sum(adj_batch * f_diff.pow(2), dim=2)
    center_adj = adj[center_idx].unsqueeze(0)
    center_gamma = gamma_f[:, center_idx].unsqueeze(1)
    delta_gamma = torch.sum(center_adj * (gamma_f - center_gamma), dim=1)
    delta_f_diff = delta_f - delta_f[:, center_idx].unsqueeze(1)
    center_f = F[:, center_idx].unsqueeze(1)
    gamma_f_delta = 0.5 * torch.sum(center_adj * (F - center_f) * delta_f_diff, dim=1)
    return 0.5 * delta_gamma - gamma_f_delta


def _safe_gamma2_batch_size(adj_local: torch.Tensor, target_mb: int = 128) -> int:
    n = max(int(adj_local.size(0)), 1)
    bytes_per_value = max(int(adj_local.element_size()), 1)
    bytes_per_sample = n * n * bytes_per_value
    budget = target_mb * 1024 * 1024
    return max(1, budget // max(bytes_per_sample, 1))


def _gamma2_batch_chunked(
    F: torch.Tensor,
    adj: torch.Tensor,
    center_idx: int,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if F.numel() == 0:
        return F.new_zeros((0,))

    chunk_size = chunk_size or _safe_gamma2_batch_size(adj)
    outputs = []
    for start in range(0, F.size(0), chunk_size):
        outputs.append(_gamma2_batch(F[start:start + chunk_size], adj, center_idx))
    return torch.cat(outputs, dim=0)


def _gamma2_at_center(f: torch.Tensor, adj: torch.Tensor, center_idx: int) -> torch.Tensor:
    f_diff = f.unsqueeze(0) - f.unsqueeze(1)
    delta_f = torch.sum(adj * f_diff, dim=1)
    gamma_f = 0.5 * torch.sum(adj * f_diff.pow(2), dim=1)
    delta_gamma = torch.sum(adj[center_idx] * (gamma_f - gamma_f[center_idx]))
    delta_f_diff = delta_f - delta_f[center_idx]
    gamma_f_delta = 0.5 * torch.sum(adj[center_idx] * (f - f[center_idx]) * delta_f_diff)
    return 0.5 * delta_gamma - gamma_f_delta


def _quadratic_form_matrix(
    adj_local: torch.Tensor,
    center_local_idx: int,
    form_fn,
) -> torch.Tensor:
    num_vars = adj_local.size(0) - 1
    if num_vars <= 0:
        return adj_local.new_zeros((0, 0))

    centerless_nodes = [idx for idx in range(adj_local.size(0)) if idx != center_local_idx]
    basis_values = []
    for node_idx in centerless_nodes:
        f = adj_local.new_zeros(adj_local.size(0))
        f[node_idx] = 1.0
        basis_values.append(f)

    matrix = adj_local.new_zeros((num_vars, num_vars))
    diag = [form_fn(vec, adj_local, center_local_idx) for vec in basis_values]
    for i in range(num_vars):
        matrix[i, i] = diag[i]
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            value = form_fn(basis_values[i] + basis_values[j], adj_local, center_local_idx)
            cross = 0.5 * (value - diag[i] - diag[j])
            matrix[i, j] = cross
            matrix[j, i] = cross
    return 0.5 * (matrix + matrix.t())


def _gamma_matrix_diagonal(
    adj_local: torch.Tensor,
    center_local_idx: int,
) -> torch.Tensor:
    centerless_mask = torch.ones(adj_local.size(0), dtype=torch.bool, device=adj_local.device)
    centerless_mask[center_local_idx] = False
    return 0.5 * adj_local[center_local_idx, centerless_mask]


def _gamma2_matrix_batch(
    adj_local: torch.Tensor,
    center_local_idx: int,
) -> torch.Tensor:
    num_vars = adj_local.size(0) - 1
    if num_vars <= 0:
        return adj_local.new_zeros((0, 0))

    centerless_mask = torch.ones(adj_local.size(0), dtype=torch.bool, device=adj_local.device)
    centerless_mask[center_local_idx] = False

    basis = torch.eye(adj_local.size(0), dtype=adj_local.dtype, device=adj_local.device)[centerless_mask]
    chunk_size = _safe_gamma2_batch_size(adj_local)
    diag = _gamma2_batch_chunked(basis, adj_local, center_local_idx, chunk_size=chunk_size)

    matrix = adj_local.new_zeros((num_vars, num_vars))
    matrix.diagonal().copy_(diag)

    if num_vars == 1:
        return matrix

    pair_i, pair_j = torch.triu_indices(num_vars, num_vars, offset=1, device=adj_local.device)
    for start in range(0, pair_i.numel(), chunk_size):
        pair_i_chunk = pair_i[start:start + chunk_size]
        pair_j_chunk = pair_j[start:start + chunk_size]
        pair_basis = basis.index_select(0, pair_i_chunk) + basis.index_select(0, pair_j_chunk)
        pair_values = _gamma2_batch(pair_basis, adj_local, center_local_idx)
        cross = 0.5 * (
            pair_values
            - diag.index_select(0, pair_i_chunk)
            - diag.index_select(0, pair_j_chunk)
        )
        matrix[pair_i_chunk, pair_j_chunk] = cross
        matrix[pair_j_chunk, pair_i_chunk] = cross
    return 0.5 * (matrix + matrix.t())


def _exact_node_curvature_from_adj(
    adj: torch.Tensor,
    node_idx: int,
    eig_tol: float = 1e-9,
) -> torch.Tensor:
    one_hop = torch.nonzero(adj[node_idx] > 0, as_tuple=False).view(-1)
    if one_hop.numel() == 0:
        return adj.new_tensor(0.0)

    two_hop_mask = torch.any(adj[one_hop] > 0, dim=0)
    local_nodes = torch.nonzero(two_hop_mask | (adj[node_idx] > 0), as_tuple=False).view(-1)
    if not (local_nodes == node_idx).any():
        local_nodes = torch.cat([local_nodes.new_tensor([node_idx]), local_nodes])
    else:
        local_nodes = torch.cat([local_nodes[local_nodes == node_idx], local_nodes[local_nodes != node_idx]])

    adj_local = adj.index_select(0, local_nodes).index_select(1, local_nodes)
    gamma_diag = _gamma_matrix_diagonal(adj_local, 0)
    if gamma_diag.numel() == 0:
        return adj.new_tensor(0.0)

    gamma2_matrix = _gamma2_matrix_batch(adj_local, 0)
    valid = gamma_diag > eig_tol
    if valid.sum() == 0:
        return adj.new_tensor(0.0)

    basis_evals = gamma_diag[valid]
    reduced = gamma2_matrix[valid][:, valid]
    scaled = reduced / torch.sqrt(basis_evals.unsqueeze(0) * basis_evals.unsqueeze(1))
    scaled = 0.5 * (scaled + scaled.t())
    return torch.linalg.eigvalsh(scaled).min()


def _compute_exact_node_curvatures(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    adj = _build_dense_adjacency(edge_index, num_nodes, edge_weight=edge_weight).to(dtype=torch.float64)
    node_curvatures = []
    for node_idx in range(num_nodes):
        node_curvatures.append(_exact_node_curvature_from_adj(adj, node_idx))
    return torch.stack(node_curvatures).to(dtype=adj.dtype, device=edge_index.device)


def _transition_matrix_from_adjacency(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = adj.sum(dim=1)
    P = torch.zeros_like(adj)
    valid = mu > 0
    if valid.any():
        P[valid] = adj[valid] / mu[valid].unsqueeze(1)
    return P, mu


def _combinatorial_transition_from_adjacency(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = torch.ones(adj.size(0), dtype=adj.dtype, device=adj.device)
    return adj.clone(), mu


def _exact_node_curvature_cushing_from_transition(
    P: torch.Tensor,
    support: torch.Tensor,
    mu: torch.Tensor,
    node_idx: int,
    eig_tol: float = 1e-12,
) -> torch.Tensor:
    s1_mask = support[node_idx].clone()
    s1_mask[node_idx] = False
    s1 = torch.nonzero(s1_mask, as_tuple=False).view(-1)
    if s1.numel() == 0:
        # Engineering convention for isolated vertices.
        return P.new_tensor(0.0)

    reach2_mask = torch.any(support.index_select(0, s1), dim=0)
    reach2_mask[node_idx] = False
    reach2_mask[s1] = False
    s2 = torch.nonzero(reach2_mask, as_tuple=False).view(-1)

    p_x_s1 = P[node_idx].index_select(0, s1)
    if s2.numel() == 0:
        q = 0.5 * torch.diag(p_x_s1.square())
        scale = torch.sqrt(torch.clamp(torch.outer(p_x_s1, p_x_s1), min=eig_tol))
        a_inf = 2.0 * q / scale
        a_inf = 0.5 * (a_inf + a_inf.t())
        return torch.linalg.eigvalsh(a_inf).min()

    p_s1_to_s1 = P.index_select(0, s1).index_select(1, s1)
    p_s1_to_s2 = P.index_select(0, s1).index_select(1, s2)
    p_s1_to_x = P.index_select(0, s1)[:, node_idx]
    p_x_s2_2 = p_x_s1 @ p_s1_to_s2

    positive_s2 = p_x_s2_2 > eig_tol
    if not positive_s2.all():
        s2 = s2[positive_s2]
        p_s1_to_s2 = p_s1_to_s2[:, positive_s2]
        p_x_s2_2 = p_x_s2_2[positive_s2]

    s1_row_sum_to_s2 = p_s1_to_s2.sum(dim=1)
    weighted_s2 = p_s1_to_s2 / torch.sqrt(p_x_s2_2).unsqueeze(0)
    schur_core = weighted_s2 @ weighted_s2.t()
    schur_full = torch.outer(p_x_s1, p_x_s1) * schur_core
    schur_diag = p_x_s1.square() * weighted_s2.square().sum(dim=1)

    d_x = support[node_idx].to(P.dtype).sum()
    dx_mu_ratio = d_x / mu[node_idx].clamp_min(eig_tol)

    q_diag = (
        0.5 * p_x_s1.square()
        + 0.75 * p_x_s1 * p_s1_to_x
        - 0.25 * dx_mu_ratio * p_x_s1
        + 0.75 * p_x_s1 * s1_row_sum_to_s2
        + 0.25 * (3.0 * p_x_s1 * p_s1_to_s1.sum(dim=1) + p_s1_to_s1.t() @ p_x_s1)
        - p_x_s1.square() * schur_diag
    )

    q = 0.5 * torch.outer(p_x_s1, p_x_s1) - 0.5 * (
        p_x_s1.unsqueeze(1) * p_s1_to_s1 + p_x_s1.unsqueeze(0) * p_s1_to_s1.t()
    ) - schur_full
    q.fill_diagonal_(0.0)
    q.diagonal().copy_(q_diag)

    scale = torch.sqrt(torch.clamp(torch.outer(p_x_s1, p_x_s1), min=eig_tol))
    a_inf = 2.0 * q / scale
    a_inf = 0.5 * (a_inf + a_inf.t())
    return torch.linalg.eigvalsh(a_inf).min()


def _compute_exact_node_curvatures_cushing(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    adj = _build_dense_adjacency(edge_index, num_nodes, edge_weight=edge_weight).to(dtype=torch.float64)
    support = adj > 0
    P, mu = _combinatorial_transition_from_adjacency(adj)
    node_curvatures = []
    for node_idx in range(num_nodes):
        node_curvatures.append(
            _exact_node_curvature_cushing_from_transition(P, support, mu, node_idx)
        )
    return torch.stack(node_curvatures).to(dtype=adj.dtype, device=edge_index.device)


def initialize_edge_curvature(
    base_edge_index: torch.Tensor,
    num_nodes: int,
    target_edge_index: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    mode: str = "proxy",
) -> torch.Tensor:
    target_edge_index = base_edge_index if target_edge_index is None else target_edge_index

    if mode == "proxy":
        row = base_edge_index[0]
        if edge_weight is None:
            deg = torch.bincount(row, minlength=num_nodes).float()
        else:
            deg = torch.zeros(num_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
            deg.scatter_add_(0, row, edge_weight)
        src_deg = deg[target_edge_index[0]]
        dst_deg = deg[target_edge_index[1]]
        return (1.0 / (1.0 + src_deg + dst_deg)).detach()

    if mode == "exact_cushing":
        node_curvature = _compute_exact_node_curvatures_cushing(
            base_edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
        edge_curvature = 0.5 * (
            node_curvature[target_edge_index[0]] + node_curvature[target_edge_index[1]]
        )
        return edge_curvature.to(dtype=torch.float32, device=target_edge_index.device).detach()

    if mode == "exact_cushing_affine":
        node_curvature = _compute_exact_node_curvatures_cushing(
            base_edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
        edge_curvature = 0.5 * (
            node_curvature[target_edge_index[0]] + node_curvature[target_edge_index[1]]
        )

        # Research-only calibration: match the old exact edge distribution
        # on the same graph so we can test whether scale/offset mismatch is
        # the dominant cause of the training gap.
        old_node_curvature = _compute_exact_node_curvatures(
            base_edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
        old_edge_curvature = 0.5 * (
            old_node_curvature[target_edge_index[0]] + old_node_curvature[target_edge_index[1]]
        )

        new_mean = edge_curvature.mean()
        new_std = edge_curvature.std(unbiased=False)
        old_mean = old_edge_curvature.mean()
        old_std = old_edge_curvature.std(unbiased=False)
        scale = old_std / new_std.clamp_min(1e-12)
        calibrated = (edge_curvature - new_mean) * scale + old_mean
        return calibrated.to(dtype=torch.float32, device=target_edge_index.device).detach()

    if mode != "exact":
        raise NotImplementedError(f"Unsupported curvature initialization mode: {mode}")

    node_curvature = _compute_exact_node_curvatures(base_edge_index, num_nodes, edge_weight=edge_weight)
    edge_curvature = 0.5 * (
        node_curvature[target_edge_index[0]] + node_curvature[target_edge_index[1]]
    )
    return edge_curvature.to(dtype=torch.float32, device=target_edge_index.device).detach()
