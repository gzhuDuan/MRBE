from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset_utils import DataLoader
from .utils import accuracy, random_planetoid_splits, random_ratio_splits

from .task_curvature.model import TaskCurvatureGCN, TaskCurvatureGAT, TaskCurvatureGraphSAGE
from .task_curvature.rewiring import build_candidate_graph
from .task_curvature.surrogate import surrogate_update
from .task_curvature.target import compute_target_curvature, target_curvature_stats


def build_logger(log_file: Path, console: bool = False) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("task_curvature_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 2:
            return torch.device("cuda:2")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def write_config(run_dir: Path, args: argparse.Namespace, resolved_device: torch.device) -> Path:
    config_path = run_dir / "config.json"
    payload = vars(args).copy()
    payload["resolved_device"] = str(resolved_device)
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return config_path


def load_config_into_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    initial_args, _ = parser.parse_known_args()
    if not getattr(initial_args, "config", ""):
        return parser.parse_args()

    config_path = Path(initial_args.config)
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    parser.set_defaults(**config_data)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metric(mean_value: float, std_value: float) -> str:
    return f"{mean_value * 100:.2f} ± {std_value * 100:.2f}"


def log_box(logger: logging.Logger, title: str, rows: list[str]) -> None:
    width = max(len(title) + 4, *(len(row) + 4 for row in rows))
    border = "=" * width
    logger.info(border)
    logger.info(f"| {title.ljust(width - 4)} |")
    logger.info(border)
    for row in rows:
        logger.info(f"| {row.ljust(width - 4)} |")
    logger.info(border)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset", default="wisconsin")
    parser.add_argument("--device", default="2")
    parser.add_argument("--task-type", default="node_classification")
    parser.add_argument("--method-name", default="task_curvature")
    parser.add_argument("--model-type", default="task_curvature_gcn", choices=["task_curvature_gcn", "task_curvature_gat", "task_curvature_graphsage"])
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads (GAT only)")
    parser.add_argument("--protocol", default="dev", choices=["dev", "final", "custom"])
    parser.add_argument("--split-mode", default="ratio", choices=["ratio", "planetoid"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--edge-weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rate", type=float, default=0.48)
    parser.add_argument("--val-rate", type=float, default=0.32)
    parser.add_argument("--percls-trn", type=int, default=5)
    parser.add_argument("--val-lb", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=5)
    parser.add_argument("--candidate-init", type=float, default=0.08)
    parser.add_argument("--init-curvature-mode", default="exact", choices=["proxy", "exact", "exact_cushing", "exact_cushing_affine"])
    parser.add_argument("--target-gamma", type=float, default=1.0)
    parser.add_argument("--target-tau", type=float, default=1.0)
    parser.add_argument("--target-lambda", type=float, default=0.5)
    parser.add_argument("--target-entropy-eps", type=float, default=1e-12)
    parser.add_argument("--surrogate-mix", type=float, default=0.5)
    parser.add_argument("--surrogate-alpha", type=float, default=0.5)
    parser.add_argument("--surrogate-beta", type=float, default=0.5)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--refresh-interval", type=int, default=5)
    parser.add_argument("--rebuild-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbosity", default="compact", choices=["compact", "detailed"])
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--console-log", action="store_true")
    args = load_config_into_args(parser)

    if args.protocol == "dev":
        args.runs = 10
        args.epochs = 200
    elif args.protocol == "final":
        args.runs = 20
        args.epochs = 200

    device = resolve_device(args.device)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or "run"
    run_dir = Path(args.log_dir) / args.task_type / args.dataset / timestamp
    log_file = run_dir / f"{run_name}.log"
    logger = build_logger(log_file, console=args.console_log)
    config_path = write_config(run_dir, args, device)

    dataset = DataLoader(args.dataset)
    logger.info(
        f"device={device} "
        f"task_type={args.task_type} "
        f"method_name={args.method_name} "
        f"model_type={args.model_type} "
        f"protocol={args.protocol} "
        f"split_mode={args.split_mode} "
        f"dataset={args.dataset} "
        f"runs={args.runs} "
        f"seed={args.seed} "
        f"epochs={args.epochs} "
        f"hidden={args.hidden} "
        f"lr={args.lr} "
        f"weight_decay={args.weight_decay} "
        f"edge_weight_decay={args.edge_weight_decay} "
        f"dropout={args.dropout} "
        f"candidate_init={args.candidate_init} "
        f"init_curvature_mode={args.init_curvature_mode} "
        f"target_tau={args.target_tau} "
        f"target_lambda={args.target_lambda} "
        f"surrogate_mix={args.surrogate_mix} "
        f"early_stopping={args.early_stopping} "
        f"refresh_interval={args.refresh_interval}"
    )
    if device.type == "cuda":
        logger.info(f"cuda_device={torch.cuda.get_device_name(device)}")
    logger.info(f"log_dir={run_dir}")
    logger.info(f"config_path={config_path}")
    log_box(
        logger,
        "Experiment Protocol",
        [
            f"protocol: {args.protocol}",
            f"runs: {args.runs}",
            f"epochs_per_run: {args.epochs}",
            f"split_mode: {args.split_mode}",
            f"train/val/test: {args.train_rate:.2f}/{args.val_rate:.2f}/{1.0 - args.train_rate - args.val_rate:.2f}" if args.split_mode == "ratio" else f"planetoid: percls_trn={args.percls_trn}, val_lb={args.val_lb}",
            f"selection_rule: best test at best val",
            f"init_curvature: {args.init_curvature_mode}",
            f"target(lambda,tau): {args.target_lambda:.3f}/{args.target_tau:.3f}",
            f"surrogate(mix): {args.surrogate_mix:.3f}",
            f"weight_decay(edge): {args.weight_decay}/{args.edge_weight_decay}",
            f"early_stopping: {args.early_stopping}",
            f"device: {device}",
        ],
    )

    run_summaries: list[dict[str, float]] = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_seed(run_seed)

        data = dataset[0].clone()
        if args.split_mode == "ratio":
            data = random_ratio_splits(data, train_rate=args.train_rate, val_rate=args.val_rate)
        else:
            data = random_planetoid_splits(
                data,
                dataset.num_classes,
                percls_trn=args.percls_trn,
                val_lb=args.val_lb,
            )
        x = data.x.float().to(device)
        y = data.y.long().to(device)
        idx_train = data.train_mask.to(device)
        idx_val = data.val_mask.to(device)
        idx_test = data.test_mask.to(device)

        rewiring_state = build_candidate_graph(
            data.edge_index.to(device),
            x,
            data.num_nodes,
            k=args.candidate_k,
            candidate_init=args.candidate_init,
        )

        model_kwargs = dict(
            num_features=x.size(1),
            hidden_dim=args.hidden,
            num_classes=int(y.max().item()) + 1,
            rewiring_state=rewiring_state,
            dropout=args.dropout,
            init_curvature_mode=args.init_curvature_mode,
        )
        if args.model_type == "task_curvature_gat":
            model_kwargs["heads"] = args.heads
            model = TaskCurvatureGAT(**model_kwargs).to(device)
        elif args.model_type == "task_curvature_graphsage":
            model = TaskCurvatureGraphSAGE(**model_kwargs).to(device)
        else:
            model = TaskCurvatureGCN(**model_kwargs).to(device)
        curvature_state = model.init_curvature_state()
        curvature_state.current_kappa_edge = curvature_state.current_kappa_edge.to(device)

        logger.info(
            f"[run {run_idx + 1:02d}/{args.runs:02d}] seed={run_seed} "
            f"base_edges={rewiring_state.edge_index_base.size(1)} "
            f"candidate_edges={rewiring_state.edge_index_candidate.size(1)}"
        )

        optimizer = torch.optim.Adam(
            [
                {"params": model.gnn_parameters(), "weight_decay": args.weight_decay},
                {"params": model.edge_weights.parameters(), "weight_decay": args.edge_weight_decay},
            ],
            lr=args.lr,
        )
        best_val_acc = float("-inf")
        best_test_acc = float("-inf")
        best_train_acc = float("-inf")
        best_epoch = -1
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(x)
            loss_task = F.cross_entropy(out.logits[idx_train], y[idx_train])
            loss_curv = torch.tensor(0.0, device=device)

            target_kappa = compute_target_curvature(
                out.embeddings,
                out.logits,
                out.edge_index,
                model.kappa0_edge,
                tau=args.target_tau,
                lam=args.target_lambda,
                entropy_eps=args.target_entropy_eps,
            )
            loss_curv = F.mse_loss(curvature_state.current_kappa_edge, target_kappa)

            loss = loss_task + args.alpha * loss_curv
            loss.backward()
            old_active_weight = out.active_edge_weight.detach()
            optimizer.step()

            new_active_weight = model.edge_weights.active_weights().detach()
            delta_weight = new_active_weight - old_active_weight
            curvature_state.current_kappa_edge = surrogate_update(
                curvature_state.current_kappa_edge,
                old_active_weight,
                delta_weight,
                out.edge_index,
                num_nodes=x.size(0),
                mix=args.surrogate_mix,
            ).detach()
            if args.refresh_interval > 0 and (epoch + 1) % args.refresh_interval == 0:
                curvature_state.current_kappa_edge = model.refresh_curvature()
                if args.verbosity == "detailed":
                    logger.info(
                        f"[run {run_idx + 1:02d}] refresh at epoch {epoch + 1}: "
                        f"kappa_mean={curvature_state.current_kappa_edge.mean().item():.6f} "
                        f"kappa_std={curvature_state.current_kappa_edge.std().item():.6f}"
                    )

            if args.rebuild_interval > 0 and (epoch + 1) % args.rebuild_interval == 0:
                before = model.rewiring_state.edge_index_candidate.size(1)
                model.hard_rebuild()
                after = model.rewiring_state.edge_index_candidate.size(1)
                curvature_state.current_kappa_edge = model.kappa0_edge.detach().clone()
                if args.verbosity == "detailed":
                    logger.info(f"[run {run_idx + 1:02d}] hard rebuild at epoch {epoch + 1}: edges {before} -> {after}")

            model.eval()
            with torch.inference_mode():
                eval_out = model(x)
                train_acc = accuracy(eval_out.logits[idx_train], y[idx_train]).item()
                val_acc = accuracy(eval_out.logits[idx_val], y[idx_val]).item()
                test_acc = accuracy(eval_out.logits[idx_test], y[idx_test]).item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_train_acc = train_acc
                best_epoch = epoch
                patience_counter = 0
                logger.info(
                    f"[run {run_idx + 1:02d}] [best update] epoch={epoch:03d} "
                    f"train={train_acc:.4f} val={val_acc:.4f} test={test_acc:.4f}"
                )
            else:
                patience_counter += 1

            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                break

            if args.verbosity == "detailed" and (epoch % args.log_interval == 0 or epoch == args.epochs - 1):
                t_stats = target_curvature_stats(
                    eval_out.embeddings,
                    eval_out.logits,
                    eval_out.edge_index,
                    model.kappa0_edge,
                    tau=args.target_tau,
                    lam=args.target_lambda,
                    entropy_eps=args.target_entropy_eps,
                )
                logger.info(
                    f"[run {run_idx + 1:02d}] [epoch {epoch:03d}] "
                    f"loss={loss.item():.4f} "
                    f"task={loss_task.item():.4f} "
                    f"curv={loss_curv.item():.4f} "
                    f"train={train_acc:.4f} "
                    f"val={val_acc:.4f} "
                    f"test={test_acc:.4f} "
                    f"active_edges={(eval_out.active_edge_weight > 0).sum().item()} "
                    f"sim_mean={t_stats.sim_mean:.4f} "
                    f"gate_mean={t_stats.gate_mean:.4f} "
                    f"target_mean={t_stats.target_mean:.4f}"
                )

        run_summaries.append(
            {
                "best_epoch": float(best_epoch),
                "best_train": best_train_acc,
                "best_val": best_val_acc,
                "best_test_at_best_val": best_test_acc,
            }
        )
        logger.info(
            f"[run {run_idx + 1:02d}] [summary] "
            f"best_epoch={best_epoch:03d} "
            f"best_train={best_train_acc:.4f} "
            f"best_val={best_val_acc:.4f} "
            f"best_test_at_best_val={best_test_acc:.4f}"
        )

    best_train_values = np.array([item["best_train"] for item in run_summaries], dtype=float)
    best_val_values = np.array([item["best_val"] for item in run_summaries], dtype=float)
    best_test_values = np.array([item["best_test_at_best_val"] for item in run_summaries], dtype=float)

    logger.info("[final summary]")
    log_box(
        logger,
        "Final Summary",
        [
            f"method: {args.method_name}",
            f"model: {args.model_type}",
            f"dataset: {args.dataset}",
            f"runs: {args.runs}",
            f"best_train: {format_metric(best_train_values.mean(), best_train_values.std(ddof=1))}",
            f"best_val: {format_metric(best_val_values.mean(), best_val_values.std(ddof=1))}",
            f"best_test_at_best_val: {format_metric(best_test_values.mean(), best_test_values.std(ddof=1))}",
        ],
    )
    logger.info("training finished")
    print(f"log written to {log_file}")
    print(f"config written to {config_path}")


if __name__ == "__main__":
    main()
