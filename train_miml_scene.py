import argparse
import datetime
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_utils.SceneMIMLDataLoader import SceneMIMLDataLoader
from miml_metrics import compute_metrics, sigmoid
from models.pointnet_miml_cls import get_loss, get_model


BASE_DIR = Path(__file__).resolve().parent


# 中文注释：返回当前项目内默认数据文件的绝对路径。
def default_data_path():
    return str(BASE_DIR / "data" / "scene" / "scene_full_pointnet.npz")


# 中文注释：解析命令行参数并返回本次运行配置。
def parse_args():
    parser = argparse.ArgumentParser("PointNet MIML training on Scene")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold_selection", default="validation", choices=["fixed", "validation"])
    parser.add_argument("--threshold_metric", default="hamming_loss", choices=["hamming_loss", "micro_f1", "macro_f1"])
    parser.add_argument("--threshold_candidates", default="0.05:0.95:0.05")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--norm_type", default="layer", choices=["batch", "layer"])
    parser.add_argument(
        "--mil_pooling",
        default="lse",
        choices=["max", "lse", "mean", "noisy_or", "attention"],
    )
    parser.add_argument("--lse_temperature", type=float, default=1.0)
    parser.add_argument("--set_attention_layers", type=int, default=0)
    parser.add_argument("--set_attention_heads", type=int, default=4)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.1)
    parser.add_argument("--ranking_margin", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_path", default=default_data_path())
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--smoke_test", action="store_true", default=False)
    parser.add_argument("--show_samples", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=False)
    return parser.parse_args()


# 中文注释：将模块中的 ReLU 设置为原地计算以减少显存占用。
def inplace_relu(module):
    classname = module.__class__.__name__
    if "ReLU" in classname:
        module.inplace = True


# 中文注释：统一设置 Python、NumPy 与 PyTorch 的随机种子。
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 中文注释：按给定比例和随机种子拆分训练集与验证集。
def split_train_val(dataset, val_fraction, seed):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1")

    num_samples = len(dataset)
    val_size = int(round(num_samples * val_fraction))
    val_size = max(1, min(num_samples - 1, val_size))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices


# 中文注释：解析阈值列表或范围表达式并校验取值。
def parse_threshold_candidates(spec):
    if ":" in spec:
        parts = [float(part) for part in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("--threshold_candidates range must be start:end:step")
        start, end, step = parts
        if step <= 0:
            raise ValueError("--threshold_candidates step must be positive")
        values = []
        current = start
        while current <= end + step * 1e-6:
            values.append(round(current, 10))
            current += step
    else:
        values = [float(part.strip()) for part in spec.split(",") if part.strip()]

    values = sorted(set(values))
    if not values:
        raise ValueError("--threshold_candidates must contain at least one threshold")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold candidates must be between 0 and 1")
    return values


# 中文注释：按照指标方向判断候选值是否优于当前值。
def metric_is_better(candidate, incumbent, metric_name):
    if incumbent is None:
        return True
    if metric_name == "hamming_loss":
        return candidate < incumbent
    return candidate > incumbent


# 中文注释：仅使用验证集指标选择最终预测阈值。
def select_threshold(logits, targets, candidates, metric_name):
    best_threshold = candidates[0]
    best_metric_value = None
    best_metrics = None
    for threshold in candidates:
        metrics = compute_metrics(logits, targets, threshold=threshold)
        metric_value = metrics[metric_name]
        if metric_is_better(metric_value, best_metric_value, metric_name):
            best_threshold = threshold
            best_metric_value = metric_value
            best_metrics = metrics
    return best_threshold, best_metric_value, best_metrics


# 中文注释：计算阈值相关指标并附加给定损失值。
def metrics_with_loss(logits, targets, loss, threshold):
    metrics = compute_metrics(logits, targets, threshold=threshold)
    metrics["loss"] = loss
    return metrics


# 中文注释：在不更新参数的条件下评估模型并汇总 logits、标签和指标。
def evaluate(classifier, criterion, loader, device, threshold, desc="Eval"):
    classifier.eval()
    loss_sum = 0.0
    sample_count = 0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        progress = tqdm(total=len(loader), desc=desc, leave=False)
        for points, target in loader:
            points = points.to(device).transpose(2, 1)
            target = target.to(device)
            logits, _ = classifier(points)
            loss = criterion(logits, target)
            batch_size = target.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")
            progress.update(1)
        progress.close()

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(logits, targets, threshold=threshold)
    metrics["loss"] = float(loss_sum / sample_count)
    return metrics, logits, targets


# 中文注释：把概率向量转换为超过阈值的标签名称列表。
def labels_from_prob(prob, label_names, threshold):
    selected = [label for label, score in zip(label_names, prob) if score >= threshold]
    return selected if selected else ["<none>"]


# 中文注释：返回经典 MIML 指标的含义与优化方向。
def metric_legend():
    return {
        "hamming_loss": "fraction of wrongly predicted label decisions, lower is better",
        "one_error": "fraction whose top-ranked label is not relevant, lower is better",
        "coverage": "average rank depth needed to cover all true labels, lower is better",
        "ranking_loss": "fraction of positive-negative label pairs ranked incorrectly, lower is better",
        "average_precision": "example-based MIML average precision, higher is better",
    }


# 中文注释：返回用于论文对比的经典 MIML 指标名称。
def classic_metric_names():
    return ["hamming_loss", "one_error", "coverage", "ranking_loss", "average_precision"]


# 中文注释：打印经典 MIML 指标。
def print_classic_metrics(prefix, metrics):
    text = ", ".join(f"{name}={metrics[name]:.6f}" for name in classic_metric_names())
    print(f"{prefix}: {text}")


# 中文注释：打印单个 epoch 的损失与经典指标。
def print_epoch_metrics(prefix, metrics):
    text = ", ".join(f"{name}={metrics[name]:.6f}" for name in classic_metric_names())
    print(f"{prefix}: loss={metrics['loss']:.6f}, {text}")


# 中文注释：以表格形式打印训练、验证和测试指标。
def print_split_metrics_table(split_metrics):
    print("Best model performance on train/validation/test sets:")
    header = f"{'split':<8} {'hamming_loss':>14} {'one_error':>12} {'coverage':>12} {'ranking_loss':>14} {'avg_precision':>15}"
    print(header)
    print("-" * len(header))
    for split, metrics in split_metrics:
        print(
            f"{split:<8} "
            f"{metrics['hamming_loss']:>14.6f} "
            f"{metrics['one_error']:>12.6f} "
            f"{metrics['coverage']:>12.6f} "
            f"{metrics['ranking_loss']:>14.6f} "
            f"{metrics['average_precision']:>15.6f}"
        )


# 中文注释：打印各个 MIML 指标的解释。
def print_metric_legend():
    print("Metric labels:")
    for name, description in metric_legend().items():
        print(f"  {name}: {description}")


# 中文注释：将可序列化结果以 UTF-8 JSON 格式写入磁盘。
def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# 中文注释：执行 main 对应的数据处理或模型计算流程。
def main(args):
    set_random_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    threshold_candidates = parse_threshold_candidates(args.threshold_candidates)
    selected_threshold = args.threshold
    selected_threshold_metric_value = None

    train_dataset = SceneMIMLDataLoader(args.data_path, split="train")
    test_dataset = SceneMIMLDataLoader(args.data_path, split="test")
    train_subset, val_subset, train_indices, val_indices = split_train_val(
        train_dataset, args.val_fraction, args.seed
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        generator=train_generator,
    )
    train_eval_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    exp_root = BASE_DIR / "log_miml" / "scene"
    exp_root.mkdir(parents=True, exist_ok=True)
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = exp_root / (args.log_dir if args.log_dir else timestr)
    checkpoints_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    classifier = get_model(
        num_labels=train_dataset.num_labels,
        input_dim=train_dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pooling=args.mil_pooling,
        lse_temperature=args.lse_temperature,
        set_attention_layers=args.set_attention_layers,
        set_attention_heads=args.set_attention_heads,
    )
    criterion = get_loss(
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
    )
    classifier.apply(inplace_relu)
    classifier = classifier.to(device)
    criterion = criterion.to(device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    print("Dataset summary:")
    print(train_dataset.summary())
    print(test_dataset.summary())
    print(
        f"Validation split: {len(train_subset)} train / {len(val_subset)} validation "
        f"from original training set (seed={args.seed}, val_fraction={args.val_fraction})"
    )
    print(f"Device: {device}")
    print(f"Experiment directory: {exp_dir}")
    print(
        f"Threshold protocol: {args.threshold_selection} "
        f"(metric={args.threshold_metric}, candidates={threshold_candidates})"
    )
    print_metric_legend()

    shutil.copy(str(BASE_DIR / "models" / "pointnet_miml_cls.py"), str(exp_dir))
    shutil.copy(str(BASE_DIR / "train_miml_scene.py"), str(exp_dir))

    best_ap = -1.0
    best_epoch = 0
    best_train_loss = None
    best_train_metrics = None
    best_val_metrics = None
    best_test_metrics = None
    test_logits = None
    test_targets = None
    best_model_path = checkpoints_dir / "best_model.pth"
    total_epochs = 1 if args.smoke_test else args.epoch
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(total_epochs):
        completed_epochs = epoch + 1
        classifier.train()
        loss_sum = 0.0
        sample_count = 0
        train_total = 1 if args.smoke_test else len(train_loader)
        progress = tqdm(total=train_total, desc=f"Epoch {epoch + 1}/{total_epochs} [train]", smoothing=0.9)
        for batch_id, (points, target) in enumerate(train_loader):
            optimizer.zero_grad()
            points = points.to(device).transpose(2, 1)
            target = target.to(device)
            logits, _ = classifier(points)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            batch_size = target.size(0)
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{loss_sum / sample_count:.4f}",
                lr=f"{current_lr:.2e}",
            )
            progress.update(1)

            if args.smoke_test and batch_id >= 0:
                break
        progress.close()

        scheduler.step()
        train_loss = float(loss_sum / sample_count)
        val_metrics, _, _ = evaluate(
            classifier,
            criterion,
            val_loader,
            device,
            args.threshold,
            desc=f"Epoch {epoch + 1}/{total_epochs} [val]",
        )
        print(f"Epoch {epoch + 1}/{total_epochs}")
        print(f"Train loss: {train_loss:.6f}")
        print_epoch_metrics("Validation", val_metrics)

        if val_metrics["average_precision"] > best_ap + args.early_stop_min_delta:
            best_ap = val_metrics["average_precision"]
            epochs_without_improvement = 0
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_train_metrics, _, _ = evaluate(
                classifier,
                criterion,
                train_eval_loader,
                device,
                args.threshold,
                desc=f"Epoch {epoch + 1}/{total_epochs} [best-train-eval]",
            )
            best_val_metrics = dict(val_metrics)
            torch.save(
                {
                    "epoch": best_epoch,
                    "selection_metric": "validation_average_precision",
                    "validation_average_precision": best_ap,
                    "train_metrics": best_train_metrics,
                    "validation_metrics": best_val_metrics,
                    "train_loss": best_train_loss,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "train_indices": train_indices,
                    "validation_indices": val_indices,
                    "label_names": train_dataset.label_names,
                },
                best_model_path,
            )
            save_json(
                exp_dir / "best_summary.json",
                {
                    "best_epoch": best_epoch,
                    "completed_epochs": completed_epochs,
                    "selection_metric": "validation_average_precision",
                    "best_selection_value": best_ap,
                    "train_loss": best_train_loss,
                    "train_metrics": best_train_metrics,
                    "validation_metrics": best_val_metrics,
                    "test_metrics": None,
                    "selected_threshold": None,
                    "threshold_selection": args.threshold_selection,
                    "threshold_metric": args.threshold_metric,
                    "threshold_candidates": threshold_candidates,
                    "classic_metric_names": classic_metric_names(),
                    "parameters": vars(args),
                    "data_path": args.data_path,
                    "train_indices": train_indices,
                    "validation_indices": val_indices,
                    "label_names": train_dataset.label_names,
                    "checkpoint": str(best_model_path),
                    "metric_legend": metric_legend(),
                },
            )
            print(f"New best model: epoch={best_epoch}, validation_average_precision={best_ap:.6f}")
        else:
            epochs_without_improvement += 1
            print(
                "No validation AP improvement: "
                f"{epochs_without_improvement}/{args.early_stop_patience}"
            )

        if (
            not args.smoke_test
            and args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"validation average_precision did not improve for "
                f"{args.early_stop_patience} epochs."
            )
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    final_train_metrics, train_logits, train_targets = evaluate(
        classifier,
        criterion,
        train_eval_loader,
        device,
        args.threshold,
        desc="Final train [best checkpoint]",
    )
    final_val_metrics, val_logits, val_targets = evaluate(
        classifier,
        criterion,
        val_loader,
        device,
        args.threshold,
        desc="Final validation [threshold selection]",
    )
    if args.threshold_selection == "validation":
        (
            selected_threshold,
            selected_threshold_metric_value,
            selected_threshold_val_metrics,
        ) = select_threshold(
            val_logits,
            val_targets,
            threshold_candidates,
            args.threshold_metric,
        )
    else:
        selected_threshold = args.threshold
        selected_threshold_val_metrics = compute_metrics(
            val_logits,
            val_targets,
            threshold=selected_threshold,
        )
        selected_threshold_metric_value = selected_threshold_val_metrics[args.threshold_metric]

    best_train_metrics = metrics_with_loss(
        train_logits,
        train_targets,
        final_train_metrics["loss"],
        selected_threshold,
    )
    best_val_metrics = metrics_with_loss(
        val_logits,
        val_targets,
        final_val_metrics["loss"],
        selected_threshold,
    )
    if args.skip_test:
        best_test_metrics = None
        test_logits = None
        test_targets = None
        print("Final test skipped by --skip_test.")
    else:
        best_test_metrics, test_logits, test_targets = evaluate(
            classifier,
            criterion,
            test_loader,
            device,
            selected_threshold,
            desc="Final test [best checkpoint]",
        )
    checkpoint["selected_threshold"] = selected_threshold
    checkpoint["threshold_selection"] = args.threshold_selection
    checkpoint["threshold_metric"] = args.threshold_metric
    checkpoint["threshold_metric_value"] = selected_threshold_metric_value
    checkpoint["threshold_candidates"] = threshold_candidates
    torch.save(checkpoint, best_model_path)
    save_json(
        exp_dir / "best_summary.json",
        {
            "best_epoch": best_epoch,
            "completed_epochs": completed_epochs,
            "selection_metric": "validation_average_precision",
            "best_selection_value": best_ap,
            "train_loss": best_train_loss,
            "train_metrics": best_train_metrics,
            "validation_metrics": best_val_metrics,
            "test_metrics": best_test_metrics,
            "selected_threshold": selected_threshold,
            "threshold_selection": args.threshold_selection,
            "threshold_metric": args.threshold_metric,
            "threshold_metric_value": selected_threshold_metric_value,
            "threshold_candidates": threshold_candidates,
            "classic_metric_names": classic_metric_names(),
            "parameters": vars(args),
            "data_path": args.data_path,
            "train_indices": train_indices,
            "validation_indices": val_indices,
            "label_names": train_dataset.label_names,
            "checkpoint": str(best_model_path),
            "metric_legend": metric_legend(),
        },
    )

    print("\nBest model summary")
    print(f"Best epoch: {best_epoch}/{completed_epochs}")
    print(f"Selection metric: validation_average_precision={best_ap:.6f}")
    print(
        f"Selected threshold: {selected_threshold:.6f} "
        f"({args.threshold_selection}, {args.threshold_metric}={selected_threshold_metric_value:.6f})"
    )
    print(f"Checkpoint: {best_model_path}")
    print("Best training parameters:")
    for name in [
        "data_path",
        "batch_size",
        "epoch",
        "learning_rate",
        "optimizer",
        "decay_rate",
        "threshold",
        "threshold_selection",
        "threshold_metric",
        "threshold_candidates",
        "val_fraction",
        "seed",
        "hidden_dim",
        "dropout",
        "norm_type",
        "mil_pooling",
        "lse_temperature",
        "set_attention_layers",
        "set_attention_heads",
        "ranking_loss_weight",
        "ranking_margin",
        "early_stop_patience",
        "early_stop_min_delta",
        "gpu",
        "use_cpu",
        "skip_test",
    ]:
        print(f"  {name}: {getattr(args, name)}")
    print(f"Best train loss: {best_train_loss:.6f}")
    split_metrics = [
        ("train", best_train_metrics),
        ("val", best_val_metrics),
    ]
    if best_test_metrics is not None:
        split_metrics.append(("test", best_test_metrics))
    print_split_metrics_table(split_metrics)
    print_classic_metrics("Best train", best_train_metrics)
    print_classic_metrics("Best validation", best_val_metrics)
    if best_test_metrics is not None:
        print_classic_metrics("Best test", best_test_metrics)
    print(f"Best summary JSON: {exp_dir / 'best_summary.json'}")

    if args.show_samples and test_logits is not None:
        probs = sigmoid(test_logits[:5])
        print("Sample predictions:")
        for i, prob in enumerate(probs):
            pred_labels = labels_from_prob(prob, train_dataset.label_names, selected_threshold)
            true_labels = labels_from_prob(test_targets[i], train_dataset.label_names, 0.5)
            prob_text = ", ".join(
                f"{name}={score:.3f}" for name, score in zip(train_dataset.label_names, prob)
            )
            print(f"[{i}] prob: {prob_text}")
            print(f"    pred: {pred_labels}")
            print(f"    true: {true_labels}")
    elif args.show_samples:
        print("Sample predictions skipped because --skip_test was set.")


if __name__ == "__main__":
    main(parse_args())
