import argparse
import csv
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EXP_ROOT = BASE_DIR / "log_miml" / "scene"


# 中文注释：解析命令行参数并返回本次运行配置。
def parse_args():
    parser = argparse.ArgumentParser("Run PointNet MIML experiments over multiple seeds")
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train_miml_scene.py after a -- separator",
    )
    return parser.parse_args()


# 中文注释：把逗号分隔的文本解析为随机种子列表。
def parse_seeds(spec):
    seeds = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one seed")
    return seeds


# 中文注释：清理透传参数并阻止覆盖多种子运行器控制的参数。
def clean_train_args(train_args):
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    forbidden = {"--seed", "--log_dir", "--skip_test"}
    used_forbidden = [arg for arg in train_args if arg in forbidden]
    if used_forbidden:
        raise ValueError(
            "run_miml_multiseed.py controls --seed and --log_dir; "
            f"remove {used_forbidden} from forwarded train args"
        )
    return train_args


# 中文注释：读取单次实验生成的 JSON 汇总。
def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 中文注释：计算多次实验数值的均值、样本标准差和原始列表。
def aggregate_values(values):
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": [float(value) for value in values],
    }


# 中文注释：按数据划分和指标聚合多次实验结果。
def aggregate_summaries(summaries):
    aggregate = {
        "num_runs": len(summaries),
        "seeds": [summary["parameters"]["seed"] for summary in summaries],
        "selected_threshold": aggregate_values(
            [summary["selected_threshold"] for summary in summaries]
        ),
        "completed_epochs": aggregate_values(
            [summary["completed_epochs"] for summary in summaries]
        ),
        "splits": {},
    }

    for split_key in ["train_metrics", "validation_metrics", "test_metrics"]:
        metric_names = sorted(summaries[0][split_key].keys())
        aggregate["splits"][split_key] = {}
        for metric_name in metric_names:
            values = [summary[split_key][metric_name] for summary in summaries]
            aggregate["splits"][split_key][metric_name] = aggregate_values(values)
    return aggregate


# 中文注释：将多种子聚合结果写入 CSV 文件。
def write_aggregate_csv(path, aggregate):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "metric", "mean", "std", "values"])
        writer.writerow(
            [
                "run",
                "selected_threshold",
                aggregate["selected_threshold"]["mean"],
                aggregate["selected_threshold"]["std"],
                json.dumps(aggregate["selected_threshold"]["values"]),
            ]
        )
        writer.writerow(
            [
                "run",
                "completed_epochs",
                aggregate["completed_epochs"]["mean"],
                aggregate["completed_epochs"]["std"],
                json.dumps(aggregate["completed_epochs"]["values"]),
            ]
        )
        for split_key, metrics in aggregate["splits"].items():
            split = split_key.replace("_metrics", "")
            for metric_name, stats in metrics.items():
                writer.writerow(
                    [
                        split,
                        metric_name,
                        stats["mean"],
                        stats["std"],
                        json.dumps(stats["values"]),
                    ]
                )


# 中文注释：打印多种子测试集经典指标的均值与标准差。
def print_test_summary(aggregate):
    metrics = ["hamming_loss", "one_error", "coverage", "ranking_loss", "average_precision"]
    print("Multi-seed test summary:")
    for metric_name in metrics:
        stats = aggregate["splits"]["test_metrics"][metric_name]
        print(f"  {metric_name}: {stats['mean']:.6f} +/- {stats['std']:.6f}")


# 中文注释：执行 main 对应的数据处理或模型计算流程。
def main(args):
    seeds = parse_seeds(args.seeds)
    train_args = clean_train_args(args.train_args)
    run_name = args.run_name or datetime.datetime.now().strftime("multiseed_%Y-%m-%d_%H-%M")
    aggregate_dir = EXP_ROOT / run_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in seeds:
        log_dir = f"{run_name}/seed_{seed}"
        command = [
            sys.executable,
            str(BASE_DIR / "train_miml_scene.py"),
            "--seed",
            str(seed),
            "--log_dir",
            log_dir,
            *train_args,
        ]
        print(f"Running seed {seed}: {' '.join(command)}")
        subprocess.run(command, cwd=str(BASE_DIR), check=True)

        summary_path = EXP_ROOT / log_dir / "best_summary.json"
        summary = load_summary(summary_path)
        summaries.append(summary)

    aggregate = aggregate_summaries(summaries)
    payload = {
        "run_name": run_name,
        "seeds": seeds,
        "train_args": train_args,
        "summaries": [
            {
                "seed": summary["parameters"]["seed"],
                "path": str(EXP_ROOT / run_name / f"seed_{summary['parameters']['seed']}" / "best_summary.json"),
                "best_epoch": summary["best_epoch"],
                "completed_epochs": summary["completed_epochs"],
                "selected_threshold": summary["selected_threshold"],
                "test_metrics": summary["test_metrics"],
            }
            for summary in summaries
        ],
        "aggregate": aggregate,
    }
    json_path = aggregate_dir / "multi_seed_summary.json"
    csv_path = aggregate_dir / "multi_seed_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_aggregate_csv(csv_path, aggregate)

    print_test_summary(aggregate)
    print(f"Aggregate JSON: {json_path}")
    print(f"Aggregate CSV: {csv_path}")


if __name__ == "__main__":
    main(parse_args())
