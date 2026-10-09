import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUN = BASE_DIR / "log_miml" / "scene" / "full_multiseed_5" / "multi_seed_summary.json"
DEFAULT_BASELINE = BASE_DIR / "baselines" / "scene_literature.json"
DEFAULT_OUTPUT = BASE_DIR / "results" / "scene_baseline_comparison.json"


# 中文注释：解析命令行参数并返回本次运行配置。
def parse_args():
    parser = argparse.ArgumentParser("Compare PointNet-MIML with published Scene baselines")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# 中文注释：读取并解析指定 JSON 文件。
def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# 中文注释：依据指标方向找出最强的文献基线及其均值。
def strongest_baseline(baseline, metric):
    direction = baseline["metrics"][metric]
    values = {
        name: metrics[metric][0]
        for name, metrics in baseline["algorithms"].items()
    }
    selector = min if direction == "lower" else max
    name = selector(values, key=values.get)
    return name, values[name]


# 中文注释：执行 main 对应的数据处理或模型计算流程。
def main():
    args = parse_args()
    run = load_json(args.run)
    baseline = load_json(args.baseline)
    rows = []
    wins = 0

    for metric, direction in baseline["metrics"].items():
        pointnet = run["aggregate"]["splits"]["test_metrics"][metric]
        baseline_name, baseline_mean = strongest_baseline(baseline, metric)
        delta = pointnet["mean"] - baseline_mean
        meets_or_beats = delta <= 0 if direction == "lower" else delta >= 0
        wins += int(meets_or_beats)
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "pointnet_mean": pointnet["mean"],
                "pointnet_std": pointnet["std"],
                "strongest_published_model": baseline_name,
                "strongest_published_mean": baseline_mean,
                "pointnet_minus_published": delta,
                "meets_or_beats_strongest": meets_or_beats,
            }
        )

    comparison = {
        "pointnet_run": str(args.run.resolve()),
        "pointnet_protocol": {
            "seeds": run["seeds"],
            "partition": "fixed 1500/500 split generated with seed 42",
            "validation": "20% of the 1500 training bags per seed",
            "test_usage": "once per seed after validation checkpoint and threshold selection",
        },
        "published_source": baseline["source"],
        "published_protocol": baseline["protocol"],
        "protocol_comparable_but_not_identical": True,
        "metrics_meeting_or_beating_strongest_published": wins,
        "metrics_total": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)

    print(f"{'metric':<20} {'PointNet mean+/-std':>23} {'published best':>24} {'result':>10}")
    print("-" * 82)
    for row in rows:
        pointnet_text = f"{row['pointnet_mean']:.4f} +/- {row['pointnet_std']:.4f}"
        published_text = (
            f"{row['strongest_published_mean']:.4f} "
            f"({row['strongest_published_model']})"
        )
        result = "meets" if row["meets_or_beats_strongest"] else "below best"
        print(f"{row['metric']:<20} {pointnet_text:>23} {published_text:>24} {result:>10}")
    print(f"\nMeets or beats the strongest published mean on {wins}/{len(rows)} metrics.")
    print("Protocols are comparable but not identical; see the output JSON for details.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
