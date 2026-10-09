import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils.SceneMIMLDataLoader import SceneMIMLDataLoader
from models.pointnet_miml_cls import get_loss, get_model


BASE_DIR = Path(__file__).resolve().parent


# 中文注释：返回当前项目内默认数据文件的绝对路径。
def default_data_path():
    return str(BASE_DIR / "data" / "scene" / "scene_full_pointnet.npz")


# 中文注释：解析命令行参数并返回本次运行配置。
def parse_args():
    parser = argparse.ArgumentParser("PointNet MIML smoke test")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
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
    parser.add_argument("--data_path", default=default_data_path())
    return parser.parse_args()


# 中文注释：执行 main 对应的数据处理或模型计算流程。
def main(args):
    use_cuda = torch.cuda.is_available() and not args.use_cpu
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    dataset = SceneMIMLDataLoader(args.data_path, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    points, target = next(iter(loader))

    classifier = get_model(
        num_labels=dataset.num_labels,
        input_dim=dataset.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        norm_type=args.norm_type,
        pooling=args.mil_pooling,
        lse_temperature=args.lse_temperature,
        set_attention_layers=args.set_attention_layers,
        set_attention_heads=args.set_attention_heads,
    ).to(device)
    criterion = get_loss(
        ranking_loss_weight=args.ranking_loss_weight,
        ranking_margin=args.ranking_margin,
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    points = points.to(device).transpose(2, 1)
    target = target.to(device)
    logits, aux = classifier(points)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    prob = torch.sigmoid(logits).detach().cpu()
    pred = prob >= args.threshold
    label_names = dataset.label_names
    first_pred = [name for name, flag in zip(label_names, pred[0].tolist()) if flag]
    first_true = [name for name, flag in zip(label_names, target[0].cpu().tolist()) if flag >= 0.5]

    print("Smoke test passed.")
    print(f"data_path: {args.data_path}")
    print(f"input before transpose: [batch, instances, feature_dim] = {tuple(points.transpose(2, 1).shape)}")
    print(f"input to model: [batch, feature_dim, instances] = {tuple(points.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    if isinstance(aux, dict):
        instance_logits = aux.get("instance_logits")
        attention_weights = aux.get("attention_weights")
        if instance_logits is not None:
            print(f"instance logits shape: {tuple(instance_logits.shape)}")
        if attention_weights is not None:
            print(f"attention weights shape: {tuple(attention_weights.shape)}")
    print(f"loss after one backward step: {loss.item():.6f}")
    print("first sample probabilities:")
    print({name: round(float(score), 4) for name, score in zip(label_names, prob[0])})
    print(f"first sample predicted labels: {first_pred if first_pred else ['<none>']}")
    print(f"first sample true labels: {first_true if first_true else ['<none>']}")


if __name__ == "__main__":
    main(parse_args())
