"""
将Scene MIML数据集从MATLAB格式转换为NumPy压缩格式，同时打乱并划分数据集，打乱种子默认为42。
输出至data文件夹下的scene_full_pointnet.npz文件，包含训练集和测试集的实例、标签、索引、划分种子、标签名称和数据源URL
"""
import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

LABEL_NAMES = np.asarray(["desert", "mountains", "sea", "sunset", "trees"])

def parse_args():
    """
    解析命令行参数并返回本次运行配置
    """
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser("Convert the complete LAMDA Scene MIML data")
    parser.add_argument(
        "--source",
        type=Path,
        default=project_dir / "data" / "scene" / "miml data.mat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "scene" / "scene_full_pointnet.npz",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=1500)
    return parser.parse_args()


def stack_bags(mat_bags):
    """
    校验MATLAB bag形状并堆叠为统一三维数组
    """
    bags = [np.asarray(bag, dtype=np.float32) for bag in mat_bags.ravel()]
    shapes = {bag.shape for bag in bags}
    if len(shapes) != 1:
        raise ValueError(f"Expected equal-size Scene bags, found shapes: {sorted(shapes)}")
    return np.stack(bags, axis=0)


def convert_labels(mat_targets):
    """
    把MATLAB的{-1, 1}标签矩阵转换为{0, 1}
    """
    labels = np.asarray(mat_targets.T) #因为MATLAB的targets是一个5xN矩阵，转置后变为Nx5
    return (labels == 1).astype(np.float32)


def main():
    """
    执行数据处理流程
    """
    args = parse_args()
    data = sio.loadmat(args.source)
    instances = stack_bags(data["bags"])
    labels = convert_labels(data["targets"])
    if len(instances) != len(labels):
        raise ValueError("Bag and target counts do not match")
    if not 0 < args.train_size < len(instances):
        raise ValueError("--train_size must leave at least one train and test bag")

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(instances))
    train_indices = indices[: args.train_size]
    test_indices = indices[args.train_size :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=instances[train_indices],
        y_train=labels[train_indices],
        x_test=instances[test_indices],
        y_test=labels[test_indices],
        train_indices=train_indices,
        test_indices=test_indices,
        split_seed=np.asarray(args.seed),
        label_names=LABEL_NAMES,
        source_url=np.asarray(
            "https://www.lamda.nju.edu.cn/data_MIMLimage.ashx"
        ),
    )
    print(f"wrote {args.output}")
    print(f"all={instances.shape}, labels={labels.shape}")
    print(f"train={len(train_indices)}, test={len(test_indices)}, seed={args.seed}")


if __name__ == "__main__":
    main()
