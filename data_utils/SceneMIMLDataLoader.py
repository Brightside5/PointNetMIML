"""
设置DataSet便于传入DataLoader
"""
import numpy as np
from torch.utils.data import Dataset

class SceneMIMLDataLoader(Dataset):
    """
    加载由MATLAB数据转换得到的Scene MIML数据划分
    """
    def __init__(self, data_path, split="train"):
        if split not in {"train", "test"}:
            raise ValueError("split 必须是 'train' 或 'test'")
        self.data_path = data_path
        self.split = split

        data = np.load(data_path, allow_pickle=True)
        self.instances = data[f"x_{split}"].astype(np.float32)
        self.labels = data[f"y_{split}"].astype(np.float32)
        self.label_names = [str(name) for name in data["label_names"]]

        if self.instances.ndim != 3:
            raise ValueError("实例数组必须具有 [bag, instance, feature] 三个维度")
        if self.labels.ndim != 2:
            raise ValueError("标签数组必须具有 [bag, label] 两个维度")
        if self.instances.shape[0] != self.labels.shape[0]:
            raise ValueError("实例数组与标签数组的 bag 数量不一致")
        if self.labels.shape[1] != len(self.label_names):
            raise ValueError("标签列数与 label_names 数量不一致")

        self.num_bags = self.instances.shape[0]
        self.num_instances = self.instances.shape[1]
        self.feature_dim = self.instances.shape[2]
        self.num_labels = self.labels.shape[1]

    #返回当前数据集或场景集合包含的样本数量。
    def __len__(self):
        return self.num_bags

    #读取并预处理指定索引对应的数据样本。
    def __getitem__(self, index):
        return self.instances[index], self.labels[index]

    #执行summary对应的数据处理或模型计算流程。
    def summary(self):
        return {
            "split": self.split,
            "num_bags": self.num_bags,
            "num_instances": self.num_instances,
            "feature_dim": self.feature_dim,
            "num_labels": self.num_labels,
            "label_names": self.label_names,
            "positive_per_label": self.labels.sum(axis=0).astype(int).tolist(),
            "avg_labels_per_bag": float(self.labels.sum(axis=1).mean()),
        }
