"""
PointNet-MIML模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """
    对形状为[batch_size, channels, num_instances]的张量在通道维度上执行层标准化
    """
    #初始化ChannelLayerNorm的配置，在PyTorch中创建一个LayerNorm自定义模块。
    def __init__(self, num_channels):
        super(ChannelLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(num_channels)

    #执行ChannelLayerNorm的前向计算并返回结果。
    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


#执行make_channel_norm对应的数据处理或模型计算流程。
def make_channel_norm(norm_type, num_channels):
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "layer":
        return ChannelLayerNorm(num_channels)
    raise ValueError("norm_type must be one of: batch, layer")


class SetSelfAttentionBlock(nn.Module):
    """
    在包实例上的置换等变自注意力模块(暂时没启用)
    """
    #初始化SetSelfAttentionBlock
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.3):
        super(SetSelfAttentionBlock, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    #初始化SetSelfAttentionBlock的配置、网络层或数据索引。
    def forward(self, x):
        # x: [batch_size, num_instances, hidden_dim]
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MIMLInstanceEncoder(nn.Module):
    """
    无序MIML包的共享实例编码器
    """
    #初始化MIMLInstanceEncoder的配置
    def __init__(
        self,
        input_dim=15,
        hidden_dim=128,
        dropout=0.3,
        norm_type="layer",  #默认使用层归一化
        set_attention_layers=0,  #暂时没使用注意力，可更改
        set_attention_heads=4,
    ):
        super(MIMLInstanceEncoder, self).__init__()
        if set_attention_layers < 0:
            raise ValueError("set_attention_layers must be non-negative")

        #共享MLP实例编码器
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.norm1 = make_channel_norm(norm_type, 64)
        self.norm2 = make_channel_norm(norm_type, 128)
        self.norm3 = make_channel_norm(norm_type, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.set_attention = nn.ModuleList(
            [
                SetSelfAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=set_attention_heads,
                    dropout=dropout,
                )
                for _ in range(set_attention_layers)
            ]
        )

    #执行MIMLInstanceEncoder的前向计算并返回结果。
    def forward(self, x):
        # x: [batch_size, input_dim, num_instances]
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        for block in self.set_attention:
            x = block(x)
        return x


class get_model(nn.Module):
    """
    初始化get_model的配置
    处理经过感知机后的特征，进行池化得到logits
    """
    def __init__(
        self,
        num_labels=5,
        input_dim=15,
        hidden_dim=128,
        dropout=0.3,
        norm_type="layer",
        pooling="lse",  #默认是lse池化，温度可调
        lse_temperature=1.0,
        set_attention_layers=0,
        set_attention_heads=4,
    ):
        super(get_model, self).__init__()
        self.num_labels = num_labels
        self.pooling = pooling
        self.lse_temperature = lse_temperature
        self.feat = MIMLInstanceEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            norm_type=norm_type,
            set_attention_layers=set_attention_layers,
            set_attention_heads=set_attention_heads,
        )
        self.instance_classifier = nn.Linear(hidden_dim, num_labels)
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        elif pooling not in {"max", "lse", "mean", "noisy_or"}:
            raise ValueError("pooling must be one of: max, lse, mean, noisy_or, attention")

    #执行get_model的前向计算并返回结果。
    def forward(self, x):
        # Returns raw bag-level logits. Use sigmoid(logits) only for inference.
        instance_features = self.feat(x)
        instance_logits = self.instance_classifier(instance_features)
        attention_weights = None

        if self.pooling == "max":
            logits = instance_logits.max(dim=1)[0]
        elif self.pooling == "mean":
            logits = instance_logits.mean(dim=1)
        elif self.pooling == "lse":
            logits = self._logsumexp_pool(instance_logits)
        elif self.pooling == "noisy_or":
            logits = self._noisy_or_pool(instance_logits)
        else:
            attention_logits = self.attention(instance_features)
            attention_weights = F.softmax(attention_logits, dim=1)
            logits = (attention_weights * instance_logits).sum(dim=1)

        aux = {"instance_logits": instance_logits}
        if attention_weights is not None:
            aux["attention_weights"] = attention_weights
        return logits, aux

    #执行_logsumexp_pool对应的数据处理或模型计算流程。
    def _logsumexp_pool(self, instance_logits):
        temperature = max(float(self.lse_temperature), 1e-6)
        pooled = torch.logsumexp(instance_logits / temperature, dim=1) * temperature
        return pooled - temperature * torch.log(
            torch.tensor(
                instance_logits.size(1),
                dtype=instance_logits.dtype,
                device=instance_logits.device,
            )
        )

    #执行_noisy_or_pool对应的数据处理或模型计算流程。
    @staticmethod
    def _noisy_or_pool(instance_logits):
        probs = torch.sigmoid(instance_logits)  #压到01使成为概率
        bag_probs = 1.0 - torch.prod(1.0 - probs.clamp(max=1.0 - 1e-6), dim=1)
        bag_probs = bag_probs.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.logit(bag_probs)


class get_loss(nn.Module):
    """
    初始化损失函数，损失函数由两部分构成: 
    二元交叉熵损失和成对排序损失。二元交叉熵损失用于多标签分类任务，
    而成对排序损失用于确保正样本的预测分数高于负样本的预测分数。
    这两个损失的权重以及排序损失的参数可以在这里调整。
    """
    def __init__(self, ranking_loss_weight=0.1, ranking_margin=1.0):
        super(get_loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_margin = float(ranking_margin)
        if self.ranking_loss_weight < 0:
            raise ValueError("ranking_loss_weight must be non-negative")
        if self.ranking_margin < 0:
            raise ValueError("ranking_margin must be non-negative")

    #执行get_loss的前向计算并返回结果。
    def forward(self, pred, target):
        target = target.float()
        bce_loss = self.loss(pred, target)
        ranking_loss = self._pairwise_ranking_loss(pred, target)
        return bce_loss + self.ranking_loss_weight * ranking_loss

    #执行_pairwise_ranking_loss对应的数据处理或模型计算流程。
    def _pairwise_ranking_loss(self, pred, target):
        # For each example, every positive label should outrank every negative label.
        positive = target > 0.5
        negative = ~positive
        pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)  #掩码筛选
        pair_count = pair_mask.sum(dim=(1, 2))

        pair_loss = F.relu(
            self.ranking_margin
            - pred.unsqueeze(2)
            + pred.unsqueeze(1)
        )
        per_example_loss = (pair_loss * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1.0)  #防止除零
        valid_examples = pair_count > 0
        if valid_examples.any():
            return per_example_loss[valid_examples].mean()
        return pred.new_zeros(())
