import numpy as np


#稳定地把 logits 转换为标签概率。
def sigmoid(x):
    # 裁剪只影响已饱和的极端分数，可避免 exp 产生无意义的溢出警告。
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


#计算所有样本标签位上的汉明损失。
def hamming_loss(y_true, y_pred):
    return float(np.mean(y_true.astype(bool) != y_pred.astype(bool)))


#基于全局真阳性、假阳性和假阴性计算 micro-F1。
def micro_f1(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    fn = np.logical_and(y_true, ~y_pred).sum()
    denom = 2 * tp + fp + fn
    return float(0.0 if denom == 0 else (2 * tp) / denom)


#计算逐样本平均精度并在样本间取平均。
def example_average_precision(y_true, y_score):
    aps = []
    for labels, scores in zip(y_true.astype(bool), y_score):
        positives = labels.sum()
        if positives == 0:
            continue
        order = np.argsort(-scores)
        hits = labels[order]
        precision_at_k = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append(float((precision_at_k * hits).sum() / positives))
    return float(np.mean(aps)) if aps else 0.0


#计算单个二分类标签的平均精度。
def binary_average_precision(y_true, y_score):
    y_true = y_true.astype(bool)
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    order = np.argsort(-y_score)
    hits = y_true[order]
    precision_at_k = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision_at_k * hits).sum() / positives)


#计算最高分标签不属于真实标签集的样本比例。
def one_error(y_true, y_score):
    top = np.argmax(y_score, axis=1)
    return float(np.mean(y_true[np.arange(y_true.shape[0]), top] != 1))


#计算正标签得分不高于负标签得分的标签对比例。
def ranking_loss(y_true, y_score):
    losses = []
    y_true = y_true.astype(bool)
    for labels, scores in zip(y_true, y_score):
        pos_scores = scores[labels]
        neg_scores = scores[~labels]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        violations = (neg_scores[:, None] >= pos_scores[None, :]).sum()
        losses.append(float(violations / (len(pos_scores) * len(neg_scores))))
    return float(np.mean(losses)) if losses else 0.0


#计算覆盖全部真实标签所需的平均最大排名。
def coverage(y_true, y_score):
    values = []
    y_true = y_true.astype(bool)
    for labels, scores in zip(y_true, y_score):
        if labels.sum() == 0:
            continue
        order = np.argsort(-scores)
        positive_ranks = np.where(labels[order])[0]
        values.append(float(positive_ranks.max()))
    return float(np.mean(values)) if values else 0.0


#由 logits 和真实标签统一计算全部 MIML 指标。
def compute_metrics(logits, y_true, threshold=0.5):
    probs = sigmoid(logits)
    pred = (probs >= threshold).astype(np.float32)
    per_label = per_label_metrics(y_true, probs, pred)
    return {
        "hamming_loss": hamming_loss(y_true, pred),
        "micro_f1": micro_f1(y_true, pred),
        "macro_f1": float(np.mean([item["f1"] for item in per_label])),
        "macro_label_ap": float(np.mean([item["average_precision"] for item in per_label])),
        "average_precision": example_average_precision(y_true, probs),
        "one_error": one_error(y_true, probs),
        "ranking_loss": ranking_loss(y_true, probs),
        "coverage": coverage(y_true, probs),
    }


#逐标签统计支持数、混淆计数、F1 与平均精度。
def per_label_metrics(y_true, y_score, y_pred=None, label_names=None, threshold=0.5):
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(np.float32)
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]

    rows = []
    for i, name in enumerate(label_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        tp = int(np.logical_and(true_i, pred_i).sum())
        fp = int(np.logical_and(~true_i, pred_i).sum())
        fn = int(np.logical_and(true_i, ~pred_i).sum())
        tn = int(np.logical_and(~true_i, ~pred_i).sum())
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
        rows.append(
            {
                "label": str(name),
                "support": int(true_i.sum()),
                "predicted_positive": int(pred_i.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "average_precision": binary_average_precision(true_i, y_score[:, i]),
            }
        )
    return rows
