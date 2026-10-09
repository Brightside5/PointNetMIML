# PointNetMIML

基于共享实例编码器的 PointNet-MIML Scene 多实例多标签学习项目。包含完整 Scene 特征数据、训练与评估代码、五随机种子实验及经典文献结果比较。

## 安装

在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

已验证环境：Python 3.13.3、PyTorch 2.11.0+cu128、NumPy 2.2.6、SciPy 1.16.0、tqdm 4.67.3。GPU 训练需要可用的 CUDA 版 PyTorch；也支持 CPU。

## 运行

```powershell
# 一次前向、反向与参数更新检查
python smoke_test_miml.py --gpu 0

# 五随机种子训练
python run_miml_multiseed.py --seeds '1,2,3,4,5' --run_name scene_5seeds -- --gpu 0 --epoch 200 --batch_size 64 --early_stop_patience 30

# 比较本次结果与经典文献值
python compare_scene_results.py --run log_miml/scene/scene_5seeds/multi_seed_summary.json --output results/scene_5seeds_comparison.json
```

无 GPU 时，将训练参数中的 `--gpu 0` 替换为 `--use_cpu`。重复实验请使用新的 `--run_name`，避免覆盖之前的结果。

训练生成的检查点与汇总保存在 `log_miml/scene/<run_name>/`。每次训练最多 200 epoch，验证集 average precision 连续 30 epoch 不改善时提前停止。

## 数据

`data/scene/` 包含官方 `miml data.mat` 及可直接读取的 `scene_full_pointnet.npz`，共覆盖全部 2000 个 Scene bag。每个 bag 有 9 个 15 维 SBN 实例，五个标签为 desert、mountains、sea、sunset、trees。

| NPZ 数组 | 形状 |
|---|---|
| `x_train` / `y_train` | `[1500, 9, 15]` / `[1500, 5]` |
| `x_test` / `y_test` | `[500, 9, 15]` / `[500, 5]` |

MAT → NPZ 仅转换格式、标签编码与数据划分，未重新提取图像特征。训练不依赖原始图片或仓库外的数据文件。需要重建 NPZ 时运行：

```powershell
python tools/convert_scene_full_mat_to_npz.py --seed 42 --train_size 1500
```

官方数据出处和引用见 [SOURCE.txt](data/scene/SOURCE.txt)。

## 默认设置与指标

默认使用 LayerNorm、LSE 池化（温度 1）、128 维实例表示、dropout=0.3，以及 BCE + 0.1×成对排序损失。集合自注意力默认关闭；自注意力和其他池化选项仍保留，代码与提交前版本一致。

固定外层划分为 1500/500，划分种子为 42。每个实验种子从 1500 个训练 bag 内取 20% 作为验证集；检查点按验证 average precision 选择，阈值按验证 Hamming loss 选择，最后评估测试集。

报告 Hamming loss、One-error、Coverage、Ranking loss 和逐样本 Average precision。五种子汇总采用均值与样本标准差。

`baselines/scene_literature.json` 保存经典论文的报告值，比较脚本不运行这些经典模型。文献的随机划分协议与本地固定测试集协议有差异。作者公开稿中 MIMLSVM 的 Hamming loss 缺少小数点，本文件记录的 0.189 是注明依据的排版校正。

## 许可与引用

代码保留原有 [MIT License](LICENSE)。Scene 数据使用条件独立于代码许可，以 [SOURCE.txt](data/scene/SOURCE.txt) 和官方数据页面为准。

Z.-H. Zhou and M.-L. Zhang. *Multi-instance multi-label learning with application to scene classification*. Advances in Neural Information Processing Systems 19, 2007, pp. 1609–1616.
