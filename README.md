# New_model

一个从零开始的轻量级分子势能模型（MLP）项目骨架。

## 数据集策略更新（针对 OMA 对比场景）

你提到后续希望与 MACE、AlphaNet 等模型在 **OMA（Omat + MPtrj + Alex）相关场景**中展示结果。  
在工期限制（约 1 个月）下，本项目建议采用：

1. **主方案：OMA 标准答案子集（推荐）**
   - 从你们已有的 OMA 数据导出一个统一 `.npz` 文件（至少包含 `z/R/E/F`），
   - 通过本项目脚本抽样到约 100k，做 train/val/test 划分。
   - 好处：实验叙事与目标基线一致，后续对比解释最自然。

2. **备选方案：rMD17 100k 快速基线**
   - 用于先打通端到端流程（数据->训练->评估->MD）。
   - 若 OMA 子集准备稍慢，可先用它验证 pipeline，再切换到 OMA 子集。

## 当前已完成

第 1 步：**通用数据准备脚本**（支持 rMD17 与 OMA 风格 NPZ 子集）。

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置数据

编辑 `configs/data_prep_rmd17.yaml`：
- 默认 profile 是 `oma_mptrj_100k`。
- 请把 `datasets.oma_mptrj_100k.source_url` 填成可下载的 NPZ 地址（内网或公开地址均可）。

NPZ 至少需要键：
- `z`: 原子序号，形状通常为 `[n_atoms]`
- `R`: 坐标，`[n_samples, n_atoms, 3]`
- `E`: 能量，`[n_samples]` 或 `[n_samples, 1]`
- `F`: 力，`[n_samples, n_atoms, 3]`

可选键：`cell`, `pbc`, `stress`, `virial`。

### 3) 处理 OMA 子集（推荐）

```bash
python scripts/prepare_rmd17.py \
  --dataset-name oma_mptrj_100k \
  --max-samples 100000 \
  --seed 42
```

### 4) 处理 rMD17 快速基线（备选）

```bash
python scripts/prepare_rmd17.py \
  --dataset-name rmd17_aspirin_100k \
  --molecule aspirin \
  --max-samples 100000 \
  --seed 42
```

输出：
- 处理后数据：`data/processed/<profile>_<N>.npz`
- 摘要统计：`data/processed/<profile>_<N>_summary.json`

## 项目结构

```text
.
├── configs/
│   └── data_prep_rmd17.yaml
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   └── prepare_rmd17.py
├── requirements.txt
└── README.md
```

## 下一步建议（我可以继续帮你做）

1. 建立 MLP baseline（energy + force 联合损失）。
2. 增加训练脚本（早停、学习率调度、日志）。
3. 加评估脚本（energy/force MAE + 简化 MD 稳定性）。
4. 固化对比协议：MACE / AlphaNet / 你的模型在同一 100k 子集上比较。


## 模型开发进度（Step 2 - 第一部分）

已新增一个可训练的**轻量级 MLP 势能模型骨架**（先实现核心前向，不一次写完全部训练系统）：

- 代码位置：`src/new_model/models/mlp_potential.py`
- 核心结构：
  1. `Embedding(z)`：把原子序号映射到可学习向量；
  2. `RadialBasisLayer(d_ij)`：把两两距离做高斯径向基展开；
  3. 邻域聚合：对每个原子汇总邻域径向特征；
  4. `AtomMLP`：输出每个原子的能量贡献，最后求和得到总能量；
  5. 力通过自动微分计算：`F = -dE/dR`。

本阶段目标是把“模型主干 + 能量/力接口”先稳定下来，下一步再补：
- 训练脚本（dataloader、loss、optimizer、日志）
- 评估脚本（energy/force MAE、简单 MD 稳定性）
- 与 MACE/AlphaNet 的统一对比入口

新增配置文件：`configs/model_mlp_baseline.yaml`（先放模型与损失权重默认值）。


## 模型开发进度（Step 2 - 第二部分）

已继续完成训练最小闭环（仍保持轻量）：

- `src/new_model/data/npz_dataset.py`
  - 新增 `MolecularNpzDataset`，可读取 `prepare_rmd17.py` 导出的 processed `.npz`。
  - 支持按 `train_idx/val_idx/test_idx` 切分加载。
- `scripts/train_mlp.py`
  - 新增训练脚本：DataLoader、energy/force 双损失、AdamW、每轮验证、best/last checkpoint 保存。
  - 训练日志逐 epoch 打印 JSON，并落盘 `history.json`。
- `configs/train_mlp.yaml`
  - 新增训练配置模板（模型超参、数据路径、优化器与loss权重）。

### 训练命令（当前版本）

```bash
python scripts/train_mlp.py --config configs/train_mlp.yaml
```

> 注意：请先把 `configs/train_mlp.yaml` 中的 `data.processed_npz` 改成你的真实 processed 数据路径。

下一步可继续补：
1. 学习率调度器与早停；
2. test 集评估脚本（MAE/RMSE）；
3. 更稳健的邻域构造与归一化策略；
4. 简化 MD rollout 验证脚本。
