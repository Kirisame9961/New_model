# New_model

一个从零开始的轻量级分子势能模型（MLP）项目骨架，目标是完成：
- 结构能量预测
- 力预测
- 简化 MD（NVE）验证

## 当前完整流程（已补齐）

1. **MPtrj JSON -> NPZ 转换**：`scripts/convert_mptrj_json.py`
2. **数据准备与切分**：`scripts/prepare_dataset.py`
3. **模型训练**：`scripts/train_mlp.py`
4. **测试集评估**：`scripts/eval_mlp.py`
5. **单构型能量/力推理**：`scripts/predict_mlp.py`
6. **简化 MD(NVE) rollout**：`scripts/run_md_nve.py`

## 数据来源（MPtrj）

- figshare 页面（你提供）：  
  `https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842?file=41619375`
- 对应下载直链：  
  `https://figshare.com/ndownloader/files/41619375`
- 下载文件名示例：`MPtrj_2022.9_full.json`

> 该文件是 JSON，不是训练直接可用的 NPZ，请先执行转换脚本。

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) JSON 转换为 NPZ

```bash
python scripts/convert_mptrj_json.py \
  --input data/raw/MPtrj_2022.9_full.json \
  --output data/raw/mptrj_full.npz \
  --group-key material_id \
  --max-samples 150000 \
  --seed 42
```

### 3) 标准化准备与切分（默认 MPtrj 10%）

```bash
python scripts/prepare_dataset.py --dataset-name mptrj_10pct --seed 42
```

### 4) 训练

```bash
python scripts/train_mlp.py --config configs/train_mlp.yaml
```

训练脚本已支持：
- 随机种子固定
- 梯度裁剪
- ReduceLROnPlateau 调度
- early stopping
- resume 断点续训

### 5) 评估

```bash
python scripts/eval_mlp.py --config configs/eval_mlp.yaml
```

### 6) 单样本预测（能量+力）

```bash
python scripts/predict_mlp.py \
  --checkpoint outputs/mlp_baseline/best.pt \
  --npz data/processed/mptrj_10pct_150000.npz \
  --split test \
  --index 0
```

### 7) 简化 MD NVE 验证

```bash
python scripts/run_md_nve.py --config configs/md_nve.yaml
```

输出：`outputs/md_nve/rollout.npz`，并打印总能量相对漂移（`relative_energy_drift`）。

## 关键配置

- 数据准备：`configs/data_prep_mptrj.yaml`
- 训练：`configs/train_mlp.yaml`
- 评估：`configs/eval_mlp.yaml`
- MD：`configs/md_nve.yaml`
- 数据来源说明：`docs/mptrj_source.md`
