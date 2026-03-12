# MPtrj 数据来源与获取说明

本项目默认使用 **MPtrj 10% 子集（约 150k）** 作为首选训练数据。

## 1. 数据来源建议

已确认可用的 figshare 下载直链（文件ID=41619375）：
- `https://figshare.com/ndownloader/files/41619375`
- 文件名示例：`MPtrj_2022.9_full.json`


由于不同团队对 OMA/MPtrj 数据的托管方式不同，建议按以下优先级获取：

1. **团队内部镜像（推荐）**
   - 优先使用你们内部已校验版本（对象存储/文件服务器）。
   - 优点：稳定、可审计、版本一致，便于复现实验。

2. **项目统一公开地址（如你们组维护的发布链接）**
   - 若存在公开下载链接，建议固定到具体版本并记录 checksum。

3. **上游原始数据重导出（兜底）**
   - 从上游来源导出为统一 NPZ 格式后再接入本项目。

## 2. 本项目要求的 NPZ 字段

最少包含：
- `z`: `[n_atoms]`
- `R`: `[n_samples, n_atoms, 3]`
- `E`: `[n_samples]` 或 `[n_samples, 1]`
- `F`: `[n_samples, n_atoms, 3]`

可选字段：
- `cell`, `pbc`, `stress`, `virial`

## 3. 接入步骤

### 步骤 A：填入下载地址

编辑 `configs/data_prep_mptrj.yaml` 中：

```yaml
datasets:
  mptrj_10pct:
    source_url: "<你的MPtrj npz下载地址>"
```

### 步骤 B：若源文件是 JSON，先转换为 NPZ

```bash
python scripts/convert_mptrj_json.py \
  --input data/raw/MPtrj_2022.9_full.json \
  --output data/raw/mptrj_full.npz \
  --group-key material_id \
  --max-samples 150000 \
  --seed 42
```

### 步骤 C：执行数据准备（默认 10%）

```bash
python scripts/prepare_dataset.py --dataset-name mptrj_10pct --seed 42
```

### 步骤 D：切换为全量（可选）

```bash
python scripts/prepare_dataset.py --dataset-name mptrj_full --use-all --seed 42
```

## 4. 建议补充的可复现信息

建议把以下信息写入你们的实验记录：
- 数据下载 URL（或内部路径）
- 文件 checksum（如 SHA256）
- 拉取日期
- 数据版本号/快照号
- 本项目 commit hash

这样后续与 MACE、AlphaNet 对比时，数据口径能完全一致。
