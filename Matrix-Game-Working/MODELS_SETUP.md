# Matrix Game 2.0 模型設置指南

## 📋 概述

Matrix Game 2.0 需要下載大型預訓練模型文件才能正常運行。這些模型文件總大小約 **50GB+**，因此不包含在 Git 儲存庫中。

## 🚀 快速設置

### 方法 1: 使用自動下載腳本 (推薦)

```bash
# 運行自動下載腳本
bash download_models.sh
```

### 方法 2: 手動下載

```bash
# 使用 Hugging Face CLI 下載所有模型
huggingface-cli download Skywork/Matrix-Game-2.0 \
  --local-dir Matrix-Game-2.0 \
  --local-dir-use-symlinks
```

## 📂 模型文件結構

下載完成後，您應該看到以下目錄結構：

```
Matrix-Game-2.0/
├── README.md
├── architecture.png
├── config.json
├── Wan2.1_VAE.pth                    # VAE 模型 (~8GB)
├── models_clip_*.pth                 # CLIP 模型 (~2GB)
├── base_model/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors  # 基礎模型 (~20GB)
├── base_distilled_model/
│   ├── config.json
│   └── base_distill.safetensors      # 精煉基礎模型 (~10GB)
├── gta_distilled_model/
│   ├── config.json
│   └── gta_keyboard2dim.safetensors  # GTA 專用模型 (~5GB)
├── templerun_distilled_model/
│   ├── config.json
│   └── templerun_7dim_onlykey.safetensors  # Temple Run 專用模型 (~5GB)
└── xlm-roberta-large/               # 語言模型 (~2GB)
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── sentencepiece.bpe.model
```

## 🔍 驗證下載

檢查模型是否正確下載：

```bash
# 檢查模型文件是否存在
python -c "
import os
models = [
    'Matrix-Game-2.0/Wan2.1_VAE.pth',
    'Matrix-Game-2.0/base_distilled_model/base_distill.safetensors',
    'Matrix-Game-2.0/base_model/diffusion_pytorch_model.safetensors'
]
for model in models:
    if os.path.exists(model):
        size = os.path.getsize(model) / (1024**3)
        print(f'✅ {model} ({size:.1f}GB)')
    else:
        print(f'❌ {model} - 文件不存在')
"
```

## 🛠️ 故障排除

### 問題 1: 下載中斷
如果下載過程中斷，可以重新運行下載命令，Hugging Face CLI 會自動恢復下載。

### 問題 2: 磁盤空間不足
確保至少有 **60GB** 的可用磁盤空間。

### 問題 3: 網絡連接問題
```bash
# 使用鏡像站點 (中國用戶)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0 --local-dir-use-symlinks
```

### 問題 4: 符號鏈接問題
如果您在 Windows 或某些文件系統上遇到符號鏈接問題：
```bash
# 不使用符號鏈接
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
```

## 🔒 安全性

所有模型文件都來自官方的 [Skywork/Matrix-Game-2.0](https://huggingface.co/Skywork/Matrix-Game-2.0) 儲存庫，並經過 SHA256 驗證。

## 📈 存儲優化

### 使用符號鏈接 (推薦)
預設情況下，Hugging Face CLI 會使用符號鏈接，這樣：
- 多個項目可以共享同一套模型
- 節省磁盤空間
- 自動管理緩存

### 模型緩存位置
模型實際存儲在：
- **macOS/Linux**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

## 🚦 下一步

模型下載完成後，您可以：

1. **運行基礎推理**:
   ```bash
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference.py \
     --config_path configs/inference_yaml/inference_universal.yaml \
     --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
     --img_path demo_images/universal/0000.png \
     --output_folder outputs \
     --num_output_frames 5 \
     --seed 42 \
     --pretrained_model_path Matrix-Game-2.0
   ```

2. **運行串流推理**:
   ```bash
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference_streaming.py \
     --config_path configs/inference_yaml/inference_universal.yaml \
     --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
     --pretrained_model_path Matrix-Game-2.0
   ```

## 📞 支援

如果遇到任何問題，請檢查：
1. [Apple Silicon 適配報告](FINAL_VERIFICATION_REPORT.md)
2. [變更說明](APPLE_SILICON_CHANGES.md)
3. [原始 README](README.md)

---

**注意**: 首次下載需要較長時間，請耐心等待。建議在穩定的網絡環境下進行下載。