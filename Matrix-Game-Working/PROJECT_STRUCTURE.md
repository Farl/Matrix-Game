# Matrix Game 2.0 Apple Silicon - 項目文件結構說明

## 📋 概述

這個文檔說明了項目中各個文件和文件夾的作用，以及哪些應該包含在 Git 版本控制中。

## 🗂️ 項目結構

```
Matrix-Game-Working/
├── 📄 核心代碼文件 (✅ 包含在 Git 中)
│   ├── inference.py                    # 主推理腳本 (Apple Silicon 適配)
│   ├── inference_streaming.py          # 串流推理腳本 (Apple Silicon 適配)
│   ├── setup.py                        # 安裝腳本
│   ├── requirements.txt                # Python 依賴
│   └── pipeline/                       # 推理管道
│       └── causal_inference.py
│
├── 📄 核心模組 (✅ 包含在 Git 中)
│   └── wan/                            # 核心模型模組
│       ├── modules/
│       │   ├── model.py                # 主模型 (MPS 精度適配)
│       │   ├── attention.py            # 注意力機制 (Flash Attention 回退)
│       │   ├── causal_model.py         # 因果模型 (複數精度適配)
│       │   └── ...
│       └── vae/                        # VAE 模組
│
├── 📄 配置文件 (✅ 包含在 Git 中)
│   └── configs/
│       ├── inference_yaml/             # 推理配置
│       └── distilled_model/            # 模型配置
│
├── 📄 示例文件 (✅ 包含在 Git 中)
│   └── demo_images/                    # 示例圖片
│       ├── universal/
│       ├── gta_drive/
│       └── temple_run/
│
├── 📄 文檔文件 (✅ 包含在 Git 中)
│   ├── README.md                       # 主文檔 (已更新 Apple Silicon 支援)
│   ├── README_Apple_Silicon.md         # Apple Silicon 特定說明
│   ├── FINAL_VERIFICATION_REPORT.md    # 最終驗證報告
│   ├── APPLE_SILICON_CHANGES.md        # 技術變更記錄
│   ├── MODELS_SETUP.md                 # 模型設置指南 (新)
│   └── PROJECT_STRUCTURE.md            # 本文檔 (新)
│
├── 📄 自動化腳本 (✅ 包含在 Git 中)
│   ├── download_models.sh              # 模型自動下載腳本 (新)
│   └── .gitignore                      # Git 忽略規則 (新)
│
├── 📁 模型文件 (❌ 不包含在 Git 中)
│   └── Matrix-Game-2.0/                # 大型模型文件文件夾
│       ├── Wan2.1_VAE.pth              # VAE 模型 (~8GB)
│       ├── base_distilled_model/
│       │   └── base_distill.safetensors # 基礎模型 (~10GB)
│       ├── base_model/
│       │   └── diffusion_pytorch_model.safetensors # (~20GB)
│       ├── gta_distilled_model/
│       │   └── gta_keyboard2dim.safetensors # GTA 模型 (~5GB)
│       ├── templerun_distilled_model/
│       │   └── templerun_7dim_onlykey.safetensors # Temple Run 模型 (~5GB)
│       ├── models_clip_*.pth           # CLIP 模型 (~2GB)
│       └── xlm-roberta-large/          # 語言模型文件夾 (~2GB)
│
└── 📁 輸出文件 (❌ 不包含在 Git 中)
    ├── test_outputs/                   # 測試輸出 (生成的影片)
    ├── test_outputs_streaming/         # 串流測試輸出
    └── outputs/                        # 用戶輸出文件夾
```

## 🔄 Git 版本控制策略

### ✅ **包含在 Git 中的文件**
- **源代碼**: 所有 `.py` 文件和模組
- **配置文件**: YAML 配置和 JSON 設置
- **示例資源**: 小型示例圖片 (< 1MB)
- **文檔**: README、說明文檔、報告
- **腳本**: 安裝和自動化腳本
- **元數據**: requirements.txt、setup.py 等

### ❌ **不包含在 Git 中的文件** 
- **大型模型文件**: `.safetensors`, `.pth`, `.bin` (由 `.gitignore` 排除)
- **輸出文件**: 生成的影片和中間結果
- **緩存文件**: `__pycache__/`, `.cache/`
- **臨時文件**: 日誌、調試輸出
- **環境相關**: 虛擬環境文件夾

## 🔗 模型文件管理

### 符號鏈接機制
```bash
Matrix-Game-2.0/base_distilled_model/base_distill.safetensors 
└── 符號鏈接指向 → ~/.cache/huggingface/hub/models--Skywork--Matrix-Game-2.0/blobs/xxx
```

**優點**:
- 多個項目共享同一套模型
- 節省磁盤空間
- 自動緩存管理
- Git 友好 (符號鏈接很小)

### 自動下載流程
1. **運行腳本**: `bash download_models.sh`
2. **Hugging Face CLI** 下載模型到緩存
3. **創建符號鏈接** 指向緩存中的文件
4. **驗證完整性** 確保所有模型文件就位

## 📊 磁盤空間使用

| 類別 | 大小 | 說明 |
|------|------|------|
| **源代碼** | ~10 MB | 包含在 Git 中 |
| **配置和文檔** | ~5 MB | 包含在 Git 中 |
| **示例圖片** | ~50 MB | 包含在 Git 中 |
| **模型文件** | ~50 GB | 不包含在 Git (符號鏈接) |
| **輸出文件** | 變動 | 不包含在 Git |

## 🚀 新用戶設置流程

1. **Clone 儲存庫**:
   ```bash
   git clone [your-repo-url] Matrix-Game-Working
   cd Matrix-Game-Working
   ```

2. **安裝依賴**:
   ```bash
   pip install -r requirements.txt
   python setup.py develop
   ```

3. **下載模型**:
   ```bash
   bash download_models.sh
   ```

4. **驗證設置**:
   ```bash
   # 檢查模型是否就位
   ls -la Matrix-Game-2.0/
   ```

5. **運行推理**:
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

## 🔧 維護注意事項

### 開發者
- **提交代碼前**: 確認沒有意外包含大文件
- **測試輸出**: 清理 `test_outputs/` 文件夾
- **依賴更新**: 及時更新 `requirements.txt`

### 用戶
- **定期清理**: 刪除不需要的輸出文件
- **空間監控**: 關注 Hugging Face 緩存大小
- **更新模型**: 重新運行 `download_models.sh` 獲取最新版本

---

**總結**: 這個項目結構設計確保了代碼和配置的完整版本控制，同時避免了大文件帶來的問題。通過自動化腳本和符號鏈接機制，提供了流暢的用戶體驗。