#!/bin/bash

# ==============================================
# Matrix Game 2.0 Apple Silicon - 模型自動下載腳本
# ==============================================

set -e  # 遇到錯誤立即退出

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日誌函數
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 檢查依賴
check_dependencies() {
    log_info "檢查系統依賴..."
    
    # 檢查 Python
    if ! command -v python &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            log_error "Python 未安裝。請先安裝 Python 3.8+"
            exit 1
        else
            PYTHON_CMD="python3"
        fi
    else
        PYTHON_CMD="python"
    fi
    
    # 檢查 pip
    if ! command -v pip &> /dev/null; then
        if ! command -v pip3 &> /dev/null; then
            log_error "pip 未安裝。請先安裝 pip"
            exit 1
        else
            PIP_CMD="pip3"
        fi
    else
        PIP_CMD="pip"
    fi
    
    log_success "Python: $($PYTHON_CMD --version)"
    log_success "pip: $($PIP_CMD --version)"
}

# 安裝 Hugging Face CLI
install_huggingface_cli() {
    log_info "檢查 Hugging Face CLI..."
    
    if ! command -v huggingface-cli &> /dev/null; then
        log_warning "Hugging Face CLI 未安裝，正在安裝..."
        $PIP_CMD install huggingface_hub[cli]
        log_success "Hugging Face CLI 安裝完成"
    else
        log_success "Hugging Face CLI 已安裝: $(huggingface-cli --version)"
    fi
}

# 檢查磁盤空間
check_disk_space() {
    log_info "檢查磁盤空間..."
    
    # 獲取可用空間 (GB)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        available_space=$(df -g . | awk 'NR==2 {print $4}')
    else
        # Linux
        available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    
    required_space=60
    
    if [ "$available_space" -lt "$required_space" ]; then
        log_error "磁盤空間不足！需要至少 ${required_space}GB，目前可用: ${available_space}GB"
        exit 1
    fi
    
    log_success "磁盤空間足夠: ${available_space}GB 可用"
}

# 設置鏡像 (可選，針對網絡較慢的地區)
setup_mirror() {
    if [ "${HF_USE_MIRROR:-}" = "true" ]; then
        log_info "使用 Hugging Face 鏡像站點..."
        export HF_ENDPOINT=https://hf-mirror.com
        log_success "已設置鏡像: $HF_ENDPOINT"
    fi
}

# 下載模型
download_models() {
    log_info "開始下載 Matrix Game 2.0 模型文件..."
    log_warning "注意：首次下載大約需要 50GB+ 空間和較長時間"
    
    # 創建模型目錄
    mkdir -p Matrix-Game-2.0
    
    # 下載模型 (使用符號鏈接以節省空間)
    huggingface-cli download Skywork/Matrix-Game-2.0 \
        --local-dir Matrix-Game-2.0 \
        --local-dir-use-symlinks \
        --resume-download
    
    log_success "模型下載完成！"
}

# 驗證下載
verify_models() {
    log_info "驗證模型文件..."
    
    # 關鍵模型文件列表
    models=(
        "Matrix-Game-2.0/Wan2.1_VAE.pth"
        "Matrix-Game-2.0/base_distilled_model/base_distill.safetensors"
        "Matrix-Game-2.0/base_model/diffusion_pytorch_model.safetensors"
        "Matrix-Game-2.0/gta_distilled_model/gta_keyboard2dim.safetensors"
        "Matrix-Game-2.0/templerun_distilled_model/templerun_7dim_onlykey.safetensors"
        "Matrix-Game-2.0/xlm-roberta-large/tokenizer.json"
    )
    
    all_present=true
    
    for model in "${models[@]}"; do
        if [ -e "$model" ]; then
            # 計算文件大小
            if [[ "$OSTYPE" == "darwin"* ]]; then
                size=$(du -h "$model" | cut -f1)
            else
                size=$(du -h "$model" | cut -f1)
            fi
            log_success "$model ($size)"
        else
            log_error "$model - 文件不存在"
            all_present=false
        fi
    done
    
    if [ "$all_present" = true ]; then
        log_success "所有關鍵模型文件驗證通過！"
        return 0
    else
        log_error "部分模型文件缺失，請重新運行下載"
        return 1
    fi
}

# 顯示使用說明
show_usage() {
    log_success "模型設置完成！您現在可以："
    echo
    echo "1. 運行基礎推理："
    echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference.py \\"
    echo "     --config_path configs/inference_yaml/inference_universal.yaml \\"
    echo "     --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
    echo "     --img_path demo_images/universal/0000.png \\"
    echo "     --output_folder outputs \\"
    echo "     --num_output_frames 5 \\"
    echo "     --seed 42 \\"
    echo "     --pretrained_model_path Matrix-Game-2.0"
    echo
    echo "2. 運行串流推理："
    echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 TORCH_COMPILE_DISABLE=1 python inference_streaming.py \\"
    echo "     --config_path configs/inference_yaml/inference_universal.yaml \\"
    echo "     --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \\"
    echo "     --pretrained_model_path Matrix-Game-2.0"
    echo
    echo "詳細信息請查看 MODELS_SETUP.md 文檔"
}

# 主函數
main() {
    echo "========================================"
    echo "Matrix Game 2.0 Apple Silicon 模型下載器"
    echo "========================================"
    echo
    
    # 檢查依賴
    check_dependencies
    
    # 安裝 Hugging Face CLI
    install_huggingface_cli
    
    # 檢查磁盤空間
    check_disk_space
    
    # 設置鏡像（如果需要）
    setup_mirror
    
    # 下載模型
    download_models
    
    # 驗證下載
    if verify_models; then
        show_usage
    else
        log_error "模型驗證失敗，請檢查網絡連接並重試"
        exit 1
    fi
    
    log_success "🎉 Matrix Game 2.0 模型設置完成！"
}

# 處理命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-mirror)
            export HF_USE_MIRROR=true
            shift
            ;;
        --help|-h)
            echo "使用方法: $0 [選項]"
            echo "選項:"
            echo "  --use-mirror    使用 Hugging Face 鏡像站點 (適用於網絡較慢地區)"
            echo "  --help, -h      顯示此幫助信息"
            exit 0
            ;;
        *)
            log_error "未知選項: $1"
            echo "使用 --help 查看幫助"
            exit 1
            ;;
    esac
done

# 運行主函數
main