#!/bin/bash
# macOS OpenCV GUI 修復腳本
echo "🔧 修復 macOS OpenCV GUI 支援..."

# 卸載可能有問題的版本
echo "1️⃣  清理現有 OpenCV 安裝..."
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# 重新安裝完整版本
echo "2️⃣  安裝 OpenCV 完整版本..."
pip install opencv-contrib-python

# 安裝 matplotlib 作為備選
echo "3️⃣  安裝 matplotlib 備選方案..."
pip install matplotlib

echo "✅ OpenCV GUI 修復完成！"
echo "💡 現在重新運行你的程序，應該可以看到預覽窗口了"
echo ""
echo "⚠️  記住: max_num_output_frames 必須是3的倍數!"
echo "   正確示例: 9, 12, 15, 18, 21, 90, 180, 360"
echo "   運行示例: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python inference_streaming_apple_silicon.py --test_mode --show_preview --max_num_output_frames 12"

