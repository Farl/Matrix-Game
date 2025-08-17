#!/bin/bash
EXPERIMENT_NAME=$1
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "使用方法: $0 <experiment-name>"
    echo "例如: $0 exp-01-flash-attention"
    exit 1
fi

echo "重置實驗環境: $EXPERIMENT_NAME"
rm -rf "$EXPERIMENT_NAME"
cp -r base-clean "$EXPERIMENT_NAME"
echo "✅ 實驗環境 ${EXPERIMENT_NAME} 已重置完成"