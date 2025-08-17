# Matrix Game 2.0 Apple Silicon 適配計劃 (最終版)
*結合實作性和技術分析的平衡方案*

## 🎯 項目目標
從技術分析出發，實作 Matrix Game 2.0 的 Apple Silicon 適配版本，重視實測和可恢復性

## 📁 專案結構保護策略
**原則：每次實驗都要可以快速回滾，不破壞專案結構**

```
matrix-game-2.0/
├── Matrix-Game-Original/          # 原始代碼（只讀，永不修改）
├── Matrix-Game/                   # 現有珍貴文件（保護）
├── experiments/                   # 實驗目錄（可隨意刪除重建）
│   ├── exp-01-flash-attention/
│   ├── exp-02-mps-datatype/
│   └── exp-03-device-management/
├── Matrix-Game-Working/           # 最終工作版本
└── test-results/                  # 測試結果記錄
```

## 🧪 實驗驅動的開發流程

### 階段 1：建立實驗環境（保護現有結構）

#### 步驟 1.1：建立安全的實驗空間
```bash
# 建立實驗目錄結構
mkdir -p experiments test-results

# 建立乾淨的工作基礎（從原始碼複製）
cp -r Matrix-Game-Original/Matrix-Game-2 ./experiments/base-clean
cd experiments/base-clean

# 保護性複製珍貴文檔
cp ../../Matrix-Game/Matrix-Game-2/README_Apple_Silicon.md ./

# 建立快速重置腳本
cat > ../reset-experiment.sh << 'EOF'
#!/bin/bash
EXPERIMENT_NAME=$1
rm -rf experiments/${EXPERIMENT_NAME}
cp -r experiments/base-clean experiments/${EXPERIMENT_NAME}
echo "實驗環境 ${EXPERIMENT_NAME} 已重置"
EOF
chmod +x ../reset-experiment.sh
```

#### 步驟 1.2：建立最小測試用例
**目的**：在修改任何代碼前，先有一個可以快速驗證的基準測試

```bash
# 建立最小測試腳本
cat > test-minimal.py << 'EOF'
#!/usr/bin/env python3
"""最小化測試：檢查核心模組是否可以導入和初始化"""
import torch
import sys
import traceback

def test_import():
    """測試基本導入"""
    try:
        from wan.modules.action_module import ActionModule
        from wan.modules.model import WanModel
        from pipeline.causal_inference import CausalInferencePipeline
        print("✅ 基本導入測試通過")
        return True
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        traceback.print_exc()
        return False

def test_device_detection():
    """測試設備檢測"""
    print(f"MPS 可用: {torch.backends.mps.is_available()}")
    print(f"MPS 已建置: {torch.backends.mps.is_built()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"將使用設備: {device}")
    
    # 測試基本張量操作
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)
        print("✅ 基本張量操作通過")
        return True
    except Exception as e:
        print(f"❌ 張量操作失敗: {e}")
        return False

if __name__ == "__main__":
    print("開始最小化測試...")
    success = test_import() and test_device_detection()
    sys.exit(0 if success else 1)
EOF
```

### 階段 2：逐個技術點實驗（每個實驗獨立可回滾）

#### 實驗 2.1：Flash Attention 替代方案測試

```bash
# 建立實驗環境
../reset-experiment.sh exp-01-flash-attention
cd ../exp-01-flash-attention

# 建立 A/B 測試腳本
cat > test-attention-methods.py << 'EOF'
import torch
import time
from typing import Callable

def test_attention_method(attention_func: Callable, name: str, q, k, v):
    """測試注意力方法的效能和正確性"""
    print(f"\n測試 {name}:")
    
    try:
        start_time = time.time()
        result = attention_func(q, k, v)
        end_time = time.time()
        
        print(f"  ✅ 執行成功")
        print(f"  ⏱️  執行時間: {end_time - start_time:.4f}s")
        print(f"  📊 輸出形狀: {result.shape}")
        print(f"  📈 記憶體使用: {torch.mps.current_allocated_memory() / 1024**2:.1f}MB")
        
        return result, True
    except Exception as e:
        print(f"  ❌ 執行失敗: {e}")
        return None, False

# 定義不同的注意力實現
def pytorch_native_attention(q, k, v):
    """PyTorch 2.0 原生實現"""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def manual_attention(q, k, v):
    """手動實現（README 方案）"""
    scale = q.size(-1) ** -0.5
    attention = torch.matmul(q * scale, k.transpose(-2, -1))
    attention = torch.softmax(attention, dim=-1)
    return torch.matmul(attention, v)

def run_attention_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 測試數據
    batch_size, seq_len, dim = 2, 128, 64
    q = torch.randn(batch_size, seq_len, dim, device=device)
    k = torch.randn(batch_size, seq_len, dim, device=device)  
    v = torch.randn(batch_size, seq_len, dim, device=device)
    
    results = {}
    
    # 測試所有方法
    methods = [
        (pytorch_native_attention, "PyTorch 原生"),
        (manual_attention, "手動實現")
    ]
    
    for method, name in methods:
        result, success = test_attention_method(method, name, q, k, v)
        if success:
            results[name] = result
    
    # 比較結果一致性
    if len(results) > 1:
        print("\n結果一致性檢查:")
        base_result = list(results.values())[0]
        for name, result in results.items():
            diff = torch.abs(base_result - result).mean().item()
            print(f"  {name}: 平均差異 {diff:.6f}")
    
    return results

if __name__ == "__main__":
    run_attention_comparison()
EOF
```

**實驗執行和記錄**：
```bash
# 執行實驗
python test-attention-methods.py > ../../test-results/exp-01-attention-$(date +%Y%m%d_%H%M%S).log 2>&1

# 根據結果決定最佳方案，然後修改代碼
# 如果實驗失敗，直接重置環境：../reset-experiment.sh exp-01-flash-attention
```

#### 實驗 2.2：MPS 數據類型最佳化測試

```bash
../reset-experiment.sh exp-02-mps-datatype  
cd ../exp-02-mps-datatype

cat > test-precision-impact.py << 'EOF'
import torch
import numpy as np

def test_sinusoidal_precision():
    """測試 sinusoidal embedding 的精度影響"""
    def sinusoidal_float64(dim, position):
        position = position.type(torch.float64)
        # ... 實現

    def sinusoidal_float32(dim, position):
        position = position.type(torch.float32) 
        # ... 實現

    def sinusoidal_smart(dim, position):
        """智能精度：根據設備選擇"""
        if position.device.type == 'mps':
            position = position.type(torch.float32)
        else:
            position = position.type(torch.float64)
        # ... 實現

    # A/B/C 測試所有方案
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test_positions = torch.arange(100, device=device)
    
    results = {}
    for name, func in [("Float64", sinusoidal_float64), 
                       ("Float32", sinusoidal_float32),
                       ("Smart", sinusoidal_smart)]:
        try:
            start_time = time.time()
            result = func(128, test_positions)
            end_time = time.time()
            results[name] = {
                'result': result,
                'time': end_time - start_time,
                'success': True
            }
            print(f"✅ {name}: {end_time - start_time:.4f}s")
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"❌ {name}: {e}")
    
    # 精度比較
    if results['Float64']['success'] and results['Float32']['success']:
        diff = torch.abs(results['Float64']['result'] - results['Float32']['result']).mean()
        print(f"Float32 vs Float64 精度差異: {diff.item():.8f}")
    
    return results
EOF
```

#### 實驗 2.3：整合測試和基準建立

```bash
../reset-experiment.sh exp-03-integration
cd ../exp-03-integration

# 應用所有通過實驗的修改
# 建立完整的基準測試

cat > test-full-pipeline.py << 'EOF'
#!/usr/bin/env python3
"""完整管道測試：模擬真實使用情境"""

def test_minimal_inference():
    """最小推理測試：不需要大模型，只測試代碼路徑"""
    try:
        # 使用最小參數建立模型
        # 執行一次 forward pass
        # 記錄記憶體使用、執行時間
        # 驗證輸出形狀正確
        pass
    except Exception as e:
        print(f"推理測試失敗: {e}")
        return False
    return True

def test_memory_limits():
    """記憶體限制測試：確保不超過 18GB"""
    # 監控記憶體使用
    # 測試邊界條件
    pass

def run_benchmark():
    """執行完整基準測試"""
    results = {
        'inference_success': test_minimal_inference(),
        'memory_within_limits': test_memory_limits(),
        'timestamp': time.time()
    }
    
    # 記錄到結果文件
    with open('../../test-results/benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return all(results.values())
EOF
```

### 階段 3：建立最終工作版本

```bash
# 只有所有實驗都通過，才建立最終版本
if [ -f "test-results/all-experiments-passed" ]; then
    cp -r experiments/exp-03-integration ./Matrix-Game-Working
    echo "✅ 最終工作版本建立完成"
else
    echo "❌ 實驗尚未全部通過，不建立最終版本"
fi
```

## 🔄 故障恢復機制

### 快速重置命令
```bash
# 重置單個實驗
./experiments/reset-experiment.sh exp-01-flash-attention

# 重置所有實驗
rm -rf experiments/exp-* && echo "所有實驗已重置"

# 回到安全狀態
rm -rf Matrix-Game-Working experiments/exp-*
echo "已回到安全狀態，只保留原始代碼和珍貴文檔"
```

### 實驗記錄和追蹤
每次實驗都會產生：
- 時間戳標記的日誌文件
- JSON 格式的測試結果
- 效能基準數據
- 失敗時的完整 traceback

## 🎯 成功標準（基於實測）

每個實驗階段必須達到：
- ✅ **功能測試**: 基本 import 和初始化成功
- ✅ **相容性測試**: 在 MPS 設備上正常運行
- ✅ **精度測試**: 數值差異在可接受範圍內
- ✅ **效能測試**: 執行時間合理
- ✅ **記憶體測試**: 不超過 18GB 限制

## 📊 實驗決策矩陣

| 技術點 | 方案A | 方案B | 方案C | 選擇標準 |
|--------|-------|-------|-------|----------|
| Flash Attention | PyTorch原生 | 手動實現 | 混合策略 | 效能+穩定性 |
| MPS數據類型 | 全Float32 | 智能切換 | MPS原生 | 精度+相容性 |
| 設備管理 | 簡單if-else | 智能管理器 | 硬編碼 | 簡單+有效 |

**原則：實測勝過理論，簡單勝過複雜，可恢復勝過完美**