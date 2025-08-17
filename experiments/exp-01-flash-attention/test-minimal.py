#!/usr/bin/env python3
"""最小化測試：檢查核心模組是否可以導入和初始化"""
import torch
import sys
import traceback
import os

def test_import():
    """測試基本導入"""
    print("🧪 測試模組導入...")
    try:
        # 添加當前目錄到 Python path
        sys.path.insert(0, os.getcwd())
        
        from wan.modules.action_module import ActionModule
        print("  ✅ ActionModule 導入成功")
        
        # 測試 ActionModule 初始化
        action_module = ActionModule(mouse_dim_in=2, keyboard_dim_in=6)
        print("  ✅ ActionModule 初始化成功")
        
        from wan.modules.model import WanModel
        print("  ✅ WanModel 導入成功")
        
        from pipeline.causal_inference import get_current_action
        print("  ✅ causal_inference 導入成功")
        
        print("✅ 基本導入測試通過")
        return True
    except Exception as e:
        print(f"❌ 導入失敗: {e}")
        traceback.print_exc()
        return False

def test_device_detection():
    """測試設備檢測"""
    print("\n🔍 測試設備檢測...")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  MPS 可用: {torch.backends.mps.is_available()}")
    print(f"  MPS 已建置: {torch.backends.mps.is_built()}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  將使用設備: {device}")
    
    # 測試基本張量操作
    try:
        print("  測試基本張量操作...")
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)
        print(f"  張量形狀: {y.shape}")
        print("✅ 基本張量操作通過")
        return True
    except Exception as e:
        print(f"❌ 張量操作失敗: {e}")
        traceback.print_exc()
        return False

def test_flash_attention_current_state():
    """測試當前 Flash Attention 狀態"""
    print("\n⚡ 測試 Flash Attention 當前狀態...")
    try:
        from flash_attn import flash_attn_func
        print("✅ Flash Attention 可用")
        return True
    except ImportError as e:
        print(f"❌ Flash Attention 不可用: {e}")
        return False

def test_original_requirements():
    """測試原始 requirements 在 Apple Silicon 上的相容性"""
    print("\n📦 測試關鍵依賴...")
    
    # 檢查 NVIDIA 相關依賴的狀態
    nvidia_deps = ["nvidia-pyindex", "nvidia-tensorrt", "pycuda"]
    for dep in nvidia_deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  ⚠️  {dep} 已安裝（可能會有問題）")
        except ImportError:
            print(f"  ✅ {dep} 未安裝（符合預期）")
    
    # 檢查關鍵依賴
    key_deps = ["torch", "diffusers", "transformers", "accelerate"]
    for dep in key_deps:
        try:
            module = __import__(dep)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✅ {dep}: {version}")
        except ImportError:
            print(f"  ❌ {dep} 未安裝")
            return False
    
    return True

if __name__ == "__main__":
    print("🚀 開始最小化測試...")
    print("=" * 50)
    
    tests = [
        ("設備檢測", test_device_detection),
        ("依賴檢查", test_original_requirements),
        ("Flash Attention狀態", test_flash_attention_current_state),
        ("模組導入", test_import),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 執行測試: {test_name}")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 測試結果總覽:")
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
    
    overall_success = all(results.values())
    print(f"\n🎯 總體結果: {'✅ 所有測試通過' if overall_success else '❌ 部分測試失敗'}")
    
    sys.exit(0 if overall_success else 1)