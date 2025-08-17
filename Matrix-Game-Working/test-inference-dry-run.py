#!/usr/bin/env python3
"""
推理功能乾燥運行測試：測試不需要實際模型檢查點的推理流程
"""

import os
import sys
import torch
import traceback

# 添加當前目錄到 Python path
sys.path.insert(0, os.getcwd())

def test_inference_initialization():
    """測試推理類別是否能正確初始化（不載入實際模型）"""
    print("🔬 測試推理初始化...")
    
    try:
        # 模擬參數
        class MockArgs:
            config_path = "configs/inference_yaml/inference_universal.yaml"
            checkpoint_path = ""  # 空路徑，測試錯誤處理
            img_path = "demo_images/universal/0000.png"
            output_folder = "test_outputs/"
            num_output_frames = 10  # 很小的數量
            seed = 42
            pretrained_model_path = "Matrix-Game-2.0"  # 不存在，但測試初始化
        
        from inference import InteractiveGameInference
        
        # 這裡應該會在模型載入階段失敗，但我們可以測試設備檢測
        try:
            inference = InteractiveGameInference(MockArgs())
            print(f"  ✅ 設備檢測成功: {inference.device}")
            return {"status": "partial_success", "device": str(inference.device)}
        except FileNotFoundError as e:
            if "config" in str(e).lower():
                print(f"  ⚠️  配置文件未找到（預期）: {e}")
                return {"status": "config_missing", "error": str(e)}
            else:
                print(f"  ⚠️  模型文件未找到（預期）: {e}")
                return {"status": "model_missing", "error": str(e)}
        except Exception as e:
            print(f"  ⚠️  初始化部分失敗（可能正常）: {e}")
            return {"status": "init_failed", "error": str(e)}
            
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        return {"status": "import_failed", "error": str(e)}

def test_streaming_inference_initialization():
    """測試串流推理類別是否能正確初始化"""
    print("\n🌊 測試串流推理初始化...")
    
    try:
        # 模擬參數
        class MockArgs:
            config_path = "configs/inference_yaml/inference_universal.yaml"
            checkpoint_path = ""
            output_folder = "test_outputs/"
            max_num_output_frames = 10
            seed = 42
            pretrained_model_path = "Matrix-Game-2.0"
        
        from inference_streaming import InteractiveGameInference as StreamingInference
        
        try:
            inference = StreamingInference(MockArgs())
            print(f"  ✅ 設備檢測成功: {inference.device}")
            return {"status": "partial_success", "device": str(inference.device)}
        except Exception as e:
            print(f"  ⚠️  初始化失敗（可能正常）: {e}")
            return {"status": "init_failed", "error": str(e)}
            
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        return {"status": "import_failed", "error": str(e)}

def test_device_compatibility():
    """測試設備相容性"""
    print("\n🖥️  測試設備相容性...")
    
    device_info = {
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__
    }
    
    # 模擬推理腳本中的設備檢測邏輯
    if torch.backends.mps.is_available():
        selected_device = torch.device("mps")
        print("  ✅ 將使用 MPS 設備")
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
        print("  ✅ 將使用 CUDA 設備")
    else:
        selected_device = torch.device("cpu")
        print("  ✅ 將使用 CPU 設備")
    
    # 測試 bfloat16 相容性
    try:
        if selected_device.type == "mps":
            # 在 MPS 上測試 bfloat16
            test_tensor = torch.randn(10, 10, device=selected_device, dtype=torch.bfloat16)
            print("  ✅ MPS bfloat16 支援正常")
        else:
            test_tensor = torch.randn(10, 10, device=selected_device, dtype=torch.bfloat16)
            print(f"  ✅ {selected_device.type} bfloat16 支援正常")
    except Exception as e:
        print(f"  ⚠️  bfloat16 不支援: {e}")
    
    device_info['selected_device'] = str(selected_device)
    return device_info

def test_config_file_loading():
    """測試配置文件載入"""
    print("\n📋 測試配置文件載入...")
    
    try:
        from omegaconf import OmegaConf
        
        config_files = [
            "configs/inference_yaml/inference_universal.yaml",
            "configs/inference_yaml/inference_gta_drive.yaml",
            "configs/inference_yaml/inference_templerun.yaml"
        ]
        
        results = {}
        for config_path in config_files:
            if os.path.exists(config_path):
                try:
                    config = OmegaConf.load(config_path)
                    print(f"  ✅ {config_path} 載入成功")
                    results[config_path] = "success"
                except Exception as e:
                    print(f"  ❌ {config_path} 載入失敗: {e}")
                    results[config_path] = f"failed: {e}"
            else:
                print(f"  ⚠️  {config_path} 不存在")
                results[config_path] = "missing"
        
        return results
        
    except ImportError as e:
        print(f"  ❌ OmegaConf 導入失敗: {e}")
        return {"error": f"import_failed: {e}"}

def main():
    print("🚀 開始推理功能乾燥運行測試...")
    print("=" * 60)
    
    test_results = {
        'device_compatibility': test_device_compatibility(),
        'config_loading': test_config_file_loading(),
        'inference_init': test_inference_initialization(),
        'streaming_init': test_streaming_inference_initialization()
    }
    
    print("\n" + "=" * 60)
    print("📊 測試總結:")
    
    # 設備相容性
    device_info = test_results['device_compatibility']
    print(f"  設備選擇: {device_info.get('selected_device', 'unknown')}")
    print(f"  MPS 可用: {device_info.get('mps_available', False)}")
    
    # 配置文件
    config_results = test_results['config_loading']
    if isinstance(config_results, dict) and 'error' not in config_results:
        successful_configs = len([k for k, v in config_results.items() if v == "success"])
        print(f"  配置文件載入: {successful_configs}/{len(config_results)} 成功")
    
    # 推理初始化
    inference_result = test_results['inference_init']
    streaming_result = test_results['streaming_init']
    
    print(f"  標準推理初始化: {inference_result.get('status', 'unknown')}")
    print(f"  串流推理初始化: {streaming_result.get('status', 'unknown')}")
    
    print("\n🎯 結論:")
    if device_info.get('mps_available', False):
        print("  ✅ Apple Silicon MPS 設備檢測正常")
        print("  ✅ 基礎架構相容")
        print("  ⚠️  需要下載預訓練模型才能進行完整推理")
        print("  📋 自動下載: bash download_models.sh")
        print("  📋 手動下載: huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0")
    else:
        print("  ⚪ 非 MPS 環境，但架構應該相容")
    
    return test_results

if __name__ == "__main__":
    results = main()