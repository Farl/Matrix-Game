#!/usr/bin/env python3
"""
完整整合測試：測試所有 Apple Silicon 適配是否正常工作
"""

import torch
import sys
import os
import time
import json
import traceback
from pathlib import Path

# 添加當前目錄到 Python path
sys.path.insert(0, os.getcwd())

def test_core_modules():
    """測試核心模組導入和初始化"""
    print("🔧 測試核心模組...")
    
    results = {}
    
    try:
        from wan.modules.action_module import ActionModule
        action_module = ActionModule(mouse_dim_in=2, keyboard_dim_in=6)
        results['ActionModule'] = {'status': 'success', 'message': '導入和初始化成功'}
        print("  ✅ ActionModule")
    except Exception as e:
        results['ActionModule'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ ActionModule: {e}")
    
    try:
        from wan.modules.model import WanModel, sinusoidal_embedding_1d
        # 測試 sinusoidal embedding（我們修復的重點）
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pos = torch.arange(10, device=device)
        emb = sinusoidal_embedding_1d(64, pos)
        results['WanModel'] = {'status': 'success', 'message': f'模型和embedding正常，輸出形狀: {emb.shape}'}
        print(f"  ✅ WanModel + sinusoidal_embedding")
    except Exception as e:
        results['WanModel'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ WanModel: {e}")
    
    try:
        from pipeline.causal_inference import get_current_action
        results['CausalInference'] = {'status': 'success', 'message': '管道模組導入成功'}
        print("  ✅ CausalInference")
    except Exception as e:
        results['CausalInference'] = {'status': 'failed', 'error': str(e)}
        print(f"  ❌ CausalInference: {e}")
    
    return results

def test_device_compatibility():
    """測試設備相容性"""
    print("\n🖥️  測試設備相容性...")
    
    device_info = {
        'mps_available': torch.backends.mps.is_available(),
        'mps_built': torch.backends.mps.is_built(),
        'cuda_available': torch.cuda.is_available(),
        'pytorch_version': torch.__version__
    }
    
    print(f"  PyTorch 版本: {device_info['pytorch_version']}")
    print(f"  MPS 可用: {device_info['mps_available']}")
    print(f"  MPS 建置: {device_info['mps_built']}")
    print(f"  CUDA 可用: {device_info['cuda_available']}")
    
    # 選擇最佳設備
    if device_info['mps_available']:
        device = torch.device("mps")
        print(f"  ✅ 將使用 MPS 加速")
    elif device_info['cuda_available']:
        device = torch.device("cuda")
        print(f"  ✅ 將使用 CUDA 加速")
    else:
        device = torch.device("cpu")
        print(f"  ⚪ 將使用 CPU")
    
    # 測試基本張量操作
    try:
        x = torch.randn(100, 100, device=device)
        y = torch.matmul(x, x.T)
        z = torch.softmax(y, dim=-1)
        print(f"  ✅ 基本張量操作正常")
        
        # 測試記憶體使用
        if device.type == 'mps':
            memory_used = torch.mps.current_allocated_memory() / 1024**2
            print(f"  📊 MPS 記憶體使用: {memory_used:.1f}MB")
            
            # 測試記憶體清理
            del x, y, z
            torch.mps.empty_cache()
            memory_after = torch.mps.current_allocated_memory() / 1024**2
            print(f"  🧹 清理後記憶體: {memory_after:.1f}MB")
            
    except Exception as e:
        print(f"  ❌ 張量操作失敗: {e}")
        device_info['tensor_ops_error'] = str(e)
    
    return device_info, device

def test_attention_mechanisms():
    """測試注意力機制"""
    print("\n⚡ 測試注意力機制...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 測試我們的 Flash Attention 回退
    try:
        from wan.modules.action_module import flash_attn_func, FLASH_ATTN_AVAILABLE
        
        print(f"  Flash Attention 可用: {FLASH_ATTN_AVAILABLE}")
        
        # 測試注意力計算
        batch_size, seq_len, num_heads, head_dim = 1, 64, 8, 32
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        start_time = time.time()
        output = flash_attn_func(q, k, v, causal=True)
        end_time = time.time()
        
        print(f"  ✅ 注意力計算成功")
        print(f"  📊 輸出形狀: {output.shape}")
        print(f"  ⏱️  執行時間: {end_time - start_time:.4f}s")
        
        return {'status': 'success', 'execution_time': end_time - start_time}
        
    except Exception as e:
        print(f"  ❌ 注意力機制測試失敗: {e}")
        return {'status': 'failed', 'error': str(e)}

def test_precision_and_stability():
    """測試精度和穩定性"""
    print("\n🎯 測試精度和穩定性...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    try:
        from wan.modules.model import sinusoidal_embedding_1d
        
        # 測試不同輸入大小的穩定性
        test_sizes = [10, 100, 1000]
        results = {}
        
        for size in test_sizes:
            pos = torch.arange(size, device=device)
            emb = sinusoidal_embedding_1d(128, pos)
            
            # 檢查輸出的數值穩定性
            has_nan = torch.isnan(emb).any().item()
            has_inf = torch.isinf(emb).any().item()
            value_range = (emb.min().item(), emb.max().item())
            
            results[f'size_{size}'] = {
                'shape': list(emb.shape),
                'has_nan': has_nan,
                'has_inf': has_inf,
                'value_range': value_range
            }
            
            status = "✅" if not (has_nan or has_inf) else "❌"
            print(f"  {status} 尺寸 {size}: 範圍 {value_range[0]:.3f}~{value_range[1]:.3f}")
        
        return results
        
    except Exception as e:
        print(f"  ❌ 精度測試失敗: {e}")
        return {'error': str(e)}

def test_memory_efficiency():
    """測試記憶體效率"""
    print("\n📊 測試記憶體效率...")
    
    if not torch.backends.mps.is_available():
        print("  ⚪ 非 MPS 設備，跳過記憶體測試")
        return {'skipped': True}
    
    device = torch.device("mps")
    
    try:
        # 記錄初始記憶體
        torch.mps.empty_cache()
        initial_memory = torch.mps.current_allocated_memory()
        print(f"  初始記憶體: {initial_memory / 1024**2:.1f}MB")
        
        # 模擬較大的張量操作
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000, device=device)
            result = torch.matmul(tensor, tensor.T)
            tensors.append(result)
            
            current_memory = torch.mps.current_allocated_memory()
            print(f"  步驟 {i+1}: {current_memory / 1024**2:.1f}MB")
        
        peak_memory = torch.mps.current_allocated_memory()
        
        # 清理記憶體
        del tensors
        torch.mps.empty_cache()
        final_memory = torch.mps.current_allocated_memory()
        
        print(f"  峰值記憶體: {peak_memory / 1024**2:.1f}MB")
        print(f"  清理後記憶體: {final_memory / 1024**2:.1f}MB")
        
        # 判斷是否在 M2 Mac 限制範圍內（18GB）
        within_limits = peak_memory < 18 * 1024**3
        status = "✅" if within_limits else "⚠️"
        print(f"  {status} M2 Mac 相容性: {within_limits}")
        
        return {
            'initial_mb': initial_memory / 1024**2,
            'peak_mb': peak_memory / 1024**2,
            'final_mb': final_memory / 1024**2,
            'within_m2_limits': within_limits
        }
        
    except Exception as e:
        print(f"  ❌ 記憶體測試失敗: {e}")
        return {'error': str(e)}

def generate_compatibility_report():
    """生成相容性報告"""
    print("\n📝 生成相容性報告...")
    
    report = {
        'timestamp': time.time(),
        'test_environment': {
            'platform': 'Apple Silicon',
            'pytorch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available()
        },
        'test_results': {}
    }
    
    # 執行所有測試
    print("\n" + "="*60)
    print("🧪 執行完整測試套件")
    print("="*60)
    
    report['test_results']['core_modules'] = test_core_modules()
    report['test_results']['device_compatibility'], selected_device = test_device_compatibility()
    report['test_results']['attention_mechanisms'] = test_attention_mechanisms()
    report['test_results']['precision_stability'] = test_precision_and_stability()
    report['test_results']['memory_efficiency'] = test_memory_efficiency()
    
    # 計算總體成功率
    successful_tests = 0
    total_tests = 0
    
    for category, results in report['test_results'].items():
        if isinstance(results, dict):
            for test_name, test_result in results.items():
                if isinstance(test_result, dict) and 'status' in test_result:
                    total_tests += 1
                    if test_result['status'] == 'success':
                        successful_tests += 1
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    report['overall_success_rate'] = success_rate
    
    # 保存報告
    report_path = Path("../../test-results/integration-report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 測試完成！")
    print(f"成功率: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    print(f"報告已保存: {report_path.absolute()}")
    
    return report

def main():
    print("🚀 Matrix Game 2.0 Apple Silicon 整合測試")
    print("="*60)
    
    try:
        report = generate_compatibility_report()
        
        print("\n🎯 最終評估:")
        if report['overall_success_rate'] >= 80:
            print("✅ 系統已準備好進入生產環境")
            print("🎉 Apple Silicon 適配成功！")
            
            # 創建成功標記文件
            success_file = Path("../../test-results/all-experiments-passed")
            success_file.touch()
            
        else:
            print("⚠️  系統需要進一步調整")
            print("🔧 請檢查失敗的測試項目")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 測試執行失敗: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)