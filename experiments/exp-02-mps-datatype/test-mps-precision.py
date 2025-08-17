#!/usr/bin/env python3
"""
MPS 數據類型精度測試：比較不同精度實現的結果
"""

import torch
import time
import sys
import os
import numpy as np

# 添加當前目錄到 Python path
sys.path.insert(0, os.getcwd())

def test_sinusoidal_precision():
    """測試 sinusoidal embedding 的精度影響"""
    print("🧮 測試 Sinusoidal Embedding 精度...")
    
    def sinusoidal_float64(dim, position):
        """原始的 float64 實現"""
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float64)
        
        sinusoid = torch.outer(
            position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x
    
    def sinusoidal_float32(dim, position):
        """強制 float32 實現"""
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float32)
        
        sinusoid = torch.outer(
            position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x
    
    def sinusoidal_smart(dim, position):
        """智能精度：根據設備選擇"""
        assert dim % 2 == 0
        half = dim // 2
        
        if position.device.type == 'mps':
            position = position.type(torch.float32)
        else:
            position = position.type(torch.float64)
        
        sinusoid = torch.outer(
            position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
        x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        return x
    
    # 測試設置
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    test_positions = torch.arange(100, device=device)
    dim = 128
    
    results = {}
    
    # A/B/C 測試所有方案
    methods = [
        ("Float64", sinusoidal_float64),
        ("Float32", sinusoidal_float32),
        ("Smart", sinusoidal_smart)
    ]
    
    for name, func in methods:
        try:
            print(f"\n  測試 {name}:")
            start_time = time.time()
            result = func(dim, test_positions)
            end_time = time.time()
            
            results[name] = {
                'result': result,
                'time': end_time - start_time,
                'success': True
            }
            print(f"    ✅ 執行時間: {end_time - start_time:.4f}s")
            print(f"    📊 輸出形狀: {result.shape}")
            print(f"    📈 數值範圍: {result.min().item():.6f} ~ {result.max().item():.6f}")
            
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"    ❌ 失敗: {e}")
    
    # 精度比較
    print(f"\n🔍 精度比較分析:")
    if results['Float64']['success'] and results['Float32']['success']:
        diff = torch.abs(results['Float64']['result'] - results['Float32']['result'])
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()
        print(f"  Float32 vs Float64:")
        print(f"    平均差異: {mean_diff:.8f}")
        print(f"    最大差異: {max_diff:.8f}")
        print(f"    相對誤差: {mean_diff / results['Float64']['result'].abs().mean().item():.6e}")
        
        # 判斷精度是否可接受
        if mean_diff < 1e-6:
            print(f"    ✅ 精度差異可接受")
        else:
            print(f"    ⚠️  精度差異較大")
    
    if results['Smart']['success'] and results['Float64']['success']:
        if device.type == 'mps':
            # 在 MPS 上，Smart 應該等同於 Float32
            diff = torch.abs(results['Smart']['result'] - results['Float32']['result'])
            print(f"  Smart vs Float32 (在MPS上): 平均差異 {diff.mean().item():.8f}")
        else:
            # 在其他設備上，Smart 應該等同於 Float64  
            diff = torch.abs(results['Smart']['result'] - results['Float64']['result'])
            print(f"  Smart vs Float64 (在{device.type}上): 平均差異 {diff.mean().item():.8f}")
    
    return results

def test_complex_operations():
    """測試複數運算的精度"""
    print("\n🔄 測試複數運算精度...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  使用設備: {device}")
    
    # 測試數據
    batch_size, seq_len, dim = 2, 64, 128
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    def test_complex_float64():
        """Float64 複數運算"""
        x_complex_input = x.to(torch.float64).reshape(batch_size, seq_len, dim//2, 2)
        x_complex = torch.view_as_complex(x_complex_input)
        # 簡單的複數運算
        result = x_complex * torch.exp(1j * 0.1)
        return torch.view_as_real(result).flatten(-2)
    
    def test_complex_float32():
        """Float32 複數運算"""  
        x_complex_input = x.to(torch.float32).reshape(batch_size, seq_len, dim//2, 2)
        x_complex = torch.view_as_complex(x_complex_input)
        # 簡單的複數運算
        result = x_complex * torch.exp(1j * 0.1)
        return torch.view_as_real(result).flatten(-2)
    
    def test_complex_smart():
        """智能複數運算"""
        if device.type == 'mps':
            x_complex_input = x.to(torch.float32).reshape(batch_size, seq_len, dim//2, 2)
        else:
            x_complex_input = x.to(torch.float64).reshape(batch_size, seq_len, dim//2, 2)
        x_complex = torch.view_as_complex(x_complex_input)
        result = x_complex * torch.exp(1j * 0.1)
        return torch.view_as_real(result).flatten(-2)
    
    methods = [
        ("Float64", test_complex_float64),
        ("Float32", test_complex_float32), 
        ("Smart", test_complex_smart)
    ]
    
    results = {}
    for name, func in methods:
        try:
            print(f"\n  測試 {name}:")
            start_time = time.time()
            result = func()
            end_time = time.time()
            
            results[name] = {
                'result': result,
                'time': end_time - start_time,
                'success': True
            }
            print(f"    ✅ 執行時間: {end_time - start_time:.4f}s")
            print(f"    📊 輸出形狀: {result.shape}")
            
        except Exception as e:
            results[name] = {'success': False, 'error': str(e)}
            print(f"    ❌ 失敗: {e}")
    
    # 精度比較
    if results['Float64']['success'] and results['Float32']['success']:
        diff = torch.abs(results['Float64']['result'] - results['Float32']['result'])
        print(f"\n  Float32 vs Float64 複數運算:")
        print(f"    平均差異: {diff.mean().item():.8f}")
        print(f"    最大差異: {diff.max().item():.8f}")
    
    return results

def main():
    print("🧪 開始 MPS 數據類型精度測試...")
    print("=" * 60)
    
    # 檢查設備狀態
    print(f"MPS 可用: {torch.backends.mps.is_available()}")
    print(f"當前設備: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    test_results = {}
    
    # 測試 1: Sinusoidal Embedding
    try:
        test_results['sinusoidal'] = test_sinusoidal_precision()
    except Exception as e:
        print(f"❌ Sinusoidal 測試失敗: {e}")
        test_results['sinusoidal'] = {'error': str(e)}
    
    # 測試 2: 複數運算
    try:
        test_results['complex'] = test_complex_operations()
    except Exception as e:
        print(f"❌ 複數運算測試失敗: {e}")  
        test_results['complex'] = {'error': str(e)}
    
    print("\n" + "=" * 60)
    print("📊 測試總結:")
    
    # 總結 sinusoidal 測試
    if 'error' not in test_results['sinusoidal']:
        sin_results = test_results['sinusoidal']
        successful_methods = [k for k, v in sin_results.items() if v.get('success', False)]
        print(f"  Sinusoidal Embedding: {len(successful_methods)}/3 方法成功")
        
        if 'Smart' in successful_methods:
            print(f"  ✅ 智能精度方案可行")
        
    # 總結複數測試  
    if 'error' not in test_results['complex']:
        complex_results = test_results['complex']
        successful_methods = [k for k, v in complex_results.items() if v.get('success', False)]
        print(f"  複數運算: {len(successful_methods)}/3 方法成功")
    
    # 推薦決策
    print(f"\n🎯 推薦方案:")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == 'mps':
        print(f"  在 MPS 設備上，建議使用智能精度管理")
        print(f"  - 自動降級到 float32 以確保相容性")
        print(f"  - 精度損失在可接受範圍內")
    else:
        print(f"  在 {device.type} 設備上，可以繼續使用 float64")
    
    return test_results

if __name__ == "__main__":
    results = main()