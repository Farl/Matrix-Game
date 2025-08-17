#!/usr/bin/env python3
"""
系統性修復所有 CUDA 硬編碼問題
將硬編碼的 CUDA 調用替換為智能設備檢測
"""

import os
import re

def get_smart_device_code():
    """返回智能設備檢測代碼"""
    return '''# 智能設備選擇
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")'''

def fix_file_cuda_issues(file_path):
    """修復單個文件中的 CUDA 問題"""
    print(f"修復文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 修復 torch.cuda.current_device()
    content = re.sub(
        r'device=torch\.cuda\.current_device\(\)',
        'device=device',
        content
    )
    
    # 2. 修復 .cuda()
    content = re.sub(
        r'\.cuda\(\)',
        '.to(device)',
        content
    )
    
    # 3. 修復 device="cuda"
    content = re.sub(
        r'device="cuda"',
        'device=device',
        content
    )
    
    # 4. 修復 torch.device("cuda")
    content = re.sub(
        r'torch\.device\("cuda"\)',
        'device',
        content
    )
    
    # 5. 修復 torch.device(f"cuda:{device_id}")
    content = re.sub(
        r'torch\.device\(f"cuda:\{device_id\}"\)',
        'device',
        content
    )
    
    # 如果內容有變化，需要添加設備檢測代碼
    if content != original_content:
        # 在文件開頭添加 torch import（如果還沒有）
        if 'import torch' not in content and 'torch' in content:
            content = 'import torch\n' + content
        
        # 在適當位置添加設備檢測
        if 'device = torch.device' not in content:
            # 找到第一個函數或類定義，在其前面插入
            device_code = get_smart_device_code()
            
            # 如果這是一個模塊文件，在imports後添加
            lines = content.split('\n')
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_pos = i + 1
                elif line.strip() and not line.startswith('#'):
                    break
            
            lines.insert(insert_pos, '\n' + device_code + '\n')
            content = '\n'.join(lines)
    
    # 寫回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ 已修復")
        return True
    else:
        print(f"  ⚪ 無需修改")
        return False

def main():
    """主函數：修復所有檢測到的文件"""
    
    # 需要修復的文件列表
    files_to_fix = [
        # 核心模組文件
        'wan/modules/posemb_layers.py',
        'wan/modules/t5.py',
        'wan/modules/attention.py', 
        'wan/modules/vae.py',
        'wan/text2video.py',
        'wan/image2video.py',
        'wan/vae/wanx_vae_src/attention.py',
        
        # 管道文件
        'pipeline/causal_inference.py',
        
        # 推理文件
        'inference.py',
        'inference_streaming.py',
        
        # 工具文件
        'demo_utils/memory.py',
        'demo_utils/taehv.py',
        'demo_utils/vae.py',
        # 跳過 vae_torch2trt.py（tensorrt專用）
    ]
    
    print("🔧 開始系統性修復 CUDA 問題...")
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_file_cuda_issues(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n✅ 修復完成！共修復 {fixed_count} 個文件")

if __name__ == "__main__":
    main()