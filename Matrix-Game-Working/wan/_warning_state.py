# 統一的警告狀態控制模組
# 用於避免跨模組重複警告，確保進度條不被打斷

class WarningState:
    """統一的警告狀態管理類"""
    def __init__(self):
        self._flash_attention_warning_shown = False
        self._mps_warning_shown = False
    
    def show_flash_attention_warning(self, message="⚠️  Flash Attention 不可用，使用標準注意力回退"):
        """顯示 Flash Attention 警告，只顯示一次"""
        if not self._flash_attention_warning_shown:
            print(message)
            self._flash_attention_warning_shown = True
            return True
        return False
    
    def show_mps_warning(self, message="ℹ️  MPS 設備：使用 float32 進行 sinusoidal embedding"):
        """顯示 MPS 警告，只顯示一次"""
        if not self._mps_warning_shown:
            print(message)
            self._mps_warning_shown = True
            return True
        return False

# 全域實例，所有模組共享
warning_state = WarningState()