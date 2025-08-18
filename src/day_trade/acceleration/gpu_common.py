"""
GPU計算共通ユーティリティ

CUDAとNumPyの共通計算パターンを抽象化
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Union
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ArrayLike(Protocol):
    """配列風オブジェクトのプロトコル"""
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...


class GPUArrayOperations(ABC):
    """GPU配列操作の抽象基底クラス"""
    
    @abstractmethod
    def zeros_like(self, array: ArrayLike) -> ArrayLike:
        """ゼロ配列作成"""
        pass
    
    @abstractmethod
    def ones(self, size: int) -> ArrayLike:
        """1配列作成"""
        pass
    
    @abstractmethod
    def convolve(self, a: ArrayLike, v: ArrayLike, mode: str = "valid") -> ArrayLike:
        """畳み込み演算"""
        pass
    
    @abstractmethod
    def where(self, condition: ArrayLike, x: ArrayLike, y: Union[ArrayLike, float]) -> ArrayLike:
        """条件分岐"""
        pass
    
    @abstractmethod
    def mean(self, array: ArrayLike) -> float:
        """平均値計算"""
        pass
    
    @abstractmethod
    def std(self, array: ArrayLike) -> float:
        """標準偏差計算"""
        pass
    
    @abstractmethod
    def max(self, array: ArrayLike) -> float:
        """最大値"""
        pass
    
    @abstractmethod
    def min(self, array: ArrayLike) -> float:
        """最小値"""
        pass
    
    @abstractmethod
    def diff(self, array: ArrayLike) -> ArrayLike:
        """差分計算"""
        pass
    
    @abstractmethod
    def pad(self, array: ArrayLike, pad_width: tuple, mode: str = "edge") -> ArrayLike:
        """パディング"""
        pass


class CUDAOperations(GPUArrayOperations):
    """CUDA実装"""
    
    def __init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy is not available")
    
    def zeros_like(self, array):
        return cp.zeros_like(array)
    
    def ones(self, size: int):
        return cp.ones(size)
    
    def convolve(self, a, v, mode: str = "valid"):
        return cp.convolve(a, v, mode=mode)
    
    def where(self, condition, x, y):
        return cp.where(condition, x, y)
    
    def mean(self, array):
        return cp.mean(array)
    
    def std(self, array):
        return cp.std(array)
    
    def max(self, array):
        return cp.max(array)
    
    def min(self, array):
        return cp.min(array)
    
    def diff(self, array):
        return cp.diff(array)
    
    def pad(self, array, pad_width, mode: str = "edge"):
        return cp.pad(array, pad_width, mode=mode)


class NumPyOperations(GPUArrayOperations):
    """NumPy実装（CPUフォールバック）"""
    
    def zeros_like(self, array):
        return np.zeros_like(array)
    
    def ones(self, size: int):
        return np.ones(size)
    
    def convolve(self, a, v, mode: str = "valid"):
        return np.convolve(a, v, mode=mode)
    
    def where(self, condition, x, y):
        return np.where(condition, x, y)
    
    def mean(self, array):
        return np.mean(array)
    
    def std(self, array):
        return np.std(array)
    
    def max(self, array):
        return np.max(array)
    
    def min(self, array):
        return np.min(array)
    
    def diff(self, array):
        return np.diff(array)
    
    def pad(self, array, pad_width, mode: str = "edge"):
        return np.pad(array, pad_width, mode=mode)


class TechnicalIndicatorCalculator:
    """テクニカル指標計算の統一インターフェース"""
    
    def __init__(self, use_gpu: bool = True):
        """
        初期化
        
        Args:
            use_gpu: GPU使用フラグ
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            self.ops = CUDAOperations()
            logger.info("GPU（CUDA）モードで初期化")
        else:
            self.ops = NumPyOperations()
            logger.info("CPU（NumPy）モードで初期化")
    
    def calculate_sma(self, prices: ArrayLike, period: int) -> ArrayLike:
        """
        単純移動平均（SMA）計算
        
        Args:
            prices: 価格配列
            period: 期間
            
        Returns:
            ArrayLike: SMA値配列
        """
        kernel = self.ops.ones(period) / period
        padded_prices = self.ops.pad(prices, (period - 1, 0), mode="edge")
        return self.ops.convolve(padded_prices, kernel, mode="valid")
    
    def calculate_ema(self, prices: ArrayLike, period: int) -> ArrayLike:
        """
        指数移動平均（EMA）計算
        
        Args:
            prices: 価格配列
            period: 期間
            
        Returns:
            ArrayLike: EMA値配列
        """
        alpha = 2.0 / (period + 1)
        ema = self.ops.zeros_like(prices)
        
        # 初期値設定
        if len(prices) > 0:
            ema[0] = prices[0]
        
        # 逐次計算（GPU最適化の余地あり）
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def calculate_rsi(self, prices: ArrayLike, period: int) -> ArrayLike:
        """
        RSI計算
        
        Args:
            prices: 価格配列
            period: 期間
            
        Returns:
            ArrayLike: RSI値配列
        """
        delta = self.ops.diff(prices)
        gain = self.ops.where(delta > 0, delta, 0)
        loss = self.ops.where(delta < 0, -delta, 0)
        
        avg_gain = self.ops.zeros_like(prices)
        avg_loss = self.ops.zeros_like(prices)
        
        # 最初の期間の平均
        if len(gain) >= period:
            avg_gain[period] = self.ops.mean(gain[:period])
            avg_loss[period] = self.ops.mean(loss[:period])
        
        # 指数移動平均による更新
        for i in range(period + 1, len(prices)):
            avg_gain[i] = ((period - 1) * avg_gain[i - 1] + gain[i - 1]) / period
            avg_loss[i] = ((period - 1) * avg_loss[i - 1] + loss[i - 1]) / period
        
        rs = avg_gain / self.ops.where(avg_loss != 0, avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self, 
        prices: ArrayLike, 
        period: int, 
        std_multiplier: float = 2.0
    ) -> Dict[str, ArrayLike]:
        """
        ボリンジャーバンド計算
        
        Args:
            prices: 価格配列
            period: 期間
            std_multiplier: 標準偏差の倍数
            
        Returns:
            Dict[str, ArrayLike]: ボリンジャーバンド（upper, middle, lower）
        """
        middle = self.calculate_sma(prices, period)
        
        # 移動標準偏差計算
        rolling_std = self.ops.zeros_like(prices)
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1 : i + 1]
            rolling_std[i] = self.ops.std(window)
        
        upper_band = middle + std_multiplier * rolling_std
        lower_band = middle - std_multiplier * rolling_std
        
        return {
            "upper": upper_band,
            "middle": middle,
            "lower": lower_band
        }
    
    def calculate_macd(
        self, 
        prices: ArrayLike, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, ArrayLike]:
        """
        MACD計算
        
        Args:
            prices: 価格配列
            fast_period: 高速EMA期間
            slow_period: 低速EMA期間
            signal_period: シグナル線期間
            
        Returns:
            Dict[str, ArrayLike]: MACD（macd, signal, histogram）
        """
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def to_numpy(self, array: ArrayLike) -> np.ndarray:
        """
        NumPy配列に変換
        
        Args:
            array: 変換対象配列
            
        Returns:
            np.ndarray: NumPy配列
        """
        if self.use_gpu and hasattr(array, 'get'):
            return array.get()  # CuPy → NumPy
        return np.asarray(array)


def create_indicator_calculator(prefer_gpu: bool = True) -> TechnicalIndicatorCalculator:
    """
    テクニカル指標計算器のファクトリ関数
    
    Args:
        prefer_gpu: GPU使用を優先するか
        
    Returns:
        TechnicalIndicatorCalculator: 計算器インスタンス
    """
    return TechnicalIndicatorCalculator(use_gpu=prefer_gpu)