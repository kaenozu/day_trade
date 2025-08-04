#!/usr/bin/env python3
"""
最適化されたテクニカル指標計算モジュール

Issue #165: アプリケーション全体の処理速度向上に向けた最適化
このモジュールは、NumPy/Pandasのベクトル化機能を最大限活用した
高速なテクニカル指標計算を提供します。
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pandas as pd

# オプショナルな依存関係
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Numbaが利用できない場合のダミー実装
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    prange = range

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

from ..utils.logging_config import get_context_logger
from ..utils.performance_optimizer import PerformanceProfiler


@dataclass
class IndicatorResult:
    """指標計算結果"""

    name: str
    values: Union[pd.Series, pd.DataFrame]
    parameters: Dict
    execution_time: float
    data_points: int


class OptimizedIndicatorCalculator:
    """最適化されたテクニカル指標計算クラス"""

    def __init__(self, enable_parallel: bool = True, n_jobs: int = None):
        self.logger = get_context_logger(__name__)
        self.profiler = PerformanceProfiler()
        self.enable_parallel = enable_parallel
        self.n_jobs = n_jobs or mp.cpu_count()

        # TALibが利用可能かチェック
        self.talib_available = TALIB_AVAILABLE
        if not self.talib_available:
            self.logger.warning("TALibが利用できません。純粋Python実装を使用します。")

    def calculate_multiple_indicators(
        self, data: pd.DataFrame, indicators: List[Dict], use_parallel: bool = None
    ) -> Dict[str, IndicatorResult]:
        """
        複数の指標を並列計算

        Args:
            data: OHLCV データ
            indicators: 指標設定のリスト
                例: [{"name": "sma", "period": 20}, {"name": "rsi", "period": 14}]
            use_parallel: 並列処理を使用するか

        Returns:
            指標名をキーとする計算結果辞書
        """
        use_parallel = (
            use_parallel if use_parallel is not None else self.enable_parallel
        )

        if use_parallel and len(indicators) > 1:
            return self._calculate_parallel(data, indicators)
        else:
            return self._calculate_sequential(data, indicators)

    def _calculate_sequential(
        self, data: pd.DataFrame, indicators: List[Dict]
    ) -> Dict[str, IndicatorResult]:
        """逐次計算"""
        results = {}

        for indicator_config in indicators:
            name = indicator_config["name"]
            try:
                result = self._calculate_single_indicator(data, indicator_config)
                results[name] = result
            except Exception as e:
                self.logger.error(f"指標計算エラー: {name}", error=str(e))

        return results

    def _calculate_parallel(
        self, data: pd.DataFrame, indicators: List[Dict]
    ) -> Dict[str, IndicatorResult]:
        """並列計算"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # タスクを投入
            future_to_indicator = {
                executor.submit(self._calculate_single_indicator, data, config): config
                for config in indicators
            }

            # 結果を収集
            for future in as_completed(future_to_indicator):
                config = future_to_indicator[future]
                name = config["name"]

                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    self.logger.error(f"並列指標計算エラー: {name}", error=str(e))

        return results

    def _calculate_single_indicator(
        self, data: pd.DataFrame, config: Dict
    ) -> IndicatorResult:
        """単一指標の計算"""
        import time

        start_time = time.perf_counter()

        name = config["name"]
        method = getattr(self, f"calculate_{name}", None)

        if method is None:
            raise ValueError(f"未対応の指標: {name}")

        # パラメータを抽出
        params = {k: v for k, v in config.items() if k != "name"}

        # 計算実行
        values = method(data, **params)

        execution_time = time.perf_counter() - start_time

        return IndicatorResult(
            name=name,
            values=values,
            parameters=params,
            execution_time=execution_time,
            data_points=len(data),
        )

    # === 移動平均系指標 ===

    def calculate_sma(
        self, data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.Series:
        """単純移動平均（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.SMA(data[column].values, timeperiod=period),
                index=data.index,
                name=f"sma_{period}",
            )
        else:
            return self._sma_optimized(data[column], period)

    def calculate_ema(
        self, data: pd.DataFrame, period: int = 12, column: str = "close"
    ) -> pd.Series:
        """指数移動平均（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.EMA(data[column].values, timeperiod=period),
                index=data.index,
                name=f"ema_{period}",
            )
        else:
            return self._ema_optimized(data[column], period)

    def calculate_wma(
        self, data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.Series:
        """加重移動平均（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.WMA(data[column].values, timeperiod=period),
                index=data.index,
                name=f"wma_{period}",
            )
        else:
            return self._wma_optimized(data[column], period)

    # === オシレーター系指標 ===

    def calculate_rsi(
        self, data: pd.DataFrame, period: int = 14, column: str = "close"
    ) -> pd.Series:
        """RSI（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.RSI(data[column].values, timeperiod=period),
                index=data.index,
                name=f"rsi_{period}",
            )
        else:
            return self._rsi_optimized(data[column], period)

    def calculate_stoch(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        slow_period: int = 3,
    ) -> pd.DataFrame:
        """ストキャスティクス（最適化版）"""
        if self.talib_available:
            k, d = talib.STOCH(
                data["high"].values,
                data["low"].values,
                data["close"].values,
                fastk_period=k_period,
                slowk_period=slow_period,
                slowd_period=d_period,
            )
            return pd.DataFrame(
                {f"stoch_k_{k_period}": k, f"stoch_d_{d_period}": d}, index=data.index
            )
        else:
            return self._stoch_optimized(data, k_period, d_period, slow_period)

    def calculate_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """MACD（最適化版）"""
        if self.talib_available:
            macd, signal, histogram = talib.MACD(
                data[column].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )
            return pd.DataFrame(
                {"macd": macd, "macd_signal": signal, "macd_histogram": histogram},
                index=data.index,
            )
        else:
            return self._macd_optimized(
                data[column], fast_period, slow_period, signal_period
            )

    # === ボラティリティ系指標 ===

    def calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> pd.DataFrame:
        """ボリンジャーバンド（最適化版）"""
        if self.talib_available:
            upper, middle, lower = talib.BBANDS(
                data[column].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            return pd.DataFrame(
                {
                    f"bb_upper_{period}": upper,
                    f"bb_middle_{period}": middle,
                    f"bb_lower_{period}": lower,
                },
                index=data.index,
            )
        else:
            return self._bollinger_optimized(data[column], period, std_dev)

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.ATR(
                    data["high"].values,
                    data["low"].values,
                    data["close"].values,
                    timeperiod=period,
                ),
                index=data.index,
                name=f"atr_{period}",
            )
        else:
            return self._atr_optimized(data, period)

    # === 出来高系指標 ===

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """OBV（最適化版）"""
        if self.talib_available:
            return pd.Series(
                talib.OBV(data["close"].values, data["volume"].values),
                index=data.index,
                name="obv",
            )
        else:
            return self._obv_optimized(data)

    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """VWAP（最適化版）"""
        return self._vwap_optimized(data)

    # === NumPy/Numba最適化版の実装 ===

    @staticmethod
    @jit(nopython=True)
    def _sma_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba最適化版SMA"""
        n = len(prices)
        result = np.full(n, np.nan)

        if n < period:
            return result

        # 初回計算
        result[period - 1] = np.mean(prices[:period])

        # 増分計算
        for i in prange(period, n):
            result[i] = result[i - 1] + (prices[i] - prices[i - period]) / period

        return result

    def _sma_optimized(self, prices: pd.Series, period: int) -> pd.Series:
        """最適化版SMA（Numba使用）"""
        values = self._sma_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f"sma_{period}")

    @staticmethod
    @jit(nopython=True)
    def _ema_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba最適化版EMA"""
        n = len(prices)
        result = np.full(n, np.nan)

        if n == 0:
            return result

        alpha = 2.0 / (period + 1)
        result[0] = prices[0]

        for i in prange(1, n):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

        return result

    def _ema_optimized(self, prices: pd.Series, period: int) -> pd.Series:
        """最適化版EMA（Numba使用）"""
        values = self._ema_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f"ema_{period}")

    @staticmethod
    @jit(nopython=True)
    def _wma_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba最適化版WMA"""
        n = len(prices)
        result = np.full(n, np.nan)

        if n < period:
            return result

        weights = np.arange(1, period + 1, dtype=np.float64)
        weight_sum = np.sum(weights)

        for i in prange(period - 1, n):
            window = prices[i - period + 1 : i + 1]
            result[i] = np.sum(window * weights) / weight_sum

        return result

    def _wma_optimized(self, prices: pd.Series, period: int) -> pd.Series:
        """最適化版WMA（Numba使用）"""
        values = self._wma_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f"wma_{period}")

    @staticmethod
    @jit(nopython=True)
    def _rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
        """Numba最適化版RSI"""
        n = len(prices)
        result = np.full(n, np.nan)

        if n < period + 1:
            return result

        # 価格変化を計算
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # 初回平均計算
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            result[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[period] = 100.0 - (100.0 / (1.0 + rs))

        # EMアベレージによる更新
        alpha = 1.0 / period
        for i in prange(period + 1, n):
            avg_gain = (1 - alpha) * avg_gain + alpha * gains[i - 1]
            avg_loss = (1 - alpha) * avg_loss + alpha * losses[i - 1]

            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100.0 - (100.0 / (1.0 + rs))

        return result

    def _rsi_optimized(self, prices: pd.Series, period: int) -> pd.Series:
        """最適化版RSI（Numba使用）"""
        values = self._rsi_numba(prices.values, period)
        return pd.Series(values, index=prices.index, name=f"rsi_{period}")

    def _bollinger_optimized(
        self, prices: pd.Series, period: int, std_dev: float
    ) -> pd.DataFrame:
        """最適化版ボリンジャーバンド"""
        # Pandasのローリング関数を使用（十分に最適化されている）
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return pd.DataFrame(
            {
                f"bb_upper_{period}": sma + (std * std_dev),
                f"bb_middle_{period}": sma,
                f"bb_lower_{period}": sma - (std * std_dev),
            },
            index=prices.index,
        )

    def _macd_optimized(
        self, prices: pd.Series, fast_period: int, slow_period: int, signal_period: int
    ) -> pd.DataFrame:
        """最適化版MACD"""
        ema_fast = self._ema_optimized(prices, fast_period)
        ema_slow = self._ema_optimized(prices, slow_period)
        macd = ema_fast - ema_slow
        signal = self._ema_optimized(macd, signal_period)
        histogram = macd - signal

        return pd.DataFrame(
            {"macd": macd, "macd_signal": signal, "macd_histogram": histogram},
            index=prices.index,
        )

    def _stoch_optimized(
        self, data: pd.DataFrame, k_period: int, d_period: int, slow_period: int
    ) -> pd.DataFrame:
        """最適化版ストキャスティクス"""
        lowest_low = data["low"].rolling(window=k_period).min()
        highest_high = data["high"].rolling(window=k_period).max()

        k_percent = 100 * ((data["close"] - lowest_low) / (highest_high - lowest_low))
        k_slow = k_percent.rolling(window=slow_period).mean()
        d_slow = k_slow.rolling(window=d_period).mean()

        return pd.DataFrame(
            {f"stoch_k_{k_period}": k_slow, f"stoch_d_{d_period}": d_slow},
            index=data.index,
        )

    def _atr_optimized(self, data: pd.DataFrame, period: int) -> pd.Series:
        """最適化版ATR"""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()

        return pd.Series(atr, index=data.index, name=f"atr_{period}")

    def _obv_optimized(self, data: pd.DataFrame) -> pd.Series:
        """最適化版OBV"""
        price_change = np.sign(data["close"].diff())
        obv = (price_change * data["volume"]).cumsum()

        return pd.Series(obv, index=data.index, name="obv")

    def _vwap_optimized(self, data: pd.DataFrame) -> pd.Series:
        """最適化版VWAP"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

        return pd.Series(vwap, index=data.index, name="vwap")

    # === 一括計算用の高速化メソッド ===

    def calculate_comprehensive_analysis(
        self, data: pd.DataFrame, include_advanced: bool = True
    ) -> pd.DataFrame:
        """包括的テクニカル分析（一括最適化計算）"""
        result = data.copy()

        # 基本的な移動平均
        basic_indicators = [
            {"name": "sma", "period": 5},
            {"name": "sma", "period": 20},
            {"name": "sma", "period": 50},
            {"name": "ema", "period": 12},
            {"name": "ema", "period": 26},
            {"name": "rsi", "period": 14},
            {"name": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9},
            {"name": "bollinger_bands", "period": 20, "std_dev": 2.0},
            {"name": "atr", "period": 14},
        ]

        # 高速一括計算
        indicator_results = self.calculate_multiple_indicators(
            data, basic_indicators, use_parallel=True
        )

        # 結果をDataFrameに統合
        for _name, indicator_result in indicator_results.items():
            values = indicator_result.values
            if isinstance(values, pd.Series):
                result[values.name] = values
            elif isinstance(values, pd.DataFrame):
                for col in values.columns:
                    result[col] = values[col]

        if include_advanced:
            # 高度な指標を追加
            result = self._add_advanced_indicators(result)

        return result

    def _add_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な指標を追加"""
        # 各種高度な指標を効率的に計算
        result = data.copy()

        # Ichimoku Cloudの簡易版
        if "high" in data.columns and "low" in data.columns:
            result["ichimoku_conversion"] = (
                data["high"].rolling(9).max() + data["low"].rolling(9).min()
            ) / 2
            result["ichimoku_base"] = (
                data["high"].rolling(26).max() + data["low"].rolling(26).min()
            ) / 2

        # パラボリックSAR（簡易版）
        if "high" in data.columns and "low" in data.columns:
            result["sar"] = self._parabolic_sar_optimized(data)

        return result

    def _parabolic_sar_optimized(self, data: pd.DataFrame) -> pd.Series:
        """最適化版パラボリックSAR"""
        # 簡易実装（実際の本格実装は別途）
        high = data["high"]
        low = data["low"]
        data["close"]

        # 初期値設定
        sar = low.iloc[0]
        trend = 1  # 1: uptrend, -1: downtrend
        af = 0.02  # acceleration factor

        sar_values = [sar]

        for i in range(1, len(data)):
            if trend == 1:  # uptrend
                sar = sar + af * (high.iloc[i - 1] - sar)
                if low.iloc[i] <= sar:
                    trend = -1
                    sar = high.iloc[i - 1]
                    af = 0.02
            else:  # downtrend
                sar = sar + af * (low.iloc[i - 1] - sar)
                if high.iloc[i] >= sar:
                    trend = 1
                    sar = low.iloc[i - 1]
                    af = 0.02

            sar_values.append(sar)

        return pd.Series(sar_values, index=data.index, name="sar")


# 使用例とベンチマーク
if __name__ == "__main__":
    import time

    from ..utils.performance_optimizer import create_sample_data

    print("🚀 最適化テクニカル指標計算 - パフォーマンステスト")

    # テストデータ作成
    test_data = create_sample_data(10000)
    test_data.columns = ["date", "open", "high", "low", "close", "volume"]
    test_data.set_index("date", inplace=True)

    # 計算エンジン初期化
    calculator = OptimizedIndicatorCalculator(enable_parallel=True)

    # 個別指標テスト
    print(f"\n📊 {len(test_data)}データポイントでテスト実行")

    start_time = time.perf_counter()

    # 包括的分析実行
    comprehensive_result = calculator.calculate_comprehensive_analysis(
        test_data, include_advanced=True
    )

    execution_time = time.perf_counter() - start_time

    print("✅ 包括的分析完了:")
    print(f"   実行時間: {execution_time:.3f}秒")
    print(
        f"   計算指標数: {len(comprehensive_result.columns) - len(test_data.columns)}"
    )
    print(f"   スループット: {len(test_data) / execution_time:.0f} records/sec")

    # パフォーマンス統計表示
    summary = calculator.profiler.get_summary_report()
    if summary.get("slowest_functions"):
        print("\n⏱️ 最も時間のかかった処理:")
        for func_metrics in summary["slowest_functions"][:3]:
            print(
                f"   {func_metrics.function_name}: {func_metrics.execution_time:.3f}秒"
            )
