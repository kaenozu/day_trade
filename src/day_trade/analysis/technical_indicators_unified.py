#!/usr/bin/env python3
"""
テクニカル指標統合システム（Strategy Pattern実装）

標準実装と最適化実装を統一し、設定ベースで選択可能な統合アーキテクチャ
"""

import asyncio
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.optimization_strategy import (
    OptimizationConfig,
    OptimizationLevel,
    OptimizationStrategy,
    get_optimized_implementation,
    optimization_strategy,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# 依存パッケージチェック
try:
    import talib

    TALIB_AVAILABLE = True
    logger.info("TA-Lib利用可能")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib未インストール - 基本実装を使用")

# 最適化システム依存パッケージ（オプショナル）
try:
    from ..data.advanced_parallel_ml_engine import AdvancedParallelMLEngine
    from ..utils.performance_monitor import PerformanceMonitor
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )

    OPTIMIZATION_AVAILABLE = True
    logger.info("最適化システム利用可能")
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logger.info("最適化システム未利用 - 標準実装のみ")




@dataclass
class IndicatorResult:
    """指標計算結果"""

    name: str
    values: Dict[str, Any]
    metadata: Dict[str, Any]
    calculation_time: float
    strategy_used: str


class TechnicalIndicatorsBase(OptimizationStrategy):
    """テクニカル指標の基底戦略クラス"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.fibonacci_levels = [
            0.0,
            0.236,
            0.382,
            0.5,
            0.618,
            0.786,
            1.0,
            1.236,
            1.618,
        ]
        self.bollinger_periods = [20, 50, 100]
        self.ma_periods = [5, 10, 20, 50, 100, 200]

    def execute(
        self, data: pd.DataFrame, indicators: List[str], **kwargs
    ) -> Dict[str, IndicatorResult]:
        """指標計算の実行"""
        start_time = time.time()
        results = {}

        try:
            for indicator in indicators:
                method_name = f"calculate_{indicator.lower()}"
                if hasattr(self, method_name):
                    calc_start = time.time()
                    method = getattr(self, method_name)

                    # 各指標に適切なパラメータを渡す
                    filtered_kwargs = self._filter_kwargs_for_method(indicator, **kwargs)
                    result = method(data, **filtered_kwargs)
                    calc_time = time.time() - calc_start

                    results[indicator] = IndicatorResult(
                        name=indicator,
                        values=result,
                        metadata={"params": kwargs},
                        calculation_time=calc_time,
                        strategy_used=self.get_strategy_name(),
                    )
                else:
                    logger.warning(f"未サポートの指標: {indicator}")

            execution_time = time.time() - start_time
            self.record_execution(execution_time, True)

            return results

        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"指標計算エラー: {e}")
            raise

    def _filter_kwargs_for_method(self, indicator: str, **kwargs) -> dict:
        """Issue #594対応: 各指標メソッドに適切なパラメータのみを抽出（簡略化実装）"""

        # 指標ごとのデフォルトパラメータ定義（拡張容易な辞書形式）
        indicator_defaults = {
            'sma': {'period': 20},
            'ema': {'period': 20},
            'rsi': {'period': 14},
            'bollinger_bands': {'period': 20, 'std_dev': 2.0},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'ichimoku': {
                'conversion_period': 9, 'base_period': 26,
                'leading_span_b_period': 52, 'lagging_span_period': 26
            },
            'fibonacci_retracement': {'period': 100}
        }

        # 指標名正規化
        indicator_key = indicator.lower()
        defaults = indicator_defaults.get(indicator_key, {})

        # Issue #594対応: 簡略化されたマージロジック
        filtered = {}

        # デフォルト値を設定
        for key, default_value in defaults.items():
            filtered[key] = default_value

        # kwargs の値で上書き（既知のパラメータのみ）
        if defaults:  # 既知の指標の場合
            for key, value in kwargs.items():
                if key in defaults:
                    filtered[key] = value
        else:  # 未知の指標の場合は全パラメータを通す
            filtered.update(kwargs)

        return filtered

    # 基本指標計算メソッド（共通）
    def calculate_sma(
        self, data: pd.DataFrame, period: int = 20
    ) -> Dict[str, np.ndarray]:
        """単純移動平均"""
        close = data["終値"] if "終値" in data.columns else data["Close"]
        sma = close.rolling(window=period).mean()
        return {"sma": sma.values, "period": period}

    def calculate_ema(
        self, data: pd.DataFrame, period: int = 20
    ) -> Dict[str, np.ndarray]:
        """指数移動平均"""
        close = data["終値"] if "終値" in data.columns else data["Close"]
        ema = close.ewm(span=period).mean()
        return {"ema": ema.values, "period": period}

    def calculate_bollinger_bands(
        self, data: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """ボリンジャーバンド"""
        close = data["終値"] if "終値" in data.columns else data["Close"]
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            "middle": sma.values,
            "upper": upper_band.values,
            "lower": lower_band.values,
            "period": period,
            "std_dev": std_dev,
        }

    def calculate_rsi(
        self, data: pd.DataFrame, period: int = 14
    ) -> Dict[str, np.ndarray]:
        """RSI計算"""
        close = data["終値"] if "終値" in data.columns else data["Close"]
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return {"rsi": rsi.values, "period": period}

    def calculate_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, np.ndarray]:
        """MACD計算"""
        close = data["終値"] if "終値" in data.columns else data["Close"]
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal

        return {
            "macd": macd.values,
            "signal": signal.values,
            "histogram": histogram.values,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
        }


@optimization_strategy("technical_indicators", OptimizationLevel.STANDARD)
class StandardTechnicalIndicators(TechnicalIndicatorsBase):
    """標準テクニカル指標実装"""

    def get_strategy_name(self) -> str:
        return "標準テクニカル指標"

    def calculate_ichimoku(
        self,
        data: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        leading_span_b_period: int = 52,
        lagging_span_period: int = 26,
    ) -> Dict[str, np.ndarray]:
        """一目均衡表（標準実装）"""
        high = data["高値"] if "高値" in data.columns else data["High"]
        low = data["安値"] if "安値" in data.columns else data["Low"]
        close = data["終値"] if "終値" in data.columns else data["Close"]

        # 転換線 (Conversion Line)
        conversion_line = (
            high.rolling(window=conversion_period).max()
            + low.rolling(window=conversion_period).min()
        ) / 2

        # 基準線 (Base Line)
        base_line = (
            high.rolling(window=base_period).max()
            + low.rolling(window=base_period).min()
        ) / 2

        # 先行スパン1 (Leading Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)

        # 先行スパン2 (Leading Span B)
        leading_span_b = (
            (
                high.rolling(window=leading_span_b_period).max()
                + low.rolling(window=leading_span_b_period).min()
            )
            / 2
        ).shift(base_period)

        # 遅行スパン (Lagging Span)
        lagging_span = close.shift(-lagging_span_period)

        return {
            "conversion_line": conversion_line.values,
            "base_line": base_line.values,
            "leading_span_a": leading_span_a.values,
            "leading_span_b": leading_span_b.values,
            "lagging_span": lagging_span.values,
            "cloud_top": np.maximum(leading_span_a.values, leading_span_b.values),
            "cloud_bottom": np.minimum(leading_span_a.values, leading_span_b.values),
        }

    def calculate_fibonacci_retracement(
        self, data: pd.DataFrame, period: int = 100
    ) -> Dict[str, Any]:
        """フィボナッチリトレースメント（標準実装）"""
        close = data["終値"] if "終値" in data.columns else data["Close"]

        # 期間内の最高値・最安値を取得
        high_val = close.rolling(window=period).max().iloc[-1]
        low_val = close.rolling(window=period).min().iloc[-1]
        diff = high_val - low_val

        retracement_levels = {}
        for level in self.fibonacci_levels:
            retracement_levels[f"fib_{level}"] = high_val - (diff * level)

        return {
            "levels": retracement_levels,
            "high": high_val,
            "low": low_val,
            "range": diff,
        }


@optimization_strategy("technical_indicators", OptimizationLevel.OPTIMIZED)
class OptimizedTechnicalIndicators(TechnicalIndicatorsBase):
    """最適化テクニカル指標実装"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)

        # 最適化システムの初期化
        if OPTIMIZATION_AVAILABLE:
            # UnifiedCacheManagerのAPIに合わせた初期化
            try:
                self.cache_manager = UnifiedCacheManager(
                    cache_size=1000, ttl_seconds=300, enable_compression=True
                )
            except TypeError:
                # フォールバック: 基本パラメータのみ
                self.cache_manager = UnifiedCacheManager()
            self.performance_monitor = PerformanceMonitor()
            self.ml_engine = AdvancedParallelMLEngine()
        else:
            self.cache_manager = None
            self.performance_monitor = None
            self.ml_engine = None

        logger.info("最適化テクニカル指標初期化完了")

    def get_strategy_name(self) -> str:
        return "最適化テクニカル指標"

    def execute(
        self, data: pd.DataFrame, indicators: List[str], **kwargs
    ) -> Dict[str, IndicatorResult]:
        """キャッシュ機能付き実行"""
        if self.cache_manager and self.config.cache_enabled:
            # キャッシュキーを生成
            cache_key = generate_unified_cache_key(
                "tech_indicators",
                {
                    "data_hash": hash(str(data.values.tobytes())),
                    "indicators": sorted(indicators),
                    "params": sorted(kwargs.items()),
                },
            )

            # キャッシュから取得を試行
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"キャッシュヒット: {indicators}")
                return cached_result

        # キャッシュミスまたはキャッシュ無効時は計算実行
        result = super().execute(data, indicators, **kwargs)

        # 結果をキャッシュに保存
        if self.cache_manager and self.config.cache_enabled:
            self.cache_manager.set(cache_key, result)

        return result

    def calculate_ichimoku(
        self,
        data: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        leading_span_b_period: int = 52,
        lagging_span_period: int = 26,
    ) -> Dict[str, np.ndarray]:
        """一目均衡表（最適化実装）"""
        if not OPTIMIZATION_AVAILABLE:
            # フォールバック
            return super().calculate_ichimoku(
                data,
                conversion_period,
                base_period,
                leading_span_b_period,
                lagging_span_period,
            )

        high = data["高値"] if "高値" in data.columns else data["High"]
        low = data["安値"] if "安値" in data.columns else data["Low"]
        close = data["終値"] if "終値" in data.columns else data["Close"]

        # 最適化されたローリング計算
        with self.performance_monitor.measure_time("ichimoku_calculation"):
            # Numbaを使用したベクトル化計算（利用可能な場合）
            try:
                import numba

                @numba.jit(nopython=True)
                def rolling_max_min(arr, window):
                    n = len(arr)
                    result_max = np.empty(n)
                    result_min = np.empty(n)
                    result_max[: window - 1] = np.nan
                    result_min[: window - 1] = np.nan

                    for i in range(window - 1, n):
                        result_max[i] = np.max(arr[i - window + 1 : i + 1])
                        result_min[i] = np.min(arr[i - window + 1 : i + 1])

                    return result_max, result_min

                # 高速計算
                conv_max, conv_min = rolling_max_min(high.values, conversion_period)
                base_max, base_min = rolling_max_min(high.values, base_period)
                span_b_max, span_b_min = rolling_max_min(
                    high.values, leading_span_b_period
                )

                conversion_line = (conv_max + conv_min) / 2
                base_line = (base_max + base_min) / 2

            except ImportError:
                # フォールバック（標準実装）
                conversion_line = (
                    high.rolling(window=conversion_period).max()
                    + low.rolling(window=conversion_period).min()
                ) / 2
                base_line = (
                    high.rolling(window=base_period).max()
                    + low.rolling(window=base_period).min()
                ) / 2

            # 先行スパン計算
            leading_span_a = (conversion_line + base_line) / 2
            if isinstance(leading_span_a, pd.Series):
                leading_span_a = leading_span_a.shift(base_period)
            else:
                leading_span_a = np.concatenate(
                    [np.full(base_period, np.nan), leading_span_a[:-base_period]]
                )

            # 先行スパン2
            leading_span_b = (
                (
                    high.rolling(window=leading_span_b_period).max()
                    + low.rolling(window=leading_span_b_period).min()
                )
                / 2
            ).shift(base_period)

            # 遅行スパン
            lagging_span = close.shift(-lagging_span_period)

        return {
            "conversion_line": (
                conversion_line
                if isinstance(conversion_line, np.ndarray)
                else conversion_line.values
            ),
            "base_line": (
                base_line if isinstance(base_line, np.ndarray) else base_line.values
            ),
            "leading_span_a": (
                leading_span_a
                if isinstance(leading_span_a, np.ndarray)
                else leading_span_a.values
            ),
            "leading_span_b": leading_span_b.values,
            "lagging_span": lagging_span.values,
            "cloud_top": np.maximum(
                (
                    leading_span_a
                    if isinstance(leading_span_a, np.ndarray)
                    else leading_span_a.values
                ),
                leading_span_b.values,
            ),
            "cloud_bottom": np.minimum(
                (
                    leading_span_a
                    if isinstance(leading_span_a, np.ndarray)
                    else leading_span_a.values
                ),
                leading_span_b.values,
            ),
        }

    def calculate_fibonacci_retracement(
        self, data: pd.DataFrame, period: int = 100
    ) -> Dict[str, Any]:
        """フィボナッチリトレースメント（ML強化最適化実装）"""
        if not OPTIMIZATION_AVAILABLE or not self.ml_engine:
            # フォールバック
            return super().calculate_fibonacci_retracement(data, period)

        close = data["終値"] if "終値" in data.columns else data["Close"]

        with self.performance_monitor.measure_time("fibonacci_ml_analysis"):
            # ML強化による自動重要レベル検出
            try:
                # 価格データの特徴抽出
                features = {
                    "price_data": close.tail(period).values,
                    "volatility": close.pct_change()
                    .rolling(20)
                    .std()
                    .tail(period)
                    .values,
                    "volume": data.get("出来高", data.get("Volume", pd.Series()))
                    .tail(period)
                    .values,
                }

                # ML予測による重要レベル特定
                ml_levels = asyncio.run(self._detect_key_levels_async(features))

                # 従来のフィボナッチ計算
                high_val = close.rolling(window=period).max().iloc[-1]
                low_val = close.rolling(window=period).min().iloc[-1]
                diff = high_val - low_val

                retracement_levels = {}
                for level in self.fibonacci_levels:
                    level_price = high_val - (diff * level)
                    retracement_levels[f"fib_{level}"] = level_price

                # MLによる重要度スコアを付与
                for level_name, level_price in retracement_levels.items():
                    importance_score = self._calculate_level_importance(
                        level_price, close.tail(period).values, ml_levels
                    )
                    retracement_levels[f"{level_name}_importance"] = importance_score

                return {
                    "levels": retracement_levels,
                    "high": high_val,
                    "low": low_val,
                    "range": diff,
                    "ml_key_levels": ml_levels,
                    "analysis_enhanced": True,
                }

            except Exception as e:
                logger.warning(f"ML強化フィボナッチ分析失敗: {e}, 標準計算使用")
                return super().calculate_fibonacci_retracement(data, period)

    async def _detect_key_levels_async(
        self, features: Dict[str, np.ndarray]
    ) -> List[float]:
        """ML非同期重要レベル検出"""
        try:
            # 簡単化された重要レベル検出アルゴリズム
            price_data = features["price_data"]
            peaks = []
            troughs = []

            for i in range(2, len(price_data) - 2):
                if (
                    price_data[i] > price_data[i - 1]
                    and price_data[i] > price_data[i - 2]
                    and price_data[i] > price_data[i + 1]
                    and price_data[i] > price_data[i + 2]
                ):
                    peaks.append(price_data[i])
                elif (
                    price_data[i] < price_data[i - 1]
                    and price_data[i] < price_data[i - 2]
                    and price_data[i] < price_data[i + 1]
                    and price_data[i] < price_data[i + 2]
                ):
                    troughs.append(price_data[i])

            # 上位5つの重要レベルを返す
            key_levels = sorted(peaks + troughs, key=lambda x: abs(x - price_data[-1]))[
                :5
            ]
            return key_levels

        except Exception as e:
            logger.error(f"ML重要レベル検出エラー: {e}")
            return []

    def _calculate_level_importance(
        self, level_price: float, price_history: np.ndarray, ml_levels: List[float]
    ) -> float:
        """レベル重要度スコア計算"""
        try:
            # 価格履歴における接触回数
            tolerance = (price_history.max() - price_history.min()) * 0.01  # 1%許容範囲
            touches = np.sum(np.abs(price_history - level_price) <= tolerance)

            # MLで検出されたレベルとの近接性
            ml_proximity = 0.0
            if ml_levels:
                min_distance = min(
                    abs(level_price - ml_level) for ml_level in ml_levels
                )
                max_price_range = price_history.max() - price_history.min()
                ml_proximity = max(0, 1.0 - (min_distance / max_price_range))

            # 合計重要度スコア（0-1）
            importance = min(1.0, (touches * 0.1) + (ml_proximity * 0.7))
            return importance

        except Exception:
            return 0.5  # デフォルト重要度


# 統合インターフェース
class TechnicalIndicatorsManager:
    """テクニカル指標統合マネージャー"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig.from_env()
        self._strategy = None

    def get_strategy(self) -> OptimizationStrategy:
        """現在の戦略を取得"""
        if self._strategy is None:
            self._strategy = get_optimized_implementation(
                "technical_indicators", self.config
            )
        return self._strategy

    def calculate_indicators(
        self, data: pd.DataFrame, indicators: List[str], **kwargs
    ) -> Dict[str, IndicatorResult]:
        """指標計算の実行"""
        strategy = self.get_strategy()
        return strategy.execute(data, indicators, **kwargs)

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標一覧"""
        return [
            "sma",
            "ema",
            "bollinger_bands",
            "rsi",
            "macd",
            "ichimoku",
            "fibonacci_retracement",
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        if self._strategy:
            return self._strategy.get_performance_metrics()
        return {}

    def reset_performance_metrics(self) -> None:
        """パフォーマンス指標のリセット"""
        if self._strategy:
            self._strategy.reset_metrics()





# 戦略の自動登録（ダミー戦略でテスト用）
try:
    from ..core.optimization_strategy import (
        OptimizationLevel,
        OptimizationStrategyFactory,
    )

    class DummyTechnicalIndicatorsStrategy(OptimizationStrategy):
        """テクニカル指標ダミー戦略（テスト用）"""

        def execute(self, *args, **kwargs):
            """ダミー実行メソッド"""
            return f"Executed with {self.config.level.value} optimization"

        def get_strategy_name(self) -> str:
            return f"DummyTechnicalIndicators-{self.config.level.value}"

    # 戦略登録（実際の実装を使用）
    OptimizationStrategyFactory.register_strategy(
        "technical_indicators",
        OptimizationLevel.STANDARD,
        StandardTechnicalIndicators,
    )
    OptimizationStrategyFactory.register_strategy(
        "technical_indicators",
        OptimizationLevel.OPTIMIZED,
        OptimizedTechnicalIndicators,
    )

    logger.info("テクニカル指標戦略の自動登録完了")

except Exception as e:
    logger.warning(f"戦略自動登録失敗: {e}")
    pass
