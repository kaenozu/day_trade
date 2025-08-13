#!/usr/bin/env python3
"""
Issue #619対応: テクニカル指標計算ロジックの統合

全てのテクニカル指標計算を統一インターフェースで提供する統合システム
- 重複コード除去
- パフォーマンス最適化
- 一貫したAPI提供
"""

import asyncio
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class IndicatorCategory(Enum):
    """指標カテゴリ"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    CYCLE = "cycle"


class SignalStrength(Enum):
    """シグナル強度"""
    VERY_STRONG_BUY = 5
    STRONG_BUY = 4
    BUY = 3
    NEUTRAL = 0
    SELL = -3
    STRONG_SELL = -4
    VERY_STRONG_SELL = -5


@dataclass
class IndicatorConfig:
    """統合指標設定"""
    enabled_categories: List[IndicatorCategory] = None
    timeframes: List[str] = None
    lookback_periods: int = 252
    smoothing_factor: float = 0.1
    signal_threshold: float = 0.7
    combine_signals: bool = True
    parallel_computation: bool = True
    cache_results: bool = True
    use_talib: bool = True  # Issue #619対応: TA-Lib使用設定

    def __post_init__(self):
        if self.enabled_categories is None:
            self.enabled_categories = list(IndicatorCategory)
        if self.timeframes is None:
            self.timeframes = ["1D"]


@dataclass
class IndicatorResult:
    """統合指標結果"""
    name: str
    category: IndicatorCategory
    values: Union[np.ndarray, Dict[str, np.ndarray]]
    signals: np.ndarray
    signal_strength: SignalStrength
    confidence: float
    timeframe: str
    calculation_time: float
    metadata: Dict[str, Any]
    implementation_used: str  # Issue #619対応: 使用実装の記録


class ConsolidatedTechnicalIndicators:
    """Issue #619対応: 統合テクニカル指標計算エンジン"""

    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.cache = {}
        self.performance_metrics = {
            "calculation_times": {},
            "total_calculations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "talib_usage": 0,
            "fallback_usage": 0
        }

        logger.info(f"統合テクニカル指標エンジン初期化完了 (TA-Lib: {TALIB_AVAILABLE})")

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        symbols: List[str] = None,
        timeframe: str = "1D",
        **kwargs
    ) -> Dict[str, List[IndicatorResult]]:
        """Issue #619対応: 統合指標計算実行"""

        start_time = time.time()

        if symbols is None:
            symbols = ["Close"] if "Close" in data.columns else [data.columns[0]]

        results = {}

        for symbol in symbols:
            results[symbol] = []

            # データ準備
            symbol_data = self._prepare_data(data, symbol)
            if symbol_data is None:
                continue

            for indicator_name in indicators:
                try:
                    # キャッシュチェック
                    cache_key = self._generate_cache_key(symbol_data, indicator_name, timeframe, kwargs)

                    if self.config.cache_results and cache_key in self.cache:
                        result = self.cache[cache_key]
                        self.performance_metrics["cache_hits"] += 1
                    else:
                        # 指標計算実行
                        result = self._calculate_single_indicator(
                            symbol_data, indicator_name, symbol, timeframe, **kwargs
                        )

                        if self.config.cache_results:
                            self.cache[cache_key] = result
                        self.performance_metrics["cache_misses"] += 1

                    results[symbol].append(result)

                except Exception as e:
                    logger.error(f"指標計算エラー {indicator_name} ({symbol}): {e}")
                    continue

        total_time = time.time() - start_time
        self.performance_metrics["total_calculations"] += 1

        logger.info(f"統合指標計算完了: {len(results)}銘柄, {total_time:.3f}秒")
        return results

    def _prepare_data(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Issue #619対応: データ準備統一処理"""
        try:
            if symbol in data.columns:
                # シンボルが列名の場合
                if isinstance(data[symbol], pd.Series):
                    return pd.DataFrame({'Close': data[symbol]})
                else:
                    return data[[symbol]]
            elif "Close" in data.columns:
                # OHLCV データの場合
                return data
            else:
                logger.warning(f"データが見つかりません: {symbol}")
                return None
        except Exception as e:
            logger.error(f"データ準備エラー {symbol}: {e}")
            return None

    def _calculate_single_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        symbol: str,
        timeframe: str,
        **kwargs
    ) -> IndicatorResult:
        """Issue #619対応: 単一指標計算統一処理"""

        calc_start = time.time()

        # 指標計算メソッドの選択と実行
        if hasattr(self, f"_calc_{indicator_name}"):
            method = getattr(self, f"_calc_{indicator_name}")
            values, metadata, impl_used = method(data, **kwargs)
        else:
            # デフォルト処理
            values = self._calc_default(data, indicator_name, **kwargs)
            metadata = {"indicator": indicator_name}
            impl_used = "default"

        # シグナル計算
        signals = self._calculate_signals(values, indicator_name)
        signal_strength = self._evaluate_signal_strength(signals)
        confidence = self._calculate_confidence(values, signals)

        # カテゴリ決定
        category = self._get_indicator_category(indicator_name)

        calc_time = time.time() - calc_start

        # パフォーマンス記録
        if indicator_name not in self.performance_metrics["calculation_times"]:
            self.performance_metrics["calculation_times"][indicator_name] = []
        self.performance_metrics["calculation_times"][indicator_name].append(calc_time)

        return IndicatorResult(
            name=indicator_name,
            category=category,
            values=values,
            signals=signals,
            signal_strength=signal_strength,
            confidence=confidence,
            timeframe=timeframe,
            calculation_time=calc_time,
            metadata={**metadata, "symbol": symbol},
            implementation_used=impl_used
        )

    # === Issue #619対応: 統合指標計算メソッド群 ===

    def _calc_sma(self, data: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[np.ndarray, Dict, str]:
        """単純移動平均統一計算"""
        close = self._get_close_price(data)

        if TALIB_AVAILABLE and self.config.use_talib:
            try:
                values = talib.SMA(close.values, timeperiod=period)
                self.performance_metrics["talib_usage"] += 1
                return values, {"period": period}, "talib"
            except Exception as e:
                logger.debug(f"TA-Lib SMA失敗: {e}, フォールバック使用")

        # フォールバック実装
        values = close.rolling(window=period).mean().values
        self.performance_metrics["fallback_usage"] += 1
        return values, {"period": period}, "pandas"

    def _calc_ema(self, data: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[np.ndarray, Dict, str]:
        """指数移動平均統一計算"""
        close = self._get_close_price(data)

        if TALIB_AVAILABLE and self.config.use_talib:
            try:
                values = talib.EMA(close.values, timeperiod=period)
                self.performance_metrics["talib_usage"] += 1
                return values, {"period": period}, "talib"
            except Exception:
                pass

        values = close.ewm(span=period).mean().values
        self.performance_metrics["fallback_usage"] += 1
        return values, {"period": period}, "pandas"

    def _calc_rsi(self, data: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[np.ndarray, Dict, str]:
        """RSI統一計算"""
        close = self._get_close_price(data)

        if TALIB_AVAILABLE and self.config.use_talib:
            try:
                values = talib.RSI(close.values, timeperiod=period)
                self.performance_metrics["talib_usage"] += 1
                return values, {"period": period}, "talib"
            except Exception:
                pass

        # フォールバック実装
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        self.performance_metrics["fallback_usage"] += 1
        return rsi.values, {"period": period}, "pandas"

    def _calc_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict, str]:
        """MACD統一計算"""
        close = self._get_close_price(data)

        if TALIB_AVAILABLE and self.config.use_talib:
            try:
                macd, signal, hist = talib.MACD(
                    close.values,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                    signalperiod=signal_period
                )
                values = {
                    "macd": macd,
                    "signal": signal,
                    "histogram": hist
                }
                metadata = {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period
                }
                self.performance_metrics["talib_usage"] += 1
                return values, metadata, "talib"
            except Exception:
                pass

        # フォールバック実装
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period).mean()
        histogram = macd - signal

        values = {
            "macd": macd.values,
            "signal": signal.values,
            "histogram": histogram.values
        }
        metadata = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period
        }

        self.performance_metrics["fallback_usage"] += 1
        return values, metadata, "pandas"

    def _calc_bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict, str]:
        """ボリンジャーバンド統一計算"""
        close = self._get_close_price(data)

        if TALIB_AVAILABLE and self.config.use_talib:
            try:
                upper, middle, lower = talib.BBANDS(
                    close.values,
                    timeperiod=period,
                    nbdevup=std_dev,
                    nbdevdn=std_dev
                )
                values = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower
                }
                metadata = {"period": period, "std_dev": std_dev}
                self.performance_metrics["talib_usage"] += 1
                return values, metadata, "talib"
            except Exception:
                pass

        # フォールバック実装
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        values = {
            "upper": upper.values,
            "middle": middle.values,
            "lower": lower.values
        }
        metadata = {"period": period, "std_dev": std_dev}

        self.performance_metrics["fallback_usage"] += 1
        return values, metadata, "pandas"

    def _calc_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict, str]:
        """ストキャスティクス統一計算"""

        if TALIB_AVAILABLE and self.config.use_talib and self._has_ohlc(data):
            try:
                high = data["High"].values if "High" in data.columns else data["高値"].values
                low = data["Low"].values if "Low" in data.columns else data["安値"].values
                close = self._get_close_price(data).values

                k_percent, d_percent = talib.STOCH(
                    high, low, close,
                    fastk_period=k_period,
                    slowk_period=d_period,
                    slowd_period=d_period
                )

                values = {
                    "k_percent": k_percent,
                    "d_percent": d_percent
                }
                metadata = {"k_period": k_period, "d_period": d_period}
                self.performance_metrics["talib_usage"] += 1
                return values, metadata, "talib"
            except Exception:
                pass

        # フォールバック実装
        close = self._get_close_price(data)
        if self._has_ohlc(data):
            high = data["High"] if "High" in data.columns else data["高値"]
            low = data["Low"] if "Low" in data.columns else data["安値"]
            highest_high = high.rolling(k_period).max()
            lowest_low = low.rolling(k_period).min()
        else:
            # 単一価格データの場合の近似計算
            highest_high = close.rolling(k_period).max()
            lowest_low = close.rolling(k_period).min()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()

        values = {
            "k_percent": k_percent.values,
            "d_percent": d_percent.values
        }
        metadata = {"k_period": k_period, "d_period": d_period}

        self.performance_metrics["fallback_usage"] += 1
        return values, metadata, "pandas"

    def _calc_ichimoku(
        self,
        data: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        leading_span_b_period: int = 52,
        lagging_span_period: int = 26,
        **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict, str]:
        """一目均衡表統一計算"""

        if self._has_ohlc(data):
            high = data["High"] if "High" in data.columns else data["高値"]
            low = data["Low"] if "Low" in data.columns else data["安値"]
            close = self._get_close_price(data)
        else:
            # 単一価格データの場合の近似
            high = low = close = self._get_close_price(data)

        # 転換線
        conversion_line = (
            high.rolling(window=conversion_period).max() +
            low.rolling(window=conversion_period).min()
        ) / 2

        # 基準線
        base_line = (
            high.rolling(window=base_period).max() +
            low.rolling(window=base_period).min()
        ) / 2

        # 先行スパン1
        leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)

        # 先行スパン2
        leading_span_b = (
            (high.rolling(window=leading_span_b_period).max() +
             low.rolling(window=leading_span_b_period).min()) / 2
        ).shift(base_period)

        # 遅行スパン
        lagging_span = close.shift(-lagging_span_period)

        values = {
            "conversion_line": conversion_line.values,
            "base_line": base_line.values,
            "leading_span_a": leading_span_a.values,
            "leading_span_b": leading_span_b.values,
            "lagging_span": lagging_span.values,
            "cloud_top": np.maximum(leading_span_a.values, leading_span_b.values),
            "cloud_bottom": np.minimum(leading_span_a.values, leading_span_b.values)
        }

        metadata = {
            "conversion_period": conversion_period,
            "base_period": base_period,
            "leading_span_b_period": leading_span_b_period,
            "lagging_span_period": lagging_span_period
        }

        return values, metadata, "pandas"

    def _calc_fibonacci_retracement(
        self,
        data: pd.DataFrame,
        period: int = 100,
        **kwargs
    ) -> Tuple[Dict[str, float], Dict, str]:
        """フィボナッチリトレースメント統一計算"""
        close = self._get_close_price(data)

        # 期間内の最高値・最安値
        high_val = close.rolling(window=period).max().iloc[-1]
        low_val = close.rolling(window=period).min().iloc[-1]
        diff = high_val - low_val

        fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.618]

        values = {}
        for level in fibonacci_levels:
            values[f"fib_{level}"] = high_val - (diff * level)

        values.update({
            "high": high_val,
            "low": low_val,
            "range": diff
        })

        metadata = {"period": period, "levels_count": len(fibonacci_levels)}

        return values, metadata, "pandas"

    def _calc_default(self, data: pd.DataFrame, indicator_name: str, **kwargs) -> np.ndarray:
        """デフォルト指標計算（未実装指標用）"""
        close = self._get_close_price(data)
        logger.warning(f"未実装指標のデフォルト計算: {indicator_name}")
        return close.values

    # === ヘルパーメソッド ===

    def _get_close_price(self, data: pd.DataFrame) -> pd.Series:
        """終値データ取得統一処理"""
        if "Close" in data.columns:
            return data["Close"]
        elif "終値" in data.columns:
            return data["終値"]
        elif len(data.columns) == 1:
            return data.iloc[:, 0]
        else:
            # OHLC データの場合、最後の列を終値とみなす
            return data.iloc[:, -1]

    def _has_ohlc(self, data: pd.DataFrame) -> bool:
        """OHLC データかどうかの判定"""
        ohlc_patterns = [
            ["Open", "High", "Low", "Close"],
            ["始値", "高値", "安値", "終値"],
            ["open", "high", "low", "close"]
        ]

        for pattern in ohlc_patterns:
            if all(col in data.columns for col in pattern):
                return True

        # 列数での判定（4列以上でOHLCの可能性）
        return len(data.columns) >= 4

    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        timeframe: str,
        kwargs: Dict
    ) -> str:
        """キャッシュキー生成"""
        try:
            data_hash = hash(str(data.values.tobytes()))
            params_hash = hash(str(sorted(kwargs.items())))
            return f"{indicator_name}_{timeframe}_{data_hash}_{params_hash}"
        except Exception:
            # フォールバック
            return f"{indicator_name}_{timeframe}_{hash(str(data.shape))}"

    def _calculate_signals(self, values: Union[np.ndarray, Dict], indicator_name: str) -> np.ndarray:
        """シグナル計算統一処理"""
        try:
            if isinstance(values, dict):
                # 複数値の場合（MACD等）
                if "macd" in values and "signal" in values:
                    return np.where(values["macd"] > values["signal"], 1, -1)
                elif "k_percent" in values:
                    return np.where(values["k_percent"] < 20, 1, np.where(values["k_percent"] > 80, -1, 0))
                else:
                    # デフォルト: 最初の値を使用
                    first_key = list(values.keys())[0]
                    return self._calculate_simple_signal(values[first_key], indicator_name)
            else:
                return self._calculate_simple_signal(values, indicator_name)
        except Exception as e:
            logger.debug(f"シグナル計算エラー {indicator_name}: {e}")
            return np.zeros(len(values) if hasattr(values, '__len__') else 1)

    def _calculate_simple_signal(self, values: np.ndarray, indicator_name: str) -> np.ndarray:
        """単純シグナル計算"""
        if "rsi" in indicator_name.lower():
            return np.where(values < 30, 1, np.where(values > 70, -1, 0))
        elif any(ma_type in indicator_name.lower() for ma_type in ["sma", "ema", "ma"]):
            # 移動平均のトレンドシグナル
            if len(values) > 1:
                return np.where(values[1:] > values[:-1], 1, -1)
            return np.zeros_like(values)
        else:
            # デフォルト: ゼロライン基準
            return np.where(values > 0, 1, -1)

    def _evaluate_signal_strength(self, signals: np.ndarray) -> SignalStrength:
        """シグナル強度評価統一処理"""
        if len(signals) == 0:
            return SignalStrength.NEUTRAL

        recent_signals = signals[-20:] if len(signals) > 20 else signals
        avg_signal = np.mean(recent_signals)

        if avg_signal > 0.8:
            return SignalStrength.VERY_STRONG_BUY
        elif avg_signal > 0.5:
            return SignalStrength.STRONG_BUY
        elif avg_signal > 0.2:
            return SignalStrength.BUY
        elif avg_signal < -0.8:
            return SignalStrength.VERY_STRONG_SELL
        elif avg_signal < -0.5:
            return SignalStrength.STRONG_SELL
        elif avg_signal < -0.2:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL

    def _calculate_confidence(self, values: Union[np.ndarray, Dict], signals: np.ndarray) -> float:
        """信頼度計算統一処理"""
        try:
            if isinstance(values, dict):
                # 複数値の場合は最初の値を使用
                values = list(values.values())[0]

            if len(values) == 0 or len(signals) == 0:
                return 0.5

            # 値の安定性
            recent_values = values[-10:] if len(values) >= 10 else values
            if len(recent_values) > 1 and np.mean(np.abs(recent_values)) > 0:
                stability = 1.0 - (np.std(recent_values) / np.mean(np.abs(recent_values)))
                stability = max(0, min(1, stability))
            else:
                stability = 0.5

            # シグナルの一貫性
            recent_signals = signals[-10:] if len(signals) >= 10 else signals
            consistency = abs(np.mean(recent_signals))

            confidence = (stability + consistency) / 2
            return max(0.1, min(0.9, confidence))

        except Exception:
            return 0.5

    def _get_indicator_category(self, indicator_name: str) -> IndicatorCategory:
        """指標カテゴリ決定"""
        indicator_lower = indicator_name.lower()

        if any(trend in indicator_lower for trend in ["sma", "ema", "ma", "ichimoku", "adx"]):
            return IndicatorCategory.TREND
        elif any(momentum in indicator_lower for momentum in ["rsi", "macd", "stoch", "williams"]):
            return IndicatorCategory.MOMENTUM
        elif any(vol in indicator_lower for vol in ["bollinger", "atr", "keltner"]):
            return IndicatorCategory.VOLATILITY
        elif any(volume in indicator_lower for volume in ["obv", "vwap", "vroc"]):
            return IndicatorCategory.VOLUME
        elif any(sr in indicator_lower for sr in ["fibonacci", "pivot", "support", "resistance"]):
            return IndicatorCategory.SUPPORT_RESISTANCE
        elif any(pattern in indicator_lower for pattern in ["doji", "hammer", "pattern"]):
            return IndicatorCategory.PATTERN
        else:
            return IndicatorCategory.TREND  # デフォルト

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標一覧"""
        base_indicators = [
            "sma", "ema", "rsi", "macd", "bollinger_bands",
            "stochastic", "ichimoku", "fibonacci_retracement"
        ]

        # TA-Lib利用可能な場合の追加指標
        if TALIB_AVAILABLE and self.config.use_talib:
            talib_indicators = [
                "atr", "adx", "cci", "williams_r", "roc",
                "obv", "vwap", "keltner_channels"
            ]
            base_indicators.extend(talib_indicators)

        return sorted(base_indicators)

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要取得"""
        total_calcs = self.performance_metrics["total_calculations"]
        cache_hits = self.performance_metrics["cache_hits"]
        cache_misses = self.performance_metrics["cache_misses"]
        talib_usage = self.performance_metrics["talib_usage"]
        fallback_usage = self.performance_metrics["fallback_usage"]

        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        talib_rate = talib_usage / (talib_usage + fallback_usage) if (talib_usage + fallback_usage) > 0 else 0

        avg_times = {}
        for indicator, times in self.performance_metrics["calculation_times"].items():
            avg_times[indicator] = np.mean(times) if times else 0

        return {
            "total_calculations": total_calcs,
            "cache_hit_rate": cache_hit_rate,
            "talib_usage_rate": talib_rate,
            "fallback_usage": fallback_usage,
            "average_calculation_times": avg_times,
            "fastest_indicator": min(avg_times.keys(), key=lambda k: avg_times[k]) if avg_times else None,
            "slowest_indicator": max(avg_times.keys(), key=lambda k: avg_times[k]) if avg_times else None,
            "cache_size": len(self.cache),
            "available_indicators": len(self.get_available_indicators())
        }


# === 統合インターフェース ===

class TechnicalIndicatorsManager:
    """Issue #619対応: 統合テクニカル指標マネージャー"""

    def __init__(self, config: IndicatorConfig = None):
        self.config = config or IndicatorConfig()
        self.engine = ConsolidatedTechnicalIndicators(self.config)
        logger.info("統合テクニカル指標マネージャー初期化完了")

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str],
        symbols: List[str] = None,
        timeframe: str = "1D",
        **kwargs
    ) -> Dict[str, List[IndicatorResult]]:
        """統合指標計算実行"""
        return self.engine.calculate_indicators(data, indicators, symbols, timeframe, **kwargs)

    def get_available_indicators(self) -> List[str]:
        """利用可能な指標一覧"""
        return self.engine.get_available_indicators()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        return self.engine.get_performance_summary()

    def reset_cache(self) -> None:
        """キャッシュリセット"""
        self.engine.cache.clear()
        logger.info("統合指標キャッシュをリセットしました")


# === 便利関数 ===

def calculate_technical_indicators(
    data: pd.DataFrame,
    indicators: List[str],
    config: IndicatorConfig = None,
    **kwargs
) -> Dict[str, List[IndicatorResult]]:
    """Issue #619対応: テクニカル指標計算のヘルパー関数"""
    manager = TechnicalIndicatorsManager(config)
    return manager.calculate_indicators(data, indicators, **kwargs)


# === 後方互換性サポート ===

class AdvancedTechnicalIndicators:
    """後方互換性のための旧クラス"""

    def __init__(self):
        self._manager = TechnicalIndicatorsManager()
        logger.warning("AdvancedTechnicalIndicatorsは非推奨です。TechnicalIndicatorsManagerを使用してください")

    def __getattr__(self, name):
        # 旧メソッドの委譲
        return getattr(self._manager, name)


# グローバルインスタンス
_indicator_manager = None


def get_indicator_manager(config: IndicatorConfig = None) -> TechnicalIndicatorsManager:
    """統合テクニカル指標マネージャー取得"""
    global _indicator_manager
    if _indicator_manager is None:
        _indicator_manager = TechnicalIndicatorsManager(config)
    return _indicator_manager


# === 指標別便利関数（Issue #619対応） ===

def calculate_sma(data: pd.DataFrame, period: int = 20) -> np.ndarray:
    """SMA計算便利関数"""
    manager = get_indicator_manager()
    result = manager.calculate_indicators(data, ["sma"], period=period)
    return list(result.values())[0][0].values

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    """RSI計算便利関数"""
    manager = get_indicator_manager()
    result = manager.calculate_indicators(data, ["rsi"], period=period)
    return list(result.values())[0][0].values

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """MACD計算便利関数"""
    manager = get_indicator_manager()
    result = manager.calculate_indicators(
        data, ["macd"],
        fast_period=fast, slow_period=slow, signal_period=signal
    )
    return list(result.values())[0][0].values