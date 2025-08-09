"""
マルチタイムフレーム分析統合システム（Strategy Pattern実装）

標準マルチタイムフレーム分析と最適化版を統一し、設定ベースで選択可能
"""

import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ..core.optimization_strategy import (
    OptimizationStrategy,
    OptimizationLevel,
    OptimizationConfig,
    optimization_strategy,
    get_optimized_implementation
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# オプショナル依存関係
try:
    from .technical_indicators_unified import TechnicalIndicatorsManager
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    logger.warning("統合テクニカル指標未利用 - 基本分析のみ")

try:
    from ..utils.performance_monitor import PerformanceMonitor
    from ..utils.unified_cache_manager import UnifiedCacheManager
    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    logger.info("最適化ユーティリティ未利用")

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TimeframeConfig:
    """タイムフレーム設定"""
    def __init__(self):
        self.timeframes = {
            "daily": {"period": "D", "name": "日足", "weight": 0.4, "min_periods": 30},
            "weekly": {"period": "W", "name": "週足", "weight": 0.35, "min_periods": 12},
            "monthly": {"period": "M", "name": "月足", "weight": 0.25, "min_periods": 6},
        }
        
        self.analysis_indicators = [
            "sma", "ema", "bollinger_bands", "rsi", "macd"
        ]
        
        self.trend_weights = {
            "strong_bullish": 2.0,
            "bullish": 1.0,
            "neutral": 0.0,
            "bearish": -1.0,
            "strong_bearish": -2.0
        }


class MultiTimeframeResult:
    """マルチタイムフレーム分析結果"""
    def __init__(self):
        self.timeframe_results = {}
        self.integrated_trend = "neutral"
        self.confidence_score = 0.0
        self.trend_consistency = 0.0
        self.analysis_time = 0.0
        self.strategy_used = ""
        self.cache_hits = 0
        self.total_indicators = 0


class MultiTimeframeAnalysisBase(OptimizationStrategy):
    """マルチタイムフレーム分析の基底戦略クラス"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.timeframe_config = TimeframeConfig()
        self.analysis_cache = {}
        
        # テクニカル指標マネージャーの初期化
        if TECHNICAL_INDICATORS_AVAILABLE:
            self.indicators_manager = TechnicalIndicatorsManager(config)
        else:
            self.indicators_manager = None
    
    def execute(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """マルチタイムフレーム分析の実行"""
        start_time = time.time()
        
        try:
            result = self._analyze_multi_timeframe(data, symbols, **kwargs)
            execution_time = time.time() - start_time
            result.analysis_time = execution_time
            result.strategy_used = self.get_strategy_name()
            
            self.record_execution(execution_time, True)
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.record_execution(execution_time, False)
            logger.error(f"マルチタイムフレーム分析エラー: {e}")
            raise
    
    def _analyze_multi_timeframe(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """マルチタイムフレーム分析の実装"""
        result = MultiTimeframeResult()
        
        # 各タイムフレームで分析
        for tf_name, tf_config in self.timeframe_config.timeframes.items():
            try:
                # データのリサンプリング
                resampled_data = self._resample_to_timeframe(data, tf_config["period"])
                
                if len(resampled_data) < tf_config["min_periods"]:
                    logger.warning(f"{tf_name}タイムフレーム: データ不足 ({len(resampled_data)}行)")
                    continue
                
                # タイムフレーム別分析
                tf_result = self._analyze_timeframe(resampled_data, tf_name, tf_config)
                result.timeframe_results[tf_name] = tf_result
                
            except Exception as e:
                logger.error(f"{tf_name}タイムフレーム分析エラー: {e}")
                continue
        
        # 統合トレンド判定
        if result.timeframe_results:
            result.integrated_trend, result.confidence_score = self._integrate_trends(result.timeframe_results)
            result.trend_consistency = self._calculate_trend_consistency(result.timeframe_results)
        
        return result
    
    def _resample_to_timeframe(self, data: pd.DataFrame, period: str) -> pd.DataFrame:
        """指定タイムフレームにリサンプリング"""
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'Date' in data.columns:
                data = data.set_index('Date')
            else:
                # フォールバック: インデックスを日付として解釈
                data.index = pd.to_datetime(data.index)
        
        # OHLCV形式への対応
        agg_dict = {}
        
        if '始値' in data.columns or 'Open' in data.columns:
            open_col = '始値' if '始値' in data.columns else 'Open'
            agg_dict[open_col] = 'first'
        
        if '高値' in data.columns or 'High' in data.columns:
            high_col = '高値' if '高値' in data.columns else 'High'
            agg_dict[high_col] = 'max'
        
        if '安値' in data.columns or 'Low' in data.columns:
            low_col = '安値' if '安値' in data.columns else 'Low'
            agg_dict[low_col] = 'min'
        
        if '終値' in data.columns or 'Close' in data.columns:
            close_col = '終値' if '終値' in data.columns else 'Close'
            agg_dict[close_col] = 'last'
        
        if '出来高' in data.columns or 'Volume' in data.columns:
            volume_col = '出来高' if '出来高' in data.columns else 'Volume'
            agg_dict[volume_col] = 'sum'
        
        # その他のカラムは最終値を使用
        for col in data.columns:
            if col not in agg_dict:
                agg_dict[col] = 'last'
        
        return data.resample(period).agg(agg_dict).dropna()
    
    def _analyze_timeframe(self, data: pd.DataFrame, tf_name: str, tf_config: Dict[str, Any]) -> Dict[str, Any]:
        """単一タイムフレームの分析"""
        tf_result = {
            "timeframe": tf_name,
            "data_points": len(data),
            "indicators": {},
            "trend": "neutral",
            "trend_strength": 0.0
        }
        
        # テクニカル指標計算
        if self.indicators_manager:
            try:
                indicators_result = self.indicators_manager.calculate_indicators(
                    data, self.timeframe_config.analysis_indicators
                )
                
                for indicator_name, indicator_result in indicators_result.items():
                    tf_result["indicators"][indicator_name] = self._extract_indicator_values(indicator_result)
                
            except Exception as e:
                logger.warning(f"{tf_name}指標計算エラー: {e}")
        else:
            # 基本指標のフォールバック
            tf_result["indicators"] = self._calculate_basic_indicators(data)
        
        # トレンド判定
        tf_result["trend"], tf_result["trend_strength"] = self._determine_trend(tf_result["indicators"], data)
        
        return tf_result
    
    def _extract_indicator_values(self, indicator_result) -> Dict[str, Any]:
        """指標結果から値を抽出"""
        if hasattr(indicator_result, 'values') and isinstance(indicator_result.values, dict):
            return indicator_result.values
        else:
            return {"raw_result": str(indicator_result)}
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """基本指標のフォールバック計算"""
        close_col = '終値' if '終値' in data.columns else 'Close'
        close_prices = data[close_col]
        
        indicators = {}
        
        # 単純移動平均
        indicators["sma_20"] = close_prices.rolling(20).mean().iloc[-1]
        indicators["sma_50"] = close_prices.rolling(50).mean().iloc[-1] if len(close_prices) >= 50 else np.nan
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators["rsi"] = rsi.iloc[-1] if not rsi.empty else 50
        
        # 価格変動率
        if len(close_prices) >= 20:
            indicators["price_change_20d"] = (close_prices.iloc[-1] / close_prices.iloc[-20] - 1) * 100
        
        return indicators
    
    def _determine_trend(self, indicators: Dict[str, Any], data: pd.DataFrame) -> Tuple[str, float]:
        """トレンド判定"""
        trend_signals = []
        
        # SMA基準トレンド
        if "sma_20" in indicators and "sma_50" in indicators:
            sma_20 = indicators["sma_20"]
            sma_50 = indicators["sma_50"]
            if not np.isnan(sma_20) and not np.isnan(sma_50):
                if sma_20 > sma_50 * 1.02:  # 2%以上の差
                    trend_signals.append(1)
                elif sma_20 < sma_50 * 0.98:
                    trend_signals.append(-1)
                else:
                    trend_signals.append(0)
        
        # RSI基準トレンド
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi > 70:
                trend_signals.append(-0.5)  # 買われすぎ
            elif rsi < 30:
                trend_signals.append(0.5)   # 売られすぎ
            else:
                trend_signals.append(0)
        
        # 価格変動基準
        if "price_change_20d" in indicators:
            change = indicators["price_change_20d"]
            if change > 5:
                trend_signals.append(1)
            elif change < -5:
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
        
        # 統合判定
        if not trend_signals:
            return "neutral", 0.0
        
        avg_signal = np.mean(trend_signals)
        strength = abs(avg_signal)
        
        if avg_signal > 0.5:
            trend = "strong_bullish" if avg_signal > 1.0 else "bullish"
        elif avg_signal < -0.5:
            trend = "strong_bearish" if avg_signal < -1.0 else "bearish"
        else:
            trend = "neutral"
        
        return trend, strength
    
    def _integrate_trends(self, timeframe_results: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
        """各タイムフレームのトレンドを統合"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for tf_name, tf_result in timeframe_results.items():
            if tf_name in self.timeframe_config.timeframes:
                weight = self.timeframe_config.timeframes[tf_name]["weight"]
                trend = tf_result.get("trend", "neutral")
                trend_weight = self.timeframe_config.trend_weights.get(trend, 0.0)
                
                weighted_score += trend_weight * weight
                total_weight += weight
        
        if total_weight == 0:
            return "neutral", 0.0
        
        final_score = weighted_score / total_weight
        confidence = min(abs(final_score), 1.0)
        
        # トレンド決定
        if final_score > 0.7:
            integrated_trend = "strong_bullish"
        elif final_score > 0.3:
            integrated_trend = "bullish"
        elif final_score < -0.7:
            integrated_trend = "strong_bearish"
        elif final_score < -0.3:
            integrated_trend = "bearish"
        else:
            integrated_trend = "neutral"
        
        return integrated_trend, confidence
    
    def _calculate_trend_consistency(self, timeframe_results: Dict[str, Dict[str, Any]]) -> float:
        """トレンド一貫性の計算"""
        trends = [result.get("trend", "neutral") for result in timeframe_results.values()]
        
        if len(trends) <= 1:
            return 1.0
        
        # 各トレンドを数値化
        trend_values = [self.timeframe_config.trend_weights.get(trend, 0.0) for trend in trends]
        
        # 標準偏差を基準とした一貫性スコア
        if len(trend_values) > 1:
            std_dev = np.std(trend_values)
            # 標準偏差が小さいほど一貫性が高い（最大2.0の範囲で正規化）
            consistency = max(0.0, 1.0 - (std_dev / 2.0))
        else:
            consistency = 1.0
        
        return consistency


@optimization_strategy("multi_timeframe_analysis", OptimizationLevel.STANDARD)
class StandardMultiTimeframeAnalysis(MultiTimeframeAnalysisBase):
    """標準マルチタイムフレーム分析実装"""
    
    def get_strategy_name(self) -> str:
        return "標準マルチタイムフレーム分析"


@optimization_strategy("multi_timeframe_analysis", OptimizationLevel.OPTIMIZED)
class OptimizedMultiTimeframeAnalysis(MultiTimeframeAnalysisBase):
    """最適化マルチタイムフレーム分析実装"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
        # 最適化機能の初期化
        if OPTIMIZATION_UTILS_AVAILABLE:
            self.cache_manager = UnifiedCacheManager()
            self.performance_monitor = PerformanceMonitor()
        else:
            self.cache_manager = None
            self.performance_monitor = None
            
        logger.info("最適化マルチタイムフレーム分析初期化完了")
    
    def get_strategy_name(self) -> str:
        return "最適化マルチタイムフレーム分析"
    
    def _analyze_multi_timeframe(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """並列処理による高速マルチタイムフレーム分析"""
        if not self.config.parallel_processing or len(self.timeframe_config.timeframes) <= 2:
            # 標準処理にフォールバック
            return super()._analyze_multi_timeframe(data, symbols, **kwargs)
        
        return self._analyze_multi_timeframe_parallel(data, symbols, **kwargs)
    
    def _analyze_multi_timeframe_parallel(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """並列マルチタイムフレーム分析"""
        result = MultiTimeframeResult()
        
        with ThreadPoolExecutor(max_workers=min(4, len(self.timeframe_config.timeframes))) as executor:
            # 並列タスク投入
            future_to_timeframe = {}
            
            for tf_name, tf_config in self.timeframe_config.timeframes.items():
                future = executor.submit(self._analyze_single_timeframe_cached, data, tf_name, tf_config)
                future_to_timeframe[future] = tf_name
            
            # 結果収集
            for future in as_completed(future_to_timeframe):
                tf_name = future_to_timeframe[future]
                try:
                    tf_result = future.result(timeout=60)
                    if tf_result:
                        result.timeframe_results[tf_name] = tf_result
                        logger.debug(f"{tf_name}タイムフレーム分析完了")
                except Exception as e:
                    logger.error(f"{tf_name}タイムフレーム並列分析エラー: {e}")
        
        # 統合トレンド判定
        if result.timeframe_results:
            result.integrated_trend, result.confidence_score = self._integrate_trends(result.timeframe_results)
            result.trend_consistency = self._calculate_trend_consistency(result.timeframe_results)
        
        return result
    
    def _analyze_single_timeframe_cached(self, data: pd.DataFrame, tf_name: str, tf_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """キャッシュ機能付き単一タイムフレーム分析"""
        # キャッシュキー生成
        cache_key = None
        if self.cache_manager:
            import hashlib
            data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:16]
            cache_key = f"mtf_{tf_name}_{data_hash}"
            
            # キャッシュから取得試行
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"{tf_name}タイムフレーム分析キャッシュヒット")
                return cached_result
        
        # リサンプリング
        try:
            resampled_data = self._resample_to_timeframe(data, tf_config["period"])
            
            if len(resampled_data) < tf_config["min_periods"]:
                logger.warning(f"{tf_name}タイムフレーム: データ不足")
                return None
            
            # 分析実行
            tf_result = self._analyze_timeframe(resampled_data, tf_name, tf_config)
            
            # キャッシュに保存
            if self.cache_manager and cache_key:
                self.cache_manager.set(cache_key, tf_result)
            
            return tf_result
            
        except Exception as e:
            logger.error(f"{tf_name}単一タイムフレーム分析エラー: {e}")
            return None
    
    async def analyze_multi_timeframe_async(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """非同期マルチタイムフレーム分析"""
        result = MultiTimeframeResult()
        
        # 非同期タスク作成
        tasks = []
        for tf_name, tf_config in self.timeframe_config.timeframes.items():
            task = asyncio.create_task(self._analyze_timeframe_async(data, tf_name, tf_config))
            tasks.append((task, tf_name))
        
        # 並列実行
        for task, tf_name in tasks:
            try:
                tf_result = await task
                if tf_result:
                    result.timeframe_results[tf_name] = tf_result
            except Exception as e:
                logger.error(f"{tf_name}非同期分析エラー: {e}")
        
        # 統合判定
        if result.timeframe_results:
            result.integrated_trend, result.confidence_score = self._integrate_trends(result.timeframe_results)
            result.trend_consistency = self._calculate_trend_consistency(result.timeframe_results)
        
        return result
    
    async def _analyze_timeframe_async(self, data: pd.DataFrame, tf_name: str, tf_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """非同期単一タイムフレーム分析"""
        loop = asyncio.get_event_loop()
        try:
            # ブロッキング処理を別スレッドで実行
            result = await loop.run_in_executor(
                None, self._analyze_single_timeframe_cached, data, tf_name, tf_config
            )
            return result
        except Exception as e:
            logger.error(f"{tf_name}非同期タイムフレーム分析エラー: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計取得"""
        if self.cache_manager:
            return {
                "cache_enabled": True,
                "cache_stats": "N/A"  # UnifiedCacheManagerのAPIに依存
            }
        return {"cache_enabled": False}


# 統合インターフェース
class MultiTimeframeAnalysisManager:
    """マルチタイムフレーム分析統合マネージャー"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig.from_env()
        self._strategy = None
    
    def get_strategy(self) -> OptimizationStrategy:
        """現在の戦略を取得"""
        if self._strategy is None:
            self._strategy = get_optimized_implementation("multi_timeframe_analysis", self.config)
        return self._strategy
    
    def analyze_multi_timeframe(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """マルチタイムフレーム分析実行"""
        strategy = self.get_strategy()
        return strategy.execute(data, symbols, **kwargs)
    
    async def analyze_async(self, data: pd.DataFrame, symbols: Optional[List[str]] = None, **kwargs) -> MultiTimeframeResult:
        """非同期マルチタイムフレーム分析"""
        strategy = self.get_strategy()
        if hasattr(strategy, 'analyze_multi_timeframe_async'):
            return await strategy.analyze_multi_timeframe_async(data, symbols, **kwargs)
        else:
            # 同期実行にフォールバック
            return strategy.execute(data, symbols, **kwargs)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        if self._strategy:
            return self._strategy.get_performance_metrics()
        return {}


# 便利関数
def analyze_multi_timeframe(
    data: pd.DataFrame,
    symbols: Optional[List[str]] = None,
    config: Optional[OptimizationConfig] = None,
    **kwargs
) -> MultiTimeframeResult:
    """マルチタイムフレーム分析のヘルパー関数"""
    manager = MultiTimeframeAnalysisManager(config)
    return manager.analyze_multi_timeframe(data, symbols, **kwargs)