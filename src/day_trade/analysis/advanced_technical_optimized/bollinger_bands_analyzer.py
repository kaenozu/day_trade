#!/usr/bin/env python3
"""
Bollinger Bands変動率分析モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化基盤活用:
- キャッシュ最適化（Issue #324）
- ML特徴量最適化（Issue #325）
- 高精度ボラティリティ分析（Issue #322）
"""

import time
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pandas as pd

from .core_system import CoreAdvancedTechnicalSystem
from .data_structures import BollingerBandsAnalysis

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import generate_unified_cache_key
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    def generate_unified_cache_key(*args, **kwargs):
        return f"advanced_technical_{hash(str(args) + str(kwargs))}"


logger = get_context_logger(__name__)


class BollingerBandsAnalyzer(CoreAdvancedTechnicalSystem):
    """
    Bollinger Bands変動率分析（統合最適化版）

    高精度テクニカル分析:
    - Bollinger Bands変動率分析
    - スクイーズ判定
    - ブレイクアウト確率計算
    - トレンド強度分析
    """

    async def analyze_bollinger_bands_optimized(
        self, data: pd.DataFrame, symbol: str, period: int = 20, std_dev: float = 2.0
    ) -> BollingerBandsAnalysis:
        """
        Bollinger Bands変動率分析（統合最適化版）

        統合最適化基盤活用:
        - キャッシュ最適化（Issue #324）
        - ML特徴量最適化（Issue #325）
        """
        start_time = time.time()
        self.performance_monitor.start_monitoring(f"bollinger_bands_{symbol}")

        try:
            # Issue #324: 統合キャッシュチェック
            if self.cache_enabled:
                cache_key = generate_unified_cache_key(
                    "advanced_bollinger_bands",
                    "optimized_analysis",
                    symbol,
                    {"period": period, "std_dev": std_dev, "optimization": True},
                    time_bucket_minutes=self.cache_ttl_minutes,
                )
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    self.performance_stats["cache_hits"] += 1
                    logger.info(f"Bollinger Bands最適化キャッシュヒット: {symbol}")
                    return BollingerBandsAnalysis(**cached_result)

            # データ検証
            if len(data) < period + 20:  # 最適化のため余裕を持った検証
                logger.warning(f"データ不足: {symbol} - {len(data)}日分")
                return self._create_default_bb_analysis()

            # Issue #325: ML最適化による高速特徴量計算
            close_prices = data["Close"].copy()
            high_prices = data["High"].copy()
            low_prices = data["Low"].copy()
            volume = (
                data["Volume"].copy()
                if "Volume" in data.columns
                else pd.Series([1] * len(data))
            )

            # 最適化されたBollinger Bands計算
            sma = close_prices.rolling(
                window=period, min_periods=period // 2
            ).mean()
            std = close_prices.rolling(
                window=period, min_periods=period // 2
            ).std()

            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)

            # 現在値取得
            current_price = close_prices.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_middle = sma.iloc[-1]
            current_lower = lower_band.iloc[-1]

            # 高度化されたBB位置計算
            bb_width = current_upper - current_lower
            bb_position = (
                (current_price - current_lower) / bb_width 
                if bb_width > 0 else 0.5
            )

            # Issue #322: 多角的分析による高精度スクイーズ判定
            # ボラティリティ履歴分析
            bb_width_series = (upper_band - lower_band) / sma
            avg_bb_width = bb_width_series.rolling(window=50).mean().iloc[-1]
            current_bb_width = (current_upper - current_lower) / current_middle
            squeeze_ratio = (
                current_bb_width / avg_bb_width if avg_bb_width > 0 else 1.0
            )

            # 高度ボラティリティ分析
            recent_volatility = close_prices.pct_change().tail(20).std()
            historical_volatility = close_prices.pct_change().tail(100).std()
            volatility_ratio = (
                recent_volatility / historical_volatility
                if historical_volatility > 0
                else 1.0
            )

            # ボラティリティレジーム判定（高度化）
            if squeeze_ratio < 0.6 and volatility_ratio < 0.8:
                volatility_regime = "low"
                breakout_probability = min(
                    0.9, 0.4 + (0.6 - squeeze_ratio) * 1.5
                )
            elif squeeze_ratio > 1.4 or volatility_ratio > 1.3:
                volatility_regime = "high"
                breakout_probability = 0.15
            else:
                volatility_regime = "normal"
                breakout_probability = 0.35

            # Issue #325: ML強化トレンド分析
            trend_periods = [5, 10, 20]
            trend_strengths = []

            for period_len in trend_periods:
                if len(close_prices) >= period_len:
                    trend_direction = (
                        current_price - sma.iloc[-period_len]
                    ) / sma.iloc[-period_len]
                    trend_strengths.append(abs(trend_direction))

            trend_strength = (
                np.mean(trend_strengths) * 5 if trend_strengths else 0
            )
            trend_strength = min(1.0, trend_strength)

            # 高度シグナル生成（多角的要素統合）
            signal, confidence = self._generate_optimized_bb_signal(
                bb_position,
                squeeze_ratio,
                trend_strength,
                breakout_probability,
                volatility_ratio,
                current_price / current_middle,
            )

            # パフォーマンススコア計算
            performance_score = self._calculate_bb_performance_score(
                bb_position, squeeze_ratio, trend_strength, confidence
            )

            # 分析結果作成
            analysis = BollingerBandsAnalysis(
                upper_band=current_upper,
                middle_band=current_middle,
                lower_band=current_lower,
                current_price=current_price,
                bb_position=bb_position,
                squeeze_ratio=squeeze_ratio,
                volatility_regime=volatility_regime,
                breakout_probability=breakout_probability,
                trend_strength=trend_strength,
                signal=signal,
                confidence=confidence,
                performance_score=performance_score,
            )

            # Issue #324: 統合キャッシュ保存
            if self.cache_enabled:
                self.cache_manager.put(
                    cache_key,
                    asdict(analysis),
                    priority=5.0,  # 高優先度（高度分析結果）
                )

            # パフォーマンス統計更新
            processing_time = time.time() - start_time
            self.performance_stats["total_analyses"] += 1
            self.performance_stats["avg_processing_time"] = (
                self.performance_stats["avg_processing_time"] * 0.9
                + processing_time * 0.1
            )

            metrics = self.performance_monitor.get_metrics(
                f"bollinger_bands_{symbol}"
            )
            self.performance_monitor.stop_monitoring(f"bollinger_bands_{symbol}")

            logger.info(
                f"Bollinger Bands最適化分析完了: {symbol} ({processing_time:.3f}s)"
            )
            return analysis

        except Exception as e:
            logger.error(f"Bollinger Bands最適化分析エラー: {symbol} - {e}")
            self.performance_monitor.stop_monitoring(f"bollinger_bands_{symbol}")
            return self._create_default_bb_analysis()

    def _generate_optimized_bb_signal(
        self,
        bb_position: float,
        squeeze_ratio: float,
        trend_strength: float,
        breakout_probability: float,
        volatility_ratio: float,
        price_ma_ratio: float,
    ) -> Tuple[str, float]:
        """最適化されたBollinger Bandsシグナル生成"""

        signal_strength = 0
        base_confidence = 0.5

        # オーバーボート/オーバーソールド判定（高度化）
        if bb_position > 0.85:  # 強いオーバーボート
            if squeeze_ratio < 0.7:  # スクイーズ中 → ブレイクアウト警戒
                signal_strength = -0.3
                base_confidence = 0.6 + breakout_probability * 0.3
            else:  # 通常 → 売り
                signal_strength = -0.7
                base_confidence = 0.75
        elif bb_position > 0.75:  # 中程度オーバーボート
            signal_strength = -0.5
            base_confidence = 0.65
        elif bb_position < 0.15:  # 強いオーバーソールド
            if squeeze_ratio < 0.7:  # スクイーズ中
                signal_strength = 0.3
                base_confidence = 0.6 + breakout_probability * 0.3
            else:
                signal_strength = 0.7
                base_confidence = 0.75
        elif bb_position < 0.25:  # 中程度オーバーソールド
            signal_strength = 0.5
            base_confidence = 0.65

        # トレンド強度による調整
        signal_strength += (
            trend_strength * 0.2 * (1 if bb_position > 0.5 else -1)
        )

        # ボラティリティ比による調整
        if volatility_ratio < 0.7:  # 低ボラティリティ → レンジ相場
            signal_strength *= 0.8
        elif volatility_ratio > 1.3:  # 高ボラティリティ → 注意
            base_confidence *= 0.9

        # 価格MA比による調整
        if abs(price_ma_ratio - 1.0) > 0.05:  # MA乖離大
            signal_strength *= 1.1

        # 最終シグナル決定
        if signal_strength > 0.4:
            signal = "BUY"
            confidence = min(
                0.95, base_confidence + abs(signal_strength) * 0.2
            )
        elif signal_strength < -0.4:
            signal = "SELL"
            confidence = min(
                0.95, base_confidence + abs(signal_strength) * 0.2
            )
        else:
            signal = "HOLD"
            confidence = base_confidence

        return signal, confidence

    def _calculate_bb_performance_score(
        self,
        bb_position: float,
        squeeze_ratio: float,
        trend_strength: float,
        confidence: float,
    ) -> float:
        """Bollinger Bandsパフォーマンススコア計算"""

        # 位置スコア（極端な位置は高スコア）
        position_score = max(bb_position, 1 - bb_position) * 2 - 1

        # スクイーズスコア（スクイーズは高スコア）
        squeeze_score = max(0, 1 - squeeze_ratio)

        # トレンドスコア
        trend_score = trend_strength

        # 総合スコア
        performance_score = (
            position_score * 0.4
            + squeeze_score * 0.3
            + trend_score * 0.2
            + confidence * 0.1
        )

        return max(0, min(1, performance_score))