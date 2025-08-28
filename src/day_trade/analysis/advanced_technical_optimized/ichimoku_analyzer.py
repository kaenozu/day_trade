#!/usr/bin/env python3
"""
一目均衡表総合分析モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化基盤活用:
- 高速計算（Issue #325）
- キャッシュ最適化（Issue #324）
- 高精度判定（Issue #322）
"""

import time
from dataclasses import asdict

import pandas as pd

from .core_system import CoreAdvancedTechnicalSystem
from .data_structures import IchimokuAnalysis
from .ichimoku_utils import IchimokuUtils

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


class IchimokuAnalyzer(CoreAdvancedTechnicalSystem):
    """
    一目均衡表総合分析（統合最適化版）

    高精度テクニカル分析:
    - 一目均衡表総合判定
    - 雲の詳細分析
    - TKクロス強度分析
    - 遅行スパン高精度分析
    """

    def __init__(self, **kwargs):
        """初期化"""
        super().__init__(**kwargs)
        self.ichimoku_utils = IchimokuUtils()

    async def analyze_ichimoku_cloud_optimized(
        self,
        data: pd.DataFrame,
        symbol: str,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
    ) -> IchimokuAnalysis:
        """
        一目均衡表総合分析（統合最適化版）

        統合最適化基盤活用:
        - 高速計算（Issue #325）
        - キャッシュ最適化（Issue #324）
        - 高精度判定（Issue #322）
        """
        start_time = time.time()
        self.performance_monitor.start_monitoring(f"ichimoku_{symbol}")

        try:
            # 統合キャッシュチェック
            if self.cache_enabled:
                cache_key = generate_unified_cache_key(
                    "advanced_ichimoku",
                    "optimized_analysis",
                    symbol,
                    {
                        "tenkan": tenkan_period,
                        "kijun": kijun_period,
                        "senkou_b": senkou_b_period,
                        "optimization": True,
                    },
                    time_bucket_minutes=self.cache_ttl_minutes,
                )
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    self.performance_stats["cache_hits"] += 1
                    return IchimokuAnalysis(**cached_result)

            # データ検証
            required_periods = (
                max(tenkan_period, kijun_period, senkou_b_period) + 30
            )
            if len(data) < required_periods:
                logger.warning(
                    f"データ不足 (一目最適化): {symbol} - {len(data)}日分"
                )
                return self._create_default_ichimoku_analysis()

            # Issue #325: 高速価格データ計算
            high = data["High"]
            low = data["Low"]
            close = data["Close"]

            # 一目均衡表ライン計算
            lines = self.ichimoku_utils.calculate_ichimoku_lines(
                high, low, close, tenkan_period, kijun_period, senkou_b_period
            )

            # 現在値取得（安全な取得）
            current_price = close.iloc[-1]
            current_values = self._get_safe_current_values(
                lines, current_price, kijun_period
            )

            # Issue #322: 高精度雲分析
            cloud_analysis = self.ichimoku_utils.calculate_cloud_analysis(
                current_values["senkou_a"],
                current_values["senkou_b"],
                current_price,
            )

            # 高度クロス分析
            tk_cross_strength = self.ichimoku_utils.analyze_tk_cross(
                lines["tenkan_sen"],
                lines["kijun_sen"],
                current_values["tenkan"],
                current_values["kijun"],
            )

            tk_cross = self._determine_tk_cross(
                current_values["tenkan"], current_values["kijun"]
            )

            # 遅行スパン高精度分析
            chikou_signal, chikou_signal_strength = (
                self.ichimoku_utils.analyze_chikou_signal(
                    current_values["chikou"], current_price
                )
            )

            # 総合判定（重み付き）
            total_signal = self.ichimoku_utils.calculate_weighted_ichimoku_signal(
                cloud_analysis["price_vs_cloud"],
                tk_cross,
                tk_cross_strength,
                chikou_signal,
                chikou_signal_strength,
                cloud_analysis["cloud_color"],
                cloud_analysis["cloud_thickness"],
                current_price,
            )

            # シグナル・信頼度決定
            overall_signal, confidence = self._determine_final_signal(
                total_signal
            )

            # トレンド強度計算
            trend_strength = self._calculate_trend_strength(
                total_signal,
                cloud_analysis["cloud_thickness"],
                current_price,
                tk_cross_strength,
            )

            # パフォーマンススコア計算
            performance_score = self._calculate_ichimoku_performance_score(
                total_signal,
                confidence,
                trend_strength,
                cloud_analysis["cloud_thickness"] / current_price,
            )

            analysis = IchimokuAnalysis(
                tenkan_sen=current_values["tenkan"],
                kijun_sen=current_values["kijun"],
                senkou_span_a=current_values["senkou_a"],
                senkou_span_b=current_values["senkou_b"],
                chikou_span=current_values["chikou"],
                current_price=current_price,
                cloud_thickness=cloud_analysis["cloud_thickness"],
                cloud_color=cloud_analysis["cloud_color"],
                price_vs_cloud=cloud_analysis["price_vs_cloud"],
                tk_cross=tk_cross,
                chikou_signal=chikou_signal,
                overall_signal=overall_signal,
                trend_strength=trend_strength,
                confidence=confidence,
                performance_score=performance_score,
            )

            # 統合キャッシュ保存
            if self.cache_enabled:
                self.cache_manager.put(
                    cache_key, asdict(analysis), priority=5.0
                )

            # パフォーマンス統計更新
            self._update_performance_stats(start_time)
            self.performance_monitor.stop_monitoring(f"ichimoku_{symbol}")

            logger.info(
                f"一目均衡表最適化分析完了: {symbol} "
                f"({time.time() - start_time:.3f}s)"
            )
            return analysis

        except Exception as e:
            logger.error(f"一目均衡表最適化分析エラー: {symbol} - {e}")
            self.performance_monitor.stop_monitoring(f"ichimoku_{symbol}")
            return self._create_default_ichimoku_analysis()

    def _get_safe_current_values(self, lines, current_price, kijun_period):
        """安全な現在値取得"""
        current_tenkan = (
            lines["tenkan_sen"].iloc[-1]
            if not pd.isna(lines["tenkan_sen"].iloc[-1])
            else current_price
        )
        current_kijun = (
            lines["kijun_sen"].iloc[-1]
            if not pd.isna(lines["kijun_sen"].iloc[-1])
            else current_price
        )
        current_senkou_a = (
            lines["senkou_span_a"].iloc[-1]
            if not pd.isna(lines["senkou_span_a"].iloc[-1])
            else current_price
        )
        current_senkou_b = (
            lines["senkou_span_b"].iloc[-1]
            if not pd.isna(lines["senkou_span_b"].iloc[-1])
            else current_price
        )

        # 遅行スパンの安全な取得
        chikou_index = max(0, len(lines["chikou_span"]) - kijun_period - 1)
        if (
            chikou_index < len(lines["chikou_span"])
            and not pd.isna(lines["chikou_span"].iloc[chikou_index])
        ):
            current_chikou = lines["chikou_span"].iloc[chikou_index]
        else:
            current_chikou = current_price

        return {
            "tenkan": current_tenkan,
            "kijun": current_kijun,
            "senkou_a": current_senkou_a,
            "senkou_b": current_senkou_b,
            "chikou": current_chikou,
        }

    def _determine_tk_cross(self, current_tenkan, current_kijun):
        """TKクロス方向決定"""
        if current_tenkan > current_kijun:
            return "bullish"
        elif current_tenkan < current_kijun:
            return "bearish"
        else:
            return "neutral"

    def _determine_final_signal(self, total_signal):
        """最終シグナル・信頼度決定"""
        if total_signal > 0.4:
            overall_signal = "BUY"
            confidence = min(0.95, 0.6 + abs(total_signal) * 0.8)
        elif total_signal < -0.4:
            overall_signal = "SELL"
            confidence = min(0.95, 0.6 + abs(total_signal) * 0.8)
        else:
            overall_signal = "HOLD"
            confidence = 0.5 + abs(total_signal) * 0.5

        return overall_signal, confidence

    def _calculate_trend_strength(
        self, total_signal, cloud_thickness, current_price, tk_cross_strength
    ):
        """トレンド強度計算"""
        return min(
            1.0,
            abs(total_signal)
            + (cloud_thickness / current_price * 50)
            + tk_cross_strength * 0.3,
        )

    def _update_performance_stats(self, start_time):
        """パフォーマンス統計更新"""
        processing_time = time.time() - start_time
        self.performance_stats["total_analyses"] += 1

    def _calculate_ichimoku_performance_score(
        self,
        total_signal: float,
        confidence: float,
        trend_strength: float,
        cloud_ratio: float,
    ) -> float:
        """一目均衡表パフォーマンススコア計算"""
        signal_score = abs(total_signal)
        confidence_score = confidence
        trend_score = trend_strength
        cloud_score = min(1.0, cloud_ratio * 20)  # 雲の厚さ

        performance_score = (
            signal_score * 0.3
            + confidence_score * 0.3
            + trend_score * 0.2
            + cloud_score * 0.2
        )

        return max(0, min(1, performance_score))