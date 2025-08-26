#!/usr/bin/env python3
"""
分析ヘルパー機能モジュール
Issue #315: 高度テクニカル指標・ML機能拡張

簡易分析機能:
- 複合移動平均分析
- フィボナッチ分析
- 基本的なヘルパー関数
"""

import pandas as pd

from .data_structures import ComplexMAAnalysis, FibonacciAnalysis

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AnalysisHelpers:
    """分析ヘルパー機能クラス"""

    @staticmethod
    async def analyze_complex_ma_optimized(
        data: pd.DataFrame, symbol: str
    ) -> ComplexMAAnalysis:
        """複合移動平均分析（最適化版）"""
        try:
            close_prices = data["Close"]

            # 移動平均計算
            ma_5 = (
                close_prices.rolling(5).mean().iloc[-1]
                if len(close_prices) >= 5
                else 0.0
            )
            ma_25 = (
                close_prices.rolling(25).mean().iloc[-1]
                if len(close_prices) >= 25
                else 0.0
            )
            ma_75 = (
                close_prices.rolling(75).mean().iloc[-1]
                if len(close_prices) >= 75
                else 0.0
            )
            ma_200 = (
                close_prices.rolling(200).mean().iloc[-1]
                if len(close_prices) >= 200
                else 0.0
            )

            current_price = close_prices.iloc[-1]

            # 簡易アライメント判定
            ma_values = [ma for ma in [ma_5, ma_25, ma_75, ma_200] if ma > 0]
            if len(ma_values) >= 2:
                if all(
                    ma_values[i] >= ma_values[i + 1]
                    for i in range(len(ma_values) - 1)
                ):
                    ma_alignment = "bullish"
                elif all(
                    ma_values[i] <= ma_values[i + 1]
                    for i in range(len(ma_values) - 1)
                ):
                    ma_alignment = "bearish"
                else:
                    ma_alignment = "mixed"
            else:
                ma_alignment = "mixed"

            # ゴールデンクロス/デッドクロス判定
            golden_cross = (
                ma_5 > ma_25 and ma_25 > ma_75
                if ma_5 > 0 and ma_25 > 0 and ma_75 > 0
                else False
            )
            death_cross = (
                ma_5 < ma_25 and ma_25 < ma_75
                if ma_5 > 0 and ma_25 > 0 and ma_75 > 0
                else False
            )

            # 簡易シグナル生成
            if ma_alignment == "bullish" and current_price > ma_5:
                signal = "BUY"
                confidence = 0.7
            elif ma_alignment == "bearish" and current_price < ma_5:
                signal = "SELL"
                confidence = 0.7
            else:
                signal = "HOLD"
                confidence = 0.5

        except Exception as e:
            logger.warning(f"複合MA分析エラー {symbol}: {e}")
            ma_5 = ma_25 = ma_75 = ma_200 = 0.0
            current_price = data["Close"].iloc[-1] if len(data) > 0 else 0.0
            ma_alignment = "mixed"
            golden_cross = death_cross = False
            signal = "HOLD"
            confidence = 0.5

        return ComplexMAAnalysis(
            ma_5=ma_5,
            ma_25=ma_25,
            ma_75=ma_75,
            ma_200=ma_200,
            current_price=current_price,
            ma_alignment=ma_alignment,
            golden_cross=golden_cross,
            death_cross=death_cross,
            support_resistance={},
            trend_phase="accumulation",
            momentum_score=0.5,
            signal=signal,
            confidence=confidence,
            performance_score=0.5,
        )

    @staticmethod
    async def analyze_fibonacci_optimized(
        data: pd.DataFrame, symbol: str
    ) -> FibonacciAnalysis:
        """フィボナッチ分析（最適化版）"""
        try:
            high_price = data["High"].max()
            low_price = data["Low"].min()
            current_price = data["Close"].iloc[-1]

            # フィボナッチリトレースメントレベル計算
            price_range = high_price - low_price
            retracement_levels = {
                "0%": high_price,
                "23.6%": high_price - (price_range * 0.236),
                "38.2%": high_price - (price_range * 0.382),
                "50%": high_price - (price_range * 0.5),
                "61.8%": high_price - (price_range * 0.618),
                "100%": low_price,
            }

            # エクステンションレベル計算
            extension_levels = {
                "161.8%": high_price + (price_range * 0.618),
                "261.8%": high_price + (price_range * 1.618),
                "423.6%": high_price + (price_range * 3.236),
            }

            # 現在レベル判定
            current_level = "50%"
            for level, price in retracement_levels.items():
                if abs(current_price - price) < price_range * 0.05:
                    current_level = level
                    break

            # サポート・レジスタンスレベル
            support_level = min(
                [
                    price
                    for price in retracement_levels.values()
                    if price < current_price
                ],
                default=low_price,
            )
            resistance_level = min(
                [
                    price
                    for price in retracement_levels.values()
                    if price > current_price
                ],
                default=high_price,
            )

        except Exception as e:
            logger.warning(f"フィボナッチ分析エラー {symbol}: {e}")
            retracement_levels = extension_levels = {}
            current_level = "50%"
            support_level = data["Low"].min() if len(data) > 0 else 0.0
            resistance_level = data["High"].max() if len(data) > 0 else 0.0

        return FibonacciAnalysis(
            retracement_levels=retracement_levels,
            extension_levels=extension_levels,
            current_level=current_level,
            support_level=support_level,
            resistance_level=resistance_level,
            signal="HOLD",
            confidence=0.5,
            performance_score=0.5,
        )