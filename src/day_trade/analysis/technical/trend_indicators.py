#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend Technical Indicators
トレンド系技術指標計算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple


class TrendIndicators:
    """トレンド系指標計算クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """トレンド系指標計算"""
        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']

            # 移動平均群
            indicators['SMA_5'] = float(closes.rolling(5).mean().iloc[-1])
            indicators['SMA_20'] = float(closes.rolling(20).mean().iloc[-1])
            indicators['SMA_50'] = float(
                closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
            )

            indicators['EMA_12'] = float(closes.ewm(span=12).mean().iloc[-1])
            indicators['EMA_26'] = float(closes.ewm(span=26).mean().iloc[-1])

            # MACD
            macd_line = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = float(macd_line.iloc[-1])
            indicators['MACD_Signal'] = float(signal_line.iloc[-1])
            indicators['MACD_Histogram'] = float(macd_line.iloc[-1] - signal_line.iloc[-1])

            # ADX（方向性指数）
            indicators['ADX'] = self._calculate_adx(highs, lows, closes)

            # パラボリックSAR
            indicators['PSAR'] = self._calculate_psar(highs, lows, closes)

            # 一目均衡表
            ichimoku = self._calculate_ichimoku(highs, lows, closes)
            indicators.update(ichimoku)

            # SuperTrend
            indicators['SuperTrend'] = self._calculate_supertrend(highs, lows, closes)

        except Exception as e:
            self.logger.error(f"Trend indicators calculation error: {e}")

        return indicators

    def _calculate_adx(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 14) -> float:
        """ADX計算（簡易版）"""
        try:
            # True Rangeベースの簡易ADX
            atr = self._calculate_atr(highs, lows, closes, period)
            return min(100, max(0, atr / closes.iloc[-1] * 100 * 5))
        except:
            return 25.0

    def _calculate_psar(self, highs: pd.Series, lows: pd.Series, 
                       closes: pd.Series) -> float:
        """パラボリックSAR計算（簡易版）"""
        try:
            # 簡易PSAR（実際の実装はより複雑）
            recent_low = lows.iloc[-10:].min()
            recent_high = highs.iloc[-10:].max()

            if closes.iloc[-1] > closes.iloc[-5]:  # 上昇トレンド
                return float(recent_low * 0.98)
            else:  # 下降トレンド
                return float(recent_high * 1.02)
        except:
            return float(closes.iloc[-1])

    def _calculate_ichimoku(self, highs: pd.Series, lows: pd.Series, 
                           closes: pd.Series) -> Dict[str, float]:
        """一目均衡表計算"""
        try:
            # 転換線 (9日間の高値と安値の平均)
            tenkan = (highs.rolling(9).max() + lows.rolling(9).min()) / 2

            # 基準線 (26日間の高値と安値の平均)
            kijun = (highs.rolling(26).max() + lows.rolling(26).min()) / 2

            return {
                'Ichimoku_Tenkan': float(tenkan.iloc[-1] if not tenkan.empty else closes.iloc[-1]),
                'Ichimoku_Kijun': float(kijun.iloc[-1] if not kijun.empty else closes.iloc[-1])
            }
        except:
            return {
                'Ichimoku_Tenkan': float(closes.iloc[-1]), 
                'Ichimoku_Kijun': float(closes.iloc[-1])
            }

    def _calculate_supertrend(self, highs: pd.Series, lows: pd.Series, 
                             closes: pd.Series) -> float:
        """SuperTrend計算（簡易版）"""
        try:
            atr = self._calculate_atr(highs, lows, closes)
            hl2 = (highs + lows) / 2

            upper_band = hl2.iloc[-1] + 2 * atr
            lower_band = hl2.iloc[-1] - 2 * atr

            if closes.iloc[-1] > closes.iloc[-5]:  # 上昇
                return float(lower_band)
            else:  # 下降
                return float(upper_band)
        except:
            return float(closes.iloc[-1])

    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 14) -> float:
        """ATR計算（サポートメソッド）"""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0

    def calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """トレンド強度計算"""
        try:
            strength = 0.0

            # MACD強度
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                macd_strength = (indicators['MACD'] - indicators['MACD_Signal']) * 10
                strength += max(-30, min(30, macd_strength))

            # ADX強度
            if 'ADX' in indicators:
                adx = indicators['ADX']
                if adx > 25:
                    strength += min(25, adx)
                else:
                    strength += adx / 2

            # 移動平均の位置関係
            if all(key in indicators for key in ['SMA_5', 'SMA_20', 'SMA_50']):
                if indicators['SMA_5'] > indicators['SMA_20'] > indicators['SMA_50']:
                    strength += 20  # 上昇トレンド
                elif indicators['SMA_5'] < indicators['SMA_20'] < indicators['SMA_50']:
                    strength -= 20  # 下降トレンド

            return max(-100, min(100, strength))

        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.0