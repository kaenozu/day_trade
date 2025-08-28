#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Technical Indicators
ボラティリティ系技術指標計算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple


class VolatilityIndicators:
    """ボラティリティ系指標計算クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volatility_regimes = {
            "超低ボラ": (0, 10),
            "低ボラ": (10, 20),
            "通常ボラ": (20, 35),
            "高ボラ": (35, 50),
            "超高ボラ": (50, 100)
        }

    async def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """ボラティリティ系指標計算"""
        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']

            # ボリンジャーバンド
            bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(closes)
            indicators['BB_Middle'] = bb_middle
            indicators['BB_Upper'] = bb_upper
            indicators['BB_Lower'] = bb_lower
            try:
                indicators['BB_Width'] = float((bb_upper - bb_lower) / bb_middle * 100)
                indicators['BB_Position'] = float(
                    (closes.iloc[-1] - bb_lower) / (bb_upper - bb_lower) * 100
                )
            except (TypeError, ValueError):
                indicators['BB_Width'] = 2.0  # デフォルト値
                indicators['BB_Position'] = 50.0

            # ATR（真の値幅）
            indicators['ATR_14'] = self._calculate_atr(highs, lows, closes, 14)
            indicators['ATR_21'] = self._calculate_atr(highs, lows, closes, 21)

            # 標準偏差
            indicators['StdDev_20'] = float(closes.rolling(20).std())

            # Keltner Channel
            kc_middle, kc_upper, kc_lower = self._calculate_keltner_channel(highs, lows, closes)
            indicators['KC_Middle'] = kc_middle
            indicators['KC_Upper'] = kc_upper
            indicators['KC_Lower'] = kc_lower

            # 歴史的ボラティリティ
            returns = closes.pct_change().dropna()
            indicators['Historical_Vol'] = float(returns.std() * np.sqrt(252) * 100)

            # VIX風指標（ATRベース）
            indicators['VIX_Proxy'] = float(indicators['ATR_14'] / closes.iloc[-1] * 100)

        except Exception as e:
            self.logger.error(f"Volatility indicators calculation error: {e}")

        return indicators

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2) -> Tuple[float, float, float]:
        """ボリンジャーバンド計算"""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return float(middle.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
        except:
            last_price = float(prices.iloc[-1])
            return last_price, last_price * 1.02, last_price * 0.98

    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 14) -> float:
        """ATR計算"""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0

    def _calculate_keltner_channel(self, highs: pd.Series, lows: pd.Series, 
                                  closes: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """Keltner Channel計算"""
        try:
            middle = closes.rolling(period).mean()
            atr = self._calculate_atr(highs, lows, closes, period)

            upper = middle + 2 * atr
            lower = middle - 2 * atr

            return float(middle.iloc[-1]), float(upper), float(lower)
        except:
            price = float(closes.iloc[-1])
            return price, price * 1.02, price * 0.98

    def determine_volatility_regime(self, indicators: Dict[str, float]) -> str:
        """ボラティリティ局面判定"""
        try:
            if 'Historical_Vol' in indicators:
                vol = indicators['Historical_Vol']
                for regime, (low, high) in self.volatility_regimes.items():
                    if low <= vol < high:
                        return regime

            return "通常ボラ"

        except Exception as e:
            self.logger.error(f"Volatility regime determination error: {e}")
            return "通常ボラ"