#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Technical Indicators
モメンタム系技術指標計算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple


class MomentumIndicators:
    """モメンタム系指標計算クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """モメンタム系指標計算"""
        indicators = {}

        try:
            closes = df['Close']
            highs = df['High']
            lows = df['Low']
            volumes = df['Volume']

            # RSI群
            indicators['RSI_14'] = self._calculate_rsi(closes, 14)
            indicators['RSI_21'] = self._calculate_rsi(closes, 21)

            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes)
            indicators['Stoch_K'] = stoch_k
            indicators['Stoch_D'] = stoch_d

            # Williams %R
            indicators['Williams_R'] = self._calculate_williams_r(highs, lows, closes)

            # CCI（商品チャンネル指数）
            indicators['CCI'] = self._calculate_cci(highs, lows, closes)

            # ROC（変化率）
            indicators['ROC_12'] = float(
                (closes.iloc[-1] / closes.iloc[-13] - 1) * 100 if len(closes) >= 13 else 0
            )

            # Money Flow Index
            indicators['MFI'] = self._calculate_mfi(highs, lows, closes, volumes)

            # Ultimate Oscillator
            indicators['UO'] = self._calculate_ultimate_oscillator(highs, lows, closes)

            # Awesome Oscillator
            indicators['AO'] = self._calculate_awesome_oscillator(highs, lows)

        except Exception as e:
            self.logger.error(f"Momentum indicators calculation error: {e}")

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI計算"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0

    def _calculate_stochastic(self, highs: pd.Series, lows: pd.Series, 
                             closes: pd.Series, k_period: int = 14, 
                             d_period: int = 3) -> Tuple[float, float]:
        """ストキャスティクス計算"""
        try:
            lowest_low = lows.rolling(window=k_period).min()
            highest_high = highs.rolling(window=k_period).max()
            k_percent = 100 * ((closes - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return float(k_percent.iloc[-1]), float(d_percent.iloc[-1])
        except:
            return 50.0, 50.0

    def _calculate_williams_r(self, highs: pd.Series, lows: pd.Series, 
                             closes: pd.Series, period: int = 14) -> float:
        """Williams %R計算"""
        try:
            highest_high = highs.rolling(window=period).max()
            lowest_low = lows.rolling(window=period).min()
            wr = -100 * ((highest_high - closes) / (highest_high - lowest_low))
            return float(wr.iloc[-1])
        except:
            return -50.0

    def _calculate_cci(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 20) -> float:
        """CCI計算"""
        try:
            tp = (highs + lows + closes) / 3
            sma = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: pd.Series(x).mad())
            cci = (tp - sma) / (0.015 * mad)
            return float(cci.iloc[-1])
        except:
            return 0.0

    def _calculate_mfi(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, volumes: pd.Series, 
                      period: int = 14) -> float:
        """MFI計算"""
        try:
            tp = (highs + lows + closes) / 3
            mf = tp * volumes

            pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()

            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            return float(mfi.iloc[-1])
        except:
            return 50.0

    def _calculate_ultimate_oscillator(self, highs: pd.Series, lows: pd.Series, 
                                      closes: pd.Series) -> float:
        """Ultimate Oscillator計算（簡易版）"""
        try:
            # 3つの期間のモメンタム平均
            periods = [7, 14, 28]
            values = []

            for period in periods:
                if len(closes) >= period:
                    roc = (closes.iloc[-1] / closes.iloc[-period] - 1) * 100
                    values.append(roc)

            if values:
                uo = sum(values) / len(values)
                return float(max(-100, min(100, uo + 50)))  # -100~100を0~100に変換

            return 50.0
        except:
            return 50.0

    def _calculate_awesome_oscillator(self, highs: pd.Series, lows: pd.Series) -> float:
        """Awesome Oscillator計算"""
        try:
            median_price = (highs + lows) / 2
            ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
            return float(ao.iloc[-1])
        except:
            return 0.0

    def calculate_momentum_score(self, indicators: Dict[str, float]) -> float:
        """モメンタムスコア計算"""
        try:
            score = 0.0
            count = 0

            # RSI評価
            if 'RSI_14' in indicators:
                rsi = indicators['RSI_14']
                rsi_score = (rsi - 50) * 2  # -100~100に正規化
                score += rsi_score
                count += 1

            # Stochastic評価
            if 'Stoch_K' in indicators:
                stoch = indicators['Stoch_K']
                stoch_score = (stoch - 50) * 2
                score += stoch_score
                count += 1

            # Williams %R評価
            if 'Williams_R' in indicators:
                wr = indicators['Williams_R']
                wr_score = (wr + 50) * 2  # -100~0を-100~100に変換
                score += wr_score
                count += 1

            # ROC評価
            if 'ROC_12' in indicators:
                roc = indicators['ROC_12']
                roc_score = max(-50, min(50, roc * 2))
                score += roc_score
                count += 1

            return (score / count) if count > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {e}")
            return 0.0