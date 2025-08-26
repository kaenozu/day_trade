#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volume Technical Indicators
出来高系技術指標計算
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict


class VolumeIndicators:
    """出来高系指標計算クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """出来高系指標計算"""
        indicators = {}

        try:
            closes = df['Close']
            volumes = df['Volume']

            # 出来高移動平均
            indicators['Volume_SMA_20'] = float(volumes.rolling(20).mean().iloc[-1])
            indicators['Volume_Ratio'] = float(
                volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1]
            )

            # OBV（オンバランスボリューム）
            indicators['OBV'] = self._calculate_obv(closes, volumes)

            # A/D Line（集積/配布ライン）
            indicators['AD_Line'] = self._calculate_ad_line(df)

            # Chaikin Money Flow
            indicators['CMF'] = self._calculate_cmf(df)

            # Volume Oscillator
            vol_short = volumes.rolling(5).mean()
            vol_long = volumes.rolling(20).mean()
            indicators['Volume_Osc'] = float(
                (vol_short.iloc[-1] - vol_long.iloc[-1]) / vol_long.iloc[-1] * 100
            )

            # Price Volume Trend
            indicators['PVT'] = self._calculate_pvt(closes, volumes)

            # VWAP（出来高加重平均価格）
            indicators['VWAP'] = self._calculate_vwap(df)

        except Exception as e:
            self.logger.error(f"Volume indicators calculation error: {e}")

        return indicators

    def _calculate_obv(self, closes: pd.Series, volumes: pd.Series) -> float:
        """OBV計算"""
        try:
            price_change = closes.diff()
            obv = volumes.copy()
            obv[price_change < 0] = -obv[price_change < 0]
            obv[price_change == 0] = 0
            return float(obv.cumsum().iloc[-1])
        except:
            return 0.0

    def _calculate_ad_line(self, df: pd.DataFrame) -> float:
        """A/D Line計算"""
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (
                df['High'] - df['Low']
            )
            ad_line = (clv * df['Volume']).cumsum()
            return float(ad_line.iloc[-1])
        except:
            return 0.0

    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> float:
        """Chaikin Money Flow計算"""
        try:
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (
                df['High'] - df['Low']
            )
            cmf = (clv * df['Volume']).rolling(period).sum() / df['Volume'].rolling(
                period
            ).sum()
            return float(cmf.iloc[-1])
        except:
            return 0.0

    def _calculate_pvt(self, closes: pd.Series, volumes: pd.Series) -> float:
        """Price Volume Trend計算"""
        try:
            pvt = ((closes.pct_change()) * volumes).cumsum()
            return float(pvt.iloc[-1])
        except:
            return 0.0

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP計算"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            return float(vwap.iloc[-1])
        except:
            return float(df['Close'].iloc[-1])