#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Volatility and Volume Indicators - ボラティリティ・出来高系指標計算モジュール
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple


class VolatilityVolumeCalculator:
    """ボラティリティ・出来高系指標計算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        ボラティリティ系指標計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            ボラティリティ系指標辞書
        """
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
            kc_middle, kc_upper, kc_lower = self._calculate_keltner_channel(
                highs, lows, closes
            )
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
    
    async def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        出来高系指標計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            出来高系指標辞書
        """
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
    
    def _calculate_atr(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 14) -> float:
        """
        ATR計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            period: 計算期間
            
        Returns:
            ATR値
        """
        try:
            high_low = highs - lows
            high_close = np.abs(highs - closes.shift())
            low_close = np.abs(lows - closes.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return float(atr.iloc[-1])
        except:
            return 0.0
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                  std_dev: float = 2) -> Tuple[float, float, float]:
        """
        ボリンジャーバンド計算
        
        Args:
            prices: 価格系列
            period: 計算期間
            std_dev: 標準偏差倍率
            
        Returns:
            (中央線, 上線, 下線)
        """
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return (
                float(middle.iloc[-1]), 
                float(upper.iloc[-1]), 
                float(lower.iloc[-1])
            )
        except:
            last_price = float(prices.iloc[-1])
            return last_price, last_price * 1.02, last_price * 0.98
    
    def _calculate_keltner_channel(self, highs: pd.Series, lows: pd.Series, 
                                  closes: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """
        Keltner Channel計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            period: 計算期間
            
        Returns:
            (中央線, 上線, 下線)
        """
        try:
            middle = closes.rolling(period).mean()
            atr = self._calculate_atr(highs, lows, closes, period)

            upper = middle + 2 * atr
            lower = middle - 2 * atr

            return (
                float(middle.iloc[-1]), 
                float(upper), 
                float(lower)
            )
        except:
            price = float(closes.iloc[-1])
            return price, price * 1.02, price * 0.98
    
    def _calculate_obv(self, closes: pd.Series, volumes: pd.Series) -> float:
        """
        OBV計算
        
        Args:
            closes: 終値系列
            volumes: 出来高系列
            
        Returns:
            OBV値
        """
        try:
            price_change = closes.diff()
            obv = volumes.copy()
            obv[price_change < 0] = -obv[price_change < 0]
            obv[price_change == 0] = 0
            return float(obv.cumsum().iloc[-1])
        except:
            return 0.0
    
    def _calculate_ad_line(self, df: pd.DataFrame) -> float:
        """
        A/D Line計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            A/D Line値
        """
        try:
            clv = (
                (df['Close'] - df['Low']) - (df['High'] - df['Close'])
            ) / (df['High'] - df['Low'])
            ad_line = (clv * df['Volume']).cumsum()
            return float(ad_line.iloc[-1])
        except:
            return 0.0
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        Chaikin Money Flow計算
        
        Args:
            df: 価格データ DataFrame
            period: 計算期間
            
        Returns:
            CMF値
        """
        try:
            clv = (
                (df['Close'] - df['Low']) - (df['High'] - df['Close'])
            ) / (df['High'] - df['Low'])
            cmf = (
                (clv * df['Volume']).rolling(period).sum() / 
                df['Volume'].rolling(period).sum()
            )
            return float(cmf.iloc[-1])
        except:
            return 0.0
    
    def _calculate_pvt(self, closes: pd.Series, volumes: pd.Series) -> float:
        """
        Price Volume Trend計算
        
        Args:
            closes: 終値系列
            volumes: 出来高系列
            
        Returns:
            PVT値
        """
        try:
            pvt = ((closes.pct_change()) * volumes).cumsum()
            return float(pvt.iloc[-1])
        except:
            return 0.0
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        VWAP計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            VWAP値
        """
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (
                (typical_price * df['Volume']).cumsum() / 
                df['Volume'].cumsum()
            )
            return float(vwap.iloc[-1])
        except:
            return float(df['Close'].iloc[-1])