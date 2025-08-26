#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Momentum Indicators - モメンタム系指標計算モジュール
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple


class MomentumIndicatorCalculator:
    """モメンタム系指標計算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        モメンタム系指標計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            モメンタム系指標辞書
        """
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
                (closes.iloc[-1] / closes.iloc[-13] - 1) * 100 
                if len(closes) >= 13 else 0
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
        """
        RSI計算
        
        Args:
            prices: 価格系列
            period: 計算期間
            
        Returns:
            RSI値
        """
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
        """
        ストキャスティクス計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            k_period: %K期間
            d_period: %D期間
            
        Returns:
            (%K値, %D値)
        """
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
        """
        Williams %R計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            period: 計算期間
            
        Returns:
            Williams %R値
        """
        try:
            highest_high = highs.rolling(window=period).max()
            lowest_low = lows.rolling(window=period).min()
            wr = -100 * ((highest_high - closes) / (highest_high - lowest_low))
            return float(wr.iloc[-1])
        except:
            return -50.0
    
    def _calculate_cci(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = 20) -> float:
        """
        CCI計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            period: 計算期間
            
        Returns:
            CCI値
        """
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
        """
        MFI計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            volumes: 出来高系列
            period: 計算期間
            
        Returns:
            MFI値
        """
        try:
            tp = (highs + lows + closes) / 3
            mf = tp * volumes

            pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
            neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()

            mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
            return float(mfi.iloc[-1])
        except:
            return 50.0
    
    def _calculate_ultimate_oscillator(self, highs: pd.Series, 
                                     lows: pd.Series, closes: pd.Series) -> float:
        """
        Ultimate Oscillator計算（簡易版）
        
        Args:
            highs: 高値系列
            lows: 安値系列
            closes: 終値系列
            
        Returns:
            Ultimate Oscillator値
        """
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
        """
        Awesome Oscillator計算
        
        Args:
            highs: 高値系列
            lows: 安値系列
            
        Returns:
            Awesome Oscillator値
        """
        try:
            median_price = (highs + lows) / 2
            ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
            return float(ao.iloc[-1])
        except:
            return 0.0