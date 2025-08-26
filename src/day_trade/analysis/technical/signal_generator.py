#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Generator for Technical Analysis
技術分析シグナル生成
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from .base import TechnicalSignal, SignalStrength


class SignalGenerator:
    """技術分析シグナル生成クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate_signals(self, df: pd.DataFrame, trend: Dict, momentum: Dict,
                              volatility: Dict, volume: Dict) -> Tuple[List[TechnicalSignal], List[TechnicalSignal]]:
        """シグナル生成"""

        primary_signals = []
        secondary_signals = []

        try:
            current_price = float(df['Close'].iloc[-1])
            timestamp = datetime.now()

            # MACD買いシグナル
            if 'MACD' in trend and 'MACD_Signal' in trend:
                if trend['MACD'] > trend['MACD_Signal'] and trend['MACD_Histogram'] > 0:
                    signal = TechnicalSignal(
                        indicator_name="MACD",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=75.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=trend['MACD'],
                        trend_direction="UP"
                    )
                    primary_signals.append(signal)

            # RSI逆張りシグナル
            if 'RSI_14' in momentum:
                rsi = momentum['RSI_14']
                if rsi < 30:
                    signal = TechnicalSignal(
                        indicator_name="RSI_Oversold",
                        signal_type="BUY",
                        strength=SignalStrength.STRONG,
                        confidence=80.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=rsi,
                        threshold_lower=30.0
                    )
                    primary_signals.append(signal)
                elif rsi > 70:
                    signal = TechnicalSignal(
                        indicator_name="RSI_Overbought",
                        signal_type="SELL",
                        strength=SignalStrength.MODERATE,
                        confidence=70.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=rsi,
                        threshold_upper=70.0
                    )
                    secondary_signals.append(signal)

            # ボリンジャーバンドブレイクアウト
            if all(key in volatility for key in ['BB_Upper', 'BB_Lower', 'BB_Position']):
                bb_pos = volatility['BB_Position']
                if bb_pos > 95:  # 上限突破
                    signal = TechnicalSignal(
                        indicator_name="BB_Breakout_Up",
                        signal_type="BUY",
                        strength=SignalStrength.STRONG,
                        confidence=85.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=bb_pos,
                        target_price=current_price * 1.03
                    )
                    primary_signals.append(signal)
                elif bb_pos < 5:  # 下限反発
                    signal = TechnicalSignal(
                        indicator_name="BB_Bounce_Up",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=75.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=bb_pos,
                        target_price=current_price * 1.02
                    )
                    secondary_signals.append(signal)

            # 出来高急増シグナル
            if 'Volume_Ratio' in volume and volume['Volume_Ratio'] > 2.0:
                signal = TechnicalSignal(
                    indicator_name="Volume_Surge",
                    signal_type="HOLD",
                    strength=SignalStrength.MODERATE,
                    confidence=70.0,
                    price_level=current_price,
                    timestamp=timestamp,
                    indicator_value=volume['Volume_Ratio']
                )
                secondary_signals.append(signal)

            # トレンド系シグナル
            await self._generate_trend_signals(
                primary_signals, secondary_signals, trend, current_price, timestamp
            )

            # モメンタム系シグナル
            await self._generate_momentum_signals(
                primary_signals, secondary_signals, momentum, current_price, timestamp
            )

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")

        return primary_signals, secondary_signals

    async def _generate_trend_signals(self, primary_signals: List[TechnicalSignal],
                                     secondary_signals: List[TechnicalSignal],
                                     trend: Dict, current_price: float, timestamp: datetime):
        """トレンド系シグナル生成"""
        try:
            # ADXシグナル
            if 'ADX' in trend and trend['ADX'] > 25:
                # 強いトレンド発生
                signal = TechnicalSignal(
                    indicator_name="ADX_Strong_Trend",
                    signal_type="HOLD",
                    strength=SignalStrength.MODERATE,
                    confidence=65.0,
                    price_level=current_price,
                    timestamp=timestamp,
                    indicator_value=trend['ADX']
                )
                secondary_signals.append(signal)

            # 一目均衡表シグナル
            if 'Ichimoku_Tenkan' in trend and 'Ichimoku_Kijun' in trend:
                if trend['Ichimoku_Tenkan'] > trend['Ichimoku_Kijun']:
                    signal = TechnicalSignal(
                        indicator_name="Ichimoku_Bull",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=70.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=trend['Ichimoku_Tenkan']
                    )
                    secondary_signals.append(signal)

        except Exception as e:
            self.logger.error(f"Trend signal generation error: {e}")

    async def _generate_momentum_signals(self, primary_signals: List[TechnicalSignal],
                                        secondary_signals: List[TechnicalSignal],
                                        momentum: Dict, current_price: float, timestamp: datetime):
        """モメンタム系シグナル生成"""
        try:
            # Stochasticシグナル
            if 'Stoch_K' in momentum and 'Stoch_D' in momentum:
                k = momentum['Stoch_K']
                d = momentum['Stoch_D']
                
                if k > d and k < 20:  # 売られすぎからの反転
                    signal = TechnicalSignal(
                        indicator_name="Stoch_Oversold_Cross",
                        signal_type="BUY",
                        strength=SignalStrength.MODERATE,
                        confidence=75.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=k
                    )
                    secondary_signals.append(signal)

            # CCIシグナル
            if 'CCI' in momentum:
                cci = momentum['CCI']
                if cci < -100:  # 売られすぎ
                    signal = TechnicalSignal(
                        indicator_name="CCI_Oversold",
                        signal_type="BUY",
                        strength=SignalStrength.WEAK,
                        confidence=60.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=cci
                    )
                    secondary_signals.append(signal)
                elif cci > 100:  # 買われすぎ
                    signal = TechnicalSignal(
                        indicator_name="CCI_Overbought",
                        signal_type="SELL",
                        strength=SignalStrength.WEAK,
                        confidence=60.0,
                        price_level=current_price,
                        timestamp=timestamp,
                        indicator_value=cci
                    )
                    secondary_signals.append(signal)

        except Exception as e:
            self.logger.error(f"Momentum signal generation error: {e}")