#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators Module - テクニカル分析指標
Issue #937対応: 売買判断システムの実装
Issue #939対応: Polarsへの移行による高速化
"""

import pandas as pd
import polars as pl
from polars import DataFrame, Series
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TechnicalSignal:
    signal_type: SignalType
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reason: str = ""
    timestamp: Optional[str] = None

class TechnicalIndicators:
    """テクニカル分析指標計算クラス (Pandas and Polars)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # --- Polars Implementations ---
    def calculate_rsi_pl(self, prices: Series, period: int = 14) -> Series:
        delta = prices.diff()
        gain = delta.clip_lower(0).rolling_mean(window_size=period)
        loss = -delta.clip_upper(0).rolling_mean(window_size=period)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.alias("rsi")

    def calculate_macd_pl(self, prices: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Series]:
        exp1 = prices.ewm_mean(span=fast)
        exp2 = prices.ewm_mean(span=slow)
        macd = (exp1 - exp2).alias('macd')
        signal_line = macd.ewm_mean(span=signal).alias('signal')
        histogram = (macd - signal_line).alias('histogram')
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

    def calculate_moving_averages_pl(self, prices: Series, periods: List[int] = [5, 10, 20, 50]) -> Dict[str, Series]:
        mas = {}
        for period in periods:
            mas[f'sma_{period}'] = prices.rolling_mean(window_size=period)
            mas[f'ema_{period}'] = prices.ewm_mean(span=period)
        return mas

    def calculate_bollinger_bands_pl(self, prices: Series, period: int = 20, std_dev: float = 2) -> Dict[str, Series]:
        sma = prices.rolling_mean(window_size=period)
        std = prices.rolling_std(window_size=period)
        return {
            'upper': (sma + (std * std_dev)).alias('upper'),
            'middle': sma.alias('middle'),
            'lower': (sma - (std * std_dev)).alias('lower')
        }

    def calculate_stochastic_pl(self, high: Series, low: Series, close: Series, k_period: int = 14, d_period: int = 3) -> Dict[str, Series]:
        lowest_low = low.rolling_min(window_size=k_period)
        highest_high = high.rolling_max(window_size=k_period)
        k_percent = (100 * ((close - lowest_low) / (highest_high - lowest_low))).alias('k_percent')
        d_percent = k_percent.rolling_mean(window_size=d_period).alias('d_percent')
        return {'k_percent': k_percent, 'd_percent': d_percent}

class SignalGenerator:
    """売買シグナル生成クラス (Polars対応)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()
    
    def generate_comprehensive_signal_pl(self, data: DataFrame) -> TechnicalSignal:
        try:
            if len(data) < 50:
                return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason="データ不足")

            close_prices = data['Close']
            high_prices = data['High'] if 'High' in data.columns else close_prices
            low_prices = data['Low'] if 'Low' in data.columns else close_prices

            rsi = self.indicators.calculate_rsi_pl(close_prices)
            macd_data = self.indicators.calculate_macd_pl(close_prices)
            mas = self.indicators.calculate_moving_averages_pl(close_prices)
            bb = self.indicators.calculate_bollinger_bands_pl(close_prices)
            stoch = self.indicators.calculate_stochastic_pl(high_prices, low_prices, close_prices)

            current_price = close_prices[-1]
            current_rsi = rsi[-1]
            current_macd = macd_data['macd'][-1]
            current_signal = macd_data['signal'][-1]
            current_sma_20 = mas['sma_20'][-1]
            current_bb_upper = bb['upper'][-1]
            current_bb_lower = bb['lower'][-1]
            current_k = stoch['k_percent'][-1]

            signals = []
            if current_rsi < 30: signals.append(('BUY', 0.7, 'RSI過売り'))
            elif current_rsi > 70: signals.append(('SELL', 0.7, 'RSI過買い'))

            if current_macd > current_signal and macd_data['macd'][-2] <= macd_data['signal'][-2]:
                signals.append(('BUY', 0.8, 'MACDゴールデンクロス'))
            elif current_macd < current_signal and macd_data['macd'][-2] >= macd_data['signal'][-2]:
                signals.append(('SELL', 0.8, 'MACDデッドクロス'))

            if current_price > current_sma_20: signals.append(('BUY', 0.6, '20日移動平均線上抜け'))
            else: signals.append(('SELL', 0.6, '20日移動平均線下抜け'))

            if current_price < current_bb_lower: signals.append(('BUY', 0.7, 'ボリンジャーバンド下限タッチ'))
            elif current_price > current_bb_upper: signals.append(('SELL', 0.7, 'ボリンジャーバンド上限タッチ'))

            if current_k < 20: signals.append(('BUY', 0.6, 'ストキャスティクス過売り'))
            elif current_k > 80: signals.append(('SELL', 0.6, 'ストキャスティクス過買い'))

            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']

            if len(buy_signals) >= 2:
                avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
                reasons = ", ".join(s[2] for s in buy_signals)
                signal_type = SignalType.STRONG_BUY if avg_confidence > 0.7 else SignalType.BUY
                return TechnicalSignal(signal_type=signal_type, confidence=min(avg_confidence, 0.95), price_target=current_price * 1.05, stop_loss=current_price * 0.95, reason=reasons)
            
            elif len(sell_signals) >= 2:
                avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
                reasons = ", ".join(s[2] for s in sell_signals)
                signal_type = SignalType.STRONG_SELL if avg_confidence > 0.7 else SignalType.SELL
                return TechnicalSignal(signal_type=signal_type, confidence=min(avg_confidence, 0.95), price_target=current_price * 0.95, stop_loss=current_price * 1.05, reason=reasons)
            
            else:
                return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.5, reason="明確なシグナルなし")
                
        except Exception as e:
            self.logger.error(f"シグナル生成エラー (Polars): {e}")
            return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason=f"計算エラー: {str(e)}")

class RiskManager:
    """リスク管理クラス"""
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.02):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, account_balance: float, stock_price: float, stop_loss_price: float) -> int:
        risk_per_share = abs(stock_price - stop_loss_price)
        if risk_per_share <= 0: return 0
        max_risk_amount = account_balance * self.max_daily_loss
        position_value = max_risk_amount / risk_per_share
        max_position_value = account_balance * self.max_position_size
        final_position_value = min(position_value, max_position_value)
        return int(final_position_value / stock_price)

def create_trading_recommendation_pl(symbol: str, data: DataFrame, account_balance: float = 1000000) -> Dict[str, Any]:
    """取引推奨情報の生成 (Polars)"""
    signal_generator = SignalGenerator()
    risk_manager = RiskManager()
    
    signal = signal_generator.generate_comprehensive_signal_pl(data)
    
    current_price = data['Close'][-1]
    position_size = 0
    if signal.stop_loss:
        position_size = risk_manager.calculate_position_size(account_balance, current_price, signal.stop_loss)
    
    return {
        'symbol': symbol,
        'signal': signal.signal_type.value,
        'confidence': signal.confidence,
        'current_price': current_price,
        'target_price': signal.price_target,
        'stop_loss': signal.stop_loss,
        'position_size': position_size,
        'investment_amount': position_size * current_price if position_size > 0 else 0,
        'reason': signal.reason,
        'timestamp': pd.Timestamp.now().isoformat() # Keep pandas for timestamp
    }