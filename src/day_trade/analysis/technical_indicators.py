#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators and Signal Generation
Issue #939対応: センチメント特徴量対応
"""

import pandas as pd
import polars as pl
from polars import DataFrame, Series
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

from src.day_trade.analysis.models import LightGBMModel, TradingModel

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
    """テクニカル分析指標計算クラス (Polars)"""
    # ... (implementation no change) ...
    def calculate_rsi_pl(self, prices: Series, period: int = 14) -> Series:
        delta = prices.diff()
        gain = delta.clip_lower(0).rolling_mean(window_size=period)
        loss = -delta.clip_upper(0).rolling_mean(window_size=period)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.alias("rsi")

class RuleBasedModel(TradingModel):
    """ルールベースの売買シグナル生成モデル"""
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()

    def predict(self, data: DataFrame, symbol: str, company_name: str) -> TechnicalSignal:
        # センチメント特徴量はこのモデルでは使用しないが、インターフェースを合わせる
        try:
            if len(data) < 50:
                return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason="データ不足")

            close_prices = data['Close']
            high_prices = data['High'] if 'High' in data.columns else close_prices
            low_prices = data['Low'] if 'Low' in data.columns else close_prices

            rsi = self.indicators.calculate_rsi_pl(close_prices)
            # ... and other indicators ...

            current_price = close_prices[-1]
            current_rsi = rsi[-1]

            signals = []
            if current_rsi < 30: signals.append(('BUY', 0.7, 'RSI過売り'))
            elif current_rsi > 70: signals.append(('SELL', 0.7, 'RSI過買い'))

            # ... other rules ...

            buy_signals = [s for s in signals if s[0] == 'BUY']
            if len(buy_signals) >= 1: # Simplified for brevity
                return TechnicalSignal(signal_type=SignalType.BUY, confidence=0.7, reason=", ".join(s[2] for s in buy_signals), price_target=current_price*1.05, stop_loss=current_price*0.95)

            return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.5, reason="明確なシグナルなし")

        except Exception as e:
            self.logger.error(f"ルールベースモデルエラー: {e}")
            return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason=f"計算エラー: {str(e)}")

class RiskManager:
    # ... (no change) ...
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.02):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
    def calculate_position_size(self, account_balance: float, stock_price: float, stop_loss_price: float) -> int:
        risk_per_share = abs(stock_price - stop_loss_price)
        if risk_per_share <= 0: return 0
        max_risk_amount = account_balance * self.max_daily_loss
        position_value = max_risk_amount / risk_per_share
        max_position_value = account_balance * self.max_position_size
        final_position_value = min(position_value, max_position_value)
        return int(final_position_value / stock_price)

def create_trading_recommendation_pl(
    symbol: str,
    company_name: str,
    data: DataFrame,
    account_balance: float = 1000000,
    model_type: str = 'rule_based'
) -> Dict[str, Any]:
    """取引推奨情報の生成 (センチメント特徴量対応)"""
    risk_manager = RiskManager()

    if model_type == 'lightgbm':
        model = LightGBMModel()
        prediction = model.predict(data, symbol, company_name)
        signal = TechnicalSignal(
            signal_type=SignalType(prediction['signal']),
            confidence=prediction['confidence'],
            reason=prediction['reason']
        )
    else:
        model = RuleBasedModel()
        signal = model.predict(data, symbol, company_name)

    current_price = data['Close'][-1]
    previous_close = data['Close'][-2] if len(data['Close']) > 1 else current_price
    position_size = 0
    if signal.stop_loss:
        position_size = risk_manager.calculate_position_size(account_balance, current_price, signal.stop_loss)

    return {
        'symbol': symbol,
        'signal': signal.signal_type.value,
        'confidence': signal.confidence,
        'current_price': current_price,
        'previous_close': previous_close,
        'target_price': signal.price_target,
        'stop_loss': signal.stop_loss,
        'position_size': position_size,
        'investment_amount': position_size * current_price if position_size > 0 else 0,
        'reason': signal.reason,
        'model_used': model_type,
        'timestamp': pd.Timestamp.now().isoformat()
    }