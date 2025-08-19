#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators and Signal Generation
Issue #939対応: モデル選択ロジックの導入
"""

import pandas as pd
import polars as pl
from polars import DataFrame, Series
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# モデルのインポート
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
    def calculate_rsi_pl(self, prices: Series, period: int = 14) -> Series:
        # ... (実装は変更なし) ...
        delta = prices.diff()
        gain = delta.clip_lower(0).rolling_mean(window_size=period)
        loss = -delta.clip_upper(0).rolling_mean(window_size=period)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.alias("rsi")

    def calculate_macd_pl(self, prices: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Series]:
        # ... (実装は変更なし) ...
        exp1 = prices.ewm_mean(span=fast)
        exp2 = prices.ewm_mean(span=slow)
        macd = (exp1 - exp2).alias('macd')
        signal_line = macd.ewm_mean(span=signal).alias('signal')
        histogram = (macd - signal_line).alias('histogram')
        return {'macd': macd, 'signal': signal_line, 'histogram': histogram}

    # ... 他の指標計算メソッドも同様 ...

class RuleBasedModel(TradingModel):
    """ルールベースの売買シグナル生成モデル"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()
    
    def predict(self, data: DataFrame) -> TechnicalSignal:
        # (旧 generate_comprehensive_signal_pl)
        try:
            if len(data) < 50:
                return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason="データ不足")

            close_prices = data['Close']
            # ... (シグナル生成ロジックは変更なし) ...
            high_prices = data['High'] if 'High' in data.columns else close_prices
            low_prices = data['Low'] if 'Low' in data.columns else close_prices

            rsi = self.indicators.calculate_rsi_pl(close_prices)
            macd_data = self.indicators.calculate_macd_pl(close_prices)

            current_price = close_prices[-1]
            current_rsi = rsi[-1]
            current_macd = macd_data['macd'][-1]
            current_signal = macd_data['signal'][-1]

            signals = []
            if current_rsi < 30: signals.append(('BUY', 0.7, 'RSI過売り'))
            elif current_rsi > 70: signals.append(('SELL', 0.7, 'RSI過買い'))

            if current_macd > current_signal and macd_data['macd'][-2] <= macd_data['signal'][-2]:
                signals.append(('BUY', 0.8, 'MACDゴールデンクロス'))
            elif current_macd < current_signal and macd_data['macd'][-2] >= macd_data['signal'][-2]:
                signals.append(('SELL', 0.8, 'MACDデッドクロス'))

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
            self.logger.error(f"ルールベースモデルエラー: {e}")
            return TechnicalSignal(signal_type=SignalType.HOLD, confidence=0.1, reason=f"計算エラー: {str(e)}")

class RiskManager:
    # ... (変更なし) ...
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

def create_trading_recommendation_pl(
    symbol: str, 
    data: DataFrame, 
    account_balance: float = 1000000, 
    model_type: str = 'rule_based'
) -> Dict[str, Any]:
    """取引推奨情報の生成 (モデル選択対応)"""
    risk_manager = RiskManager()
    
    # モデル選択
    if model_type == 'lightgbm':
        model = LightGBMModel()
        prediction = model.predict(data)
        # LightGBMの出力をTechnicalSignalに変換
        signal = TechnicalSignal(
            signal_type=SignalType(prediction['signal']),
            confidence=prediction['confidence'],
            reason=prediction['reason']
        )
    else: # デフォルトはルールベース
        model = RuleBasedModel()
        signal = model.predict(data)

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
        'model_used': model_type,
        'timestamp': pd.Timestamp.now().isoformat()
    }
