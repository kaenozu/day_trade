#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Technical Indicators Module - テクニカル分析指標
Issue #937対応: 売買判断システムの実装

主要機能:
1. RSI (Relative Strength Index)
2. MACD (Moving Average Convergence Divergence)
3. Moving Averages (SMA, EMA)
4. Bollinger Bands
5. Stochastic Oscillator
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

class SignalType(Enum):
    """シグナルタイプ"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TechnicalSignal:
    """テクニカルシグナル"""
    signal_type: SignalType
    confidence: float  # 0.0-1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reason: str = ""
    timestamp: Optional[str] = None

class TechnicalIndicators:
    """テクニカル分析指標計算クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD計算"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_moving_averages(self, prices: pd.Series, periods: List[int] = [5, 10, 20, 50]) -> Dict[str, pd.Series]:
        """移動平均計算"""
        mas = {}
        for period in periods:
            mas[f'sma_{period}'] = prices.rolling(window=period).mean()
            mas[f'ema_{period}'] = prices.ewm(span=period).mean()
        return mas
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """ボリンジャーバンド計算"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """ストキャスティクス計算"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }

class SignalGenerator:
    """売買シグナル生成クラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.indicators = TechnicalIndicators()
    
    def generate_comprehensive_signal(self, data: pd.DataFrame) -> TechnicalSignal:
        """総合的な売買シグナル生成"""
        try:
            if len(data) < 50:
                return TechnicalSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.1,
                    reason="データ不足"
                )
            
            close_prices = data['Close']
            high_prices = data['High'] if 'High' in data.columns else close_prices
            low_prices = data['Low'] if 'Low' in data.columns else close_prices
            
            # 各指標の計算
            rsi = self.indicators.calculate_rsi(close_prices)
            macd_data = self.indicators.calculate_macd(close_prices)
            mas = self.indicators.calculate_moving_averages(close_prices)
            bb = self.indicators.calculate_bollinger_bands(close_prices)
            stoch = self.indicators.calculate_stochastic(high_prices, low_prices, close_prices)
            
            # 最新値を取得
            current_price = close_prices.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            current_sma_20 = mas['sma_20'].iloc[-1]
            current_bb_upper = bb['upper'].iloc[-1]
            current_bb_lower = bb['lower'].iloc[-1]
            current_k = stoch['k_percent'].iloc[-1]
            
            # シグナル判定
            signals = []
            
            # RSI判定
            if current_rsi < 30:
                signals.append(('BUY', 0.7, 'RSI過売り'))
            elif current_rsi > 70:
                signals.append(('SELL', 0.7, 'RSI過買い'))
            
            # MACD判定
            if current_macd > current_signal and macd_data['macd'].iloc[-2] <= macd_data['signal'].iloc[-2]:
                signals.append(('BUY', 0.8, 'MACDゴールデンクロス'))
            elif current_macd < current_signal and macd_data['macd'].iloc[-2] >= macd_data['signal'].iloc[-2]:
                signals.append(('SELL', 0.8, 'MACDデッドクロス'))
            
            # 移動平均判定
            if current_price > current_sma_20:
                signals.append(('BUY', 0.6, '20日移動平均線上抜け'))
            else:
                signals.append(('SELL', 0.6, '20日移動平均線下抜け'))
            
            # ボリンジャーバンド判定
            if current_price < current_bb_lower:
                signals.append(('BUY', 0.7, 'ボリンジャーバンド下限タッチ'))
            elif current_price > current_bb_upper:
                signals.append(('SELL', 0.7, 'ボリンジャーバンド上限タッチ'))
            
            # ストキャスティクス判定
            if current_k < 20:
                signals.append(('BUY', 0.6, 'ストキャスティクス過売り'))
            elif current_k > 80:
                signals.append(('SELL', 0.6, 'ストキャスティクス過買い'))
            
            # 総合判定
            buy_signals = [s for s in signals if s[0] == 'BUY']
            sell_signals = [s for s in signals if s[0] == 'SELL']
            
            if len(buy_signals) >= 2:
                avg_confidence = sum(s[1] for s in buy_signals) / len(buy_signals)
                reasons = ", ".join(s[2] for s in buy_signals)
                signal_type = SignalType.STRONG_BUY if avg_confidence > 0.7 else SignalType.BUY
                
                return TechnicalSignal(
                    signal_type=signal_type,
                    confidence=min(avg_confidence, 0.95),
                    price_target=current_price * 1.05,  # 5%上昇目標
                    stop_loss=current_price * 0.95,    # 5%損切り
                    reason=reasons
                )
            
            elif len(sell_signals) >= 2:
                avg_confidence = sum(s[1] for s in sell_signals) / len(sell_signals)
                reasons = ", ".join(s[2] for s in sell_signals)
                signal_type = SignalType.STRONG_SELL if avg_confidence > 0.7 else SignalType.SELL
                
                return TechnicalSignal(
                    signal_type=signal_type,
                    confidence=min(avg_confidence, 0.95),
                    price_target=current_price * 0.95,  # 5%下落予想
                    stop_loss=current_price * 1.05,    # 5%ストップ
                    reason=reasons
                )
            
            else:
                return TechnicalSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.5,
                    reason="明確なシグナルなし"
                )
                
        except Exception as e:
            self.logger.error(f"シグナル生成エラー: {e}")
            return TechnicalSignal(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                reason=f"計算エラー: {str(e)}"
            )

class RiskManager:
    """リスク管理クラス"""
    
    def __init__(self, max_position_size: float = 0.1, max_daily_loss: float = 0.02):
        self.max_position_size = max_position_size  # ポートフォリオの10%まで
        self.max_daily_loss = max_daily_loss        # 1日2%まで損失許容
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(self, account_balance: float, stock_price: float, 
                              stop_loss_price: float) -> int:
        """適切なポジションサイズ計算"""
        # リスク金額の計算
        risk_per_share = abs(stock_price - stop_loss_price)
        max_risk_amount = account_balance * self.max_daily_loss
        
        # ポジションサイズ計算
        if risk_per_share > 0:
            position_value = max_risk_amount / risk_per_share
            max_position_value = account_balance * self.max_position_size
            
            final_position_value = min(position_value, max_position_value)
            shares = int(final_position_value / stock_price)
            
            return max(shares, 0)
        
        return 0
    
    def validate_trade(self, signal: TechnicalSignal, account_balance: float, 
                      current_positions: Dict[str, Any]) -> bool:
        """取引の妥当性検証"""
        # 信頼度チェック
        if signal.confidence < 0.6:
            return False
        
        # ポジション数制限
        if len(current_positions) >= 10:  # 最大10銘柄まで
            return False
        
        return True

def create_trading_recommendation(symbol: str, data: pd.DataFrame, 
                                account_balance: float = 1000000) -> Dict[str, Any]:
    """取引推奨情報の生成"""
    signal_generator = SignalGenerator()
    risk_manager = RiskManager()
    
    # シグナル生成
    signal = signal_generator.generate_comprehensive_signal(data)
    
    # ポジションサイズ計算
    current_price = data['Close'].iloc[-1]
    position_size = 0
    
    if signal.stop_loss:
        position_size = risk_manager.calculate_position_size(
            account_balance, current_price, signal.stop_loss
        )
    
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
        'timestamp': pd.Timestamp.now().isoformat()
    }