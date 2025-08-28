#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Analysis Engine - 分析エンジン・スコア計算・シグナル生成モジュール
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

from .types_and_enums import TechnicalSignal, SignalStrength


class AnalysisEngine:
    """分析エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.volatility_regimes = {
            "超低ボラ": (0, 10),
            "低ボラ": (10, 20),
            "通常ボラ": (20, 35),
            "高ボラ": (35, 50),
            "超高ボラ": (50, 100)
        }
    
    def calculate_composite_score(self, trend: Dict, momentum: Dict, 
                                volatility: Dict) -> float:
        """
        複合スコア計算
        
        Args:
            trend: トレンド指標辞書
            momentum: モメンタム指標辞書
            volatility: ボラティリティ指標辞書
            
        Returns:
            複合スコア（0-100）
        """
        try:
            score = 50.0  # 中立からスタート

            # トレンド評価
            if 'SMA_5' in trend and 'SMA_20' in trend:
                if trend['SMA_5'] > trend['SMA_20']:
                    score += 10
                else:
                    score -= 10

            # MACD評価
            if 'MACD' in trend and 'MACD_Signal' in trend:
                if trend['MACD'] > trend['MACD_Signal']:
                    score += 8
                else:
                    score -= 8

            # RSI評価
            if 'RSI_14' in momentum:
                rsi = momentum['RSI_14']
                if 30 <= rsi <= 70:
                    score += 5  # 適正レンジ
                elif rsi > 80:
                    score -= 10  # 買われすぎ
                elif rsi < 20:
                    score += 10  # 売られすぎ（反転期待）

            # ボラティリティ評価
            if 'BB_Position' in volatility:
                bb_pos = volatility['BB_Position']
                if 20 <= bb_pos <= 80:
                    score += 3  # 適正範囲
                else:
                    score -= 3  # 極端な位置

            return max(0, min(100, score))

        except Exception as e:
            self.logger.error(f"Composite score calculation error: {e}")
            return 50.0
    
    def calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """
        トレンド強度計算
        
        Args:
            indicators: トレンド指標辞書
            
        Returns:
            トレンド強度（-100～100）
        """
        try:
            strength = 0.0

            # MACD強度
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                macd_strength = (indicators['MACD'] - indicators['MACD_Signal']) * 10
                strength += max(-30, min(30, macd_strength))

            # ADX強度
            if 'ADX' in indicators:
                adx = indicators['ADX']
                if adx > 25:
                    strength += min(25, adx)
                else:
                    strength += adx / 2

            # 移動平均の位置関係
            if all(key in indicators for key in ['SMA_5', 'SMA_20', 'SMA_50']):
                if indicators['SMA_5'] > indicators['SMA_20'] > indicators['SMA_50']:
                    strength += 20  # 上昇トレンド
                elif indicators['SMA_5'] < indicators['SMA_20'] < indicators['SMA_50']:
                    strength -= 20  # 下降トレンド

            return max(-100, min(100, strength))

        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.0
    
    def calculate_momentum_score(self, indicators: Dict[str, float]) -> float:
        """
        モメンタムスコア計算
        
        Args:
            indicators: モメンタム指標辞書
            
        Returns:
            モメンタムスコア（-100～100）
        """
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
    
    def determine_volatility_regime(self, indicators: Dict[str, float]) -> str:
        """
        ボラティリティ局面判定
        
        Args:
            indicators: ボラティリティ指標辞書
            
        Returns:
            ボラティリティ局面
        """
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
    
    async def generate_signals(self, df: pd.DataFrame, trend: Dict, 
                             momentum: Dict, volatility: Dict, 
                             volume: Dict) -> Tuple[List[TechnicalSignal], 
                                                  List[TechnicalSignal]]:
        """
        シグナル生成
        
        Args:
            df: 価格データ DataFrame
            trend: トレンド指標辞書
            momentum: モメンタム指標辞書
            volatility: ボラティリティ指標辞書
            volume: 出来高指標辞書
            
        Returns:
            (プライマリシグナルリスト, セカンダリシグナルリスト)
        """
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

        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")

        return primary_signals, secondary_signals
    
    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        統計分析実行
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            統計プロファイル辞書
        """
        try:
            closes = df['Close']
            returns = closes.pct_change().dropna()

            return {
                'mean_return': float(returns.mean() * 252),  # 年率化
                'volatility': float(returns.std() * np.sqrt(252)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(closes),
                'var_95': float(np.percentile(returns, 5) * 100),
                'autocorrelation': float(returns.autocorr(lag=1))
            }

        except Exception as e:
            self.logger.error(f"Statistical analysis error: {e}")
            return {}
    
    def calculate_anomaly_score(self, df: pd.DataFrame) -> float:
        """
        異常度スコア計算
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            異常度スコア（0-100）
        """
        try:
            # 価格変動率の異常度
            returns = df['Close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.0

            # Z-score計算
            z_score = abs((returns.iloc[-1] - returns.mean()) / returns.std())

            # 0-100スケールに変換
            anomaly_score = min(100, z_score * 20)

            return float(anomaly_score)

        except Exception as e:
            self.logger.error(f"Anomaly score calculation error: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        最大ドローダウン計算
        
        Args:
            prices: 価格系列
            
        Returns:
            最大ドローダウン（%）
        """
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min() * 100)
        except:
            return 0.0