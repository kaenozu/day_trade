#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Analysis System - シグナル分析システム

技術指標からのシグナル生成と分析機能
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

from .types import TechnicalSignal, PatternMatch


class SignalAnalyzer:
    """シグナル分析システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_momentum_signals(self, indicators: Dict[str, pd.Series], 
                               data: pd.DataFrame) -> List[TechnicalSignal]:
        """モメンタムシグナル分析"""

        signals = []
        current_price = data['Close'].iloc[-1]

        try:
            # RSI系シグナル
            if 'connors_rsi' in indicators:
                crsi = indicators['connors_rsi'].iloc[-1]
                if not np.isnan(crsi):
                    if crsi < 20:
                        signals.append(TechnicalSignal(
                            indicator_name="Connors RSI",
                            signal_type="BUY",
                            strength=80,
                            confidence=0.8,
                            timeframe="short",
                            description=f"Connors RSI過売り水準 ({crsi:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif crsi > 80:
                        signals.append(TechnicalSignal(
                            indicator_name="Connors RSI",
                            signal_type="SELL",
                            strength=80,
                            confidence=0.8,
                            timeframe="short",
                            description=f"Connors RSI過買い水準 ({crsi:.1f})",
                            timestamp=datetime.now()
                        ))

            # Money Flow Index
            if 'money_flow_index' in indicators:
                mfi = indicators['money_flow_index'].iloc[-1]
                if not np.isnan(mfi):
                    if mfi < 20:
                        signals.append(TechnicalSignal(
                            indicator_name="Money Flow Index",
                            signal_type="BUY",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description=f"MFI過売り水準 ({mfi:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif mfi > 80:
                        signals.append(TechnicalSignal(
                            indicator_name="Money Flow Index",
                            signal_type="SELL",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description=f"MFI過買い水準 ({mfi:.1f})",
                            timestamp=datetime.now()
                        ))

            # Awesome Oscillator
            if 'awesome_oscillator' in indicators and len(indicators['awesome_oscillator']) > 1:
                ao_current = indicators['awesome_oscillator'].iloc[-1]
                ao_prev = indicators['awesome_oscillator'].iloc[-2]
                if not np.isnan(ao_current) and not np.isnan(ao_prev):
                    if ao_current > 0 and ao_prev <= 0:
                        signals.append(TechnicalSignal(
                            indicator_name="Awesome Oscillator",
                            signal_type="BUY",
                            strength=60,
                            confidence=0.6,
                            timeframe="medium",
                            description="AO ゼロライン上抜け",
                            timestamp=datetime.now()
                        ))
                    elif ao_current < 0 and ao_prev >= 0:
                        signals.append(TechnicalSignal(
                            indicator_name="Awesome Oscillator",
                            signal_type="SELL",
                            strength=60,
                            confidence=0.6,
                            timeframe="medium",
                            description="AO ゼロライン下抜け",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"モメンタムシグナル分析エラー: {e}")

        return signals

    def analyze_trend_signals(self, indicators: Dict[str, pd.Series], 
                            data: pd.DataFrame) -> List[TechnicalSignal]:
        """トレンドシグナル分析"""

        signals = []
        current_price = data['Close'].iloc[-1]

        try:
            # Parabolic SAR
            if 'parabolic_sar' in indicators:
                psar = indicators['parabolic_sar'].iloc[-1]
                if not np.isnan(psar):
                    if current_price > psar:
                        signals.append(TechnicalSignal(
                            indicator_name="Parabolic SAR",
                            signal_type="BUY",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description="価格がSAR上位",
                            timestamp=datetime.now()
                        ))
                    else:
                        signals.append(TechnicalSignal(
                            indicator_name="Parabolic SAR",
                            signal_type="SELL",
                            strength=70,
                            confidence=0.7,
                            timeframe="medium",
                            description="価格がSAR下位",
                            timestamp=datetime.now()
                        ))

            # ADX
            if all(k in indicators for k in ['adx', 'dmi_positive', 'dmi_negative']):
                adx = indicators['adx'].iloc[-1]
                dmi_pos = indicators['dmi_positive'].iloc[-1]
                dmi_neg = indicators['dmi_negative'].iloc[-1]

                if not any(np.isnan([adx, dmi_pos, dmi_neg])):
                    if adx > 25:  # 強いトレンド
                        if dmi_pos > dmi_neg:
                            signals.append(TechnicalSignal(
                                indicator_name="ADX/DMI",
                                signal_type="BUY",
                                strength=75,
                                confidence=0.75,
                                timeframe="medium",
                                description=f"強い上昇トレンド (ADX:{adx:.1f})",
                                timestamp=datetime.now()
                            ))
                        else:
                            signals.append(TechnicalSignal(
                                indicator_name="ADX/DMI",
                                signal_type="SELL",
                                strength=75,
                                confidence=0.75,
                                timeframe="medium",
                                description=f"強い下降トレンド (ADX:{adx:.1f})",
                                timestamp=datetime.now()
                            ))

            # CCI
            if 'cci' in indicators:
                cci = indicators['cci'].iloc[-1]
                if not np.isnan(cci):
                    if cci > 100:
                        signals.append(TechnicalSignal(
                            indicator_name="CCI",
                            signal_type="SELL",
                            strength=65,
                            confidence=0.65,
                            timeframe="short",
                            description=f"CCI過買い水準 ({cci:.1f})",
                            timestamp=datetime.now()
                        ))
                    elif cci < -100:
                        signals.append(TechnicalSignal(
                            indicator_name="CCI",
                            signal_type="BUY",
                            strength=65,
                            confidence=0.65,
                            timeframe="short",
                            description=f"CCI過売り水準 ({cci:.1f})",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"トレンドシグナル分析エラー: {e}")

        return signals

    def analyze_volatility_signals(self, indicators: Dict[str, pd.Series], 
                                 data: pd.DataFrame) -> List[TechnicalSignal]:
        """ボラティリティシグナル分析"""

        signals = []

        try:
            # Bollinger Bands
            if 'bollinger_percent' in indicators:
                bb_percent = indicators['bollinger_percent'].iloc[-1]
                if not np.isnan(bb_percent):
                    if bb_percent > 0.95:
                        signals.append(TechnicalSignal(
                            indicator_name="Bollinger Bands",
                            signal_type="SELL",
                            strength=60,
                            confidence=0.6,
                            timeframe="short",
                            description="ボリンジャーバンド上限近く",
                            timestamp=datetime.now()
                        ))
                    elif bb_percent < 0.05:
                        signals.append(TechnicalSignal(
                            indicator_name="Bollinger Bands",
                            signal_type="BUY",
                            strength=60,
                            confidence=0.6,
                            timeframe="short",
                            description="ボリンジャーバンド下限近く",
                            timestamp=datetime.now()
                        ))

            # Volatility Ratio
            if 'volatility_ratio' in indicators:
                vol_ratio = indicators['volatility_ratio'].iloc[-1]
                if not np.isnan(vol_ratio):
                    if vol_ratio > 1.5:
                        signals.append(TechnicalSignal(
                            indicator_name="Volatility Ratio",
                            signal_type="HOLD",
                            strength=40,
                            confidence=0.5,
                            timeframe="short",
                            description="高ボラティリティ環境",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"ボラティリティシグナル分析エラー: {e}")

        return signals

    def analyze_volume_signals(self, indicators: Dict[str, pd.Series], 
                             data: pd.DataFrame) -> List[TechnicalSignal]:
        """ボリュームシグナル分析"""

        signals = []

        try:
            # Chaikin Money Flow
            if 'chaikin_money_flow' in indicators:
                cmf = indicators['chaikin_money_flow'].iloc[-1]
                if not np.isnan(cmf):
                    if cmf > 0.2:
                        signals.append(TechnicalSignal(
                            indicator_name="Chaikin Money Flow",
                            signal_type="BUY",
                            strength=65,
                            confidence=0.65,
                            timeframe="medium",
                            description=f"強い買い圧力 (CMF:{cmf:.2f})",
                            timestamp=datetime.now()
                        ))
                    elif cmf < -0.2:
                        signals.append(TechnicalSignal(
                            indicator_name="Chaikin Money Flow",
                            signal_type="SELL",
                            strength=65,
                            confidence=0.65,
                            timeframe="medium",
                            description=f"強い売り圧力 (CMF:{cmf:.2f})",
                            timestamp=datetime.now()
                        ))

            # Volume Oscillator
            if 'volume_oscillator' in indicators:
                vol_osc = indicators['volume_oscillator'].iloc[-1]
                if not np.isnan(vol_osc):
                    if vol_osc > 10:
                        signals.append(TechnicalSignal(
                            indicator_name="Volume Oscillator",
                            signal_type="BUY",
                            strength=50,
                            confidence=0.5,
                            timeframe="short",
                            description="ボリューム増加トレンド",
                            timestamp=datetime.now()
                        ))

        except Exception as e:
            self.logger.error(f"ボリュームシグナル分析エラー: {e}")

        return signals

    def calculate_overall_sentiment(self, signals: List[TechnicalSignal]) -> Tuple[str, float]:
        """総合センチメント計算"""

        if not signals:
            return "NEUTRAL", 0.5

        buy_score = sum(s.strength * s.confidence for s in signals if s.signal_type == "BUY")
        sell_score = sum(s.strength * s.confidence for s in signals if s.signal_type == "SELL")
        total_signals = len([s for s in signals if s.signal_type in ["BUY", "SELL"]])

        if total_signals == 0:
            return "NEUTRAL", 0.5

        net_score = (buy_score - sell_score) / (buy_score + sell_score + 1e-10)
        confidence = min(1.0, (buy_score + sell_score) / (total_signals * 100))

        if net_score > 0.3:
            sentiment = "BULLISH"
        elif net_score < -0.3:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return sentiment, confidence

    def assess_risk_level(self, indicators: Dict[str, float], 
                         volatility_indicators: Dict[str, pd.Series]) -> str:
        """リスクレベル評価"""

        risk_factors = []

        # ボラティリティリスク
        if 'historical_vol_10' in indicators:
            vol = indicators['historical_vol_10']
            if vol > 0.3:  # 30%以上の年率ボラティリティ
                risk_factors.append("高ボラティリティ")

        # トレンド強度
        if 'adx' in indicators:
            adx = indicators['adx']
            if adx < 20:
                risk_factors.append("弱いトレンド")

        risk_score = len(risk_factors)

        if risk_score >= 2:
            return "HIGH"
        elif risk_score == 1:
            return "MEDIUM"
        else:
            return "LOW"

    def generate_recommendations(self, signals: List[TechnicalSignal],
                               patterns: List[PatternMatch], sentiment: str) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        # センチメントベース推奨
        if sentiment == "BULLISH":
            recommendations.append("技術指標は上昇トレンドを示唆")
            recommendations.append("押し目での買い機会を検討")
        elif sentiment == "BEARISH":
            recommendations.append("技術指標は下降トレンドを示唆")
            recommendations.append("戻り売り機会を検討")
        else:
            recommendations.append("中立的な市場環境")
            recommendations.append("明確なトレンド確認まで様子見推奨")

        # 強いシグナルから推奨
        strong_signals = [s for s in signals if s.strength > 70]
        if strong_signals:
            for signal in strong_signals[:3]:  # 上位3つ
                recommendations.append(f"{signal.indicator_name}: {signal.description}")

        # パターンから推奨
        reliable_patterns = [p for p in patterns if p.reliability > 0.8]
        if reliable_patterns:
            for pattern in reliable_patterns[:2]:  # 上位2つ
                recommendations.append(f"{pattern.pattern_name}パターン検出")

        return recommendations