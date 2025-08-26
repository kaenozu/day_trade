#!/usr/bin/env python3
"""
投資機会アラートシステム - テクニカル分析器
"""

import time
from datetime import datetime
from typing import Dict, Optional

from .enums import OpportunitySeverity, TradingAction
from .models import InvestmentOpportunity, OpportunityConfig


class TechnicalAnalyzer:
    """テクニカル分析器"""

    async def analyze_technical_breakout(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """テクニカルブレイクアウト分析"""

        # ブレイクアウト条件
        sma_20 = indicators.get("SMA_20", current_price)
        bb_upper = indicators.get("BB_Upper", current_price * 1.02)
        volume_ratio = indicators.get("Volume_Ratio", 1.0)

        # 上方ブレイクアウト判定
        price_above_sma = current_price > sma_20 * 1.02  # 2%以上上
        volume_confirmation = volume_ratio > 1.5  # 出来高1.5倍以上
        near_bb_upper = current_price > bb_upper * 0.98  # ボリンジャー上限付近

        if price_above_sma and volume_confirmation and near_bb_upper:
            confidence = 0.6 + (volume_ratio - 1.5) * 0.1  # 出来高に応じて信頼度調整
            confidence = min(confidence, 0.95)

            profit_potential = ((bb_upper * 1.05 - current_price) / current_price) * 100

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"breakout_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.HIGH,
                    recommended_action=TradingAction.BUY,
                    target_price=bb_upper * 1.05,
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 上方ブレイクアウト機会検出",
                )

        return None

    async def analyze_momentum_signal(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """モメンタム分析"""

        rsi = indicators.get("RSI", 50)
        macd = indicators.get("MACD", 0)
        macd_signal = indicators.get("MACD_Signal", 0)
        price_change_5d = indicators.get("Price_Change_5D", 0)

        # 強い上昇モメンタム判定
        rsi_bullish = 50 < rsi < config.rsi_overbought
        macd_bullish = macd > macd_signal and macd > 0
        positive_momentum = price_change_5d > 2.0  # 5日間で2%以上上昇

        if rsi_bullish and macd_bullish and positive_momentum:
            confidence = 0.65 + (price_change_5d / 20.0)  # モメンタムに応じて調整
            confidence = min(confidence, 0.9)

            # 利益目標をモメンタムに基づいて設定
            profit_potential = price_change_5d * 1.5  # 現在のモメンタムの1.5倍

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"momentum_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.BUY,
                    target_price=current_price * (1 + profit_potential / 100),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 強い上昇モメンタム検出",
                )

        return None

    async def analyze_reversal_pattern(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """リバーサルパターン分析"""

        rsi = indicators.get("RSI", 50)
        bb_position = indicators.get("BB_Position", 0.5)
        price_change_1d = indicators.get("Price_Change_1D", 0)

        # 過売り反転パターン
        oversold_rsi = rsi < config.rsi_oversold
        near_bb_lower = bb_position < 0.2  # ボリンジャー下限付近
        recent_decline = price_change_1d < -2.0  # 直近で2%以上下落

        if oversold_rsi and near_bb_lower and recent_decline:
            # RSIが低いほど反転の可能性が高い
            confidence = 0.5 + ((config.rsi_oversold - rsi) / config.rsi_oversold) * 0.3
            confidence = min(confidence, 0.85)

            # 反転による利益目標
            profit_potential = abs(price_change_1d) * 2.0  # 下落分の2倍の反転

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"reversal_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.BUY,
                    target_price=current_price * (1 + profit_potential / 100),
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="high",  # 反転狙いはリスクが高い
                    risk_reward_ratio=profit_potential / config.stop_loss_percentage,
                    stop_loss_price=current_price
                    * (1 - config.stop_loss_percentage / 100),
                    take_profit_price=current_price * (1 + profit_potential / 100),
                    technical_indicators=indicators,
                    message=f"{symbol} 過売り反転パターン検出",
                )

        return None

    async def analyze_volatility_squeeze(
        self,
        symbol: str,
        config: OpportunityConfig,
        indicators: Dict[str, float],
        current_price: float,
    ) -> Optional[InvestmentOpportunity]:
        """ボラティリティスクイーズ分析"""

        bb_upper = indicators.get("BB_Upper", current_price * 1.02)
        bb_lower = indicators.get("BB_Lower", current_price * 0.98)
        bb_position = indicators.get("BB_Position", 0.5)

        # ボリンジャーバンド幅
        bb_width = (bb_upper - bb_lower) / current_price

        # ボラティリティスクイーズ判定（バンド幅が狭い）
        narrow_bands = bb_width < 0.04  # 4%以下の狭いバンド
        central_position = 0.3 < bb_position < 0.7  # 中央付近

        if narrow_bands and central_position:
            confidence = 0.7 + (0.04 - bb_width) * 10  # バンドが狭いほど信頼度高い
            confidence = min(confidence, 0.85)

            # スクイーズ後の拡張を予想した利益目標
            profit_potential = bb_width * 150  # バンド幅の1.5倍の拡張予想

            if (
                confidence >= config.confidence_threshold
                and profit_potential >= config.profit_potential_threshold
            ):
                return InvestmentOpportunity(
                    opportunity_id=f"squeeze_{symbol}_{int(time.time())}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    opportunity_type=config.opportunity_type,
                    severity=OpportunitySeverity.MEDIUM,
                    recommended_action=TradingAction.HOLD,  # ブレイクアウト方向待ち
                    target_price=None,  # 方向が不明のため未設定
                    current_price=current_price,
                    profit_potential=profit_potential,
                    confidence_score=confidence,
                    time_horizon=config.time_horizon,
                    risk_level="medium",
                    risk_reward_ratio=2.0,  # デフォルト値
                    technical_indicators=indicators,
                    message=f"{symbol} ボラティリティスクイーズ検出",
                )

        return None