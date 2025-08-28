#!/usr/bin/env python3
"""
投資機会アラートシステム - データモデル定義
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import OpportunitySeverity, OpportunityType, TimeHorizon, TradingAction


@dataclass
class OpportunityConfig:
    """機会検出設定"""

    rule_id: str
    rule_name: str
    opportunity_type: OpportunityType
    symbols: List[str]
    time_horizon: TimeHorizon

    # 検出しきい値
    confidence_threshold: float = 0.7
    profit_potential_threshold: float = 5.0  # %
    risk_reward_ratio: float = 2.0

    # テクニカル指標設定
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    bollinger_deviation: float = 2.0
    macd_signal_threshold: float = 0
    volume_spike_threshold: float = 2.0

    # ファンダメンタル設定
    pe_ratio_threshold: float = 15.0
    earnings_growth_threshold: float = 20.0
    dividend_yield_threshold: float = 3.0

    # リスク管理設定
    max_position_size: float = 0.1  # 10%
    stop_loss_percentage: float = 5.0  # 5%
    take_profit_percentage: float = 10.0  # 10%

    enabled: bool = True
    check_interval_minutes: int = 15


@dataclass
class InvestmentOpportunity:
    """投資機会"""

    opportunity_id: str
    timestamp: datetime
    symbol: str
    opportunity_type: OpportunityType
    severity: OpportunitySeverity

    # 機会詳細
    recommended_action: TradingAction
    target_price: Optional[float]
    current_price: float
    profit_potential: float  # %
    confidence_score: float
    time_horizon: TimeHorizon

    # リスク・リワード
    risk_level: str
    risk_reward_ratio: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]

    # 分析データ
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    fundamental_data: Dict[str, Any] = field(default_factory=dict)

    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    executed_at: Optional[datetime] = None


@dataclass
class MarketCondition:
    """市場状況"""

    timestamp: datetime
    market_trend: str  # bull, bear, sideways
    volatility_level: str  # low, medium, high
    volume_trend: str  # increasing, decreasing, stable
    sector_performance: Dict[str, float]
    market_sentiment: float  # -1 to 1
    fear_greed_index: Optional[float] = None


@dataclass
class AlertConfig:
    """アラート設定"""

    # 基本アラート設定
    enable_email_notifications: bool = False
    enable_push_notifications: bool = False
    enable_trade_execution: bool = False

    # 機会フィルタリング
    min_confidence_score: float = 0.6
    min_profit_potential: float = 3.0
    min_risk_reward_ratio: float = 1.5

    # 実行設定
    max_opportunities_per_hour: int = 10
    opportunity_cooldown_minutes: int = 30
    alert_history_retention_days: int = 30

    # 市場条件フィルター
    enable_market_condition_filter: bool = True
    avoid_high_volatility_periods: bool = True
    min_market_sentiment: float = -0.3