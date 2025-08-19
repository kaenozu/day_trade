from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

from .enums import MarketDirection, PredictionConfidence, RiskLevel


@dataclass
class MarketSentiment:
    """市場センチメント"""
    sentiment_score: float        # -1.0 to 1.0
    confidence: float            # 0.0 to 1.0
    key_factors: List[str]       # 主要要因
    news_sentiment: float        # ニュースセンチメント
    technical_sentiment: float   # テクニカルセンチメント
    fundamental_sentiment: float # ファンダメンタルセンチメント
    timestamp: datetime


@dataclass
class RiskMetrics:
    """リスク指標"""
    volatility: float              # ボラティリティ
    var_95: float                 # 95% VaR
    expected_shortfall: float     # 期待ショートフォール
    maximum_drawdown: float       # 最大ドローダウン
    sharpe_ratio: float           # シャープレシオ
    beta: float                   # ベータ値
    correlation_market: float     # 市場相関


@dataclass
class PositionRecommendation:
    """ポジション推奨"""
    symbol: str
    direction: MarketDirection
    confidence: PredictionConfidence
    entry_price: float
    target_price: float
    stop_loss_price: float
    position_size_percentage: float  # ポートフォリオに対する%
    risk_level: RiskLevel
    holding_period: str              # 想定保有期間
    rationale: str                   # 根拠


@dataclass
class BacktestResult:
    """バックテスト結果"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    average_trade_return: float


@dataclass
class NextMorningPrediction:
    """翌朝場予測"""
    symbol: str
    prediction_date: datetime
    market_direction: MarketDirection
    predicted_change_percent: float
    confidence: PredictionConfidence
    confidence_score: float
    market_sentiment: MarketSentiment
    risk_metrics: RiskMetrics
    position_recommendation: PositionRecommendation
    supporting_data: Dict[str, Any]
    model_used: str
    data_sources: List[str]
