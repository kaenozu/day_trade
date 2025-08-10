#!/usr/bin/env python3
"""
Request Models
リクエストモデル

API呼び出しや分析要求で使用される標準化されたリクエストモデル
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .unified_models import Asset, Portfolio, Position, AssetType, RiskLevel

class AnalysisType(Enum):
    """分析タイプ"""
    PORTFOLIO_RISK = "portfolio_risk"
    POSITION_RISK = "position_risk"
    ASSET_RISK = "asset_risk"
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    FRAUD_DETECTION = "fraud_detection"
    COMPLIANCE_CHECK = "compliance_check"
    STRESS_TEST = "stress_test"

class TimeFrame(Enum):
    """時間枠"""
    REALTIME = "realtime"
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class Priority(Enum):
    """優先度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# 基本リクエストクラス

@dataclass
class BaseRequest:
    """基本リクエスト"""
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

# リスク分析リクエスト

@dataclass
class RiskAnalysisRequest(BaseRequest):
    """リスク分析リクエスト"""
    analysis_type: AnalysisType
    target: Union[Asset, Position, Portfolio]
    time_frame: TimeFrame = TimeFrame.DAILY
    confidence_level: float = 0.95
    include_stress_test: bool = False
    include_scenario_analysis: bool = False
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.5 <= self.confidence_level <= 0.99:
            raise ValueError("confidence_level must be between 0.5 and 0.99")

@dataclass
class PortfolioRiskRequest(RiskAnalysisRequest):
    """ポートフォリオリスク分析リクエスト"""
    portfolio: Portfolio
    calculate_var: bool = True
    calculate_cvar: bool = True
    calculate_beta: bool = True
    calculate_correlation: bool = True
    benchmark_symbol: Optional[str] = None

    def __init__(self, **kwargs):
        if 'target' not in kwargs:
            kwargs['target'] = kwargs.get('portfolio')
        if 'analysis_type' not in kwargs:
            kwargs['analysis_type'] = AnalysisType.PORTFOLIO_RISK
        super().__init__(**kwargs)

@dataclass
class AssetRiskRequest(RiskAnalysisRequest):
    """資産リスク分析リクエスト"""
    asset: Asset
    market_data_days: int = 252
    include_fundamental_analysis: bool = False
    include_technical_analysis: bool = True

    def __init__(self, **kwargs):
        if 'target' not in kwargs:
            kwargs['target'] = kwargs.get('asset')
        if 'analysis_type' not in kwargs:
            kwargs['analysis_type'] = AnalysisType.ASSET_RISK
        super().__init__(**kwargs)

@dataclass
class FraudDetectionRequest(RiskAnalysisRequest):
    """不正検知リクエスト"""
    transaction_data: List[Dict[str, Any]]
    model_type: str = "ensemble"
    anomaly_threshold: float = 0.5
    include_explanation: bool = True

    def __init__(self, **kwargs):
        if 'analysis_type' not in kwargs:
            kwargs['analysis_type'] = AnalysisType.FRAUD_DETECTION
        super().__init__(**kwargs)

# バッチ処理リクエスト

@dataclass
class BatchAnalysisRequest(BaseRequest):
    """バッチ分析リクエスト"""
    analysis_requests: List[RiskAnalysisRequest]
    parallel_execution: bool = True
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 3600
    stop_on_error: bool = False

    def __post_init__(self):
        if self.max_concurrent_tasks <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

@dataclass
class ScheduledAnalysisRequest(BaseRequest):
    """スケジュール分析リクエスト"""
    analysis_request: RiskAnalysisRequest
    schedule_expression: str  # CRON式
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    enabled: bool = True
    max_executions: Optional[int] = None

    def __post_init__(self):
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

# データ取得リクエスト

@dataclass
class MarketDataRequest(BaseRequest):
    """市場データリクエスト"""
    symbols: List[str]
    data_type: str  # "price", "volume", "ohlc", "all"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    interval: str = "1d"  # "1m", "5m", "1h", "1d", etc.
    include_extended_hours: bool = False

    def __post_init__(self):
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

@dataclass
class HistoricalDataRequest(BaseRequest):
    """履歴データリクエスト"""
    symbol: str
    asset_type: AssetType
    lookback_days: int = 252
    data_fields: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    include_dividends: bool = False
    include_splits: bool = False

    def __post_init__(self):
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if not self.data_fields:
            raise ValueError("data_fields cannot be empty")

# 設定・管理リクエスト

@dataclass
class ConfigurationRequest(BaseRequest):
    """設定リクエスト"""
    operation: str  # "get", "set", "update", "delete"
    config_path: str
    config_value: Optional[Any] = None
    validate_config: bool = True
    backup_existing: bool = True

@dataclass
class CacheRequest(BaseRequest):
    """キャッシュリクエスト"""
    operation: str  # "get", "set", "delete", "clear", "stats"
    cache_key: Optional[str] = None
    cache_value: Optional[Any] = None
    ttl_seconds: Optional[int] = None
    cache_provider: Optional[str] = None  # "redis", "memory", "file"

@dataclass
class AlertRequest(BaseRequest):
    """アラートリクエスト"""
    operation: str  # "create", "update", "acknowledge", "resolve", "delete"
    alert_id: Optional[str] = None
    alert_level: Optional[RiskLevel] = None
    title: Optional[str] = None
    message: Optional[str] = None
    target: Optional[Union[Asset, Position, Portfolio]] = None
    notification_channels: List[str] = field(default_factory=list)  # "email", "slack", "webhook"

# ストリーミング・リアルタイムリクエスト

@dataclass
class StreamingRequest(BaseRequest):
    """ストリーミングリクエスト"""
    stream_type: str  # "market_data", "risk_alerts", "analysis_results"
    symbols: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    buffer_size: int = 1000
    compression: bool = False

@dataclass
class RealtimeAnalysisRequest(BaseRequest):
    """リアルタイム分析リクエスト"""
    analysis_type: AnalysisType
    target: Union[Asset, Position, Portfolio]
    update_interval_seconds: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    auto_stop_conditions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.update_interval_seconds <= 0:
            raise ValueError("update_interval_seconds must be positive")

# バックテスト・シミュレーションリクエスト

@dataclass
class BacktestRequest(BaseRequest):
    """バックテストリクエスト"""
    strategy_config: Dict[str, Any]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    benchmark_symbol: Optional[str] = None
    commission_rate: float = 0.001
    slippage_rate: float = 0.001
    include_transaction_costs: bool = True

    def __post_init__(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

@dataclass
class StressTestRequest(BaseRequest):
    """ストレステストリクエスト"""
    portfolio: Portfolio
    stress_scenarios: List[Dict[str, Any]]
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    include_monte_carlo: bool = True
    monte_carlo_iterations: int = 10000
    correlation_adjustment: bool = True

    def __post_init__(self):
        if not self.stress_scenarios:
            raise ValueError("stress_scenarios cannot be empty")
        if self.monte_carlo_iterations <= 0:
            raise ValueError("monte_carlo_iterations must be positive")

# リクエスト検証

class RequestValidator:
    """リクエスト検証"""

    @staticmethod
    def validate_base_request(request: BaseRequest) -> List[str]:
        """基本リクエスト検証"""
        errors = []

        if not request.request_id or len(request.request_id.strip()) == 0:
            errors.append("request_id is required")

        if request.timestamp > datetime.now() + timedelta(minutes=5):
            errors.append("timestamp cannot be too far in the future")

        return errors

    @staticmethod
    def validate_risk_analysis_request(request: RiskAnalysisRequest) -> List[str]:
        """リスク分析リクエスト検証"""
        errors = RequestValidator.validate_base_request(request)

        if not hasattr(request, 'target') or request.target is None:
            errors.append("analysis target is required")

        if not 0.5 <= request.confidence_level <= 0.99:
            errors.append("confidence_level must be between 0.5 and 0.99")

        return errors

    @staticmethod
    def validate_market_data_request(request: MarketDataRequest) -> List[str]:
        """市場データリクエスト検証"""
        errors = RequestValidator.validate_base_request(request)

        if not request.symbols:
            errors.append("symbols list cannot be empty")

        # シンボルの妥当性チェック
        for symbol in request.symbols:
            if not symbol or len(symbol.strip()) == 0:
                errors.append(f"invalid symbol: {symbol}")
            elif len(symbol) > 20:
                errors.append(f"symbol too long: {symbol}")

        # 日付範囲チェック
        if request.start_date and request.end_date:
            if request.start_date >= request.end_date:
                errors.append("start_date must be before end_date")

            # 過去5年以内の制限
            max_lookback = datetime.now() - timedelta(days=5*365)
            if request.start_date < max_lookback:
                errors.append("start_date cannot be more than 5 years ago")

        return errors

# リクエスト作成ヘルパー関数

def create_portfolio_risk_request(
    portfolio: Portfolio,
    confidence_level: float = 0.95,
    include_var: bool = True,
    priority: Priority = Priority.MEDIUM
) -> PortfolioRiskRequest:
    """ポートフォリオリスクリクエスト作成"""
    import uuid

    return PortfolioRiskRequest(
        request_id=str(uuid.uuid4()),
        portfolio=portfolio,
        confidence_level=confidence_level,
        calculate_var=include_var,
        priority=priority
    )

def create_asset_risk_request(
    asset: Asset,
    market_data_days: int = 252,
    priority: Priority = Priority.MEDIUM
) -> AssetRiskRequest:
    """資産リスクリクエスト作成"""
    import uuid

    return AssetRiskRequest(
        request_id=str(uuid.uuid4()),
        asset=asset,
        market_data_days=market_data_days,
        priority=priority
    )

def create_fraud_detection_request(
    transaction_data: List[Dict[str, Any]],
    threshold: float = 0.5,
    priority: Priority = Priority.HIGH
) -> FraudDetectionRequest:
    """不正検知リクエスト作成"""
    import uuid

    return FraudDetectionRequest(
        request_id=str(uuid.uuid4()),
        transaction_data=transaction_data,
        anomaly_threshold=threshold,
        priority=priority,
        target=None  # ダミー値
    )

def create_market_data_request(
    symbols: List[str],
    data_type: str = "all",
    lookback_days: int = 30
) -> MarketDataRequest:
    """市場データリクエスト作成"""
    import uuid

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    return MarketDataRequest(
        request_id=str(uuid.uuid4()),
        symbols=symbols,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date
    )
