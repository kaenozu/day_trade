#!/usr/bin/env python3
"""
Response Models
レスポンスモデル

API応答や分析結果で使用される標準化されたレスポンスモデル
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .unified_models import (
    AnalysisResult,
    AssetType,
    RiskFactor,
)


class ResponseStatus(Enum):
    """レスポンスステータス"""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ErrorType(Enum):
    """エラータイプ"""

    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    DATA_ERROR = "data_error"
    ANALYSIS_ERROR = "analysis_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"


# 基本レスポンス


@dataclass
class BaseResponse:
    """基本レスポンス"""

    request_id: str
    status: ResponseStatus
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorResponse(BaseResponse):
    """エラーレスポンス"""

    error_type: ErrorType
    error_code: str
    error_message: str
    error_details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    retry_possible: bool = False
    retry_after_seconds: Optional[int] = None

    def __init__(self, **kwargs):
        if "status" not in kwargs:
            kwargs["status"] = ResponseStatus.ERROR
        super().__init__(**kwargs)


@dataclass
class SuccessResponse(BaseResponse):
    """成功レスポンス"""

    data: Any
    warnings: List[str] = field(default_factory=list)

    def __init__(self, **kwargs):
        if "status" not in kwargs:
            kwargs["status"] = ResponseStatus.SUCCESS
        super().__init__(**kwargs)


# リスク分析レスポンス


@dataclass
class RiskAnalysisResponse(SuccessResponse):
    """リスク分析レスポンス"""

    analysis_result: AnalysisResult
    confidence_score: float
    recommendation_summary: str
    next_analysis_time: Optional[datetime] = None

    def __init__(self, **kwargs):
        if "data" not in kwargs:
            kwargs["data"] = kwargs.get("analysis_result")
        super().__init__(**kwargs)


@dataclass
class PortfolioRiskResponse(RiskAnalysisResponse):
    """ポートフォリオリスク分析レスポンス"""

    portfolio_metrics: Dict[str, float]
    position_risks: List[Dict[str, Any]]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    diversification_score: Optional[float] = None
    concentration_risk: Optional[Dict[str, float]] = None

    @property
    def total_portfolio_risk(self) -> float:
        """総ポートフォリオリスク"""
        return self.portfolio_metrics.get("total_risk", 0.0)

    @property
    def risk_adjusted_return(self) -> Optional[float]:
        """リスク調整後リターン"""
        return self.portfolio_metrics.get("sharpe_ratio")


@dataclass
class AssetRiskResponse(RiskAnalysisResponse):
    """資産リスク分析レスポンス"""

    asset_symbol: str
    asset_type: AssetType
    risk_factors: List[RiskFactor]
    historical_volatility: Optional[float] = None
    beta: Optional[float] = None
    correlation_to_market: Optional[float] = None
    liquidity_risk: Optional[float] = None
    fundamental_score: Optional[float] = None
    technical_score: Optional[float] = None


@dataclass
class FraudDetectionResponse(RiskAnalysisResponse):
    """不正検知レスポンス"""

    fraud_probability: float
    fraud_indicators: List[Dict[str, Any]]
    transaction_scores: List[Dict[str, float]]
    model_explanations: Dict[str, Any] = field(default_factory=dict)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_fraud_detected(self) -> bool:
        """不正検知フラグ"""
        return self.fraud_probability > 0.5


@dataclass
class StressTestResponse(RiskAnalysisResponse):
    """ストレステストレスポンス"""

    scenario_results: List[Dict[str, Any]]
    worst_case_loss: float
    expected_shortfall: float
    survival_probability: float
    recovery_time_estimate: Optional[int] = None  # days

    def get_scenario_result(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """シナリオ結果取得"""
        for result in self.scenario_results:
            if result.get("scenario_name") == scenario_name:
                return result
        return None


# バッチ処理レスポンス


@dataclass
class BatchAnalysisResponse(BaseResponse):
    """バッチ分析レスポンス"""

    total_analyses: int
    successful_analyses: int
    failed_analyses: int
    individual_results: List[Union[RiskAnalysisResponse, ErrorResponse]]
    batch_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses

    def get_failed_results(self) -> List[ErrorResponse]:
        """失敗結果の取得"""
        return [result for result in self.individual_results if isinstance(result, ErrorResponse)]

    def get_successful_results(self) -> List[RiskAnalysisResponse]:
        """成功結果の取得"""
        return [
            result for result in self.individual_results if isinstance(result, RiskAnalysisResponse)
        ]


@dataclass
class AsyncAnalysisResponse(BaseResponse):
    """非同期分析レスポンス"""

    job_id: str
    estimated_completion_time: Optional[datetime] = None
    progress_url: Optional[str] = None
    result_url: Optional[str] = None
    cancel_url: Optional[str] = None


# データ取得レスポンス


@dataclass
class MarketDataResponse(SuccessResponse):
    """市場データレスポンス"""

    symbols: List[str]
    data_points: int
    start_date: datetime
    end_date: datetime
    market_data: Dict[str, List[Dict[str, Any]]]
    data_quality: Dict[str, float] = field(default_factory=dict)  # symbol -> quality score

    def get_symbol_data(self, symbol: str) -> List[Dict[str, Any]]:
        """シンボル別データ取得"""
        return self.market_data.get(symbol, [])

    def get_data_quality_score(self, symbol: str) -> Optional[float]:
        """データ品質スコア取得"""
        return self.data_quality.get(symbol)


@dataclass
class HistoricalDataResponse(SuccessResponse):
    """履歴データレスポンス"""

    symbol: str
    asset_type: AssetType
    data_points: int
    date_range: Dict[str, str]  # start_date, end_date
    ohlcv_data: List[Dict[str, Any]]
    technical_indicators: Dict[str, List[float]] = field(default_factory=dict)
    fundamental_data: Dict[str, Any] = field(default_factory=dict)

    def get_prices(self) -> List[float]:
        """終値リスト取得"""
        return [point["close"] for point in self.ohlcv_data if "close" in point]

    def get_volumes(self) -> List[float]:
        """出来高リスト取得"""
        return [point["volume"] for point in self.ohlcv_data if "volume" in point]


# 設定・管理レスポンス


@dataclass
class ConfigurationResponse(SuccessResponse):
    """設定レスポンス"""

    config_path: str
    config_value: Any
    previous_value: Optional[Any] = None
    validation_result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheResponse(SuccessResponse):
    """キャッシュレスポンス"""

    operation: str
    cache_key: Optional[str] = None
    cache_value: Optional[Any] = None
    cache_hit: Optional[bool] = None
    cache_stats: Optional[Dict[str, Any]] = None
    ttl_remaining: Optional[int] = None


@dataclass
class AlertResponse(SuccessResponse):
    """アラートレスポンス"""

    alert_id: str
    operation: str
    alert_details: Dict[str, Any] = field(default_factory=dict)
    notification_results: List[Dict[str, Any]] = field(default_factory=list)


# ストリーミング・リアルタイムレスポンス


@dataclass
class StreamingResponse(BaseResponse):
    """ストリーミングレスポンス"""

    stream_id: str
    stream_type: str
    connection_url: str
    subscription_details: Dict[str, Any] = field(default_factory=dict)
    expected_message_rate: Optional[float] = None  # messages per second


@dataclass
class RealtimeAnalysisResponse(RiskAnalysisResponse):
    """リアルタイム分析レスポンス"""

    is_realtime: bool = True
    update_sequence: int = 0
    next_update_time: Optional[datetime] = None
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    alert_triggered: bool = False

    def is_trend_deteriorating(self) -> bool:
        """トレンド悪化判定"""
        return self.trend_analysis.get("direction") == "deteriorating"


# バックテスト・シミュレーションレスポンス


@dataclass
class BacktestResponse(SuccessResponse):
    """バックテストレスポンス"""

    strategy_name: str
    backtest_period: Dict[str, str]  # start_date, end_date
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)
    daily_returns: List[Dict[str, Any]] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def outperformed_benchmark(self) -> bool:
        """ベンチマークを上回ったかどうか"""
        benchmark_return = self.benchmark_comparison.get("total_return", 0)
        return self.total_return > benchmark_return


# システム監視レスポンス


@dataclass
class HealthCheckResponse(BaseResponse):
    """ヘルスチェックレスポンス"""

    system_status: str  # "healthy", "degraded", "unhealthy"
    component_status: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    uptime_seconds: Optional[float] = None
    version_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsResponse(SuccessResponse):
    """メトリクスレスポンス"""

    metrics_data: Dict[str, Any]
    collection_period: Dict[str, str]  # start_time, end_time
    data_points: int
    aggregation_level: str  # "raw", "minute", "hour", "day"


@dataclass
class LogAnalysisResponse(SuccessResponse):
    """ログ分析レスポンス"""

    log_level_counts: Dict[str, int]
    error_patterns: List[Dict[str, Any]]
    performance_issues: List[Dict[str, Any]]
    security_events: List[Dict[str, Any]]
    analysis_period: Dict[str, str]
    recommendations: List[str] = field(default_factory=list)


# レスポンス作成ヘルパー


def create_success_response(
    request_id: str, data: Any, processing_time_ms: Optional[float] = None
) -> SuccessResponse:
    """成功レスポンス作成"""
    return SuccessResponse(request_id=request_id, data=data, processing_time_ms=processing_time_ms)


def create_error_response(
    request_id: str,
    error_type: ErrorType,
    error_code: str,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
) -> ErrorResponse:
    """エラーレスポンス作成"""
    return ErrorResponse(
        request_id=request_id,
        error_type=error_type,
        error_code=error_code,
        error_message=error_message,
        error_details=error_details or {},
    )


def create_portfolio_risk_response(
    request_id: str,
    analysis_result: AnalysisResult,
    portfolio_metrics: Dict[str, float],
    position_risks: List[Dict[str, Any]],
) -> PortfolioRiskResponse:
    """ポートフォリオリスクレスポンス作成"""
    return PortfolioRiskResponse(
        request_id=request_id,
        analysis_result=analysis_result,
        confidence_score=(
            analysis_result.risk_metrics.confidence_score if analysis_result.risk_metrics else 0.0
        ),
        recommendation_summary="Analysis completed successfully",
        portfolio_metrics=portfolio_metrics,
        position_risks=position_risks,
    )


def create_fraud_detection_response(
    request_id: str,
    analysis_result: AnalysisResult,
    fraud_probability: float,
    fraud_indicators: List[Dict[str, Any]],
) -> FraudDetectionResponse:
    """不正検知レスポンス作成"""
    return FraudDetectionResponse(
        request_id=request_id,
        analysis_result=analysis_result,
        confidence_score=(
            analysis_result.risk_metrics.confidence_score if analysis_result.risk_metrics else 0.0
        ),
        recommendation_summary="Fraud detection analysis completed",
        fraud_probability=fraud_probability,
        fraud_indicators=fraud_indicators,
        transaction_scores=[],
    )


def create_async_response(
    request_id: str, job_id: str, estimated_completion_time: Optional[datetime] = None
) -> AsyncAnalysisResponse:
    """非同期レスポンス作成"""
    return AsyncAnalysisResponse(
        request_id=request_id,
        status=ResponseStatus.SUCCESS,
        job_id=job_id,
        estimated_completion_time=estimated_completion_time,
    )


# レスポンス変換ユーティリティ


class ResponseConverter:
    """レスポンス変換ユーティリティ"""

    @staticmethod
    def to_dict(response: BaseResponse) -> Dict[str, Any]:
        """レスポンスを辞書に変換"""
        result = {}

        for field_name in response.__dataclass_fields__:
            value = getattr(response, field_name)

            if isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, datetime):
                result[field_name] = value.isoformat()
            elif hasattr(value, "__dataclass_fields__"):
                result[field_name] = ResponseConverter.to_dict(value)
            elif isinstance(value, list):
                result[field_name] = [
                    (
                        ResponseConverter.to_dict(item)
                        if hasattr(item, "__dataclass_fields__")
                        else item
                    )
                    for item in value
                ]
            else:
                result[field_name] = value

        return result

    @staticmethod
    def to_json(response: BaseResponse) -> str:
        """レスポンスをJSON文字列に変換"""
        import json
        from datetime import datetime
        from decimal import Decimal

        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data = ResponseConverter.to_dict(response)
        return json.dumps(data, default=json_serializer, indent=2)
