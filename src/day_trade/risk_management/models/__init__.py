"""
Risk Management Models
リスク管理モデル

統一データモデル、リクエスト・レスポンスモデル、バリデーション機能を提供
"""

# 統一データモデル
from .unified_models import (
    # 列挙型
    AssetType,
    PositionType,
    RiskLevel,
    AnalysisStatus,

    # エンティティモデル
    Asset,
    MarketData,
    Position,
    Portfolio,

    # リスク分析モデル
    RiskMetrics,
    RiskFactor,
    RiskAlert,
    AnalysisResult,

    # バッチ処理モデル
    BatchJob,
    ProcessingStats,

    # バリデーション
    ValidationResult,

    # データ変換関数
    asset_from_dict,
    position_from_dict,
    portfolio_from_dict,

    # バリデーション関数
    validate_asset,
    validate_position,
    validate_portfolio,
    validate_risk_metrics,

    # ユーティリティ
    ModelConverter
)

# リクエストモデル
from .request_models import (
    # 列挙型
    AnalysisType,
    TimeFrame,
    Priority,

    # 基本リクエスト
    BaseRequest,

    # リスク分析リクエスト
    RiskAnalysisRequest,
    PortfolioRiskRequest,
    AssetRiskRequest,
    FraudDetectionRequest,

    # バッチ処理リクエスト
    BatchAnalysisRequest,
    ScheduledAnalysisRequest,

    # データ取得リクエスト
    MarketDataRequest,
    HistoricalDataRequest,

    # 設定・管理リクエスト
    ConfigurationRequest,
    CacheRequest,
    AlertRequest,

    # ストリーミング・リアルタイムリクエスト
    StreamingRequest,
    RealtimeAnalysisRequest,

    # バックテスト・シミュレーションリクエスト
    BacktestRequest,
    StressTestRequest,

    # バリデーション
    RequestValidator,

    # ヘルパー関数
    create_portfolio_risk_request,
    create_asset_risk_request,
    create_fraud_detection_request,
    create_market_data_request
)

# レスポンスモデル
from .response_models import (
    # 列挙型
    ResponseStatus,
    ErrorType,

    # 基本レスポンス
    BaseResponse,
    ErrorResponse,
    SuccessResponse,

    # リスク分析レスポンス
    RiskAnalysisResponse,
    PortfolioRiskResponse,
    AssetRiskResponse,
    FraudDetectionResponse,
    StressTestResponse,

    # バッチ処理レスポンス
    BatchAnalysisResponse,
    AsyncAnalysisResponse,

    # データ取得レスポンス
    MarketDataResponse,
    HistoricalDataResponse,

    # 設定・管理レスポンス
    ConfigurationResponse,
    CacheResponse,
    AlertResponse,

    # ストリーミング・リアルタイムレスポンス
    StreamingResponse,
    RealtimeAnalysisResponse,

    # バックテスト・シミュレーションレスポンス
    BacktestResponse,

    # システム監視レスポンス
    HealthCheckResponse,
    MetricsResponse,
    LogAnalysisResponse,

    # ヘルパー関数
    create_success_response,
    create_error_response,
    create_portfolio_risk_response,
    create_fraud_detection_response,
    create_async_response,

    # ユーティリティ
    ResponseConverter
)

__all__ = [
    # 統一データモデル
    "AssetType", "PositionType", "RiskLevel", "AnalysisStatus",
    "Asset", "MarketData", "Position", "Portfolio",
    "RiskMetrics", "RiskFactor", "RiskAlert", "AnalysisResult",
    "BatchJob", "ProcessingStats", "ValidationResult",
    "asset_from_dict", "position_from_dict", "portfolio_from_dict",
    "validate_asset", "validate_position", "validate_portfolio", "validate_risk_metrics",
    "ModelConverter",

    # リクエストモデル
    "AnalysisType", "TimeFrame", "Priority",
    "BaseRequest", "RiskAnalysisRequest", "PortfolioRiskRequest", "AssetRiskRequest", "FraudDetectionRequest",
    "BatchAnalysisRequest", "ScheduledAnalysisRequest",
    "MarketDataRequest", "HistoricalDataRequest",
    "ConfigurationRequest", "CacheRequest", "AlertRequest",
    "StreamingRequest", "RealtimeAnalysisRequest",
    "BacktestRequest", "StressTestRequest",
    "RequestValidator",
    "create_portfolio_risk_request", "create_asset_risk_request",
    "create_fraud_detection_request", "create_market_data_request",

    # レスポンスモデル
    "ResponseStatus", "ErrorType",
    "BaseResponse", "ErrorResponse", "SuccessResponse",
    "RiskAnalysisResponse", "PortfolioRiskResponse", "AssetRiskResponse",
    "FraudDetectionResponse", "StressTestResponse",
    "BatchAnalysisResponse", "AsyncAnalysisResponse",
    "MarketDataResponse", "HistoricalDataResponse",
    "ConfigurationResponse", "CacheResponse", "AlertResponse",
    "StreamingResponse", "RealtimeAnalysisResponse",
    "BacktestResponse",
    "HealthCheckResponse", "MetricsResponse", "LogAnalysisResponse",
    "create_success_response", "create_error_response", "create_portfolio_risk_response",
    "create_fraud_detection_response", "create_async_response",
    "ResponseConverter"
]
