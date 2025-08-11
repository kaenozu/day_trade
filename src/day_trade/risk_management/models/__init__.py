"""
Risk Management Models
リスク管理モデル

統一データモデル、リクエスト・レスポンスモデル、バリデーション機能を提供
"""

# 統一データモデル
# リクエストモデル
from .request_models import (
    AlertRequest,
    # 列挙型
    AnalysisType,
    AssetRiskRequest,
    # バックテスト・シミュレーションリクエスト
    BacktestRequest,
    # 基本リクエスト
    BaseRequest,
    # バッチ処理リクエスト
    BatchAnalysisRequest,
    CacheRequest,
    # 設定・管理リクエスト
    ConfigurationRequest,
    FraudDetectionRequest,
    HistoricalDataRequest,
    # データ取得リクエスト
    MarketDataRequest,
    PortfolioRiskRequest,
    Priority,
    RealtimeAnalysisRequest,
    # バリデーション
    RequestValidator,
    # リスク分析リクエスト
    RiskAnalysisRequest,
    ScheduledAnalysisRequest,
    # ストリーミング・リアルタイムリクエスト
    StreamingRequest,
    StressTestRequest,
    TimeFrame,
    create_asset_risk_request,
    create_fraud_detection_request,
    create_market_data_request,
    # ヘルパー関数
    create_portfolio_risk_request,
)

# レスポンスモデル
from .response_models import (
    AlertResponse,
    AssetRiskResponse,
    AsyncAnalysisResponse,
    # バックテスト・シミュレーションレスポンス
    BacktestResponse,
    # 基本レスポンス
    BaseResponse,
    # バッチ処理レスポンス
    BatchAnalysisResponse,
    CacheResponse,
    # 設定・管理レスポンス
    ConfigurationResponse,
    ErrorResponse,
    ErrorType,
    FraudDetectionResponse,
    # システム監視レスポンス
    HealthCheckResponse,
    HistoricalDataResponse,
    LogAnalysisResponse,
    # データ取得レスポンス
    MarketDataResponse,
    MetricsResponse,
    PortfolioRiskResponse,
    RealtimeAnalysisResponse,
    # ユーティリティ
    ResponseConverter,
    # 列挙型
    ResponseStatus,
    # リスク分析レスポンス
    RiskAnalysisResponse,
    # ストリーミング・リアルタイムレスポンス
    StreamingResponse,
    StressTestResponse,
    SuccessResponse,
    create_async_response,
    create_error_response,
    create_fraud_detection_response,
    create_portfolio_risk_response,
    # ヘルパー関数
    create_success_response,
)
from .unified_models import (
    AnalysisResult,
    AnalysisStatus,
    # エンティティモデル
    Asset,
    # 列挙型
    AssetType,
    # バッチ処理モデル
    BatchJob,
    MarketData,
    # ユーティリティ
    ModelConverter,
    Portfolio,
    Position,
    PositionType,
    ProcessingStats,
    RiskAlert,
    RiskFactor,
    RiskLevel,
    # リスク分析モデル
    RiskMetrics,
    # バリデーション
    ValidationResult,
    # データ変換関数
    asset_from_dict,
    portfolio_from_dict,
    position_from_dict,
    # バリデーション関数
    validate_asset,
    validate_portfolio,
    validate_position,
    validate_risk_metrics,
)

__all__ = [
    # 統一データモデル
    "AssetType",
    "PositionType",
    "RiskLevel",
    "AnalysisStatus",
    "Asset",
    "MarketData",
    "Position",
    "Portfolio",
    "RiskMetrics",
    "RiskFactor",
    "RiskAlert",
    "AnalysisResult",
    "BatchJob",
    "ProcessingStats",
    "ValidationResult",
    "asset_from_dict",
    "position_from_dict",
    "portfolio_from_dict",
    "validate_asset",
    "validate_position",
    "validate_portfolio",
    "validate_risk_metrics",
    "ModelConverter",
    # リクエストモデル
    "AnalysisType",
    "TimeFrame",
    "Priority",
    "BaseRequest",
    "RiskAnalysisRequest",
    "PortfolioRiskRequest",
    "AssetRiskRequest",
    "FraudDetectionRequest",
    "BatchAnalysisRequest",
    "ScheduledAnalysisRequest",
    "MarketDataRequest",
    "HistoricalDataRequest",
    "ConfigurationRequest",
    "CacheRequest",
    "AlertRequest",
    "StreamingRequest",
    "RealtimeAnalysisRequest",
    "BacktestRequest",
    "StressTestRequest",
    "RequestValidator",
    "create_portfolio_risk_request",
    "create_asset_risk_request",
    "create_fraud_detection_request",
    "create_market_data_request",
    # レスポンスモデル
    "ResponseStatus",
    "ErrorType",
    "BaseResponse",
    "ErrorResponse",
    "SuccessResponse",
    "RiskAnalysisResponse",
    "PortfolioRiskResponse",
    "AssetRiskResponse",
    "FraudDetectionResponse",
    "StressTestResponse",
    "BatchAnalysisResponse",
    "AsyncAnalysisResponse",
    "MarketDataResponse",
    "HistoricalDataResponse",
    "ConfigurationResponse",
    "CacheResponse",
    "AlertResponse",
    "StreamingResponse",
    "RealtimeAnalysisResponse",
    "BacktestResponse",
    "HealthCheckResponse",
    "MetricsResponse",
    "LogAnalysisResponse",
    "create_success_response",
    "create_error_response",
    "create_portfolio_risk_response",
    "create_fraud_detection_response",
    "create_async_response",
    "ResponseConverter",
]
