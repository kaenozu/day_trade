#!/usr/bin/env python3
"""
Real-Time Trading System Module
リアルタイム取引システムモジュール

Issue #763: リアルタイム特徴量生成と予測パイプラインの構築
完全統合リアルタイム特徴量生成・予測システム
"""

# Legacy imports (existing functionality)
try:
    from .realtime_feed import RealtimeDataFeed, WebSocketClient
except ImportError:
    RealtimeDataFeed = None
    WebSocketClient = None

# New Issue #763 imports
from .feature_engine import (
    MarketDataPoint,
    FeatureValue,
    IncrementalIndicator,
    IncrementalSMA,
    IncrementalEMA,
    IncrementalRSI,
    IncrementalMACD,
    IncrementalBollingerBands,
    RealTimeFeatureEngine
)

from .streaming_processor import (
    StreamConfig,
    StreamMetrics,
    DataFilter,
    SymbolFilter,
    PriceRangeFilter,
    VolumeFilter,
    TimeRangeFilter,
    DataTransformer,
    StreamingDataProcessor
)

from .feature_store import (
    FeatureStoreConfig,
    FeatureStoreMetrics,
    FeatureSerializer,
    FeatureKeyGenerator,
    RealTimeFeatureStore
)

from .async_prediction_pipeline import (
    PredictionResult,
    PipelineConfig,
    PipelineMetrics,
    PredictionModel,
    SimpleMovingAverageModel,
    EnsembleModelWrapper,
    AlertSystem,
    AsyncPredictionPipeline
)

__version__ = "2.0.0"
__author__ = "Day Trade ML System"

# モジュールレベルの便利関数
async def create_realtime_system(
    symbols: list,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    prediction_model: str = "simple_ma"
) -> AsyncPredictionPipeline:
    """
    リアルタイムシステムの簡単セットアップ

    Args:
        symbols: 監視対象銘柄リスト
        redis_host: Redisホスト
        redis_port: Redisポート
        prediction_model: 使用する予測モデル

    Returns:
        設定済みのAsyncPredictionPipeline
    """

    # 設定作成
    feature_store_config = FeatureStoreConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=0,
        default_ttl=3600
    )

    stream_config = StreamConfig(
        url="wss://example.com/stream",  # 実際のURL
        symbols=symbols,
        buffer_size=1000,
        rate_limit=1000
    )

    pipeline_config = PipelineConfig(
        feature_store_config=feature_store_config,
        stream_config=stream_config,
        prediction_interval_ms=100,
        max_concurrent_predictions=5,
        enable_real_time_alerts=True
    )

    # パイプライン作成
    pipeline = AsyncPredictionPipeline(pipeline_config)
    pipeline.switch_model(prediction_model)

    return pipeline


def get_system_info() -> dict:
    """
    システム情報取得

    Returns:
        システム情報辞書
    """
    return {
        "module": "day_trade.realtime",
        "version": __version__,
        "components": [
            "RealTimeFeatureEngine",
            "StreamingDataProcessor",
            "RealTimeFeatureStore",
            "AsyncPredictionPipeline"
        ],
        "features": [
            "インクリメンタル特徴量計算",
            "ストリーミングデータ処理",
            "Redis特徴量ストア",
            "非同期予測パイプライン",
            "リアルタイムアラート"
        ],
        "performance": {
            "target_latency_ms": "<10",
            "target_throughput": "1000+ data points/sec",
            "supported_indicators": [
                "SMA", "EMA", "RSI", "MACD", "BollingerBands"
            ]
        }
    }


# モジュール公開API
__all__ = [
    # Legacy (backward compatibility)
    "RealtimeDataFeed",
    "WebSocketClient",

    # Core Classes
    "MarketDataPoint",
    "FeatureValue",
    "PredictionResult",

    # Feature Engine
    "RealTimeFeatureEngine",
    "IncrementalSMA",
    "IncrementalEMA",
    "IncrementalRSI",
    "IncrementalMACD",
    "IncrementalBollingerBands",

    # Streaming
    "StreamingDataProcessor",
    "StreamConfig",
    "DataTransformer",

    # Feature Store
    "RealTimeFeatureStore",
    "FeatureStoreConfig",

    # Prediction Pipeline
    "AsyncPredictionPipeline",
    "PipelineConfig",
    "SimpleMovingAverageModel",
    "EnsembleModelWrapper",

    # Utilities
    "create_realtime_system",
    "get_system_info"
]
