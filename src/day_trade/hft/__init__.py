"""
高頻度取引(HFT)パッケージ
Issue #366: 高頻度取引最適化エンジン - マイクロ秒レベル執行システム

次世代バッチ・キャッシュ・AI統合により<30μs執行速度を実現する
世界最高レベルのHFTエンジン
"""

# 既存HFTモジュール
from .hft_orchestrator import (
    HFTConfig,
    HFTOrchestrator,
    HFTStrategy,
    create_hft_orchestrator,
)
from .market_data_processor import (
    MarketUpdate,
    OrderBook,
    UltraFastMarketDataProcessor,
    create_market_data_processor,
)
from .microsecond_monitor import (
    LatencyMetrics,
    MicrosecondMonitor,
    PerformanceReport,
    create_microsecond_monitor,
)
from .realtime_decision_engine import (
    DecisionResult,
    MarketFeatures,
    RealtimeDecisionEngine,
    TradingSignal,
    create_decision_engine,
)
from .ultra_fast_executor import (
    ExecutionPlan,
    ExecutionResult,
    OrderEntry,
    UltraFastExecutor,
    create_ultra_fast_executor,
)

# 次世代HFTシステム
from .next_gen_hft_engine import (
    NextGenHFTEngine,
    NextGenHFTConfig,
    NextGenExecutionOrder,
    NextGenExecutionResult,
    HFTExecutionMode,
    HFTMarketData,
    MarketRegime,
    create_next_gen_hft_engine,
)

from .ai_market_predictor import (
    AIMarketPredictor,
    MarketPrediction,
    MarketFeatureSet,
    PredictionModel,
    MarketSignal,
    create_ai_market_predictor,
)

from .integrated_hft_test import (
    IntegratedHFTTestSuite,
    HFTTestResult,
    TestScenario,
)

__all__ = [
    # Core Execution (Legacy)
    "UltraFastExecutor",
    "ExecutionResult",
    "OrderEntry",
    "ExecutionPlan",
    "create_ultra_fast_executor",
    # Orchestration (Legacy)
    "HFTOrchestrator",
    "HFTStrategy",
    "HFTConfig",
    "create_hft_orchestrator",
    # Decision Engine (Legacy)
    "RealtimeDecisionEngine",
    "TradingSignal",
    "MarketFeatures",
    "DecisionResult",
    "create_decision_engine",
    # Monitoring (Legacy)
    "MicrosecondMonitor",
    "LatencyMetrics",
    "PerformanceReport",
    "create_microsecond_monitor",
    # Market Data (Legacy)
    "UltraFastMarketDataProcessor",
    "MarketUpdate",
    "OrderBook",
    "create_market_data_processor",
    
    # Next-Gen HFT Engine
    "NextGenHFTEngine",
    "NextGenHFTConfig",
    "NextGenExecutionOrder", 
    "NextGenExecutionResult",
    "HFTExecutionMode",
    "HFTMarketData",
    "MarketRegime",
    "create_next_gen_hft_engine",
    
    # AI Market Prediction
    "AIMarketPredictor",
    "MarketPrediction",
    "MarketFeatureSet",
    "PredictionModel",
    "MarketSignal",
    "create_ai_market_predictor",
    
    # Testing & Validation
    "IntegratedHFTTestSuite",
    "HFTTestResult",
    "TestScenario",
]

# HFT Performance Constants (Next-Gen Enhanced)
HFT_TARGET_LATENCY_US = 30  # Next-Gen Target: <30 microseconds (improved from 50μs)
HFT_MAX_LATENCY_US = 50  # Alert threshold (improved from 100μs)
HFT_THROUGHPUT_TARGET = 2000000  # 2M orders/second (doubled from 1M)

# Legacy Constants (for compatibility)
HFT_LEGACY_TARGET_LATENCY_US = 50
HFT_LEGACY_MAX_LATENCY_US = 100

# Next-Gen HFT Configuration
HFT_NEXT_GEN_CONFIG = {
    "execution_latency_target_us": HFT_TARGET_LATENCY_US,
    "market_data_buffer_size": 128 * 1024 * 1024,  # 128MB (doubled)
    "order_queue_size": 2 * 1024 * 1024,  # 2M orders (doubled)
    
    # Hardware Optimization
    "cpu_affinity_enabled": True,
    "huge_pages_enabled": True,
    "numa_optimization": True,
    "lock_free_structures": True,
    "zero_copy_networking": True,
    
    # Next-Gen Features
    "batch_optimization_enabled": True,
    "ai_prediction_enabled": True,
    "smart_cache_enabled": True,
    "adaptive_execution_mode": True,
    
    # Performance Targets
    "cache_hit_rate_target": 95.0,  # 95%
    "batch_efficiency_target": 85.0,  # 85%
    "ai_prediction_accuracy_target": 70.0,  # 70%
}

# Legacy Configuration (for backward compatibility)
HFT_DEFAULT_CONFIG = {
    "execution_latency_target_us": HFT_LEGACY_TARGET_LATENCY_US,
    "market_data_buffer_size": 64 * 1024 * 1024,  # 64MB
    "order_queue_size": 1024 * 1024,  # 1M orders
    "cpu_affinity_enabled": True,
    "huge_pages_enabled": True,
    "numa_optimization": True,
    "lock_free_structures": True,
    "zero_copy_networking": True,
}
