"""
高頻度取引(HFT)パッケージ
Issue #366: 高頻度取引最適化エンジン - マイクロ秒レベル執行システム

世界トップレベルの<50μs執行速度を実現するHFTエンジン
"""

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

__all__ = [
    # Core Execution
    "UltraFastExecutor",
    "ExecutionResult",
    "OrderEntry",
    "ExecutionPlan",
    "create_ultra_fast_executor",
    # Orchestration
    "HFTOrchestrator",
    "HFTStrategy",
    "HFTConfig",
    "create_hft_orchestrator",
    # Decision Engine
    "RealtimeDecisionEngine",
    "TradingSignal",
    "MarketFeatures",
    "DecisionResult",
    "create_decision_engine",
    # Monitoring
    "MicrosecondMonitor",
    "LatencyMetrics",
    "PerformanceReport",
    "create_microsecond_monitor",
    # Market Data
    "UltraFastMarketDataProcessor",
    "MarketUpdate",
    "OrderBook",
    "create_market_data_processor",
]

# HFT Performance Constants
HFT_TARGET_LATENCY_US = 50  # Target: <50 microseconds
HFT_MAX_LATENCY_US = 100  # Alert threshold
HFT_THROUGHPUT_TARGET = 1000000  # 1M orders/second

# HFT Configuration
HFT_DEFAULT_CONFIG = {
    "execution_latency_target_us": HFT_TARGET_LATENCY_US,
    "market_data_buffer_size": 64 * 1024 * 1024,  # 64MB
    "order_queue_size": 1024 * 1024,  # 1M orders
    "cpu_affinity_enabled": True,
    "huge_pages_enabled": True,
    "numa_optimization": True,
    "lock_free_structures": True,
    "zero_copy_networking": True,
}
