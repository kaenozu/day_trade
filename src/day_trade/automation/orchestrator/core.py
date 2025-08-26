"""
オーケストレーターコア機能
初期化、設定管理、コンポーネント管理を担当
"""

import os
from typing import Any, Dict, List, Optional

from ...config.trading_mode_config import get_current_trading_config, is_safe_mode
from ...core.portfolio import PortfolioManager
from ...data.stock_fetcher import StockFetcher
from ...models.database import get_default_database_manager
from ...utils.logging_config import get_context_logger
from ...utils.performance_monitor import PerformanceMonitor
from ...utils.stock_name_helper import get_stock_helper
from ..analysis_only_engine import AnalysisOnlyEngine
from .types import OrchestrationConfig

CI_MODE = os.getenv("CI", "false").lower() == "true"

if not CI_MODE:
    from ...data.advanced_ml_engine import ModelConfig, create_advanced_ml_engine
    from ...data.batch_data_fetcher import AdvancedBatchDataFetcher
else:
    # CI環境では軽量ダミークラス使用
    ModelConfig = None
    create_advanced_ml_engine = None
    AdvancedBatchDataFetcher = None

# 並列処理システム
try:
    from ...utils.parallel_executor_manager import ParallelExecutorManager
    PARALLEL_EXECUTOR_AVAILABLE = True
except ImportError:
    PARALLEL_EXECUTOR_AVAILABLE = False

logger = get_context_logger(__name__)


class OrchestratorCore:
    """オーケストレーターコア機能クラス"""

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        ml_config: Optional[ModelConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        コア初期化

        Args:
            config: オーケストレーション設定
            ml_config: MLモデル設定
            config_path: 設定ファイルパス（オプション）
        """
        # セーフモードチェック
        if not is_safe_mode():
            raise ValueError(
                "セーフモードでない場合は、このオーケストレーターは使用できません"
            )

        self.config = config or OrchestrationConfig()
        self.config_path = config_path
        self.trading_config = get_current_trading_config()

        # CI環境では軽量化
        if CI_MODE:
            self.config.enable_ml_engine = False
            self.config.enable_advanced_batch = False
            self.config.enable_realtime_predictions = False
            self.ml_config = None
            logger.info("CI軽量モード: ML機能を無効化")
        else:
            self.ml_config = ml_config or ModelConfig()

        # 基本コンポーネント初期化
        self._initialize_basic_components()

        # 高度コンポーネント初期化
        self._initialize_advanced_components()

        # 実行統計
        self.execution_history = []
        self.performance_metrics = {}

        logger.info("Next-Gen AI Orchestrator Core 初期化完了 - 完全セーフモード")
        logger.info("※ 自動取引機能は一切含まれていません")
        logger.info(
            f"設定: ML={self.config.enable_ml_engine}, Batch={self.config.enable_advanced_batch}"
        )

    def _initialize_basic_components(self):
        """基本コンポーネント初期化"""
        # コアコンポーネント初期化
        self.stock_fetcher = StockFetcher()
        self.stock_helper = get_stock_helper()
        self.analysis_engines: Dict[str, AnalysisOnlyEngine] = {}
        self.db_manager = get_default_database_manager()

    def _initialize_advanced_components(self):
        """高度コンポーネント初期化"""
        # 高度AIコンポーネント初期化（CI環境では無効化）
        if self.config.enable_ml_engine and not CI_MODE:
            from dataclasses import asdict
            self.ml_engine = create_advanced_ml_engine(asdict(self.ml_config))
        else:
            self.ml_engine = None

        if self.config.enable_advanced_batch and not CI_MODE:
            self.batch_fetcher = AdvancedBatchDataFetcher(
                max_workers=self.config.max_workers,
                enable_kafka=False,  # セーフモードではKafka無効
                enable_redis=False,  # セーフモードではRedis無効
            )
        else:
            self.batch_fetcher = None

        # パフォーマンス監視
        if self.config.enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None

        # フォールトトレラント実行（基本的なエラーハンドリング）
        if self.config.enable_fault_tolerance:
            self.fault_executor = {
                "max_retries": self.config.retry_attempts,
                "timeout_seconds": self.config.timeout_seconds,
            }
        else:
            self.fault_executor = None

        # 並列実行マネージャー
        if self.config.enable_parallel_optimization and PARALLEL_EXECUTOR_AVAILABLE:
            self.parallel_manager = ParallelExecutorManager(
                max_thread_workers=self.config.max_thread_workers,
                max_process_workers=self.config.max_process_workers,
                enable_adaptive_sizing=True,
                performance_monitoring=self.config.enable_performance_monitoring,
            )
            logger.info(
                f"並列実行最適化有効: Thread={self.config.max_thread_workers}, "
                f"Process={self.config.max_process_workers}"
            )
        else:
            self.parallel_manager = None
            if not PARALLEL_EXECUTOR_AVAILABLE:
                logger.warning(
                    "ParallelExecutorManagerが利用できません。レガシー並列処理を使用"
                )

    def get_status(self) -> Dict[str, Any]:
        """コアステータス取得"""
        return {
            "safe_mode": is_safe_mode(),
            "trading_disabled": True,
            "automatic_trading": False,
            "analysis_engines": len(self.analysis_engines),
            "config_path": self.config_path,
            "mode": "next_gen_ai_analysis",
            "components": {
                "ml_engine_enabled": self.config.enable_ml_engine,
                "advanced_batch_enabled": self.config.enable_advanced_batch,
                "performance_monitoring": self.config.enable_performance_monitoring,
                "fault_tolerance": self.config.enable_fault_tolerance,
            },
            "execution_count": len(self.execution_history),
            "last_execution": (
                self.execution_history[-1].start_time.isoformat()
                if self.execution_history
                else None
            ),
        }

    def get_execution_history(self, limit: int = 10) -> List[Any]:
        """実行履歴取得"""
        return self.execution_history[-limit:]

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了時のリソースクリーンアップ"""
        try:
            from .resource_manager import ResourceManager
            resource_manager = ResourceManager(self)
            resource_manager.cleanup()
        except Exception as e:
            logger.error(f"__exit__ クリーンアップエラー: {e}")