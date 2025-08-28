"""
データベースパッケージ
分割されたモジュールからのエクスポートとバックワード互換性を提供

Issue #120: declarative_base()の定義場所の最適化対応
- モジュール分割後の統一インターフェース
- 既存コードとの完全な互換性を維持
"""

# 各モジュールからの必要なクラス・関数をインポート
from .config import DatabaseConfig, TEST_DATABASE_URL
from .connection import ConnectionManager
from .transaction import TransactionManager
from .migration import MigrationManager
from .bulk_operations import BulkOperationsManager
from .utils import DatabaseUtils
from .manager import DatabaseManager

# 新しく追加された分割モジュールからのインポート
from .basic_operations import BasicBulkOperations
from .read_operations import ReadOperations
from .advanced_operations import AdvancedBulkOperations
from .config_security import ConfigSecurityManager
from .migration_executor import MigrationExecutor
from .health_monitor import HealthMonitor
from .optimization import DatabaseOptimizer

# Baseクラスの再エクスポート（互換性維持）
from ..base import Base
from .factory import (
    create_database_manager,
    get_default_database_manager,
    set_default_database_manager,
    get_global_db_manager,
    set_global_db_manager,
    get_session,
    create_test_database_manager,
    create_production_database_manager,
    get_db,
    init_db,
    reset_db,
    init_global_database,
    init_migration,
    create_migration,
    upgrade_db,
    downgrade_db,
    get_current_revision,
    cleanup_global_managers,
    db_manager,  # グローバルインスタンス
)

# バックワード互換性のための全エクスポート
__all__ = [
    # コア設定クラス
    "DatabaseConfig",
    "TEST_DATABASE_URL",
    
    # Baseクラス
    "Base",
    
    # 管理クラス
    "ConnectionManager",
    "TransactionManager", 
    "MigrationManager",
    "BulkOperationsManager",
    "DatabaseUtils",
    "DatabaseManager",
    
    # 新しい分割クラス
    "BasicBulkOperations",
    "ReadOperations",
    "AdvancedBulkOperations",
    "ConfigSecurityManager",
    "MigrationExecutor",
    "HealthMonitor",
    "DatabaseOptimizer",
    
    # ファクトリー関数
    "create_database_manager",
    "get_default_database_manager",
    "set_default_database_manager",
    "get_global_db_manager",
    "set_global_db_manager",
    "get_session",
    "create_test_database_manager",
    "create_production_database_manager",
    
    # 便利関数
    "get_db",
    "init_db",
    "reset_db",
    "init_global_database",
    "init_migration",
    "create_migration",
    "upgrade_db",
    "downgrade_db",
    "get_current_revision",
    "cleanup_global_managers",
    
    # グローバルインスタンス
    "db_manager",
]

# バックワード互換性のための追加エクスポート
# 元のdatabase.pyから直接インポートされていた可能性のある関数・変数

# 元のdatabase.pyにあった拡張機能の復元
def _add_enhanced_features():
    """DatabaseManagerクラスに拡張機能を動的に追加（バックワード互換性）"""
    
    # 元のコードにあった拡張メソッドを復元
    def get_connection_pool_stats(self):
        """接続プール統計情報を取得"""
        return self.connection_manager.get_connection_pool_stats()

    def health_check(self):
        """データベース接続のヘルスチェック"""
        return self.utils.health_check()

    def optimize_performance(self):
        """パフォーマンス最適化設定を適用"""
        return self.utils.optimize_performance()

    def performance_monitor(self, operation_name: str):
        """パフォーマンス監視コンテキストマネージャー"""
        return self.utils.performance_monitor(operation_name)

    def create_factory(self, config_manager=None):
        """ファクトリー方式でDatabaseManagerインスタンスを作成"""
        return DatabaseManager(
            config=DatabaseConfig(config_manager=config_manager),
            config_manager=config_manager,
        )

    def create_performance_optimized_factory(self, config_manager=None):
        """パフォーマンス最適化されたDatabaseManagerを作成"""
        try:
            from ..performance_database import PerformanceOptimizedDatabaseManager

            config = DatabaseConfig(
                config_manager=config_manager, use_performance_config=True
            )

            return PerformanceOptimizedDatabaseManager(config)
        except ImportError:
            from ...utils.logging_config import get_context_logger
            logger = get_context_logger(__name__)
            logger.warning(
                "パフォーマンス最適化モジュールが利用できません。通常版を使用します。"
            )
            return self.create_factory(config_manager)

    # メソッドを動的に追加（元のコードとの互換性維持）
    if not hasattr(DatabaseManager, 'get_connection_pool_stats'):
        DatabaseManager.get_connection_pool_stats = get_connection_pool_stats
    if not hasattr(DatabaseManager, 'health_check'):
        DatabaseManager.health_check = health_check
    if not hasattr(DatabaseManager, 'optimize_performance'):
        DatabaseManager.optimize_performance = optimize_performance
    if not hasattr(DatabaseManager, 'performance_monitor'):
        DatabaseManager.performance_monitor = performance_monitor
    if not hasattr(DatabaseManager, 'create_factory'):
        DatabaseManager.create_factory = create_factory
    if not hasattr(DatabaseManager, 'create_performance_optimized_factory'):
        DatabaseManager.create_performance_optimized_factory = create_performance_optimized_factory


# 拡張機能を適用（元のdatabase.pyとの互換性維持）
_add_enhanced_features()

# モジュール情報
__version__ = "2.0.0"
__author__ = "Day Trade Development Team"
__description__ = "Modularized database management system for day trading application"

# パッケージレベルの設定
import warnings

# 非推奨警告の抑制（必要に応じて）
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sqlalchemy')

# パッケージ初期化ログ
from ...utils.logging_config import get_context_logger
logger = get_context_logger(__name__)
logger.info(
    f"Database package initialized (version {__version__})", 
    extra={
        "modules_loaded": len(__all__),
        "compatibility_mode": True,
    }
)