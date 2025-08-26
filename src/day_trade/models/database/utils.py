"""
データベースユーティリティモジュール（統合版）
ヘルスチェック、パフォーマンス監視、最適化機能

Issue #120: declarative_base()の定義場所の最適化対応
- ユーティリティ機能の責務を明確化、モジュール分割で300行以下に最適化
- パフォーマンス監視機能の拡張
"""

from contextlib import contextmanager
from typing import Any, Dict

from ...utils.logging_config import get_context_logger
from .health_monitor import HealthMonitor
from .optimization import DatabaseOptimizer
from .connection import ConnectionManager
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class DatabaseUtils:
    """データベースユーティリティ統合クラス（300行以下に最適化）"""

    def __init__(
        self, 
        connection_manager: ConnectionManager, 
        transaction_manager: TransactionManager
    ):
        """
        データベースユーティリティの初期化

        Args:
            connection_manager: データベース接続管理
            transaction_manager: トランザクション管理
        """
        self.connection_manager = connection_manager
        self.transaction_manager = transaction_manager
        self.health_monitor = HealthMonitor(connection_manager, transaction_manager)
        self.optimizer = DatabaseOptimizer(connection_manager, transaction_manager)

    # ヘルスチェック・監視のデリゲート
    def health_check(self) -> Dict[str, Any]:
        """
        データベース接続のヘルスチェック

        Returns:
            ヘルスチェック結果の辞書
        """
        return self.health_monitor.health_check()

    def detailed_health_check(self) -> Dict[str, Any]:
        """詳細なヘルスチェック"""
        return self.health_monitor.detailed_health_check()

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """
        パフォーマンス監視コンテキストマネージャー

        Args:
            operation_name: 監視対象操作名

        Usage:
            with db_utils.performance_monitor("complex_query"):
                # データベース操作
                pass
        """
        with self.health_monitor.performance_monitor(operation_name):
            yield

    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報を取得"""
        return self.health_monitor.get_database_info()

    def check_table_exists(self, table_name: str) -> bool:
        """テーブルの存在確認"""
        return self.health_monitor.check_table_exists(table_name)

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """指定されたテーブルの情報を取得"""
        return self.health_monitor.get_table_info(table_name)

    def reset_connection_pool(self):
        """接続プールをリセット"""
        return self.health_monitor.reset_connection_pool()

    def get_connection_metrics(self) -> Dict[str, Any]:
        """接続メトリクスを取得"""
        return self.health_monitor.get_connection_metrics()

    def monitor_query_performance(self, query: str, params: dict = None) -> Dict[str, Any]:
        """クエリパフォーマンスを監視"""
        return self.health_monitor.monitor_query_performance(query, params)

    # 最適化機能のデリゲート
    def optimize_performance(self):
        """
        パフォーマンス最適化設定を適用

        SQLiteデータベースの場合のみ最適化を実行
        """
        return self.optimizer.optimize_performance()

    def optimize_database(self):
        """
        データベースの最適化を実行

        SQLiteの場合はVACUUMとANALYZEを実行
        """
        return self.optimizer.optimize_database()

    def vacuum_analyze(self):
        """
        SQLiteデータベースのVACUUMとANALYZEを実行

        Note:
            この操作は時間がかかる場合があるため、適切なタイミングで実行すること
        """
        return self.optimizer.vacuum_analyze()

    def execute_query(self, query: str, params: dict = None):
        """
        生のSQLクエリを実行（最適化されたクエリ用）

        Args:
            query: 実行するSQLクエリ
            params: クエリパラメータ

        Returns:
            クエリ結果
        """
        return self.optimizer.execute_query(query, params)

    def rebuild_indexes(self):
        """インデックスの再構築（SQLiteの場合のみ）"""
        return self.optimizer.rebuild_indexes()

    def get_sqlite_pragma_settings(self) -> Dict[str, Any]:
        """SQLiteのPRAGMA設定を取得"""
        return self.optimizer.get_sqlite_pragma_settings()

    def optimize_sqlite_for_performance(self):
        """SQLiteのパフォーマンス重視最適化"""
        return self.optimizer.optimize_sqlite_for_performance()

    def optimize_sqlite_for_size(self):
        """SQLiteのサイズ重視最適化"""
        return self.optimizer.optimize_sqlite_for_size()

    def analyze_database_statistics(self) -> Dict[str, Any]:
        """データベース統計情報を分析"""
        return self.optimizer.analyze_database_statistics()

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """最適化の推奨事項を取得"""
        return self.optimizer.get_optimization_recommendations()

    # 統合便利メソッド
    def full_health_report(self) -> Dict[str, Any]:
        """完全なヘルスレポートを生成"""
        report = {
            "basic_health": self.health_check(),
            "detailed_health": self.detailed_health_check(),
            "database_info": self.get_database_info(),
            "connection_metrics": self.get_connection_metrics(),
            "optimization_recommendations": self.get_optimization_recommendations()
        }
        
        if self.connection_manager.config.is_sqlite():
            report["sqlite_settings"] = self.get_sqlite_pragma_settings()
            report["database_statistics"] = self.analyze_database_statistics()
        
        return report

    def maintenance_operation(self, include_vacuum: bool = True, include_analyze: bool = True):
        """
        データベースメンテナンス操作を実行

        Args:
            include_vacuum: VACUUMを含めるか
            include_analyze: ANALYZEを含めるか
        """
        logger.info("Starting database maintenance operation")
        
        operations_performed = []
        
        try:
            # 最適化設定の適用
            self.optimize_performance()
            operations_performed.append("performance_optimization")
            
            # SQLite固有のメンテナンス
            if self.connection_manager.config.is_sqlite():
                if include_vacuum and include_analyze:
                    self.vacuum_analyze()
                    operations_performed.append("vacuum_analyze")
                elif include_vacuum:
                    with self.connection_manager.engine.connect() as connection:
                        from sqlalchemy import text
                        connection.execute(text("VACUUM"))
                    operations_performed.append("vacuum")
                elif include_analyze:
                    with self.connection_manager.engine.connect() as connection:
                        from sqlalchemy import text
                        connection.execute(text("ANALYZE"))
                    operations_performed.append("analyze")
                
                # インデックスの再構築
                self.rebuild_indexes()
                operations_performed.append("reindex")
            
            logger.info(f"Database maintenance completed: {', '.join(operations_performed)}")
            return {"success": True, "operations": operations_performed}
            
        except Exception as e:
            logger.error(f"Database maintenance failed: {e}")
            return {"success": False, "operations": operations_performed, "error": str(e)}

    def get_utils_info(self) -> Dict[str, Any]:
        """ユーティリティの情報を取得"""
        return {
            "components": {
                "health_monitor": "HealthMonitor",
                "optimizer": "DatabaseOptimizer",
                "connection_manager": type(self.connection_manager).__name__,
                "transaction_manager": type(self.transaction_manager).__name__
            },
            "capabilities": [
                "health_monitoring",
                "performance_monitoring",
                "database_optimization",
                "table_management",
                "connection_management",
                "sqlite_specific_features",
                "maintenance_operations"
            ],
            "database_type": self.connection_manager.config.get_database_type(),
            "current_health": self.health_check()["status"]
        }