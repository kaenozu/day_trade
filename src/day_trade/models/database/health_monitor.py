"""
データベースヘルスチェック・監視モジュール
接続状態監視、パフォーマンス監視機能

Issue #120: declarative_base()の定義場所の最適化対応
- ヘルスチェック機能の責務を明確化
- パフォーマンス監視機能の強化
"""

import time
from contextlib import contextmanager
from typing import Any, Dict

from sqlalchemy import text

from ...utils.exceptions import handle_database_exception
from ...utils.logging_config import get_context_logger, log_error_with_context
from .connection import ConnectionManager
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class HealthMonitor:
    """データベースヘルスチェック・監視クラス"""

    def __init__(
        self, 
        connection_manager: ConnectionManager, 
        transaction_manager: TransactionManager
    ):
        """
        ヘルスチェック・監視の初期化

        Args:
            connection_manager: データベース接続管理
            transaction_manager: トランザクション管理
        """
        self.connection_manager = connection_manager
        self.transaction_manager = transaction_manager

    def health_check(self) -> Dict[str, Any]:
        """
        データベース接続のヘルスチェック

        Returns:
            ヘルスチェック結果の辞書
        """
        health_status = {
            "status": "unknown",
            "database_type": self.connection_manager.config.get_database_type(),
            "database_url": self.connection_manager.config.database_url[:50] + "...",
            "connection_pool": {},
            "last_check": time.time(),
            "errors": [],
        }

        try:
            # 基本接続テスト
            with self.transaction_manager.session_scope() as session:
                result = session.execute(text("SELECT 1")).scalar()
                if result == 1:
                    health_status["status"] = "healthy"
                else:
                    health_status["status"] = "unhealthy"
                    health_status["errors"].append("Unexpected query result")

            # 接続プール統計
            health_status["connection_pool"] = self.connection_manager.get_connection_pool_stats()

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["errors"].append(str(e))
            log_error_with_context(e, {"operation": "database_health_check"})

        return health_status

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """
        パフォーマンス監視コンテキストマネージャー

        Args:
            operation_name: 監視対象操作名

        Usage:
            with health_monitor.performance_monitor("complex_query"):
                # データベース操作
                pass
        """
        start_time = time.perf_counter()
        pool_stats_before = self.connection_manager.get_connection_pool_stats()

        try:
            yield
        finally:
            elapsed_time = time.perf_counter() - start_time
            pool_stats_after = self.connection_manager.get_connection_pool_stats()

            # パフォーマンス情報をログ出力
            logger.info(
                "Database operation performance",
                operation=operation_name,
                elapsed_ms=round(elapsed_time * 1000, 2),
                pool_before=pool_stats_before,
                pool_after=pool_stats_after,
            )

    def detailed_health_check(self) -> Dict[str, Any]:
        """
        詳細なヘルスチェック

        Returns:
            詳細なヘルスチェック結果
        """
        health_status = self.health_check()
        
        # 追加の詳細情報
        try:
            # データベース情報の取得
            health_status["database_info"] = self.get_database_info()
            
            # 接続テスト（複数回実行）
            connection_tests = []
            for i in range(3):
                start_time = time.perf_counter()
                try:
                    with self.transaction_manager.session_scope() as session:
                        result = session.execute(text("SELECT 1")).scalar()
                        success = result == 1
                except Exception as e:
                    success = False
                
                elapsed_time = time.perf_counter() - start_time
                connection_tests.append({
                    "test_number": i + 1,
                    "success": success,
                    "response_time_ms": round(elapsed_time * 1000, 2)
                })
            
            health_status["connection_tests"] = connection_tests
            health_status["avg_response_time"] = round(
                sum(test["response_time_ms"] for test in connection_tests) / len(connection_tests), 2
            )
            
        except Exception as e:
            health_status["errors"].append(f"Detailed health check failed: {str(e)}")
            log_error_with_context(e, {"operation": "detailed_health_check"})

        return health_status

    def get_database_info(self) -> Dict[str, Any]:
        """
        データベース基本情報を取得

        Returns:
            データベース情報の辞書
        """
        info = {
            "database_type": self.connection_manager.config.get_database_type(),
            "database_url": self.connection_manager.config.database_url[:50] + "...",
            "is_sqlite": self.connection_manager.config.is_sqlite(),
            "is_in_memory": self.connection_manager.config.is_in_memory(),
            "pool_size": self.connection_manager.config.pool_size,
            "max_overflow": self.connection_manager.config.max_overflow,
            "pool_timeout": self.connection_manager.config.pool_timeout,
            "pool_recycle": self.connection_manager.config.pool_recycle,
            "echo": self.connection_manager.config.echo,
        }

        # 接続プール統計を追加
        pool_stats = self.connection_manager.get_connection_pool_stats()
        info["connection_pool"] = pool_stats

        return info

    def check_table_exists(self, table_name: str) -> bool:
        """
        テーブルの存在確認

        Args:
            table_name: テーブル名

        Returns:
            テーブルが存在する場合True
        """
        try:
            with self.transaction_manager.read_only_session() as session:
                if self.connection_manager.config.is_sqlite():
                    result = session.execute(
                        text(
                            "SELECT name FROM sqlite_master "
                            "WHERE type='table' AND name=:table_name"
                        ),
                        {"table_name": table_name}
                    ).fetchone()
                    return result is not None
                else:
                    # その他のデータベース（INFORMATION_SCHEMA使用）
                    result = session.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_name=:table_name"
                        ),
                        {"table_name": table_name}
                    ).fetchone()
                    return result is not None
                    
        except Exception as e:
            logger.error(f"Failed to check table existence for {table_name}: {e}")
            return False

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        指定されたテーブルの情報を取得

        Args:
            table_name: テーブル名

        Returns:
            テーブル情報の辞書
        """
        try:
            with self.transaction_manager.read_only_session() as session:
                if self.connection_manager.config.is_sqlite():
                    # SQLiteの場合
                    result = session.execute(
                        text(f"PRAGMA table_info({table_name})")
                    ).fetchall()
                    columns = [dict(row._mapping) for row in result]
                    
                    # テーブルサイズ情報
                    count_result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).scalar()
                    
                    return {
                        "table_name": table_name,
                        "columns": columns,
                        "row_count": count_result,
                        "database_type": "sqlite",
                    }
                else:
                    # その他のデータベース（簡略実装）
                    count_result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).scalar()
                    
                    return {
                        "table_name": table_name,
                        "row_count": count_result,
                        "database_type": self.connection_manager.config.get_database_type(),
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise handle_database_exception(e) from e

    def reset_connection_pool(self):
        """
        接続プールをリセット

        接続プールの問題が発生した場合に使用
        """
        try:
            logger.info("Resetting database connection pool")
            self.connection_manager.engine.dispose()
            # 新しい接続が必要な時に自動的に再作成される
            logger.info("Connection pool reset completed")
        except Exception as e:
            logger.error(f"Failed to reset connection pool: {e}")
            raise handle_database_exception(e) from e

    def get_connection_metrics(self) -> Dict[str, Any]:
        """
        接続メトリクスを取得

        Returns:
            接続メトリクスの辞書
        """
        try:
            pool_stats = self.connection_manager.get_connection_pool_stats()
            
            # 接続プールの使用率計算
            pool_size = self.connection_manager.config.pool_size
            checked_out = pool_stats.get("checked_out", 0)
            
            utilization = (checked_out / pool_size * 100) if pool_size > 0 else 0
            
            metrics = {
                "pool_stats": pool_stats,
                "pool_utilization_percent": round(utilization, 2),
                "database_type": self.connection_manager.config.get_database_type(),
                "is_healthy": self.health_check()["status"] == "healthy",
                "measurement_time": time.time()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get connection metrics: {e}")
            raise handle_database_exception(e) from e

    def monitor_query_performance(self, query: str, params: dict = None) -> Dict[str, Any]:
        """
        クエリパフォーマンスを監視

        Args:
            query: 実行するクエリ
            params: クエリパラメータ

        Returns:
            パフォーマンス測定結果
        """
        start_time = time.perf_counter()
        result_info = {"success": False, "error": None, "result_count": 0}
        
        try:
            with self.transaction_manager.read_only_session() as session:
                result = session.execute(text(query), params or {})
                results = result.fetchall()
                result_info["success"] = True
                result_info["result_count"] = len(results)
                
        except Exception as e:
            result_info["error"] = str(e)
            logger.error(f"Query execution failed: {e}")
            
        finally:
            elapsed_time = time.perf_counter() - start_time
            
            performance_data = {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "execution_time_ms": round(elapsed_time * 1000, 2),
                "result_info": result_info,
                "database_type": self.connection_manager.config.get_database_type(),
                "timestamp": time.time()
            }
            
            logger.info("Query performance measured", extra=performance_data)
            return performance_data