"""
データベース最適化モジュール
パフォーマンス最適化、VACUUM、クエリ実行機能

Issue #120: declarative_base()の定義場所の最適化対応
- データベース最適化の責務を明確化
- SQLite最適化機能の強化
"""

from typing import Dict, Any

from sqlalchemy import text

from ...utils.exceptions import handle_database_exception
from ...utils.logging_config import get_context_logger, log_error_with_context
from .connection import ConnectionManager
from .transaction import TransactionManager

logger = get_context_logger(__name__)


class DatabaseOptimizer:
    """データベース最適化クラス"""

    def __init__(
        self, 
        connection_manager: ConnectionManager, 
        transaction_manager: TransactionManager
    ):
        """
        データベース最適化の初期化

        Args:
            connection_manager: データベース接続管理
            transaction_manager: トランザクション管理
        """
        self.connection_manager = connection_manager
        self.transaction_manager = transaction_manager

    def optimize_performance(self):
        """
        パフォーマンス最適化設定を適用

        SQLiteデータベースの場合のみ最適化を実行
        """
        if not self.connection_manager.config.is_sqlite():
            logger.info("Performance optimization only available for SQLite")
            return

        try:
            with self.transaction_manager.session_scope() as session:
                # SQLiteの高度な最適化設定
                optimizations = [
                    "PRAGMA journal_mode=WAL",
                    "PRAGMA synchronous=NORMAL",
                    "PRAGMA cache_size=10000",
                    "PRAGMA temp_store=memory",
                    "PRAGMA mmap_size=268435456",
                    "PRAGMA page_size=4096",
                    "PRAGMA auto_vacuum=INCREMENTAL",
                    "PRAGMA incremental_vacuum(1000)",
                ]

                for pragma in optimizations:
                    session.execute(text(pragma))

                logger.info("Database performance optimizations applied")

        except Exception as e:
            log_error_with_context(e, {"operation": "database_optimization"})

    def optimize_database(self):
        """
        データベースの最適化を実行

        SQLiteの場合はVACUUMとANALYZEを実行
        """
        operation_logger = logger
        operation_logger.info("Starting database optimization")

        if self.connection_manager.config.is_sqlite():
            with self.connection_manager.engine.connect() as connection:
                # VACUUM操作でデータベースファイルを最適化
                operation_logger.info("Executing VACUUM")
                connection.execute(text("VACUUM"))
                # ANALYZE操作で統計情報を更新
                operation_logger.info("Executing ANALYZE")
                connection.execute(text("ANALYZE"))
        else:
            operation_logger.info("Database optimization not implemented for this database type")

        operation_logger.info("Database optimization completed")

    def vacuum_analyze(self):
        """
        SQLiteデータベースのVACUUMとANALYZEを実行

        Note:
            この操作は時間がかかる場合があるため、適切なタイミングで実行すること
        """
        if not self.connection_manager.config.is_sqlite():
            logger.warning("VACUUM/ANALYZE is only available for SQLite databases")
            return

        try:
            with self.connection_manager.engine.connect() as connection:
                logger.info("Starting VACUUM operation")
                connection.execute(text("VACUUM"))
                
                logger.info("Starting ANALYZE operation")
                connection.execute(text("ANALYZE"))
                
                logger.info("VACUUM/ANALYZE completed successfully")
                
        except Exception as e:
            logger.error(f"VACUUM/ANALYZE failed: {e}")
            raise handle_database_exception(e) from e

    def execute_query(self, query: str, params: dict = None):
        """
        生のSQLクエリを実行（最適化されたクエリ用）

        Args:
            query: 実行するSQLクエリ
            params: クエリパラメータ

        Returns:
            クエリ結果
        """
        operation_logger = logger
        operation_logger.info("Executing raw SQL query")

        with self.connection_manager.engine.connect() as connection:
            result = connection.execute(text(query), params or {})
            results = result.fetchall()
            operation_logger.info(
                "Query executed", extra={"result_count": len(results)}
            )
            return results

    def rebuild_indexes(self):
        """
        インデックスの再構築（SQLiteの場合のみ）
        """
        if not self.connection_manager.config.is_sqlite():
            logger.info("Index rebuilding only available for SQLite")
            return

        try:
            with self.connection_manager.engine.connect() as connection:
                logger.info("Starting index rebuilding")
                connection.execute(text("REINDEX"))
                logger.info("Index rebuilding completed")
                
        except Exception as e:
            logger.error(f"Index rebuilding failed: {e}")
            raise handle_database_exception(e) from e

    def get_sqlite_pragma_settings(self) -> Dict[str, Any]:
        """
        SQLiteのPRAGMA設定を取得

        Returns:
            PRAGMA設定の辞書
        """
        if not self.connection_manager.config.is_sqlite():
            return {"error": "PRAGMA settings only available for SQLite"}

        pragma_queries = [
            ("journal_mode", "PRAGMA journal_mode"),
            ("synchronous", "PRAGMA synchronous"),
            ("cache_size", "PRAGMA cache_size"),
            ("temp_store", "PRAGMA temp_store"),
            ("mmap_size", "PRAGMA mmap_size"),
            ("page_size", "PRAGMA page_size"),
            ("auto_vacuum", "PRAGMA auto_vacuum"),
            ("encoding", "PRAGMA encoding"),
            ("foreign_keys", "PRAGMA foreign_keys"),
        ]

        settings = {}
        try:
            with self.connection_manager.engine.connect() as connection:
                for setting_name, pragma_query in pragma_queries:
                    try:
                        result = connection.execute(text(pragma_query)).scalar()
                        settings[setting_name] = result
                    except Exception as e:
                        settings[setting_name] = f"Error: {str(e)}"
                        
        except Exception as e:
            logger.error(f"Failed to get PRAGMA settings: {e}")
            return {"error": f"Failed to retrieve settings: {str(e)}"}

        return settings

    def optimize_sqlite_for_performance(self):
        """
        SQLiteのパフォーマンス重視最適化
        """
        if not self.connection_manager.config.is_sqlite():
            logger.info("SQLite performance optimization only available for SQLite")
            return

        performance_pragmas = [
            "PRAGMA journal_mode=WAL",           # WALモードで同時読み書き向上
            "PRAGMA synchronous=NORMAL",         # 同期レベルを中程度に
            "PRAGMA cache_size=-64000",          # 64MB キャッシュサイズ
            "PRAGMA temp_store=MEMORY",          # 一時ファイルをメモリに
            "PRAGMA mmap_size=536870912",        # 512MB メモリマップサイズ
            "PRAGMA optimize",                   # 統計情報の最適化
        ]

        try:
            with self.connection_manager.engine.connect() as connection:
                logger.info("Applying SQLite performance optimizations")
                for pragma in performance_pragmas:
                    logger.debug(f"Executing: {pragma}")
                    connection.execute(text(pragma))
                
                logger.info("SQLite performance optimizations applied successfully")
                
        except Exception as e:
            logger.error(f"SQLite performance optimization failed: {e}")
            raise handle_database_exception(e) from e

    def optimize_sqlite_for_size(self):
        """
        SQLiteのサイズ重視最適化
        """
        if not self.connection_manager.config.is_sqlite():
            logger.info("SQLite size optimization only available for SQLite")
            return

        size_pragmas = [
            "PRAGMA auto_vacuum=FULL",           # 自動バキュームを完全モードに
            "PRAGMA cache_size=2000",            # キャッシュサイズを小さく
            "PRAGMA temp_store=FILE",            # 一時ファイルをディスクに
        ]

        try:
            with self.connection_manager.engine.connect() as connection:
                logger.info("Applying SQLite size optimizations")
                for pragma in size_pragmas:
                    logger.debug(f"Executing: {pragma}")
                    connection.execute(text(pragma))
                
                # VACUUMでファイルサイズを削減
                logger.info("Executing VACUUM for size optimization")
                connection.execute(text("VACUUM"))
                
                logger.info("SQLite size optimizations applied successfully")
                
        except Exception as e:
            logger.error(f"SQLite size optimization failed: {e}")
            raise handle_database_exception(e) from e

    def analyze_database_statistics(self) -> Dict[str, Any]:
        """
        データベース統計情報を分析

        Returns:
            統計情報の辞書
        """
        try:
            stats = {
                "database_type": self.connection_manager.config.get_database_type(),
                "optimization_applied": False,
                "statistics": {}
            }

            if self.connection_manager.config.is_sqlite():
                with self.connection_manager.engine.connect() as connection:
                    # SQLite固有の統計
                    queries = [
                        ("database_pages", "PRAGMA page_count"),
                        ("page_size", "PRAGMA page_size"),
                        ("freelist_count", "PRAGMA freelist_count"),
                        ("schema_version", "PRAGMA schema_version"),
                        ("user_version", "PRAGMA user_version"),
                    ]
                    
                    for stat_name, query in queries:
                        try:
                            result = connection.execute(text(query)).scalar()
                            stats["statistics"][stat_name] = result
                        except Exception as e:
                            stats["statistics"][stat_name] = f"Error: {str(e)}"
                    
                    # データベースサイズ計算
                    page_count = stats["statistics"].get("database_pages", 0)
                    page_size = stats["statistics"].get("page_size", 0)
                    if isinstance(page_count, int) and isinstance(page_size, int):
                        stats["statistics"]["database_size_bytes"] = page_count * page_size
                        stats["statistics"]["database_size_mb"] = round((page_count * page_size) / (1024 * 1024), 2)
                    
                    stats["optimization_applied"] = True
            else:
                stats["message"] = "Detailed statistics only available for SQLite"

            return stats
            
        except Exception as e:
            logger.error(f"Failed to analyze database statistics: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        最適化の推奨事項を取得

        Returns:
            推奨事項の辞書
        """
        recommendations = {
            "database_type": self.connection_manager.config.get_database_type(),
            "recommendations": [],
            "warnings": []
        }

        try:
            if self.connection_manager.config.is_sqlite():
                # SQLiteの推奨事項
                settings = self.get_sqlite_pragma_settings()
                
                if settings.get("journal_mode") != "wal":
                    recommendations["recommendations"].append(
                        "Consider using WAL mode for better concurrency: PRAGMA journal_mode=WAL"
                    )
                
                if settings.get("synchronous") not in ["NORMAL", "normal", 1]:
                    recommendations["recommendations"].append(
                        "Consider NORMAL synchronous mode for balanced performance: PRAGMA synchronous=NORMAL"
                    )
                
                cache_size = settings.get("cache_size", 0)
                if isinstance(cache_size, int) and cache_size > 0 and cache_size < 10000:
                    recommendations["recommendations"].append(
                        "Consider increasing cache size for better performance: PRAGMA cache_size=10000"
                    )
                
                if settings.get("temp_store") != "memory":
                    recommendations["recommendations"].append(
                        "Consider storing temporary data in memory: PRAGMA temp_store=memory"
                    )
                
            else:
                recommendations["recommendations"].append(
                    "Optimization recommendations are primarily available for SQLite databases"
                )

        except Exception as e:
            recommendations["warnings"].append(f"Failed to analyze settings: {str(e)}")

        return recommendations