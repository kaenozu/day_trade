"""
データベース最適化モジュール

データベース操作のパフォーマンス最適化機能を提供。
インデックス最適化、クエリ最適化、バッチ処理、コネクションプール管理。
"""

import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..utils.logging_config import get_context_logger, log_performance_metric
from ..utils.performance_analyzer import profile_performance
from .database import DatabaseConfig, DatabaseManager

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


@dataclass
class QueryPerformanceMetrics:
    """クエリパフォーマンス指標"""

    query_hash: str
    execution_time: float
    rows_affected: int
    cache_hit: bool = False
    optimization_level: str = "none"
    index_usage: List[str] = field(default_factory=list)
    execution_plan: Optional[str] = None


@dataclass
class OptimizationConfig:
    """最適化設定"""

    enable_query_cache: bool = True
    enable_batch_processing: bool = True
    enable_connection_pooling: bool = True
    enable_index_optimization: bool = True

    # クエリキャッシュ設定
    query_cache_size: int = 1000
    query_cache_ttl_minutes: int = 30

    # バッチ処理設定
    batch_size: int = 1000
    max_batch_memory_mb: int = 100

    # コネクションプール設定
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 60
    pool_recycle: int = 1800

    # インデックス設定
    auto_create_indexes: bool = True
    analyze_query_patterns: bool = True


class QueryCache:
    """クエリ結果キャッシュシステム"""

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)

    def _generate_cache_key(self, query: str, params: Dict = None) -> str:
        """キャッシュキー生成"""
        params_str = str(sorted(params.items())) if params else ""
        return f"{hash(query + params_str)}"

    def get(self, query: str, params: Dict = None) -> Optional[Any]:
        """キャッシュから結果取得"""
        key = self._generate_cache_key(query, params)

        if key in self.cache:
            cached_item = self.cache[key]

            # TTLチェック
            if datetime.now() - cached_item["timestamp"] < self.ttl:
                self.access_times[key] = datetime.now()
                return cached_item["result"]
            else:
                # 期限切れ削除
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        return None

    def put(self, query: str, result: Any, params: Dict = None):
        """キャッシュに結果保存"""
        key = self._generate_cache_key(query, params)

        # キャッシュサイズ制限
        if len(self.cache) >= self.max_size:
            oldest_key = min(
                self.access_times.keys(), key=lambda k: self.access_times[k]
            )
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = {"result": result, "timestamp": datetime.now()}
        self.access_times[key] = datetime.now()

    def clear(self):
        """キャッシュクリア"""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl.total_seconds() / 60,
        }


class BatchProcessor:
    """バッチ処理システム"""

    def __init__(self, batch_size: int = 1000, max_memory_mb: int = 100):
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.pending_operations = []

    def add_operation(self, operation_type: str, table: str, data: Dict[str, Any]):
        """操作をバッチに追加"""
        self.pending_operations.append(
            {
                "type": operation_type,
                "table": table,
                "data": data,
                "timestamp": datetime.now(),
            }
        )

        # バッチサイズに達した場合は自動実行
        return len(self.pending_operations) >= self.batch_size

    def get_batch_operations(self, operation_type: str = None) -> List[Dict]:
        """バッチ操作取得"""
        if operation_type:
            return [
                op for op in self.pending_operations if op["type"] == operation_type
            ]
        return self.pending_operations.copy()

    def clear_operations(self, operation_type: str = None):
        """操作クリア"""
        if operation_type:
            self.pending_operations = [
                op for op in self.pending_operations if op["type"] != operation_type
            ]
        else:
            self.pending_operations.clear()

    def estimate_memory_usage(self) -> float:
        """メモリ使用量推定（MB）"""
        # 簡易推定
        return len(self.pending_operations) * 0.01  # 1操作あたり0.01MB


class OptimizedDatabaseManager(DatabaseManager):
    """最適化済みデータベース管理クラス"""

    def __init__(
        self,
        config: Optional[DatabaseConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """
        Args:
            config: データベース設定
            optimization_config: 最適化設定
        """
        self.optimization_config = optimization_config or OptimizationConfig()

        # 最適化コンポーネント
        self.query_cache = (
            QueryCache(
                max_size=self.optimization_config.query_cache_size,
                ttl_minutes=self.optimization_config.query_cache_ttl_minutes,
            )
            if self.optimization_config.enable_query_cache
            else None
        )

        self.batch_processor = (
            BatchProcessor(
                batch_size=self.optimization_config.batch_size,
                max_memory_mb=self.optimization_config.max_batch_memory_mb,
            )
            if self.optimization_config.enable_batch_processing
            else None
        )

        # パフォーマンス統計
        self.query_metrics = []
        self.index_usage_stats = {}
        self.connection_pool_stats = {"hits": 0, "misses": 0}

        # 最適化済み設定でスーパークラス初期化
        optimized_config = self._create_optimized_config(config)
        super().__init__(optimized_config)

        # インデックス最適化
        if self.optimization_config.enable_index_optimization:
            self._initialize_index_optimization()

        logger.info(
            "最適化済みデータベース管理システム初期化完了",
            section="optimized_database_init",
            query_cache_enabled=self.optimization_config.enable_query_cache,
            batch_processing_enabled=self.optimization_config.enable_batch_processing,
            index_optimization_enabled=self.optimization_config.enable_index_optimization,
        )

    def _create_optimized_config(
        self, base_config: Optional[DatabaseConfig]
    ) -> DatabaseConfig:
        """最適化済みデータベース設定作成"""

        if base_config is None:
            base_config = DatabaseConfig()

        # 最適化設定を適用
        if self.optimization_config.enable_connection_pooling:
            optimized_config = DatabaseConfig(
                database_url=base_config.database_url,
                echo=base_config.echo,
                pool_size=self.optimization_config.pool_size,
                max_overflow=self.optimization_config.max_overflow,
                pool_timeout=self.optimization_config.pool_timeout,
                pool_recycle=self.optimization_config.pool_recycle,
                connect_args=base_config.connect_args,
            )
        else:
            optimized_config = base_config

        return optimized_config

    def _initialize_index_optimization(self):
        """インデックス最適化初期化"""
        try:
            if self.optimization_config.auto_create_indexes:
                self._create_performance_indexes()

            logger.info(
                "インデックス最適化初期化完了", extra={"section": "index_optimization"}
            )

        except Exception as e:
            logger.warning(
                f"インデックス最適化初期化エラー: {e}",
                extra={"section": "index_optimization"},
            )

    def _create_performance_indexes(self):
        """パフォーマンス向上インデックス作成"""

        # よく使われるクエリパターンに基づくインデックス
        performance_indexes = [
            # 取引履歴テーブル用インデックス
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_price ON trades(symbol, price)",
            # ポートフォリオテーブル用インデックス
            "CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_updated_at ON portfolio(updated_at)",
            # ウォッチリストテーブル用インデックス
            "CREATE INDEX IF NOT EXISTS idx_watchlist_symbol ON watchlist(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_watchlist_created_at ON watchlist(created_at)",
            # アラートテーブル用インデックス
            "CREATE INDEX IF NOT EXISTS idx_alerts_symbol_active ON alerts(symbol, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_trigger_price ON alerts(trigger_price)",
        ]

        try:
            with self.get_session() as session:
                for index_sql in performance_indexes:
                    try:
                        session.execute(text(index_sql))
                        session.commit()

                        logger.debug(
                            f"インデックス作成: {index_sql}",
                            extra={"section": "index_creation"},
                        )

                    except Exception as e:
                        logger.debug(
                            f"インデックス作成スキップ: {e}",
                            extra={"section": "index_creation"},
                        )
                        session.rollback()

        except Exception as e:
            logger.warning(
                f"パフォーマンスインデックス作成エラー: {e}",
                section="index_optimization",
            )

    @profile_performance
    def execute_optimized_query(
        self,
        query: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        fetch_all: bool = True,
    ) -> Any:
        """最適化済みクエリ実行"""

        start_time = time.time()
        cache_hit = False
        result = None

        # キャッシュチェック
        if use_cache and self.query_cache:
            cached_result = self.query_cache.get(query, params)
            if cached_result is not None:
                cache_hit = True
                result = cached_result

                logger.debug(
                    "クエリキャッシュヒット",
                    section="query_optimization",
                    query_hash=hash(query),
                )

        # キャッシュミスの場合は実行
        if result is None:
            try:
                with self.get_session() as session:
                    executed_query = session.execute(text(query), params or {})

                    if fetch_all:
                        result = executed_query.fetchall()
                    else:
                        result = executed_query.fetchone()

                    session.commit()

                    # キャッシュに保存
                    if use_cache and self.query_cache:
                        self.query_cache.put(query, result, params)

            except Exception as e:
                logger.error(
                    "最適化クエリ実行エラー",
                    section="query_optimization",
                    query=query[:100],
                    error=str(e),
                )
                raise

        # パフォーマンス統計記録
        execution_time = time.time() - start_time
        rows_affected = len(result) if isinstance(result, list) else 1

        metrics = QueryPerformanceMetrics(
            query_hash=str(hash(query)),
            execution_time=execution_time,
            rows_affected=rows_affected,
            cache_hit=cache_hit,
        )

        self.query_metrics.append(metrics)

        # ログ記録
        log_performance_metric(
            metric_name="optimized_query_execution",
            value=execution_time,
            unit="seconds",
            additional_data={
                "rows_affected": rows_affected,
                "cache_hit": cache_hit,
                "query_hash": metrics.query_hash,
            },
        )

        return result

    @profile_performance
    def bulk_insert_optimized(
        self, table_name: str, data: List[Dict[str, Any]], use_batch: bool = True
    ) -> int:
        """最適化済み一括挿入"""

        if not data:
            return 0

        if use_batch and self.batch_processor:
            return self._bulk_insert_batch(table_name, data)
        else:
            return self._bulk_insert_direct(table_name, data)

    def _bulk_insert_batch(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """バッチ処理による一括挿入"""

        inserted_count = 0
        batch_size = self.optimization_config.batch_size

        for i in range(0, len(data), batch_size):
            batch_data = data[i : i + batch_size]

            try:
                with self.get_session() as session:
                    # バッチ挿入実行
                    for item in batch_data:
                        # テーブルに応じた挿入処理
                        insert_sql = self._generate_insert_sql(table_name, item)
                        session.execute(text(insert_sql), item)

                    session.commit()
                    inserted_count += len(batch_data)

                    logger.debug(
                        f"バッチ挿入完了: {table_name}",
                        section="batch_processing",
                        batch_size=len(batch_data),
                        total_inserted=inserted_count,
                    )

            except Exception as e:
                logger.error(
                    f"バッチ挿入エラー: {table_name}",
                    section="batch_processing",
                    error=str(e),
                )
                raise

        return inserted_count

    def _bulk_insert_direct(self, table_name: str, data: List[Dict[str, Any]]) -> int:
        """直接一括挿入"""

        try:
            with self.get_session() as session:
                for item in data:
                    insert_sql = self._generate_insert_sql(table_name, item)
                    session.execute(text(insert_sql), item)

                session.commit()

                logger.debug(
                    f"直接一括挿入完了: {table_name}",
                    section="direct_insert",
                    records_count=len(data),
                )

                return len(data)

        except Exception as e:
            logger.error(
                f"直接一括挿入エラー: {table_name}",
                section="direct_insert",
                error=str(e),
            )
            raise

    def _generate_insert_sql(self, table_name: str, data: Dict[str, Any]) -> str:
        """挿入SQL生成"""

        columns = list(data.keys())
        placeholders = [f":{col}" for col in columns]

        sql = f"""
        INSERT INTO {table_name} ({", ".join(columns)})
        VALUES ({", ".join(placeholders)})
        """

        return sql

    @profile_performance
    def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """テーブル最適化実行"""

        optimization_results = {
            "table_name": table_name,
            "optimizations_applied": [],
            "performance_improvement": 0.0,
            "errors": [],
        }

        try:
            with self.get_session() as session:
                # 1. 統計情報更新（SQLiteの場合）
                if self.config.is_sqlite():
                    session.execute(text(f"ANALYZE {table_name}"))
                    optimization_results["optimizations_applied"].append(
                        "analyze_table"
                    )

                # 2. インデックス使用状況分析
                index_analysis = self._analyze_table_indexes(session, table_name)
                optimization_results["index_analysis"] = index_analysis

                # 3. クエリプラン分析
                query_plan_analysis = self._analyze_query_plans(session, table_name)
                optimization_results["query_plan_analysis"] = query_plan_analysis

                session.commit()

                logger.info(
                    f"テーブル最適化完了: {table_name}",
                    section="table_optimization",
                    optimizations=optimization_results["optimizations_applied"],
                )

        except Exception as e:
            error_msg = f"テーブル最適化エラー: {e}"
            logger.error(error_msg, extra={"section": "table_optimization"})
            optimization_results["errors"].append(error_msg)

        return optimization_results

    def _analyze_table_indexes(
        self, session: Session, table_name: str
    ) -> Dict[str, Any]:
        """テーブルインデックス分析"""

        index_analysis = {
            "existing_indexes": [],
            "recommended_indexes": [],
            "unused_indexes": [],
        }

        try:
            # 既存インデックス取得
            if self.config.is_sqlite():
                result = session.execute(text(f"PRAGMA index_list({table_name})"))
                existing_indexes = [row[1] for row in result.fetchall()]
                index_analysis["existing_indexes"] = existing_indexes

            # インデックス推奨事項（簡易版）
            # 実際の実装では、クエリログ分析に基づく推奨が望ましい
            index_analysis["recommended_indexes"] = [
                f"idx_{table_name}_created_at",
                f"idx_{table_name}_updated_at",
            ]

        except Exception as e:
            logger.warning(
                f"インデックス分析エラー: {e}", extra={"section": "index_analysis"}
            )

        return index_analysis

    def _analyze_query_plans(self, session: Session, table_name: str) -> Dict[str, Any]:
        """クエリプラン分析"""

        plan_analysis = {"sample_queries": [], "optimization_opportunities": []}

        try:
            # サンプルクエリのプラン分析
            sample_queries = [
                f"SELECT * FROM {table_name} LIMIT 10",
                f"SELECT COUNT(*) FROM {table_name}",
            ]

            for query in sample_queries:
                if self.config.is_sqlite():
                    plan_result = session.execute(text(f"EXPLAIN QUERY PLAN {query}"))
                    plan = [row for row in plan_result.fetchall()]

                    plan_analysis["sample_queries"].append(
                        {"query": query, "plan": str(plan)}
                    )

        except Exception as e:
            logger.warning(
                f"クエリプラン分析エラー: {e}", extra={"section": "query_plan_analysis"}
            )

        return plan_analysis

    def get_performance_statistics(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""

        stats = {
            "query_metrics": {
                "total_queries": len(self.query_metrics),
                "avg_execution_time": 0.0,
                "cache_hit_rate": 0.0,
                "slowest_queries": [],
            },
            "cache_stats": {},
            "batch_stats": {},
            "connection_pool_stats": self.connection_pool_stats,
        }

        # クエリ統計
        if self.query_metrics:
            total_time = sum(m.execution_time for m in self.query_metrics)
            stats["query_metrics"]["avg_execution_time"] = total_time / len(
                self.query_metrics
            )

            cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
            stats["query_metrics"]["cache_hit_rate"] = (
                cache_hits / len(self.query_metrics) * 100
            )

            # 最も遅いクエリ
            sorted_metrics = sorted(
                self.query_metrics, key=lambda x: x.execution_time, reverse=True
            )
            stats["query_metrics"]["slowest_queries"] = [
                {
                    "query_hash": m.query_hash,
                    "execution_time": m.execution_time,
                    "rows_affected": m.rows_affected,
                }
                for m in sorted_metrics[:5]
            ]

        # キャッシュ統計
        if self.query_cache:
            stats["cache_stats"] = self.query_cache.get_stats()

        # バッチ統計
        if self.batch_processor:
            stats["batch_stats"] = {
                "pending_operations": len(self.batch_processor.pending_operations),
                "estimated_memory_mb": self.batch_processor.estimate_memory_usage(),
            }

        return stats

    def vacuum_optimize(self) -> Dict[str, Any]:
        """データベース真空・最適化"""

        optimization_results = {
            "vacuum_executed": False,
            "space_reclaimed_mb": 0.0,
            "errors": [],
        }

        try:
            if self.config.is_sqlite():
                with self.get_session() as session:
                    # ファイルサイズ取得（最適化前）
                    size_before = self._get_database_size()

                    # VACUUM実行
                    session.execute(text("VACUUM"))
                    session.commit()

                    # ファイルサイズ取得（最適化後）
                    size_after = self._get_database_size()

                    optimization_results["vacuum_executed"] = True
                    optimization_results["space_reclaimed_mb"] = (
                        (size_before - size_after) / 1024 / 1024
                    )

                    logger.info(
                        "データベース真空最適化完了",
                        section="vacuum_optimization",
                        space_reclaimed_mb=optimization_results["space_reclaimed_mb"],
                    )
            else:
                logger.info(
                    "真空最適化はSQLiteでのみサポートされています",
                    section="vacuum_optimization",
                )

        except Exception as e:
            error_msg = f"真空最適化エラー: {e}"
            logger.error(error_msg, extra={"section": "vacuum_optimization"})
            optimization_results["errors"].append(error_msg)

        return optimization_results

    def _get_database_size(self) -> int:
        """データベースファイルサイズ取得（バイト）"""

        if self.config.is_sqlite() and not self.config.is_in_memory():
            import os

            db_path = self.config.database_url.replace("sqlite:///", "")
            if os.path.exists(db_path):
                return os.path.getsize(db_path)

        return 0

    @contextmanager
    def optimized_transaction(self):
        """最適化済みトランザクション"""

        session = self.get_session()
        try:
            yield session
            session.commit()

            # 統計更新
            self.connection_pool_stats["hits"] += 1

        except Exception as e:
            session.rollback()
            self.connection_pool_stats["misses"] += 1

            logger.error(
                "最適化トランザクションエラー",
                section="optimized_transaction",
                error=str(e),
            )
            raise
        finally:
            session.close()


# 使用例とデモ
if __name__ == "__main__":
    logger.info("最適化データベース管理システムデモ開始", extra={"section": "demo"})

    try:
        # 最適化設定
        optimization_config = OptimizationConfig(
            enable_query_cache=True,
            enable_batch_processing=True,
            enable_index_optimization=True,
            batch_size=100,
            query_cache_size=500,
        )

        # 最適化済みデータベースマネージャー
        db_manager = OptimizedDatabaseManager(
            config=DatabaseConfig.for_testing(), optimization_config=optimization_config
        )

        # テーブル作成
        db_manager.create_tables()

        # クエリ最適化テスト
        result = db_manager.execute_optimized_query(
            "SELECT 1 as test_column", use_cache=True
        )

        # パフォーマンス統計
        perf_stats = db_manager.get_performance_statistics()

        # 真空最適化
        vacuum_results = db_manager.vacuum_optimize()

        logger.info(
            "最適化データベース管理システムデモ完了",
            section="demo",
            performance_stats=perf_stats,
            vacuum_results=vacuum_results,
        )

    except Exception as e:
        logger.error(f"デモ実行エラー: {e}", extra={"section": "demo"})

    finally:
        # リソースクリーンアップ
        if "db_manager" in locals() and db_manager.query_cache:
            db_manager.query_cache.clear()
