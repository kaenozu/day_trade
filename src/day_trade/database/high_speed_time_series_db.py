#!/usr/bin/env python3
"""
高速時系列データベースシステム
Issue #317: 高速データ管理システム - Phase 1

TimescaleDB最適化による時系列データ高速処理システム
- 時系列データ最適化
- クエリパフォーマンス改善
- インデックス戦略最適化
- データ圧縮・パーティション設計
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class TimeSeriesConfig:
    """時系列データベース設定"""

    # 接続設定
    host: str = "localhost"
    port: int = 5432
    database: str = "day_trade_timeseries"
    user: str = "daytrader"
    password: str = ""

    # TimescaleDB設定
    chunk_time_interval: str = "1 day"  # チャンクサイズ
    compression_interval: str = "7 days"  # 圧縮間隔
    retention_policy: str = "2 years"  # データ保持期間

    # パフォーマンス設定
    connection_pool_size: int = 20
    query_timeout: int = 30
    batch_insert_size: int = 1000
    enable_compression: bool = True
    enable_continuous_aggregates: bool = True


@dataclass
class QueryPerformanceMetrics:
    """クエリパフォーマンスメトリクス"""

    query_id: str
    execution_time_ms: float
    rows_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    index_usage: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataCompressionResult:
    """データ圧縮結果"""

    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    compression_algorithm: str
    compression_time_ms: float
    decompression_time_ms: float
    data_integrity_verified: bool
    timestamp: datetime = field(default_factory=datetime.now)


class HighSpeedTimeSeriesDB:
    """高速時系列データベースシステム"""

    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.performance_metrics: List[QueryPerformanceMetrics] = []
        self.compression_results: List[DataCompressionResult] = []
        self._initialized = False

    async def initialize(self) -> bool:
        """データベース初期化"""
        try:
            # 接続プール作成
            self.connection_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=5,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout,
            )

            # TimescaleDB拡張確認・有効化
            await self._enable_timescaledb_extension()

            # テーブル作成・設定
            await self._create_hypertables()
            await self._setup_compression_policies()
            await self._setup_retention_policies()
            await self._create_indexes()

            # 連続集計の設定
            if self.config.enable_continuous_aggregates:
                await self._setup_continuous_aggregates()

            self._initialized = True
            logger.info("高速時系列データベースシステム初期化完了")
            return True

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            return False

    async def _enable_timescaledb_extension(self) -> None:
        """TimescaleDB拡張の有効化"""
        async with self.connection_pool.acquire() as conn:
            try:
                # TimescaleDB拡張確認
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
                )

                if not result:
                    # TimescaleDB拡張作成
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                    logger.info("TimescaleDB拡張を有効化しました")

            except Exception as e:
                logger.warning(f"TimescaleDB拡張確認/作成エラー: {e}")
                # フォールバック: PostgreSQL標準機能で継続

    async def _create_hypertables(self) -> None:
        """ハイパーテーブル作成"""
        async with self.connection_pool.acquire() as conn:
            # 株価時系列テーブル
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stock_prices_ts (
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    open_price DECIMAL(12,3) NOT NULL,
                    high_price DECIMAL(12,3) NOT NULL,
                    low_price DECIMAL(12,3) NOT NULL,
                    close_price DECIMAL(12,3) NOT NULL,
                    adjusted_close DECIMAL(12,3) NOT NULL,
                    volume BIGINT NOT NULL,
                    market_cap DECIMAL(18,2),
                    data_source VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """
            )

            # ハイパーテーブル化
            try:
                await conn.execute(
                    """
                    SELECT create_hypertable(
                        'stock_prices_ts',
                        'timestamp',
                        chunk_time_interval => INTERVAL %s,
                        if_not_exists => TRUE
                    );
                """,
                    self.config.chunk_time_interval,
                )

            except Exception as e:
                logger.warning(f"ハイパーテーブル作成スキップ: {e}")

            # テクニカル指標時系列テーブル
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS technical_indicators_ts (
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    indicator_type VARCHAR(50) NOT NULL,
                    indicator_value DECIMAL(15,6),
                    indicator_signal VARCHAR(20),
                    confidence_score DECIMAL(5,4),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """
            )

            try:
                await conn.execute(
                    """
                    SELECT create_hypertable(
                        'technical_indicators_ts',
                        'timestamp',
                        chunk_time_interval => INTERVAL %s,
                        if_not_exists => TRUE
                    );
                """,
                    self.config.chunk_time_interval,
                )

            except Exception:
                pass

            # ML予測結果時系列テーブル
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_predictions_ts (
                    symbol VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    prediction_type VARCHAR(50) NOT NULL,
                    predicted_value DECIMAL(15,6),
                    confidence_interval_lower DECIMAL(15,6),
                    confidence_interval_upper DECIMAL(15,6),
                    model_name VARCHAR(100),
                    model_version VARCHAR(20),
                    prediction_horizon_days INTEGER,
                    feature_importance JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """
            )

            try:
                await conn.execute(
                    """
                    SELECT create_hypertable(
                        'ml_predictions_ts',
                        'timestamp',
                        chunk_time_interval => INTERVAL %s,
                        if_not_exists => TRUE
                    );
                """,
                    self.config.chunk_time_interval,
                )

            except Exception:
                pass

            logger.info("ハイパーテーブル作成完了")

    async def _setup_compression_policies(self) -> None:
        """データ圧縮ポリシー設定"""
        if not self.config.enable_compression:
            return

        async with self.connection_pool.acquire() as conn:
            try:
                # 株価データ圧縮ポリシー
                await conn.execute(
                    """
                    SELECT add_compression_policy(
                        'stock_prices_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.compression_interval,
                )

                # テクニカル指標データ圧縮ポリシー
                await conn.execute(
                    """
                    SELECT add_compression_policy(
                        'technical_indicators_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.compression_interval,
                )

                # ML予測データ圧縮ポリシー
                await conn.execute(
                    """
                    SELECT add_compression_policy(
                        'ml_predictions_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.compression_interval,
                )

                logger.info("データ圧縮ポリシー設定完了")

            except Exception as e:
                logger.warning(f"圧縮ポリシー設定エラー: {e}")

    async def _setup_retention_policies(self) -> None:
        """データ保持ポリシー設定"""
        async with self.connection_pool.acquire() as conn:
            try:
                # データ保持ポリシー設定
                await conn.execute(
                    """
                    SELECT add_retention_policy(
                        'stock_prices_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.retention_policy,
                )

                await conn.execute(
                    """
                    SELECT add_retention_policy(
                        'technical_indicators_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.retention_policy,
                )

                await conn.execute(
                    """
                    SELECT add_retention_policy(
                        'ml_predictions_ts',
                        INTERVAL %s
                    );
                """,
                    self.config.retention_policy,
                )

                logger.info("データ保持ポリシー設定完了")

            except Exception as e:
                logger.warning(f"保持ポリシー設定エラー: {e}")

    async def _create_indexes(self) -> None:
        """最適化インデックス作成"""
        async with self.connection_pool.acquire() as conn:
            # 株価テーブルインデックス
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time ON stock_prices_ts (symbol, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_volume ON stock_prices_ts (volume) WHERE volume > 1000000",
                "CREATE INDEX IF NOT EXISTS idx_stock_prices_price_change ON stock_prices_ts ((close_price - open_price)/open_price) WHERE ABS((close_price - open_price)/open_price) > 0.05",
                # テクニカル指標インデックス
                "CREATE INDEX IF NOT EXISTS idx_technical_symbol_type_time ON technical_indicators_ts (symbol, indicator_type, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_technical_signal ON technical_indicators_ts (indicator_signal) WHERE indicator_signal IN ('BUY', 'SELL')",
                "CREATE INDEX IF NOT EXISTS idx_technical_confidence ON technical_indicators_ts (confidence_score) WHERE confidence_score > 0.8",
                # ML予測インデックス
                "CREATE INDEX IF NOT EXISTS idx_ml_symbol_model_time ON ml_predictions_ts (symbol, model_name, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ml_prediction_type ON ml_predictions_ts (prediction_type, timestamp DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ml_confidence_interval ON ml_predictions_ts ((confidence_interval_upper - confidence_interval_lower)) WHERE (confidence_interval_upper - confidence_interval_lower) < 0.1",
            ]

            for index_sql in indexes:
                try:
                    await conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"インデックス作成エラー: {e}")

            logger.info("最適化インデックス作成完了")

    async def _setup_continuous_aggregates(self) -> None:
        """連続集計設定"""
        async with self.connection_pool.acquire() as conn:
            try:
                # 日次集計ビュー
                await conn.execute(
                    """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS stock_daily_aggregates
                    WITH (timescaledb.continuous) AS
                    SELECT
                        symbol,
                        time_bucket('1 day', timestamp) AS day,
                        first(open_price, timestamp) AS open_price,
                        max(high_price) AS high_price,
                        min(low_price) AS low_price,
                        last(close_price, timestamp) AS close_price,
                        sum(volume) AS total_volume,
                        avg(close_price) AS avg_price,
                        stddev(close_price) AS price_volatility,
                        count(*) AS data_points
                    FROM stock_prices_ts
                    GROUP BY symbol, day;
                """
                )

                # 週次集計ビュー
                await conn.execute(
                    """
                    CREATE MATERIALIZED VIEW IF NOT EXISTS stock_weekly_aggregates
                    WITH (timescaledb.continuous) AS
                    SELECT
                        symbol,
                        time_bucket('1 week', timestamp) AS week,
                        first(open_price, timestamp) AS open_price,
                        max(high_price) AS high_price,
                        min(low_price) AS low_price,
                        last(close_price, timestamp) AS close_price,
                        sum(volume) AS total_volume,
                        avg(close_price) AS avg_price,
                        stddev(close_price) AS price_volatility
                    FROM stock_prices_ts
                    GROUP BY symbol, week;
                """
                )

                # 自動リフレッシュポリシー設定
                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy(
                        'stock_daily_aggregates',
                        start_offset => INTERVAL '1 month',
                        end_offset => INTERVAL '1 hour',
                        schedule_interval => INTERVAL '1 hour'
                    );
                """
                )

                await conn.execute(
                    """
                    SELECT add_continuous_aggregate_policy(
                        'stock_weekly_aggregates',
                        start_offset => INTERVAL '3 months',
                        end_offset => INTERVAL '1 day',
                        schedule_interval => INTERVAL '1 day'
                    );
                """
                )

                logger.info("連続集計設定完了")

            except Exception as e:
                logger.warning(f"連続集計設定エラー: {e}")

    async def insert_stock_data_batch(self, stock_data: List[Dict[str, Any]]) -> bool:
        """株価データバッチ挿入"""
        if not self._initialized:
            raise RuntimeError("データベースが初期化されていません")

        start_time = time.time()

        try:
            async with self.connection_pool.acquire() as conn:
                # バッチ挿入準備
                insert_sql = """
                    INSERT INTO stock_prices_ts (
                        symbol, timestamp, open_price, high_price, low_price,
                        close_price, adjusted_close, volume, market_cap, data_source
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (symbol, timestamp)
                    DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        adjusted_close = EXCLUDED.adjusted_close,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap,
                        data_source = EXCLUDED.data_source;
                """

                # データ準備
                batch_data = []
                for data in stock_data:
                    batch_data.append(
                        (
                            data.get("symbol"),
                            data.get("timestamp"),
                            data.get("open_price"),
                            data.get("high_price"),
                            data.get("low_price"),
                            data.get("close_price"),
                            data.get("adjusted_close"),
                            data.get("volume"),
                            data.get("market_cap"),
                            data.get("data_source", "system"),
                        )
                    )

                # バッチ実行
                await conn.executemany(insert_sql, batch_data)

                execution_time = (time.time() - start_time) * 1000

                # パフォーマンス記録
                metrics = QueryPerformanceMetrics(
                    query_id=f"batch_insert_{int(time.time())}",
                    execution_time_ms=execution_time,
                    rows_processed=len(stock_data),
                    memory_usage_mb=0.0,  # 実装で取得可能
                    cpu_usage_percent=0.0,  # 実装で取得可能
                    cache_hit_rate=0.0,
                )
                self.performance_metrics.append(metrics)

                logger.info(
                    f"株価データバッチ挿入完了: {len(stock_data)}件, {execution_time:.2f}ms"
                )
                return True

        except Exception as e:
            logger.error(f"バッチ挿入エラー: {e}")
            return False

    async def query_stock_data_optimized(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        aggregation_level: str = "raw",  # raw, daily, weekly, monthly
    ) -> Optional[pd.DataFrame]:
        """最適化された株価データクエリ"""
        if not self._initialized:
            raise RuntimeError("データベースが初期化されていません")

        start_time = time.time()

        try:
            async with self.connection_pool.acquire() as conn:
                if aggregation_level == "daily":
                    # 日次集計データ使用
                    query = """
                        SELECT
                            symbol,
                            day as timestamp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            total_volume as volume,
                            avg_price,
                            price_volatility
                        FROM stock_daily_aggregates
                        WHERE symbol = $1
                        AND day >= $2
                        AND day <= $3
                        ORDER BY day;
                    """

                elif aggregation_level == "weekly":
                    # 週次集計データ使用
                    query = """
                        SELECT
                            symbol,
                            week as timestamp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            total_volume as volume,
                            avg_price,
                            price_volatility
                        FROM stock_weekly_aggregates
                        WHERE symbol = $1
                        AND week >= $2
                        AND week <= $3
                        ORDER BY week;
                    """

                else:
                    # 生データ使用
                    query = """
                        SELECT
                            symbol,
                            timestamp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            adjusted_close,
                            volume,
                            market_cap
                        FROM stock_prices_ts
                        WHERE symbol = $1
                        AND timestamp >= $2
                        AND timestamp <= $3
                        ORDER BY timestamp;
                    """

                rows = await conn.fetch(query, symbol, start_date, end_date)

                execution_time = (time.time() - start_time) * 1000

                # パフォーマンス記録
                metrics = QueryPerformanceMetrics(
                    query_id=f"query_{aggregation_level}_{int(time.time())}",
                    execution_time_ms=execution_time,
                    rows_processed=len(rows),
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    cache_hit_rate=0.0,
                )
                self.performance_metrics.append(metrics)

                if rows:
                    # pandasデータフレーム変換
                    df = pd.DataFrame([dict(row) for row in rows])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)

                    logger.info(f"データクエリ完了: {len(rows)}件, {execution_time:.2f}ms")
                    return df
                else:
                    return pd.DataFrame()

        except Exception as e:
            logger.error(f"データクエリエラー: {e}")
            return None

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        if not self.performance_metrics:
            return {}

        # 統計計算
        execution_times = [m.execution_time_ms for m in self.performance_metrics]
        rows_processed = [m.rows_processed for m in self.performance_metrics]

        return {
            "total_queries": len(self.performance_metrics),
            "avg_execution_time_ms": np.mean(execution_times),
            "max_execution_time_ms": np.max(execution_times),
            "min_execution_time_ms": np.min(execution_times),
            "p95_execution_time_ms": np.percentile(execution_times, 95),
            "total_rows_processed": sum(rows_processed),
            "avg_rows_per_query": np.mean(rows_processed),
            "queries_per_second": len(self.performance_metrics)
            / (
                (datetime.now() - self.performance_metrics[0].timestamp).total_seconds()
                if self.performance_metrics
                else 1
            ),
        }

    async def optimize_database(self) -> Dict[str, Any]:
        """データベース最適化実行"""
        optimization_results = {}

        try:
            async with self.connection_pool.acquire() as conn:
                start_time = time.time()

                # VACUUM ANALYZE実行
                await conn.execute("VACUUM ANALYZE stock_prices_ts;")
                await conn.execute("VACUUM ANALYZE technical_indicators_ts;")
                await conn.execute("VACUUM ANALYZE ml_predictions_ts;")

                # 統計情報更新
                await conn.execute("ANALYZE;")

                # チャンクサイズ最適化確認
                chunk_info = await conn.fetch(
                    """
                    SELECT
                        hypertable_name,
                        chunk_name,
                        chunk_size
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name IN ('stock_prices_ts', 'technical_indicators_ts', 'ml_predictions_ts')
                    ORDER BY chunk_size DESC
                    LIMIT 10;
                """
                )

                optimization_time = (time.time() - start_time) * 1000

                optimization_results = {
                    "optimization_time_ms": optimization_time,
                    "tables_optimized": 3,
                    "top_chunks": [dict(row) for row in chunk_info],
                    "status": "completed",
                }

                logger.info(f"データベース最適化完了: {optimization_time:.2f}ms")

        except Exception as e:
            logger.error(f"データベース最適化エラー: {e}")
            optimization_results = {"status": "failed", "error": str(e)}

        return optimization_results

    async def cleanup(self) -> None:
        """リソースクリーンアップ"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("データベース接続プールを閉じました")


# 使用例・テスト用ファクトリー関数
async def create_high_speed_timeseries_db(
    config: Optional[TimeSeriesConfig] = None,
) -> HighSpeedTimeSeriesDB:
    """高速時系列DB作成・初期化"""
    db = HighSpeedTimeSeriesDB(config)

    if await db.initialize():
        return db
    else:
        raise RuntimeError("高速時系列データベースの初期化に失敗しました")


# パフォーマンステスト用関数
async def benchmark_database_performance(
    db: HighSpeedTimeSeriesDB, test_data_size: int = 10000
) -> Dict[str, Any]:
    """データベースパフォーマンステスト"""
    # テストデータ生成
    test_data = []
    base_time = datetime.now()

    for i in range(test_data_size):
        test_data.append(
            {
                "symbol": f"TEST{i % 100:04d}",
                "timestamp": base_time + timedelta(minutes=i),
                "open_price": 1000 + np.random.normal(0, 50),
                "high_price": 1050 + np.random.normal(0, 50),
                "low_price": 950 + np.random.normal(0, 50),
                "close_price": 1000 + np.random.normal(0, 50),
                "adjusted_close": 1000 + np.random.normal(0, 50),
                "volume": int(1000000 + np.random.normal(0, 500000)),
                "market_cap": 1000000000,
                "data_source": "benchmark",
            }
        )

    # 挿入性能テスト
    insert_start = time.time()
    success = await db.insert_stock_data_batch(test_data)
    insert_time = (time.time() - insert_start) * 1000

    # クエリ性能テスト
    query_start = time.time()
    df = await db.query_stock_data_optimized("TEST0001", base_time, base_time + timedelta(hours=1))
    query_time = (time.time() - query_start) * 1000

    return {
        "insert_success": success,
        "insert_time_ms": insert_time,
        "insert_throughput_records_per_second": test_data_size / (insert_time / 1000),
        "query_time_ms": query_time,
        "query_result_rows": len(df) if df is not None else 0,
        "performance_metrics": await db.get_performance_metrics(),
    }


if __name__ == "__main__":

    async def main():
        # 設定
        config = TimeSeriesConfig(
            database="test_timeseries", user="test_user", password="test_pass"
        )

        # データベース作成・初期化
        try:
            db = await create_high_speed_timeseries_db(config)

            # パフォーマンステスト
            results = await benchmark_database_performance(db, 1000)
            print("パフォーマンステスト結果:")
            for key, value in results.items():
                print(f"  {key}: {value}")

            # クリーンアップ
            await db.cleanup()

        except Exception as e:
            print(f"エラー: {e}")

    asyncio.run(main())
