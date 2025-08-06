"""
パフォーマンス最適化設定モジュール

データベース、キャッシュ、計算処理の最適化設定を管理する。
Phase 2: パフォーマンス最適化プロジェクト対応
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DatabasePerformanceConfig:
    """データベースパフォーマンス設定"""

    # コネクションプール設定（高性能化）
    pool_size: int = 20  # デフォルト5から増量
    max_overflow: int = 30  # デフォルト10から増量
    pool_timeout: int = 60  # デフォルト30から延長
    pool_recycle: int = 1800  # デフォルト3600から短縮（30分）
    pool_pre_ping: bool = True  # コネクション事前検証

    # SQLAlchemy 2.0 最適化設定
    echo_pool: bool = False  # プール状態ログ
    query_cache_size: int = 500  # クエリキャッシュサイズ
    compiled_cache_size: int = 1000  # コンパイル済みクエリキャッシュ

    # バルク操作設定
    bulk_batch_size: int = 1000  # バルク処理のバッチサイズ
    bulk_synchronize_session: str = "evaluate"  # バルク操作の同期方法

    # 接続設定（SQLite固有）
    sqlite_optimize: bool = True  # SQLite最適化の有効化
    sqlite_wal_mode: bool = True  # WALモード（Write-Ahead Logging）
    sqlite_cache_size: int = -32000  # キャッシュサイズ（32MB）
    sqlite_temp_store: str = "memory"  # 一時データをメモリに保存
    sqlite_synchronous: str = "NORMAL"  # 同期モード（FULL < NORMAL < OFF）
    sqlite_journal_mode: str = "WAL"  # ジャーナルモード


@dataclass
class ComputePerformanceConfig:
    """計算処理パフォーマンス設定"""

    # 並列処理設定
    max_workers: int = min(32, os.cpu_count() + 4)  # 最大ワーカー数
    use_multiprocessing: bool = True  # マルチプロセシング有効化
    chunk_size: int = 10000  # 処理チャンクサイズ

    # NumPy/pandas最適化
    numpy_threads: int = min(8, os.cpu_count())  # NumPyスレッド数
    pandas_threads: int = min(8, os.cpu_count())  # pandasスレッド数
    use_numba: bool = True  # Numba JIT最適化

    # メモリ最適化
    memory_limit_mb: int = 2048  # メモリ制限（2GB）
    gc_threshold: tuple = (700, 10, 10)  # ガベージコレクション閾値
    use_memory_mapping: bool = True  # メモリマッピング使用


@dataclass
class CachePerformanceConfig:
    """キャッシュパフォーマンス設定"""

    # L1キャッシュ（インメモリ）
    l1_cache_size: int = 1000  # L1キャッシュアイテム数
    l1_ttl_seconds: int = 300  # L1キャッシュTTL（5分）

    # L2キャッシュ（永続化）
    l2_cache_enabled: bool = True  # L2キャッシュ有効化
    l2_cache_size: int = 10000  # L2キャッシュアイテム数
    l2_ttl_seconds: int = 3600  # L2キャッシュTTL（1時間）

    # キャッシュウォーミング
    enable_cache_warming: bool = True  # キャッシュ事前ロード
    warm_up_batch_size: int = 100  # ウォームアップバッチサイズ

    # Redis設定（将来の拡張用）
    redis_enabled: bool = False  # Redis使用有効化
    redis_url: Optional[str] = None  # Redis接続URL
    redis_max_connections: int = 10  # Redis最大コネクション数


@dataclass
class APIPerformanceConfig:
    """API呼び出しパフォーマンス設定"""

    # 非同期処理設定
    async_enabled: bool = True  # 非同期処理有効化
    concurrent_requests: int = 10  # 同時リクエスト数
    request_timeout: float = 30.0  # リクエストタイムアウト（秒）

    # レート制限設定
    rate_limit_per_second: int = 10  # 秒間リクエスト制限
    rate_limit_burst: int = 20  # バーストリクエスト許可数

    # リトライ設定
    max_retries: int = 3  # 最大リトライ回数
    retry_backoff: float = 1.0  # リトライ間隔（秒）
    retry_exponential: bool = True  # 指数バックオフ


@dataclass
class PerformanceOptimizationConfig:
    """包括的なパフォーマンス最適化設定"""

    database: DatabasePerformanceConfig = None
    compute: ComputePerformanceConfig = None
    cache: CachePerformanceConfig = None
    api: APIPerformanceConfig = None

    def __post_init__(self):
        """初期化後処理"""
        if self.database is None:
            self.database = DatabasePerformanceConfig()
        if self.compute is None:
            self.compute = ComputePerformanceConfig()
        if self.cache is None:
            self.cache = CachePerformanceConfig()
        if self.api is None:
            self.api = APIPerformanceConfig()

    # 全体設定
    profiling_enabled: bool = True  # プロファイリング有効化
    monitoring_enabled: bool = True  # パフォーマンス監視有効化
    optimization_level: str = "high"  # 最適化レベル（low, medium, high, extreme）

    @classmethod
    def for_development(cls) -> "PerformanceOptimizationConfig":
        """開発環境用設定"""
        config = cls()
        # 開発環境では安全性を重視
        config.database.pool_size = 5
        config.database.max_overflow = 10
        config.compute.max_workers = 2
        config.optimization_level = "medium"
        return config

    @classmethod
    def for_production(cls) -> "PerformanceOptimizationConfig":
        """本番環境用設定"""
        config = cls()
        # 本番環境では最大パフォーマンスを追求
        config.database.pool_size = 30
        config.database.max_overflow = 50
        config.compute.max_workers = min(32, os.cpu_count() + 4)
        config.optimization_level = "high"
        return config

    @classmethod
    def for_testing(cls) -> "PerformanceOptimizationConfig":
        """テスト環境用設定"""
        config = cls()
        # テスト環境では軽量設定
        config.database.pool_size = 2
        config.database.max_overflow = 5
        config.compute.max_workers = 1
        config.cache.l1_cache_size = 100
        config.optimization_level = "low"
        config.profiling_enabled = False
        config.monitoring_enabled = False
        return config

    def get_sqlite_pragma_settings(self) -> Dict[str, Any]:
        """SQLite PRAGMA設定を取得"""
        if not self.database.sqlite_optimize:
            return {}

        settings = {
            "cache_size": self.database.sqlite_cache_size,
            "temp_store": self.database.sqlite_temp_store,
            "synchronous": self.database.sqlite_synchronous,
            "journal_mode": self.database.sqlite_journal_mode,
        }

        return settings

    def get_engine_settings(self) -> Dict[str, Any]:
        """SQLAlchemyエンジン設定を取得"""
        settings = {
            "pool_size": self.database.pool_size,
            "max_overflow": self.database.max_overflow,
            "pool_timeout": self.database.pool_timeout,
            "pool_recycle": self.database.pool_recycle,
            "pool_pre_ping": self.database.pool_pre_ping,
            "echo_pool": self.database.echo_pool,
        }

        # 実行環境に応じた最適化
        if self.optimization_level == "extreme":
            settings["pool_size"] *= 2
            settings["max_overflow"] *= 2

        return settings


# グローバル設定インスタンス
_performance_config: Optional[PerformanceOptimizationConfig] = None


def get_performance_config() -> PerformanceOptimizationConfig:
    """グローバルパフォーマンス設定を取得"""
    global _performance_config

    if _performance_config is None:
        # 環境変数に基づいて設定を決定
        env = os.getenv("ENVIRONMENT", "development").lower()

        if env == "production":
            _performance_config = PerformanceOptimizationConfig.for_production()
        elif env == "testing":
            _performance_config = PerformanceOptimizationConfig.for_testing()
        else:
            _performance_config = PerformanceOptimizationConfig.for_development()

    return _performance_config


def set_performance_config(config: PerformanceOptimizationConfig) -> None:
    """グローバルパフォーマンス設定を設定"""
    global _performance_config
    _performance_config = config
