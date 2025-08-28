#!/usr/bin/env python3
"""
特徴量ストア基本モジュール

基本的なデータクラスと設定を提供します。
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class FeatureMetadata:
    """
    特徴量メタデータ
    
    特徴量の詳細情報を保持するデータクラス。
    ファイル情報、生成時刻、統計情報などを含みます。
    """

    feature_id: str
    symbol: str
    start_date: str
    end_date: str
    config_hash: str
    feature_names: List[str]
    created_at: datetime
    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    generation_time_seconds: float
    strategy_used: str

    def to_dict(self) -> Dict[str, Any]:
        """
        辞書形式に変換
        
        Returns:
            メタデータの辞書表現
        """
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """
        辞書から復元
        
        Args:
            data: メタデータの辞書
            
        Returns:
            復元されたFeatureMetadataインスタンス
        """
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FeatureStoreConfig:
    """
    特徴量ストア設定
    
    特徴量ストアの動作を制御する設定パラメータを定義します。
    """

    # 基本設定
    base_path: str = "data/features"
    max_cache_age_days: int = 30
    max_cache_size_mb: int = 1024  # 1GB
    enable_compression: bool = True
    enable_metadata_indexing: bool = True
    cleanup_on_startup: bool = True
    auto_versioning: bool = True
    
    # バッチ処理並列化設定 (Issue #719対応)
    enable_parallel_batch_processing: bool = True
    max_parallel_workers: int = 4
    parallel_backend: str = "threading"  # "threading", "joblib"
    batch_chunk_size: int = 10
    
    # 特徴データ保存形式最適化設定 (Issue #718対応)
    storage_format: str = "auto"  # "pickle", "joblib", "numpy", "parquet", "auto"
    auto_format_threshold_mb: float = 1.0  # 1MB以上でnumpy/parquet使用
    enable_parquet: bool = True
    parquet_compression: str = "snappy"  # "snappy", "gzip", "lz4"
    
    # メタデータインデックスI/O効率改善設定 (Issue #717対応)
    metadata_index_format: str = "auto"  # "json", "sqlite", "pickle", "joblib", "auto"
    metadata_sqlite_cache_size: int = 1000  # SQLite キャッシュサイズ（KB）
    metadata_batch_size: int = 100  # バッチ挿入サイズ
    metadata_index_threshold: int = 1000  # 1000件以上でSQLite使用


def generate_feature_id(symbol: str, start_date: str, end_date: str, config_hash: str) -> str:
    """
    特徴量IDの生成
    
    シンボル、日付範囲、設定ハッシュから一意な特徴量IDを生成します。
    
    Args:
        symbol: 銘柄コード
        start_date: 開始日
        end_date: 終了日
        config_hash: 設定のハッシュ値
        
    Returns:
        生成された特徴量ID
    """
    id_string = f"{symbol}_{start_date}_{end_date}_{config_hash}"
    feature_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
    return feature_id


def generate_config_hash(feature_config) -> str:
    """
    特徴量設定のハッシュ値生成
    
    Args:
        feature_config: FeatureConfig インスタンス
        
    Returns:
        設定のハッシュ値
    """
    config_dict = asdict(feature_config)
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    return config_hash


class StatisticsTracker:
    """
    統計情報トラッカー
    
    特徴量ストアの利用統計を追跡します。
    """

    def __init__(self):
        """初期化"""
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "features_generated": 0,
            "features_loaded": 0,
            "total_generation_time": 0.0,
            "total_loading_time": 0.0,
        }

    def increment_cache_hits(self):
        """キャッシュヒット数を増加"""
        self.stats["cache_hits"] += 1

    def increment_cache_misses(self):
        """キャッシュミス数を増加"""
        self.stats["cache_misses"] += 1

    def increment_features_generated(self):
        """生成された特徴量数を増加"""
        self.stats["features_generated"] += 1

    def increment_features_loaded(self):
        """読み込まれた特徴量数を増加"""
        self.stats["features_loaded"] += 1

    def add_generation_time(self, time_seconds: float):
        """生成時間を追加"""
        self.stats["total_generation_time"] += time_seconds

    def add_loading_time(self, time_seconds: float):
        """読み込み時間を追加"""
        self.stats["total_loading_time"] += time_seconds

    def get_stats(self, cache_size_mb: float, features_in_cache: int) -> Dict[str, Any]:
        """
        統計情報の取得
        
        Args:
            cache_size_mb: キャッシュサイズ（MB）
            features_in_cache: キャッシュ内の特徴量数
            
        Returns:
            統計情報の辞書
        """
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        )

        return {
            **self.stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "features_in_cache": features_in_cache,
            "cache_size_mb": round(cache_size_mb, 2),
            "avg_generation_time_ms": round(
                (
                    self.stats["total_generation_time"]
                    / max(1, self.stats["features_generated"])
                )
                * 1000,
                2,
            ),
            "avg_loading_time_ms": round(
                (
                    self.stats["total_loading_time"]
                    / max(1, self.stats["features_loaded"])
                )
                * 1000,
                2,
            ),
        }


# 利用可能な外部ライブラリの確認
try:
    from joblib import Parallel, delayed
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False