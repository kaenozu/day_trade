#!/usr/bin/env python3
"""
特徴量ストア（Feature Store）システム
Issue #380: 機械学習モデルの最適化 - 特徴量生成の効率化

主要機能:
1. 特徴量の効率的保存・読み込み
2. 重複計算の排除と再利用機能
3. バッチ特徴量生成
4. バージョン管理とメタデータ追跡
"""

import hashlib
import json
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Issue #719対応: 並列処理のためのオプション
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Issue #718対応: 最適化された保存形式のオプション
try:
    import joblib
    JOBLIB_SAVE_AVAILABLE = True
except ImportError:
    JOBLIB_SAVE_AVAILABLE = False

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

from ..analysis.feature_engineering_unified import (
    FeatureConfig,
    FeatureEngineeringManager,
    FeatureResult,
)
from ..core.optimization_strategy import OptimizationConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class FeatureMetadata:
    """特徴量メタデータ"""

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
        """辞書形式に変換"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMetadata":
        """辞書から復元"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class FeatureStoreConfig:
    """特徴量ストア設定"""

    base_path: str = "data/features"
    max_cache_age_days: int = 30
    max_cache_size_mb: int = 1024  # 1GB
    enable_compression: bool = True
    enable_metadata_indexing: bool = True
    cleanup_on_startup: bool = True
    auto_versioning: bool = True
    # Issue #719対応: バッチ処理並列化設定
    enable_parallel_batch_processing: bool = True
    max_parallel_workers: int = 4
    parallel_backend: str = "threading"  # "threading", "joblib"
    batch_chunk_size: int = 10
    # Issue #718対応: 特徴データ保存形式最適化設定
    storage_format: str = "auto"  # "pickle", "joblib", "numpy", "parquet", "auto"
    auto_format_threshold_mb: float = 1.0  # 1MB以上でnumpy/parquet使用
    enable_parquet: bool = True
    parquet_compression: str = "snappy"  # "snappy", "gzip", "lz4"
    # Issue #717対応: メタデータインデックスI/O効率改善設定
    metadata_index_format: str = "auto"  # "json", "sqlite", "pickle", "joblib", "auto"
    metadata_sqlite_cache_size: int = 1000  # SQLite キャッシュサイズ（KB）
    metadata_batch_size: int = 100  # バッチ挿入サイズ
    metadata_index_threshold: int = 1000  # 1000件以上でSQLite使用


class FeatureStore:
    """特徴量ストア - 特徴量の保存・再利用システム"""

    def __init__(self, config: FeatureStoreConfig = None):
        """初期化"""
        self.config = config or FeatureStoreConfig()
        self.base_path = Path(self.config.base_path)
        self.metadata_path = self.base_path / "metadata"

        # ディレクトリ作成
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)

        # メタデータインデックス
        self.metadata_index: Dict[str, FeatureMetadata] = {}

        # Issue #717対応: SQLiteデータベース接続（オンデマンド）
        self._sqlite_conn: Optional[sqlite3.Connection] = None
        self._sqlite_db_path = self.metadata_path / "index.db"

        # 統計情報
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "features_generated": 0,
            "features_loaded": 0,
            "total_generation_time": 0.0,
            "total_loading_time": 0.0,
        }

        # 初期化処理
        self._initialize()

    def _initialize(self):
        """初期化処理"""
        logger.info(
            "特徴量ストア初期化開始",
            extra={"base_path": str(self.base_path), "config": asdict(self.config)},
        )

        # メタデータの読み込み
        self._load_metadata_index()

        # クリーンアップ（設定による）
        if self.config.cleanup_on_startup:
            self._cleanup_old_features()

        logger.info(
            "特徴量ストア初期化完了",
            extra={
                "features_count": len(self.metadata_index),
                "cache_size_mb": self._get_cache_size_mb(),
            },
        )

    def _load_metadata_index(self):
        """メタデータインデックスの読み込み"""
        # Issue #717対応: 形式自動選択または設定に基づく読み込み
        if self.config.metadata_index_format == "auto":
            # メタデータ件数推定に基づいて形式を決定
            estimated_count = self._estimate_metadata_count()
            if estimated_count >= self.config.metadata_index_threshold:
                format_to_use = "sqlite"
            else:
                format_to_use = "json"
        else:
            format_to_use = self.config.metadata_index_format

        logger.debug(f"メタデータインデックス読み込み形式: {format_to_use}")

        try:
            if format_to_use == "sqlite":
                self._load_metadata_index_sqlite()
            elif format_to_use == "pickle":
                self._load_metadata_index_pickle()
            elif format_to_use == "joblib" and JOBLIB_SAVE_AVAILABLE:
                self._load_metadata_index_joblib()
            else:
                self._load_metadata_index_json()
                
            logger.info(
                f"メタデータインデックス読み込み完了: {len(self.metadata_index)}件 (形式: {format_to_use})"
            )
        except Exception as e:
            logger.error(f"メタデータインデックス読み込みエラー ({format_to_use}): {e}")
            self.metadata_index = {}

    def _save_metadata_index(self):
        """メタデータインデックスの保存"""
        # Issue #717対応: 形式自動選択または設定に基づく保存
        if self.config.metadata_index_format == "auto":
            # メタデータ件数に基づいて形式を決定
            if len(self.metadata_index) >= self.config.metadata_index_threshold:
                format_to_use = "sqlite"
            else:
                format_to_use = "json"
        else:
            format_to_use = self.config.metadata_index_format

        try:
            if format_to_use == "sqlite":
                self._save_metadata_index_sqlite()
            elif format_to_use == "pickle":
                self._save_metadata_index_pickle()
            elif format_to_use == "joblib" and JOBLIB_SAVE_AVAILABLE:
                self._save_metadata_index_joblib()
            else:
                self._save_metadata_index_json()
                
            logger.debug(
                f"メタデータインデックス保存完了: {len(self.metadata_index)}件 (形式: {format_to_use})"
            )
        except Exception as e:
            logger.error(f"メタデータインデックス保存エラー ({format_to_use}): {e}")

    def _generate_feature_id(
        self, symbol: str, start_date: str, end_date: str, config_hash: str
    ) -> str:
        """特徴量IDの生成"""
        # 一意な識別子の生成
        id_string = f"{symbol}_{start_date}_{end_date}_{config_hash}"
        feature_id = hashlib.md5(id_string.encode()).hexdigest()[:16]
        return feature_id

    def _generate_config_hash(self, feature_config: FeatureConfig) -> str:
        """特徴量設定のハッシュ値生成"""
        # 設定を正規化してハッシュ化
        config_dict = asdict(feature_config)
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return config_hash

    def _get_feature_file_path(self, feature_id: str) -> Path:
        """特徴量ファイルパスの取得"""
        # 2レベルディレクトリ構造（パフォーマンス向上）
        dir1 = feature_id[:2]
        dir2 = feature_id[2:4]

        feature_dir = self.base_path / dir1 / dir2
        feature_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{feature_id}.pkl"
        if self.config.enable_compression:
            filename += ".gz"

        return feature_dir / filename

    def _cleanup_old_features(self):
        """古い特徴量のクリーンアップ"""
        if not self.config.max_cache_age_days:
            return

        cutoff_date = datetime.now() - timedelta(days=self.config.max_cache_age_days)
        removed_count = 0

        feature_ids_to_remove = []

        for feature_id, metadata in self.metadata_index.items():
            if metadata.created_at < cutoff_date:
                feature_ids_to_remove.append(feature_id)

        for feature_id in feature_ids_to_remove:
            try:
                self._remove_feature(feature_id)
                removed_count += 1
            except Exception as e:
                logger.warning(f"特徴量削除エラー: {feature_id} - {e}")

        if removed_count > 0:
            logger.info(f"古い特徴量をクリーンアップ: {removed_count}件")

    def _remove_feature(self, feature_id: str):
        """特徴量の削除"""
        if feature_id in self.metadata_index:
            metadata = self.metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # ファイル削除
            if file_path.exists():
                file_path.unlink()

            # メタデータから削除
            del self.metadata_index[feature_id]

            logger.debug(f"特徴量削除完了: {feature_id}")

    def _get_cache_size_mb(self) -> float:
        """キャッシュサイズの取得"""
        total_size = 0
        for metadata in self.metadata_index.values():
            total_size += metadata.file_size_bytes
        return total_size / 1024 / 1024

    # Issue #717対応: メタデータインデックスI/O効率改善メソッド群

    def _estimate_metadata_count(self) -> int:
        """メタデータ件数の推定（形式選択用）"""
        # 既存ファイルからの推定
        json_file = self.metadata_path / "index.json"
        sqlite_file = self._sqlite_db_path
        pickle_file = self.metadata_path / "index.pkl"
        joblib_file = self.metadata_path / "index.joblib"

        if sqlite_file.exists():
            try:
                conn = sqlite3.connect(sqlite_file)
                cursor = conn.execute("SELECT COUNT(*) FROM metadata")
                count = cursor.fetchone()[0]
                conn.close()
                return count
            except Exception:
                pass

        if json_file.exists():
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                return len(data)
            except Exception:
                pass

        if pickle_file.exists():
            try:
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                return len(data)
            except Exception:
                pass

        if joblib_file.exists() and JOBLIB_SAVE_AVAILABLE:
            try:
                data = joblib.load(joblib_file)
                return len(data)
            except Exception:
                pass

        return 0

    @contextmanager
    def _get_sqlite_connection(self):
        """SQLiteデータベース接続の取得（コンテキストマネージャー）"""
        conn = None
        try:
            conn = sqlite3.connect(
                self._sqlite_db_path,
                timeout=30,  # 30秒タイムアウト
                check_same_thread=False
            )
            
            # Issue #717対応: SQLiteキャッシュサイズ設定
            conn.execute(f"PRAGMA cache_size = {self.config.metadata_sqlite_cache_size}")
            conn.execute("PRAGMA synchronous = NORMAL")  # パフォーマンス向上
            conn.execute("PRAGMA journal_mode = WAL")     # 並行アクセス改善
            conn.execute("PRAGMA temp_store = MEMORY")    # 一時データメモリ格納
            
            yield conn
        finally:
            if conn:
                conn.close()

    def _create_metadata_table(self, conn: sqlite3.Connection):
        """メタデータテーブルの作成"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                feature_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                feature_names TEXT NOT NULL,  -- JSON形式で保存
                created_at TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                generation_time_seconds REAL NOT NULL,
                strategy_used TEXT NOT NULL
            )
        """)
        
        # インデックス作成（クエリパフォーマンス向上）
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON metadata(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date_range ON metadata(start_date, end_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_config_hash ON metadata(config_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON metadata(created_at)")
        
        conn.commit()

    def _load_metadata_index_json(self):
        """JSON形式からのメタデータインデックス読み込み"""
        index_file = self.metadata_path / "index.json"

        if index_file.exists():
            try:
                with open(index_file, encoding="utf-8") as f:
                    index_data = json.load(f)

                for feature_id, metadata_dict in index_data.items():
                    self.metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"JSON形式メタデータ読み込みエラー: {e}")
                raise

    def _load_metadata_index_sqlite(self):
        """SQLite形式からのメタデータインデックス読み込み"""
        if not self._sqlite_db_path.exists():
            logger.info("新しいSQLiteメタデータインデックスを作成")
            return

        try:
            with self._get_sqlite_connection() as conn:
                self._create_metadata_table(conn)
                
                cursor = conn.execute("""
                    SELECT feature_id, symbol, start_date, end_date, config_hash,
                           feature_names, created_at, file_path, file_size_bytes,
                           row_count, column_count, generation_time_seconds, strategy_used
                    FROM metadata
                """)
                
                for row in cursor:
                    feature_id = row[0]
                    feature_names = json.loads(row[5])  # JSON文字列をリストに変換
                    
                    metadata = FeatureMetadata(
                        feature_id=feature_id,
                        symbol=row[1],
                        start_date=row[2],
                        end_date=row[3],
                        config_hash=row[4],
                        feature_names=feature_names,
                        created_at=datetime.fromisoformat(row[6]),
                        file_path=row[7],
                        file_size_bytes=row[8],
                        row_count=row[9],
                        column_count=row[10],
                        generation_time_seconds=row[11],
                        strategy_used=row[12]
                    )
                    
                    self.metadata_index[feature_id] = metadata

        except Exception as e:
            logger.error(f"SQLite形式メタデータ読み込みエラー: {e}")
            raise

    def _load_metadata_index_pickle(self):
        """Pickle形式からのメタデータインデックス読み込み"""
        index_file = self.metadata_path / "index.pkl"

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    index_data = pickle.load(f)

                for feature_id, metadata_dict in index_data.items():
                    self.metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"Pickle形式メタデータ読み込みエラー: {e}")
                raise

    def _load_metadata_index_joblib(self):
        """Joblib形式からのメタデータインデックス読み込み"""
        index_file = self.metadata_path / "index.joblib"

        if index_file.exists():
            try:
                index_data = joblib.load(index_file)

                for feature_id, metadata_dict in index_data.items():
                    self.metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"Joblib形式メタデータ読み込みエラー: {e}")
                raise

    def _save_metadata_index_json(self):
        """JSON形式でのメタデータインデックス保存"""
        index_file = self.metadata_path / "index.json"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in self.metadata_index.items()
            }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"JSON形式メタデータ保存エラー: {e}")
            raise

    def _save_metadata_index_sqlite(self):
        """SQLite形式でのメタデータインデックス保存"""
        try:
            with self._get_sqlite_connection() as conn:
                self._create_metadata_table(conn)
                
                # 全データをクリアして再挿入（簡易実装）
                conn.execute("DELETE FROM metadata")
                
                # バッチ挿入でパフォーマンス向上
                batch_data = []
                for feature_id, metadata in self.metadata_index.items():
                    batch_data.append((
                        feature_id,
                        metadata.symbol,
                        metadata.start_date,
                        metadata.end_date,
                        metadata.config_hash,
                        json.dumps(metadata.feature_names),  # リストをJSON文字列に変換
                        metadata.created_at.isoformat(),
                        metadata.file_path,
                        metadata.file_size_bytes,
                        metadata.row_count,
                        metadata.column_count,
                        metadata.generation_time_seconds,
                        metadata.strategy_used
                    ))
                
                # バッチサイズに分けて挿入
                batch_size = self.config.metadata_batch_size
                for i in range(0, len(batch_data), batch_size):
                    batch = batch_data[i:i + batch_size]
                    conn.executemany("""
                        INSERT OR REPLACE INTO metadata 
                        (feature_id, symbol, start_date, end_date, config_hash,
                         feature_names, created_at, file_path, file_size_bytes,
                         row_count, column_count, generation_time_seconds, strategy_used)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                
                conn.commit()

        except Exception as e:
            logger.error(f"SQLite形式メタデータ保存エラー: {e}")
            raise

    def _save_metadata_index_pickle(self):
        """Pickle形式でのメタデータインデックス保存"""
        index_file = self.metadata_path / "index.pkl"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in self.metadata_index.items()
            }

            with open(index_file, "wb") as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            logger.error(f"Pickle形式メタデータ保存エラー: {e}")
            raise

    def _save_metadata_index_joblib(self):
        """Joblib形式でのメタデータインデックス保存"""
        index_file = self.metadata_path / "index.joblib"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in self.metadata_index.items()
            }

            joblib.dump(index_data, index_file, compress=3)  # 圧縮レベル3

        except Exception as e:
            logger.error(f"Joblib形式メタデータ保存エラー: {e}")
            raise

    def has_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> bool:
        """特徴量の存在確認"""
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        if feature_id not in self.metadata_index:
            return False

        # ファイルの実際の存在確認
        metadata = self.metadata_index[feature_id]
        file_path = Path(metadata.file_path)

        if not file_path.exists():
            # メタデータから削除
            del self.metadata_index[feature_id]
            return False

        return True

    def load_feature(
        self, symbol: str, start_date: str, end_date: str, feature_config: FeatureConfig
    ) -> Optional[FeatureResult]:
        """
        Issue #718対応: 特徴量の読み込み（最適化された保存形式対応）
        """
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        if not self.has_feature(symbol, start_date, end_date, feature_config):
            self.stats["cache_misses"] += 1
            return None

        start_time = time.time()

        try:
            metadata = self.metadata_index[feature_id]
            file_path = Path(metadata.file_path)

            # Issue #718対応: 保存形式の判定と読み込み
            storage_format = self._detect_storage_format(file_path)
            feature_data = self._load_feature_data(file_path, storage_format)

            # FeatureResultの復元
            feature_result = FeatureResult(
                features=feature_data["features"],
                feature_names=metadata.feature_names,
                metadata={
                    **feature_data["metadata"],
                    "loaded_from_cache": True,
                    "cache_feature_id": feature_id,
                },
                generation_time=metadata.generation_time_seconds,
                strategy_used=metadata.strategy_used,
            )

            loading_time = time.time() - start_time

            # 統計更新
            self.stats["cache_hits"] += 1
            self.stats["features_loaded"] += 1
            self.stats["total_loading_time"] += loading_time

            logger.info(
                "特徴量キャッシュから読み込み完了",
                extra={
                    "feature_id": feature_id,
                    "symbol": symbol,
                    "loading_time_ms": round(loading_time * 1000, 2),
                    "feature_count": len(metadata.feature_names),
                    "data_shape": feature_data["features"].shape,
                },
            )

            return feature_result

        except Exception as e:
            logger.error(f"特徴量読み込みエラー: {feature_id} - {e}")
            # 破損したキャッシュの削除
            self._remove_feature(feature_id)
            self.stats["cache_misses"] += 1
            return None

    def save_feature(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        feature_result: FeatureResult,
    ) -> str:
        """
        Issue #718対応: 特徴量の保存（最適化された保存形式対応）
        """
        config_hash = self._generate_config_hash(feature_config)
        feature_id = self._generate_feature_id(
            symbol, start_date, end_date, config_hash
        )

        start_time = time.time()

        try:
            # 保存データの準備
            feature_data = {
                "features": feature_result.features,
                "metadata": feature_result.metadata,
            }

            # Issue #718対応: 保存形式の決定
            storage_format = self._determine_storage_format(feature_data)
            
            # ファイルパス取得（形式に応じた拡張子）
            file_path = self._get_feature_file_path(feature_id, storage_format)

            # データ保存
            self._save_feature_data(feature_data, file_path, storage_format)

            # ファイルサイズ取得
            file_size = file_path.stat().st_size

            # メタデータ作成
            metadata = FeatureMetadata(
                feature_id=feature_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                config_hash=config_hash,
                feature_names=feature_result.feature_names,
                created_at=datetime.now(),
                file_path=str(file_path),
                file_size_bytes=file_size,
                row_count=len(feature_result.features),
                column_count=len(feature_result.feature_names),
                generation_time_seconds=feature_result.generation_time,
                strategy_used=feature_result.strategy_used,
            )

            # メタデータインデックスに追加
            self.metadata_index[feature_id] = metadata

            # インデックス保存
            self._save_metadata_index()

            saving_time = time.time() - start_time

            # 統計更新
            self.stats["features_generated"] += 1

            logger.info(
                "特徴量保存完了",
                extra={
                    "feature_id": feature_id,
                    "symbol": symbol,
                    "file_size_mb": round(file_size / 1024 / 1024, 3),
                    "saving_time_ms": round(saving_time * 1000, 2),
                    "feature_count": len(feature_result.feature_names),
                    "data_shape": feature_result.features.shape,
                    "compression": self.config.enable_compression,
                    # Issue #718対応: 保存形式情報の追加
                    "storage_format": storage_format,
                },
            )

            return feature_id

        except Exception as e:
            logger.error(f"特徴量保存エラー: {feature_id} - {e}")
            raise

    def get_or_generate_feature(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> FeatureResult:
        """特徴量の取得または生成（メインインターフェース）"""
        # キャッシュから読み込み試行
        cached_feature = self.load_feature(symbol, start_date, end_date, feature_config)
        if cached_feature:
            return cached_feature

        # 特徴量生成
        logger.info(
            "特徴量生成開始",
            extra={
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "data_shape": data.shape,
            },
        )

        manager = FeatureEngineeringManager(optimization_config)
        feature_result = manager.generate_features(data, feature_config)

        # 保存
        feature_id = self.save_feature(
            symbol, start_date, end_date, feature_config, feature_result
        )

        # メタデータに特徴量IDを追加
        feature_result.metadata["feature_id"] = feature_id

        return feature_result

    def batch_generate_features(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        Issue #719対応: バッチ特徴量生成（並列化対応）
        """
        logger.info(
            "バッチ特徴量生成開始",
            extra={
                "symbols_count": len(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "parallel_enabled": self.config.enable_parallel_batch_processing,
                "parallel_backend": self.config.parallel_backend,
                "max_workers": self.config.max_parallel_workers,
            },
        )

        # Issue #719対応: 並列化が有効で複数シンボルがある場合は並列処理
        if (self.config.enable_parallel_batch_processing and len(symbols) > 1):
            return self._batch_generate_features_parallel(
                symbols, data_dict, start_date, end_date, feature_config, optimization_config
            )
        else:
            return self._batch_generate_features_sequential(
                symbols, data_dict, start_date, end_date, feature_config, optimization_config
            )

    def _batch_generate_features_sequential(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """順次バッチ特徴量生成（従来の実装）"""
        results = {}
        cache_hits = 0
        cache_misses = 0

        for symbol in symbols:
            if symbol not in data_dict:
                logger.warning(f"データが見つかりません: {symbol}")
                continue

            try:
                # キャッシュ確認
                if self.has_feature(symbol, start_date, end_date, feature_config):
                    cached_result = self.load_feature(
                        symbol, start_date, end_date, feature_config
                    )
                    if cached_result:
                        results[symbol] = cached_result
                        cache_hits += 1
                        continue

                # 生成
                data = data_dict[symbol]
                feature_result = self.get_or_generate_feature(
                    symbol,
                    data,
                    start_date,
                    end_date,
                    feature_config,
                    optimization_config,
                )
                results[symbol] = feature_result
                cache_misses += 1

            except Exception as e:
                logger.error(f"バッチ特徴量生成エラー: {symbol} - {e}")
                continue

        logger.info(
            "順次バッチ特徴量生成完了",
            extra={
                "processed_symbols": len(results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": (
                    f"{cache_hits / (cache_hits + cache_misses) * 100:.1f}%"
                    if (cache_hits + cache_misses) > 0
                    else "0%"
                ),
            },
        )

        return results

    def _batch_generate_features_parallel(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Dict[str, FeatureResult]:
        """
        Issue #719対応: 並列バッチ特徴量生成
        """
        results = {}
        total_cache_hits = 0
        total_cache_misses = 0

        # バックエンドに応じた並列処理
        if self.config.parallel_backend == "joblib" and JOBLIB_AVAILABLE:
            results, total_cache_hits, total_cache_misses = self._parallel_process_joblib(
                symbols, data_dict, start_date, end_date, feature_config, optimization_config
            )
        else:
            # Threading並列処理（デフォルト）
            results, total_cache_hits, total_cache_misses = self._parallel_process_threading(
                symbols, data_dict, start_date, end_date, feature_config, optimization_config
            )

        logger.info(
            "並列バッチ特徴量生成完了",
            extra={
                "processed_symbols": len(results),
                "cache_hits": total_cache_hits,
                "cache_misses": total_cache_misses,
                "cache_hit_rate": (
                    f"{total_cache_hits / (total_cache_hits + total_cache_misses) * 100:.1f}%"
                    if (total_cache_hits + total_cache_misses) > 0
                    else "0%"
                ),
                "parallel_backend": self.config.parallel_backend,
            },
        )

        return results

    def _process_single_symbol(
        self,
        symbol: str,
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[str, Optional[FeatureResult], bool, str]:
        """
        Issue #719対応: 単一シンボル処理（並列実行用）
        
        Returns:
            (symbol, result, is_cache_hit, error_message)
        """
        if symbol not in data_dict:
            return symbol, None, False, f"データが見つかりません: {symbol}"

        try:
            # キャッシュ確認
            if self.has_feature(symbol, start_date, end_date, feature_config):
                cached_result = self.load_feature(
                    symbol, start_date, end_date, feature_config
                )
                if cached_result:
                    return symbol, cached_result, True, ""

            # 生成
            data = data_dict[symbol]
            feature_result = self.get_or_generate_feature(
                symbol,
                data,
                start_date,
                end_date,
                feature_config,
                optimization_config,
            )
            return symbol, feature_result, False, ""

        except Exception as e:
            error_msg = f"バッチ特徴量生成エラー: {symbol} - {e}"
            return symbol, None, False, error_msg

    def _parallel_process_threading(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[Dict[str, FeatureResult], int, int]:
        """
        Issue #719対応: Threading並列処理
        
        Returns:
            (results, cache_hits, cache_misses)
        """
        results = {}
        cache_hits = 0
        cache_misses = 0

        with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
            # 各シンボルを並列で処理
            future_to_symbol = {
                executor.submit(
                    self._process_single_symbol,
                    symbol, data_dict, start_date, end_date, feature_config, optimization_config
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                try:
                    symbol, result, is_cache_hit, error_msg = future.result()
                    
                    if result is not None:
                        results[symbol] = result
                        if is_cache_hit:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    elif error_msg:
                        logger.error(error_msg)
                        
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.error(f"Threading並列処理エラー ({symbol}): {e}")

        return results, cache_hits, cache_misses

    def _parallel_process_joblib(
        self,
        symbols: List[str],
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        feature_config: FeatureConfig,
        optimization_config: OptimizationConfig = None,
    ) -> Tuple[Dict[str, FeatureResult], int, int]:
        """
        Issue #719対応: Joblib並列処理
        
        Returns:
            (results, cache_hits, cache_misses)
        """
        try:
            # Joblib並列実行
            results_list = Parallel(n_jobs=self.config.max_parallel_workers)(
                delayed(self._process_single_symbol)(
                    symbol, data_dict, start_date, end_date, feature_config, optimization_config
                )
                for symbol in symbols
            )

            # 結果を集約
            results = {}
            cache_hits = 0
            cache_misses = 0

            for symbol, result, is_cache_hit, error_msg in results_list:
                if result is not None:
                    results[symbol] = result
                    if is_cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1
                elif error_msg:
                    logger.error(error_msg)

            return results, cache_hits, cache_misses

        except Exception as e:
            logger.error(f"Joblib並列処理エラー: {e}")
            # フォールバック: Threading並列処理
            logger.warning("Joblib処理失敗、Threading並列処理にフォールバック")
            return self._parallel_process_threading(
                symbols, data_dict, start_date, end_date, feature_config, optimization_config
            )

    def _determine_storage_format(self, feature_data: Dict[str, Any]) -> str:
        """
        Issue #718対応: データ特性に基づく保存形式の決定
        
        Args:
            feature_data: 保存するデータ
            
        Returns:
            保存形式 ('pickle', 'joblib', 'numpy', 'parquet')
        """
        if self.config.storage_format != "auto":
            return self.config.storage_format
            
        features = feature_data.get("features")
        
        # データサイズの推定
        data_size_mb = 0
        if hasattr(features, 'nbytes'):
            data_size_mb = features.nbytes / (1024 * 1024)
        elif hasattr(features, '__len__'):
            # 概算
            data_size_mb = len(str(feature_data)) / (1024 * 1024)
        
        # NumPy配列の場合
        if NUMPY_AVAILABLE and hasattr(features, 'dtype') and hasattr(features, 'shape'):
            if data_size_mb > self.config.auto_format_threshold_mb:
                return 'numpy'
            else:
                return 'joblib' if JOBLIB_SAVE_AVAILABLE else 'pickle'
        
        # DataFrame の場合
        if hasattr(features, 'to_parquet') and PARQUET_AVAILABLE and self.config.enable_parquet:
            if data_size_mb > self.config.auto_format_threshold_mb:
                return 'parquet'
        
        # デフォルト
        if data_size_mb > self.config.auto_format_threshold_mb and JOBLIB_SAVE_AVAILABLE:
            return 'joblib'
        else:
            return 'pickle'

    def _get_feature_file_path(self, feature_id: str, storage_format: str = "pickle") -> Path:
        """
        Issue #718対応: 保存形式に応じたファイルパス取得
        
        Args:
            feature_id: 特徴ID
            storage_format: 保存形式
            
        Returns:
            ファイルパス
        """
        # 拡張子マッピング
        extension_map = {
            'pickle': '.pkl',
            'joblib': '.joblib',
            'numpy': '.npy',
            'parquet': '.parquet'
        }
        
        extension = extension_map.get(storage_format, '.pkl')
        if self.config.enable_compression and storage_format in ['pickle', 'joblib']:
            extension += '.gz'
            
        filename = f"{feature_id}{extension}"
        return self.base_path / filename

    def _save_feature_data(self, feature_data: Dict[str, Any], file_path: Path, storage_format: str):
        """
        Issue #718対応: 保存形式に応じたデータ保存
        
        Args:
            feature_data: 保存するデータ
            file_path: 保存先ファイルパス
            storage_format: 保存形式
        """
        if storage_format == 'numpy' and NUMPY_AVAILABLE:
            self._save_as_numpy(feature_data, file_path)
        elif storage_format == 'joblib' and JOBLIB_SAVE_AVAILABLE:
            self._save_as_joblib(feature_data, file_path)
        elif storage_format == 'parquet' and PARQUET_AVAILABLE:
            self._save_as_parquet(feature_data, file_path)
        else:
            # デフォルト: pickle
            self._save_as_pickle(feature_data, file_path)

    def _load_feature_data(self, file_path: Path, storage_format: str) -> Dict[str, Any]:
        """
        Issue #718対応: 保存形式に応じたデータ読み込み
        
        Args:
            file_path: 読み込み元ファイルパス
            storage_format: 保存形式
            
        Returns:
            読み込まれたデータ
        """
        if storage_format == 'numpy' and NUMPY_AVAILABLE:
            return self._load_from_numpy(file_path)
        elif storage_format == 'joblib' and JOBLIB_SAVE_AVAILABLE:
            return self._load_from_joblib(file_path)
        elif storage_format == 'parquet' and PARQUET_AVAILABLE:
            return self._load_from_parquet(file_path)
        else:
            # デフォルト: pickle
            return self._load_from_pickle(file_path)

    def _detect_storage_format(self, file_path: Path) -> str:
        """
        Issue #718対応: ファイル拡張子からの保存形式判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            保存形式
        """
        suffix = file_path.suffix.lower()
        # 圧縮ファイルの場合は二重拡張子をチェック
        if suffix == '.gz':
            suffix = file_path.suffixes[-2].lower() if len(file_path.suffixes) >= 2 else '.gz'
        
        format_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle', 
            '.joblib': 'joblib',
            '.npy': 'numpy',
            '.parquet': 'parquet'
        }
        
        return format_map.get(suffix, 'pickle')

    def _save_as_pickle(self, feature_data: Dict[str, Any], file_path: Path):
        """Pickle形式で保存"""
        if self.config.enable_compression:
            import gzip
            with gzip.open(file_path, "wb") as f:
                pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(feature_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_from_pickle(self, file_path: Path) -> Dict[str, Any]:
        """Pickle形式から読み込み"""
        if self.config.enable_compression and file_path.suffix == ".gz":
            import gzip
            with gzip.open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(file_path, "rb") as f:
                return pickle.load(f)

    def _save_as_joblib(self, feature_data: Dict[str, Any], file_path: Path):
        """Joblib形式で保存（NumPyデータに最適化）"""
        if self.config.enable_compression:
            joblib.dump(feature_data, file_path, compress=3)  # 圧縮レベル3
        else:
            joblib.dump(feature_data, file_path)

    def _load_from_joblib(self, file_path: Path) -> Dict[str, Any]:
        """Joblib形式から読み込み"""
        return joblib.load(file_path)

    def _save_as_numpy(self, feature_data: Dict[str, Any], file_path: Path):
        """NumPy形式で保存（特徴量がNumPy配列の場合）"""
        features = feature_data["features"]
        metadata = feature_data["metadata"]
        
        # ファイル名から.npyを除去してベースパスを作成
        base_path = str(file_path).rstrip('.npy')
        
        # NumPy配列を保存
        features_path = f"{base_path}_features.npy"
        np.save(features_path, features)
        
        # メタデータはPickleで保存
        metadata_path = f"{base_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def _load_from_numpy(self, file_path: Path) -> Dict[str, Any]:
        """NumPy形式から読み込み"""
        # ファイル名から.npyを除去してベースパスを作成
        base_path = str(file_path).rstrip('.npy')
        
        # NumPy配列を読み込み
        features_path = f"{base_path}_features.npy"
        features = np.load(features_path)
        
        # メタデータを読み込み
        metadata_path = f"{base_path}_metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return {"features": features, "metadata": metadata}

    def _save_as_parquet(self, feature_data: Dict[str, Any], file_path: Path):
        """Parquet形式で保存（DataFrameの場合）"""
        features = feature_data["features"]
        metadata = feature_data["metadata"]
        
        if hasattr(features, 'to_parquet'):
            # DataFrameの場合
            features.to_parquet(file_path, compression=self.config.parquet_compression)
            
            # メタデータは別ファイルで保存
            metadata_path = str(file_path).replace('.parquet', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        else:
            # DataFrameでない場合はfallback
            self._save_as_pickle(feature_data, file_path.with_suffix('.pkl'))

    def _load_from_parquet(self, file_path: Path) -> Dict[str, Any]:
        """Parquet形式から読み込み"""
        import pandas as pd
        
        # Parquetファイルを読み込み
        features = pd.read_parquet(file_path)
        
        # メタデータを読み込み
        metadata_path = str(file_path).replace('.parquet', '_metadata.pkl')
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except FileNotFoundError:
            metadata = {}
        
        return {"features": features, "metadata": metadata}

    def get_stats(self) -> Dict[str, Any]:
        """統計情報の取得"""
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / cache_total * 100) if cache_total > 0 else 0
        )

        return {
            **self.stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "features_in_cache": len(self.metadata_index),
            "cache_size_mb": round(self._get_cache_size_mb(), 2),
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

    def cleanup_cache(self, force: bool = False):
        """キャッシュのクリーンアップ"""
        if force or self._get_cache_size_mb() > self.config.max_cache_size_mb:
            # 最も古い特徴量から削除
            sorted_metadata = sorted(
                self.metadata_index.items(), key=lambda x: x[1].created_at
            )

            removed_count = 0
            target_size_mb = self.config.max_cache_size_mb * 0.8  # 80%まで削減

            for feature_id, metadata in sorted_metadata:
                if self._get_cache_size_mb() <= target_size_mb:
                    break

                self._remove_feature(feature_id)
                removed_count += 1

            self._save_metadata_index()

            # Issue #717対応: SQLiteデータベースVACUUM（最適化）
            if self._sqlite_db_path.exists():
                try:
                    with self._get_sqlite_connection() as conn:
                        conn.execute("VACUUM")  # データベース最適化
                    logger.debug("SQLiteデータベース最適化完了")
                except Exception as e:
                    logger.warning(f"SQLiteデータベース最適化エラー: {e}")

            logger.info(f"キャッシュクリーンアップ完了: {removed_count}件削除")

    @contextmanager
    def batch_context(self):
        """バッチ処理用コンテキストマネージャー"""
        try:
            logger.info("バッチ処理開始")
            yield self
        finally:
            # バッチ処理後のクリーンアップ
            self._save_metadata_index()
            logger.info("バッチ処理完了")


# 便利関数
def create_feature_store(config: FeatureStoreConfig = None) -> FeatureStore:
    """特徴量ストアの作成"""
    return FeatureStore(config)
