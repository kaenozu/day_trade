#!/usr/bin/env python3
"""
メタデータマネージャー

特徴量メタデータのインデックス管理機能を提供します。
Issue #717対応: メタデータインデックスI/O効率改善
"""

import json
import pickle
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

from .base import (
    JOBLIB_AVAILABLE,
    FeatureMetadata,
    FeatureStoreConfig,
)
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

if JOBLIB_AVAILABLE:
    import joblib


class MetadataIndex:
    """
    メタデータインデックス基底クラス
    
    メタデータの保存・読み込みインターフェースを定義します。
    """

    def __init__(self, metadata_path: Path, config: FeatureStoreConfig):
        """
        初期化
        
        Args:
            metadata_path: メタデータ保存パス
            config: 特徴量ストア設定
        """
        self.metadata_path = metadata_path
        self.config = config

    def load(self) -> Dict[str, FeatureMetadata]:
        """メタデータ読み込み（サブクラスで実装）"""
        raise NotImplementedError

    def save(self, metadata_index: Dict[str, FeatureMetadata]):
        """メタデータ保存（サブクラスで実装）"""
        raise NotImplementedError


class JsonMetadataIndex(MetadataIndex):
    """
    JSON形式のメタデータインデックス
    
    小〜中規模のメタデータに適しています。
    """

    def load(self) -> Dict[str, FeatureMetadata]:
        """JSON形式からのメタデータ読み込み"""
        index_file = self.metadata_path / "index.json"
        metadata_index = {}

        if index_file.exists():
            try:
                with open(index_file, encoding="utf-8") as f:
                    index_data = json.load(f)

                for feature_id, metadata_dict in index_data.items():
                    metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"JSON形式メタデータ読み込みエラー: {e}")
                raise

        return metadata_index

    def save(self, metadata_index: Dict[str, FeatureMetadata]):
        """JSON形式でのメタデータ保存"""
        index_file = self.metadata_path / "index.json"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in metadata_index.items()
            }

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"JSON形式メタデータ保存エラー: {e}")
            raise


class PickleMetadataIndex(MetadataIndex):
    """
    Pickle形式のメタデータインデックス
    
    高速な保存・読み込みが可能です。
    """

    def load(self) -> Dict[str, FeatureMetadata]:
        """Pickle形式からのメタデータ読み込み"""
        index_file = self.metadata_path / "index.pkl"
        metadata_index = {}

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    index_data = pickle.load(f)

                for feature_id, metadata_dict in index_data.items():
                    metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"Pickle形式メタデータ読み込みエラー: {e}")
                raise

        return metadata_index

    def save(self, metadata_index: Dict[str, FeatureMetadata]):
        """Pickle形式でのメタデータ保存"""
        index_file = self.metadata_path / "index.pkl"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in metadata_index.items()
            }

            with open(index_file, "wb") as f:
                pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            logger.error(f"Pickle形式メタデータ保存エラー: {e}")
            raise


class JoblibMetadataIndex(MetadataIndex):
    """
    Joblib形式のメタデータインデックス
    
    圧縮により効率的なストレージを提供します。
    """

    def load(self) -> Dict[str, FeatureMetadata]:
        """Joblib形式からのメタデータ読み込み"""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is not available")

        index_file = self.metadata_path / "index.joblib"
        metadata_index = {}

        if index_file.exists():
            try:
                index_data = joblib.load(index_file)

                for feature_id, metadata_dict in index_data.items():
                    metadata_index[feature_id] = FeatureMetadata.from_dict(
                        metadata_dict
                    )
            except Exception as e:
                logger.error(f"Joblib形式メタデータ読み込みエラー: {e}")
                raise

        return metadata_index

    def save(self, metadata_index: Dict[str, FeatureMetadata]):
        """Joblib形式でのメタデータ保存"""
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is not available")

        index_file = self.metadata_path / "index.joblib"

        try:
            index_data = {
                feature_id: metadata.to_dict()
                for feature_id, metadata in metadata_index.items()
            }

            joblib.dump(index_data, index_file, compress=3)  # 圧縮レベル3

        except Exception as e:
            logger.error(f"Joblib形式メタデータ保存エラー: {e}")
            raise


class SqliteMetadataIndex(MetadataIndex):
    """
    SQLite形式のメタデータインデックス
    
    大規模なメタデータとクエリ性能に最適化されています。
    """

    def __init__(self, metadata_path: Path, config: FeatureStoreConfig):
        """
        初期化
        
        Args:
            metadata_path: メタデータ保存パス
            config: 特徴量ストア設定
        """
        super().__init__(metadata_path, config)
        self._sqlite_db_path = self.metadata_path / "index.db"

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

    def load(self) -> Dict[str, FeatureMetadata]:
        """SQLite形式からのメタデータ読み込み"""
        metadata_index = {}

        if not self._sqlite_db_path.exists():
            logger.info("新しいSQLiteメタデータインデックスを作成")
            return metadata_index

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

                    metadata_index[feature_id] = metadata

        except Exception as e:
            logger.error(f"SQLite形式メタデータ読み込みエラー: {e}")
            raise

        return metadata_index

    def save(self, metadata_index: Dict[str, FeatureMetadata]):
        """SQLite形式でのメタデータ保存"""
        try:
            with self._get_sqlite_connection() as conn:
                self._create_metadata_table(conn)

                # 全データをクリアして再挿入（簡易実装）
                conn.execute("DELETE FROM metadata")

                # バッチ挿入でパフォーマンス向上
                batch_data = []
                for feature_id, metadata in metadata_index.items():
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

    def vacuum(self):
        """データベース最適化（VACUUM実行）"""
        try:
            with self._get_sqlite_connection() as conn:
                conn.execute("VACUUM")  # データベース最適化
            logger.debug("SQLiteデータベース最適化完了")
        except Exception as e:
            logger.warning(f"SQLiteデータベース最適化エラー: {e}")


class MetadataManager:
    """
    メタデータマネージャー
    
    適切なメタデータインデックスを選択し、メタデータの管理を行います。
    """

    def __init__(self, metadata_path: Path, config: FeatureStoreConfig):
        """
        初期化
        
        Args:
            metadata_path: メタデータ保存パス
            config: 特徴量ストア設定
        """
        self.metadata_path = metadata_path
        self.config = config
        self._current_index: Optional[MetadataIndex] = None

    def _estimate_metadata_count(self) -> int:
        """メタデータ件数の推定（形式選択用）"""
        # 既存ファイルからの推定
        json_file = self.metadata_path / "index.json"
        sqlite_file = self.metadata_path / "index.db"
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

        if joblib_file.exists() and JOBLIB_AVAILABLE:
            try:
                data = joblib.load(joblib_file)
                return len(data)
            except Exception:
                pass

        return 0

    def _get_index_format(self) -> str:
        """使用するインデックス形式を決定"""
        if self.config.metadata_index_format == "auto":
            # メタデータ件数推定に基づいて形式を決定
            estimated_count = self._estimate_metadata_count()
            if estimated_count >= self.config.metadata_index_threshold:
                return "sqlite"
            else:
                return "json"
        else:
            return self.config.metadata_index_format

    def _get_index(self) -> MetadataIndex:
        """適切なメタデータインデックスを取得"""
        if self._current_index is not None:
            return self._current_index

        format_to_use = self._get_index_format()

        if format_to_use == "sqlite":
            self._current_index = SqliteMetadataIndex(self.metadata_path, self.config)
        elif format_to_use == "pickle":
            self._current_index = PickleMetadataIndex(self.metadata_path, self.config)
        elif format_to_use == "joblib" and JOBLIB_AVAILABLE:
            self._current_index = JoblibMetadataIndex(self.metadata_path, self.config)
        else:
            self._current_index = JsonMetadataIndex(self.metadata_path, self.config)

        return self._current_index

    def load_metadata_index(self) -> Dict[str, FeatureMetadata]:
        """メタデータインデックスの読み込み"""
        format_to_use = self._get_index_format()
        logger.debug(f"メタデータインデックス読み込み形式: {format_to_use}")

        try:
            index = self._get_index()
            metadata_index = index.load()

            logger.info(
                f"メタデータインデックス読み込み完了: {len(metadata_index)}件 (形式: {format_to_use})"
            )
            return metadata_index
        except Exception as e:
            logger.error(f"メタデータインデックス読み込みエラー ({format_to_use}): {e}")
            return {}

    def save_metadata_index(self, metadata_index: Dict[str, FeatureMetadata]):
        """メタデータインデックスの保存"""
        format_to_use = self._get_index_format()

        try:
            index = self._get_index()
            index.save(metadata_index)

            logger.debug(
                f"メタデータインデックス保存完了: {len(metadata_index)}件 (形式: {format_to_use})"
            )
        except Exception as e:
            logger.error(f"メタデータインデックス保存エラー ({format_to_use}): {e}")

    def vacuum_database(self):
        """データベース最適化（SQLite使用時）"""
        if isinstance(self._current_index, SqliteMetadataIndex):
            self._current_index.vacuum()


# datetimeのインポートを追加
from datetime import datetime