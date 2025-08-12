#!/usr/bin/env python3
"""
エンハンスドデータバージョンコントロールシステム
Issue #420: データ管理とデータ品質保証メカニズムの強化

DVC（Data Version Control）の本格実装:
- Gitライクなバージョンワークフロー
- 分散データ管理
- ML実験追跡統合
- 自動品質チェック統合
- クラウドストレージ対応
"""

import asyncio
import gzip
import hashlib
import json
import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    from ..monitoring.structured_logging_enhancement import (
        StructuredLoggingEnhancementSystem,
    )
    from ..utils.unified_cache_manager import UnifiedCacheManager
    from .comprehensive_data_quality_system import (
        ComprehensiveDataQualitySystem,
        DataQualityReport,
    )
    from .data_version_manager import DataStatus, DataVersion, VersionOperation

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

    # Fallback definitions
    class DataVersion:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class DataStatus(Enum):
        ACTIVE = "active"
        DEPRECATED = "deprecated"
        ARCHIVED = "archived"

    class VersionOperation(Enum):
        CREATE = "create"
        UPDATE = "update"


class DVCStorageBackend(Enum):
    """DVC ストレージバックエンド"""

    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    HDFS = "hdfs"


class ExperimentStatus(Enum):
    """実験ステータス"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DVCConfig:
    """DVC設定"""

    repository_path: str = "data_versions"
    storage_backend: DVCStorageBackend = DVCStorageBackend.LOCAL
    remote_url: Optional[str] = None
    compression_enabled: bool = True
    encryption_enabled: bool = False
    max_versions_per_dataset: int = 100
    auto_cleanup_enabled: bool = True
    quality_check_enabled: bool = True
    backup_retention_days: int = 90


@dataclass
class MLExperiment:
    """ML実験情報"""

    experiment_id: str
    name: str
    model_type: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    datasets: List[str]  # データセットバージョンID
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DataLineage:
    """データ系譜情報"""

    dataset_id: str
    version_id: str
    parent_datasets: List[str]
    transformation_code: Optional[str] = None
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    quality_report_id: Optional[str] = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnhancedDataVersionControl:
    """エンハンスドデータバージョンコントロール"""

    def __init__(self, config: DVCConfig = None):
        self.config = config or DVCConfig()
        self.logger = logging.getLogger(__name__)

        # ディレクトリ初期化
        self.repo_path = Path(self.config.repository_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        (self.repo_path / "objects").mkdir(exist_ok=True)
        (self.repo_path / "refs").mkdir(exist_ok=True)
        (self.repo_path / "experiments").mkdir(exist_ok=True)
        (self.repo_path / "metadata").mkdir(exist_ok=True)
        (self.repo_path / "cache").mkdir(exist_ok=True)

        # データベース初期化
        self.db_path = self.repo_path / "dvc.db"
        self._initialize_database()

        # コンポーネント初期化
        self.cache_manager = UnifiedCacheManager() if DEPENDENCIES_AVAILABLE else None
        self.quality_system = None

        if self.config.quality_check_enabled:
            try:
                self.quality_system = ComprehensiveDataQualitySystem()
            except Exception as e:
                self.logger.warning(f"品質システム初期化失敗: {e}")

        # 統計
        self.stats = {
            "versions_created": 0,
            "experiments_tracked": 0,
            "data_quality_checks": 0,
            "storage_usage_bytes": 0,
        }

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # バージョン管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_versions (
                    version_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    parent_version TEXT,
                    branch TEXT NOT NULL DEFAULT 'main',
                    tag TEXT,
                    author TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    message TEXT,
                    data_hash TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    file_count INTEGER DEFAULT 0,
                    quality_score REAL,
                    storage_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # 実験追跡テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT,
                    parameters TEXT,
                    metrics TEXT,
                    datasets TEXT,
                    status TEXT NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    artifacts TEXT,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # データ系譜テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    parent_datasets TEXT,
                    transformation_code TEXT,
                    transformation_params TEXT,
                    quality_report_id TEXT,
                    created_by TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES data_versions(version_id)
                )
            """
            )

            # ブランチ管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS branches (
                    branch_name TEXT PRIMARY KEY,
                    head_version TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    description TEXT,
                    FOREIGN KEY (head_version) REFERENCES data_versions(version_id)
                )
            """
            )

            # タグテーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    tag_name TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    description TEXT,
                    FOREIGN KEY (version_id) REFERENCES data_versions(version_id)
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_dataset ON data_versions(dataset_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_branch ON data_versions(branch, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_status ON ml_experiments(status, start_time)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON data_lineage(dataset_id, version_id)"
            )

            # デフォルトブランチ作成
            conn.execute(
                """
                INSERT OR IGNORE INTO branches (branch_name, head_version, created_by, created_at, description)
                VALUES ('main', 'initial', 'system', ?, 'Default main branch')
            """,
                (datetime.now(timezone.utc).isoformat(),),
            )

            conn.commit()

    def _calculate_data_hash(self, data: Union[pd.DataFrame, bytes, str]) -> str:
        """データハッシュ計算"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame用ハッシュ
                content = data.to_json(sort_keys=True).encode()
            elif isinstance(data, str):
                content = data.encode()
            elif isinstance(data, bytes):
                content = data
            else:
                content = str(data).encode()

            return hashlib.sha256(content).hexdigest()

        except Exception as e:
            self.logger.error(f"ハッシュ計算エラー: {e}")
            return hashlib.sha256(str(time.time()).encode()).hexdigest()

    async def create_version(
        self,
        dataset_id: str,
        data: Union[pd.DataFrame, bytes, str, Path],
        branch: str = "main",
        message: str = "",
        author: str = "system",
        metadata: Dict[str, Any] = None,
        parent_datasets: List[str] = None,
    ) -> str:
        """データバージョン作成"""
        try:
            metadata = metadata or {}
            parent_datasets = parent_datasets or []

            # データの正規化
            if isinstance(data, (str, Path)):
                data_path = Path(data)
                if data_path.exists():
                    if data_path.suffix.lower() == ".csv":
                        data = pd.read_csv(data_path)
                    elif data_path.suffix.lower() in [".pkl", ".pickle"]:
                        with open(data_path, "rb") as f:
                            data = pickle.load(f)
                    else:
                        with open(data_path, "rb") as f:
                            data = f.read()
                else:
                    raise FileNotFoundError(f"ファイルが見つかりません: {data}")

            # データハッシュ計算
            data_hash = self._calculate_data_hash(data)

            # 既存バージョンチェック
            existing_version = await self._find_version_by_hash(dataset_id, data_hash)
            if existing_version:
                self.logger.info(f"既存バージョンが見つかりました: {existing_version}")
                return existing_version

            # 新バージョンID生成
            timestamp = datetime.now(timezone.utc)
            version_id = f"{dataset_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{data_hash[:8]}"

            # 親バージョン取得
            parent_version = await self._get_branch_head(branch)

            # データ保存
            storage_path = await self._store_data(version_id, data)

            # 品質チェック実行
            quality_score = None
            quality_report_id = None

            if (
                self.config.quality_check_enabled
                and self.quality_system
                and isinstance(data, pd.DataFrame)
            ):
                try:
                    quality_report = await self.quality_system.process_dataset(
                        data, dataset_id, metadata=metadata
                    )
                    quality_score = quality_report.overall_score
                    quality_report_id = f"qr_{version_id}"

                    self.stats["data_quality_checks"] += 1

                except Exception as e:
                    self.logger.warning(f"品質チェックエラー: {e}")

            # データサイズ計算
            if isinstance(data, pd.DataFrame):
                size_bytes = data.memory_usage(deep=True).sum()
                file_count = 1
            elif isinstance(data, bytes):
                size_bytes = len(data)
                file_count = 1
            else:
                size_bytes = len(str(data).encode())
                file_count = 1

            # バージョン情報保存
            version = DataVersion(
                version_id=version_id,
                parent_version=parent_version,
                branch=branch,
                author=author,
                timestamp=timestamp,
                message=message,
                data_hash=data_hash,
                metadata=metadata,
                status=DataStatus.ACTIVE,
                size_bytes=size_bytes,
                file_count=file_count,
            )

            await self._save_version(version, quality_score, storage_path)

            # ブランチ更新
            await self._update_branch_head(branch, version_id)

            # データ系譜記録
            if parent_datasets:
                lineage = DataLineage(
                    dataset_id=dataset_id,
                    version_id=version_id,
                    parent_datasets=parent_datasets,
                    transformation_params=metadata,
                    quality_report_id=quality_report_id,
                    created_by=author,
                )
                await self._save_lineage(lineage)

            # 統計更新
            self.stats["versions_created"] += 1
            self.stats["storage_usage_bytes"] += size_bytes

            self.logger.info(f"バージョン作成完了: {version_id}")
            return version_id

        except Exception as e:
            self.logger.error(f"バージョン作成エラー: {e}")
            raise

    async def _find_version_by_hash(self, dataset_id: str, data_hash: str) -> Optional[str]:
        """ハッシュによるバージョン検索"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id FROM data_versions
                    WHERE dataset_id = ? AND data_hash = ? AND status != 'deleted'
                    ORDER BY timestamp DESC LIMIT 1
                """,
                    (dataset_id, data_hash),
                )

                result = cursor.fetchone()
                return result[0] if result else None

        except Exception as e:
            self.logger.error(f"バージョン検索エラー: {e}")
            return None

    async def _get_branch_head(self, branch: str) -> Optional[str]:
        """ブランチのHEAD取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT head_version FROM branches WHERE branch_name = ?
                """,
                    (branch,),
                )

                result = cursor.fetchone()
                if result and result[0] != "initial":
                    return result[0]
                else:
                    return None

        except Exception as e:
            self.logger.error(f"ブランチHEAD取得エラー: {e}")
            return None

    async def _store_data(self, version_id: str, data: Any) -> str:
        """データ保存"""
        try:
            # オブジェクト保存パス
            storage_path = self.repo_path / "objects" / f"{version_id}.pkl"

            # データ圧縮・保存
            if self.config.compression_enabled:
                with gzip.open(storage_path, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(storage_path, "wb") as f:
                    pickle.dump(data, f)

            return str(storage_path.relative_to(self.repo_path))

        except Exception as e:
            self.logger.error(f"データ保存エラー: {e}")
            raise

    async def _save_version(
        self, version: DataVersion, quality_score: Optional[float], storage_path: str
    ):
        """バージョン情報保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_versions
                    (version_id, dataset_id, parent_version, branch, tag, author, timestamp,
                     message, data_hash, metadata, status, size_bytes, file_count,
                     quality_score, storage_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        version.version_id,
                        version.version_id.split("_")[0],  # dataset_id extraction
                        version.parent_version,
                        version.branch,
                        version.tag,
                        version.author,
                        version.timestamp.isoformat(),
                        version.message,
                        version.data_hash,
                        json.dumps(version.metadata, ensure_ascii=False),
                        version.status.value,
                        version.size_bytes,
                        version.file_count,
                        quality_score,
                        storage_path,
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"バージョン保存エラー: {e}")
            raise

    async def _update_branch_head(self, branch: str, version_id: str):
        """ブランチHEAD更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # ブランチ存在確認
                cursor = conn.execute("SELECT 1 FROM branches WHERE branch_name = ?", (branch,))
                if not cursor.fetchone():
                    # 新ブランチ作成
                    conn.execute(
                        """
                        INSERT INTO branches (branch_name, head_version, created_by, created_at, description)
                        VALUES (?, ?, 'system', ?, ?)
                    """,
                        (
                            branch,
                            version_id,
                            datetime.now(timezone.utc).isoformat(),
                            f"Auto-created branch: {branch}",
                        ),
                    )
                else:
                    # 既存ブランチ更新
                    conn.execute(
                        """
                        UPDATE branches SET head_version = ? WHERE branch_name = ?
                    """,
                        (version_id, branch),
                    )

                conn.commit()

        except Exception as e:
            self.logger.error(f"ブランチ更新エラー: {e}")

    async def _save_lineage(self, lineage: DataLineage):
        """データ系譜保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO data_lineage
                    (dataset_id, version_id, parent_datasets, transformation_code,
                     transformation_params, quality_report_id, created_by, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        lineage.dataset_id,
                        lineage.version_id,
                        json.dumps(lineage.parent_datasets, ensure_ascii=False),
                        lineage.transformation_code,
                        json.dumps(lineage.transformation_params, ensure_ascii=False),
                        lineage.quality_report_id,
                        lineage.created_by,
                        lineage.created_at.isoformat(),
                    ),
                )
                conn.commit()

        except Exception as e:
            self.logger.error(f"系譜保存エラー: {e}")

    async def get_version(self, version_id: str) -> Optional[Any]:
        """バージョンデータ取得"""
        try:
            # バージョン情報取得
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT storage_path FROM data_versions WHERE version_id = ?
                """,
                    (version_id,),
                )

                result = cursor.fetchone()
                if not result:
                    return None

                storage_path = self.repo_path / result[0]

            # データ読み込み
            if self.config.compression_enabled:
                with gzip.open(storage_path, "rb") as f:
                    return pickle.load(f)
            else:
                with open(storage_path, "rb") as f:
                    return pickle.load(f)

        except Exception as e:
            self.logger.error(f"バージョン取得エラー: {e}")
            return None

    async def create_branch(
        self, branch_name: str, from_version: str = None, author: str = "system"
    ) -> bool:
        """ブランチ作成"""
        try:
            # 開始バージョン決定
            if not from_version:
                from_version = await self._get_branch_head("main")

            if not from_version:
                raise ValueError("開始バージョンが見つかりません")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO branches (branch_name, head_version, created_by, created_at, description)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        branch_name,
                        from_version,
                        author,
                        datetime.now(timezone.utc).isoformat(),
                        f"Branch created from {from_version}",
                    ),
                )
                conn.commit()

            self.logger.info(f"ブランチ作成: {branch_name}")
            return True

        except Exception as e:
            self.logger.error(f"ブランチ作成エラー: {e}")
            return False

    async def create_tag(
        self,
        tag_name: str,
        version_id: str,
        description: str = "",
        author: str = "system",
    ) -> bool:
        """タグ作成"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # タグ作成
                conn.execute(
                    """
                    INSERT INTO tags (tag_name, version_id, created_by, created_at, description)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        tag_name,
                        version_id,
                        author,
                        datetime.now(timezone.utc).isoformat(),
                        description,
                    ),
                )

                # バージョンにタグ設定
                conn.execute(
                    """
                    UPDATE data_versions SET tag = ? WHERE version_id = ?
                """,
                    (tag_name, version_id),
                )

                conn.commit()

            self.logger.info(f"タグ作成: {tag_name} -> {version_id}")
            return True

        except Exception as e:
            self.logger.error(f"タグ作成エラー: {e}")
            return False

    async def track_experiment(
        self,
        experiment_name: str,
        model_type: str,
        parameters: Dict[str, Any],
        dataset_versions: List[str],
        author: str = "system",
    ) -> str:
        """ML実験追跡開始"""
        try:
            experiment_id = f"exp_{int(time.time())}_{hash(experiment_name) % 10000:04d}"

            experiment = MLExperiment(
                experiment_id=experiment_id,
                name=experiment_name,
                model_type=model_type,
                parameters=parameters,
                metrics={},
                datasets=dataset_versions,
                status=ExperimentStatus.RUNNING,
                start_time=datetime.now(timezone.utc),
            )

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO ml_experiments
                    (experiment_id, name, model_type, parameters, metrics, datasets,
                     status, start_time, artifacts, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        experiment.experiment_id,
                        experiment.name,
                        experiment.model_type,
                        json.dumps(experiment.parameters, ensure_ascii=False),
                        json.dumps(experiment.metrics, ensure_ascii=False),
                        json.dumps(experiment.datasets, ensure_ascii=False),
                        experiment.status.value,
                        experiment.start_time.isoformat(),
                        json.dumps(experiment.artifacts, ensure_ascii=False),
                        experiment.notes,
                    ),
                )
                conn.commit()

            self.stats["experiments_tracked"] += 1
            self.logger.info(f"実験追跡開始: {experiment_id}")

            return experiment_id

        except Exception as e:
            self.logger.error(f"実験追跡開始エラー: {e}")
            raise

    async def complete_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        artifacts: List[str] = None,
        notes: str = "",
    ) -> bool:
        """ML実験完了"""
        try:
            artifacts = artifacts or []

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE ml_experiments
                    SET metrics = ?, status = ?, end_time = ?, artifacts = ?, notes = ?
                    WHERE experiment_id = ?
                """,
                    (
                        json.dumps(metrics, ensure_ascii=False),
                        status.value,
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(artifacts, ensure_ascii=False),
                        notes,
                        experiment_id,
                    ),
                )
                conn.commit()

            self.logger.info(f"実験完了: {experiment_id}")
            return True

        except Exception as e:
            self.logger.error(f"実験完了エラー: {e}")
            return False

    async def get_version_history(
        self, dataset_id: str, branch: str = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """バージョン履歴取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if branch:
                    cursor = conn.execute(
                        """
                        SELECT * FROM data_versions
                        WHERE dataset_id = ? AND branch = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """,
                        (dataset_id, branch, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM data_versions
                        WHERE dataset_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """,
                        (dataset_id, limit),
                    )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"履歴取得エラー: {e}")
            return []

    async def get_data_lineage(self, version_id: str) -> Dict[str, Any]:
        """データ系譜取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 直接の系譜
                cursor = conn.execute(
                    """
                    SELECT * FROM data_lineage WHERE version_id = ?
                """,
                    (version_id,),
                )

                lineage_data = cursor.fetchone()
                if not lineage_data:
                    return {"version_id": version_id, "lineage": None}

                columns = [desc[0] for desc in cursor.description]
                lineage = dict(zip(columns, lineage_data))

                # 親データセットの詳細取得
                if lineage.get("parent_datasets"):
                    parent_ids = json.loads(lineage["parent_datasets"])
                    parent_details = []

                    for parent_id in parent_ids:
                        cursor = conn.execute(
                            """
                            SELECT version_id, dataset_id, timestamp, quality_score
                            FROM data_versions WHERE version_id = ?
                        """,
                            (parent_id,),
                        )

                        parent_info = cursor.fetchone()
                        if parent_info:
                            parent_details.append(
                                {
                                    "version_id": parent_info[0],
                                    "dataset_id": parent_info[1],
                                    "timestamp": parent_info[2],
                                    "quality_score": parent_info[3],
                                }
                            )

                    lineage["parent_details"] = parent_details

                return {"version_id": version_id, "lineage": lineage}

        except Exception as e:
            self.logger.error(f"データ系譜取得エラー: {e}")
            return {"version_id": version_id, "error": str(e)}

    async def get_system_stats(self) -> Dict[str, Any]:
        """システム統計取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # バージョン統計
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM data_versions")
                version_count, total_size = cursor.fetchone()

                # ブランチ統計
                cursor = conn.execute("SELECT COUNT(*) FROM branches")
                branch_count = cursor.fetchone()[0]

                # 実験統計
                cursor = conn.execute("SELECT status, COUNT(*) FROM ml_experiments GROUP BY status")
                experiment_stats = {row[0]: row[1] for row in cursor.fetchall()}

                # 品質統計
                cursor = conn.execute(
                    """
                    SELECT AVG(quality_score), MIN(quality_score), MAX(quality_score)
                    FROM data_versions WHERE quality_score IS NOT NULL
                """
                )
                quality_stats = cursor.fetchone()

                return {
                    "versions": {
                        "total_count": version_count or 0,
                        "total_size_bytes": total_size or 0,
                        "total_size_mb": round((total_size or 0) / 1024 / 1024, 2),
                    },
                    "branches": {"total_count": branch_count or 0},
                    "experiments": experiment_stats,
                    "quality": {
                        "average_score": round(quality_stats[0], 2) if quality_stats[0] else None,
                        "min_score": quality_stats[1],
                        "max_score": quality_stats[2],
                    },
                    "system_stats": self.stats,
                    "config": {
                        "storage_backend": self.config.storage_backend.value,
                        "compression_enabled": self.config.compression_enabled,
                        "quality_check_enabled": self.config.quality_check_enabled,
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"システム統計取得エラー: {e}")
            return {"error": str(e)}


# Factory function
def create_enhanced_dvc_system(config: DVCConfig = None) -> EnhancedDataVersionControl:
    """エンハンスドDVCシステム作成"""
    return EnhancedDataVersionControl(config)


# Global instance
_enhanced_dvc_system = None


def get_enhanced_dvc_system() -> EnhancedDataVersionControl:
    """グローバルエンハンスドDVCシステム取得"""
    global _enhanced_dvc_system
    if _enhanced_dvc_system is None:
        _enhanced_dvc_system = create_enhanced_dvc_system()
    return _enhanced_dvc_system


if __name__ == "__main__":
    # テスト実行
    async def test_enhanced_dvc_system():
        print("=== エンハンスドデータバージョンコントロールシステムテスト ===")

        try:
            # システム初期化
            config = DVCConfig(
                repository_path="test_dvc_repo",
                compression_enabled=True,
                quality_check_enabled=True,
                max_versions_per_dataset=10,
            )

            dvc_system = create_enhanced_dvc_system(config)

            print("\n1. DVCシステム初期化完了")
            print(f"   リポジトリパス: {config.repository_path}")
            print(f"   品質チェック: {config.quality_check_enabled}")
            print(f"   圧縮: {config.compression_enabled}")

            # テストデータ作成
            test_data_1 = pd.DataFrame(
                {
                    "symbol": ["AAPL", "GOOGL", "MSFT", "TSLA"],
                    "price": [150.0, 2800.0, 300.0, 800.0],
                    "volume": [1000, 500, 800, 1200],
                    "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
                }
            )

            test_data_2 = pd.DataFrame(
                {
                    "symbol": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
                    "price": [155.0, 2850.0, 305.0, 820.0, 3200.0],
                    "volume": [1100, 550, 850, 1300, 600],
                    "date": [
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-02",
                    ],
                }
            )

            print("\n2. テストデータ作成完了")

            # 初期バージョン作成
            print("\n3. データバージョン作成...")

            version_1 = await dvc_system.create_version(
                dataset_id="stock_prices",
                data=test_data_1,
                message="初期株価データセット",
                author="test_user",
                metadata={"data_source": "test", "version_type": "initial"},
            )

            print(f"   バージョン1作成: {version_1}")

            # 更新バージョン作成
            version_2 = await dvc_system.create_version(
                dataset_id="stock_prices",
                data=test_data_2,
                message="株価データ更新（AMZN追加）",
                author="test_user",
                metadata={"data_source": "test", "version_type": "update"},
                parent_datasets=[version_1],
            )

            print(f"   バージョン2作成: {version_2}")

            # ブランチ作成テスト
            print("\n4. ブランチ作成テスト...")

            branch_created = await dvc_system.create_branch(
                "feature/add_indicators", from_version=version_1, author="test_user"
            )

            print(f"   ブランチ作成: {'成功' if branch_created else '失敗'}")

            # タグ作成テスト
            print("\n5. タグ作成テスト...")

            tag_created = await dvc_system.create_tag(
                "v1.0.0", version_2, "安定版リリース", "test_user"
            )

            print(f"   タグ作成: {'成功' if tag_created else '失敗'}")

            # ML実験追跡テスト
            print("\n6. ML実験追跡テスト...")

            experiment_id = await dvc_system.track_experiment(
                "LSTM株価予測実験",
                "LSTM",
                {"hidden_size": 128, "learning_rate": 0.001, "epochs": 100},
                [version_1, version_2],
                "test_user",
            )

            print(f"   実験開始: {experiment_id}")

            # 実験完了
            experiment_completed = await dvc_system.complete_experiment(
                experiment_id,
                {"accuracy": 0.85, "loss": 0.23, "val_accuracy": 0.82},
                ExperimentStatus.COMPLETED,
                artifacts=["model.pkl", "predictions.csv"],
                notes="初回実験完了",
            )

            print(f"   実験完了: {'成功' if experiment_completed else '失敗'}")

            # データ取得テスト
            print("\n7. データ取得テスト...")

            retrieved_data = await dvc_system.get_version(version_2)
            if retrieved_data is not None:
                print(f"   データ取得成功: {len(retrieved_data)}行")
                print(f"   取得データ列: {list(retrieved_data.columns)}")

            # 履歴取得テスト
            print("\n8. バージョン履歴取得...")

            history = await dvc_system.get_version_history("stock_prices")
            print(f"   履歴件数: {len(history)}")

            for i, version in enumerate(history[:3]):
                print(f"   {i+1}. {version['version_id']} ({version['timestamp']})")
                if version["quality_score"]:
                    print(f"      品質スコア: {version['quality_score']:.2f}")

            # データ系譜取得テスト
            print("\n9. データ系譜取得...")

            lineage = await dvc_system.get_data_lineage(version_2)
            if lineage.get("lineage"):
                lineage_info = lineage["lineage"]
                print("   系譜データ取得成功")
                print(
                    f"   親データセット数: {len(json.loads(lineage_info.get('parent_datasets', '[]')))}"
                )
                if lineage_info.get("parent_details"):
                    print(f"   親データセット詳細: {len(lineage_info['parent_details'])}件")

            # システム統計取得
            print("\n10. システム統計...")

            stats = await dvc_system.get_system_stats()
            print(f"   総バージョン数: {stats['versions']['total_count']}")
            print(f"   総データサイズ: {stats['versions']['total_size_mb']} MB")
            print(f"   ブランチ数: {stats['branches']['total_count']}")
            print(f"   実験数: {sum(stats['experiments'].values())}")

            if stats["quality"]["average_score"]:
                print(f"   平均品質スコア: {stats['quality']['average_score']}")

            print("\n[成功] エンハンスドデータバージョンコントロールシステムテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_enhanced_dvc_system())
