#!/usr/bin/env python3
"""
データバージョン管理システム（DVC）
Issue #420: データ管理とデータ品質保証メカニズムの強化

MLモデルの再現性とデータ系譜管理のための包括的なバージョン管理システム:
- データセットバージョニング
- メタデータ管理
- 変更履歴追跡
- ブランチング・マージ
- データリネージュ
- スナップショット管理
- 自動バックアップ
"""

import asyncio
import hashlib
import json
import logging
import shutil
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    from ..utils.logging_config import get_context_logger
    from ..utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class UnifiedCacheManager:
        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"version_key_{hash(str(args))}"


logger = get_context_logger(__name__)

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class VersionOperation(Enum):
    """バージョン操作種別"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    BRANCH = "branch"
    TAG = "tag"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"


class DataStatus(Enum):
    """データステータス"""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"
    APPROVED = "approved"


class ConflictResolution(Enum):
    """競合解決方式"""

    MANUAL = "manual"
    AUTO_LATEST = "auto_latest"
    AUTO_MERGE = "auto_merge"
    KEEP_BOTH = "keep_both"


@dataclass
class DataVersion:
    """データバージョン情報"""

    version_id: str
    parent_version: Optional[str] = None
    branch: str = "main"
    tag: Optional[str] = None
    author: str = "system"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""
    data_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DataStatus = DataStatus.ACTIVE
    size_bytes: int = 0
    file_count: int = 0
    checksum: str = ""


@dataclass
class DataBranch:
    """データブランチ情報"""

    branch_name: str
    created_from: str  # parent version_id
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    is_protected: bool = False
    latest_version: Optional[str] = None
    merge_policy: ConflictResolution = ConflictResolution.MANUAL


@dataclass
class DataTag:
    """データタグ情報"""

    tag_name: str
    version_id: str
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    is_release: bool = False


@dataclass
class VersionConflict:
    """バージョン競合情報"""

    conflict_id: str
    source_version: str
    target_version: str
    conflict_type: str  # "file", "metadata", "schema"
    file_path: Optional[str] = None
    description: str = ""
    resolution_strategy: ConflictResolution = ConflictResolution.MANUAL
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class DataVersionManager:
    """データバージョン管理システム"""

    def __init__(
        self,
        repository_path: str = "data/versions",
        enable_cache: bool = True,
        max_versions: int = 100,
        auto_backup: bool = True,
    ):
        self.repository_path = Path(repository_path)
        self.enable_cache = enable_cache
        self.max_versions = max_versions
        self.auto_backup = auto_backup

        # ディレクトリ構造初期化
        self._initialize_repository()

        # データベース初期化
        self.db_path = self.repository_path / "versions.db"
        self._initialize_database()

        # キャッシュマネージャー初期化
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=64, l2_memory_mb=256, l3_disk_mb=1024
                )
                logger.info("バージョン管理キャッシュシステム初期化完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
        else:
            self.cache_manager = None

        # 内部管理状態
        self.current_branch = "main"
        self.version_cache: Dict[str, DataVersion] = {}
        self.branch_cache: Dict[str, DataBranch] = {}
        self.conflict_cache: List[VersionConflict] = []

        logger.info("データバージョン管理システム初期化完了")
        logger.info(f"  - リポジトリパス: {self.repository_path}")
        logger.info(f"  - キャッシュ: {'有効' if enable_cache else '無効'}")
        logger.info(f"  - 最大バージョン数: {max_versions}")
        logger.info(f"  - 自動バックアップ: {'有効' if auto_backup else '無効'}")

    def _initialize_repository(self):
        """リポジトリディレクトリ初期化"""
        self.repository_path.mkdir(parents=True, exist_ok=True)

        # サブディレクトリ作成
        (self.repository_path / "data").mkdir(exist_ok=True)
        (self.repository_path / "metadata").mkdir(exist_ok=True)
        (self.repository_path / "snapshots").mkdir(exist_ok=True)
        (self.repository_path / "backups").mkdir(exist_ok=True)
        (self.repository_path / "temp").mkdir(exist_ok=True)

        # .dvconfig 設定ファイル作成
        config_file = self.repository_path / ".dvconfig"
        if not config_file.exists():
            config = {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "default_branch": "main",
                "auto_backup": self.auto_backup,
                "max_versions": self.max_versions,
            }
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # バージョン管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    parent_version TEXT,
                    branch TEXT NOT NULL,
                    tag TEXT,
                    author TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    message TEXT,
                    data_hash TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    file_count INTEGER DEFAULT 0,
                    checksum TEXT
                )
            """
            )

            # ブランチ管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS branches (
                    branch_name TEXT PRIMARY KEY,
                    created_from TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    description TEXT,
                    is_protected INTEGER DEFAULT 0,
                    latest_version TEXT,
                    merge_policy TEXT
                )
            """
            )

            # タグ管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    tag_name TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    description TEXT,
                    is_release INTEGER DEFAULT 0
                )
            """
            )

            # 競合管理テーブル
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    source_version TEXT NOT NULL,
                    target_version TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    file_path TEXT,
                    description TEXT,
                    resolution_strategy TEXT,
                    resolved INTEGER DEFAULT 0,
                    resolved_by TEXT,
                    resolved_at TEXT
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_branch ON versions(branch)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON versions(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_versions_status ON versions(status)"
            )

            # デフォルトブランチ作成
            self._ensure_default_branch(conn)

    def _ensure_default_branch(self, conn: sqlite3.Connection):
        """デフォルトブランチ確保"""
        cursor = conn.execute(
            "SELECT COUNT(*) FROM branches WHERE branch_name = ?", ("main",)
        )
        if cursor.fetchone()[0] == 0:
            main_branch = DataBranch(
                branch_name="main",
                created_from="root",
                created_by="system",
                description="メインブランチ",
                is_protected=True,
            )

            conn.execute(
                """
                INSERT INTO branches
                (branch_name, created_from, created_by, created_at, description, is_protected, merge_policy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    main_branch.branch_name,
                    main_branch.created_from,
                    main_branch.created_by,
                    main_branch.created_at.isoformat(),
                    main_branch.description,
                    1 if main_branch.is_protected else 0,
                    main_branch.merge_policy.value,
                ),
            )

            logger.info("デフォルトmainブランチ作成完了")

    async def commit_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
        message: str,
        author: str = "system",
        branch: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """データコミット"""
        start_time = time.time()
        branch = branch or self.current_branch
        metadata = metadata or {}

        logger.info(f"データコミット開始: {branch} - {message}")

        try:
            # データハッシュ計算
            data_hash = self._calculate_data_hash(data)

            # 重複チェック
            existing_version = await self._find_version_by_hash(data_hash, branch)
            if existing_version:
                logger.info(f"同一データが既存: {existing_version.version_id}")
                return existing_version.version_id

            # バージョンID生成
            version_id = f"v_{int(time.time())}_{data_hash[:8]}"

            # 親バージョン取得
            parent_version = await self._get_latest_version(branch)
            parent_version_id = parent_version.version_id if parent_version else None

            # データ保存
            data_path = await self._save_version_data(version_id, data)

            # メタデータ保存
            full_metadata = {
                **metadata,
                "commit_time": datetime.utcnow().isoformat(),
                "data_path": str(data_path),
                "data_type": type(data).__name__,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

            if isinstance(data, pd.DataFrame):
                full_metadata.update(
                    {
                        "shape": data.shape,
                        "columns": list(data.columns),
                        "dtypes": data.dtypes.to_dict(),
                        "memory_usage": data.memory_usage(deep=True).sum(),
                    }
                )

            # バージョン情報作成
            version = DataVersion(
                version_id=version_id,
                parent_version=parent_version_id,
                branch=branch,
                author=author,
                timestamp=datetime.utcnow(),
                message=message,
                data_hash=data_hash,
                metadata=full_metadata,
                status=DataStatus.ACTIVE,
                size_bytes=self._get_data_size(data),
                file_count=1,
                checksum=self._calculate_checksum(data),
            )

            # データベース保存
            await self._save_version_to_db(version)

            # ブランチの最新バージョン更新
            await self._update_branch_latest_version(branch, version_id)

            # キャッシュ更新
            if self.cache_manager:
                cache_key = generate_unified_cache_key("version", version_id)
                self.cache_manager.put(cache_key, version, priority=5.0)

            self.version_cache[version_id] = version

            # バージョン数制限チェック
            await self._cleanup_old_versions(branch)

            # 自動バックアップ
            if self.auto_backup and len(await self._list_versions(branch)) % 10 == 0:
                await self._create_automatic_backup(version_id)

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"データコミット完了: {version_id} ({duration_ms:.1f}ms)")

            return version_id

        except Exception as e:
            logger.error(f"データコミットエラー: {e}")
            raise

    async def checkout_data(self, version_id: str) -> Tuple[Any, DataVersion]:
        """データチェックアウト"""
        logger.info(f"データチェックアウト: {version_id}")

        try:
            # バージョン情報取得
            version = await self._get_version(version_id)
            if not version:
                raise ValueError(f"バージョンが見つかりません: {version_id}")

            if version.status == DataStatus.DELETED:
                raise ValueError(f"削除されたバージョンです: {version_id}")

            # データロード
            data = await self._load_version_data(version_id)

            logger.info(f"データチェックアウト完了: {version_id}")
            return data, version

        except Exception as e:
            logger.error(f"データチェックアウトエラー: {e}")
            raise

    async def create_branch(
        self,
        branch_name: str,
        from_version: Optional[str] = None,
        description: str = "",
        created_by: str = "system",
    ) -> str:
        """ブランチ作成"""
        logger.info(f"ブランチ作成: {branch_name}")

        try:
            # ブランチ名重複チェック
            existing_branch = await self._get_branch(branch_name)
            if existing_branch:
                raise ValueError(f"ブランチが既に存在します: {branch_name}")

            # 元バージョン決定
            if from_version is None:
                latest_version = await self._get_latest_version(self.current_branch)
                from_version = latest_version.version_id if latest_version else "root"

            # ブランチ作成
            branch = DataBranch(
                branch_name=branch_name,
                created_from=from_version,
                created_by=created_by,
                description=description,
            )

            # データベース保存
            await self._save_branch_to_db(branch)

            # キャッシュ更新
            self.branch_cache[branch_name] = branch

            logger.info(f"ブランチ作成完了: {branch_name}")
            return branch_name

        except Exception as e:
            logger.error(f"ブランチ作成エラー: {e}")
            raise

    async def merge_branches(
        self,
        source_branch: str,
        target_branch: str,
        strategy: ConflictResolution = ConflictResolution.MANUAL,
        author: str = "system",
    ) -> Tuple[str, List[VersionConflict]]:
        """ブランチマージ"""
        logger.info(f"ブランチマージ開始: {source_branch} -> {target_branch}")

        try:
            # ブランチ存在確認
            source_branch_info = await self._get_branch(source_branch)
            target_branch_info = await self._get_branch(target_branch)

            if not source_branch_info or not target_branch_info:
                raise ValueError("ブランチが見つかりません")

            # 最新バージョン取得
            source_version = await self._get_latest_version(source_branch)
            target_version = await self._get_latest_version(target_branch)

            if not source_version:
                raise ValueError(
                    f"マージ元ブランチにバージョンがありません: {source_branch}"
                )

            # 競合検出
            conflicts = await self._detect_merge_conflicts(
                source_version, target_version
            )

            if conflicts and strategy == ConflictResolution.MANUAL:
                logger.warning(f"マージ競合が検出されました: {len(conflicts)}件")
                return "", conflicts

            # 自動競合解決
            if conflicts:
                conflicts = await self._resolve_conflicts_automatically(
                    conflicts, strategy
                )

            # マージコミット作成
            merge_data = await self._create_merge_data(
                source_version, target_version, strategy
            )

            merge_message = f"Merge branch '{source_branch}' into '{target_branch}'"
            merge_version_id = await self.commit_data(
                merge_data,
                merge_message,
                author,
                target_branch,
                {
                    "merge_source": source_branch,
                    "merge_target": target_branch,
                    "merge_strategy": strategy.value,
                    "conflicts_resolved": len([c for c in conflicts if c.resolved]),
                },
            )

            # マージ操作記録
            await self._record_merge_operation(
                merge_version_id, source_branch, target_branch, conflicts
            )

            logger.info(f"ブランチマージ完了: {merge_version_id}")
            return merge_version_id, conflicts

        except Exception as e:
            logger.error(f"ブランチマージエラー: {e}")
            raise

    async def create_tag(
        self,
        tag_name: str,
        version_id: str,
        description: str = "",
        is_release: bool = False,
        created_by: str = "system",
    ) -> str:
        """タグ作成"""
        logger.info(f"タグ作成: {tag_name} -> {version_id}")

        try:
            # タグ名重複チェック
            existing_tag = await self._get_tag(tag_name)
            if existing_tag:
                raise ValueError(f"タグが既に存在します: {tag_name}")

            # バージョン存在確認
            version = await self._get_version(version_id)
            if not version:
                raise ValueError(f"バージョンが見つかりません: {version_id}")

            # タグ作成
            tag = DataTag(
                tag_name=tag_name,
                version_id=version_id,
                created_by=created_by,
                description=description,
                is_release=is_release,
            )

            # データベース保存
            await self._save_tag_to_db(tag)

            # バージョンにタグ情報更新
            version.tag = tag_name
            await self._update_version_in_db(version)

            logger.info(f"タグ作成完了: {tag_name}")
            return tag_name

        except Exception as e:
            logger.error(f"タグ作成エラー: {e}")
            raise

    async def create_snapshot(self, branch: Optional[str] = None) -> str:
        """スナップショット作成"""
        branch = branch or self.current_branch
        snapshot_id = f"snapshot_{branch}_{int(time.time())}"

        logger.info(f"スナップショット作成: {snapshot_id}")

        try:
            # ブランチの全バージョン取得
            versions = await self._list_versions(branch)

            # スナップショットディレクトリ作成
            snapshot_path = self.repository_path / "snapshots" / snapshot_id
            snapshot_path.mkdir(parents=True, exist_ok=True)

            # メタデータ作成
            snapshot_metadata = {
                "snapshot_id": snapshot_id,
                "branch": branch,
                "created_at": datetime.utcnow().isoformat(),
                "version_count": len(versions),
                "versions": [v.version_id for v in versions],
                "total_size_bytes": sum(v.size_bytes for v in versions),
            }

            # メタデータ保存
            metadata_file = snapshot_path / "snapshot.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_metadata, f, indent=2, ensure_ascii=False)

            # データファイルコピー
            for version in versions:
                source_path = (
                    self.repository_path / "data" / f"{version.version_id}.json"
                )
                if source_path.exists():
                    target_path = snapshot_path / f"{version.version_id}.json"
                    shutil.copy2(source_path, target_path)

            logger.info(f"スナップショット作成完了: {snapshot_id}")
            return snapshot_id

        except Exception as e:
            logger.error(f"スナップショット作成エラー: {e}")
            raise

    async def get_version_history(
        self, branch: Optional[str] = None, limit: int = 50
    ) -> List[DataVersion]:
        """バージョン履歴取得"""
        branch = branch or self.current_branch

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE branch = ? AND status != ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (branch, DataStatus.DELETED.value, limit),
                )

                versions = []
                for row in cursor.fetchall():
                    version = DataVersion(
                        version_id=row[0],
                        parent_version=row[1],
                        branch=row[2],
                        tag=row[3],
                        author=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        message=row[6],
                        data_hash=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                        status=DataStatus(row[9]),
                        size_bytes=row[10],
                        file_count=row[11],
                        checksum=row[12],
                    )
                    versions.append(version)

                return versions

        except Exception as e:
            logger.error(f"バージョン履歴取得エラー: {e}")
            return []

    async def get_version_diff(self, version1: str, version2: str) -> Dict[str, Any]:
        """バージョン間差分取得"""
        logger.info(f"バージョン差分計算: {version1} <-> {version2}")

        try:
            # バージョン情報取得
            v1 = await self._get_version(version1)
            v2 = await self._get_version(version2)

            if not v1 or not v2:
                raise ValueError("指定されたバージョンが見つかりません")

            # データロード
            data1, _ = await self.checkout_data(version1)
            data2, _ = await self.checkout_data(version2)

            # 差分計算
            diff_result = {
                "version1": version1,
                "version2": version2,
                "timestamp1": v1.timestamp.isoformat(),
                "timestamp2": v2.timestamp.isoformat(),
                "metadata_diff": self._calculate_metadata_diff(
                    v1.metadata, v2.metadata
                ),
                "data_diff": self._calculate_data_diff(data1, data2),
                "size_diff_bytes": v2.size_bytes - v1.size_bytes,
                "hash_changed": v1.data_hash != v2.data_hash,
            }

            return diff_result

        except Exception as e:
            logger.error(f"バージョン差分計算エラー: {e}")
            raise

    def _calculate_data_hash(self, data: Any) -> str:
        """データハッシュ計算"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrame構造とデータのハッシュ
                content = (
                    f"{list(data.columns)}_{data.shape}_{str(data.dtypes.to_dict())}"
                )
                if len(data) > 0:
                    # サンプルデータも含める（大規模データ対応）
                    sample_data = (
                        data.head(100).to_json() if len(data) > 100 else data.to_json()
                    )
                    content += f"_{sample_data}"
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True)
            elif isinstance(data, list):
                content = json.dumps(data)
            else:
                content = str(data)

            return hashlib.sha256(content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"データハッシュ計算エラー: {e}")
            return f"error_hash_{int(time.time())}"

    def _calculate_checksum(self, data: Any) -> str:
        """チェックサム計算"""
        try:
            content = str(data)
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"チェックサム計算エラー: {e}")
            return f"error_checksum_{int(time.time())}"

    def _get_data_size(self, data: Any) -> int:
        """データサイズ計算（バイト）"""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, (dict, list)):
                return len(json.dumps(data).encode())
            else:
                return len(str(data).encode())
        except Exception:
            return 0

    async def _save_version_data(self, version_id: str, data: Any) -> Path:
        """バージョンデータ保存"""
        data_path = self.repository_path / "data" / f"{version_id}.json"

        try:
            # シリアライズ可能な形に変換
            if isinstance(data, pd.DataFrame):
                serialized_data = {
                    "type": "DataFrame",
                    "data": data.to_dict(orient="records"),
                    "index": data.index.tolist(),
                    "columns": data.columns.tolist(),
                    "dtypes": data.dtypes.to_dict(),
                }
            else:
                serialized_data = {"type": type(data).__name__, "data": data}

            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False, default=str)

            return data_path

        except Exception as e:
            logger.error(f"バージョンデータ保存エラー: {e}")
            raise

    async def _load_version_data(self, version_id: str) -> Any:
        """バージョンデータロード"""
        data_path = self.repository_path / "data" / f"{version_id}.json"

        try:
            if not data_path.exists():
                raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

            with open(data_path, encoding="utf-8") as f:
                serialized_data = json.load(f)

            data_type = serialized_data.get("type", "unknown")

            if data_type == "DataFrame":
                df = pd.DataFrame(serialized_data["data"])
                if "index" in serialized_data:
                    df.index = serialized_data["index"]
                return df
            else:
                return serialized_data["data"]

        except Exception as e:
            logger.error(f"バージョンデータロードエラー: {e}")
            raise

    async def _save_version_to_db(self, version: DataVersion):
        """バージョンをデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO versions
                (version_id, parent_version, branch, tag, author, timestamp, message,
                 data_hash, metadata, status, size_bytes, file_count, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    version.version_id,
                    version.parent_version,
                    version.branch,
                    version.tag,
                    version.author,
                    version.timestamp.isoformat(),
                    version.message,
                    version.data_hash,
                    json.dumps(version.metadata, default=str),
                    version.status.value,
                    version.size_bytes,
                    version.file_count,
                    version.checksum,
                ),
            )

    async def _save_branch_to_db(self, branch: DataBranch):
        """ブランチをデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO branches
                (branch_name, created_from, created_by, created_at, description,
                 is_protected, latest_version, merge_policy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    branch.branch_name,
                    branch.created_from,
                    branch.created_by,
                    branch.created_at.isoformat(),
                    branch.description,
                    1 if branch.is_protected else 0,
                    branch.latest_version,
                    branch.merge_policy.value,
                ),
            )

    async def _save_tag_to_db(self, tag: DataTag):
        """タグをデータベース保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO tags
                (tag_name, version_id, created_by, created_at, description, is_release)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    tag.tag_name,
                    tag.version_id,
                    tag.created_by,
                    tag.created_at.isoformat(),
                    tag.description,
                    1 if tag.is_release else 0,
                ),
            )

    async def _get_version(self, version_id: str) -> Optional[DataVersion]:
        """バージョン取得"""
        # キャッシュチェック
        if version_id in self.version_cache:
            return self.version_cache[version_id]

        if self.cache_manager:
            cache_key = generate_unified_cache_key("version", version_id)
            cached_version = self.cache_manager.get(cache_key)
            if cached_version:
                self.version_cache[version_id] = cached_version
                return cached_version

        # データベース検索
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions WHERE version_id = ?
                """,
                    (version_id,),
                )

                row = cursor.fetchone()
                if row:
                    version = DataVersion(
                        version_id=row[0],
                        parent_version=row[1],
                        branch=row[2],
                        tag=row[3],
                        author=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        message=row[6],
                        data_hash=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                        status=DataStatus(row[9]),
                        size_bytes=row[10],
                        file_count=row[11],
                        checksum=row[12],
                    )

                    # キャッシュ更新
                    self.version_cache[version_id] = version
                    if self.cache_manager:
                        cache_key = generate_unified_cache_key("version", version_id)
                        self.cache_manager.put(cache_key, version, priority=4.0)

                    return version

                return None

        except Exception as e:
            logger.error(f"バージョン取得エラー: {e}")
            return None

    async def _get_branch(self, branch_name: str) -> Optional[DataBranch]:
        """ブランチ取得"""
        if branch_name in self.branch_cache:
            return self.branch_cache[branch_name]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT branch_name, created_from, created_by, created_at, description,
                           is_protected, latest_version, merge_policy
                    FROM branches WHERE branch_name = ?
                """,
                    (branch_name,),
                )

                row = cursor.fetchone()
                if row:
                    branch = DataBranch(
                        branch_name=row[0],
                        created_from=row[1],
                        created_by=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        description=row[4],
                        is_protected=bool(row[5]),
                        latest_version=row[6],
                        merge_policy=ConflictResolution(row[7]),
                    )

                    self.branch_cache[branch_name] = branch
                    return branch

                return None

        except Exception as e:
            logger.error(f"ブランチ取得エラー: {e}")
            return None

    async def _get_tag(self, tag_name: str) -> Optional[DataTag]:
        """タグ取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT tag_name, version_id, created_by, created_at, description, is_release
                    FROM tags WHERE tag_name = ?
                """,
                    (tag_name,),
                )

                row = cursor.fetchone()
                if row:
                    return DataTag(
                        tag_name=row[0],
                        version_id=row[1],
                        created_by=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        description=row[4],
                        is_release=bool(row[5]),
                    )

                return None

        except Exception as e:
            logger.error(f"タグ取得エラー: {e}")
            return None

    async def _get_latest_version(self, branch: str) -> Optional[DataVersion]:
        """最新バージョン取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE branch = ? AND status != ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """,
                    (branch, DataStatus.DELETED.value),
                )

                row = cursor.fetchone()
                if row:
                    return DataVersion(
                        version_id=row[0],
                        parent_version=row[1],
                        branch=row[2],
                        tag=row[3],
                        author=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        message=row[6],
                        data_hash=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                        status=DataStatus(row[9]),
                        size_bytes=row[10],
                        file_count=row[11],
                        checksum=row[12],
                    )

                return None

        except Exception as e:
            logger.error(f"最新バージョン取得エラー: {e}")
            return None

    async def _list_versions(self, branch: str) -> List[DataVersion]:
        """ブランチのバージョンリスト取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE branch = ? AND status != ?
                    ORDER BY timestamp ASC
                """,
                    (branch, DataStatus.DELETED.value),
                )

                versions = []
                for row in cursor.fetchall():
                    version = DataVersion(
                        version_id=row[0],
                        parent_version=row[1],
                        branch=row[2],
                        tag=row[3],
                        author=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        message=row[6],
                        data_hash=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                        status=DataStatus(row[9]),
                        size_bytes=row[10],
                        file_count=row[11],
                        checksum=row[12],
                    )
                    versions.append(version)

                return versions

        except Exception as e:
            logger.error(f"バージョンリスト取得エラー: {e}")
            return []

    async def _find_version_by_hash(
        self, data_hash: str, branch: str
    ) -> Optional[DataVersion]:
        """データハッシュによるバージョン検索"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id, parent_version, branch, tag, author, timestamp,
                           message, data_hash, metadata, status, size_bytes, file_count, checksum
                    FROM versions
                    WHERE data_hash = ? AND branch = ? AND status != ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """,
                    (data_hash, branch, DataStatus.DELETED.value),
                )

                row = cursor.fetchone()
                if row:
                    return DataVersion(
                        version_id=row[0],
                        parent_version=row[1],
                        branch=row[2],
                        tag=row[3],
                        author=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        message=row[6],
                        data_hash=row[7],
                        metadata=json.loads(row[8]) if row[8] else {},
                        status=DataStatus(row[9]),
                        size_bytes=row[10],
                        file_count=row[11],
                        checksum=row[12],
                    )

                return None

        except Exception as e:
            logger.error(f"ハッシュによるバージョン検索エラー: {e}")
            return None

    async def _update_branch_latest_version(self, branch: str, version_id: str):
        """ブランチ最新バージョン更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE branches
                    SET latest_version = ?
                    WHERE branch_name = ?
                """,
                    (version_id, branch),
                )

            # キャッシュ更新
            if branch in self.branch_cache:
                self.branch_cache[branch].latest_version = version_id

        except Exception as e:
            logger.error(f"ブランチ最新バージョン更新エラー: {e}")

    async def _update_version_in_db(self, version: DataVersion):
        """バージョン情報データベース更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE versions
                    SET tag = ?, status = ?, metadata = ?
                    WHERE version_id = ?
                """,
                    (
                        version.tag,
                        version.status.value,
                        json.dumps(version.metadata, default=str),
                        version.version_id,
                    ),
                )

            # キャッシュ更新
            self.version_cache[version.version_id] = version

        except Exception as e:
            logger.error(f"バージョン情報更新エラー: {e}")

    async def _cleanup_old_versions(self, branch: str):
        """古いバージョンのクリーンアップ"""
        if self.max_versions <= 0:
            return

        try:
            versions = await self._list_versions(branch)
            if len(versions) <= self.max_versions:
                return

            # 古いバージョンを特定（タグ付きとリリースは保護）
            versions_to_delete = []
            keep_count = 0

            for version in reversed(versions):  # 新しい順
                if version.tag or version.status == DataStatus.APPROVED:
                    continue  # 保護対象

                if keep_count < self.max_versions:
                    keep_count += 1
                else:
                    versions_to_delete.append(version)

            # 削除実行
            for version in versions_to_delete:
                await self._archive_version(version.version_id)

            if versions_to_delete:
                logger.info(f"古いバージョンをアーカイブ: {len(versions_to_delete)}件")

        except Exception as e:
            logger.error(f"バージョンクリーンアップエラー: {e}")

    async def _archive_version(self, version_id: str):
        """バージョンアーカイブ"""
        try:
            # ステータス更新
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE versions
                    SET status = ?
                    WHERE version_id = ?
                """,
                    (DataStatus.ARCHIVED.value, version_id),
                )

            # データファイル移動
            source_path = self.repository_path / "data" / f"{version_id}.json"
            archive_path = (
                self.repository_path / "backups" / f"archived_{version_id}.json"
            )

            if source_path.exists():
                shutil.move(str(source_path), str(archive_path))

            # キャッシュクリア
            self.version_cache.pop(version_id, None)

        except Exception as e:
            logger.error(f"バージョンアーカイブエラー: {e}")

    async def _create_automatic_backup(self, version_id: str):
        """自動バックアップ作成"""
        try:
            backup_id = f"auto_backup_{version_id}_{int(time.time())}"
            backup_path = self.repository_path / "backups" / backup_id
            backup_path.mkdir(parents=True, exist_ok=True)

            # データベースバックアップ
            backup_db = backup_path / "versions.db"
            shutil.copy2(self.db_path, backup_db)

            # 設定ファイルバックアップ
            config_source = self.repository_path / ".dvconfig"
            if config_source.exists():
                shutil.copy2(config_source, backup_path / ".dvconfig")

            # バックアップメタデータ
            backup_metadata = {
                "backup_id": backup_id,
                "trigger_version": version_id,
                "created_at": datetime.utcnow().isoformat(),
                "backup_type": "automatic",
            }

            with open(backup_path / "backup.json", "w", encoding="utf-8") as f:
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"自動バックアップ作成完了: {backup_id}")

        except Exception as e:
            logger.error(f"自動バックアップ作成エラー: {e}")

    def _calculate_metadata_diff(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """メタデータ差分計算"""
        diff = {"added": {}, "removed": {}, "changed": {}}

        all_keys = set(meta1.keys()) | set(meta2.keys())

        for key in all_keys:
            if key not in meta1:
                diff["added"][key] = meta2[key]
            elif key not in meta2:
                diff["removed"][key] = meta1[key]
            elif meta1[key] != meta2[key]:
                diff["changed"][key] = {"old": meta1[key], "new": meta2[key]}

        return diff

    def _calculate_data_diff(self, data1: Any, data2: Any) -> Dict[str, Any]:
        """データ差分計算"""
        diff = {
            "type_changed": type(data1) != type(data2),
            "content_summary": "データが変更されました",
        }

        if isinstance(data1, pd.DataFrame) and isinstance(data2, pd.DataFrame):
            diff.update(
                {
                    "shape_diff": {"old_shape": data1.shape, "new_shape": data2.shape},
                    "columns_diff": {
                        "added": list(set(data2.columns) - set(data1.columns)),
                        "removed": list(set(data1.columns) - set(data2.columns)),
                    },
                    "row_count_diff": len(data2) - len(data1),
                }
            )

        return diff

    async def _detect_merge_conflicts(
        self, source_version: DataVersion, target_version: Optional[DataVersion]
    ) -> List[VersionConflict]:
        """マージ競合検出"""
        conflicts = []

        if not target_version:
            return conflicts  # ターゲットが空の場合は競合なし

        # データハッシュ競合
        if source_version.data_hash != target_version.data_hash:
            conflict = VersionConflict(
                conflict_id=f"conflict_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="data",
                description="データ内容が異なります",
            )
            conflicts.append(conflict)

        # メタデータ競合
        source_meta = source_version.metadata
        target_meta = target_version.metadata

        conflicting_keys = []
        for key in set(source_meta.keys()) & set(target_meta.keys()):
            if source_meta[key] != target_meta[key]:
                conflicting_keys.append(key)

        if conflicting_keys:
            conflict = VersionConflict(
                conflict_id=f"conflict_meta_{int(time.time())}",
                source_version=source_version.version_id,
                target_version=target_version.version_id,
                conflict_type="metadata",
                description=f"メタデータ競合: {', '.join(conflicting_keys)}",
            )
            conflicts.append(conflict)

        return conflicts

    async def _resolve_conflicts_automatically(
        self, conflicts: List[VersionConflict], strategy: ConflictResolution
    ) -> List[VersionConflict]:
        """自動競合解決"""
        resolved_conflicts = []

        for conflict in conflicts:
            if strategy == ConflictResolution.AUTO_LATEST:
                # 最新バージョンを採用
                conflict.resolved = True
                conflict.resolved_by = "auto_resolver"
                conflict.resolved_at = datetime.utcnow()
                conflict.resolution_strategy = ConflictResolution.AUTO_LATEST

            elif strategy == ConflictResolution.AUTO_MERGE:
                # 自動マージ（簡易版）
                conflict.resolved = True
                conflict.resolved_by = "auto_merger"
                conflict.resolved_at = datetime.utcnow()
                conflict.resolution_strategy = ConflictResolution.AUTO_MERGE

            resolved_conflicts.append(conflict)

        return resolved_conflicts

    async def _create_merge_data(
        self,
        source_version: DataVersion,
        target_version: Optional[DataVersion],
        strategy: ConflictResolution,
    ) -> Any:
        """マージデータ作成"""
        # ソースデータをロード
        source_data, _ = await self.checkout_data(source_version.version_id)

        if not target_version:
            return source_data

        # ターゲットデータをロード
        target_data, _ = await self.checkout_data(target_version.version_id)

        # 戦略に基づくマージ
        if strategy == ConflictResolution.AUTO_LATEST:
            return source_data
        elif strategy == ConflictResolution.AUTO_MERGE:
            # 簡易マージ（実際のプロジェクトでは詳細な実装が必要）
            if isinstance(source_data, pd.DataFrame) and isinstance(
                target_data, pd.DataFrame
            ):
                # DataFrameの場合、行を結合
                return pd.concat([target_data, source_data]).drop_duplicates()
            else:
                return source_data
        else:
            return source_data

    async def _record_merge_operation(
        self,
        merge_version_id: str,
        source_branch: str,
        target_branch: str,
        conflicts: List[VersionConflict],
    ):
        """マージ操作記録"""
        try:
            # 競合情報をデータベース保存
            with sqlite3.connect(self.db_path) as conn:
                for conflict in conflicts:
                    conn.execute(
                        """
                        INSERT INTO conflicts
                        (conflict_id, source_version, target_version, conflict_type,
                         file_path, description, resolution_strategy, resolved,
                         resolved_by, resolved_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            conflict.conflict_id,
                            conflict.source_version,
                            conflict.target_version,
                            conflict.conflict_type,
                            conflict.file_path,
                            conflict.description,
                            conflict.resolution_strategy.value,
                            1 if conflict.resolved else 0,
                            conflict.resolved_by,
                            conflict.resolved_at.isoformat()
                            if conflict.resolved_at
                            else None,
                        ),
                    )

        except Exception as e:
            logger.error(f"マージ操作記録エラー: {e}")

    async def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 統計情報取得
                version_count = conn.execute(
                    "SELECT COUNT(*) FROM versions"
                ).fetchone()[0]
                branch_count = conn.execute("SELECT COUNT(*) FROM branches").fetchone()[
                    0
                ]
                tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]

                # ブランチ別バージョン数
                branch_stats = {}
                cursor = conn.execute(
                    """
                    SELECT branch, COUNT(*) as count, SUM(size_bytes) as total_size
                    FROM versions WHERE status != ?
                    GROUP BY branch
                """,
                    (DataStatus.DELETED.value,),
                )

                for branch, count, total_size in cursor.fetchall():
                    branch_stats[branch] = {
                        "version_count": count,
                        "total_size_bytes": total_size or 0,
                    }

            return {
                "repository_path": str(self.repository_path),
                "current_branch": self.current_branch,
                "cache_enabled": self.enable_cache,
                "auto_backup": self.auto_backup,
                "max_versions": self.max_versions,
                "statistics": {
                    "total_versions": version_count,
                    "total_branches": branch_count,
                    "total_tags": tag_count,
                    "branch_statistics": branch_stats,
                },
                "cache_status": {
                    "version_cache_size": len(self.version_cache),
                    "branch_cache_size": len(self.branch_cache),
                },
            }

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """リソースクリーンアップ"""
        logger.info("データバージョン管理システム クリーンアップ開始")

        # キャッシュクリア
        self.version_cache.clear()
        self.branch_cache.clear()

        # 一時ファイルクリーンアップ
        temp_path = self.repository_path / "temp"
        if temp_path.exists():
            for temp_file in temp_path.glob("*"):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                except Exception as e:
                    logger.warning(f"一時ファイル削除エラー: {e}")

        logger.info("データバージョン管理システム クリーンアップ完了")


# Factory function
def create_data_version_manager(
    repository_path: str = "data/versions",
    enable_cache: bool = True,
    max_versions: int = 100,
    auto_backup: bool = True,
) -> DataVersionManager:
    """データバージョン管理システム作成"""
    return DataVersionManager(
        repository_path=repository_path,
        enable_cache=enable_cache,
        max_versions=max_versions,
        auto_backup=auto_backup,
    )


if __name__ == "__main__":
    # テスト実行
    async def test_data_version_manager():
        print("=== Issue #420 データバージョン管理システム（DVC）テスト ===")

        try:
            # バージョン管理システム初期化
            dvc = create_data_version_manager(
                repository_path="test_versions",
                enable_cache=True,
                max_versions=10,
                auto_backup=True,
            )

            print("\n1. データバージョン管理システム初期化完了")
            status = await dvc.get_system_status()
            print(f"   リポジトリパス: {status['repository_path']}")
            print(f"   現在ブランチ: {status['current_branch']}")
            print(f"   バージョン数: {status['statistics']['total_versions']}")

            # テストデータ準備
            print("\n2. テストデータ準備...")
            test_data_v1 = pd.DataFrame(
                {
                    "symbol": ["7203", "8306", "9984"],
                    "price": [2500, 800, 4200],
                    "volume": [1000000, 500000, 2000000],
                }
            )

            test_data_v2 = pd.DataFrame(
                {
                    "symbol": ["7203", "8306", "9984", "6758"],
                    "price": [2520, 810, 4150, 12500],
                    "volume": [1200000, 450000, 1800000, 800000],
                }
            )

            # 初回コミット
            print("\n3. 初回データコミット...")
            version1_id = await dvc.commit_data(
                test_data_v1,
                "初回データコミット",
                "test_user",
                metadata={"source": "test_data", "version": "1.0"},
            )
            print(f"   バージョン1: {version1_id}")

            # 更新コミット
            print("\n4. 更新データコミット...")
            version2_id = await dvc.commit_data(
                test_data_v2,
                "データ更新 - 新銘柄追加",
                "test_user",
                metadata={"source": "test_data", "version": "2.0"},
            )
            print(f"   バージョン2: {version2_id}")

            # データチェックアウト
            print("\n5. データチェックアウトテスト...")
            retrieved_data, retrieved_version = await dvc.checkout_data(version1_id)
            print(f"   チェックアウト成功: {retrieved_version.version_id}")
            print(f"   データ形状: {retrieved_data.shape}")

            # ブランチ作成
            print("\n6. ブランチ作成テスト...")
            feature_branch = await dvc.create_branch(
                "feature/new-analysis", version1_id, "新しい分析機能用ブランチ"
            )
            print(f"   新ブランチ: {feature_branch}")

            # ブランチ切り替えしてコミット
            dvc.current_branch = feature_branch
            test_data_feature = test_data_v1.copy()
            test_data_feature["analysis_flag"] = True

            version3_id = await dvc.commit_data(
                test_data_feature, "分析フラグ追加", "test_user", branch=feature_branch
            )
            print(f"   フィーチャーブランチバージョン: {version3_id}")

            # タグ作成
            print("\n7. タグ作成テスト...")
            tag_name = await dvc.create_tag(
                "v1.0.0", version2_id, "初回リリース版", is_release=True
            )
            print(f"   作成タグ: {tag_name}")

            # バージョン履歴取得
            print("\n8. バージョン履歴取得...")
            main_history = await dvc.get_version_history("main", 10)
            print(f"   mainブランチ履歴: {len(main_history)}件")
            for version in main_history:
                print(
                    f"     - {version.version_id}: {version.message} ({version.author})"
                )

            # バージョン差分計算
            print("\n9. バージョン差分計算...")
            diff_result = await dvc.get_version_diff(version1_id, version2_id)
            print("   差分計算完了:")
            print(f"     - サイズ変更: {diff_result['size_diff_bytes']} bytes")
            print(f"     - ハッシュ変更: {diff_result['hash_changed']}")

            # ブランチマージ
            print("\n10. ブランチマージテスト...")
            try:
                merge_version_id, conflicts = await dvc.merge_branches(
                    feature_branch, "main", ConflictResolution.AUTO_LATEST
                )
                if merge_version_id:
                    print(f"    マージ成功: {merge_version_id}")
                else:
                    print(f"    マージ競合: {len(conflicts)}件")
            except Exception as e:
                print(f"    マージエラー: {e}")

            # スナップショット作成
            print("\n11. スナップショット作成...")
            snapshot_id = await dvc.create_snapshot("main")
            print(f"    スナップショット: {snapshot_id}")

            # システム状態確認
            print("\n12. 最終システム状態...")
            final_status = await dvc.get_system_status()
            print(f"    総バージョン数: {final_status['statistics']['total_versions']}")
            print(f"    総ブランチ数: {final_status['statistics']['total_branches']}")
            print(f"    総タグ数: {final_status['statistics']['total_tags']}")

            # クリーンアップ
            await dvc.cleanup()

            print("\n✅ Issue #420 データバージョン管理システムテスト完了")

        except Exception as e:
            print(f"❌ テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_data_version_manager())
