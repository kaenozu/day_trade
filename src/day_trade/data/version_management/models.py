#!/usr/bin/env python3
"""
データバージョン管理システム - モデル定義

Issue #420: データ管理とデータ品質保証メカニズムの強化

データバージョン管理に関するデータクラスとEnumの定義
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

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
    """データバージョン情報

    Attributes:
        version_id: バージョンの一意識別子
        parent_version: 親バージョンのID
        branch: 所属するブランチ名
        tag: タグ名（オプション）
        author: 作成者
        timestamp: 作成日時
        message: コミットメッセージ
        data_hash: データのハッシュ値
        metadata: メタデータ情報
        status: データステータス
        size_bytes: データサイズ（バイト）
        file_count: ファイル数
        checksum: チェックサム値
    """

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
    """データブランチ情報

    Attributes:
        branch_name: ブランチ名
        created_from: 作成元バージョンID
        created_by: 作成者
        created_at: 作成日時
        description: ブランチの説明
        is_protected: 保護されたブランチかどうか
        latest_version: 最新バージョンID
        merge_policy: マージ時の競合解決方針
    """

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
    """データタグ情報

    Attributes:
        tag_name: タグ名
        version_id: 対象バージョンID
        created_by: 作成者
        created_at: 作成日時
        description: タグの説明
        is_release: リリース用タグかどうか
    """

    tag_name: str
    version_id: str
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    is_release: bool = False


@dataclass
class VersionConflict:
    """バージョン競合情報

    Attributes:
        conflict_id: 競合の一意識別子
        source_version: ソースバージョンID
        target_version: ターゲットバージョンID
        conflict_type: 競合タイプ（file, metadata, schema）
        file_path: 競合ファイルパス（オプション）
        description: 競合の説明
        resolution_strategy: 解決戦略
        resolved: 解決済みかどうか
        resolved_by: 解決者
        resolved_at: 解決日時
    """

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