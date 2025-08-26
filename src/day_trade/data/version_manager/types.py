#!/usr/bin/env python3
"""
データバージョン管理システムの型定義

このモジュールは、データバージョン管理システムで使用される
すべての列挙型とデータクラスを定義します。

Classes:
    VersionOperation: バージョン操作種別
    DataStatus: データステータス
    ConflictResolution: 競合解決方式
    DataVersion: データバージョン情報
    DataBranch: データブランチ情報
    DataTag: データタグ情報
    VersionConflict: バージョン競合情報
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class VersionOperation(Enum):
    """バージョン操作種別
    
    データバージョン管理で実行される操作の種類を定義します。
    """
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    BRANCH = "branch"
    TAG = "tag"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"


class DataStatus(Enum):
    """データステータス
    
    データバージョンのライフサイクルを管理するためのステータスです。
    """
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    DELETED = "deleted"
    DRAFT = "draft"
    APPROVED = "approved"


class ConflictResolution(Enum):
    """競合解決方式
    
    ブランチマージ時の競合解決戦略を定義します。
    """
    MANUAL = "manual"
    AUTO_LATEST = "auto_latest"
    AUTO_MERGE = "auto_merge"
    KEEP_BOTH = "keep_both"


@dataclass
class DataVersion:
    """データバージョン情報
    
    データの各バージョンに関する詳細情報を保持するデータクラス。
    データのハッシュ、メタデータ、ファイルサイズなどを含みます。
    
    Attributes:
        version_id: 一意のバージョン識別子
        parent_version: 親バージョンのID（ブランチ分岐の追跡用）
        branch: 所属ブランチ名
        tag: 関連付けられたタグ名（任意）
        author: バージョン作成者
        timestamp: 作成タイムスタンプ
        message: コミットメッセージ
        data_hash: データのハッシュ値（重複検出用）
        metadata: 追加のメタデータ情報
        status: 現在のデータステータス
        size_bytes: データサイズ（バイト単位）
        file_count: 関連ファイル数
        checksum: データのチェックサム
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

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "branch": self.branch,
            "tag": self.tag,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataVersion":
        """辞書から生成"""
        return cls(
            version_id=data["version_id"],
            parent_version=data.get("parent_version"),
            branch=data.get("branch", "main"),
            tag=data.get("tag"),
            author=data.get("author", "system"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message=data.get("message", ""),
            data_hash=data.get("data_hash", ""),
            metadata=data.get("metadata", {}),
            status=DataStatus(data.get("status", "active")),
            size_bytes=data.get("size_bytes", 0),
            file_count=data.get("file_count", 0),
            checksum=data.get("checksum", ""),
        )


@dataclass
class DataBranch:
    """データブランチ情報
    
    データのブランチ（並行開発ライン）を管理するための情報。
    
    Attributes:
        branch_name: ブランチ名
        created_from: 作成元のバージョンID
        created_by: ブランチ作成者
        created_at: 作成タイムスタンプ
        description: ブランチの説明
        is_protected: 保護フラグ（削除禁止等）
        latest_version: 最新バージョンID
        merge_policy: デフォルトのマージ戦略
    """
    branch_name: str
    created_from: str  # parent version_id
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    is_protected: bool = False
    latest_version: Optional[str] = None
    merge_policy: ConflictResolution = ConflictResolution.MANUAL

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "branch_name": self.branch_name,
            "created_from": self.created_from,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "is_protected": self.is_protected,
            "latest_version": self.latest_version,
            "merge_policy": self.merge_policy.value,
        }


@dataclass
class DataTag:
    """データタグ情報
    
    特定のバージョンに名前を付けるためのタグ機能。
    リリースバージョンの管理等に使用します。
    
    Attributes:
        tag_name: タグ名
        version_id: 対象バージョンID
        created_by: タグ作成者
        created_at: 作成タイムスタンプ
        description: タグの説明
        is_release: リリースタグかどうか
    """
    tag_name: str
    version_id: str
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    is_release: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "tag_name": self.tag_name,
            "version_id": self.version_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "is_release": self.is_release,
        }


@dataclass
class VersionConflict:
    """バージョン競合情報
    
    ブランチマージ時に発生する競合を管理するための情報。
    
    Attributes:
        conflict_id: 競合の一意識別子
        source_version: マージ元バージョンID
        target_version: マージ先バージョンID
        conflict_type: 競合タイプ（file, metadata, schema等）
        file_path: 競合ファイルパス（任意）
        description: 競合の詳細説明
        resolution_strategy: 適用された解決戦略
        resolved: 解決済みフラグ
        resolved_by: 解決者
        resolved_at: 解決タイムスタンプ
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

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "conflict_id": self.conflict_id,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "conflict_type": self.conflict_type,
            "file_path": self.file_path,
            "description": self.description,
            "resolution_strategy": self.resolution_strategy.value,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": (
                self.resolved_at.isoformat() if self.resolved_at else None
            ),
        }

    def mark_resolved(
        self, resolved_by: str, strategy: ConflictResolution
    ) -> None:
        """競合を解決済みとしてマーク"""
        self.resolved = True
        self.resolved_by = resolved_by
        self.resolved_at = datetime.utcnow()
        self.resolution_strategy = strategy