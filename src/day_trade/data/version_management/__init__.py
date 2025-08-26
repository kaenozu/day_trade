#!/usr/bin/env python3
"""
データバージョン管理システム - パッケージ初期化

Issue #420: データ管理とデータ品質保証メカニズムの強化

後方互換性を保つため、元のクラス・関数を再エクスポート
"""

# 主要なクラスとデータ型のインポート
from .models import (
    VersionOperation,
    DataStatus,
    ConflictResolution,
    DataVersion,
    DataBranch,
    DataTag,
    VersionConflict,
)

# メインマネージャークラスのインポート
from .manager import DataVersionManager, create_data_version_manager

# 各サブマネージャークラスのインポート
from .database import DatabaseManager
from .version_operations import VersionOperations
from .version_core import VersionCore
from .version_db_operations import VersionDbOperations
from .branch_manager import BranchManager
from .merge_manager import MergeManager
from .snapshot_manager import SnapshotManager
from .data_manager import DataManager

# ユーティリティ関数のインポート
from .utils import (
    initialize_repository_structure,
    create_config_file,
    cleanup_temp_files,
    validate_version_id,
    validate_branch_name,
    validate_tag_name,
    format_file_size,
    format_duration,
    calculate_repository_size,
    get_system_info,
)

# パッケージ情報
__version__ = "1.0.0"
__author__ = "Data Management Team"
__description__ = "データバージョン管理システム - 分割モジュラー版"

# 後方互換性のためのエクスポート
__all__ = [
    # データ型・Enum
    "VersionOperation",
    "DataStatus", 
    "ConflictResolution",
    "DataVersion",
    "DataBranch",
    "DataTag",
    "VersionConflict",
    
    # メインマネージャー
    "DataVersionManager",
    "create_data_version_manager",
    
    # サブマネージャー
    "DatabaseManager",
    "VersionOperations",
    "VersionCore",
    "VersionDbOperations",
    "BranchManager",
    "MergeManager",
    "SnapshotManager",
    "DataManager",
    
    # ユーティリティ
    "initialize_repository_structure",
    "create_config_file",
    "cleanup_temp_files",
    "validate_version_id",
    "validate_branch_name", 
    "validate_tag_name",
    "format_file_size",
    "format_duration",
    "calculate_repository_size",
    "get_system_info",
]