#!/usr/bin/env python3
"""
データバージョン管理システム - 統合パッケージ

このパッケージは、元のdata_version_manager.pyの機能を複数のモジュールに分割し、
バックワード互換性を保ちながら、より保守しやすい構造を提供します。

公開API:
    DataVersionManager: メインマネージャークラス
    create_data_version_manager: ファクトリー関数
    
    データクラス:
        DataVersion: バージョン情報
        DataBranch: ブランチ情報
        DataTag: タグ情報
        VersionConflict: 競合情報
        
    列挙型:
        VersionOperation: バージョン操作種別
        DataStatus: データステータス
        ConflictResolution: 競合解決方式

使用例:
    ```python
    from day_trade.data.version_manager import DataVersionManager, create_data_version_manager
    
    # 旧来の方法（互換性）
    dvc = create_data_version_manager()
    
    # 新しい方法
    dvc = DataVersionManager(repository_path="my_versions")
    
    # データコミット
    version_id = await dvc.commit_data(data, "コミットメッセージ")
    
    # データチェックアウト
    data, version = await dvc.checkout_data(version_id)
    ```
"""

# 型定義をインポート
from .types import (
    ConflictResolution,
    DataBranch,
    DataStatus,
    DataTag,
    DataVersion,
    VersionConflict,
    VersionOperation,
)

# サブマネージャーをインポート（必要に応じて）
from .branch_manager import BranchManager
from .data_operations import DataOperations
from .database import DatabaseManager
from .diff_calculator import DiffCalculator
from .merge_manager import MergeManager
from .snapshot_manager import SnapshotManager
from .tag_manager import TagManager

# メインマネージャーとファクトリー関数をインポート
from .manager import DataVersionManager, create_data_version_manager

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "データバージョン管理システム - 分割モジュール版"

# パブリックAPI
__all__ = [
    # メインクラス
    "DataVersionManager",
    "create_data_version_manager",
    
    # データクラス
    "DataVersion",
    "DataBranch",
    "DataTag", 
    "VersionConflict",
    
    # 列挙型
    "VersionOperation",
    "DataStatus",
    "ConflictResolution",
    
    # サブマネージャー（上級ユーザー向け）
    "DatabaseManager",
    "DataOperations",
    "BranchManager",
    "TagManager",
    "MergeManager",
    "SnapshotManager",
    "DiffCalculator",
    
    # メタデータ
    "__version__",
    "__author__",
    "__description__",
]

# 互換性のためのエイリアス
# 元のdata_version_manager.pyからの移行を容易にする
DataVersionControl = DataVersionManager  # 旧名称との互換性

# モジュール情報
MODULE_INFO = {
    "name": "version_manager",
    "version": __version__,
    "description": __description__,
    "modules": {
        "types": "データクラスと列挙型の定義",
        "database": "SQLiteデータベース管理",
        "data_operations": "データシリアライゼーション・ハッシュ計算",
        "branch_manager": "ブランチ作成・管理機能",
        "tag_manager": "タグ作成・管理機能",
        "merge_manager": "ブランチマージ・競合解決機能",
        "snapshot_manager": "スナップショット・バックアップ機能",
        "diff_calculator": "バージョン間差分計算機能",
        "manager": "統合メインマネージャー",
    },
    "features": [
        "データバージョニング",
        "ブランチ管理",
        "タグ管理",
        "マージ・競合解決",
        "スナップショット作成",
        "差分計算",
        "自動バックアップ",
        "キャッシュ機能",
        "メタデータ管理",
    ],
}


def get_version_info() -> dict:
    """バージョン情報を取得
    
    Returns:
        バージョン情報辞書
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "module_info": MODULE_INFO,
    }


def validate_environment() -> bool:
    """実行環境の妥当性をチェック
    
    Returns:
        環境が有効な場合True
    """
    try:
        import pandas as pd
        import sqlite3
        from pathlib import Path
        
        # 基本的な依存関係チェック
        required_modules = ['pandas', 'sqlite3', 'pathlib', 'json', 'hashlib']
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                return False
                
        return True
        
    except Exception:
        return False


# モジュール読み込み時の初期化処理
def _initialize_module():
    """モジュール初期化処理"""
    import logging
    
    # 環境チェック
    if not validate_environment():
        logging.warning(
            "データバージョン管理システムの実行環境に問題があります。"
            "必要な依存関係を確認してください。"
        )
    
    # 設定の読み込み（可能であれば）
    try:
        # ロギング設定
        logger = logging.getLogger(__name__)
        logger.debug(f"データバージョン管理システム v{__version__} 読み込み完了")
        
    except Exception as e:
        print(f"データバージョン管理システム初期化警告: {e}")


# モジュール読み込み時の初期化実行
_initialize_module()