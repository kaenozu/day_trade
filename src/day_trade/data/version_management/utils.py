#!/usr/bin/env python3
"""
データバージョン管理システム - ユーティリティ関数

Issue #420: データ管理とデータ品質保証メカニズムの強化

共通のユーティリティ関数とヘルパー機能を提供
"""

import shutil
import warnings
from pathlib import Path
from typing import Any, Dict

# 警告抑制
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


def initialize_repository_structure(repository_path: Path) -> None:
    """
    リポジトリディレクトリ構造初期化
    
    Args:
        repository_path: リポジトリのパス
    """
    repository_path.mkdir(parents=True, exist_ok=True)

    # サブディレクトリ作成
    (repository_path / "data").mkdir(exist_ok=True)
    (repository_path / "metadata").mkdir(exist_ok=True)
    (repository_path / "snapshots").mkdir(exist_ok=True)
    (repository_path / "backups").mkdir(exist_ok=True)
    (repository_path / "temp").mkdir(exist_ok=True)

    logger.debug(f"リポジトリ構造初期化完了: {repository_path}")


def create_config_file(repository_path: Path, config: Dict[str, Any]) -> None:
    """
    設定ファイル作成
    
    Args:
        repository_path: リポジトリのパス
        config: 設定情報
    """
    import json
    from datetime import datetime
    
    config_file = repository_path / ".dvconfig"
    if not config_file.exists():
        default_config = {
            "version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "default_branch": "main",
            **config
        }
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"設定ファイル作成完了: {config_file}")


def cleanup_temp_files(repository_path: Path) -> int:
    """
    一時ファイルクリーンアップ
    
    Args:
        repository_path: リポジトリのパス
        
    Returns:
        削除されたファイル数
    """
    temp_path = repository_path / "temp"
    if not temp_path.exists():
        return 0

    deleted_count = 0
    for temp_file in temp_path.glob("*"):
        try:
            if temp_file.is_file():
                temp_file.unlink()
                deleted_count += 1
            elif temp_file.is_dir():
                shutil.rmtree(temp_file)
                deleted_count += 1
        except Exception as e:
            logger.warning(f"一時ファイル削除エラー: {e}")

    logger.debug(f"一時ファイルクリーンアップ完了: {deleted_count}件削除")
    return deleted_count


def validate_version_id(version_id: str) -> bool:
    """
    バージョンIDの妥当性検証
    
    Args:
        version_id: 検証するバージョンID
        
    Returns:
        妥当性フラグ
    """
    if not version_id or not isinstance(version_id, str):
        return False
    
    if len(version_id) < 3 or len(version_id) > 100:
        return False
    
    # 基本的な形式チェック（例：v_123456789_abcd1234）
    if version_id.startswith("v_") and "_" in version_id[2:]:
        return True
    
    return False


def validate_branch_name(branch_name: str) -> bool:
    """
    ブランチ名の妥当性検証
    
    Args:
        branch_name: 検証するブランチ名
        
    Returns:
        妥当性フラグ
    """
    if not branch_name or not isinstance(branch_name, str):
        return False
    
    if len(branch_name) < 1 or len(branch_name) > 50:
        return False
    
    # 基本的な命名規則チェック
    invalid_chars = {" ", "\t", "\n", "\r", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"}
    if any(char in branch_name for char in invalid_chars):
        return False
    
    return True


def validate_tag_name(tag_name: str) -> bool:
    """
    タグ名の妥当性検証
    
    Args:
        tag_name: 検証するタグ名
        
    Returns:
        妥当性フラグ
    """
    if not tag_name or not isinstance(tag_name, str):
        return False
    
    if len(tag_name) < 1 or len(tag_name) > 50:
        return False
    
    # 基本的な命名規則チェック
    invalid_chars = {" ", "\t", "\n", "\r", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"}
    if any(char in tag_name for char in invalid_chars):
        return False
    
    return True


def format_file_size(size_bytes: int) -> str:
    """
    ファイルサイズを人間に読みやすい形式でフォーマット
    
    Args:
        size_bytes: サイズ（バイト）
        
    Returns:
        フォーマットされたサイズ文字列
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


def format_duration(duration_seconds: float) -> str:
    """
    処理時間を人間に読みやすい形式でフォーマット
    
    Args:
        duration_seconds: 処理時間（秒）
        
    Returns:
        フォーマットされた時間文字列
    """
    if duration_seconds < 1:
        return f"{duration_seconds * 1000:.1f}ms"
    elif duration_seconds < 60:
        return f"{duration_seconds:.2f}s"
    elif duration_seconds < 3600:
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_repository_size(repository_path: Path) -> Dict[str, int]:
    """
    リポジトリサイズ計算
    
    Args:
        repository_path: リポジトリのパス
        
    Returns:
        各ディレクトリのサイズ情報
    """
    size_info = {
        "data": 0,
        "metadata": 0,
        "snapshots": 0,
        "backups": 0,
        "temp": 0,
        "total": 0
    }
    
    try:
        for dir_name in ["data", "metadata", "snapshots", "backups", "temp"]:
            dir_path = repository_path / dir_name
            if dir_path.exists():
                dir_size = sum(
                    f.stat().st_size for f in dir_path.rglob('*') if f.is_file()
                )
                size_info[dir_name] = dir_size
                size_info["total"] += dir_size
                
    except Exception as e:
        logger.error(f"リポジトリサイズ計算エラー: {e}")
    
    return size_info


def get_system_info() -> Dict[str, Any]:
    """
    システム情報取得
    
    Returns:
        システム情報
    """
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }