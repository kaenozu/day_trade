#!/usr/bin/env python3
"""
銘柄マスタ関連の設定

銘柄一括登録機能で使用される設定値を定義する。
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_stock_master_config() -> Dict[str, Any]:
    """
    銘柄マスタ設定を取得

    Returns:
        設定辞書
    """
    # デフォルト設定
    default_config = {
        "session_management": {
            "request_timeout": 30,
            "connection_timeout": 10,
            "retry_count": 3
        },
        "performance": {
            "default_bulk_batch_size": 100,
            "fetch_batch_size": 50,
            "max_concurrent_requests": 5
        },
        "validation": {
            "validate_symbol_format": True,
            "validate_company_name": True,
            "skip_invalid_records": True
        }
    }

    # 設定ファイルからの読み込みを試行
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "stock_master_config.json"

    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                # デフォルト設定にファイル設定をマージ
                _merge_config(default_config, file_config)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"設定ファイル読み込みエラー: {e}、デフォルト設定を使用")

    return default_config


def _merge_config(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """
    設定辞書をマージ

    Args:
        base: ベース設定（更新される）
        overlay: オーバーレイ設定
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def save_stock_master_config(config: Dict[str, Any]) -> None:
    """
    銘柄マスタ設定を保存

    Args:
        config: 設定辞書
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "stock_master_config.json"

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
