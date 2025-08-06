#!/usr/bin/env python3
"""
パターン分析関連の設定

テクニカル分析パターンの設定値を定義する。
"""

import json
from pathlib import Path
from typing import Dict, Any


def get_patterns_config() -> Dict[str, Any]:
    """
    パターン分析設定を取得

    Returns:
        設定辞書
    """
    # デフォルト設定
    default_config = {
        "golden_dead_cross": {
            "fast_period": 5,
            "slow_period": 25,
            "confidence_threshold": 70
        },
        "support_resistance": {
            "window_size": 20,
            "strength_threshold": 3
        },
        "breakout_detection": {
            "volume_factor": 1.5,
            "price_change_threshold": 2.0
        },
        "rsi": {
            "period": 14,
            "oversold": 30,
            "overbought": 70
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2
        }
    }

    # 設定ファイルからの読み込みを試行
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "patterns_config.json"

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


def save_patterns_config(config: Dict[str, Any]) -> None:
    """
    パターン分析設定を保存

    Args:
        config: 設定辞書
    """
    config_dir = Path(__file__).parent.parent.parent.parent / "config"
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "patterns_config.json"

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
