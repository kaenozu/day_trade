#!/usr/bin/env python3
"""
Day Trade Personal - 共通ユーティリティ

リファクタリングにより統合された共通機能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class CommonUtils:
    """共通ユーティリティクラス"""

    @staticmethod
    def setup_paths():
        """パス設定"""
        base_dir = Path(__file__).parent.parent.parent.parent
        src_dir = base_dir / "src"

        if str(src_dir) not in sys.path:
            sys.path.append(str(src_dir))

        return base_dir, src_dir

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """ロガー取得"""
        return logging.getLogger(name)

    @staticmethod
    def format_currency(amount: float) -> str:
        """通貨フォーマット"""
        return f"¥{amount:,.0f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """パーセンテージフォーマット"""
        return f"{value:.2%}"

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全な除算"""
        return numerator / denominator if denominator != 0 else default


class FileUtils:
    """ファイル操作ユーティリティ"""

    @staticmethod
    def ensure_directory(path: Path):
        """ディレクトリ確保"""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_read_json(file_path: Path, default: Dict = None) -> Dict:
        """安全なJSON読み込み"""
        if default is None:
            default = {}

        try:
            if file_path.exists():
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"JSON読み込みエラー {file_path}: {e}")

        return default

    @staticmethod
    def safe_write_json(file_path: Path, data: Dict):
        """安全なJSON書き込み"""
        try:
            import json
            FileUtils.ensure_directory(file_path.parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"JSON書き込みエラー {file_path}: {e}")


class DateUtils:
    """日付ユーティリティ"""

    @staticmethod
    def get_current_timestamp() -> str:
        """現在のタイムスタンプ"""
        return datetime.now().isoformat()

    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """日時フォーマット"""
        return dt.strftime(format_str)

    @staticmethod
    def is_market_hours() -> bool:
        """市場時間判定（簡易版）"""
        now = datetime.now()
        # 平日の9:00-15:00を市場時間とする（簡易版）
        return (
            now.weekday() < 5 and  # 平日
            9 <= now.hour < 15      # 9時-15時
        )
