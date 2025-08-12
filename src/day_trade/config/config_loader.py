"""
統合設定ローダー

分割された設定ファイルを統合して読み込む機能を提供します。
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.day_trade.utils.logging_config import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """分割設定ファイルを統合して読み込むクラス"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._cache = {}

    def load_config(self) -> Dict[str, Any]:
        """全設定ファイルを統合して読み込み"""
        if self._cache:
            return self._cache

        config = {}

        # 設定ファイルの読み込み順序（依存関係を考慮）
        config_files = [
            "base.json",
            "watchlist.json",
            "analysis.json",
            "ml.json",
            "risk_management.json",
            "monitoring.json",
        ]

        for config_file in config_files:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        file_config = json.load(f)
                        config.update(file_config)
                        logger.info(f"設定ファイル読み込み成功: {config_file}")
                except Exception as e:
                    logger.error(
                        f"設定ファイル読み込みエラー: {config_file}",
                        extra={"error": str(e)},
                    )
                    continue
            else:
                logger.warning(f"設定ファイルが見つかりません: {config_file}")

        # 基本構造の検証
        if not config:
            raise ValueError("有効な設定ファイルが見つかりませんでした")

        self._cache = config
        logger.info("統合設定読み込み完了", extra={"sections": list(config.keys())})
        return config

    def get_section(self, section_name: str) -> Dict[str, Any]:
        """特定のセクションを取得"""
        config = self.load_config()
        return config.get(section_name, {})

    def reload(self):
        """設定キャッシュをクリアして再読み込み"""
        self._cache = {}
        logger.info("設定キャッシュをクリアしました")

    def validate_config(self) -> bool:
        """設定の基本的な整合性チェック"""
        try:
            config = self.load_config()

            # 必須セクションの確認
            required_sections = ["watchlist", "analysis", "risk_management"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"必須セクションが見つかりません: {section}")
                    return False

            # watchlistの基本チェック
            if "symbols" not in config.get("watchlist", {}):
                logger.error("watchlist.symbolsが見つかりません")
                return False

            symbols = config["watchlist"]["symbols"]
            if not symbols or not isinstance(symbols, list):
                logger.error("有効な銘柄リストが見つかりません")
                return False

            logger.info("設定の整合性チェック完了", extra={"symbols_count": len(symbols)})
            return True

        except Exception as e:
            logger.error("設定検証エラー", extra={"error": str(e)})
            return False


# グローバルインスタンス
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """統合設定ローダーのシングルトンインスタンスを取得"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_integrated_config() -> Dict[str, Any]:
    """統合設定を読み込み（便利関数）"""
    return get_config_loader().load_config()


def get_config_section(section_name: str) -> Dict[str, Any]:
    """設定セクションを取得（便利関数）"""
    return get_config_loader().get_section(section_name)
