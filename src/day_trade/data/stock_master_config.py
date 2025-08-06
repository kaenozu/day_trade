"""
銘柄マスタ関連の設定

銘柄一括登録機能で使用される設定値を定義する。
StockMasterManager設定管理
パフォーマンス設定と動作設定の外部化
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


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
            "retry_count": 3,
        },
        "performance": {
            "default_bulk_batch_size": 100,
            "fetch_batch_size": 50,
            "max_concurrent_requests": 5,
        },
        "validation": {
            "validate_symbol_format": True,
            "validate_company_name": True,
            "skip_invalid_records": True,
        },
        "limits": {
            "max_stock_count": None,  # None = 制限なし、整数 = 最大銘柄数
            "default_search_limit": 50,
            "max_search_limit": 1000,
        },
    }

    # 設定ファイルからの読み込みを試行
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "config"
        / "stock_master_config.json"
    )

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
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

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# Issue #122互換性のために、関数ベースのインターフェースを保持
# mainブランチのクラスベース実装もインポートして利用可能にする
from ..utils.logging_config import get_context_logger  # noqa: E402

logger = get_context_logger(__name__)


class StockMasterConfig:
    """StockMaster設定管理クラス"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            # デフォルトの設定ファイルパスを設定
            current_dir = Path(__file__).parent.parent.parent.parent
            config_path = current_dir / "config" / "stock_master_config.json"

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込み"""
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"StockMaster設定を読み込み: {self.config_path}")
                return config
            else:
                logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "session_management": {
                "default_detached": False,
                "auto_expunge": False,
                "eager_loading": True,
                "session_scope_isolation": True,
                "request_timeout": 30,
                "connection_timeout": 10,
                "retry_count": 3,
            },
            "performance": {
                "default_bulk_batch_size": 100,  # Issue #122デフォルト
                "fetch_batch_size": 50,
                "fetch_delay_seconds": 0.1,
                "search_limit_default": 50,
                "search_limit_max": 1000,
                "max_concurrent_requests": 5,
            },
            "validation": {
                "require_code": True,
                "require_name": True,
                "validate_code_format": True,
                "max_name_length": 255,
                "validate_symbol_format": True,
                "validate_company_name": True,
                "skip_invalid_records": True,
                "allowed_markets": [
                    "東証プライム",
                    "東証スタンダード",
                    "東証グロース",
                    "ETF",
                ],
            },
            "logging": {
                "log_bulk_operations": True,
                "log_performance_metrics": True,
                "log_cache_operations": False,
                "detailed_error_logging": True,
            },
            "limits": {
                "max_stock_count": None,  # None = 制限なし、整数 = 最大銘柄数
                "default_search_limit": 50,
                "max_search_limit": 1000,
                "test_mode_limit": 100,  # テストモード用制限
            },
        }

    # Issue #122互換メソッド（従来の関数ベースconfig用）
    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（Issue #122互換）"""
        return self._config.copy()


# グローバル設定インスタンス
_global_config_class = None


def get_stock_master_config_class() -> StockMasterConfig:
    """グローバル設定クラスインスタンスを取得"""
    global _global_config_class
    if _global_config_class is None:
        _global_config_class = StockMasterConfig()
    return _global_config_class
