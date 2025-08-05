"""
StockMasterManager設定管理
パフォーマンス設定と動作設定の外部化
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.logging_config import get_context_logger

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
                with open(self.config_path, encoding='utf-8') as f:
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
                "session_scope_isolation": True
            },
            "performance": {
                "default_bulk_batch_size": 1000,
                "fetch_batch_size": 50,
                "fetch_delay_seconds": 0.1,
                "search_limit_default": 50,
                "search_limit_max": 1000
            },
            "data_fetching": {
                "enable_retry": True,
                "retry_attempts": 3,
                "retry_delay": 1.0,
                "enable_caching": True,
                "cache_ttl_seconds": 3600
            },
            "market_estimation": {
                "use_market_cap_estimation": True,
                "prime_market_cap_threshold": 100_000_000_000,  # 1000億ドル
                "standard_market_cap_threshold": 10_000_000_000,  # 100億ドル
                "default_market_segment": "東証プライム",
                "etf_code_ranges": [
                    {"start": 1300, "end": 1399},
                    {"start": 1500, "end": 1599}
                ],
                "growth_code_ranges": [
                    {"start": 2000, "end": 2999}
                ]
            },
            "validation": {
                "require_code": True,
                "require_name": True,
                "validate_code_format": True,
                "max_name_length": 255,
                "allowed_markets": [
                    "東証プライム", "東証スタンダード", "東証グロース", "ETF"
                ]
            },
            "logging": {
                "log_bulk_operations": True,
                "log_performance_metrics": True,
                "log_cache_operations": False,
                "detailed_error_logging": True
            }
        }

    # セッション管理設定
    def get_default_detached(self) -> bool:
        return self._config.get("session_management", {}).get("default_detached", False)

    def should_auto_expunge(self) -> bool:
        return self._config.get("session_management", {}).get("auto_expunge", False)

    def should_use_eager_loading(self) -> bool:
        return self._config.get("session_management", {}).get("eager_loading", True)

    def should_use_session_scope_isolation(self) -> bool:
        return self._config.get("session_management", {}).get("session_scope_isolation", True)

    # パフォーマンス設定
    def get_default_bulk_batch_size(self) -> int:
        return self._config.get("performance", {}).get("default_bulk_batch_size", 1000)

    def get_fetch_batch_size(self) -> int:
        return self._config.get("performance", {}).get("fetch_batch_size", 50)

    def get_fetch_delay_seconds(self) -> float:
        return self._config.get("performance", {}).get("fetch_delay_seconds", 0.1)

    def get_search_limit_default(self) -> int:
        return self._config.get("performance", {}).get("search_limit_default", 50)

    def get_search_limit_max(self) -> int:
        return self._config.get("performance", {}).get("search_limit_max", 1000)

    # データ取得設定
    def should_enable_retry(self) -> bool:
        return self._config.get("data_fetching", {}).get("enable_retry", True)

    def get_retry_attempts(self) -> int:
        return self._config.get("data_fetching", {}).get("retry_attempts", 3)

    def get_retry_delay(self) -> float:
        return self._config.get("data_fetching", {}).get("retry_delay", 1.0)

    def should_enable_caching(self) -> bool:
        return self._config.get("data_fetching", {}).get("enable_caching", True)

    def get_cache_ttl_seconds(self) -> int:
        return self._config.get("data_fetching", {}).get("cache_ttl_seconds", 3600)

    # 市場推定設定
    def should_use_market_cap_estimation(self) -> bool:
        return self._config.get("market_estimation", {}).get("use_market_cap_estimation", True)

    def get_prime_market_cap_threshold(self) -> int:
        return self._config.get("market_estimation", {}).get("prime_market_cap_threshold", 100_000_000_000)

    def get_standard_market_cap_threshold(self) -> int:
        return self._config.get("market_estimation", {}).get("standard_market_cap_threshold", 10_000_000_000)

    def get_default_market_segment(self) -> str:
        return self._config.get("market_estimation", {}).get("default_market_segment", "東証プライム")

    def get_etf_code_ranges(self) -> list:
        return self._config.get("market_estimation", {}).get("etf_code_ranges", [
            {"start": 1300, "end": 1399},
            {"start": 1500, "end": 1599}
        ])

    def get_growth_code_ranges(self) -> list:
        return self._config.get("market_estimation", {}).get("growth_code_ranges", [
            {"start": 2000, "end": 2999}
        ])

    # バリデーション設定
    def should_require_code(self) -> bool:
        return self._config.get("validation", {}).get("require_code", True)

    def should_require_name(self) -> bool:
        return self._config.get("validation", {}).get("require_name", True)

    def should_validate_code_format(self) -> bool:
        return self._config.get("validation", {}).get("validate_code_format", True)

    def get_max_name_length(self) -> int:
        return self._config.get("validation", {}).get("max_name_length", 255)

    def get_allowed_markets(self) -> list:
        return self._config.get("validation", {}).get("allowed_markets", [
            "東証プライム", "東証スタンダード", "東証グロース", "ETF"
        ])

    # ログ設定
    def should_log_bulk_operations(self) -> bool:
        return self._config.get("logging", {}).get("log_bulk_operations", True)

    def should_log_performance_metrics(self) -> bool:
        return self._config.get("logging", {}).get("log_performance_metrics", True)

    def should_log_cache_operations(self) -> bool:
        return self._config.get("logging", {}).get("log_cache_operations", False)

    def should_use_detailed_error_logging(self) -> bool:
        return self._config.get("logging", {}).get("detailed_error_logging", True)

    # セクター管理設定
    def should_skip_existing_sector_info(self) -> bool:
        return self._config.get("sector_management", {}).get("skip_existing_sector_info", True)

    def should_auto_update_on_fetch(self) -> bool:
        return self._config.get("sector_management", {}).get("auto_update_on_fetch", False)

    def get_sector_cache_ttl_hours(self) -> int:
        return self._config.get("sector_management", {}).get("sector_cache_ttl_hours", 24)

    def get_max_concurrent_updates(self) -> int:
        return self._config.get("sector_management", {}).get("max_concurrent_updates", 5)

    def get_allowed_sectors(self) -> list:
        return self._config.get("validation", {}).get("allowed_sectors", [
            "テクノロジー", "金融", "ヘルスケア", "消費財", "エネルギー",
            "不動産", "素材", "資本財", "通信", "公共サービス"
        ])

    # 設定更新・保存
    def update_config(self, new_config: Dict[str, Any]):
        """設定を更新"""
        self._config.update(new_config)

    def save_config(self):
        """設定をファイルに保存"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            logger.info(f"設定を保存: {self.config_path}")
        except Exception as e:
            logger.error(f"設定保存エラー: {e}")

    def get_config_dict(self) -> Dict[str, Any]:
        """設定辞書を取得（デバッグ用）"""
        return self._config.copy()


# グローバル設定インスタンス
_global_config = None


def get_stock_master_config() -> StockMasterConfig:
    """グローバル設定インスタンスを取得"""
    global _global_config
    if _global_config is None:
        _global_config = StockMasterConfig()
    return _global_config


def set_stock_master_config(config: StockMasterConfig):
    """グローバル設定インスタンスを設定"""
    global _global_config
    _global_config = config
