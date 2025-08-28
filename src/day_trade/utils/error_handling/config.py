"""
エラーハンドラー設定クラス
ConfigManagerとの統合対応
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EnhancedErrorHandlerConfig:
    """エラーハンドラー設定クラス（config_manager統合対応）"""

    def __init__(self, config_manager=None):
        """
        Args:
            config_manager: ConfigManagerインスタンス（依存性注入）
        """
        # デフォルト設定値
        self._defaults = {
            "debug_mode": False,
            "enable_sanitization": True,
            "enable_rich_display": True,
            "log_technical_details": True,
            "max_context_items": 10,
            "max_solution_items": 5,
            "console_width": 120,
            "panel_padding": (1, 2),
            "lock_timeout_seconds": 1.0,
            "enable_performance_logging": True,
        }

        self.config_manager = config_manager
        self._load_config()

    def _load_config(self):
        """設定を読み込み（config_manager優先、環境変数フォールバック）"""
        # config_managerから取得を試行
        error_handler_settings = {}
        if self.config_manager:
            try:
                # ConfigManagerからエラーハンドラー設定を取得
                # ConfigManagerがPydanticモデルになったので、get_error_handler_settings()を呼び出す形に
                error_handler_settings = (
                    self.config_manager.get_error_handler_settings().model_dump()
                )  # model_dump()でdictに変換
                logger.info("Error handler settings loaded from ConfigManager")
            except AttributeError:  # config_managerに該当メソッドがない場合など
                logger.warning(
                    "ConfigManager does not have get_error_handler_settings, using defaults"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load error handler settings from ConfigManager: {e}, using defaults"
                )

        # 設定値の決定（優先度: config_managerからの値 > デフォルト）
        # 環境変数からの読み込みはConfigManagerで一元管理されるためここでは不要
        self.debug_mode = error_handler_settings.get(
            "debug_mode", self._defaults["debug_mode"]
        )
        self.enable_sanitization = error_handler_settings.get(
            "enable_sanitization", self._defaults["enable_sanitization"]
        )
        self.enable_rich_display = error_handler_settings.get(
            "enable_rich_display", self._defaults["enable_rich_display"]
        )
        self.log_technical_details = error_handler_settings.get(
            "log_technical_details", self._defaults["log_technical_details"]
        )
        self.max_context_items = error_handler_settings.get(
            "max_context_items", self._defaults["max_context_items"]
        )
        self.max_solution_items = error_handler_settings.get(
            "max_solution_items", self._defaults["max_solution_items"]
        )
        self.console_width = error_handler_settings.get(
            "console_width", self._defaults["console_width"]
        )
        self.lock_timeout_seconds = error_handler_settings.get(
            "lock_timeout_seconds", self._defaults["lock_timeout_seconds"]
        )
        self.enable_performance_logging = error_handler_settings.get(
            "enable_performance_logging", self._defaults["enable_performance_logging"]
        )

        # パネル設定
        panel_padding = error_handler_settings.get("panel_padding")
        if (
            panel_padding
            and isinstance(panel_padding, (list, tuple))
            and len(panel_padding) == 2
        ):
            self.panel_padding = tuple(panel_padding)
        else:
            self.panel_padding = self._defaults["panel_padding"]

    def _parse_bool(self, config_value, env_value, default_value):
        """設定値をbooleanに変換"""
        # ConfigManagerから直接bool値が来ることを想定。環境変数からの変換ロジックは不要になる
        if config_value is not None:
            return bool(config_value)
        return default_value

    def reload(self):
        """設定を再読み込み"""
        self._load_config()
        logger.info("Error handler configuration reloaded")