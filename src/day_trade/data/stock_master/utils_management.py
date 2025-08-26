"""
銘柄マスタの管理機能モジュール

このモジュールは制限管理、セッション管理等の管理機能を提供します。
"""

import os
from typing import Optional

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockManagementUtils:
    """銘柄データ管理機能クラス"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}

    def apply_stock_limit(self, requested_limit: int) -> int:
        """
        設定に基づいて銘柄数制限を適用

        Args:
            requested_limit: 要求された制限数

        Returns:
            適用する実際の制限数
        """
        # 設定から最大銘柄数制限を取得
        limits_config = self.config.get("limits", {})
        max_stock_count = limits_config.get("max_stock_count")
        max_search_limit = limits_config.get("max_search_limit", 1000)

        # テストモード判定（環境変数またはconfig）
        is_test_mode = (
            os.getenv("TEST_MODE", "false").lower() == "true"
            or os.getenv("PYTEST_CURRENT_TEST") is not None
        )

        if is_test_mode:
            test_limit = limits_config.get("test_mode_limit", 100)
            effective_limit = min(requested_limit, test_limit)
            logger.info(
                f"テストモード: 銘柄数制限を適用 {requested_limit} -> {effective_limit}"
            )
            return effective_limit

        # 通常モードでの制限適用
        effective_limit = min(requested_limit, max_search_limit)

        if max_stock_count is not None:
            effective_limit = min(effective_limit, max_stock_count)
            if effective_limit < requested_limit:
                logger.info(
                    f"銘柄数制限を適用: {requested_limit} -> {effective_limit} "
                    f"(max_stock_count: {max_stock_count})"
                )

        return effective_limit

    def set_stock_limit(self, limit: Optional[int]) -> None:
        """
        動的に銘柄数制限を設定

        Args:
            limit: 設定する制限数（Noneで制限解除）
        """
        if "limits" not in self.config:
            self.config["limits"] = {}

        self.config["limits"]["max_stock_count"] = limit

        if limit is None:
            logger.info("銘柄数制限を解除しました")
        else:
            logger.info(f"銘柄数制限を設定しました: {limit}")

    def get_stock_limit(self) -> Optional[int]:
        """
        現在の銘柄数制限を取得

        Returns:
            現在の制限数（Noneは制限なし）
        """
        return self.config.get("limits", {}).get("max_stock_count")

    def apply_session_management(self, stock: Stock, session, detached: bool) -> None:
        """
        セッション管理の適用

        Args:
            stock: 対象のStockオブジェクト
            session: データベースセッション
            detached: セッションから切り離すかどうか
        """
        # eager loadingで属性を事前読み込み
        if self.config.get("use_eager_loading", True):
            _ = (
                stock.id,
                stock.code,
                stock.name,
                stock.market,
                stock.sector,
                stock.industry,
            )

        # detachedが指定されている場合はセッションから切り離す
        if detached and self.config.get("auto_expunge", False):
            session.expunge(stock)

    def get_memory_usage_info(self) -> dict:
        """
        メモリ使用量の情報を取得

        Returns:
            メモリ使用量の情報
        """
        try:
            import psutil
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2),
            }
            
        except ImportError:
            logger.warning("psutil not available, cannot get memory info")
            return {"error": "psutil not available"}
        except Exception as e:
            logger.error(f"メモリ情報取得エラー: {e}")
            return {"error": str(e)}

    def cleanup_expired_sessions(self) -> int:
        """
        期限切れのセッションをクリーンアップ

        Returns:
            クリーンアップされたセッション数
        """
        try:
            # SQLAlchemyのセッションレジストリをクリア
            from sqlalchemy.orm import close_all_sessions
            
            close_all_sessions()
            logger.info("SQLAlchemyセッションをクリーンアップしました")
            return 1
            
        except Exception as e:
            logger.error(f"セッションクリーンアップエラー: {e}")
            return 0

    def get_configuration_summary(self) -> dict:
        """
        現在の設定状況のサマリーを取得

        Returns:
            設定サマリー
        """
        limits_config = self.config.get("limits", {})
        
        return {
            "max_stock_count": limits_config.get("max_stock_count", "無制限"),
            "max_search_limit": limits_config.get("max_search_limit", 1000),
            "test_mode_limit": limits_config.get("test_mode_limit", 100),
            "use_eager_loading": self.config.get("use_eager_loading", True),
            "auto_expunge": self.config.get("auto_expunge", False),
            "require_code": self.config.get("require_code", True),
            "require_name": self.config.get("require_name", True),
            "validate_code_format": self.config.get("validate_code_format", True),
            "max_name_length": self.config.get("max_name_length", 100),
        }

    def reset_configuration(self) -> None:
        """
        設定をデフォルトにリセット
        """
        default_config = {
            "limits": {
                "max_search_limit": 1000,
                "test_mode_limit": 100,
            },
            "use_eager_loading": True,
            "auto_expunge": False,
            "require_code": True,
            "require_name": True,
            "validate_code_format": True,
            "max_name_length": 100,
        }
        
        self.config.clear()
        self.config.update(default_config)
        logger.info("設定をデフォルト値にリセットしました")

    def is_test_mode(self) -> bool:
        """
        テストモードかどうかを判定

        Returns:
            テストモードの場合True
        """
        return (
            os.getenv("TEST_MODE", "false").lower() == "true"
            or os.getenv("PYTEST_CURRENT_TEST") is not None
            or self.config.get("test_mode", False)
        )