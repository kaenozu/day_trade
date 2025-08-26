"""
銘柄マスタのデータ取得・更新機能モジュール（統合版）

このモジュールはStockFetcherと連携して銘柄情報の取得と更新を行います。
単体取得、複数取得、セクター情報更新等の機能を統合して提供します。
"""

from typing import Dict, List, Optional

from ...utils.logging_config import get_context_logger
from .fetching_bulk import StockDataFetcherBulk
from .fetching_core import StockDataFetcherCore

logger = get_context_logger(__name__)


class StockDataFetcher:
    """銘柄データ取得・更新クラス（統合版）"""

    def __init__(self, db_manager, stock_fetcher, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            stock_fetcher: StockFetcherインスタンス
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.stock_fetcher = stock_fetcher
        self.config = config or {}
        
        # 各機能モジュールを初期化
        self.core = StockDataFetcherCore(db_manager, stock_fetcher, config)
        self.bulk = StockDataFetcherBulk(db_manager, stock_fetcher, config)

    # ==========================================================================
    # 基本機能（StockDataFetcherCoreに委譲）
    # ==========================================================================

    def fetch_and_update_stock_info(self, code: str) -> Optional[Dict]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（最適化版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクト
        """
        return self.core.fetch_and_update_stock_info(code)

    def fetch_and_update_stock_info_as_dict(self, code: str) -> Optional[Dict]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクトの辞書表現
        """
        return self.core.fetch_and_update_stock_info_as_dict(code)

    def fetch_and_update_stock_info_dict(self, code: str) -> Optional[Dict[str, str]]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新された銘柄情報の辞書
        """
        return self.core.fetch_and_update_stock_info_dict(code)

    # ==========================================================================
    # 一括処理機能（StockDataFetcherBulkに委譲）
    # ==========================================================================

    def bulk_fetch_and_update_companies(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        複数銘柄の企業情報を一括取得・更新（StockFetcher経由）

        Args:
            codes: 銘柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            更新結果の統計情報
        """
        return self.bulk.bulk_fetch_and_update_companies(codes, batch_size, delay)

    def update_sector_information_bulk(
        self, codes: List[str], batch_size: int = 20, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        複数銘柄のセクター情報を一括更新（Issue #133対応）

        Args:
            codes: 銘柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            更新結果の統計情報
        """
        return self.bulk.update_sector_information_bulk(codes, batch_size, delay)

    def auto_update_missing_sector_info(self, max_stocks: int = 100) -> Dict[str, int]:
        """
        セクター情報が空の銘柄を自動的に更新（ユーティリティ）

        Args:
            max_stocks: 一度に処理する銘柄の上限数

        Returns:
            更新結果の統計情報
        """
        return self.bulk.auto_update_missing_sector_info(max_stocks)