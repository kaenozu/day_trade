"""
銘柄マスタメインマネージャーモジュール

このモジュールは各機能モジュールを統合し、従来のStockMasterManagerの
インターフェースを提供します。
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger
from ..stock_fetcher import StockFetcher
from ..stock_master_config import get_stock_master_config
from .bulk_operations import StockBulkOperations
from .fetching import StockDataFetcher
from .operations import StockOperations
from .search import StockSearcher
from .utils import StockMasterUtils

logger = get_context_logger(__name__)


class StockMasterManager:
    """銘柄マスタ管理クラス（統合版）"""

    def __init__(
        self, db_manager=None, stock_fetcher: Optional[StockFetcher] = None, config=None
    ):
        """
        初期化（依存性注入対応）

        Args:
            db_manager: データベースマネージャー
            stock_fetcher: 株価データ取得インスタンス
            config: 設定インスタンス
        """
        if db_manager is None:
            from ...models.database import db_manager as default_db_manager

            self.db_manager = default_db_manager
        else:
            self.db_manager = db_manager

        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.config = config or get_stock_master_config()

        # 各機能モジュールを初期化
        self.operations = StockOperations(self.db_manager)
        self.searcher = StockSearcher(self.db_manager, self.config)
        self.data_fetcher = StockDataFetcher(self.db_manager, self.stock_fetcher, self.config)
        self.bulk_ops = StockBulkOperations(self.db_manager)
        self.utils = StockMasterUtils(self.db_manager, self.config)

    # ==========================================================================
    # 基本CRUD操作（StockOperationsに委譲）
    # ==========================================================================

    def add_stock(
        self,
        code: str,
        name: str,
        market: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        session: Optional[Session] = None,
    ) -> Optional[Stock]:
        """
        銘柄をマスタに追加

        Args:
            code: 証券コード
            name: 銘柄名
            market: 市場区分
            sector: セクター
            industry: 業種
            session: データベースセッション

        Returns:
            作成されたStockオブジェクト
        """
        return self.operations.add_stock(code, name, market, sector, industry, session)

    def update_stock(
        self,
        code: str,
        name: Optional[str] = None,
        market: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> Optional[Stock]:
        """
        銘柄情報を更新

        Args:
            code: 証券コード
            name: 銘柄名
            market: 市場区分
            sector: セクター
            industry: 業種

        Returns:
            更新されたStockオブジェクト
        """
        return self.operations.update_stock(code, name, market, sector, industry)

    def get_stock_by_code(
        self, code: str, detached: Optional[bool] = None
    ) -> Optional[Stock]:
        """
        証券コードで銘柄を取得（最適化版）

        Args:
            code: 証券コード
            detached: セッションから切り離すかどうか

        Returns:
            Stockオブジェクト
        """
        return self.operations.get_stock_by_code(code, detached)

    def delete_stock(self, code: str) -> bool:
        """
        銘柄を削除

        Args:
            code: 証券コード

        Returns:
            削除成功フラグ
        """
        return self.operations.delete_stock(code)

    def get_stock_count(self) -> int:
        """
        登録銘柄数を取得

        Returns:
            銘柄数
        """
        return self.operations.get_stock_count()

    def get_all_sectors(self) -> List[str]:
        """
        全セクターリストを取得

        Returns:
            セクター名のリスト
        """
        return self.operations.get_all_sectors()

    def get_all_industries(self) -> List[str]:
        """
        全業種リストを取得

        Returns:
            業種名のリスト
        """
        return self.operations.get_all_industries()

    def get_all_markets(self) -> List[str]:
        """
        全市場区分リストを取得

        Returns:
            市場区分のリスト
        """
        return self.operations.get_all_markets()

    # ==========================================================================
    # 検索機能（StockSearcherに委譲）
    # ==========================================================================

    def search_stocks_by_name(
        self, name_pattern: str, limit: int = 50, detached: bool = False
    ) -> List[Stock]:
        """
        銘柄名で部分一致検索（最適化版）

        Args:
            name_pattern: 銘柄名の一部
            limit: 結果の上限数（設定上限あり）
            detached: セッションから切り離すかどうか

        Returns:
            Stockオブジェクトのリスト
        """
        return self.searcher.search_stocks_by_name(name_pattern, limit, detached)

    def search_stocks_by_sector(self, sector: str, limit: int = 100) -> List[Stock]:
        """
        セクターで銘柄を検索

        Args:
            sector: セクター名
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        return self.searcher.search_stocks_by_sector(sector, limit)

    def search_stocks_by_industry(self, industry: str, limit: int = 100) -> List[Stock]:
        """
        業種で銘柄を検索

        Args:
            industry: 業種名
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        return self.searcher.search_stocks_by_industry(industry, limit)

    def search_stocks(
        self,
        code: Optional[str] = None,
        name: Optional[str] = None,
        market: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        limit: int = 50,
    ) -> List[Stock]:
        """
        複合条件で銘柄を検索

        Args:
            code: 証券コード（部分一致）
            name: 銘柄名（部分一致）
            market: 市場区分（完全一致）
            sector: セクター（完全一致）
            industry: 業種（完全一致）
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        return self.searcher.search_stocks(code, name, market, sector, industry, limit)

    def get_stocks_without_sector_info(self, limit: int = 1000) -> List[str]:
        """
        セクター情報が空の銘柄コードを取得

        Args:
            limit: 結果の上限数

        Returns:
            セクター情報が空の銘柄コードのリスト
        """
        return self.searcher.get_stocks_without_sector_info(limit)

    # ==========================================================================
    # データ取得・更新機能（StockDataFetcherに委譲）
    # ==========================================================================

    def fetch_and_update_stock_info(self, code: str) -> Optional[Stock]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（最適化版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクト
        """
        return self.data_fetcher.fetch_and_update_stock_info(code)

    def fetch_and_update_stock_info_as_dict(self, code: str) -> Optional[Dict]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクトの辞書表現
        """
        return self.data_fetcher.fetch_and_update_stock_info_as_dict(code)

    def fetch_and_update_stock_info_dict(self, code: str) -> Optional[Dict[str, str]]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新された銘柄情報の辞書
        """
        return self.data_fetcher.fetch_and_update_stock_info_dict(code)

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
        return self.data_fetcher.bulk_fetch_and_update_companies(codes, batch_size, delay)

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
        return self.data_fetcher.update_sector_information_bulk(codes, batch_size, delay)

    def auto_update_missing_sector_info(self, max_stocks: int = 100) -> Dict[str, int]:
        """
        セクター情報が空の銘柄を自動的に更新（ユーティリティ）

        Args:
            max_stocks: 一度に処理する銘柄の上限数

        Returns:
            更新結果の統計情報
        """
        return self.data_fetcher.auto_update_missing_sector_info(max_stocks)

    # ==========================================================================
    # 一括処理機能（StockBulkOperationsに委譲）
    # ==========================================================================

    def bulk_add_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括追加（AdvancedBulkOperations使用・パフォーマンス最適化版）

        Args:
            stocks_data: 銘柄データのリスト
                例: [{'code': '1000', 'name': '株式会社A', 'market': '東証プライム', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            追加結果の統計情報
        """
        return self.bulk_ops.bulk_add_stocks(stocks_data, batch_size)

    def bulk_update_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括更新（AdvancedBulkOperations使用・パフォーマンス最適化版）

        Args:
            stocks_data: 更新する銘柄データのリスト
                例: [{'code': '1000', 'name': '新社名', 'sector': '新セクター', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            更新結果の統計情報
        """
        return self.bulk_ops.bulk_update_stocks(stocks_data, batch_size)

    def bulk_upsert_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        銘柄の一括upsert（AdvancedBulkOperations使用・存在すれば更新、なければ追加）

        Args:
            stocks_data: 銘柄データのリスト
            batch_size: バッチサイズ

        Returns:
            実行結果の統計情報
        """
        return self.bulk_ops.bulk_upsert_stocks(stocks_data, batch_size)

    # ==========================================================================
    # ユーティリティ機能（StockMasterUtilsに委譲）
    # ==========================================================================

    def _validate_stock_data(self, code: str, name: str = None) -> bool:
        """銘柄データのバリデーション"""
        return self.utils.validate_stock_data(code, name)

    def _apply_session_management(self, stock: Stock, session, detached: bool):
        """セッション管理の適用"""
        return self.utils.apply_session_management(stock, session, detached)

    def _apply_stock_limit(self, requested_limit: int) -> int:
        """
        設定に基づいて銘柄数制限を適用

        Args:
            requested_limit: 要求された制限数

        Returns:
            適用する実際の制限数
        """
        return self.utils.apply_stock_limit(requested_limit)

    def set_stock_limit(self, limit: Optional[int]) -> None:
        """
        動的に銘柄数制限を設定

        Args:
            limit: 設定する制限数（Noneで制限解除）
        """
        return self.utils.set_stock_limit(limit)

    def get_stock_limit(self) -> Optional[int]:
        """
        現在の銘柄数制限を取得

        Returns:
            現在の制限数（Noneは制限なし）
        """
        return self.utils.get_stock_limit()

    # ==========================================================================
    # 追加のユーティリティメソッド（後方互換性のため）
    # ==========================================================================
    
    def bulk_fetch_and_update_prices_optimized(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, int]:
        """最適化された一括価格取得・更新（未実装）"""
        logger.warning("bulk_fetch_and_update_prices_optimized は未実装です。")
        return {"total": len(codes), "updated": 0, "failed": len(codes)}


# グローバルインスタンス（改善版）
def create_stock_master_manager(
    db_manager=None, stock_fetcher=None
) -> StockMasterManager:
    """
    StockMasterManagerのファクトリー関数（依存性注入対応）

    Args:
        db_manager: データベースマネージャー
        stock_fetcher: 株価データ取得インスタンス

    Returns:
        StockMasterManagerインスタンス
    """
    return StockMasterManager(db_manager=db_manager, stock_fetcher=stock_fetcher)


# 後方互換性のためのグローバルインスタンス
stock_master = StockMasterManager()