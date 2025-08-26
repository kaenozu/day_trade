"""
銘柄マスタの検索機能モジュール

このモジュールは銘柄の様々な検索機能を提供します。
名前検索、セクター検索、業種検索、複合条件検索等を実装しています。
"""

from typing import List, Optional

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockSearcher:
    """銘柄検索機能クラス"""

    def __init__(self, db_manager, config=None):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
            config: 設定オブジェクト
        """
        self.db_manager = db_manager
        self.config = config or {}

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
        with self.db_manager.session_scope() as session:
            try:
                # 部分一致検索（大文字小文字区別なし）
                pattern = f"%{name_pattern}%"
                effective_limit = self._apply_stock_limit(limit)
                stocks = (
                    session.query(Stock)
                    .filter(Stock.name.ilike(pattern))
                    .limit(effective_limit)
                    .all()
                )

                # セッションスコープを抜ける前に属性をアクセスして遅延読み込みを解決
                for stock in stocks:
                    self._preload_stock_attributes(stock)

                logger.debug(
                    f"銘柄名検索結果: {len(stocks)}件 (パターン: {name_pattern})"
                )
                return stocks

            except Exception as e:
                logger.error(f"銘柄名検索エラー ({name_pattern}): {e}")
                return []

    def search_stocks_by_sector(self, sector: str, limit: int = 100) -> List[Stock]:
        """
        セクターで銘柄を検索

        Args:
            sector: セクター名
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                effective_limit = self._apply_stock_limit(limit)
                stocks = (
                    session.query(Stock)
                    .filter(Stock.sector == sector)
                    .limit(effective_limit)
                    .all()
                )

                # 属性を事前に読み込み（セッションスコープ内で遅延読み込み解決）
                for stock in stocks:
                    self._preload_stock_attributes(stock)

                return stocks

            except Exception as e:
                logger.error(f"セクター検索エラー ({sector}): {e}")
                return []

    def search_stocks_by_industry(self, industry: str, limit: int = 100) -> List[Stock]:
        """
        業種で銘柄を検索

        Args:
            industry: 業種名
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                effective_limit = self._apply_stock_limit(limit)
                stocks = (
                    session.query(Stock)
                    .filter(Stock.industry == industry)
                    .limit(effective_limit)
                    .all()
                )

                # 属性を事前に読み込み（セッションスコープ内で遅延読み込み解決）
                for stock in stocks:
                    self._preload_stock_attributes(stock)

                return stocks

            except Exception as e:
                logger.error(f"業種検索エラー ({industry}): {e}")
                return []

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
        with self.db_manager.session_scope() as session:
            try:
                query = session.query(Stock)

                # 条件を追加
                if code:
                    query = query.filter(Stock.code.ilike(f"%{code}%"))
                if name:
                    query = query.filter(Stock.name.ilike(f"%{name}%"))
                if market:
                    query = query.filter(Stock.market == market)
                if sector:
                    query = query.filter(Stock.sector == sector)
                if industry:
                    query = query.filter(Stock.industry == industry)

                # 銘柄制限設定を適用
                effective_limit = self._apply_stock_limit(limit)
                stocks = query.limit(effective_limit).all()

                # 属性を事前に読み込み（セッションスコープ内で遅延読み込み解決）
                for stock in stocks:
                    self._preload_stock_attributes(stock)

                return stocks

            except Exception as e:
                logger.error(f"銘柄複合検索エラー: {e}")
                return []

    def get_stocks_without_sector_info(self, limit: int = 1000) -> List[str]:
        """
        セクター情報が空の銘柄コードを取得

        Args:
            limit: 結果の上限数

        Returns:
            セクター情報が空の銘柄コードのリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                stocks = (
                    session.query(Stock)
                    .filter(
                        (Stock.sector.is_(None))
                        | (Stock.sector == "")
                        | (Stock.industry.is_(None))
                        | (Stock.industry == "")
                    )
                    .limit(limit)
                    .all()
                )

                codes = [stock.code for stock in stocks]
                logger.info(f"セクター情報が空の銘柄: {len(codes)}件")
                return codes

            except Exception as e:
                logger.error(f"セクター情報の空銘柄取得エラー: {e}")
                return []


    def _apply_stock_limit(self, requested_limit: int) -> int:
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
        import os

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

    def _preload_stock_attributes(self, stock: Stock) -> None:
        """
        Stockオブジェクトの属性を事前読み込み

        Args:
            stock: 読み込み対象のStockオブジェクト
        """
        _ = (
            stock.id,
            stock.code,
            stock.name,
            stock.market,
            stock.sector,
            stock.industry,
        )