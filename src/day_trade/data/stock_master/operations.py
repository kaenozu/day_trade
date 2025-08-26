"""
銘柄マスタの基本CRUD操作モジュール

このモジュールは銘柄の基本的な作成、読み取り、更新、削除操作を提供します。
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockOperations:
    """銘柄の基本CRUD操作クラス"""

    def __init__(self, db_manager):
        """
        初期化

        Args:
            db_manager: データベースマネージャー
        """
        self.db_manager = db_manager

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
        if session:
            return self._add_stock_with_session(
                session, code, name, market, sector, industry
            )

        with self.db_manager.session_scope() as session:
            return self._add_stock_with_session(
                session, code, name, market, sector, industry
            )

    def _add_stock_with_session(
        self,
        session: Session,
        code: str,
        name: str,
        market: Optional[str],
        sector: Optional[str],
        industry: Optional[str],
    ) -> Optional[Stock]:
        """セッション付きで銘柄を追加（内部メソッド）"""
        try:
            # 既存チェック
            existing = session.query(Stock).filter(Stock.code == code).first()
            if existing:
                logger.info(f"銘柄は既に存在します: {code} - {existing.name}")
                # 属性を事前読み込みしてからreturn
                self._preload_stock_attributes(existing)
                return existing

            # 新規作成
            stock = Stock(
                code=code, name=name, market=market, sector=sector, industry=industry
            )
            session.add(stock)
            session.flush()  # IDを取得

            logger.info(f"銘柄を追加しました: {code} - {name}")
            # 属性を事前読み込みしてからreturn
            self._preload_stock_attributes(stock)
            return stock

        except Exception as e:
            logger.error(f"銘柄追加エラー ({code}): {e}")
            return None

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
        with self.db_manager.session_scope() as session:
            try:
                stock = session.query(Stock).filter(Stock.code == code).first()
                if not stock:
                    logger.warning(f"銘柄が見つかりません: {code}")
                    return None

                # 更新
                if name is not None:
                    stock.name = name
                if market is not None:
                    stock.market = market
                if sector is not None:
                    stock.sector = sector
                if industry is not None:
                    stock.industry = industry

                # 属性を事前に読み込み（セッションスコープ内で遅延読み込み解決）
                self._preload_stock_attributes(stock)

                logger.info(f"銘柄を更新しました: {code} - {stock.name}")
                return stock

            except Exception as e:
                logger.error(f"銘柄更新エラー ({code}): {e}")
                return None

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
        with self.db_manager.session_scope() as session:
            try:
                stock = session.query(Stock).filter(Stock.code == code).first()
                if stock:
                    # セッションスコープを抜ける前に属性をアクセスして遅延読み込みを解決
                    self._preload_stock_attributes(stock)
                    logger.debug(f"銘柄取得: {stock.code} - {stock.name}")
                return stock
            except Exception as e:
                logger.error(f"銘柄取得エラー ({code}): {e}")
                return None

    def delete_stock(self, code: str) -> bool:
        """
        銘柄を削除

        Args:
            code: 証券コード

        Returns:
            削除成功フラグ
        """
        with self.db_manager.session_scope() as session:
            try:
                stock = session.query(Stock).filter(Stock.code == code).first()
                if not stock:
                    logger.warning(f"削除対象の銘柄が見つかりません: {code}")
                    return False

                session.delete(stock)
                logger.info(f"銘柄を削除しました: {code} - {stock.name}")
                return True

            except Exception as e:
                logger.error(f"銘柄削除エラー ({code}): {e}")
                return False

    def get_stock_count(self) -> int:
        """
        登録銘柄数を取得

        Returns:
            銘柄数
        """
        with self.db_manager.session_scope() as session:
            try:
                return session.query(Stock).count()
            except Exception as e:
                logger.error(f"銘柄数取得エラー: {e}")
                return 0

    def get_all_sectors(self) -> List[str]:
        """
        全セクターリストを取得

        Returns:
            セクター名のリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                result = (
                    session.query(Stock.sector)
                    .distinct()
                    .filter(Stock.sector.isnot(None))
                    .all()
                )
                return [r.sector for r in result]

            except Exception as e:
                logger.error(f"セクター取得エラー: {e}")
                return []

    def get_all_industries(self) -> List[str]:
        """
        全業種リストを取得

        Returns:
            業種名のリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                result = (
                    session.query(Stock.industry)
                    .distinct()
                    .filter(Stock.industry.isnot(None))
                    .all()
                )
                return [r.industry for r in result]

            except Exception as e:
                logger.error(f"業種取得エラー: {e}")
                return []

    def get_all_markets(self) -> List[str]:
        """
        全市場区分リストを取得

        Returns:
            市場区分のリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                result = (
                    session.query(Stock.market)
                    .distinct()
                    .filter(Stock.market.isnot(None))
                    .all()
                )
                return [r.market for r in result]

            except Exception as e:
                logger.error(f"市場区分取得エラー: {e}")
                return []

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

    def _apply_session_management(self, stock: Stock, session, detached: bool):
        """セッション管理の適用"""
        # eager loadingで属性を事前読み込み
        self._preload_stock_attributes(stock)

        # detachedが指定されている場合はセッションから切り離す
        if detached:
            session.expunge(stock)