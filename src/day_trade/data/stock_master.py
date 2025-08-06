"""
銘柄マスタ管理モジュール
東証上場銘柄の情報を管理し、検索機能を提供する
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.bulk_operations import AdvancedBulkOperations
from ..models.database import db_manager
from ..models.stock import Stock
from ..utils.logging_config import get_context_logger
from .stock_fetcher import StockFetcher
from .stock_master_config import get_stock_master_config

logger = get_context_logger(__name__)


class StockMasterManager:
    """銘柄マスタ管理クラス（改善版）"""

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
            from ..models.database import db_manager as default_db_manager

            self.db_manager = default_db_manager
        else:
            self.db_manager = db_manager
        self.bulk_operations = AdvancedBulkOperations(self.db_manager)
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.config = config or get_stock_master_config()

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
                _ = (
                    existing.id,
                    existing.code,
                    existing.name,
                    existing.market,
                    existing.sector,
                    existing.industry,
                )
                return existing

            # 新規作成
            stock = Stock(
                code=code, name=name, market=market, sector=sector, industry=industry
            )
            session.add(stock)
            session.flush()  # IDを取得

            logger.info(f"銘柄を追加しました: {code} - {name}")
            # 属性を事前読み込みしてからreturn
            _ = (
                stock.id,
                stock.code,
                stock.name,
                stock.market,
                stock.sector,
                stock.industry,
            )
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
                _ = (
                    stock.id,
                    stock.code,
                    stock.name,
                    stock.market,
                    stock.sector,
                    stock.industry,
                )

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
                    _ = (
                        stock.id,
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    logger.debug(f"銘柄取得: {stock.code} - {stock.name}")
                return stock
            except Exception as e:
                logger.error(f"銘柄取得エラー ({code}): {e}")
                return None

    def _validate_stock_data(self, code: str, name: str = None) -> bool:
        """銀柄データのバリデーション"""
        if self.config.should_require_code() and not code:
            return False
        if self.config.should_require_name() and not name:
            return False
        if self.config.should_validate_code_format() and not code.isdigit():
            return False
        if name and len(name) > self.config.get_max_name_length():
            return False
        return True

    def _apply_session_management(self, stock: Stock, session, detached: bool):
        """セッション管理の適用"""
        # eager loadingで属性を事前読み込み
        if self.config.should_use_eager_loading():
            _ = (
                stock.id,
                stock.code,
                stock.name,
                stock.market,
                stock.sector,
                stock.industry,
            )

        # detachedが指定されている場合はセッションから切り離す
        if detached and self.config.should_auto_expunge():
            session.expunge(stock)

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
                    _ = (
                        stock.id,
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )

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
                    _ = (
                        stock.id,
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )

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
                    _ = (
                        stock.id,
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )

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
                    _ = (
                        stock.id,
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )

                return stocks

            except Exception as e:
                logger.error(f"銘柄複合検索エラー: {e}")
                return []

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

    def fetch_and_update_stock_info(self, code: str) -> Optional[Stock]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（最適化版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクト
        """
        try:
            # StockFetcherのget_company_infoメソッドを使用（リトライ、キャッシュの恩恵を受ける）
            company_info = self.stock_fetcher.get_company_info(code)

            if not company_info:
                logger.warning(f"StockFetcherから企業情報を取得できません: {code}")
                return None

            # データを整理
            name = company_info.get("name") or ""
            sector = company_info.get("sector") or ""
            industry = company_info.get("industry") or ""

            # 市場区分を推定（改善版）
            market = self._estimate_market_segment(code, company_info)

            # 単一のセッションスコープ内で全処理を実行
            with self.db_manager.session_scope() as session:
                # 既存銘柄をチェック
                existing_stock = session.query(Stock).filter(Stock.code == code).first()

                if existing_stock:
                    logger.info(f"銘柄情報を更新: {code} - {name}")
                    # 既存銘柄を更新
                    existing_stock.name = name
                    existing_stock.market = market
                    existing_stock.sector = sector
                    existing_stock.industry = industry
                    session.flush()

                    # 属性を事前読み込み（セッション内で）
                    _ = (
                        existing_stock.code,
                        existing_stock.name,
                        existing_stock.market,
                        existing_stock.sector,
                        existing_stock.industry,
                    )
                    return existing_stock
                else:
                    logger.info(f"新規銘柄を追加: {code} - {name}")
                    # 新規銘柄を作成
                    new_stock = Stock(
                        code=code,
                        name=name,
                        market=market,
                        sector=sector,
                        industry=industry,
                    )
                    session.add(new_stock)
                    session.flush()

                    # 属性を事前読み込み（セッション内で）
                    _ = (
                        new_stock.code,
                        new_stock.name,
                        new_stock.market,
                        new_stock.sector,
                        new_stock.industry,
                    )
                    return new_stock

        except Exception as e:
            logger.error(f"銘柄情報取得・更新エラー ({code}): {e}")
            return None

    def fetch_and_update_stock_info_as_dict(self, code: str) -> Optional[Dict]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクトの辞書表現
        """
        try:
            # StockFetcherのget_company_infoメソッドを使用（リトライ、キャッシュの恩恵を受ける）
            company_info = self.stock_fetcher.get_company_info(code)

            if not company_info:
                logger.warning(f"StockFetcherから企業情報を取得できません: {code}")
                return None

            # データを整理
            name = company_info.get("name") or ""
            sector = company_info.get("sector") or ""
            industry = company_info.get("industry") or ""

            # 市場区分を推定（改善版）
            market = self._estimate_market_segment(code, company_info)

            # セッションスコープ内で更新・作成し、結果を辞書で返却
            with self.db_manager.session_scope() as session:
                # 既存銘柄を更新または新規作成
                stock = session.query(Stock).filter(Stock.code == code).first()
                if stock:
                    logger.info(f"銘柄情報を更新: {code} - {name}")
                    stock.name = name
                    stock.market = market
                    stock.sector = sector
                    stock.industry = industry
                    session.flush()
                else:
                    logger.info(f"新規銘柄を追加: {code} - {name}")
                    stock = Stock(
                        code=code,
                        name=name,
                        market=market,
                        sector=sector,
                        industry=industry,
                    )
                    session.add(stock)
                    session.flush()

                # BaseModelのto_dictメソッドを使用して辞書として返却
                return stock.to_dict()

        except Exception as e:
            logger.error(f"銘柄情報取得・更新エラー ({code}): {e}")
            return None

    def _estimate_market_segment(self, code: str, company_info: Dict) -> str:
        """
        市場区分を推定（堅牢性向上版）

        Args:
            code: 証券コード
            company_info: 企業情報

        Returns:
            推定された市場区分
        """
        try:
            # コードレンジに基づいた推定（簡単なルール）
            code_num = int(code)
            market_cap = company_info.get("market_cap")

            # ETFや特殊なコードの判定
            if 1300 <= code_num <= 1399 or 1500 <= code_num <= 1599:
                return "ETF"
            elif 2000 <= code_num <= 2999:
                return "東証グロース"  # 新興企業が多いレンジ
            elif code_num >= 9000:
                return "東証スタンダード"  # 高番台はスタンダードが多い

            # 時価総額に基づいた推定（おおよその基準）
            if market_cap:
                if (
                    market_cap > 1_000_000_000_000 or market_cap > 100_000_000_000
                ):  # 1兆ドル超
                    return "東証プライム"
                elif market_cap > 10_000_000_000:  # 100億ドル超
                    return "東証スタンダード"
                else:
                    return "東証グロース"

            # デフォルト（企業サイズが不明な場合）
            if code_num <= 1999:
                return "東証プライム"  # 1000番台はプライムが多い
            else:
                return "東証スタンダード"  # その他はスタンダードをデフォルト

        except (ValueError, TypeError):
            # コードが数値でない場合のフォールバック
            return "東証プライム"

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
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                if not stock_data.get("code") or not stock_data.get("name"):
                    logger.warning(f"無効な銘柄データをスキップ: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": stock_data["code"],
                        "name": stock_data["name"],
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用して一括挿入
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="ignore",  # 重複は無視
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括追加完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括追加エラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

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
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                code = stock_data.get("code")
                if not code:
                    logger.warning(f"銘柄コードが無効です: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": code,
                        "name": stock_data.get("name"),
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用してupsert（挿入または更新）
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括更新完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括更新エラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

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
        if not stocks_data:
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": 0}

        try:
            # データの検証と準備
            validated_data = []
            for stock_data in stocks_data:
                code = stock_data.get("code")
                if not code:
                    logger.warning(f"銘柄コードが無効です: {stock_data}")
                    continue
                validated_data.append(
                    {
                        "code": code,
                        "name": stock_data.get("name"),
                        "market": stock_data.get("market"),
                        "sector": stock_data.get("sector"),
                        "industry": stock_data.get("industry"),
                    }
                )

            # AdvancedBulkOperationsを使用してupsert
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"],
            )

            logger.info(f"銘柄一括upsert完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括upsertエラー: {e}")
            return {
                "inserted": 0,
                "updated": 0,
                "skipped": 0,
                "errors": len(stocks_data),
            }

    def fetch_and_update_stock_info_dict(self, code: str) -> Optional[Dict[str, str]]:
        """
        StockFetcherを使用して銘柄情報を取得し、マスタを更新（辞書返却版）

        Args:
            code: 証券コード

        Returns:
            更新された銘柄情報の辞書
        """
        try:
            # StockFetcherのget_company_infoメソッドを使用
            company_info = self.stock_fetcher.get_company_info(code)

            if not company_info:
                logger.warning(f"StockFetcherから企業情報を取得できません: {code}")
                return None

            # データを整理
            name = company_info.get("name") or ""
            sector = company_info.get("sector") or ""
            industry = company_info.get("industry") or ""
            market = self._estimate_market_segment(code, company_info)

            # 単一セッション内で処理
            with self.db_manager.session_scope() as session:
                # 既存銘柄をチェック
                existing_stock = session.query(Stock).filter(Stock.code == code).first()

                if existing_stock:
                    logger.info(f"銘柄情報を更新: {code} - {name}")
                    existing_stock.name = name
                    existing_stock.market = market
                    existing_stock.sector = sector
                    existing_stock.industry = industry
                    session.flush()

                    # 辞書として返却
                    return {
                        "code": existing_stock.code,
                        "name": existing_stock.name,
                        "market": existing_stock.market,
                        "sector": existing_stock.sector,
                        "industry": existing_stock.industry,
                    }
                else:
                    logger.info(f"新規銘柄を追加: {code} - {name}")
                    new_stock = Stock(
                        code=code,
                        name=name,
                        market=market,
                        sector=sector,
                        industry=industry,
                    )
                    session.add(new_stock)
                    session.flush()

                    # 辞書として返却
                    return {
                        "code": new_stock.code,
                        "name": new_stock.name,
                        "market": new_stock.market,
                        "sector": new_stock.sector,
                        "industry": new_stock.industry,
                    }

        except Exception as e:
            logger.error(f"銘柄情報取得・更新エラー ({code}): {e}")
            return None

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
        if not codes:
            return {"success": 0, "failed": 0, "skipped": 0, "total": 0}

        logger.info(f"企業情報一括取得開始: {len(codes)}銘柄")

        success_count = 0
        failed_count = 0
        skipped_count = 0
        updated_stocks = []

        # StockFetcherの一括取得機能を使用（改善版）
        import time

        start_time = time.time()

        try:
            # 新しい一括取得機能を使用
            bulk_company_data = self.stock_fetcher.bulk_get_company_info(
                codes=codes, batch_size=batch_size, delay=delay
            )

            # 取得結果を処理してデータベース更新用のデータを準備
            for code, company_info in bulk_company_data.items():
                try:
                    if company_info:
                        # 既存のStockレコードを更新または新規作成
                        with self.db_manager.session_scope() as session:
                            stock = (
                                session.query(Stock).filter(Stock.code == code).first()
                            )

                            if stock:
                                # 既存レコードを更新
                                stock.name = company_info.get("name", stock.name)
                                stock.sector = company_info.get("sector", stock.sector)
                                stock.industry = company_info.get(
                                    "industry", stock.industry
                                )
                                logger.debug(f"銘柄情報更新: {code} - {stock.name}")
                            else:
                                # 新規レコードを作成
                                stock = Stock(
                                    code=code,
                                    name=company_info.get("name", ""),
                                    market="東証プライム",  # デフォルト値
                                    sector=company_info.get("sector", ""),
                                    industry=company_info.get("industry", ""),
                                )
                                session.add(stock)
                                logger.debug(f"新規銘柄登録: {code} - {stock.name}")

                            session.commit()

                            # 更新されたデータを記録
                            updated_stocks.append(
                                {
                                    "code": stock.code,
                                    "name": stock.name,
                                    "market": stock.market,
                                    "sector": stock.sector,
                                    "industry": stock.industry,
                                }
                            )
                            success_count += 1
                    else:
                        skipped_count += 1
                        logger.warning(f"企業情報が取得できませんでした: {code}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"銘柄情報処理エラー ({code}): {e}")

        except Exception as e:
            logger.error(f"一括企業情報取得エラー: {e}")
            # フォールバック: 従来の個別処理
            logger.warning("個別処理にフォールバック")

            for i in range(0, len(codes), batch_size):
                batch_codes = codes[i : i + batch_size]
                logger.info(
                    f"フォールバック バッチ処理: {i//batch_size + 1}/{(len(codes) + batch_size - 1)//batch_size}"
                )

                for code in batch_codes:
                    try:
                        stock_info = self.fetch_and_update_stock_info_as_dict(code)
                        if stock_info:
                            updated_stocks.append(stock_info)
                            success_count += 1
                        else:
                            skipped_count += 1

                    except Exception as e:
                        logger.error(f"銘柄情報取得失敗 ({code}): {e}")
                        failed_count += 1

                # バッチ間の遅延
                if i + batch_size < len(codes) and delay > 0:
                    time.sleep(delay)

        total_elapsed = time.time() - start_time
        avg_time_per_stock = total_elapsed / len(codes) if codes else 0

        result = {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": len(codes),
            "elapsed_seconds": total_elapsed,
            "avg_time_per_stock": avg_time_per_stock,
        }

        logger.info(
            f"企業情報一括取得完了: 成功={success_count}, 失敗={failed_count}, "
            f"スキップ={skipped_count}, 合計={len(codes)} "
            f"({total_elapsed:.2f}秒, 平均{avg_time_per_stock:.3f}秒/銘柄)"
        )
        return result

    def update_sector_information_bulk(
        self, codes: List[str], batch_size: int = 20, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        複数銀柄のセクター情報を一括更新（Issue #133対応）

        Args:
            codes: 銀柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            更新結果の統計情報
        """
        if not codes:
            return {"updated": 0, "failed": 0, "skipped": 0, "total": 0}

        logger.info(f"セクター情報一括更新開始: {len(codes)}銀柄")

        updated_count = 0
        failed_count = 0
        skipped_count = 0

        # バッチ処理でAPIレートリミットを回避
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i : i + batch_size]
            logger.info(
                f"セクター情報バッチ処理: {i//batch_size + 1}/{(len(codes) + batch_size - 1)//batch_size}"
            )

            with self.db_manager.session_scope() as session:
                for code in batch_codes:
                    try:
                        # 現在の銀柄情報を取得
                        stock = session.query(Stock).filter(Stock.code == code).first()
                        if not stock:
                            logger.warning(f"銀柄が見つかりません: {code}")
                            skipped_count += 1
                            continue

                        # セクター情報が既に存在する場合はスキップ（オプション）
                        if stock.sector and stock.industry:
                            logger.debug(
                                f"セクター情報が既に存在: {code} - {stock.sector}"
                            )
                            skipped_count += 1
                            continue

                        # StockFetcherから企業情報を取得
                        company_info = self.stock_fetcher.get_company_info(code)
                        if not company_info:
                            logger.warning(f"企業情報を取得できません: {code}")
                            failed_count += 1
                            continue

                        # セクター情報を更新
                        updated = False
                        if (
                            company_info.get("sector")
                            and company_info["sector"] != stock.sector
                        ):
                            stock.sector = company_info["sector"]
                            updated = True

                        if (
                            company_info.get("industry")
                            and company_info["industry"] != stock.industry
                        ):
                            stock.industry = company_info["industry"]
                            updated = True

                        if updated:
                            session.flush()
                            logger.info(
                                f"セクター情報を更新: {code} - {stock.sector}/{stock.industry}"
                            )
                            updated_count += 1
                        else:
                            skipped_count += 1

                    except Exception as e:
                        logger.error(f"セクター情報更新エラー ({code}): {e}")
                        failed_count += 1

            # バッチ間の遅延
            if i + batch_size < len(codes) and delay > 0:
                import time

                time.sleep(delay)

        result = {
            "updated": updated_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": len(codes),
        }

        logger.info(f"セクター情報一括更新完了: {result}")
        return result

    def get_stocks_without_sector_info(self, limit: int = 1000) -> List[str]:
        """
        セクター情報が空の銀柄コードを取得

        Args:
            limit: 結果の上限数

        Returns:
            セクター情報が空の銀柄コードのリスト
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
                logger.info(f"セクター情報が空の銀柄: {len(codes)}件")
                return codes

            except Exception as e:
                logger.error(f"セクター情報の空銀柄取得エラー: {e}")
                return []

    def auto_update_missing_sector_info(self, max_stocks: int = 100) -> Dict[str, int]:
        """
        セクター情報が空の銀柄を自動的に更新（ユーティリティ）

        Args:
            max_stocks: 一度に処理する銀柄の上限数

        Returns:
            更新結果の統計情報
        """
        logger.info(f"セクター情報の自動更新を開始: 上限{max_stocks}銀柄")

        # セクター情報が空の銀柄を取得
        codes_to_update = self.get_stocks_without_sector_info(limit=max_stocks)

        if not codes_to_update:
            logger.info("セクター情報の更新が必要な銀柄はありません")
            return {"updated": 0, "failed": 0, "skipped": 0, "total": 0}

        # 一括更新を実行
        return self.update_sector_information_bulk(
            codes_to_update,
            batch_size=self.config.get_fetch_batch_size(),
            delay=self.config.get_fetch_delay_seconds(),
        )

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


# Issue #133: セクター情報永続化のユーティリティ関数
def update_all_sector_information(
    batch_size: int = 20, max_stocks: int = 1000
) -> Dict[str, int]:
    """
    全銀柄のセクター情報を更新するユーティリティ関数

    Args:
        batch_size: バッチサイズ
        max_stocks: 一度に処理する銀柄の上限数

    Returns:
        更新結果の統計情報
    """
    return stock_master.auto_update_missing_sector_info(max_stocks=max_stocks)


def get_sector_distribution() -> Dict[str, int]:
    """
    セクター別銀柄数の分布を取得

    Returns:
        セクター名: 銀柄数 の辞書
    """
    from sqlalchemy import func

    from ..models.stock import Stock

    try:
        with db_manager.session_scope() as session:
            # セクター別の銀柄数を集計
            results = (
                session.query(Stock.sector, func.count(Stock.id))
                .filter(Stock.sector.isnot(None))
                .filter(Stock.sector != "")
                .group_by(Stock.sector)
                .order_by(func.count(Stock.id).desc())
                .all()
            )

            distribution = {sector: count for sector, count in results}

            # セクター情報が空の銀柄数を追加
            missing_count = (
                session.query(func.count(Stock.id))
                .filter((Stock.sector.is_(None)) | (Stock.sector == ""))
                .scalar()
            )

            if missing_count > 0:
                distribution["(セクター情報なし)"] = missing_count

            return distribution

    except Exception as e:
        logger.error(f"セクター分布取得エラー: {e}")
        return {}

    def bulk_fetch_and_update_prices_optimized(
        self, codes: List[str], batch_size: int = 50, delay: float = 0.1
    ) -> Dict[str, int]:
        """
        最適化された一括価格取得・更新（bulk_get_current_prices_optimized使用）

        Args:
            codes: 銘柄コードのリスト
            batch_size: バッチサイズ（APIレートリミット対応）
            delay: バッチ間の遅延（秒）

        Returns:
            処理結果統計
        """
        if not codes:
            return {"total": 0, "updated": 0, "failed": 0}

        logger.info(f"最適化一括価格更新開始: {len(codes)}件 (batch_size={batch_size})")
        import time as time_module
        from datetime import datetime as dt

        start_time = time_module.time()

        # 制限適用
        effective_limit = self._apply_stock_limit(len(codes))
        if effective_limit < len(codes):
            codes = codes[:effective_limit]
            logger.info(f"銘柄制限を適用: {len(codes)}件に制限")

        try:
            # 最適化されたバルク価格取得
            price_data = self.fetcher.bulk_get_current_prices_optimized(
                codes, batch_size=batch_size, delay=delay
            )

            # データベース更新用のデータを準備
            update_data = []
            for code, data in price_data.items():
                if data is not None:
                    update_data.append(
                        {
                            "code": code,
                            "current_price": data.get("current_price"),
                            "change": data.get("change"),
                            "change_percent": data.get("change_percent"),
                            "volume": data.get("volume"),
                            "high": data.get("high"),
                            "low": data.get("low"),
                            "open_price": data.get("open"),
                            "previous_close": data.get("previous_close"),
                            "last_updated": dt.now(),
                        }
                    )

            # SQLAlchemyバルク更新
            result = {"total": len(codes), "updated": 0, "failed": 0}

            with self.db_manager.session_scope() as session:
                try:
                    # バルクアップデート実行
                    for update_info in update_data:
                        code = update_info["code"]

                        # 既存株式レコードを取得
                        stock = session.query(Stock).filter_by(code=code).first()
                        if stock:
                            # 価格情報を更新
                            if update_info.get("current_price") is not None:
                                stock.current_price = update_info["current_price"]
                            if update_info.get("change") is not None:
                                stock.change = update_info["change"]
                            if update_info.get("change_percent") is not None:
                                stock.change_percent = update_info["change_percent"]
                            if update_info.get("volume") is not None:
                                stock.volume = update_info["volume"]
                            if update_info.get("high") is not None:
                                stock.high = update_info["high"]
                            if update_info.get("low") is not None:
                                stock.low = update_info["low"]
                            if update_info.get("open_price") is not None:
                                stock.open_price = update_info["open_price"]
                            if update_info.get("previous_close") is not None:
                                stock.previous_close = update_info["previous_close"]

                            stock.last_updated = update_info["last_updated"]
                            result["updated"] += 1
                        else:
                            result["failed"] += 1

                    session.commit()

                except Exception as e:
                    logger.error(f"データベース更新エラー: {e}")
                    session.rollback()
                    result["failed"] = len(codes)
                    result["updated"] = 0

            total_elapsed = time_module.time() - start_time

            logger.info(
                f"最適化一括価格更新完了: "
                f"成功{result['updated']}/{result['total']}件 "
                f"({total_elapsed:.2f}秒, {result['updated']/total_elapsed:.1f}件/秒)"
            )

            return result

        except Exception as e:
            logger.error(f"最適化一括価格更新エラー: {e}")
            return {"total": len(codes), "updated": 0, "failed": len(codes)}
