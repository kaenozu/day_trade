"""
銘柄マスタ管理モジュール
東証上場銘柄の情報を管理し、検索機能を提供する
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.bulk_operations import AdvancedBulkOperations
from ..models.stock import Stock
from ..utils.logging_config import get_context_logger
from .stock_fetcher import StockFetcher

logger = get_context_logger(__name__)


class StockMasterManager:
    """銘柄マスタ管理クラス（改善版）"""

    def __init__(self, db_manager=None, stock_fetcher: Optional[StockFetcher] = None):
        """
        初期化（依存性注入対応）

        Args:
            db_manager: データベースマネージャー
            stock_fetcher: 株価データ取得インスタンス
        """
        self.db_manager = db_manager or globals()['db_manager']
        self.bulk_operations = AdvancedBulkOperations(self.db_manager)
        self.stock_fetcher = stock_fetcher or StockFetcher()

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
                _ = existing.id, existing.code, existing.name, existing.market, existing.sector, existing.industry
                return existing

            # 新規作成
            stock = Stock(
                code=code, name=name, market=market, sector=sector, industry=industry
            )
            session.add(stock)
            session.flush()  # IDを取得

            logger.info(f"銘柄を追加しました: {code} - {name}")
            # 属性を事前読み込みしてからreturn
            _ = stock.id, stock.code, stock.name, stock.market, stock.sector, stock.industry
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
                _ = stock.id, stock.code, stock.name, stock.market, stock.sector, stock.industry

                logger.info(f"銘柄を更新しました: {code} - {stock.name}")
                return stock

            except Exception as e:
                logger.error(f"銘柄更新エラー ({code}): {e}")
                return None

    def get_stock_by_code(self, code: str, detached: bool = False) -> Optional[Stock]:
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
                    _ = stock.id, stock.code, stock.name, stock.market, stock.sector, stock.industry
                    logger.debug(f"銘柄取得: {stock.code} - {stock.name}")
                return stock
            except Exception as e:
                logger.error(f"銘柄取得エラー ({code}): {e}")
                return None

    def search_stocks_by_name(self, name_pattern: str, limit: int = 50, detached: bool = False) -> List[Stock]:
        """
        銘柄名で部分一致検索（最適化版）

        Args:
            name_pattern: 銘柄名の一部
            limit: 結果の上限数
            detached: セッションから切り離すかどうか

        Returns:
            Stockオブジェクトのリスト
        """
        with self.db_manager.session_scope() as session:
            try:
                # 部分一致検索（大文字小文字区別なし）
                pattern = f"%{name_pattern}%"
                stocks = (
                    session.query(Stock)
                    .filter(Stock.name.ilike(pattern))
                    .limit(limit)
                    .all()
                )

                # セッションスコープを抜ける前に属性をアクセスして遅延読み込みを解決
                for stock in stocks:
                    _ = stock.id, stock.code, stock.name, stock.market, stock.sector, stock.industry

                logger.debug(f"銘柄名検索結果: {len(stocks)}件 (パターン: {name_pattern})")
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
                stocks = (
                    session.query(Stock)
                    .filter(Stock.sector == sector)
                    .limit(limit)
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
                stocks = (
                    session.query(Stock)
                    .filter(Stock.industry == industry)
                    .limit(limit)
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

                stocks = query.limit(limit).all()

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
                    _ = (existing_stock.code, existing_stock.name, existing_stock.market,
                         existing_stock.sector, existing_stock.industry)
                    return existing_stock
                else:
                    logger.info(f"新規銘柄を追加: {code} - {name}")
                    # 新規銘柄を作成
                    new_stock = Stock(
                        code=code,
                        name=name,
                        market=market,
                        sector=sector,
                        industry=industry
                    )
                    session.add(new_stock)
                    session.flush()

                    # 属性を事前読み込み（セッション内で）
                    _ = (new_stock.code, new_stock.name, new_stock.market,
                         new_stock.sector, new_stock.industry)
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
                        code=code, name=name, market=market, sector=sector, industry=industry
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
                if market_cap > 1_000_000_000_000 or market_cap > 100_000_000_000:  # 1兆ドル超
                    return "東証プライム"
                elif market_cap > 10_000_000_000:   # 100億ドル超
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

    def bulk_add_stocks(self, stocks_data: List[dict], batch_size: int = 1000) -> Dict[str, int]:
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
                validated_data.append({
                    "code": stock_data["code"],
                    "name": stock_data["name"],
                    "market": stock_data.get("market"),
                    "sector": stock_data.get("sector"),
                    "industry": stock_data.get("industry"),
                })

            # AdvancedBulkOperationsを使用して一括挿入
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="ignore",  # 重複は無視
                chunk_size=batch_size,
                unique_columns=["code"]
            )

            logger.info(f"銘柄一括追加完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括追加エラー: {e}")
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": len(stocks_data)}

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
                validated_data.append({
                    "code": code,
                    "name": stock_data.get("name"),
                    "market": stock_data.get("market"),
                    "sector": stock_data.get("sector"),
                    "industry": stock_data.get("industry"),
                })

            # AdvancedBulkOperationsを使用してupsert（挿入または更新）
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"]
            )

            logger.info(f"銘柄一括更新完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括更新エラー: {e}")
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": len(stocks_data)}

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
                validated_data.append({
                    "code": code,
                    "name": stock_data.get("name"),
                    "market": stock_data.get("market"),
                    "sector": stock_data.get("sector"),
                    "industry": stock_data.get("industry"),
                })

            # AdvancedBulkOperationsを使用してupsert
            result = self.bulk_operations.bulk_insert_with_conflict_resolution(
                Stock,
                validated_data,
                conflict_strategy="update",  # 重複時は更新
                chunk_size=batch_size,
                unique_columns=["code"]
            )

            logger.info(f"銘柄一括upsert完了: {result}")
            return result

        except Exception as e:
            logger.error(f"銘柄一括upsertエラー: {e}")
            return {"inserted": 0, "updated": 0, "skipped": 0, "errors": len(stocks_data)}


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
                        industry=industry
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

        # バッチ処理でAPIレートリミットを回避
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i + batch_size]
            logger.info(f"バッチ処理: {i//batch_size + 1}/{(len(codes) + batch_size - 1)//batch_size}")

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
                import time
                time.sleep(delay)

        result = {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total": len(codes)
        }

        logger.info(f"企業情報一括取得完了: {result}")
        return result


# グローバルインスタンス（改善版）
def create_stock_master_manager(db_manager=None, stock_fetcher=None) -> StockMasterManager:
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
