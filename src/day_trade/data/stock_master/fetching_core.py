"""
銘柄マスタのデータ取得・更新機能（基本機能）モジュール

このモジュールはStockFetcherと連携して単体銘柄の情報取得と更新を行います。
基本的な取得・更新機能を提供します。
"""

from typing import Dict, Optional

from ...models.stock import Stock
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class StockDataFetcherCore:
    """銘柄データ取得・更新クラス（基本機能）"""

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
                    self._preload_stock_attributes(existing_stock)
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
                    self._preload_stock_attributes(new_stock)
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

    def _preload_stock_attributes(self, stock: Stock) -> None:
        """
        Stockオブジェクトの属性を事前読み込み

        Args:
            stock: 読み込み対象のStockオブジェクト
        """
        _ = (
            stock.code,
            stock.name,
            stock.market,
            stock.sector,
            stock.industry,
        )