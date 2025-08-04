"""
銘柄マスタ管理モジュール
東証上場銘柄の情報を管理し、検索機能を提供する
"""

import logging
from typing import List, Optional

import yfinance as yf
from sqlalchemy.orm import Session

from ..models.database import db_manager
from ..models.stock import Stock

logger = logging.getLogger(__name__)


class StockMasterManager:
    """銘柄マスタ管理クラス"""

    def __init__(self):
        """初期化"""
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

        with db_manager.session_scope() as session:
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
                logger.info(f"銘柄は既に存在します: {code} - {name}")
                # セッションに再アタッチして返す
                session.expunge(existing)
                session.add(existing)
                return existing

            # 新規作成
            stock = Stock(
                code=code, name=name, market=market, sector=sector, industry=industry
            )
            session.add(stock)
            session.flush()  # IDを取得

            logger.info(f"銘柄を追加しました: {code} - {name}")
            # セッションから切り離して返す
            session.expunge(stock)
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
        with db_manager.session_scope() as session:
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

                # 属性を事前に読み込みしてセッションから切り離し
                _ = stock.code, stock.name, stock.market, stock.sector, stock.industry
                session.expunge(stock)

                logger.info(f"銘柄を更新しました: {code} - {stock.name}")
                return stock

            except Exception as e:
                logger.error(f"銘柄更新エラー ({code}): {e}")
                return None

    def get_stock_by_code(self, code: str) -> Optional[Stock]:
        """
        証券コードで銘柄を取得

        Args:
            code: 証券コード

        Returns:
            Stockオブジェクト
        """
        with db_manager.session_scope() as session:
            try:
                stock = session.query(Stock).filter(Stock.code == code).first()
                if stock:
                    # 必要な属性を事前に読み込み
                    _ = (
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    session.expunge(stock)  # セッションから切り離し
                return stock
            except Exception as e:
                logger.error(f"銘柄取得エラー ({code}): {e}")
                return None

    def search_stocks_by_name(self, name_pattern: str, limit: int = 50) -> List[Stock]:
        """
        銘柄名で部分一致検索

        Args:
            name_pattern: 銘柄名の一部
            limit: 結果の上限数

        Returns:
            Stockオブジェクトのリスト
        """
        with db_manager.session_scope() as session:
            try:
                # 部分一致検索（大文字小文字区別なし）
                pattern = f"%{name_pattern}%"
                stocks = (
                    session.query(Stock)
                    .filter(Stock.name.ilike(pattern))
                    .limit(limit)
                    .all()
                )

                # 属性を事前に読み込みしてセッションから切り離し
                for stock in stocks:
                    _ = (
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    session.expunge(stock)

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
        with db_manager.session_scope() as session:
            try:
                stocks = (
                    session.query(Stock)
                    .filter(Stock.sector == sector)
                    .limit(limit)
                    .all()
                )

                # 属性を事前に読み込みしてセッションから切り離し
                for stock in stocks:
                    _ = (
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    session.expunge(stock)

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
        with db_manager.session_scope() as session:
            try:
                stocks = (
                    session.query(Stock)
                    .filter(Stock.industry == industry)
                    .limit(limit)
                    .all()
                )

                # 属性を事前に読み込みしてセッションから切り離し
                for stock in stocks:
                    _ = (
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    session.expunge(stock)

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
        with db_manager.session_scope() as session:
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

                # 属性を事前に読み込みしてセッションから切り離し
                for stock in stocks:
                    _ = (
                        stock.code,
                        stock.name,
                        stock.market,
                        stock.sector,
                        stock.industry,
                    )
                    session.expunge(stock)

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
        with db_manager.session_scope() as session:
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
        with db_manager.session_scope() as session:
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
        with db_manager.session_scope() as session:
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
        with db_manager.session_scope() as session:
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
        with db_manager.session_scope() as session:
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
        yfinanceから銘柄情報を取得してマスタを更新

        Args:
            code: 証券コード

        Returns:
            更新されたStockオブジェクト
        """
        try:
            # yfinance形式のシンボルに変換
            symbol = f"{code}.T" if "." not in code else code

            # yfinanceから情報取得
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "longName" not in info:
                logger.warning(f"yfinanceから情報を取得できません: {symbol}")
                return None

            # データを整理
            name = info.get("longName") or info.get("shortName", "")
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            # 市場区分を推定（yfinanceには含まれないため）
            market = "東証プライム"  # デフォルト

            # 既存銘柄を更新または新規作成
            stock = self.get_stock_by_code(code)
            if stock:
                return self.update_stock(
                    code=code,
                    name=name,
                    market=market,
                    sector=sector,
                    industry=industry,
                )
            else:
                return self.add_stock(
                    code=code,
                    name=name,
                    market=market,
                    sector=sector,
                    industry=industry,
                )

        except Exception as e:
            logger.error(f"銘柄情報取得・更新エラー ({code}): {e}")
            return None

    def bulk_add_stocks(self, stocks_data: List[dict], batch_size: int = 1000) -> int:
        """
        銘柄の一括追加（パフォーマンス最適化版）

        Args:
            stocks_data: 銘柄データのリスト
                例: [{'code': '1000', 'name': '株式会社A', 'market': '東証プライム', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            追加された銘柄数
        """
        if not stocks_data:
            return 0

        added_count = 0

        with db_manager.session_scope() as session:
            try:
                # 既存銘柄コードを取得してセット化（高速検索用）
                existing_codes = set(
                    code[0] for code in session.query(Stock.code).all()
                )

                # 新規銘柄のみをフィルタリング
                new_stocks = [
                    stock
                    for stock in stocks_data
                    if stock.get("code") not in existing_codes
                ]

                if not new_stocks:
                    logger.info("追加すべき新規銘柄がありません")
                    return 0

                # バッチ処理で一括挿入
                for i in range(0, len(new_stocks), batch_size):
                    batch = new_stocks[i : i + batch_size]

                    # データの検証と準備
                    validated_batch = []
                    for stock_data in batch:
                        if not stock_data.get("code") or not stock_data.get("name"):
                            logger.warning(f"無効な銘柄データをスキップ: {stock_data}")
                            continue
                        validated_batch.append(
                            {
                                "code": stock_data["code"],
                                "name": stock_data["name"],
                                "market": stock_data.get("market"),
                                "sector": stock_data.get("sector"),
                                "industry": stock_data.get("industry"),
                            }
                        )

                    if validated_batch:
                        # 一括挿入実行
                        session.bulk_insert_mappings(Stock, validated_batch)
                        added_count += len(validated_batch)

                        logger.info(
                            f"バッチ {i // batch_size + 1}: {len(validated_batch)}件の銘柄を追加"
                        )

                logger.info(f"一括追加完了: 合計 {added_count}件の銘柄を追加しました")
                return added_count

            except Exception as e:
                logger.error(f"一括追加エラー: {e}")
                return 0

    def bulk_update_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> int:
        """
        銘柄の一括更新（パフォーマンス最適化版）

        Args:
            stocks_data: 更新する銘柄データのリスト
                例: [{'code': '1000', 'name': '新社名', 'sector': '新セクター', ...}, ...]
            batch_size: バッチサイズ

        Returns:
            更新された銘柄数
        """
        if not stocks_data:
            return 0

        updated_count = 0

        with db_manager.session_scope() as session:
            try:
                # 既存銘柄コードとIDのマッピングを取得
                existing_stocks = {
                    stock.code: stock.id
                    for stock in session.query(Stock.id, Stock.code).all()
                }

                # 存在する銘柄のみをフィルタリング
                update_stocks = []
                for stock_data in stocks_data:
                    code = stock_data.get("code")
                    if code and code in existing_stocks:
                        # IDを追加して更新用データを準備
                        update_data = {
                            "id": existing_stocks[code],
                            **{
                                k: v
                                for k, v in stock_data.items()
                                if k != "code" and v is not None
                            },
                        }
                        update_stocks.append(update_data)

                if not update_stocks:
                    logger.info("更新すべき銘柄がありません")
                    return 0

                # バッチ処理で一括更新
                for i in range(0, len(update_stocks), batch_size):
                    batch = update_stocks[i : i + batch_size]

                    # 一括更新実行
                    session.bulk_update_mappings(Stock, batch)
                    updated_count += len(batch)

                    logger.info(
                        f"バッチ {i // batch_size + 1}: {len(batch)}件の銘柄を更新"
                    )

                logger.info(f"一括更新完了: 合計 {updated_count}件の銘柄を更新しました")
                return updated_count

            except Exception as e:
                logger.error(f"一括更新エラー: {e}")
                return 0

    def bulk_upsert_stocks(
        self, stocks_data: List[dict], batch_size: int = 1000
    ) -> dict:
        """
        銘柄の一括upsert（存在すれば更新、なければ追加）

        Args:
            stocks_data: 銘柄データのリスト
            batch_size: バッチサイズ

        Returns:
            実行結果の辞書 {'added': int, 'updated': int, 'total': int}
        """
        if not stocks_data:
            return {"added": 0, "updated": 0, "total": 0}

        with db_manager.session_scope() as session:
            try:
                # 既存銘柄コードのセットを取得
                existing_codes = set(
                    code[0] for code in session.query(Stock.code).all()
                )

                # 新規追加と更新に分離
                new_stocks = []
                update_stocks = []

                for stock_data in stocks_data:
                    code = stock_data.get("code")
                    if not code:
                        logger.warning(f"銘柄コードが無効です: {stock_data}")
                        continue

                    if code in existing_codes:
                        update_stocks.append(stock_data)
                    else:
                        new_stocks.append(stock_data)

                # 一括追加と一括更新を実行
                added_count = (
                    self.bulk_add_stocks(new_stocks, batch_size) if new_stocks else 0
                )
                updated_count = (
                    self.bulk_update_stocks(update_stocks, batch_size)
                    if update_stocks
                    else 0
                )

                total_count = added_count + updated_count

                logger.info(
                    f"一括upsert完了: 追加 {added_count}件, 更新 {updated_count}件, 合計 {total_count}件"
                )

                return {
                    "added": added_count,
                    "updated": updated_count,
                    "total": total_count,
                }

            except Exception as e:
                logger.error(f"一括upsertエラー: {e}")
                return {"added": 0, "updated": 0, "total": 0}


# グローバルインスタンス
stock_master = StockMasterManager()
