"""
銘柄マスタ管理モジュール
東証上場銘柄の情報を管理し、検索機能を提供する
"""

import logging
from typing import Dict, List, Optional, Tuple

import yfinance as yf
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models.database import db_manager
from ..models.stock import Stock
from ..models.bulk_operations import AdvancedBulkOperations

logger = logging.getLogger(__name__)


class StockMasterManager:
    """銘柄マスタ管理クラス"""

    def __init__(self):
        """初期化"""
        self.db_manager = db_manager
        self.bulk_operations = AdvancedBulkOperations(db_manager)

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

    def bulk_insert_stocks(self, stock_data: List[Dict[str, str]], chunk_size: int = 1000) -> Tuple[int, int]:
        """
        高性能なbulk insert操作

        Args:
            stock_data: 銘柄データのリスト [{"code": "1234", "name": "銘柄名", ...}, ...]
            chunk_size: チャンクサイズ

        Returns:
            (追加件数, スキップ件数)のタプル
        """
        inserted_count = 0
        skipped_count = 0

        with db_manager.session_scope() as session:
            try:
                # 既存コードのセットを取得（メモリ効率版）
                existing_codes = set()
                for row in session.execute(text("SELECT code FROM stocks")):
                    existing_codes.add(row[0])

                # 新規データをフィルタリング
                new_stocks = []
                for stock_info in stock_data:
                    if stock_info["code"] not in existing_codes:
                        new_stocks.append(stock_info)
                    else:
                        skipped_count += 1

                # チャンクごとにbulk insert実行
                for i in range(0, len(new_stocks), chunk_size):
                    chunk = new_stocks[i:i + chunk_size]
                    if chunk:
                        session.bulk_insert_mappings(Stock, chunk)
                        inserted_count += len(chunk)

                        # 進捗ログ（大量データ対応）
                        if len(new_stocks) > chunk_size:
                            progress = min(i + chunk_size, len(new_stocks))
                            logger.info(f"Bulk insert progress: {progress}/{len(new_stocks)} ({progress/len(new_stocks)*100:.1f}%)")

                session.commit()
                logger.info(f"Bulk insert完了: 追加={inserted_count}, スキップ={skipped_count}")

            except Exception as e:
                session.rollback()
                logger.error(f"Bulk insert エラー: {e}")
                raise

        return inserted_count, skipped_count

    def bulk_update_stocks(self, update_data: List[Dict[str, str]], chunk_size: int = 1000) -> int:
        """
        高性能なbulk update操作

        Args:
            update_data: 更新データのリスト [{"code": "1234", "name": "新銘柄名", ...}, ...]
            chunk_size: チャンクサイズ

        Returns:
            更新件数
        """
        updated_count = 0

        with db_manager.session_scope() as session:
            try:
                # チャンクごとにbulk update実行
                for i in range(0, len(update_data), chunk_size):
                    chunk = update_data[i:i + chunk_size]
                    if chunk:
                        session.bulk_update_mappings(Stock, chunk)
                        updated_count += len(chunk)

                        # 進捗ログ（大量データ対応）
                        if len(update_data) > chunk_size:
                            progress = min(i + chunk_size, len(update_data))
                            logger.info(f"Bulk update progress: {progress}/{len(update_data)} ({progress/len(update_data)*100:.1f}%)")

                session.commit()
                logger.info(f"Bulk update完了: 更新件数={updated_count}")

            except Exception as e:
                session.rollback()
                logger.error(f"Bulk update エラー: {e}")
                raise

        return updated_count

    def bulk_upsert_stocks(self, stock_data: List[Dict[str, str]], chunk_size: int = 1000) -> Tuple[int, int]:
        """
        高性能なbulk upsert操作（INSERT ON CONFLICT UPDATE）

        Args:
            stock_data: 銘柄データのリスト
            chunk_size: チャンクサイズ

        Returns:
            (挿入件数, 更新件数)のタプル
        """
        inserted_count = 0
        updated_count = 0

        with db_manager.session_scope() as session:
            try:
                # SQLiteのINSERT OR REPLACEを使用
                for i in range(0, len(stock_data), chunk_size):
                    chunk = stock_data[i:i + chunk_size]

                    # 既存チェック用のコードリスト
                    codes_in_chunk = [item["code"] for item in chunk]
                    existing_codes = set()

                    if codes_in_chunk:
                        placeholders = ",".join([f"'{code}'" for code in codes_in_chunk])
                        query = text(f"SELECT code FROM stocks WHERE code IN ({placeholders})")
                        for row in session.execute(query):
                            existing_codes.add(row[0])

                    # 挿入と更新に分離
                    inserts = []
                    updates = []

                    for item in chunk:
                        if item["code"] in existing_codes:
                            updates.append(item)
                        else:
                            inserts.append(item)

                    # Bulk operations実行
                    if inserts:
                        session.bulk_insert_mappings(Stock, inserts)
                        inserted_count += len(inserts)

                    if updates:
                        session.bulk_update_mappings(Stock, updates)
                        updated_count += len(updates)

                    # 進捗ログ
                    if len(stock_data) > chunk_size:
                        progress = min(i + chunk_size, len(stock_data))
                        logger.info(f"Bulk upsert progress: {progress}/{len(stock_data)} ({progress/len(stock_data)*100:.1f}%)")

                session.commit()
                logger.info(f"Bulk upsert完了: 挿入={inserted_count}, 更新={updated_count}")

            except Exception as e:
                session.rollback()
                logger.error(f"Bulk upsert エラー: {e}")
                raise

        return inserted_count, updated_count

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


# グローバルインスタンス
stock_master = StockMasterManager()
