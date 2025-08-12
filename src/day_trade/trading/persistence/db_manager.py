"""
取引データベース管理

DB操作・同期・永続化機能
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy.exc import SQLAlchemyError

from ...models.database import db_manager
from ...models.enums import TradeType
from ...models.stock import Stock
from ...models.stock import Trade as DBTrade
from ...utils.enhanced_error_handler import get_default_error_handler
from ...utils.logging_config import get_context_logger, log_error_with_context
from ..core.types import Trade, TradeStatus

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


class TradeDatabaseManager:
    """
    取引データベース管理クラス

    取引データのDB永続化・同期・読み込み機能を提供
    """

    def __init__(self):
        """初期化"""
        self.db = db_manager
        logger.info("取引データベースマネージャー初期化完了")

    def load_trades_from_db(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        データベースから取引データ読み込み

        Args:
            symbol: 特定銘柄のみ（指定しない場合は全銘柄）

        Returns:
            取引データリスト
        """
        try:
            with self.db.get_session() as session:
                query = session.query(DBTrade)

                if symbol:
                    # 特定銘柄のみ
                    stock = session.query(Stock).filter_by(symbol=symbol).first()
                    if stock:
                        query = query.filter_by(stock_id=stock.id)
                    else:
                        logger.warning(f"銘柄{symbol}がデータベースに存在しません")
                        return []

                db_trades = query.order_by(DBTrade.timestamp.desc()).all()

                trades = []
                for db_trade in db_trades:
                    # Stock情報取得
                    stock = session.query(Stock).filter_by(id=db_trade.stock_id).first()
                    if not stock:
                        continue

                    trade = Trade(
                        id=db_trade.id,
                        symbol=stock.symbol,
                        trade_type=TradeType(db_trade.trade_type),
                        quantity=db_trade.quantity,
                        price=db_trade.price,
                        timestamp=db_trade.timestamp,
                        commission=db_trade.commission or Decimal("0"),
                        status=TradeStatus.EXECUTED,  # DBのデータは実行済み
                        notes=db_trade.notes or "",
                    )
                    trades.append(trade)

                logger.info(f"DB取引データ読み込み完了: {len(trades)}件")
                return trades

        except SQLAlchemyError as e:
            log_error_with_context("DB読み込みエラー", e, {"symbol": symbol})
            return []
        except Exception as e:
            logger.error(f"取引データ読み込み予期せぬエラー: {e}")
            return []

    def save_trade_to_db(self, trade: Trade) -> bool:
        """
        取引データをデータベースに保存

        Args:
            trade: 取引データ

        Returns:
            保存成功可否
        """
        try:
            with self.db.get_session() as session:
                # Stock情報取得または作成
                stock = session.query(Stock).filter_by(symbol=trade.symbol).first()
                if not stock:
                    stock = Stock(
                        symbol=trade.symbol,
                        name=f"銘柄{trade.symbol}",  # 暫定名
                        market="未指定",
                    )
                    session.add(stock)
                    session.flush()  # IDを取得

                # Trade作成
                db_trade = DBTrade(
                    id=trade.id,
                    stock_id=stock.id,
                    trade_type=trade.trade_type.value,
                    quantity=trade.quantity,
                    price=trade.price,
                    timestamp=trade.timestamp,
                    commission=trade.commission,
                    notes=trade.notes,
                )

                session.add(db_trade)
                session.commit()

                logger.info(f"DB取引保存完了: {trade.id} {trade.symbol}")
                return True

        except SQLAlchemyError as e:
            log_error_with_context(
                "DB取引保存エラー", e, {"trade_id": trade.id, "symbol": trade.symbol}
            )
            return False
        except Exception as e:
            logger.error(f"取引保存予期せぬエラー: {e}")
            return False

    def sync_trades_to_db(self, trades: List[Trade]) -> Dict[str, int]:
        """
        取引データをDBと同期

        Args:
            trades: 取引データリスト

        Returns:
            同期結果統計
        """
        result = {
            "total": len(trades),
            "saved": 0,
            "skipped": 0,
            "failed": 0,
        }

        try:
            with self.db.get_session() as session:
                # 既存取引ID取得
                existing_ids = set(
                    trade_id[0] for trade_id in session.query(DBTrade.id).all()
                )

                for trade in trades:
                    if trade.id in existing_ids:
                        result["skipped"] += 1
                        continue

                    # 新規取引保存
                    if self.save_trade_to_db(trade):
                        result["saved"] += 1
                    else:
                        result["failed"] += 1

                logger.info(
                    f"DB同期完了: 総数{result['total']}, "
                    f"保存{result['saved']}, スキップ{result['skipped']}, "
                    f"失敗{result['failed']}"
                )

        except Exception as e:
            logger.error(f"DB同期エラー: {e}")
            result["failed"] = result["total"] - result["saved"] - result["skipped"]

        return result

    def delete_trades_by_symbol(self, symbol: str) -> int:
        """
        特定銘柄の取引データ削除

        Args:
            symbol: 銘柄コード

        Returns:
            削除件数
        """
        try:
            with self.db.get_session() as session:
                # Stock取得
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    logger.warning(f"銘柄{symbol}がデータベースに存在しません")
                    return 0

                # 関連取引削除
                deleted_count = (
                    session.query(DBTrade).filter_by(stock_id=stock.id).delete()
                )
                session.commit()

                logger.info(f"銘柄{symbol}の取引データ削除完了: {deleted_count}件")
                return deleted_count

        except SQLAlchemyError as e:
            log_error_with_context("取引削除エラー", e, {"symbol": symbol})
            return 0
        except Exception as e:
            logger.error(f"取引削除予期せぬエラー: {e}")
            return 0

    def delete_all_trades(self) -> int:
        """
        全取引データ削除

        Returns:
            削除件数
        """
        try:
            with self.db.get_session() as session:
                deleted_count = session.query(DBTrade).delete()
                session.commit()

                logger.info(f"全取引データ削除完了: {deleted_count}件")
                return deleted_count

        except SQLAlchemyError as e:
            log_error_with_context("全取引削除エラー", e)
            return 0
        except Exception as e:
            logger.error(f"全取引削除予期せぬエラー: {e}")
            return 0

    def get_trade_count_by_symbol(self) -> Dict[str, int]:
        """
        銘柄別取引件数取得

        Returns:
            銘柄別件数辞書
        """
        try:
            with self.db.get_session() as session:
                # 銘柄別集計クエリ
                results = (
                    session.query(
                        Stock.symbol,
                        session.query(DBTrade).filter_by(stock_id=Stock.id).count(),
                    )
                    .join(DBTrade, Stock.id == DBTrade.stock_id)
                    .group_by(Stock.symbol)
                    .all()
                )

                trade_counts = dict(results)
                logger.debug(f"銘柄別取引件数取得: {len(trade_counts)}銘柄")
                return trade_counts

        except SQLAlchemyError as e:
            log_error_with_context("取引件数集計エラー", e)
            return {}
        except Exception as e:
            logger.error(f"取引件数集計予期せぬエラー: {e}")
            return {}

    def get_database_statistics(self) -> Dict:
        """
        データベース統計情報取得

        Returns:
            統計情報辞書
        """
        try:
            with self.db.get_session() as session:
                # 基本統計
                total_trades = session.query(DBTrade).count()
                total_stocks = session.query(Stock).count()

                # 取引タイプ別件数
                buy_count = (
                    session.query(DBTrade)
                    .filter_by(trade_type=TradeType.BUY.value)
                    .count()
                )
                sell_count = (
                    session.query(DBTrade)
                    .filter_by(trade_type=TradeType.SELL.value)
                    .count()
                )

                # 最新・最古取引日時
                latest_trade = (
                    session.query(DBTrade.timestamp)
                    .order_by(DBTrade.timestamp.desc())
                    .first()
                )
                earliest_trade = (
                    session.query(DBTrade.timestamp)
                    .order_by(DBTrade.timestamp.asc())
                    .first()
                )

                statistics = {
                    "total_trades": total_trades,
                    "total_stocks": total_stocks,
                    "buy_trades": buy_count,
                    "sell_trades": sell_count,
                    "latest_trade_date": (
                        latest_trade[0].isoformat() if latest_trade else None
                    ),
                    "earliest_trade_date": (
                        earliest_trade[0].isoformat() if earliest_trade else None
                    ),
                }

                logger.debug(f"DB統計取得完了: {total_trades}取引, {total_stocks}銘柄")
                return statistics

        except SQLAlchemyError as e:
            log_error_with_context("DB統計取得エラー", e)
            return {}
        except Exception as e:
            logger.error(f"DB統計取得予期せぬエラー: {e}")
            return {}

    def backup_database(self, backup_path: str) -> bool:
        """
        データベースバックアップ

        Args:
            backup_path: バックアップファイルパス

        Returns:
            バックアップ成功可否
        """
        try:
            # 全取引データ取得
            all_trades = self.load_trades_from_db()

            # JSON形式でエクスポート
            import json

            backup_data = {
                "backup_timestamp": datetime.now().isoformat(),
                "trade_count": len(all_trades),
                "trades": [trade.to_dict() for trade in all_trades],
                "statistics": self.get_database_statistics(),
            }

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"DBバックアップ完了: {backup_path} ({len(all_trades)}取引)")
            return True

        except Exception as e:
            logger.error(f"DBバックアップエラー: {e}")
            return False

    def restore_database(self, backup_path: str) -> bool:
        """
        データベース復元

        Args:
            backup_path: バックアップファイルパス

        Returns:
            復元成功可否
        """
        try:
            import json

            with open(backup_path, encoding="utf-8") as f:
                backup_data = json.load(f)

            trades_data = backup_data.get("trades", [])
            trades = [Trade.from_dict(trade_data) for trade_data in trades_data]

            # DB復元実行
            sync_result = self.sync_trades_to_db(trades)

            logger.info(
                f"DB復元完了: {backup_path} "
                f"({sync_result['saved']}件保存, {sync_result['skipped']}件スキップ)"
            )
            return sync_result["failed"] == 0

        except Exception as e:
            logger.error(f"DB復元エラー: {e}")
            return False
