"""
データベース操作機能
バッチ処理とデータクリア操作を担当
"""

from datetime import datetime
from typing import Dict, List

from ...models.database import db_manager
from ...models.stock import Stock
from ...models.stock import Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)
from .models import Trade
from .types import TradeType

logger = get_context_logger(__name__)


class DatabaseOperations:
    """データベース操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def add_trades_batch(
        self, trades_data: List[Dict], persist_to_db: bool = True
    ) -> List[str]:
        """
        複数の取引を一括追加（トランザクション保護）
        
        Args:
            trades_data: 取引データのリスト
                [{"symbol": "7203", "trade_type": TradeType.BUY, "quantity": 100, "price": Decimal("2500"), ...}, ...]
            persist_to_db: データベースに永続化するかどうか
        
        Returns:
            作成された取引IDのリスト
        
        Raises:
            Exception: いずれかの取引処理でエラーが発生した場合、すべての処理がロールバック
        """
        batch_logger = logger.bind(
            operation="add_trades_batch",
            batch_size=len(trades_data),
            persist_to_db=persist_to_db,
        )
        batch_logger.info("一括取引追加処理開始")
        
        if not trades_data:
            batch_logger.warning("空の取引データが渡されました")
            return []
        
        trade_ids = []
        
        # メモリ内データのバックアップ
        trades_backup = self.trade_manager.trades.copy()
        positions_backup = self.trade_manager.positions.copy()
        realized_pnl_backup = self.trade_manager.realized_pnl.copy()
        counter_backup = self.trade_manager._trade_counter
        
        try:
            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    for i, trade_data in enumerate(trades_data):
                        try:
                            # 取引データの検証と補完
                            symbol = trade_data["symbol"]
                            trade_type = trade_data["trade_type"]
                            quantity = trade_data["quantity"]
                            price = trade_data["price"]
                            timestamp = trade_data.get("timestamp", datetime.now())
                            commission = trade_data.get("commission")
                            notes = trade_data.get("notes", "")
                            
                            if commission is None:
                                commission = self.trade_manager.trade_operations.calculate_commission(price, quantity)
                            
                            trade_id = self.trade_manager.trade_operations.generate_trade_id()
                            
                            # 1. 銘柄マスタの存在確認・作成
                            stock = (
                                session.query(Stock)
                                .filter(Stock.code == symbol)
                                .first()
                            )
                            if not stock:
                                stock = Stock(
                                    code=symbol,
                                    name=symbol,
                                    market="未定",
                                    sector="未定",
                                    industry="未定",
                                )
                                session.add(stock)
                                session.flush()
                            
                            # 2. データベース取引記録を作成
                            DBTrade.create_buy_trade(
                                session=session,
                                stock_code=symbol,
                                quantity=quantity,
                                price=float(price),
                                commission=float(commission),
                                memo=notes,
                            ) if trade_type == TradeType.BUY else DBTrade.create_sell_trade(
                                session=session,
                                stock_code=symbol,
                                quantity=quantity,
                                price=float(price),
                                commission=float(commission),
                                memo=notes,
                            )
                            
                            # 3. メモリ内データ構造を更新
                            memory_trade = Trade(
                                id=trade_id,
                                symbol=symbol,
                                trade_type=trade_type,
                                quantity=quantity,
                                price=price,
                                timestamp=timestamp,
                                commission=commission,
                                notes=notes,
                            )
                            
                            self.trade_manager.trades.append(memory_trade)
                            self.trade_manager.trade_operations.update_position(memory_trade)
                            trade_ids.append(trade_id)
                            
                            # バッチ内の中間状態をflush
                            if (i + 1) % 10 == 0:  # 10件ごとにflush
                                session.flush()
                        
                        except Exception as trade_error:
                            batch_logger.error(
                                "個別取引処理失敗",
                                trade_index=i,
                                symbol=trade_data.get("symbol", "unknown"),
                                error=str(trade_error),
                            )
                            raise trade_error
                    
                    # 最終的なビジネスイベントログ
                    log_business_event(
                        "trades_batch_added",
                        batch_size=len(trades_data),
                        trade_ids=trade_ids,
                        persisted=True,
                    )
                    
                    batch_logger.info(
                        "一括取引追加完了（DB永続化）", trade_count=len(trade_ids)
                    )
            
            else:
                # メモリ内のみの処理
                for trade_data in trades_data:
                    symbol = trade_data["symbol"]
                    trade_type = trade_data["trade_type"]
                    quantity = trade_data["quantity"]
                    price = trade_data["price"]
                    timestamp = trade_data.get("timestamp", datetime.now())
                    commission = trade_data.get("commission")
                    notes = trade_data.get("notes", "")
                    
                    if commission is None:
                        commission = self.trade_manager.trade_operations.calculate_commission(price, quantity)
                    
                    trade_id = self.trade_manager.trade_operations.generate_trade_id()
                    
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )
                    
                    self.trade_manager.trades.append(memory_trade)
                    self.trade_manager.trade_operations.update_position(memory_trade)
                    trade_ids.append(trade_id)
                
                log_business_event(
                    "trades_batch_added",
                    batch_size=len(trades_data),
                    trade_ids=trade_ids,
                    persisted=False,
                )
                
                batch_logger.info(
                    "一括取引追加完了（メモリのみ）", trade_count=len(trade_ids)
                )
            
            return trade_ids
        
        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trade_manager.trades = trades_backup
            self.trade_manager.positions = positions_backup
            self.trade_manager.realized_pnl = realized_pnl_backup
            self.trade_manager._trade_counter = counter_backup
            
            log_error_with_context(
                e,
                {
                    "operation": "add_trades_batch",
                    "batch_size": len(trades_data),
                    "persist_to_db": persist_to_db,
                    "completed_trades": len(trade_ids),
                },
            )
            batch_logger.error(
                "一括取引追加失敗、すべての変更をロールバック", error=str(e)
            )
            raise
    
    def clear_all_data(self, persist_to_db: bool = True):
        """
        すべての取引データを削除（トランザクション保護）
        
        Args:
            persist_to_db: データベースからも削除するかどうか
        
        Warning:
            この操作は取引履歴、ポジション、実現損益をすべて削除します
        """
        clear_logger = logger.bind(
            operation="clear_all_data", persist_to_db=persist_to_db
        )
        clear_logger.warning("全データ削除処理開始")
        
        # メモリ内データのバックアップ
        trades_backup = self.trade_manager.trades.copy()
        positions_backup = self.trade_manager.positions.copy()
        realized_pnl_backup = self.trade_manager.realized_pnl.copy()
        counter_backup = self.trade_manager._trade_counter
        
        try:
            if persist_to_db:
                # データベースとメモリ両方をクリア
                with db_manager.transaction_scope() as session:
                    # データベースの取引データを削除
                    deleted_count = session.query(DBTrade).delete()
                    clear_logger.info(
                        "データベース取引データ削除", deleted_count=deleted_count
                    )
                    
                    # メモリ内データクリア
                    self.trade_manager.trades.clear()
                    self.trade_manager.positions.clear()
                    self.trade_manager.realized_pnl.clear()
                    self.trade_manager._trade_counter = 0
                    
                    log_business_event(
                        "all_data_cleared",
                        deleted_db_records=deleted_count,
                        persisted=True,
                    )
                    
                    clear_logger.warning("全データ削除完了（DB + メモリ）")
            else:
                # メモリ内のみクリア
                self.trade_manager.trades.clear()
                self.trade_manager.positions.clear()
                self.trade_manager.realized_pnl.clear()
                self.trade_manager._trade_counter = 0
                
                log_business_event(
                    "all_data_cleared", deleted_db_records=0, persisted=False
                )
                
                clear_logger.warning("全データ削除完了（メモリのみ）")
        
        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trade_manager.trades = trades_backup
            self.trade_manager.positions = positions_backup
            self.trade_manager.realized_pnl = realized_pnl_backup
            self.trade_manager._trade_counter = counter_backup
            
            log_error_with_context(
                e, {"operation": "clear_all_data", "persist_to_db": persist_to_db}
            )
            clear_logger.error("全データ削除失敗、メモリ内データを復元", error=str(e))
            raise