"""
データベース読み込み機能
データベースからの取引履歴読み込みと同期を担当
"""

from decimal import Decimal

from ...models.database import db_manager
from ...models.stock import Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_error_with_context,
)
from .models import Trade
from .types import TradeStatus, TradeType

logger = get_context_logger(__name__)


class DatabaseLoader:
    """データベース読み込み操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def load_trades_from_db(self):
        """データベースから取引履歴を読み込み（トランザクション保護版）"""
        load_logger = self.logger.bind(operation="load_trades_from_db")
        load_logger.info("データベースから取引履歴読み込み開始")
        
        try:
            # トランザクション内で一括処理
            with db_manager.transaction_scope() as session:
                # データベースから全取引を取得
                db_trades = (
                    session.query(DBTrade).order_by(DBTrade.trade_datetime).all()
                )
                
                load_logger.info("DB取引データ取得", count=len(db_trades))
                
                # メモリ内データ構造を一旦クリア（原子性保証）
                trades_backup = self.trade_manager.trades.copy()
                positions_backup = self.trade_manager.positions.copy()
                realized_pnl_backup = self.trade_manager.realized_pnl.copy()
                counter_backup = self.trade_manager._trade_counter
                
                try:
                    # メモリ内データクリア
                    self.trade_manager.trades.clear()
                    self.trade_manager.positions.clear()
                    self.trade_manager.realized_pnl.clear()
                    self.trade_manager._trade_counter = 0
                    
                    for db_trade in db_trades:
                        # セッションから切り離す前に必要な属性を読み込み
                        trade_id = db_trade.id
                        stock_code = db_trade.stock_code
                        trade_type_str = db_trade.trade_type
                        quantity = db_trade.quantity
                        price = db_trade.price
                        trade_datetime = db_trade.trade_datetime
                        commission = db_trade.commission or Decimal("0")
                        memo = db_trade.memo or ""
                        
                        # メモリ内形式に変換
                        trade_type = (
                            TradeType.BUY
                            if trade_type_str.lower() == "buy"
                            else TradeType.SELL
                        )
                        
                        memory_trade = Trade(
                            id=f"DB_{trade_id}",  # DBから読み込んだことを示すプレフィックス
                            symbol=stock_code,
                            trade_type=trade_type,
                            quantity=quantity,
                            price=Decimal(str(price)),
                            timestamp=trade_datetime,
                            commission=Decimal(str(commission)),
                            status=TradeStatus.EXECUTED,
                            notes=memo,
                        )
                        
                        self.trade_manager.trades.append(memory_trade)
                        self.trade_manager.trade_operations.update_position(memory_trade)
                    
                    # 取引カウンターを最大値+1に設定
                    if db_trades:
                        max_id = max(db_trade.id for db_trade in db_trades)
                        self.trade_manager._trade_counter = max_id + 1
                    
                    load_logger.info(
                        "データベース読み込み完了",
                        loaded_trades=len(db_trades),
                        trade_counter=self.trade_manager._trade_counter,
                    )
                
                except Exception as restore_error:
                    # メモリ内データの復元
                    self.trade_manager.trades = trades_backup
                    self.trade_manager.positions = positions_backup
                    self.trade_manager.realized_pnl = realized_pnl_backup
                    self.trade_manager._trade_counter = counter_backup
                    load_logger.error(
                        "読み込み処理失敗、メモリ内データを復元",
                        error=str(restore_error),
                    )
                    raise restore_error
        
        except Exception as e:
            log_error_with_context(e, {"operation": "load_trades_from_db"})
            load_logger.error("データベース読み込み失敗", error=str(e))
            raise
    
    def sync_with_db(self):
        """データベースとの同期を実行（原子性保証版）"""
        sync_logger = self.logger.bind(operation="sync_with_db")
        sync_logger.info("データベース同期開始")
        
        # 現在のメモリ内データをバックアップ
        trades_backup = self.trade_manager.trades.copy()
        positions_backup = self.trade_manager.positions.copy()
        realized_pnl_backup = self.trade_manager.realized_pnl.copy()
        counter_backup = self.trade_manager._trade_counter
        
        try:
            # 現在のメモリ内データをクリア
            self.trade_manager.trades.clear()
            self.trade_manager.positions.clear()
            self.trade_manager.realized_pnl.clear()
            self.trade_manager._trade_counter = 0
            
            # データベースから再読み込み（トランザクション保護済み）
            self.load_trades_from_db()
            
            sync_logger.info("データベース同期完了")
        
        except Exception as e:
            # エラー時にはバックアップデータを復元
            self.trade_manager.trades = trades_backup
            self.trade_manager.positions = positions_backup
            self.trade_manager.realized_pnl = realized_pnl_backup
            self.trade_manager._trade_counter = counter_backup
            
            log_error_with_context(e, {"operation": "sync_with_db"})
            sync_logger.error(
                "データベース同期失敗、メモリ内データを復元", error=str(e)
            )
            raise