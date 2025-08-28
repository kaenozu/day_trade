"""
株式売り注文機能
売り注文の実行を担当
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any

from ...models.database import db_manager
from ...models.stock import Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)
from .models import Trade
from .types import TradeType

logger = get_context_logger(__name__)


class SellOperations:
    """株式売り注文操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, any]:
        """
        株式売り注文を実行（完全なトランザクション保護）
        
        ポートフォリオ更新、取引履歴追加、実現損益計算を
        単一のトランザクションで処理し、データの整合性を保証する。
        
        Args:
            symbol: 銘柄コード
            quantity: 売却数量
            price: 売却価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか
        
        Returns:
            取引結果辞書（取引ID、更新後ポジション、実現損益等）
        
        Raises:
            ValueError: 無効な売却パラメータ（保有数量不足等）
            Exception: データベース処理エラー
        """
        sell_logger = logger.bind(
            operation="sell_stock",
            symbol=symbol,
            quantity=quantity,
            price=float(price),
            persist_to_db=persist_to_db,
        )
        
        sell_logger.info("株式売り注文処理開始")
        
        # パラメータ検証
        if quantity <= 0:
            raise ValueError(f"売却数量は正数である必要があります: {quantity}")
        if price <= 0:
            raise ValueError(f"売却価格は正数である必要があります: {price}")
        
        # ポジション存在確認
        if symbol not in self.trade_manager.positions:
            raise ValueError(f"銘柄 '{symbol}' のポジションが存在しません")
        
        current_position = self.trade_manager.positions[symbol]
        if current_position.quantity < quantity:
            raise ValueError(
                f"売却数量 ({quantity}) が保有数量 ({current_position.quantity}) を超過しています"
            )
        
        # メモリ内データのバックアップ
        trades_backup = self.trade_manager.trades.copy()
        positions_backup = self.trade_manager.positions.copy()
        realized_pnl_backup = self.trade_manager.realized_pnl.copy()
        counter_backup = self.trade_manager._trade_counter
        
        try:
            # 手数料計算
            commission = self.trade_manager.trade_operations.calculate_commission(price, quantity)
            timestamp = datetime.now()
            
            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 取引ID生成
                    trade_id = self.trade_manager.trade_operations.generate_trade_id()
                    
                    # 2. データベース取引記録を作成
                    db_trade = DBTrade.create_sell_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=float(price),
                        commission=float(commission),
                        memo=notes,
                    )
                    
                    # 3. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.SELL,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )
                    
                    # 4. ポジション更新と実現損益計算（原子的実行）
                    old_position = self.trade_manager.positions[symbol]
                    old_realized_pnl_count = len(self.trade_manager.realized_pnl)
                    
                    self.trade_manager.trades.append(memory_trade)
                    self.trade_manager.trade_operations.update_position(memory_trade)
                    
                    new_position = self.trade_manager.positions.get(symbol)
                    new_realized_pnl = None
                    if len(self.trade_manager.realized_pnl) > old_realized_pnl_count:
                        new_realized_pnl = self.trade_manager.realized_pnl[-1]
                    
                    # 5. 現在価格更新（指定されている場合）
                    if current_market_price and symbol in self.trade_manager.positions:
                        self.trade_manager.positions[symbol].current_price = current_market_price
                    
                    # 中間状態をflushして整合性を確認
                    session.flush()
                    
                    # ビジネスイベントログ
                    log_business_event(
                        "stock_sold",
                        trade_id=trade_id,
                        symbol=symbol,
                        quantity=quantity,
                        price=float(price),
                        commission=float(commission),
                        old_position=old_position.to_dict(),
                        new_position=new_position.to_dict() if new_position else None,
                        realized_pnl=new_realized_pnl.to_dict()
                        if new_realized_pnl
                        else None,
                        persisted=True,
                    )
                    
                    sell_logger.info(
                        "株式売り注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=float(commission),
                        realized_pnl=new_realized_pnl.pnl if new_realized_pnl else None,
                    )
            else:
                # メモリ内のみの処理
                trade_id = self.trade_manager.trade_operations.generate_trade_id()
                
                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.SELL,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )
                
                old_position = self.trade_manager.positions[symbol]
                old_realized_pnl_count = len(self.trade_manager.realized_pnl)
                
                self.trade_manager.trades.append(memory_trade)
                self.trade_manager.trade_operations.update_position(memory_trade)
                
                new_position = self.trade_manager.positions.get(symbol)
                new_realized_pnl = None
                if len(self.trade_manager.realized_pnl) > old_realized_pnl_count:
                    new_realized_pnl = self.trade_manager.realized_pnl[-1]
                
                if current_market_price and symbol in self.trade_manager.positions:
                    self.trade_manager.positions[symbol].current_price = current_market_price
                
                log_business_event(
                    "stock_sold",
                    trade_id=trade_id,
                    symbol=symbol,
                    quantity=quantity,
                    price=float(price),
                    commission=float(commission),
                    old_position=old_position.to_dict(),
                    new_position=new_position.to_dict() if new_position else None,
                    realized_pnl=new_realized_pnl.to_dict()
                    if new_realized_pnl
                    else None,
                    persisted=False,
                )
                
                sell_logger.info(
                    "株式売り注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=float(commission),
                    realized_pnl=new_realized_pnl.pnl if new_realized_pnl else None,
                )
            
            # 結果データ作成
            result = {
                "success": True,
                "trade_id": trade_id,
                "symbol": symbol,
                "quantity": quantity,
                "price": float(price),
                "commission": float(commission),
                "timestamp": timestamp.isoformat(),
                "position": new_position.to_dict() if new_position else None,
                "position_closed": new_position is None,
                "realized_pnl": new_realized_pnl.to_dict()
                if new_realized_pnl
                else None,
                "gross_proceeds": float(price * quantity - commission),
            }
            
            return result
        
        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trade_manager.trades = trades_backup
            self.trade_manager.positions = positions_backup
            self.trade_manager.realized_pnl = realized_pnl_backup
            self.trade_manager._trade_counter = counter_backup
            
            log_error_with_context(
                e,
                {
                    "operation": "sell_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": float(price),
                    "persist_to_db": persist_to_db,
                },
            )
            sell_logger.error("株式売り注文失敗、変更をロールバック", error=str(e))
            raise