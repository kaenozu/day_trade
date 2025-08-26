"""
株式買い注文機能
買い注文の実行を担当
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any

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


class BuyOperations:
    """株式買い注文操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        current_market_price: Optional[Decimal] = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> Dict[str, any]:
        """
        株式買い注文を実行（完全なトランザクション保護）
        
        ポートフォリオ更新と取引履歴追加を単一のトランザクションで処理し、
        データの整合性を保証する。
        
        Args:
            symbol: 銘柄コード
            quantity: 購入数量
            price: 購入価格
            current_market_price: 現在の市場価格（ポートフォリオ評価用）
            notes: 取引メモ
            persist_to_db: データベースに永続化するかどうか
        
        Returns:
            取引結果辞書（取引ID、更新後ポジション、手数料等）
        
        Raises:
            ValueError: 無効な購入パラメータ
            Exception: データベース処理エラー
        """
        buy_logger = logger.bind(
            operation="buy_stock",
            symbol=symbol,
            quantity=quantity,
            price=float(price),
            persist_to_db=persist_to_db,
        )
        
        buy_logger.info("株式買い注文処理開始")
        
        # パラメータ検証
        if quantity <= 0:
            raise ValueError(f"購入数量は正数である必要があります: {quantity}")
        if price <= 0:
            raise ValueError(f"購入価格は正数である必要があります: {price}")
        
        # メモリ内データのバックアップ
        trades_backup = self.trade_manager.trades.copy()
        positions_backup = self.trade_manager.positions.copy()
        counter_backup = self.trade_manager._trade_counter
        
        try:
            # 手数料計算
            commission = self.trade_manager.trade_operations.calculate_commission(price, quantity)
            timestamp = datetime.now()
            
            if persist_to_db:
                # データベース永続化の場合は全処理をトランザクション内で実行
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        buy_logger.info("銘柄マスタに未登録、新規作成")
                        stock = Stock(
                            code=symbol,
                            name=symbol,
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()
                    
                    # 2. 取引ID生成
                    trade_id = self.trade_manager.trade_operations.generate_trade_id()
                    
                    # 3. データベース取引記録を作成
                    db_trade = DBTrade.create_buy_trade(
                        session=session,
                        stock_code=symbol,
                        quantity=quantity,
                        price=float(price),
                        commission=float(commission),
                        memo=notes,
                    )
                    
                    # 4. メモリ内取引記録作成
                    memory_trade = Trade(
                        id=trade_id,
                        symbol=symbol,
                        trade_type=TradeType.BUY,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        commission=commission,
                        notes=notes,
                    )
                    
                    # 5. ポジション更新（原子的実行）
                    old_position = self.trade_manager.positions.get(symbol)
                    self.trade_manager.trades.append(memory_trade)
                    self.trade_manager.trade_operations.update_position(memory_trade)
                    new_position = self.trade_manager.positions.get(symbol)
                    
                    # 6. 現在価格更新（指定されている場合）
                    if current_market_price and symbol in self.trade_manager.positions:
                        self.trade_manager.positions[symbol].current_price = current_market_price
                    
                    # 中間状態をflushして整合性を確認
                    session.flush()
                    
                    # ビジネスイベントログ
                    log_business_event(
                        "stock_purchased",
                        trade_id=trade_id,
                        symbol=symbol,
                        quantity=quantity,
                        price=float(price),
                        commission=float(commission),
                        old_position=old_position.to_dict() if old_position else None,
                        new_position=new_position.to_dict() if new_position else None,
                        persisted=True,
                    )
                    
                    buy_logger.info(
                        "株式買い注文完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                        commission=float(commission),
                    )
            else:
                # メモリ内のみの処理
                trade_id = self.trade_manager.trade_operations.generate_trade_id()
                
                memory_trade = Trade(
                    id=trade_id,
                    symbol=symbol,
                    trade_type=TradeType.BUY,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    commission=commission,
                    notes=notes,
                )
                
                old_position = self.trade_manager.positions.get(symbol)
                self.trade_manager.trades.append(memory_trade)
                self.trade_manager.trade_operations.update_position(memory_trade)
                new_position = self.trade_manager.positions.get(symbol)
                
                if current_market_price and symbol in self.trade_manager.positions:
                    self.trade_manager.positions[symbol].current_price = current_market_price
                
                log_business_event(
                    "stock_purchased",
                    trade_id=trade_id,
                    symbol=symbol,
                    quantity=quantity,
                    price=float(price),
                    commission=float(commission),
                    old_position=old_position.to_dict() if old_position else None,
                    new_position=new_position.to_dict() if new_position else None,
                    persisted=False,
                )
                
                buy_logger.info(
                    "株式買い注文完了（メモリのみ）",
                    trade_id=trade_id,
                    commission=float(commission),
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
                "position": self.trade_manager.positions[symbol].to_dict()
                if symbol in self.trade_manager.positions
                else None,
                "total_cost": float(price * quantity + commission),
            }
            
            return result
        
        except Exception as e:
            # エラー時はメモリ内データを復元
            self.trade_manager.trades = trades_backup
            self.trade_manager.positions = positions_backup
            self.trade_manager._trade_counter = counter_backup
            
            log_error_with_context(
                e,
                {
                    "operation": "buy_stock",
                    "symbol": symbol,
                    "quantity": quantity,
                    "price": float(price),
                    "persist_to_db": persist_to_db,
                },
            )
            buy_logger.error("株式買い注文失敗、変更をロールバック", error=str(e))
            raise