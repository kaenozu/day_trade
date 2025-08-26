"""
基本取引管理機能
取引記録の追加とポジション更新を担当
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from ...models.database import db_manager
from ...models.stock import Stock
from ...models.stock import Trade as DBTrade
from ...utils.logging_config import (
    get_context_logger,
    log_business_event,
    log_error_with_context,
)
from .models import Position, RealizedPnL, Trade
from .types import TradeStatus, TradeType

logger = get_context_logger(__name__)


class TradeOperations:
    """基本取引操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def generate_trade_id(self) -> str:
        """取引IDを生成"""
        self.trade_manager._trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}{self.trade_manager._trade_counter:04d}"
    
    def calculate_commission(self, price: Decimal, quantity: int) -> Decimal:
        """手数料を計算"""
        total_value = price * Decimal(quantity)
        commission = total_value * self.trade_manager.commission_rate
        # 最低100円の手数料
        return max(commission, Decimal("100"))
    
    def add_trade(
        self,
        symbol: str,
        trade_type: TradeType,
        quantity: int,
        price: Decimal,
        timestamp: datetime = None,
        commission: Decimal = None,
        notes: str = "",
        persist_to_db: bool = True,
    ) -> str:
        """
        取引を追加（データベース永続化対応）
        
        Args:
            symbol: 銘柄コード
            trade_type: 取引タイプ
            quantity: 数量
            price: 価格
            timestamp: 取引日時
            commission: 手数料（Noneの場合は自動計算）
            notes: メモ
            persist_to_db: データベースに永続化するかどうか
        
        Returns:
            取引ID
        """
        operation_logger = logger.bind(
            operation="add_trade",
            symbol=symbol,
            trade_type=trade_type.value,
            quantity=quantity,
            price=float(price),
            persist_to_db=persist_to_db,
        )
        
        operation_logger.info("取引追加処理開始")
        
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if commission is None:
                commission = self.calculate_commission(price, quantity)
            
            trade_id = self.generate_trade_id()
            
            # メモリ内データ構造のトレード
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
            
            if persist_to_db:
                # データベース永続化（トランザクション保護）
                with db_manager.transaction_scope() as session:
                    # 1. 銘柄マスタの存在確認・作成
                    stock = session.query(Stock).filter(Stock.code == symbol).first()
                    if not stock:
                        operation_logger.info("銘柄マスタに未登録、新規作成")
                        stock = Stock(
                            code=symbol,
                            name=symbol,  # 名前が不明な場合はコードを使用
                            market="未定",
                            sector="未定",
                            industry="未定",
                        )
                        session.add(stock)
                        session.flush()  # IDを確定
                    
                    # 2. データベース取引記録を作成
                    db_trade = (
                        DBTrade.create_buy_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=float(price),
                            commission=float(commission),
                            memo=notes,
                        )
                        if trade_type == TradeType.BUY
                        else DBTrade.create_sell_trade(
                            session=session,
                            stock_code=symbol,
                            quantity=quantity,
                            price=float(price),
                            commission=float(commission),
                            memo=notes,
                        )
                    )
                    
                    # 3. メモリ内データ構造を更新
                    self.trade_manager.trades.append(memory_trade)
                    self.update_position(memory_trade)
                    
                    # 中間状態をflushして整合性を確認
                    session.flush()
                    
                    # ビジネスイベントログ
                    log_business_event(
                        "trade_added",
                        trade_id=trade_id,
                        symbol=symbol,
                        trade_type=trade_type.value,
                        quantity=quantity,
                        price=float(price),
                        commission=float(commission),
                        persisted=True,
                    )
                    
                    operation_logger.info(
                        "取引追加完了（DB永続化）",
                        trade_id=trade_id,
                        db_trade_id=db_trade.id,
                    )
            else:
                # メモリ内のみの処理（後方互換性）
                self.trade_manager.trades.append(memory_trade)
                self.update_position(memory_trade)
                
                log_business_event(
                    "trade_added",
                    trade_id=trade_id,
                    symbol=symbol,
                    trade_type=trade_type.value,
                    quantity=quantity,
                    price=float(price),
                    commission=float(commission),
                    persisted=False,
                )
                
                operation_logger.info("取引追加完了（メモリのみ）", trade_id=trade_id)
            
            return trade_id
        
        except Exception as e:
            log_error_with_context(
                e,
                {
                    "operation": "add_trade",
                    "symbol": symbol,
                    "trade_type": trade_type.value,
                    "quantity": quantity,
                    "price": float(price),
                    "persist_to_db": persist_to_db,
                },
            )
            operation_logger.error("取引追加失敗", error=str(e))
            raise
    
    def update_position(self, trade: Trade):
        """ポジションを更新"""
        symbol = trade.symbol
        
        if trade.trade_type == TradeType.BUY:
            if symbol in self.trade_manager.positions:
                # 既存ポジションに追加
                position = self.trade_manager.positions[symbol]
                total_cost = (
                    position.total_cost
                    + (trade.price * Decimal(trade.quantity))
                    + trade.commission
                )
                total_quantity = position.quantity + trade.quantity
                average_price = total_cost / Decimal(total_quantity)
                
                position.quantity = total_quantity
                position.average_price = average_price
                position.total_cost = total_cost
            else:
                # 新規ポジション
                total_cost = (trade.price * Decimal(trade.quantity)) + trade.commission
                self.trade_manager.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=trade.quantity,
                    average_price=total_cost / Decimal(trade.quantity),
                    total_cost=total_cost,
                )
        
        elif trade.trade_type == TradeType.SELL:
            if symbol in self.trade_manager.positions:
                position = self.trade_manager.positions[symbol]
                
                if position.quantity >= trade.quantity:
                    # 実現損益を計算
                    buy_price = position.average_price
                    sell_price = trade.price
                    
                    # 手数料を按分
                    buy_commission_per_share = (
                        position.total_cost / Decimal(position.quantity)
                        - position.average_price
                    )
                    buy_commission = buy_commission_per_share * Decimal(trade.quantity)
                    
                    pnl_before_tax = (
                        (sell_price - buy_price) * Decimal(trade.quantity)
                        - buy_commission
                        - trade.commission
                    )
                    
                    # 税金計算（利益が出た場合のみ）
                    tax = Decimal("0")
                    if pnl_before_tax > 0:
                        tax = pnl_before_tax * self.trade_manager.tax_rate
                    
                    pnl = pnl_before_tax - tax
                    pnl_percent = (pnl / (buy_price * Decimal(trade.quantity))) * 100
                    
                    # 実現損益を記録
                    realized_pnl = RealizedPnL(
                        symbol=symbol,
                        quantity=trade.quantity,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        buy_commission=buy_commission,
                        sell_commission=trade.commission,
                        pnl=pnl,
                        pnl_percent=pnl_percent,
                        buy_date=self.get_earliest_buy_date(symbol),
                        sell_date=trade.timestamp,
                    )
                    
                    self.trade_manager.realized_pnl.append(realized_pnl)
                    
                    # ポジション更新
                    remaining_quantity = position.quantity - trade.quantity
                    if remaining_quantity > 0:
                        # 按分してコストを調整
                        remaining_ratio = Decimal(remaining_quantity) / Decimal(
                            position.quantity
                        )
                        remaining_cost = position.total_cost * remaining_ratio
                        position.quantity = remaining_quantity
                        position.total_cost = remaining_cost
                        position.average_price = remaining_cost / Decimal(
                            remaining_quantity
                        )
                    else:
                        # ポジション完全クローズ
                        del self.trade_manager.positions[symbol]
                
                else:
                    logger.warning(
                        f"銘柄 '{symbol}' の売却数量が保有数量 ({position.quantity}) を超過しています。売却数量: {trade.quantity}。取引は処理されません。"
                    )
            else:
                logger.warning(
                    f"ポジションを保有していない銘柄 '{symbol}' の売却を試みました。取引は無視されます。"
                )
    
    def get_earliest_buy_date(self, symbol: str) -> datetime:
        """最も古い買い取引の日付を取得"""
        buy_trades = [
            t
            for t in self.trade_manager.trades
            if t.symbol == symbol and t.trade_type == TradeType.BUY
        ]
        if buy_trades:
            return min(trade.timestamp for trade in buy_trades)
        return datetime.now()