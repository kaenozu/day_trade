"""
株式売買機能（統合クラス）
BuyOperationsとSellOperationsを組み合わせた統合クラス
"""

from typing import Dict, Optional, Any

from ...utils.logging_config import get_context_logger
from .buy_operations import BuyOperations
from .sell_operations import SellOperations

logger = get_context_logger(__name__)


class StockTrading:
    """株式売買操作を管理する統合クラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
        
        # サブモジュールの初期化
        self.buy_operations = BuyOperations(trade_manager_ref)
        self.sell_operations = SellOperations(trade_manager_ref)
    
    def buy_stock(
        self,
        symbol: str,
        quantity: int,
        price,
        current_market_price: Optional = None,
        notes: str = "",
        persist_to_db: bool = True,
    ):
        """株式買い注文を実行"""
        return self.buy_operations.buy_stock(
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_market_price=current_market_price,
            notes=notes,
            persist_to_db=persist_to_db,
        )
    
    def sell_stock(
        self,
        symbol: str,
        quantity: int,
        price,
        current_market_price: Optional = None,
        notes: str = "",
        persist_to_db: bool = True,
    ):
        """株式売り注文を実行"""
        return self.sell_operations.sell_stock(
            symbol=symbol,
            quantity=quantity,
            price=price,
            current_market_price=current_market_price,
            notes=notes,
            persist_to_db=persist_to_db,
        )
    
    def execute_trade_order(
        self, trade_order: Dict[str, any], persist_to_db: bool = True
    ) -> Dict[str, any]:
        """
        取引注文を実行（買い/売りを統一インターフェースで処理）
        
        Args:
            trade_order: 取引注文辞書
                {
                    "action": "buy" | "sell",
                    "symbol": str,
                    "quantity": int,
                    "price": Decimal,
                    "current_market_price": Optional[Decimal],
                    "notes": str
                }
            persist_to_db: データベースに永続化するかどうか
        
        Returns:
            取引結果辞書
        """
        action = trade_order.get("action", "").lower()
        
        if action == "buy":
            return self.buy_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        elif action == "sell":
            return self.sell_stock(
                symbol=trade_order["symbol"],
                quantity=trade_order["quantity"],
                price=trade_order["price"],
                current_market_price=trade_order.get("current_market_price"),
                notes=trade_order.get("notes", ""),
                persist_to_db=persist_to_db,
            )
        else:
            raise ValueError(
                f"無効な取引アクション: {action}. 'buy' または 'sell' を指定してください"
            )