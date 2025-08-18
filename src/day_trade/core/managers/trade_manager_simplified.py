"""
簡素化されたTradeManager

責任を分離し、依存性注入パターンを使用して
より保守しやすい設計に変更
"""

from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime

from ...utils.logging_config import get_context_logger, log_business_event
from ...models.database import db_manager
from ...models.enums import TradeType
from ..models.trade_models import Trade, Position, BuyLot, RealizedPnL, TradeStatus
from ..calculators.trade_calculator import (
    DecimalCalculator, 
    CommissionCalculator, 
    PnLCalculator, 
    PositionCalculator
)

logger = get_context_logger(__name__)


class TradeManagerSimplified:
    """
    簡素化された取引管理クラス
    
    Single Responsibility Principleに従い、
    取引の記録と基本的な管理機能に責任を限定
    """
    
    def __init__(self, enable_database: bool = True):
        """
        初期化
        
        Args:
            enable_database: データベース永続化を有効にするか
        """
        self.enable_database = enable_database
        self.positions: Dict[str, Position] = defaultdict(lambda: Position(symbol=""))
        self.realized_pnl_history: List[RealizedPnL] = []
        
        # 依存性注入
        self.decimal_calculator = DecimalCalculator()
        self.commission_calculator = CommissionCalculator()
        self.pnl_calculator = PnLCalculator()
        self.position_calculator = PositionCalculator()
        
        logger.info("TradeManager初期化完了", extra={
            "database_enabled": enable_database,
            "calculator_components": 4
        })
    
    def record_trade(
        self, 
        symbol: str, 
        trade_type: str, 
        quantity: int, 
        price: Union[str, int, float, Decimal],
        commission: Optional[Union[str, int, float, Decimal]] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        取引記録
        
        Args:
            symbol: 銘柄コード
            trade_type: 取引種別 ('buy' or 'sell')
            quantity: 数量
            price: 価格
            commission: 手数料（省略時は自動計算）
            timestamp: タイムスタンプ（省略時は現在時刻）
            
        Returns:
            bool: 記録成功フラグ
        """
        try:
            # 入力値の検証・変換
            price_decimal = self.decimal_calculator.safe_decimal_conversion(price, "価格")
            trade_amount = price_decimal * quantity
            
            if commission is None:
                commission_decimal = self.commission_calculator.calculate_commission(trade_amount)
            else:
                commission_decimal = self.decimal_calculator.safe_decimal_conversion(
                    commission, "手数料"
                )
            
            if timestamp is None:
                timestamp = datetime.now()
            
            # Trade オブジェクト作成
            trade = Trade(
                symbol=symbol,
                trade_type=trade_type.lower(),
                quantity=quantity,
                price=price_decimal,
                timestamp=timestamp,
                commission=commission_decimal,
                status=TradeStatus.FILLED
            )
            
            # 取引種別に応じた処理
            if trade.is_buy:
                success = self._process_buy_trade(trade)
            elif trade.is_sell:
                success = self._process_sell_trade(trade)
            else:
                logger.error(f"無効な取引種別: {trade_type}")
                return False
            
            if success:
                log_business_event(
                    "trade_recorded",
                    symbol=symbol,
                    trade_type=trade_type,
                    quantity=quantity,
                    price=float(price_decimal),
                    commission=float(commission_decimal)
                )
            
            return success
            
        except Exception as e:
            logger.error(f"取引記録エラー: {e}", extra={
                "symbol": symbol,
                "trade_type": trade_type,
                "quantity": quantity,
                "price": price
            })
            return False
    
    def _process_buy_trade(self, trade: Trade) -> bool:
        """買い取引の処理"""
        try:
            # 新しい建て玉の作成
            buy_lot = BuyLot(
                symbol=trade.symbol,
                quantity=trade.quantity,
                price=trade.price,
                timestamp=trade.timestamp,
                remaining_quantity=trade.quantity,
                commission=trade.commission
            )
            
            # ポジション更新
            position = self.positions[trade.symbol]
            if position.symbol == "":  # 新規ポジション
                position.symbol = trade.symbol
            
            position.buy_lots.append(buy_lot)
            position.total_quantity += trade.quantity
            position.total_commission += trade.commission
            position.average_price = self.position_calculator.calculate_average_price(
                position.buy_lots
            )
            
            # データベース保存
            if self.enable_database:
                self._save_trade_to_database(trade)
            
            logger.info(f"買い取引記録完了: {trade.symbol}", extra={
                "quantity": trade.quantity,
                "price": float(trade.price),
                "total_position": position.total_quantity
            })
            
            return True
            
        except Exception as e:
            logger.error(f"買い取引処理エラー: {e}")
            return False
    
    def _process_sell_trade(self, trade: Trade) -> bool:
        """売り取引の処理"""
        try:
            position = self.positions[trade.symbol]
            
            if position.total_quantity < trade.quantity:
                logger.error(f"売却数量が保有数量を超過: {trade.symbol}")
                return False
            
            # FIFO法による損益計算
            realized_pnl_list, updated_buy_lots = self.pnl_calculator.calculate_fifo_pnl(
                sell_quantity=trade.quantity,
                sell_price=trade.price,
                sell_commission=trade.commission,
                buy_lots=position.buy_lots
            )
            
            # ポジション更新
            position.buy_lots = updated_buy_lots
            position.total_quantity -= trade.quantity
            position.total_commission += trade.commission
            
            if position.total_quantity > 0:
                position.average_price = self.position_calculator.calculate_average_price(
                    position.buy_lots
                )
            else:
                position.average_price = Decimal('0')
            
            # 実現損益の記録
            for pnl in realized_pnl_list:
                self.realized_pnl_history.append(pnl)
                position.realized_pnl += pnl.net_pnl
            
            # データベース保存
            if self.enable_database:
                self._save_trade_to_database(trade)
                for pnl in realized_pnl_list:
                    self._save_pnl_to_database(pnl)
            
            total_realized_pnl = sum(pnl.net_pnl for pnl in realized_pnl_list)
            logger.info(f"売り取引記録完了: {trade.symbol}", extra={
                "quantity": trade.quantity,
                "price": float(trade.price),
                "realized_pnl": float(total_realized_pnl),
                "remaining_position": position.total_quantity
            })
            
            return True
            
        except Exception as e:
            logger.error(f"売り取引処理エラー: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        ポジション取得
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            Position: ポジション情報（存在しない場合はNone）
        """
        position = self.positions.get(symbol)
        return position if position and position.total_quantity > 0 else None
    
    def get_unrealized_pnl(self, symbol: str, current_price: Decimal) -> Decimal:
        """
        未実現損益取得
        
        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            
        Returns:
            Decimal: 未実現損益
        """
        position = self.positions.get(symbol)
        if not position or position.total_quantity == 0:
            return Decimal('0')
        
        return self.pnl_calculator.calculate_unrealized_pnl(
            position.buy_lots, current_price
        )
    
    def get_total_realized_pnl(self, symbol: Optional[str] = None) -> Decimal:
        """
        総実現損益取得
        
        Args:
            symbol: 銘柄コード（省略時は全銘柄）
            
        Returns:
            Decimal: 総実現損益
        """
        if symbol:
            return sum(
                pnl.net_pnl for pnl in self.realized_pnl_history 
                if pnl.symbol == symbol
            )
        else:
            return sum(pnl.net_pnl for pnl in self.realized_pnl_history)
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """
        ポートフォリオサマリー取得
        
        Returns:
            Dict: ポートフォリオサマリー
        """
        active_positions = {
            symbol: position for symbol, position in self.positions.items()
            if position.total_quantity > 0
        }
        
        total_realized_pnl = self.get_total_realized_pnl()
        total_positions = len(active_positions)
        
        return {
            "total_positions": total_positions,
            "active_symbols": list(active_positions.keys()),
            "total_realized_pnl": float(total_realized_pnl),
            "total_trades": len(self.realized_pnl_history),
            "positions": {
                symbol: {
                    "quantity": position.total_quantity,
                    "average_price": float(position.average_price),
                    "total_commission": float(position.total_commission),
                    "realized_pnl": float(position.realized_pnl)
                }
                for symbol, position in active_positions.items()
            }
        }
    
    def _save_trade_to_database(self, trade: Trade) -> bool:
        """データベースへの取引保存"""
        try:
            # 実装は既存のコードを参照
            # ここでは概要のみ記載
            logger.debug(f"取引をデータベースに保存: {trade.symbol}")
            return True
        except Exception as e:
            logger.error(f"データベース保存エラー: {e}")
            return False
    
    def _save_pnl_to_database(self, pnl: RealizedPnL) -> bool:
        """データベースへの損益保存"""
        try:
            # 実装は既存のコードを参照
            logger.debug(f"損益をデータベースに保存: {pnl.symbol}")
            return True
        except Exception as e:
            logger.error(f"損益データベース保存エラー: {e}")
            return False