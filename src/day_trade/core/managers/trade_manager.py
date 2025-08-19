"""
統合取引管理システム

リファクタリング後の統合インターフェース
すべての取引管理機能を統一的に提供
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from .trade_manager_execution import TradeManagerExecution
from ..models.trade_models import Trade, Position, RealizedPnL
from ..models.trade_utils import safe_decimal_conversion, validate_positive_decimal
from ...models.enums import TradeType


class TradeManager(TradeManagerExecution):
    """
    統合取引管理システム（リファクタリング版）
    
    以下の機能を統合提供：
    - 基本データ管理（TradeManagerCore）
    - 取引実行機能（TradeManagerExecution）
    - FIFO会計処理
    - データベース永続化
    - CSV/JSON エクスポート
    """

    def __init__(
        self,
        commission_rate: Decimal = Decimal("0.001"),
        tax_rate: Decimal = Decimal("0.2"),
        load_from_db: bool = False,
    ):
        """
        取引管理システムを初期化

        Args:
            commission_rate: 手数料率（デフォルト0.1%）
            tax_rate: 税率（デフォルト20%）
            load_from_db: データベースから取引履歴を読み込むかどうか
        """
        super().__init__(commission_rate, tax_rate, load_from_db)
        
        self.logger.info(
            "統合取引管理システム初期化完了",
            extra={
                "commission_rate": str(commission_rate),
                "tax_rate": str(tax_rate),
                "load_from_db": load_from_db,
                "version": "2.1_refactored",
            },
        )

    def get_trade_summary(self) -> Dict[str, any]:
        """
        取引サマリーを取得（拡張版）
        
        Returns:
            詳細な取引統計情報
        """
        base_stats = self.get_summary_stats()
        
        # 追加統計
        buy_trades = [t for t in self.trades if t.trade_type == TradeType.BUY]
        sell_trades = [t for t in self.trades if t.trade_type == TradeType.SELL]
        
        # 勝率計算
        winning_trades = [pnl for pnl in self.realized_pnl if pnl.pnl > 0]
        total_closed_trades = len(self.realized_pnl)
        win_rate = (len(winning_trades) / total_closed_trades * 100) if total_closed_trades > 0 else 0
        
        # 平均損益
        avg_pnl = (
            sum(pnl.pnl for pnl in self.realized_pnl) / total_closed_trades
            if total_closed_trades > 0
            else Decimal("0")
        )
        
        # 最大損失・利益
        max_loss = min([pnl.pnl for pnl in self.realized_pnl], default=Decimal("0"))
        max_profit = max([pnl.pnl for pnl in self.realized_pnl], default=Decimal("0"))
        
        extended_stats = {
            "buy_trades_count": len(buy_trades),
            "sell_trades_count": len(sell_trades),
            "closed_trades_count": total_closed_trades,
            "win_rate_percent": round(win_rate, 2),
            "average_pnl": str(avg_pnl),
            "max_loss": str(max_loss),
            "max_profit": str(max_profit),
            "active_positions_count": len(self.positions),
        }
        
        return {**base_stats, **extended_stats}

    def get_portfolio_performance(self) -> Dict[str, any]:
        """
        ポートフォリオパフォーマンスを取得
        
        Returns:
            パフォーマンス分析結果
        """
        total_invested = sum(
            pos.total_cost for pos in self.positions.values()
        )
        
        total_market_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        
        total_realized_pnl = sum(
            pnl.pnl for pnl in self.realized_pnl
        )
        
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        # ROI計算
        roi_percent = (
            (total_pnl / total_invested * 100) 
            if total_invested > 0 
            else Decimal("0")
        )
        
        return {
            "total_invested": str(total_invested),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_pnl),
            "roi_percent": str(roi_percent.quantize(Decimal("0.01"))),
            "positions_summary": {
                symbol: {
                    "quantity": pos.quantity,
                    "unrealized_pnl": str(pos.unrealized_pnl),
                    "unrealized_pnl_percent": str(pos.unrealized_pnl_percent.quantize(Decimal("0.01"))),
                }
                for symbol, pos in self.positions.items()
            },
        }

    def update_market_prices(self, price_updates: Dict[str, Decimal]) -> None:
        """
        市場価格を一括更新
        
        Args:
            price_updates: {symbol: current_price} の辞書
        """
        updated_count = 0
        
        for symbol, current_price in price_updates.items():
            if symbol in self.positions:
                safe_price = safe_decimal_conversion(current_price, f"{symbol}の現在価格")
                validate_positive_decimal(safe_price, f"{symbol}の現在価格")
                
                self.positions[symbol].current_price = safe_price
                updated_count += 1
        
        self.logger.info(
            "市場価格一括更新完了",
            extra={
                "updated_positions": updated_count,
                "total_updates": len(price_updates),
            },
        )

    def get_position_analytics(self, symbol: str) -> Optional[Dict[str, any]]:
        """
        特定銘柄のポジション分析
        
        Args:
            symbol: 銘柄コード
            
        Returns:
            詳細なポジション分析結果
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # 関連する取引履歴
        related_trades = [t for t in self.trades if t.symbol == symbol]
        buy_trades = [t for t in related_trades if t.trade_type == TradeType.BUY]
        sell_trades = [t for t in related_trades if t.trade_type == TradeType.SELL]
        
        # 関連する実現損益
        related_pnl = [pnl for pnl in self.realized_pnl if pnl.symbol == symbol]
        
        # 統計計算
        total_bought = sum(t.quantity for t in buy_trades)
        total_sold = sum(t.quantity for t in sell_trades)
        total_realized_pnl = sum(pnl.pnl for pnl in related_pnl)
        
        # 平均買い価格（実際の加重平均）
        total_buy_cost = sum(t.price * t.quantity + t.commission for t in buy_trades)
        avg_buy_price = total_buy_cost / total_bought if total_bought > 0 else Decimal("0")
        
        return {
            "symbol": symbol,
            "current_position": position.to_dict(),
            "trade_history": {
                "total_bought": total_bought,
                "total_sold": total_sold,
                "net_position": total_bought - total_sold,
                "buy_trades_count": len(buy_trades),
                "sell_trades_count": len(sell_trades),
            },
            "performance": {
                "avg_buy_price": str(avg_buy_price),
                "current_price": str(position.current_price),
                "unrealized_pnl": str(position.unrealized_pnl),
                "unrealized_pnl_percent": str(position.unrealized_pnl_percent.quantize(Decimal("0.01"))),
                "total_realized_pnl": str(total_realized_pnl),
                "closed_trades_count": len(related_pnl),
            },
            "lot_details": [
                {
                    "quantity": lot.quantity,
                    "price": str(lot.price),
                    "commission": str(lot.commission),
                    "timestamp": lot.timestamp.isoformat(),
                    "cost_per_share": str(lot.total_cost_per_share()),
                }
                for lot in position.buy_lots
            ],
        }

    def __str__(self) -> str:
        """文字列表現"""
        stats = self.get_summary_stats()
        return (
            f"TradeManager(trades={stats['total_trades']}, "
            f"positions={stats['total_positions']}, "
            f"total_pnl={stats['total_pnl']})"
        )

    def __repr__(self) -> str:
        """詳細文字列表現"""
        return (
            f"TradeManager(commission_rate={self.commission_rate}, "
            f"tax_rate={self.tax_rate}, "
            f"trades_count={len(self.trades)}, "
            f"positions_count={len(self.positions)})"
        )