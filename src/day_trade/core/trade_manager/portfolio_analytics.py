"""
ポートフォリオ分析・税務計算機能
ポートフォリオのサマリー作成と税務計算を担当
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class PortfolioAnalytics:
    """ポートフォリオ分析・税務計算操作を管理するクラス"""
    
    def __init__(self, trade_manager_ref):
        """
        初期化
        
        Args:
            trade_manager_ref: TradeManagerオブジェクトへの参照
        """
        self.trade_manager = trade_manager_ref
        self.logger = get_context_logger(__name__)
    
    def get_portfolio_summary(self) -> Dict:
        """ポートフォリオサマリーを取得"""
        total_cost = sum(pos.total_cost for pos in self.trade_manager.positions.values())
        total_market_value = sum(pos.market_value for pos in self.trade_manager.positions.values())
        total_unrealized_pnl = total_market_value - total_cost
        total_realized_pnl = sum(pnl.pnl for pnl in self.trade_manager.realized_pnl)
        
        return {
            "total_positions": len(self.trade_manager.positions),
            "total_cost": str(total_cost),
            "total_market_value": str(total_market_value),
            "total_unrealized_pnl": str(total_unrealized_pnl),
            "total_realized_pnl": str(total_realized_pnl),
            "total_pnl": str(total_unrealized_pnl + total_realized_pnl),
            "total_trades": len(self.trade_manager.trades),
            "winning_trades": len([pnl for pnl in self.trade_manager.realized_pnl if pnl.pnl > 0]),
            "losing_trades": len([pnl for pnl in self.trade_manager.realized_pnl if pnl.pnl < 0]),
            "win_rate": (
                f"{(len([pnl for pnl in self.trade_manager.realized_pnl if pnl.pnl > 0]) / max(len(self.trade_manager.realized_pnl), 1) * 100):.1f}%"
                if self.trade_manager.realized_pnl
                else "0.0%"
            ),
        }
    
    def calculate_tax_implications(self, year: int) -> Dict:
        """税務計算"""
        try:
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)
            
            # 年内の実現損益
            year_pnl = [
                pnl
                for pnl in self.trade_manager.realized_pnl
                if year_start <= pnl.sell_date <= year_end
            ]
            
            total_gain = (
                sum(pnl.pnl for pnl in year_pnl if pnl.pnl > 0)
                if year_pnl
                else Decimal("0")
            )
            total_loss = (
                sum(abs(pnl.pnl) for pnl in year_pnl if pnl.pnl < 0)
                if year_pnl
                else Decimal("0")
            )
            net_gain = total_gain - total_loss
            
            # 税額計算
            tax_due = Decimal("0")
            if net_gain > 0:
                tax_due = net_gain * self.trade_manager.tax_rate
            
            return {
                "year": year,
                "total_trades": len(year_pnl),
                "total_gain": str(total_gain),
                "total_loss": str(total_loss),
                "net_gain": str(net_gain),
                "tax_due": str(tax_due),
                "winning_trades": len([pnl for pnl in year_pnl if pnl.pnl > 0]),
                "losing_trades": len([pnl for pnl in year_pnl if pnl.pnl < 0]),
            }
        
        except Exception as e:
            logger.error(
                f"税務計算中に予期せぬエラーが発生しました。入力データまたは計算ロジックを確認してください。詳細: {e}"
            )
            raise