"""
ポートフォリオ管理システム

ポートフォリオサマリーの計算など、取引履歴データに基づくビジネスロジックを提供します。
"""

from datetime import datetime as dt
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from ..models.enums import TradeType
from ..models.stock import Trade


class PortfolioManager:
    """ポートフォリオ管理クラス"""

    def __init__(self, session: Session):
        self.session = session

    def get_portfolio_summary(
        self, start_date: Optional[dt] = None
    ) -> Dict[str, Any]:
        """
        ポートフォリオサマリーを効率的に計算

        Args:
            start_date: 集計開始日

        Returns:
            Dict[str, Any]: ポートフォリオサマリー
        """
        query = self.session.query(Trade)
        if start_date:
            query = query.filter(Trade.trade_datetime >= start_date)

        trades = query.all()

        portfolio = {}
        total_cost = 0
        total_proceeds = 0

        for trade in trades:
            code = trade.stock_code
            if code not in portfolio:
                portfolio[code] = {
                    "quantity": 0,
                    "total_cost": 0,
                    "avg_price": 0,
                    "trades": [],
                }

            if trade.trade_type == TradeType.BUY:
                portfolio[code]["quantity"] += trade.quantity
                portfolio[code]["total_cost"] += trade.total_amount
                total_cost += trade.total_amount
            else:  # sell
                portfolio[code]["quantity"] -= trade.quantity
                total_proceeds += trade.total_amount

            portfolio[code]["trades"].append(trade)

        # 平均価格を計算
        for _code, data in portfolio.items():
            if data["quantity"] > 0:
                data["avg_price"] = data["total_cost"] / data["quantity"]

        return {
            "portfolio": portfolio,
            "total_cost": total_cost,
            "total_proceeds": total_proceeds,
            "net_position": total_proceeds - total_cost,
        }
