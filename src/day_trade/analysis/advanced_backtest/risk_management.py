"""
高度なバックテストエンジン - リスク管理システム

ポートフォリオリスクの監視と制御を管理。
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from day_trade.analysis.events import Order, OrderType
from day_trade.utils.logging_config import get_context_logger
from .position_management import PositionManager

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class RiskManager:
    """リスク管理システム"""

    def __init__(
        self,
        max_daily_loss_limit: Optional[float] = None,
        max_portfolio_heat: float = 0.02,  # 2%
        max_position_concentration: float = 0.3,  # 30%
        stop_loss_percentage: float = 0.05,  # 5%
        take_profit_percentage: float = 0.15,  # 15%
    ):
        """リスク管理システムの初期化"""
        self.max_daily_loss_limit = max_daily_loss_limit
        self.max_portfolio_heat = max_portfolio_heat
        self.max_position_concentration = max_position_concentration
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage

        # リスク監視用の履歴データ
        self.daily_equity_history: List[float] = []
        self.risk_alerts: List[Dict] = []

    def check_risk_limits(
        self, 
        position_manager: PositionManager,
        current_date: datetime,
        equity_curve: List[float]
    ) -> List[Order]:
        """リスク制限のチェックと必要な注文の生成"""
        risk_orders = []
        portfolio_value = position_manager.get_portfolio_value()

        # 最大日次損失制限チェック
        daily_loss_orders = self._check_daily_loss_limit(
            position_manager, portfolio_value, equity_curve, current_date
        )
        risk_orders.extend(daily_loss_orders)

        # ポートフォリオ熱度チェック
        heat_orders = self._check_portfolio_heat(
            position_manager, portfolio_value, current_date
        )
        risk_orders.extend(heat_orders)

        # ポジション集中度チェック
        concentration_orders = self._check_position_concentration(
            position_manager, portfolio_value, current_date
        )
        risk_orders.extend(concentration_orders)

        return risk_orders

    def _check_daily_loss_limit(
        self,
        position_manager: PositionManager,
        portfolio_value: float,
        equity_curve: List[float],
        current_date: datetime
    ) -> List[Order]:
        """最大日次損失制限のチェック"""
        if not self.max_daily_loss_limit or not equity_curve:
            return []

        daily_pnl = portfolio_value - equity_curve[-1]
        
        if daily_pnl < -self.max_daily_loss_limit:
            logger.warning(
                f"日次損失制限を超過: {daily_pnl:.2f}",
                section="risk_management",
                limit=self.max_daily_loss_limit
            )
            
            self.risk_alerts.append({
                "timestamp": current_date,
                "type": "daily_loss_limit_exceeded",
                "value": daily_pnl,
                "limit": self.max_daily_loss_limit
            })

            return self._generate_close_all_orders(position_manager, current_date)

        return []

    def _check_portfolio_heat(
        self,
        position_manager: PositionManager,
        portfolio_value: float,
        current_date: datetime
    ) -> List[Order]:
        """ポートフォリオ熱度のチェック"""
        total_heat = position_manager.calculate_position_heat(portfolio_value)

        if total_heat > self.max_portfolio_heat:
            logger.warning(
                f"ポートフォリオ熱度制限を超過: {total_heat:.4f}",
                section="risk_management",
                limit=self.max_portfolio_heat
            )

            self.risk_alerts.append({
                "timestamp": current_date,
                "type": "portfolio_heat_exceeded",
                "value": total_heat,
                "limit": self.max_portfolio_heat
            })

            return self._generate_risk_reduction_orders(position_manager, current_date)

        return []

    def _check_position_concentration(
        self,
        position_manager: PositionManager,
        portfolio_value: float,
        current_date: datetime
    ) -> List[Order]:
        """ポジション集中度のチェック"""
        concentration_orders = []

        for symbol, position in position_manager.positions.items():
            if position.quantity == 0:
                continue

            position_weight = position.market_value / portfolio_value
            
            if position_weight > self.max_position_concentration:
                logger.warning(
                    f"ポジション集中度制限を超過: {symbol} {position_weight:.4f}",
                    section="risk_management",
                    limit=self.max_position_concentration
                )

                self.risk_alerts.append({
                    "timestamp": current_date,
                    "type": "position_concentration_exceeded",
                    "symbol": symbol,
                    "value": position_weight,
                    "limit": self.max_position_concentration
                })

                # 集中度を制限内に収めるための部分決済注文を生成
                target_quantity = int(position.quantity * self.max_position_concentration / position_weight)
                reduce_quantity = position.quantity - target_quantity

                if reduce_quantity > 0:
                    order = Order(
                        order_id=f"concentration_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        side="sell",
                        quantity=reduce_quantity,
                        timestamp=current_date,
                    )
                    concentration_orders.append(order)

        return concentration_orders

    def _generate_close_all_orders(
        self, position_manager: PositionManager, current_date: datetime
    ) -> List[Order]:
        """全ポジション決済注文の生成"""
        close_orders = []
        
        for symbol, position in position_manager.positions.items():
            if position.quantity > 0:
                order = Order(
                    order_id=f"close_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=position.quantity,
                    timestamp=current_date,
                )
                close_orders.append(order)

        logger.info(
            f"全ポジション決済注文生成: {len(close_orders)}件",
            section="risk_management"
        )

        return close_orders

    def _generate_risk_reduction_orders(
        self, position_manager: PositionManager, current_date: datetime
    ) -> List[Order]:
        """リスク削減注文の生成"""
        # 損失の大きいポジションを特定
        risky_positions = [
            (symbol, pos)
            for symbol, pos in position_manager.positions.items()
            if pos.quantity > 0 and pos.unrealized_pnl < 0
        ]

        # 損失の大きい順にソート
        risky_positions.sort(key=lambda x: x[1].unrealized_pnl)

        reduction_orders = []
        
        # 上位のリスクポジションを半分決済
        for symbol, position in risky_positions[:len(risky_positions) // 2]:
            reduce_quantity = position.quantity // 2
            if reduce_quantity > 0:
                order = Order(
                    order_id=f"reduce_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=reduce_quantity,
                    timestamp=current_date,
                )
                reduction_orders.append(order)

        return reduction_orders

    def check_stop_loss_take_profit(
        self,
        position_manager: PositionManager,
        current_market_data: Dict[str, float],
        current_date: datetime
    ) -> List[Order]:
        """ストップロス・テイクプロフィットのチェック"""
        sl_tp_orders = []

        for symbol, position in position_manager.positions.items():
            if position.quantity == 0 or symbol not in current_market_data:
                continue

            current_price = current_market_data[symbol]
            entry_price = position.average_price
            
            # ポジションが存在しない場合はスキップ
            if entry_price == 0:
                continue

            # 損益率を計算
            pnl_ratio = (current_price - entry_price) / entry_price

            # ストップロス判定
            if pnl_ratio <= -self.stop_loss_percentage:
                order = Order(
                    order_id=f"stop_loss_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=position.quantity,
                    timestamp=current_date,
                )
                sl_tp_orders.append(order)
                
                logger.info(
                    f"ストップロス発動: {symbol} at {pnl_ratio:.4f}",
                    section="risk_management"
                )

            # テイクプロフィット判定
            elif pnl_ratio >= self.take_profit_percentage:
                order = Order(
                    order_id=f"take_profit_{symbol}_{current_date.strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    side="sell",
                    quantity=position.quantity,
                    timestamp=current_date,
                )
                sl_tp_orders.append(order)
                
                logger.info(
                    f"テイクプロフィット発動: {symbol} at {pnl_ratio:.4f}",
                    section="risk_management"
                )

        return sl_tp_orders

    def get_risk_summary(self) -> Dict:
        """リスクサマリの取得"""
        return {
            "total_alerts": len(self.risk_alerts),
            "alert_types": list(set(alert["type"] for alert in self.risk_alerts)),
            "recent_alerts": self.risk_alerts[-5:] if self.risk_alerts else [],
            "risk_limits": {
                "max_daily_loss_limit": self.max_daily_loss_limit,
                "max_portfolio_heat": self.max_portfolio_heat,
                "max_position_concentration": self.max_position_concentration,
                "stop_loss_percentage": self.stop_loss_percentage,
                "take_profit_percentage": self.take_profit_percentage,
            }
        }