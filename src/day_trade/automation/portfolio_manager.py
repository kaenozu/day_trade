"""
ポートフォリオ・ポジション管理システム

複数銘柄のポジション追跡、リスク管理、パフォーマンス計算を行う
包括的なポートフォリオ管理システム。
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..core.trade_manager import Trade, TradeType
from ..utils.enhanced_error_handler import get_default_error_handler
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
error_handler = get_default_error_handler()


@dataclass
class Position:
    """個別ポジション情報"""

    symbol: str
    quantity: int = 0
    average_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    market_value: Decimal = Decimal("0")

    # 損益情報
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    # コスト情報
    total_cost: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")

    # 統計情報
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    trades_count: int = 0

    # リスク指標
    max_position_value: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_profit: Decimal = Decimal("0")

    def update_market_price(self, new_price: Decimal) -> None:
        """市場価格の更新"""
        self.current_price = new_price
        self.market_value = abs(self.quantity) * new_price

        if self.quantity != 0:
            # 未実現損益の計算
            cost_basis = abs(self.quantity) * self.average_price

            if self.quantity > 0:  # ロングポジション
                self.unrealized_pnl = self.market_value - cost_basis
            else:  # ショートポジション
                self.unrealized_pnl = cost_basis - self.market_value

            self.total_pnl = self.realized_pnl + self.unrealized_pnl

            # 最大利益・最大ドローダウンの更新
            if self.unrealized_pnl > self.max_profit:
                self.max_profit = self.unrealized_pnl

            current_drawdown = self.max_profit - self.unrealized_pnl
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

    def add_trade(self, trade: Trade) -> None:
        """取引を追加してポジションを更新"""
        trade_value = trade.quantity * trade.price

        if trade.trade_type == TradeType.BUY:
            # 買い注文の場合
            if self.quantity >= 0:
                # 既存ロング/ポジション無し → ロング増加
                new_quantity = self.quantity + trade.quantity
                total_cost = (self.quantity * self.average_price) + trade_value
                self.average_price = (
                    total_cost / new_quantity if new_quantity > 0 else Decimal("0")
                )
                self.quantity = new_quantity
                self.total_cost += trade_value
            else:
                # 既存ショート → ショート減少
                if trade.quantity >= abs(self.quantity):
                    # ショートを完全にクローズして余りをロング
                    close_quantity = abs(self.quantity)
                    remaining_quantity = trade.quantity - close_quantity

                    # ショートクローズの実現損益
                    close_pnl = close_quantity * (self.average_price - trade.price)
                    self.realized_pnl += close_pnl

                    # 残りをロングポジション
                    if remaining_quantity > 0:
                        self.quantity = remaining_quantity
                        self.average_price = trade.price
                        self.total_cost = remaining_quantity * trade.price
                    else:
                        self.quantity = 0
                        self.average_price = Decimal("0")
                        self.total_cost = Decimal("0")
                else:
                    # ショート部分クローズ
                    close_pnl = trade.quantity * (self.average_price - trade.price)
                    self.realized_pnl += close_pnl
                    self.quantity += trade.quantity  # quantity は負数なので加算で減少

        else:  # TradeType.SELL
            # 売り注文の場合
            if self.quantity <= 0:
                # 既存ショート/ポジション無し → ショート増加
                new_quantity = self.quantity - trade.quantity
                total_cost = abs(self.quantity * self.average_price) + trade_value
                self.average_price = (
                    total_cost / abs(new_quantity)
                    if new_quantity != 0
                    else Decimal("0")
                )
                self.quantity = new_quantity
                self.total_cost += trade_value
            else:
                # 既存ロング → ロング減少
                if trade.quantity >= self.quantity:
                    # ロングを完全にクローズして余りをショート
                    close_quantity = self.quantity
                    remaining_quantity = trade.quantity - close_quantity

                    # ロングクローズの実現損益
                    close_pnl = close_quantity * (trade.price - self.average_price)
                    self.realized_pnl += close_pnl

                    # 残りをショートポジション
                    if remaining_quantity > 0:
                        self.quantity = -remaining_quantity
                        self.average_price = trade.price
                        self.total_cost = remaining_quantity * trade.price
                    else:
                        self.quantity = 0
                        self.average_price = Decimal("0")
                        self.total_cost = Decimal("0")
                else:
                    # ロング部分クローズ
                    close_pnl = trade.quantity * (trade.price - self.average_price)
                    self.realized_pnl += close_pnl
                    self.quantity -= trade.quantity

        # 手数料を追加
        self.total_commission += trade.commission
        self.realized_pnl -= trade.commission

        # 統計情報の更新
        self.trades_count += 1
        if not self.first_trade_time:
            self.first_trade_time = trade.timestamp
        self.last_trade_time = trade.timestamp

        # 最大ポジション価値の更新
        position_value = abs(self.quantity) * trade.price
        if position_value > self.max_position_value:
            self.max_position_value = position_value

    def is_long(self) -> bool:
        """ロングポジションかどうか"""
        return self.quantity > 0

    def is_short(self) -> bool:
        """ショートポジションかどうか"""
        return self.quantity < 0

    def is_flat(self) -> bool:
        """ポジション無しかどうか"""
        return self.quantity == 0

    def get_exposure(self) -> Decimal:
        """エクスポージャー（絶対価値）"""
        return abs(self.market_value)


@dataclass
class PortfolioSummary:
    """ポートフォリオサマリー"""

    total_market_value: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    total_cash: Decimal = Decimal("0")
    total_equity: Decimal = Decimal("0")

    # 損益情報
    total_unrealized_pnl: Decimal = Decimal("0")
    total_realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    # 手数料
    total_commission: Decimal = Decimal("0")

    # リスク指標
    total_exposure: Decimal = Decimal("0")
    long_exposure: Decimal = Decimal("0")
    short_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")

    # ポジション統計
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0

    # パフォーマンス
    total_return: float = 0.0
    daily_pnl: Decimal = Decimal("0")


class PortfolioManager:
    """
    ポートフォリオ管理システム

    機能:
    1. 複数銘柄のポジション追跡
    2. リアルタイム損益計算
    3. リスクエクスポージャー管理
    4. パフォーマンス分析
    5. ドローダウン監視
    """

    def __init__(self, initial_cash: Decimal = Decimal("1000000")):
        self.initial_cash = initial_cash
        self.current_cash = initial_cash

        # ポジション管理
        self.positions: Dict[str, Position] = {}

        # 取引履歴
        self.trade_history: List[Trade] = []

        # 日次記録
        self.daily_snapshots: Dict[str, PortfolioSummary] = {}

        # パフォーマンス追跡
        self.equity_curve: List[Tuple[datetime, Decimal]] = []
        self.max_equity: Decimal = initial_cash
        self.max_drawdown: Decimal = Decimal("0")
        self.max_drawdown_duration: int = 0

        # リスク制限
        self.max_position_size: Decimal = initial_cash * Decimal("0.1")  # 10%
        self.max_total_exposure: Decimal = initial_cash * Decimal("2.0")  # 200%
        self.max_sector_exposure: Dict[str, Decimal] = {}

        logger.info(f"ポートフォリオ管理システム初期化 - 初期資金: {initial_cash:,}")

    def add_trade(self, trade: Trade) -> None:
        """取引を追加"""
        try:
            symbol = trade.symbol

            # ポジションが存在しない場合は作成
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)

            # ポジションに取引を追加
            position = self.positions[symbol]
            position.add_trade(trade)

            # キャッシュの更新
            if trade.trade_type == TradeType.BUY:
                self.current_cash -= trade.quantity * trade.price + trade.commission
            else:
                self.current_cash += trade.quantity * trade.price - trade.commission

            # 取引履歴に追加
            self.trade_history.append(trade)

            # エクイティカーブの更新
            self._update_equity_curve()

            logger.info(
                f"取引追加: {symbol} {trade.trade_type.value} {trade.quantity}株 "
                f"@{trade.price} (ポジション: {position.quantity}株)"
            )

        except Exception as e:
            logger.error(f"取引追加エラー: {e}")
            error_handler.handle_error(e, context={"trade": trade})

    def update_market_prices(self, price_data: Dict[str, Decimal]) -> None:
        """市場価格の一括更新"""
        try:
            for symbol, price in price_data.items():
                if symbol in self.positions:
                    self.positions[symbol].update_market_price(price)

            self._update_equity_curve()

        except Exception as e:
            logger.error(f"市場価格更新エラー: {e}")
            error_handler.handle_error(e, context={"price_data": price_data})

    def get_position(self, symbol: str) -> Optional[Position]:
        """個別ポジション取得"""
        return self.positions.get(symbol)

    def get_all_positions(self, include_flat: bool = False) -> Dict[str, Position]:
        """全ポジション取得"""
        if include_flat:
            return self.positions.copy()

        return {
            symbol: position
            for symbol, position in self.positions.items()
            if not position.is_flat()
        }

    def get_portfolio_summary(self) -> PortfolioSummary:
        """ポートフォリオサマリー取得"""
        summary = PortfolioSummary()

        # 現金
        summary.total_cash = self.current_cash

        for position in self.positions.values():
            # 市場価値
            summary.total_market_value += position.market_value
            summary.total_cost += position.total_cost

            # 損益
            summary.total_unrealized_pnl += position.unrealized_pnl
            summary.total_realized_pnl += position.realized_pnl
            summary.total_commission += position.total_commission

            # エクスポージャー
            exposure = position.get_exposure()
            summary.total_exposure += exposure

            if position.is_long():
                summary.long_exposure += exposure
                summary.long_positions += 1
            elif position.is_short():
                summary.short_exposure += exposure
                summary.short_positions += 1

            if not position.is_flat():
                summary.total_positions += 1

        # 純エクスポージャー
        summary.net_exposure = summary.long_exposure - summary.short_exposure

        # 総損益
        summary.total_pnl = summary.total_unrealized_pnl + summary.total_realized_pnl

        # 総資産
        summary.total_equity = summary.total_cash + summary.total_market_value

        # リターン
        if self.initial_cash > 0:
            summary.total_return = float(
                (summary.total_equity - self.initial_cash) / self.initial_cash * 100
            )

        return summary

    def check_risk_limits(self) -> Dict[str, Any]:
        """リスク制限チェック"""
        summary = self.get_portfolio_summary()
        violations = []

        # 最大エクスポージャーチェック
        if summary.total_exposure > self.max_total_exposure:
            violations.append(
                {
                    "type": "max_exposure",
                    "current": float(summary.total_exposure),
                    "limit": float(self.max_total_exposure),
                    "violation_ratio": float(
                        summary.total_exposure / self.max_total_exposure
                    ),
                }
            )

        # 個別ポジションサイズチェック
        for symbol, position in self.positions.items():
            if position.get_exposure() > self.max_position_size:
                violations.append(
                    {
                        "type": "position_size",
                        "symbol": symbol,
                        "current": float(position.get_exposure()),
                        "limit": float(self.max_position_size),
                        "violation_ratio": float(
                            position.get_exposure() / self.max_position_size
                        ),
                    }
                )

        return {
            "violations": violations,
            "total_violations": len(violations),
            "risk_score": len(violations)
            / (len(self.positions) + 1),  # 正規化リスクスコア
        }

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """パフォーマンス指標計算"""
        if len(self.equity_curve) < 2:
            return {}

        try:
            # 直近のデータを取得
            recent_data = self.equity_curve[-days:] if days > 0 else self.equity_curve

            if len(recent_data) < 2:
                return {}

            # 日次リターンを計算
            returns = []
            for i in range(1, len(recent_data)):
                prev_equity = recent_data[i - 1][1]
                curr_equity = recent_data[i][1]
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(float(daily_return))

            if not returns:
                return {}

            # 統計値計算
            import statistics

            mean_return = statistics.mean(returns)
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0

            # シャープレシオ（リスクフリーレート = 0 と仮定）
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0

            # 最大ドローダウン
            max_dd = 0.0
            peak = recent_data[0][1]
            for _, equity in recent_data:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, drawdown)

            # 勝率計算
            winning_trades = [r for r in returns if r > 0]
            win_rate = len(winning_trades) / len(returns) * 100 if returns else 0.0

            return {
                "total_trades": len(self.trade_history),
                "avg_daily_return": mean_return * 100,
                "volatility": volatility * 100,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_dd * 100,
                "win_rate": win_rate,
                "current_equity": float(recent_data[-1][1]),
                "period_return": (
                    float(
                        (recent_data[-1][1] - recent_data[0][1])
                        / recent_data[0][1]
                        * 100
                    )
                    if recent_data[0][1] > 0
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"パフォーマンス指標計算エラー: {e}")
            return {}

    def get_position_breakdown(self) -> List[Dict[str, Any]]:
        """ポジション内訳取得"""
        breakdown = []

        for symbol, position in self.positions.items():
            if not position.is_flat():
                breakdown.append(
                    {
                        "symbol": symbol,
                        "quantity": position.quantity,
                        "side": "Long" if position.is_long() else "Short",
                        "average_price": float(position.average_price),
                        "current_price": float(position.current_price),
                        "market_value": float(position.market_value),
                        "unrealized_pnl": float(position.unrealized_pnl),
                        "unrealized_pnl_pct": (
                            float(position.unrealized_pnl / position.total_cost * 100)
                            if position.total_cost > 0
                            else 0.0
                        ),
                        "exposure": float(position.get_exposure()),
                        "trades_count": position.trades_count,
                    }
                )

        # 損益順でソート
        breakdown.sort(key=lambda x: x["unrealized_pnl"], reverse=True)
        return breakdown

    def close_position(self, symbol: str, price: Decimal) -> Optional[Trade]:
        """ポジションをクローズ"""
        if symbol not in self.positions:
            logger.warning(f"ポジションが存在しません: {symbol}")
            return None

        position = self.positions[symbol]

        if position.is_flat():
            logger.warning(f"既にフラットなポジション: {symbol}")
            return None

        try:
            # クローズ取引を作成
            close_trade = Trade(
                id=f"close_{symbol}_{int(datetime.now().timestamp() * 1000)}",
                symbol=symbol,
                trade_type=TradeType.SELL if position.is_long() else TradeType.BUY,
                quantity=abs(position.quantity),
                price=price,
                timestamp=datetime.now(),
                commission=Decimal("0"),  # 手数料は別途設定
                status="executed",
            )

            self.add_trade(close_trade)

            logger.info(
                f"ポジションクローズ: {symbol} - 実現損益: {position.realized_pnl}"
            )

            return close_trade

        except Exception as e:
            logger.error(f"ポジションクローズエラー: {e}")
            return None

    def close_all_positions(self, price_data: Dict[str, Decimal]) -> List[Trade]:
        """全ポジションをクローズ"""
        closed_trades = []

        for symbol, position in self.positions.items():
            if not position.is_flat() and symbol in price_data:
                trade = self.close_position(symbol, price_data[symbol])
                if trade:
                    closed_trades.append(trade)

        logger.info(f"全ポジションクローズ: {len(closed_trades)}件")
        return closed_trades

    def _update_equity_curve(self) -> None:
        """エクイティカーブの更新"""
        try:
            summary = self.get_portfolio_summary()
            current_time = datetime.now()

            # エクイティカーブに追加
            self.equity_curve.append((current_time, summary.total_equity))

            # 最大エクイティの更新
            if summary.total_equity > self.max_equity:
                self.max_equity = summary.total_equity

            # 最大ドローダウンの更新
            current_drawdown = (
                self.max_equity - summary.total_equity
            ) / self.max_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # 古いデータを削除（メモリ効率化）
            if len(self.equity_curve) > 10000:  # 直近10000ポイントのみ保持
                self.equity_curve = self.equity_curve[-5000:]

        except Exception as e:
            logger.error(f"エクイティカーブ更新エラー: {e}")

    def create_daily_snapshot(self) -> None:
        """日次スナップショット作成"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            summary = self.get_portfolio_summary()
            self.daily_snapshots[today] = summary

            logger.info(
                f"日次スナップショット作成: {today} - 総資産: {summary.total_equity}"
            )

        except Exception as e:
            logger.error(f"日次スナップショット作成エラー: {e}")
