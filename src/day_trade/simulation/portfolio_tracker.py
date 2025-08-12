#!/usr/bin/env python3
"""
ポートフォリオ追跡・損益計算システム

Phase 4: リアルタイム損益計算とパフォーマンス分析
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class PortfolioPosition:
    """ポートフォリオポジション"""

    symbol: str
    quantity: int
    avg_purchase_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    purchase_date: datetime
    last_update: datetime

    def update_price(self, new_price: float):
        """価格更新"""
        self.current_price = new_price
        self.market_value = self.quantity * new_price
        self.unrealized_pnl = self.market_value - (
            self.quantity * self.avg_purchase_price
        )
        self.unrealized_pnl_pct = (
            self.unrealized_pnl / (self.quantity * self.avg_purchase_price) * 100
        )
        self.last_update = datetime.now()


@dataclass
class Transaction:
    """取引記録"""

    transaction_id: str
    symbol: str
    transaction_type: str  # "BUY" or "SELL"
    quantity: int
    price: float
    total_amount: float
    commission: float
    timestamp: datetime
    strategy: str = "unknown"
    pnl: float = 0.0  # 売却時のみ

    def to_dict(self) -> Dict:
        """辞書形式変換"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class DailyPerformance:
    """日次パフォーマンス"""

    date: datetime
    portfolio_value: float
    cash_balance: float
    total_value: float
    daily_pnl: float
    daily_pnl_pct: float
    realized_pnl: float
    unrealized_pnl: float
    transaction_count: int
    active_positions: int

    def to_dict(self) -> Dict:
        """辞書形式変換"""
        result = asdict(self)
        result["date"] = self.date.isoformat()
        return result


class PortfolioTracker:
    """
    ポートフォリオ追跡システム

    リアルタイム損益計算、パフォーマンス分析、取引履歴管理
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.001,
        tax_rate: float = 0.20315,  # 20.315%(所得税+住民税+復興特別所得税)
        output_dir: str = "portfolio_tracking",
    ):
        """
        初期化

        Args:
            initial_capital: 初期資金
            commission_rate: 手数料率
            tax_rate: 税率
            output_dir: 出力ディレクトリ
        """
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate

        # ポートフォリオ状態
        self.positions: Dict[str, PortfolioPosition] = {}
        self.transactions: List[Transaction] = []
        self.daily_performance: List[DailyPerformance] = []

        # 累計統計
        self.total_realized_pnl = 0.0
        self.total_commission_paid = 0.0
        self.total_tax_paid = 0.0
        self.transaction_counter = 0

        # 出力設定
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 開始時刻
        self.start_date = datetime.now()

        logger.info(f"ポートフォリオ追跡開始 - 初期資金: ¥{initial_capital:,.0f}")

    def execute_buy_transaction(
        self, symbol: str, quantity: int, price: float, strategy: str = "unknown"
    ) -> Optional[Transaction]:
        """
        買い取引実行

        Args:
            symbol: 銘柄コード
            quantity: 株数
            price: 価格
            strategy: 戦略名

        Returns:
            取引オブジェクト
        """
        try:
            total_amount = quantity * price
            commission = total_amount * self.commission_rate
            total_cost = total_amount + commission

            # 資金チェック
            if total_cost > self.cash_balance:
                logger.warning(
                    f"資金不足: 必要¥{total_cost:,.0f} > 残高¥{self.cash_balance:,.0f}"
                )
                return None

            # 取引実行
            self.cash_balance -= total_cost
            self.total_commission_paid += commission
            self.transaction_counter += 1

            # 取引記録
            transaction = Transaction(
                transaction_id=f"TXN_{self.transaction_counter:06d}",
                symbol=symbol,
                transaction_type="BUY",
                quantity=quantity,
                price=price,
                total_amount=total_amount,
                commission=commission,
                timestamp=datetime.now(),
                strategy=strategy,
            )

            self.transactions.append(transaction)

            # ポジション更新
            self._update_position_buy(symbol, quantity, price)

            logger.info(
                f"買い取引実行: {symbol} {quantity}株 @¥{price:,.0f} (手数料¥{commission:,.0f})"
            )
            return transaction

        except Exception as e:
            logger.error(f"買い取引エラー {symbol}: {e}")
            return None

    def execute_sell_transaction(
        self, symbol: str, quantity: int, price: float, strategy: str = "unknown"
    ) -> Optional[Transaction]:
        """
        売り取引実行

        Args:
            symbol: 銘柄コード
            quantity: 株数
            price: 価格
            strategy: 戦略名

        Returns:
            取引オブジェクト
        """
        try:
            # ポジションチェック
            if symbol not in self.positions:
                logger.warning(f"ポジション無し: {symbol}")
                return None

            position = self.positions[symbol]
            if position.quantity < quantity:
                logger.warning(
                    f"保有株数不足: {symbol} 要求{quantity} > 保有{position.quantity}"
                )
                return None

            # 売却計算
            total_amount = quantity * price
            commission = total_amount * self.commission_rate

            # 損益計算
            purchase_cost = quantity * position.avg_purchase_price
            gross_pnl = total_amount - purchase_cost
            net_proceeds = total_amount - commission

            # 税金計算（利益がある場合のみ）
            tax = max(0, gross_pnl * self.tax_rate) if gross_pnl > 0 else 0
            net_pnl = gross_pnl - commission - tax

            # 取引実行
            self.cash_balance += net_proceeds - tax
            self.total_commission_paid += commission
            self.total_tax_paid += tax
            self.total_realized_pnl += net_pnl
            self.transaction_counter += 1

            # 取引記録
            transaction = Transaction(
                transaction_id=f"TXN_{self.transaction_counter:06d}",
                symbol=symbol,
                transaction_type="SELL",
                quantity=quantity,
                price=price,
                total_amount=total_amount,
                commission=commission,
                timestamp=datetime.now(),
                strategy=strategy,
                pnl=net_pnl,
            )

            self.transactions.append(transaction)

            # ポジション更新
            self._update_position_sell(symbol, quantity)

            logger.info(
                f"売り取引実行: {symbol} {quantity}株 @¥{price:,.0f} "
                f"損益¥{net_pnl:,.0f} (手数料¥{commission:,.0f} 税金¥{tax:,.0f})"
            )
            return transaction

        except Exception as e:
            logger.error(f"売り取引エラー {symbol}: {e}")
            return None

    def _update_position_buy(self, symbol: str, quantity: int, price: float):
        """買いポジション更新"""
        if symbol in self.positions:
            # 既存ポジション追加
            pos = self.positions[symbol]
            new_quantity = pos.quantity + quantity
            new_avg_price = (
                (pos.avg_purchase_price * pos.quantity) + (price * quantity)
            ) / new_quantity

            pos.quantity = new_quantity
            pos.avg_purchase_price = new_avg_price
            pos.update_price(price)

        else:
            # 新規ポジション
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=quantity,
                avg_purchase_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                purchase_date=datetime.now(),
                last_update=datetime.now(),
            )

    def _update_position_sell(self, symbol: str, quantity: int):
        """売りポジション更新"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.quantity -= quantity

        # 全株売却の場合はポジション削除
        if position.quantity <= 0:
            del self.positions[symbol]
        else:
            position.market_value = position.quantity * position.current_price
            position.unrealized_pnl = position.market_value - (
                position.quantity * position.avg_purchase_price
            )
            position.unrealized_pnl_pct = (
                position.unrealized_pnl
                / (position.quantity * position.avg_purchase_price)
                * 100
            )

    def update_market_prices(self, price_data: Dict[str, float]):
        """市場価格更新"""
        try:
            for symbol, new_price in price_data.items():
                if symbol in self.positions:
                    self.positions[symbol].update_price(new_price)

            logger.debug(f"価格更新: {len(price_data)}銘柄")

        except Exception as e:
            logger.error(f"価格更新エラー: {e}")

    def calculate_portfolio_metrics(self) -> Dict:
        """ポートフォリオ指標計算"""
        try:
            # 基本指標
            total_market_value = sum(
                pos.market_value for pos in self.positions.values()
            )
            total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.positions.values()
            )
            total_portfolio_value = self.cash_balance + total_market_value

            # 収益率
            total_return = total_portfolio_value - self.initial_capital
            total_return_pct = total_return / self.initial_capital * 100

            # リスク指標（日次パフォーマンスベース）
            if len(self.daily_performance) >= 2:
                daily_returns = [
                    p.daily_pnl_pct
                    for p in self.daily_performance[-30:]  # 過去30日
                    if p.daily_pnl_pct is not None
                ]
                if daily_returns:
                    volatility = np.std(daily_returns)
                    sharpe_ratio = (
                        np.mean(daily_returns) / volatility if volatility > 0 else 0
                    )
                else:
                    volatility = 0
                    sharpe_ratio = 0
            else:
                volatility = 0
                sharpe_ratio = 0

            # 最大ドローダウン
            if self.daily_performance:
                portfolio_values = [p.total_value for p in self.daily_performance]
                peak = portfolio_values[0]
                max_drawdown = 0

                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)

                max_drawdown_pct = max_drawdown * 100
            else:
                max_drawdown_pct = 0

            return {
                "total_portfolio_value": total_portfolio_value,
                "cash_balance": self.cash_balance,
                "total_market_value": total_market_value,
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "realized_pnl": self.total_realized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "total_commission_paid": self.total_commission_paid,
                "total_tax_paid": self.total_tax_paid,
                "active_positions": len(self.positions),
                "total_transactions": len(self.transactions),
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "days_active": (datetime.now() - self.start_date).days,
            }

        except Exception as e:
            logger.error(f"指標計算エラー: {e}")
            return {}

    def record_daily_performance(self) -> DailyPerformance:
        """日次パフォーマンス記録"""
        try:
            metrics = self.calculate_portfolio_metrics()

            # 前日比較
            previous_value = (
                self.daily_performance[-1].total_value
                if self.daily_performance
                else self.initial_capital
            )

            daily_pnl = metrics["total_portfolio_value"] - previous_value
            daily_pnl_pct = (
                daily_pnl / previous_value * 100 if previous_value > 0 else 0
            )

            # 本日の取引数
            today = datetime.now().date()
            today_transactions = [
                t for t in self.transactions if t.timestamp.date() == today
            ]

            performance = DailyPerformance(
                date=datetime.now(),
                portfolio_value=metrics["total_market_value"],
                cash_balance=self.cash_balance,
                total_value=metrics["total_portfolio_value"],
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                realized_pnl=metrics["realized_pnl"],
                unrealized_pnl=metrics["unrealized_pnl"],
                transaction_count=len(today_transactions),
                active_positions=len(self.positions),
            )

            self.daily_performance.append(performance)

            # 古い記録削除（メモリ管理）
            if len(self.daily_performance) > 365:
                self.daily_performance = self.daily_performance[-300:]

            logger.info(
                f"日次記録: 資産¥{metrics['total_portfolio_value']:,.0f} "
                f"日次損益¥{daily_pnl:,.0f} ({daily_pnl_pct:+.2f}%)"
            )

            return performance

        except Exception as e:
            logger.error(f"日次記録エラー: {e}")
            return None

    def get_position_summary(self) -> Dict:
        """ポジション要約"""
        if not self.positions:
            return {"total_positions": 0, "positions": []}

        positions_data = []
        for symbol, pos in self.positions.items():
            positions_data.append(
                {
                    "symbol": symbol,
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_purchase_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "days_held": (datetime.now() - pos.purchase_date).days,
                }
            )

        return {
            "total_positions": len(self.positions),
            "positions": sorted(
                positions_data, key=lambda x: x["market_value"], reverse=True
            ),
        }

    def generate_performance_report(self) -> Dict:
        """パフォーマンスレポート生成"""
        try:
            metrics = self.calculate_portfolio_metrics()
            position_summary = self.get_position_summary()

            # 取引統計
            buy_transactions = [
                t for t in self.transactions if t.transaction_type == "BUY"
            ]
            sell_transactions = [
                t for t in self.transactions if t.transaction_type == "SELL"
            ]
            profitable_trades = [t for t in sell_transactions if t.pnl > 0]

            win_rate = (
                len(profitable_trades) / len(sell_transactions) * 100
                if sell_transactions
                else 0
            )
            avg_profit = (
                np.mean([t.pnl for t in profitable_trades]) if profitable_trades else 0
            )
            avg_loss = np.mean([t.pnl for t in sell_transactions if t.pnl < 0])
            avg_loss = avg_loss if not np.isnan(avg_loss) else 0

            report = {
                "report_date": datetime.now().isoformat(),
                "portfolio_metrics": metrics,
                "position_summary": position_summary,
                "trading_statistics": {
                    "total_transactions": len(self.transactions),
                    "buy_transactions": len(buy_transactions),
                    "sell_transactions": len(sell_transactions),
                    "win_rate_pct": win_rate,
                    "profitable_trades": len(profitable_trades),
                    "avg_profit": avg_profit,
                    "avg_loss": avg_loss,
                    "profit_loss_ratio": (
                        abs(avg_profit / avg_loss) if avg_loss != 0 else 0
                    ),
                },
                "recent_performance": [
                    p.to_dict()
                    for p in self.daily_performance[-10:]  # 直近10日
                ],
            }

            return report

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return {"error": str(e)}

    def save_performance_data(self, filename: str = None):
        """パフォーマンスデータ保存"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"portfolio_performance_{timestamp}.json"

            filepath = self.output_dir / filename

            # 保存データ準備
            save_data = {
                "metadata": {
                    "initial_capital": self.initial_capital,
                    "start_date": self.start_date.isoformat(),
                    "commission_rate": self.commission_rate,
                    "tax_rate": self.tax_rate,
                },
                "performance_report": self.generate_performance_report(),
                "transactions": [t.to_dict() for t in self.transactions],
                "daily_performance": [p.to_dict() for p in self.daily_performance],
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"パフォーマンスデータ保存: {filepath}")

        except Exception as e:
            logger.error(f"データ保存エラー: {e}")


if __name__ == "__main__":
    # テスト実行
    tracker = PortfolioTracker(initial_capital=1000000)

    # サンプル取引
    print("=== ポートフォリオ追跡テスト ===")

    # 買い取引
    buy_txn = tracker.execute_buy_transaction("7203", 1000, 2500, "ML_BASED")
    if buy_txn:
        print(f"買い取引: {buy_txn.symbol} {buy_txn.quantity}株 @¥{buy_txn.price}")

    # 価格更新
    tracker.update_market_prices({"7203": 2600})

    # メトリクス確認
    metrics = tracker.calculate_portfolio_metrics()
    print(f"総資産: ¥{metrics['total_portfolio_value']:,.0f}")
    print(
        f"総収益: ¥{metrics['total_return']:,.0f} ({metrics['total_return_pct']:+.2f}%)"
    )

    # 売り取引
    sell_txn = tracker.execute_sell_transaction("7203", 500, 2600, "PROFIT_TAKING")
    if sell_txn:
        print(
            f"売り取引: {sell_txn.symbol} {sell_txn.quantity}株 損益¥{sell_txn.pnl:,.0f}"
        )

    # 日次記録
    daily_perf = tracker.record_daily_performance()
    if daily_perf:
        print(f"日次損益: ¥{daily_perf.daily_pnl:,.0f}")

    # レポート生成
    report = tracker.generate_performance_report()
    trading_stats = report["trading_statistics"]
    print(f"勝率: {trading_stats['win_rate_pct']:.1f}%")

    # データ保存
    tracker.save_performance_data()
    print("パフォーマンスデータ保存完了")
