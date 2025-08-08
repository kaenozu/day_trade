#!/usr/bin/env python3
"""
メイン取引シミュレーションエンジン

Phase 4の中核システム：リアルタイム取引シミュレーション
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.advanced_ml_engine import AdvancedMLEngine
from ..data.batch_data_fetcher import BatchDataFetcher
from ..data.ultra_fast_ml_engine import UltraFastMLEngine
from ..optimization.portfolio_manager import PortfolioManager
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class OrderType(Enum):
    """注文種別"""

    MARKET = "market"  # 成行
    LIMIT = "limit"  # 指値
    STOP_LOSS = "stop_loss"  # 損切り


@dataclass
class Order:
    """注文情報"""

    symbol: str
    order_type: OrderType
    side: str  # "BUY" or "SELL"
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """ポジション情報"""

    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

    def update_price(self, new_price: float):
        """価格更新"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.avg_price) * self.quantity


@dataclass
class Trade:
    """約定情報"""

    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    pnl: float = 0.0
    commission: float = 0.0


class TradingSimulator:
    """
    デイトレード自動執行シミュレーター

    ML投資助言とポートフォリオ最適化を統合した取引シミュレーション
    """

    def __init__(
        self,
        initial_capital: float = 1000000,  # 初期資金100万円
        commission_rate: float = 0.001,  # 手数料0.1%
        max_position_per_stock: float = 0.1,  # 1銘柄最大10%
        use_ultra_fast_ml: bool = True,  # 超高速ML使用
    ):
        """
        初期化

        Args:
            initial_capital: 初期資金
            commission_rate: 手数料率
            max_position_per_stock: 1銘柄あたり最大投資比率
            use_ultra_fast_ml: 超高速MLエンジン使用
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_position_per_stock = max_position_per_stock

        # MLエンジン選択（パフォーマンス重視）
        if use_ultra_fast_ml:
            self.ml_engine = UltraFastMLEngine()
            logger.info("超高速MLエンジン使用（3.6秒/85銘柄）")
        else:
            self.ml_engine = AdvancedMLEngine(fast_mode=True)
            logger.info("高速MLエンジン使用")

        # システム構成要素
        self.data_fetcher = BatchDataFetcher(max_workers=10)
        self.portfolio_manager = PortfolioManager()

        # 取引状態管理
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.trade_history: List[Trade] = []
        self.daily_pnl: List[float] = []

        # リスク管理パラメータ
        self.stop_loss_pct = 0.05  # 5%損切り
        self.take_profit_pct = 0.10  # 10%利確
        self.max_trades_per_day = 20

        self.simulation_start_time = datetime.now()
        logger.info(f"シミュレーター初期化完了 - 初期資金: ¥{initial_capital:,.0f}")

    def run_simulation(
        self, symbols: List[str], simulation_days: int = 30, data_period: str = "60d"
    ) -> Dict:
        """
        シミュレーション実行

        Args:
            symbols: 対象銘柄リスト
            simulation_days: シミュレーション日数
            data_period: データ取得期間

        Returns:
            シミュレーション結果
        """
        logger.info(f"シミュレーション開始: {len(symbols)}銘柄 x {simulation_days}日")

        try:
            # 初期データ取得
            start_time = time.time()
            stock_data = self.data_fetcher.fetch_multiple_symbols(
                symbols, period=data_period, use_parallel=True
            )

            successful_symbols = [s for s, data in stock_data.items() if not data.empty]
            data_fetch_time = time.time() - start_time

            logger.info(
                f"データ取得完了: {len(successful_symbols)}銘柄 ({data_fetch_time:.2f}秒)"
            )

            # メインシミュレーションループ
            simulation_results = []

            for day in range(simulation_days):
                daily_result = self._simulate_trading_day(
                    successful_symbols, stock_data, day
                )
                simulation_results.append(daily_result)

                # 進捗表示
                if (day + 1) % 5 == 0:
                    current_capital = self.current_capital
                    total_pnl = current_capital - self.initial_capital
                    pnl_pct = total_pnl / self.initial_capital * 100

                    logger.info(
                        f"Day {day + 1}/{simulation_days}: "
                        f"資金¥{current_capital:,.0f} "
                        f"損益¥{total_pnl:,.0f} ({pnl_pct:+.1f}%)"
                    )

            # 最終結果
            return self._generate_simulation_report(simulation_results)

        except Exception as e:
            logger.error(f"シミュレーションエラー: {e}")
            return {"error": str(e), "status": "failed"}

    def _simulate_trading_day(
        self, symbols: List[str], stock_data: Dict[str, pd.DataFrame], day: int
    ) -> Dict:
        """
        1日の取引シミュレーション

        Args:
            symbols: 銘柄リスト
            stock_data: 株価データ
            day: 日数カウンター

        Returns:
            日次結果
        """
        try:
            # ML分析実行（超高速処理）
            ml_start = time.time()

            if hasattr(self.ml_engine, "batch_ultra_fast_analysis"):
                # UltraFastMLEngine使用
                ml_recommendations = self.ml_engine.batch_ultra_fast_analysis(
                    stock_data
                )
            else:
                # AdvancedMLEngine使用
                ml_recommendations = {}
                for symbol, data in stock_data.items():
                    if not data.empty:
                        advice = self.ml_engine.generate_fast_investment_advice(
                            symbol, data
                        )
                        ml_recommendations[symbol] = advice

            ml_time = time.time() - ml_start

            # ポートフォリオ最適化
            portfolio_start = time.time()
            portfolio_recommendations = self._get_portfolio_recommendations(
                symbols, stock_data, ml_recommendations
            )
            portfolio_time = time.time() - portfolio_start

            # 取引執行
            trade_start = time.time()
            daily_trades = self._execute_trades(portfolio_recommendations, stock_data)
            trade_time = time.time() - trade_start

            # リスク管理
            risk_actions = self._apply_risk_management(stock_data)

            # 日次損益計算
            daily_pnl = self._calculate_daily_pnl(stock_data)
            self.daily_pnl.append(daily_pnl)

            return {
                "day": day + 1,
                "ml_time": ml_time,
                "portfolio_time": portfolio_time,
                "trade_time": trade_time,
                "total_processing_time": ml_time + portfolio_time + trade_time,
                "trades_executed": len(daily_trades),
                "risk_actions": len(risk_actions),
                "daily_pnl": daily_pnl,
                "total_pnl": self.current_capital - self.initial_capital,
                "active_positions": len(self.positions),
                "ml_recommendations_count": len(ml_recommendations),
            }

        except Exception as e:
            logger.error(f"日次シミュレーションエラー Day {day + 1}: {e}")
            return {"day": day + 1, "error": str(e), "daily_pnl": 0.0}

    def _get_portfolio_recommendations(
        self,
        symbols: List[str],
        stock_data: Dict[str, pd.DataFrame],
        ml_recommendations: Dict[str, Dict],
    ) -> Dict:
        """
        ポートフォリオ推奨取得

        Args:
            symbols: 銘柄リスト
            stock_data: 株価データ
            ml_recommendations: ML推奨

        Returns:
            ポートフォリオ推奨
        """
        try:
            # ポートフォリオマネージャーで最適化実行
            optimization_result = self.portfolio_manager.generate_optimized_portfolio(
                stock_data,
                investment_amount=int(self.current_capital * 0.8),  # 80%投資
            )

            if not optimization_result or "error" in optimization_result:
                logger.warning("ポートフォリオ最適化失敗、ML推奨のみ使用")
                return ml_recommendations

            # ML推奨とポートフォリオ最適化を統合
            integrated_recommendations = {}

            for symbol in symbols:
                ml_advice = ml_recommendations.get(symbol, {})
                portfolio_weight = optimization_result.get("optimal_portfolio", {}).get(
                    symbol, 0.0
                )

                # 統合判定ロジック
                if (
                    ml_advice.get("advice") == "BUY" and portfolio_weight > 0.01
                ):  # 1%以上
                    integrated_recommendations[symbol] = {
                        "action": "BUY",
                        "ml_confidence": ml_advice.get("confidence", 50),
                        "portfolio_weight": portfolio_weight,
                        "priority": ml_advice.get("confidence", 50) * portfolio_weight,
                    }
                elif ml_advice.get("advice") == "SELL":
                    integrated_recommendations[symbol] = {
                        "action": "SELL",
                        "ml_confidence": ml_advice.get("confidence", 50),
                        "portfolio_weight": 0.0,
                        "priority": ml_advice.get("confidence", 50),
                    }

            return integrated_recommendations

        except Exception as e:
            logger.error(f"ポートフォリオ推奨エラー: {e}")
            return ml_recommendations

    def _execute_trades(
        self, recommendations: Dict[str, Dict], stock_data: Dict[str, pd.DataFrame]
    ) -> List[Trade]:
        """
        取引執行

        Args:
            recommendations: 推奨取引
            stock_data: 株価データ

        Returns:
            実行された取引リスト
        """
        executed_trades = []

        if len(self.trade_history) >= self.max_trades_per_day:
            logger.debug("日次取引限度達成")
            return executed_trades

        try:
            # 推奨を優先度順にソート
            sorted_recommendations = sorted(
                recommendations.items(),
                key=lambda x: x[1].get("priority", 0),
                reverse=True,
            )

            for symbol, rec in sorted_recommendations:
                if len(executed_trades) >= self.max_trades_per_day:
                    break

                if symbol not in stock_data or stock_data[symbol].empty:
                    continue

                current_price = stock_data[symbol]["Close"].iloc[-1]
                action = rec.get("action")

                if action == "BUY":
                    trade = self._execute_buy_order(symbol, current_price, rec)
                elif action == "SELL":
                    trade = self._execute_sell_order(symbol, current_price, rec)
                else:
                    continue

                if trade:
                    executed_trades.append(trade)
                    logger.debug(
                        f"取引執行: {trade.symbol} {trade.side} {trade.quantity}株 @¥{trade.price}"
                    )

            return executed_trades

        except Exception as e:
            logger.error(f"取引執行エラー: {e}")
            return executed_trades

    def _execute_buy_order(
        self, symbol: str, price: float, recommendation: Dict
    ) -> Optional[Trade]:
        """買い注文執行"""
        try:
            # 投資金額計算
            max_investment = self.current_capital * self.max_position_per_stock
            portfolio_weight = recommendation.get(
                "portfolio_weight", 0.05
            )  # デフォルト5%
            target_investment = min(
                max_investment, self.current_capital * portfolio_weight
            )

            if target_investment < price * 100:  # 最小100株
                return None

            quantity = int(target_investment // price)
            total_cost = quantity * price
            commission = total_cost * self.commission_rate

            if total_cost + commission > self.current_capital:
                return None

            # 注文執行
            self.current_capital -= total_cost + commission

            # ポジション更新
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_quantity = pos.quantity + quantity
                new_avg_price = (
                    (pos.avg_price * pos.quantity) + (price * quantity)
                ) / new_quantity
                pos.quantity = new_quantity
                pos.avg_price = new_avg_price
                pos.update_price(price)
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    timestamp=datetime.now(),
                )

            # 取引記録
            trade = Trade(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                commission=commission,
            )

            self.trade_history.append(trade)
            return trade

        except Exception as e:
            logger.error(f"買い注文エラー {symbol}: {e}")
            return None

    def _execute_sell_order(
        self, symbol: str, price: float, recommendation: Dict
    ) -> Optional[Trade]:
        """売り注文執行"""
        try:
            if symbol not in self.positions:
                return None

            pos = self.positions[symbol]
            if pos.quantity <= 0:
                return None

            # 全株売却
            quantity = pos.quantity
            total_proceeds = quantity * price
            commission = total_proceeds * self.commission_rate

            # 損益計算
            pnl = (price - pos.avg_price) * quantity - commission

            # 注文執行
            self.current_capital += total_proceeds - commission

            # ポジション削除
            del self.positions[symbol]

            # 取引記録
            trade = Trade(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                price=price,
                timestamp=datetime.now(),
                pnl=pnl,
                commission=commission,
            )

            self.trade_history.append(trade)
            return trade

        except Exception as e:
            logger.error(f"売り注文エラー {symbol}: {e}")
            return None

    def _apply_risk_management(self, stock_data: Dict[str, pd.DataFrame]) -> List[str]:
        """リスク管理適用"""
        risk_actions = []

        try:
            for symbol, pos in list(self.positions.items()):
                if symbol not in stock_data or stock_data[symbol].empty:
                    continue

                current_price = stock_data[symbol]["Close"].iloc[-1]
                pos.update_price(current_price)

                # 損切り判定
                loss_pct = (pos.avg_price - current_price) / pos.avg_price
                if loss_pct >= self.stop_loss_pct:
                    trade = self._execute_sell_order(
                        symbol, current_price, {"action": "STOP_LOSS"}
                    )
                    if trade:
                        risk_actions.append(f"STOP_LOSS_{symbol}")
                        logger.info(f"損切り執行: {symbol} 損失{loss_pct:.1%}")

                # 利確判定
                profit_pct = (current_price - pos.avg_price) / pos.avg_price
                if profit_pct >= self.take_profit_pct:
                    trade = self._execute_sell_order(
                        symbol, current_price, {"action": "TAKE_PROFIT"}
                    )
                    if trade:
                        risk_actions.append(f"TAKE_PROFIT_{symbol}")
                        logger.info(f"利確執行: {symbol} 利益{profit_pct:.1%}")

            return risk_actions

        except Exception as e:
            logger.error(f"リスク管理エラー: {e}")
            return risk_actions

    def _calculate_daily_pnl(self, stock_data: Dict[str, pd.DataFrame]) -> float:
        """日次損益計算"""
        try:
            total_unrealized_pnl = 0.0

            for symbol, pos in self.positions.items():
                if symbol in stock_data and not stock_data[symbol].empty:
                    current_price = stock_data[symbol]["Close"].iloc[-1]
                    pos.update_price(current_price)
                    total_unrealized_pnl += pos.unrealized_pnl

            # 実現損益（本日の取引）
            today_trades = [
                t
                for t in self.trade_history
                if t.timestamp.date() == datetime.now().date()
            ]
            realized_pnl = sum(t.pnl for t in today_trades)

            return total_unrealized_pnl + realized_pnl

        except Exception as e:
            logger.error(f"損益計算エラー: {e}")
            return 0.0

    def _generate_simulation_report(self, simulation_results: List[Dict]) -> Dict:
        """シミュレーション結果レポート生成"""
        try:
            # 基本統計
            total_pnl = self.current_capital - self.initial_capital
            total_return_pct = total_pnl / self.initial_capital * 100

            # 取引統計
            total_trades = len(self.trade_history)
            buy_trades = [t for t in self.trade_history if t.side == "BUY"]
            sell_trades = [t for t in self.trade_history if t.side == "SELL"]

            profitable_trades = [t for t in sell_trades if t.pnl > 0]
            win_rate = (
                len(profitable_trades) / len(sell_trades) * 100 if sell_trades else 0
            )

            # パフォーマンス統計
            processing_times = [
                r.get("total_processing_time", 0)
                for r in simulation_results
                if "total_processing_time" in r
            ]
            avg_processing_time = np.mean(processing_times) if processing_times else 0

            # 日次損益統計
            daily_returns = np.array(self.daily_pnl)
            max_drawdown = (
                np.min(np.cumsum(daily_returns)) if len(daily_returns) > 0 else 0
            )

            report = {
                "simulation_summary": {
                    "initial_capital": self.initial_capital,
                    "final_capital": self.current_capital,
                    "total_pnl": total_pnl,
                    "total_return_pct": total_return_pct,
                    "max_drawdown": max_drawdown,
                    "simulation_days": len(simulation_results),
                },
                "trading_statistics": {
                    "total_trades": total_trades,
                    "buy_trades": len(buy_trades),
                    "sell_trades": len(sell_trades),
                    "win_rate_pct": win_rate,
                    "profitable_trades": len(profitable_trades),
                    "average_trade_pnl": np.mean([t.pnl for t in sell_trades])
                    if sell_trades
                    else 0,
                },
                "performance_metrics": {
                    "avg_processing_time_seconds": avg_processing_time,
                    "total_ml_time": sum(
                        r.get("ml_time", 0) for r in simulation_results
                    ),
                    "total_portfolio_time": sum(
                        r.get("portfolio_time", 0) for r in simulation_results
                    ),
                    "active_positions": len(self.positions),
                },
                "daily_results": simulation_results,
                "status": "completed",
            }

            logger.info(
                f"シミュレーション完了 - 総損益: ¥{total_pnl:,.0f} ({total_return_pct:+.1f}%)"
            )
            logger.info(
                f"取引統計: {total_trades}回 勝率{win_rate:.1f}% 平均処理時間{avg_processing_time:.3f}秒"
            )

            return report

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return {"error": str(e), "status": "failed"}

    def get_current_status(self) -> Dict:
        """現在の状態取得"""
        return {
            "current_capital": self.current_capital,
            "total_pnl": self.current_capital - self.initial_capital,
            "return_pct": (self.current_capital - self.initial_capital)
            / self.initial_capital
            * 100,
            "active_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for symbol, pos in self.positions.items()
            },
        }


if __name__ == "__main__":
    # テスト実行
    simulator = TradingSimulator(initial_capital=1000000, use_ultra_fast_ml=True)

    test_symbols = ["7203", "8306", "9984", "6758", "4689"]

    result = simulator.run_simulation(
        symbols=test_symbols, simulation_days=10, data_period="30d"
    )

    print("=== シミュレーション結果 ===")
    if "simulation_summary" in result:
        summary = result["simulation_summary"]
        print(f"初期資金: ¥{summary['initial_capital']:,.0f}")
        print(f"最終資金: ¥{summary['final_capital']:,.0f}")
        print(f"総損益: ¥{summary['total_pnl']:,.0f}")
        print(f"利回り: {summary['total_return_pct']:+.2f}%")

        trading = result["trading_statistics"]
        print(f"総取引数: {trading['total_trades']}")
        print(f"勝率: {trading['win_rate_pct']:.1f}%")

        perf = result["performance_metrics"]
        print(f"平均処理時間: {perf['avg_processing_time_seconds']:.3f}秒")
    else:
        print(f"エラー: {result}")
