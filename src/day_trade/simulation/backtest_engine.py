#!/usr/bin/env python3
"""
バックテスト実行エンジン

Phase 4: 過去データでの戦略検証とパフォーマンス分析
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.batch_data_fetcher import BatchDataFetcher
from ..data.ultra_fast_ml_engine import UltraFastMLEngine
from ..utils.logging_config import get_context_logger
from .portfolio_tracker import PortfolioTracker
from .strategy_executor import StrategyExecutor, StrategyParameters, StrategyType

logger = get_context_logger(__name__)


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    start_date: str
    end_date: str
    initial_capital: float
    symbols: List[str]
    strategy_type: StrategyType
    rebalance_frequency: int = 5  # 日
    commission_rate: float = 0.001
    benchmark_symbol: str = "NIKKEI225"

    # 戦略パラメータ
    risk_tolerance: float = 0.6
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15


@dataclass
class BacktestResult:
    """バックテスト結果"""

    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_return: float
    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    benchmark_return_pct: float
    alpha: float  # ベンチマーク超過収益

    # 詳細データ
    daily_returns: List[float]
    portfolio_values: List[float]
    trade_history: List[Dict]
    positions_history: List[Dict]


class BacktestEngine:
    """
    バックテストエンジン

    過去データを使用した戦略検証と詳細分析
    """

    def __init__(self, output_dir: str = "backtest_results"):
        """初期化"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 高速データ取得
        self.data_fetcher = BatchDataFetcher(max_workers=10)

        logger.info("バックテストエンジン初期化完了")

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        バックテスト実行

        Args:
            config: バックテスト設定

        Returns:
            バックテスト結果
        """
        logger.info(f"バックテスト開始: {config.start_date} - {config.end_date}")
        logger.info(f"対象銘柄: {len(config.symbols)}銘柄")
        logger.info(f"戦略: {config.strategy_type.value}")

        try:
            # データ準備
            historical_data = self._prepare_historical_data(config)
            if not historical_data:
                raise ValueError("履歴データ取得に失敗")

            # 戦略パラメータ設定
            strategy_params = self._create_strategy_parameters(config)
            strategy_executor = StrategyExecutor(strategy_params)
            tracker = PortfolioTracker(
                initial_capital=config.initial_capital,
                commission_rate=config.commission_rate,
            )

            # バックテスト期間データ分割
            date_range = pd.date_range(start=config.start_date, end=config.end_date, freq="D")
            business_days = [d for d in date_range if d.weekday() < 5]  # 平日のみ

            # 実行データ記録
            portfolio_values = []
            daily_returns = []
            trade_history = []
            positions_history = []

            ml_engine = UltraFastMLEngine()

            # 日次バックテストループ
            for i, current_date in enumerate(business_days):
                try:
                    # 当日データ取得（シミュレーション）
                    day_data = self._get_daily_data(historical_data, current_date, i)

                    if not day_data:
                        continue

                    # ML分析（超高速処理）
                    ml_start = time.time()
                    ml_recommendations = ml_engine.batch_ultra_fast_analysis(day_data)
                    ml_time = time.time() - ml_start

                    # 戦略実行
                    signals = strategy_executor.execute_strategy(
                        day_data,
                        ml_recommendations,
                        tracker.cash_balance
                        + sum(pos.market_value for pos in tracker.positions.values()),
                    )

                    # 取引執行
                    for signal in signals:
                        if signal.signal_type.value == "BUY":
                            txn = tracker.execute_buy_transaction(
                                signal.symbol,
                                signal.quantity,
                                signal.price,
                                signal.strategy.value,
                            )
                            if txn:
                                trade_history.append(
                                    {
                                        "date": current_date.isoformat(),
                                        "type": "BUY",
                                        "symbol": signal.symbol,
                                        "quantity": signal.quantity,
                                        "price": signal.price,
                                        "strategy": signal.strategy.value,
                                    }
                                )

                        elif signal.signal_type.value == "SELL":
                            txn = tracker.execute_sell_transaction(
                                signal.symbol,
                                signal.quantity,
                                signal.price,
                                signal.strategy.value,
                            )
                            if txn:
                                trade_history.append(
                                    {
                                        "date": current_date.isoformat(),
                                        "type": "SELL",
                                        "symbol": signal.symbol,
                                        "quantity": signal.quantity,
                                        "price": signal.price,
                                        "pnl": txn.pnl,
                                        "strategy": signal.strategy.value,
                                    }
                                )

                    # 価格更新
                    current_prices = {
                        symbol: data["Close"].iloc[-1]
                        for symbol, data in day_data.items()
                        if not data.empty
                    }
                    tracker.update_market_prices(current_prices)

                    # リスク管理（損切り・利確）
                    self._apply_risk_management(
                        tracker, day_data, config, trade_history, current_date
                    )

                    # 日次記録
                    daily_perf = tracker.record_daily_performance()
                    if daily_perf:
                        portfolio_values.append(daily_perf.total_value)
                        daily_returns.append(daily_perf.daily_pnl_pct)

                        positions_history.append(
                            {
                                "date": current_date.isoformat(),
                                "portfolio_value": daily_perf.total_value,
                                "cash": daily_perf.cash_balance,
                                "positions": len(tracker.positions),
                                "unrealized_pnl": daily_perf.unrealized_pnl,
                            }
                        )

                    # 進捗表示
                    if (i + 1) % 20 == 0 or i == len(business_days) - 1:
                        current_value = (
                            portfolio_values[-1] if portfolio_values else config.initial_capital
                        )
                        total_return_pct = (
                            (current_value - config.initial_capital) / config.initial_capital * 100
                        )
                        logger.info(
                            f"Day {i+1}/{len(business_days)}: "
                            f"¥{current_value:,.0f} ({total_return_pct:+.1f}%) "
                            f"ML: {ml_time:.3f}s"
                        )

                except Exception as e:
                    logger.warning(f"Day {i+1} エラー: {e}")
                    continue

            # 結果分析
            result = self._analyze_results(
                config,
                portfolio_values,
                daily_returns,
                trade_history,
                positions_history,
                tracker,
                business_days,
            )

            # 結果保存
            self._save_backtest_results(result)

            logger.info(f"バックテスト完了 - 総収益: {result.total_return_pct:+.2f}%")
            logger.info(f"シャープレシオ: {result.sharpe_ratio:.3f} 勝率: {result.win_rate:.1%}")

            return result

        except Exception as e:
            logger.error(f"バックテストエラー: {e}")
            raise

    def _prepare_historical_data(self, config: BacktestConfig) -> Dict[str, pd.DataFrame]:
        """履歴データ準備"""
        try:
            logger.info("履歴データ取得中...")

            # 期間計算（余裕をもって長めに取得）
            start_date = pd.to_datetime(config.start_date) - timedelta(days=90)
            end_date = pd.to_datetime(config.end_date) + timedelta(days=1)

            # データ取得
            data = self.data_fetcher.fetch_multiple_symbols(
                config.symbols,
                period="1y",  # 1年分取得
                use_parallel=True,
            )

            # 日付フィルタリング
            filtered_data = {}
            for symbol, df in data.items():
                if df.empty:
                    continue

                # 期間フィルタ
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                if len(df) >= 30:  # 最低30日のデータが必要
                    filtered_data[symbol] = df

            logger.info(f"履歴データ準備完了: {len(filtered_data)}銘柄")
            return filtered_data

        except Exception as e:
            logger.error(f"履歴データ準備エラー: {e}")
            return {}

    def _get_daily_data(
        self,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        day_index: int,
    ) -> Dict[str, pd.DataFrame]:
        """日次データ取得（過去データシミュレーション）"""
        try:
            daily_data = {}

            for symbol, df in historical_data.items():
                # 現在日以前のデータのみ使用（未来データ排除）
                available_data = df[df.index <= current_date]

                if len(available_data) >= 30:  # ML分析に必要な最小データ
                    # 最新30日分（ML学習用）
                    daily_data[symbol] = available_data.tail(60).copy()

            return daily_data

        except Exception as e:
            logger.debug(f"日次データ取得エラー {current_date}: {e}")
            return {}

    def _create_strategy_parameters(self, config: BacktestConfig) -> StrategyParameters:
        """戦略パラメータ作成"""
        return StrategyParameters(
            strategy_type=config.strategy_type,
            risk_tolerance=config.risk_tolerance,
            max_position_size=config.max_position_size,
            stop_loss_pct=config.stop_loss_pct,
            take_profit_pct=config.take_profit_pct,
            rebalance_frequency=config.rebalance_frequency,
            min_confidence_threshold=0.7,
        )

    def _apply_risk_management(
        self,
        tracker: PortfolioTracker,
        day_data: Dict[str, pd.DataFrame],
        config: BacktestConfig,
        trade_history: List[Dict],
        current_date: datetime,
    ):
        """リスク管理適用"""
        try:
            for symbol, position in list(tracker.positions.items()):
                if symbol not in day_data:
                    continue

                current_price = day_data[symbol]["Close"].iloc[-1]

                # 損切り判定
                loss_pct = (
                    position.avg_purchase_price - current_price
                ) / position.avg_purchase_price
                if loss_pct >= config.stop_loss_pct:
                    txn = tracker.execute_sell_transaction(
                        symbol, position.quantity, current_price, "STOP_LOSS"
                    )
                    if txn:
                        trade_history.append(
                            {
                                "date": current_date.isoformat(),
                                "type": "SELL",
                                "symbol": symbol,
                                "quantity": position.quantity,
                                "price": current_price,
                                "pnl": txn.pnl,
                                "strategy": "STOP_LOSS",
                            }
                        )

                # 利確判定
                profit_pct = (
                    current_price - position.avg_purchase_price
                ) / position.avg_purchase_price
                if profit_pct >= config.take_profit_pct:
                    txn = tracker.execute_sell_transaction(
                        symbol, position.quantity, current_price, "TAKE_PROFIT"
                    )
                    if txn:
                        trade_history.append(
                            {
                                "date": current_date.isoformat(),
                                "type": "SELL",
                                "symbol": symbol,
                                "quantity": position.quantity,
                                "price": current_price,
                                "pnl": txn.pnl,
                                "strategy": "TAKE_PROFIT",
                            }
                        )

        except Exception as e:
            logger.debug(f"リスク管理エラー: {e}")

    def _analyze_results(
        self,
        config: BacktestConfig,
        portfolio_values: List[float],
        daily_returns: List[float],
        trade_history: List[Dict],
        positions_history: List[Dict],
        tracker: PortfolioTracker,
        business_days: List[datetime],
    ) -> BacktestResult:
        """結果分析"""
        try:
            if not portfolio_values:
                raise ValueError("ポートフォリオ値データなし")

            # 基本収益指標
            final_value = portfolio_values[-1]
            total_return = final_value - config.initial_capital
            total_return_pct = total_return / config.initial_capital * 100

            # 年率換算
            days = len(business_days)
            years = days / 252.0  # 営業日ベース
            annual_return_pct = (
                ((final_value / config.initial_capital) ** (1 / years) - 1) * 100
                if years > 0
                else 0
            )

            # リスク指標
            valid_returns = [r for r in daily_returns if r is not None and not np.isnan(r)]
            if valid_returns:
                volatility = np.std(valid_returns) * np.sqrt(252)  # 年率ボラティリティ
                avg_return = np.mean(valid_returns)
                sharpe_ratio = (
                    (avg_return / np.std(valid_returns)) * np.sqrt(252)
                    if np.std(valid_returns) > 0
                    else 0
                )
            else:
                volatility = 0
                sharpe_ratio = 0

            # 最大ドローダウン
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max_drawdown * 100

            # 取引統計
            sell_trades = [t for t in trade_history if t["type"] == "SELL"]
            profitable_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
            win_rate = len(profitable_trades) / len(sell_trades) if sell_trades else 0

            # プロフィットファクター
            total_profit = sum(t.get("pnl", 0) for t in profitable_trades)
            total_loss = abs(sum(t.get("pnl", 0) for t in sell_trades if t.get("pnl", 0) < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0

            # 平均取引期間（概算）
            avg_trade_duration = days / len(sell_trades) if sell_trades else 0

            # ベンチマーク比較（簡易版）
            benchmark_return_pct = 5.0  # 仮想ベンチマーク（年率5%）
            alpha = annual_return_pct - benchmark_return_pct

            result = BacktestResult(
                config=config,
                start_date=business_days[0],
                end_date=business_days[-1],
                total_return=total_return,
                total_return_pct=total_return_pct,
                annual_return_pct=annual_return_pct,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=len(trade_history),
                avg_trade_duration=avg_trade_duration,
                benchmark_return_pct=benchmark_return_pct,
                alpha=alpha,
                daily_returns=valid_returns,
                portfolio_values=portfolio_values,
                trade_history=trade_history,
                positions_history=positions_history,
            )

            return result

        except Exception as e:
            logger.error(f"結果分析エラー: {e}")
            raise

    def _save_backtest_results(self, result: BacktestResult):
        """結果保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{result.config.strategy_type.value}_{timestamp}"

            # JSON結果
            json_data = {
                "config": {
                    "start_date": result.config.start_date,
                    "end_date": result.config.end_date,
                    "initial_capital": result.config.initial_capital,
                    "symbols": result.config.symbols,
                    "strategy_type": result.config.strategy_type.value,
                    "commission_rate": result.config.commission_rate,
                },
                "results": {
                    "total_return": result.total_return,
                    "total_return_pct": result.total_return_pct,
                    "annual_return_pct": result.annual_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "volatility": result.volatility,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                    "alpha": result.alpha,
                },
                "trade_history": result.trade_history,
                "positions_history": result.positions_history,
            }

            with open(self.output_dir / f"{filename}.json", "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            # パフォーマンスチャート生成
            self._create_performance_chart(result, filename)

            logger.info(f"バックテスト結果保存: {filename}")

        except Exception as e:
            logger.error(f"結果保存エラー: {e}")

    def _create_performance_chart(self, result: BacktestResult, filename: str):
        """パフォーマンスチャート作成"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                f"Backtest Results - {result.config.strategy_type.value.upper()}",
                fontsize=16,
            )

            # 1. ポートフォリオ値推移
            dates = pd.date_range(
                result.start_date, result.end_date, periods=len(result.portfolio_values)
            )
            ax1.plot(
                dates,
                result.portfolio_values,
                "b-",
                linewidth=2,
                label="Portfolio Value",
            )
            ax1.axhline(
                y=result.config.initial_capital,
                color="r",
                linestyle="--",
                alpha=0.7,
                label="Initial Capital",
            )
            ax1.set_title("Portfolio Value Over Time")
            ax1.set_ylabel("Value (¥)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 日次リターン分布
            if result.daily_returns:
                ax2.hist(result.daily_returns, bins=50, alpha=0.7, color="green")
                ax2.axvline(
                    x=np.mean(result.daily_returns),
                    color="r",
                    linestyle="--",
                    label=f"Mean: {np.mean(result.daily_returns):.2f}%",
                )
                ax2.set_title("Daily Returns Distribution")
                ax2.set_xlabel("Daily Return (%)")
                ax2.set_ylabel("Frequency")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. 月次リターン
            if len(result.portfolio_values) >= 20:
                monthly_values = result.portfolio_values[::20]  # 20日毎（概算月次）
                monthly_returns = [
                    ((monthly_values[i] / monthly_values[i - 1]) - 1) * 100
                    for i in range(1, len(monthly_values))
                ]
                ax3.bar(
                    range(len(monthly_returns)),
                    monthly_returns,
                    color=["g" if x > 0 else "r" for x in monthly_returns],
                )
                ax3.set_title("Monthly Returns")
                ax3.set_xlabel("Month")
                ax3.set_ylabel("Return (%)")
                ax3.grid(True, alpha=0.3)

            # 4. 取引統計
            stats_text = f"""
Total Return: {result.total_return_pct:.2f}%
Annual Return: {result.annual_return_pct:.2f}%
Max Drawdown: {result.max_drawdown_pct:.2f}%
Sharpe Ratio: {result.sharpe_ratio:.3f}
Win Rate: {result.win_rate:.1%}
Total Trades: {result.total_trades}
Profit Factor: {result.profit_factor:.2f}
            """.strip()

            ax4.text(
                0.1,
                0.9,
                stats_text,
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
            ax4.set_title("Performance Statistics")
            ax4.axis("off")

            plt.tight_layout()
            plt.savefig(self.output_dir / f"{filename}_chart.png", dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"パフォーマンスチャート生成: {filename}_chart.png")

        except Exception as e:
            logger.warning(f"チャート生成エラー: {e}")


def run_sample_backtest():
    """サンプルバックテスト実行"""
    engine = BacktestEngine()

    # バックテスト設定
    config = BacktestConfig(
        start_date="2023-06-01",
        end_date="2024-01-31",
        initial_capital=1000000,
        symbols=["7203", "8306", "9984", "6758", "4689", "4563", "4592"],
        strategy_type=StrategyType.HYBRID,
        risk_tolerance=0.7,
        max_position_size=0.12,
        stop_loss_pct=0.06,
        take_profit_pct=0.18,
    )

    # バックテスト実行
    result = engine.run_backtest(config)

    # 結果表示
    print("=== バックテスト結果 ===")
    print(f"期間: {result.start_date.date()} - {result.end_date.date()}")
    print(f"戦略: {result.config.strategy_type.value}")
    print(f"初期資金: ¥{result.config.initial_capital:,.0f}")
    print(f"最終資産: ¥{result.config.initial_capital + result.total_return:,.0f}")
    print(f"総収益: ¥{result.total_return:,.0f} ({result.total_return_pct:+.2f}%)")
    print(f"年率収益: {result.annual_return_pct:+.2f}%")
    print(f"最大ドローダウン: {result.max_drawdown_pct:.2f}%")
    print(f"シャープレシオ: {result.sharpe_ratio:.3f}")
    print(f"ボラティリティ: {result.volatility:.1f}%")
    print(f"勝率: {result.win_rate:.1%}")
    print(f"プロフィットファクター: {result.profit_factor:.2f}")
    print(f"取引回数: {result.total_trades}")
    print(f"ベンチマーク超過: {result.alpha:+.2f}%")


if __name__ == "__main__":
    # サンプル実行
    run_sample_backtest()
