#!/usr/bin/env python3
"""
戦略評価機能

Issue #323: 実データバックテスト機能開発
複数の戦略を比較評価する機能
"""

import concurrent.futures
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .backtest_engine import BacktestEngine, BacktestResults
from .risk_metrics import RiskMetrics, RiskMetricsCalculator

try:
    from ..utils.structured_logging import get_structured_logger

    logger = get_structured_logger()
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class StrategyComparison:
    """戦略比較結果"""

    strategy_name: str
    backtest_results: BacktestResults
    risk_metrics: RiskMetrics
    rank: int
    score: float


@dataclass
class StrategyEvaluationReport:
    """戦略評価レポート"""

    evaluation_date: str
    test_period: str
    strategies: List[StrategyComparison]
    best_strategy: str
    benchmark_comparison: Optional[Dict[str, Any]] = None


class StrategyEvaluator:
    """戦略評価機能"""

    def __init__(self, initial_capital: float = 1000000):
        """
        初期化

        Args:
            initial_capital: 初期資本金
        """
        self.initial_capital = initial_capital
        self.risk_calculator = RiskMetricsCalculator()

        # 評価ウェイト設定
        self.evaluation_weights = {
            "return": 0.25,  # リターン
            "risk": 0.25,  # リスク
            "stability": 0.25,  # 安定性
            "efficiency": 0.25,  # 効率性
        }

        print(f"戦略評価システム初期化: 初期資本 {initial_capital:,.0f}円")

    def evaluate_strategies(
        self,
        strategies: Dict[str, Callable],
        historical_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[List[float]] = None,
        parallel_execution: bool = True,
    ) -> StrategyEvaluationReport:
        """
        複数戦略の評価実行

        Args:
            strategies: 戦略名と関数のディクショナリ
            historical_data: 過去データ
            benchmark_returns: ベンチマークリターン
            parallel_execution: 並列実行フラグ

        Returns:
            戦略評価レポート
        """
        print(f"戦略評価開始: {len(strategies)}戦略")

        # 期間情報
        dates = self._get_common_dates(historical_data)
        start_date = dates[0].strftime("%Y-%m-%d")
        end_date = dates[-1].strftime("%Y-%m-%d")
        test_period = f"{start_date} - {end_date}"

        # 戦略評価実行
        strategy_results = []

        if parallel_execution and len(strategies) > 1:
            strategy_results = self._evaluate_strategies_parallel(
                strategies, historical_data, benchmark_returns
            )
        else:
            strategy_results = self._evaluate_strategies_sequential(
                strategies, historical_data, benchmark_returns
            )

        # 戦略ランキング
        ranked_strategies = self._rank_strategies(strategy_results)

        # 最高戦略特定
        best_strategy = (
            ranked_strategies[0].strategy_name if ranked_strategies else None
        )

        # ベンチマーク比較
        benchmark_comparison = None
        if benchmark_returns:
            benchmark_comparison = self._compare_with_benchmark(
                ranked_strategies, benchmark_returns
            )

        print(f"戦略評価完了: 最高戦略 {best_strategy}")

        return StrategyEvaluationReport(
            evaluation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            test_period=test_period,
            strategies=ranked_strategies,
            best_strategy=best_strategy,
            benchmark_comparison=benchmark_comparison,
        )

    def _evaluate_strategies_parallel(
        self,
        strategies: Dict[str, Callable],
        historical_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[List[float]],
    ) -> List[StrategyComparison]:
        """並列戦略評価"""
        strategy_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # 並列実行用のタスク作成
            future_to_strategy = {
                executor.submit(
                    self._evaluate_single_strategy,
                    name,
                    strategy,
                    historical_data,
                    benchmark_returns,
                ): name
                for name, strategy in strategies.items()
            }

            # 結果収集
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    result = future.result()
                    if result:
                        strategy_results.append(result)
                        print(f"戦略評価完了: {strategy_name}")
                except Exception as e:
                    print(f"戦略評価エラー: {strategy_name} - {e}")

        return strategy_results

    def _evaluate_strategies_sequential(
        self,
        strategies: Dict[str, Callable],
        historical_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[List[float]],
    ) -> List[StrategyComparison]:
        """順次戦略評価"""
        strategy_results = []

        for name, strategy in strategies.items():
            try:
                result = self._evaluate_single_strategy(
                    name, strategy, historical_data, benchmark_returns
                )
                if result:
                    strategy_results.append(result)
                    print(f"戦略評価完了: {name}")
            except Exception as e:
                print(f"戦略評価エラー: {name} - {e}")

        return strategy_results

    def _evaluate_single_strategy(
        self,
        strategy_name: str,
        strategy_function: Callable,
        historical_data: Dict[str, pd.DataFrame],
        benchmark_returns: Optional[List[float]],
    ) -> Optional[StrategyComparison]:
        """単一戦略評価"""
        try:
            # バックテスト実行
            engine = BacktestEngine(self.initial_capital)
            backtest_results = engine.execute_backtest(
                historical_data, strategy_function
            )

            # リスクメトリクス計算
            risk_metrics = self.risk_calculator.calculate_metrics(
                backtest_results.daily_returns,
                backtest_results.portfolio_values,
                benchmark_returns,
            )

            # 戦略スコア計算
            score = self._calculate_strategy_score(backtest_results, risk_metrics)

            return StrategyComparison(
                strategy_name=strategy_name,
                backtest_results=backtest_results,
                risk_metrics=risk_metrics,
                rank=0,  # ランキングは後で設定
                score=score,
            )

        except Exception as e:
            print(f"戦略 {strategy_name} の評価中にエラー: {e}")
            return None

    def _calculate_strategy_score(
        self, backtest_results: BacktestResults, risk_metrics: RiskMetrics
    ) -> float:
        """戦略スコア計算"""
        # リターン評価 (0-100点)
        return_score = min(
            100, max(0, (risk_metrics.annualized_return + 0.1) / 0.3 * 100)
        )

        # リスク評価 (0-100点, 低リスクほど高得点)
        risk_score = min(100, max(0, (0.3 - risk_metrics.volatility) / 0.3 * 100))

        # 安定性評価 (0-100点, 低ドローダウンほど高得点)
        stability_score = min(
            100, max(0, (0.3 - abs(risk_metrics.maximum_drawdown)) / 0.3 * 100)
        )

        # 効率性評価 (シャープレシオベース)
        efficiency_score = min(100, max(0, (risk_metrics.sharpe_ratio + 1) / 3 * 100))

        # 重み付け総合スコア
        total_score = (
            return_score * self.evaluation_weights["return"]
            + risk_score * self.evaluation_weights["risk"]
            + stability_score * self.evaluation_weights["stability"]
            + efficiency_score * self.evaluation_weights["efficiency"]
        )

        return total_score

    def _rank_strategies(
        self, strategy_results: List[StrategyComparison]
    ) -> List[StrategyComparison]:
        """戦略ランキング"""
        # スコア順でソート
        sorted_strategies = sorted(
            strategy_results, key=lambda x: x.score, reverse=True
        )

        # ランク付け
        for i, strategy in enumerate(sorted_strategies):
            strategy.rank = i + 1

        return sorted_strategies

    def _compare_with_benchmark(
        self, strategies: List[StrategyComparison], benchmark_returns: List[float]
    ) -> Dict[str, Any]:
        """ベンチマーク比較"""
        # ベンチマークメトリクス計算
        benchmark_values = [self.initial_capital]
        for ret in benchmark_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))

        benchmark_metrics = self.risk_calculator.calculate_metrics(
            benchmark_returns, benchmark_values
        )

        # 各戦略とベンチマークの比較
        comparisons = {}
        for strategy in strategies:
            comparisons[strategy.strategy_name] = {
                "excess_return": strategy.risk_metrics.annualized_return
                - benchmark_metrics.annualized_return,
                "information_ratio": strategy.risk_metrics.information_ratio,
                "outperformed": strategy.risk_metrics.annualized_return
                > benchmark_metrics.annualized_return,
            }

        return {
            "benchmark_metrics": {
                "annualized_return": benchmark_metrics.annualized_return,
                "volatility": benchmark_metrics.volatility,
                "sharpe_ratio": benchmark_metrics.sharpe_ratio,
                "max_drawdown": benchmark_metrics.maximum_drawdown,
            },
            "strategy_comparisons": comparisons,
        }

    def _get_common_dates(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """共通日付取得"""
        common_dates = None
        for data in historical_data.values():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)

        return common_dates.sort_values()

    def generate_evaluation_report(self, evaluation: StrategyEvaluationReport) -> str:
        """評価レポート生成"""
        report = []
        report.append("=" * 80)
        report.append("戦略評価レポート")
        report.append("=" * 80)

        report.append(f"評価日時: {evaluation.evaluation_date}")
        report.append(f"テスト期間: {evaluation.test_period}")
        report.append(f"評価戦略数: {len(evaluation.strategies)}")

        if evaluation.best_strategy:
            report.append(f"最優秀戦略: {evaluation.best_strategy}")

        # 戦略ランキング
        report.append("\n【戦略ランキング】")
        report.append(
            f"{'順位':<4} {'戦略名':<20} {'スコア':<8} {'年率リターン':<12} {'ボラティリティ':<12} {'シャープ':<8}"
        )
        report.append("-" * 80)

        for strategy in evaluation.strategies:
            report.append(
                f"{strategy.rank:<4} "
                f"{strategy.strategy_name:<20} "
                f"{strategy.score:<8.1f} "
                f"{strategy.risk_metrics.annualized_return:<12.2%} "
                f"{strategy.risk_metrics.volatility:<12.2%} "
                f"{strategy.risk_metrics.sharpe_ratio:<8.3f}"
            )

        # 詳細分析（上位3戦略）
        report.append("\n【上位戦略詳細分析】")
        top_strategies = evaluation.strategies[: min(3, len(evaluation.strategies))]

        for strategy in top_strategies:
            report.append(f"\n■ {strategy.strategy_name} (第{strategy.rank}位)")
            report.append(f"  総合スコア: {strategy.score:.1f}/100")
            report.append(f"  総リターン: {strategy.backtest_results.total_return:.2%}")
            report.append(
                f"  年率リターン: {strategy.risk_metrics.annualized_return:.2%}"
            )
            report.append(f"  ボラティリティ: {strategy.risk_metrics.volatility:.2%}")
            report.append(f"  シャープレシオ: {strategy.risk_metrics.sharpe_ratio:.3f}")
            report.append(
                f"  最大ドローダウン: {strategy.risk_metrics.maximum_drawdown:.2%}"
            )
            report.append(f"  勝率: {strategy.backtest_results.win_rate:.2%}")
            report.append(f"  総取引数: {strategy.backtest_results.total_trades}回")

        # ベンチマーク比較
        if evaluation.benchmark_comparison:
            report.append("\n【ベンチマーク比較】")
            benchmark = evaluation.benchmark_comparison["benchmark_metrics"]
            report.append(
                f"ベンチマーク年率リターン: {benchmark['annualized_return']:.2%}"
            )
            report.append(f"ベンチマークボラティリティ: {benchmark['volatility']:.2%}")

            report.append("\n各戦略のベンチマーク超過成果:")
            for strategy_name, comparison in evaluation.benchmark_comparison[
                "strategy_comparisons"
            ].items():
                status = "✓" if comparison["outperformed"] else "✗"
                report.append(
                    f"  {status} {strategy_name}: +{comparison['excess_return']:.2%}"
                )

        return "\n".join(report)

    def save_evaluation_results(
        self, evaluation: StrategyEvaluationReport, file_path: str
    ):
        """評価結果保存"""
        try:
            # JSON形式で詳細結果保存
            data = {
                "evaluation_date": evaluation.evaluation_date,
                "test_period": evaluation.test_period,
                "best_strategy": evaluation.best_strategy,
                "strategies": [],
            }

            for strategy in evaluation.strategies:
                strategy_data = {
                    "name": strategy.strategy_name,
                    "rank": strategy.rank,
                    "score": strategy.score,
                    "backtest_results": {
                        "total_return": strategy.backtest_results.total_return,
                        "annualized_return": strategy.backtest_results.annualized_return,
                        "volatility": strategy.backtest_results.volatility,
                        "sharpe_ratio": strategy.backtest_results.sharpe_ratio,
                        "max_drawdown": strategy.backtest_results.max_drawdown,
                        "total_trades": strategy.backtest_results.total_trades,
                        "win_rate": strategy.backtest_results.win_rate,
                    },
                    "risk_metrics": {
                        "annualized_return": strategy.risk_metrics.annualized_return,
                        "volatility": strategy.risk_metrics.volatility,
                        "sharpe_ratio": strategy.risk_metrics.sharpe_ratio,
                        "sortino_ratio": strategy.risk_metrics.sortino_ratio,
                        "maximum_drawdown": strategy.risk_metrics.maximum_drawdown,
                        "var_95": strategy.risk_metrics.var_95,
                        "win_rate": strategy.risk_metrics.win_rate,
                    },
                }
                data["strategies"].append(strategy_data)

            if evaluation.benchmark_comparison:
                data["benchmark_comparison"] = evaluation.benchmark_comparison

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"評価結果保存完了: {file_path}")

        except Exception as e:
            print(f"評価結果保存エラー: {e}")


# サンプル戦略定義
def create_sample_strategies() -> Dict[str, Callable]:
    """サンプル戦略作成"""

    def momentum_strategy(
        lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """モメンタム戦略"""
        signals = {}
        for symbol, data in lookback_data.items():
            if len(data) >= 20:
                returns_20d = data["Close"].iloc[-1] / data["Close"].iloc[-20] - 1
                if returns_20d > 0.05:
                    signals[symbol] = 0.25
                elif returns_20d < -0.05:
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.1

        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: v / total_weight for k, v in signals.items()}

        return signals

    def mean_reversion_strategy(
        lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """平均回帰戦略"""
        signals = {}
        for symbol, data in lookback_data.items():
            if len(data) >= 20:
                current_price = data["Close"].iloc[-1]
                sma_20 = data["Close"].iloc[-20:].mean()
                deviation = (current_price - sma_20) / sma_20

                if deviation < -0.1:  # 10%以上下落
                    signals[symbol] = 0.3
                elif deviation > 0.1:  # 10%以上上昇
                    signals[symbol] = 0.0
                else:
                    signals[symbol] = 0.15

        total_weight = sum(signals.values())
        if total_weight > 0:
            signals = {k: v / total_weight for k, v in signals.items()}

        return signals

    def equal_weight_strategy(
        lookback_data: Dict[str, pd.DataFrame], current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """等重み戦略"""
        symbols = list(lookback_data.keys())
        weight_per_symbol = 1.0 / len(symbols) if symbols else 0
        return {symbol: weight_per_symbol for symbol in symbols}

    return {
        "モメンタム戦略": momentum_strategy,
        "平均回帰戦略": mean_reversion_strategy,
        "等重み戦略": equal_weight_strategy,
    }


if __name__ == "__main__":
    # テスト実行
    print("戦略評価機能テスト")
    print("=" * 50)

    try:
        # サンプル戦略取得
        strategies = create_sample_strategies()

        # バックテストエンジンでサンプルデータ作成
        engine = BacktestEngine()
        test_symbols = ["7203.T", "8306.T", "9984.T"]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        historical_data = engine.load_historical_data(
            test_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        )

        if historical_data:
            # 戦略評価実行
            evaluator = StrategyEvaluator()
            evaluation = evaluator.evaluate_strategies(strategies, historical_data)

            # レポート生成
            report = evaluator.generate_evaluation_report(evaluation)
            print(report)

            # 結果保存
            evaluator.save_evaluation_results(
                evaluation, "strategy_evaluation_test.json"
            )

        else:
            print("テストデータが取得できませんでした")

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback

        traceback.print_exc()
