#!/usr/bin/env python3
"""
実データポートフォリオ最適化テスト

Issue #321: 実データでの最終動作確認テスト
実際の市場データを使用したポートフォリオ最適化の精度・実用性検証
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from day_trade.optimization.portfolio_optimizer import PortfolioOptimizer
    from day_trade.optimization.risk_manager import RiskManager
except ImportError as e:
    print(f"Module import error: {e}")
    print("Will use simplified implementation for testing")

print("REAL DATA PORTFOLIO OPTIMIZATION TEST")
print("Issue #321: Real data final operation verification")
print("=" * 60)


class RealDataPortfolioTester:
    """実データポートフォリオ最適化テスター"""

    def __init__(self):
        # 優良銘柄20銘柄（ポートフォリオ最適化用）
        self.portfolio_symbols = [
            "7203.T",  # トヨタ自動車
            "8306.T",  # 三菱UFJフィナンシャル・グループ
            "9984.T",  # ソフトバンクグループ
            "6758.T",  # ソニーグループ
            "9432.T",  # 日本電信電話
            "8001.T",  # 伊藤忠商事
            "6861.T",  # キーエンス
            "8058.T",  # 三菱商事
            "4502.T",  # 武田薬品工業
            "7974.T",  # 任天堂
            "8411.T",  # みずほフィナンシャルグループ
            "8316.T",  # 三井住友フィナンシャルグループ
            "8031.T",  # 三井物産
            "8053.T",  # 住友商事
            "7751.T",  # キヤノン
            "6981.T",  # 村田製作所
            "9983.T",  # ファーストリテイリング
            "4568.T",  # 第一三共
            "6367.T",  # ダイキン工業
            "6954.T",  # ファナック
        ]

        self.optimization_target_time = 2.0  # 2秒目標
        self.portfolio_results = {}

        print(f"Portfolio optimizer initialized: {len(self.portfolio_symbols)} symbols")

    def fetch_portfolio_data(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """ポートフォリオデータ取得"""
        print(f"\n=== Portfolio Data Fetch ({days} days) ===")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(
            f"Data period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        portfolio_data = {}
        successful_fetches = 0

        for symbol in self.portfolio_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)

                if not hist.empty and len(hist) >= 30:  # 最低30日分
                    portfolio_data[symbol] = hist
                    successful_fetches += 1
                else:
                    print(
                        f"  [SKIP] {symbol}: Insufficient data ({len(hist) if not hist.empty else 0} days)"
                    )

                time.sleep(0.1)  # API制限対策

            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")

        print("Portfolio data fetch results:")
        print(
            f"  Successful: {successful_fetches}/{len(self.portfolio_symbols)} symbols"
        )
        print(
            f"  Data coverage: {successful_fetches/len(self.portfolio_symbols)*100:.1f}%"
        )

        return portfolio_data

    def calculate_returns_and_covariance(
        self, portfolio_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """リターンと共分散行列計算"""
        print("\n=== Returns and Covariance Calculation ===")

        # 価格データ統合
        price_data = {}
        for symbol, hist in portfolio_data.items():
            price_data[symbol] = hist["Close"]

        prices = pd.DataFrame(price_data).fillna(method="ffill").dropna()

        # 日次リターン計算
        returns = prices.pct_change().dropna()

        # 期待リターン（年率）
        expected_returns = returns.mean() * 252

        # 共分散行列（年率）
        cov_matrix = returns.cov() * 252

        print("Returns calculation completed:")
        print(f"  Data points: {len(returns)} days")
        print(f"  Symbols: {len(expected_returns)} stocks")
        print(
            f"  Date range: {returns.index.min().strftime('%Y-%m-%d')} to {returns.index.max().strftime('%Y-%m-%d')}"
        )

        # 基本統計表示
        print(
            f"  Average return: {expected_returns.mean():.3f} ({expected_returns.mean()*100:.1f}% annually)"
        )
        print(
            f"  Return range: {expected_returns.min():.3f} to {expected_returns.max():.3f}"
        )
        print(f"  Average volatility: {np.sqrt(np.diag(cov_matrix)).mean():.3f}")

        return expected_returns, cov_matrix

    def optimize_portfolio(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> Dict:
        """ポートフォリオ最適化実行"""
        print(
            f"\n=== Portfolio Optimization (Target: {self.optimization_target_time}s) ==="
        )

        start_time = time.time()

        # 最適化実行（複数手法）
        optimization_results = {}

        # 1. 等重みポートフォリオ（ベースライン）
        equal_weights = np.ones(len(expected_returns)) / len(expected_returns)
        equal_return = np.dot(equal_weights, expected_returns)
        equal_risk = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
        equal_sharpe = equal_return / equal_risk if equal_risk > 0 else 0

        optimization_results["equal_weight"] = {
            "weights": equal_weights,
            "expected_return": equal_return,
            "volatility": equal_risk,
            "sharpe_ratio": equal_sharpe,
        }

        # 2. 最小分散ポートフォリオ
        min_var_weights = self._minimize_variance(cov_matrix)
        min_var_return = np.dot(min_var_weights, expected_returns)
        min_var_risk = np.sqrt(
            np.dot(min_var_weights, np.dot(cov_matrix, min_var_weights))
        )
        min_var_sharpe = min_var_return / min_var_risk if min_var_risk > 0 else 0

        optimization_results["min_variance"] = {
            "weights": min_var_weights,
            "expected_return": min_var_return,
            "volatility": min_var_risk,
            "sharpe_ratio": min_var_sharpe,
        }

        # 3. 最大シャープレシオポートフォリオ
        max_sharpe_weights = self._maximize_sharpe_ratio(expected_returns, cov_matrix)
        max_sharpe_return = np.dot(max_sharpe_weights, expected_returns)
        max_sharpe_risk = np.sqrt(
            np.dot(max_sharpe_weights, np.dot(cov_matrix, max_sharpe_weights))
        )
        max_sharpe_sharpe = (
            max_sharpe_return / max_sharpe_risk if max_sharpe_risk > 0 else 0
        )

        optimization_results["max_sharpe"] = {
            "weights": max_sharpe_weights,
            "expected_return": max_sharpe_return,
            "volatility": max_sharpe_risk,
            "sharpe_ratio": max_sharpe_sharpe,
        }

        optimization_time = time.time() - start_time

        # 結果表示
        print("Portfolio optimization completed:")
        print(
            f"  Optimization time: {optimization_time:.3f}s (target: {self.optimization_target_time:.1f}s)"
        )
        print(
            f"  Performance ratio: {optimization_time / self.optimization_target_time:.2f}x"
        )

        print("\nOptimization Results:")
        for strategy, result in optimization_results.items():
            print(f"  {strategy.upper()}:")
            print(
                f"    Expected Return: {result['expected_return']:.3f} ({result['expected_return']*100:.1f}% annually)"
            )
            print(
                f"    Volatility: {result['volatility']:.3f} ({result['volatility']*100:.1f}% annually)"
            )
            print(f"    Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            print(
                f"    Max Weight: {result['weights'].max():.3f} ({result['weights'].max()*100:.1f}%)"
            )
            print(
                f"    Min Weight: {result['weights'].min():.3f} ({result['weights'].min()*100:.1f}%)"
            )

        # パフォーマンス評価
        time_success = optimization_time <= self.optimization_target_time

        # 最適化品質評価
        best_sharpe = max(
            result["sharpe_ratio"] for result in optimization_results.values()
        )
        quality_success = best_sharpe > 0.5  # 合理的なシャープレシオ

        # 分散化評価（改良版）
        best_strategy = max(
            optimization_results.keys(),
            key=lambda k: optimization_results[k]["sharpe_ratio"],
        )
        best_weights = optimization_results[best_strategy]["weights"]
        diversification = 1 / np.sum(best_weights**2)  # 有効銘柄数
        max_weight = best_weights.max()
        # より厳しい分散化基準
        diversification_success = diversification >= 6 and max_weight <= 0.20

        overall_success = time_success and quality_success and diversification_success

        print("\nOptimization Evaluation:")
        print(
            f"  [{'OK' if time_success else 'NG'}] Time Performance: {'ACHIEVED' if time_success else 'MISSED'}"
        )
        print(
            f"  [{'OK' if quality_success else 'NG'}] Optimization Quality: {'GOOD' if quality_success else 'POOR'}"
        )
        print(
            f"  [{'OK' if diversification_success else 'NG'}] Diversification: {'ADEQUATE' if diversification_success else 'INSUFFICIENT'}"
        )
        print(
            f"  [{'OK' if overall_success else 'NG'}] Overall: {'PASSED' if overall_success else 'FAILED'}"
        )

        print(f"\nBest Strategy: {best_strategy.upper()}")
        print(
            f"  Sharpe Ratio: {optimization_results[best_strategy]['sharpe_ratio']:.3f}"
        )
        print(f"  Effective Stocks: {diversification:.1f}")
        print(f"  Max Position: {max_weight:.1%}")
        print("  Diversification Target: >=6 effective stocks, <=20% max position")

        self.portfolio_results = {
            "optimization_results": optimization_results,
            "optimization_time": optimization_time,
            "best_strategy": best_strategy,
            "overall_success": overall_success,
        }

        return self.portfolio_results

    def _minimize_variance(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """最小分散ポートフォリオ計算"""
        try:
            from scipy.optimize import minimize

            n = len(cov_matrix)

            # 目的関数（分散最小化）
            def objective(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))

            # 制約条件
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
            bounds = tuple((0, 0.25) for _ in range(n))  # 各銘柄25%上限（分散化強化）

            # 初期値
            x0 = np.ones(n) / n

            # 最適化実行
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )

            if result.success:
                return result.x
            else:
                return np.ones(n) / n  # 失敗時は等重み

        except ImportError:
            # scipy がない場合は単純な計算
            inv_cov = np.linalg.pinv(cov_matrix)
            ones = np.ones(len(cov_matrix))
            weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
            return np.clip(weights, 0, 0.25) / np.sum(np.clip(weights, 0, 0.25))

    def _maximize_sharpe_ratio(
        self, expected_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """最大シャープレシオポートフォリオ計算（分散化制約強化版）"""
        try:
            from scipy.optimize import minimize

            n = len(expected_returns)

            # 目的関数（負のシャープレシオ最小化 + 分散化ペナルティ）
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                sharpe_ratio = (
                    portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                )

                # 分散化ペナルティ：集中度が高いほどペナルティ
                concentration_penalty = np.sum(weights**2) * 0.1  # 軽いペナルティ

                return -sharpe_ratio + concentration_penalty

            # 制約条件：投資額100% + 最小分散化制約
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 投資額100%
                {
                    "type": "ineq",
                    "fun": lambda w: 1.0 / np.sum(w**2) - 6.0,
                },  # 有効銘柄数6以上
            ]
            bounds = tuple((0, 0.18) for _ in range(n))  # 各銘柄18%上限（より厳格）

            # 初期値（均等分散に近い値）
            x0 = np.ones(n) / n

            # 最適化実行
            result = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
            )

            if result.success:
                return result.x
            else:
                # 失敗時はより保守的な均等配分
                return np.ones(n) / n

        except ImportError:
            # scipy がない場合は改良された簡易計算
            # リスク調整リターンベース + 分散化強制
            risk_adj_returns = expected_returns / np.sqrt(np.diag(cov_matrix))

            # 上位銘柄に制限をかけながら配分
            weights = risk_adj_returns / risk_adj_returns.sum()

            # より強い分散化：最大18%制限
            weights = np.clip(weights, 0, 0.18)
            weights = weights / np.sum(weights)

            # さらに分散化を強制（最小重み設定）
            min_weight = 0.02  # 最低2%
            active_positions = np.sum(weights > min_weight)
            if active_positions < n * 0.6:  # 60%未満しかアクティブでない場合
                # 全銘柄にある程度の重みを付与
                weights = np.maximum(weights, min_weight)
                weights = weights / np.sum(weights)

            return weights

    def analyze_portfolio_characteristics(self) -> Dict:
        """ポートフォリオ特性分析"""
        print("\n=== Portfolio Characteristics Analysis ===")

        if not self.portfolio_results:
            print("No portfolio optimization results available")
            return {}

        optimization_results = self.portfolio_results["optimization_results"]
        best_strategy = self.portfolio_results["best_strategy"]

        # 最適ポートフォリオの詳細分析
        best_portfolio = optimization_results[best_strategy]
        weights = best_portfolio["weights"]

        # 重み分析
        sorted_weights = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)

        print(f"Best Portfolio ({best_strategy.upper()}) Analysis:")
        print(f"  Expected Annual Return: {best_portfolio['expected_return']*100:.2f}%")
        print(f"  Annual Volatility: {best_portfolio['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {best_portfolio['sharpe_ratio']:.3f}")

        print("\nTop 5 Holdings:")
        symbols = self.portfolio_symbols
        for i, (idx, weight) in enumerate(sorted_weights[:5]):
            symbol = symbols[idx]
            print(f"  {i+1}. {symbol}: {weight*100:.2f}%")

        # リスク・リターン特性
        characteristics = {
            "best_strategy": best_strategy,
            "annual_return": best_portfolio["expected_return"],
            "annual_volatility": best_portfolio["volatility"],
            "sharpe_ratio": best_portfolio["sharpe_ratio"],
            "max_weight": weights.max(),
            "concentration": np.sum(weights**2),  # ハーフィンダール指数
            "effective_stocks": 1 / np.sum(weights**2),
            "top_5_concentration": sum(weight for _, weight in sorted_weights[:5]),
        }

        print("\nPortfolio Risk Characteristics:")
        print(f"  Concentration Index: {characteristics['concentration']:.3f}")
        print(f"  Effective Stocks: {characteristics['effective_stocks']:.1f}")
        print(
            f"  Top 5 Concentration: {characteristics['top_5_concentration']*100:.1f}%"
        )
        print(f"  Maximum Position: {characteristics['max_weight']*100:.1f}%")

        # リスク・リターン効率性評価
        efficiency_score = 0
        if characteristics["sharpe_ratio"] > 1.0:
            efficiency_score += 40
        elif characteristics["sharpe_ratio"] > 0.5:
            efficiency_score += 25

        if characteristics["annual_return"] > 0.08:  # 8%以上
            efficiency_score += 30
        elif characteristics["annual_return"] > 0.05:  # 5%以上
            efficiency_score += 20

        if characteristics["effective_stocks"] > len(self.portfolio_symbols) * 0.4:
            efficiency_score += 20
        elif characteristics["effective_stocks"] > len(self.portfolio_symbols) * 0.2:
            efficiency_score += 10

        if characteristics["max_weight"] < 0.20:  # 20%未満
            efficiency_score += 10

        print(f"\nPortfolio Efficiency Score: {efficiency_score}/100")

        efficiency_passed = efficiency_score >= 60
        print(
            f"Efficiency Assessment: [{'OK' if efficiency_passed else 'NG'}] {'PASSED' if efficiency_passed else 'NEEDS_IMPROVEMENT'}"
        )

        characteristics["efficiency_score"] = efficiency_score
        characteristics["efficiency_passed"] = efficiency_passed

        return characteristics


def main():
    """メイン実行"""
    print("Starting real data portfolio optimization test...")

    try:
        tester = RealDataPortfolioTester()

        # 1. ポートフォリオデータ取得
        portfolio_data = tester.fetch_portfolio_data(days=90)

        if len(portfolio_data) < 10:
            print(f"[ERROR] Insufficient portfolio data: {len(portfolio_data)} symbols")
            return False

        # 2. リターン・共分散計算
        expected_returns, cov_matrix = tester.calculate_returns_and_covariance(
            portfolio_data
        )

        # 3. ポートフォリオ最適化実行
        optimization_results = tester.optimize_portfolio(expected_returns, cov_matrix)

        # 4. ポートフォリオ特性分析
        characteristics = tester.analyze_portfolio_characteristics()

        # 最終評価
        print(f"\n{'='*60}")
        print("FINAL EVALUATION - Real Data Portfolio Optimization")
        print(f"{'='*60}")

        evaluation_criteria = [
            ("Data Coverage", len(portfolio_data) >= 15),
            (
                "Optimization Performance",
                optimization_results["optimization_time"]
                <= tester.optimization_target_time,
            ),
            ("Portfolio Quality", optimization_results["overall_success"]),
            ("Risk-Return Efficiency", characteristics.get("efficiency_passed", False)),
            (
                "Diversification",
                characteristics.get("effective_stocks", 0) >= 6
                and characteristics.get("max_weight", 1.0) <= 0.20,
            ),
        ]

        passed_criteria = 0
        for criterion, passed in evaluation_criteria:
            status = "[OK]" if passed else "[NG]"
            print(f"  {status} {criterion}")
            if passed:
                passed_criteria += 1

        overall_success = passed_criteria >= 4  # 5項目中4項目以上で合格

        print(f"\nOverall Result: {passed_criteria}/5 criteria passed")

        if overall_success:
            print("[SUCCESS] Real Data Portfolio Optimization: PASSED")
            print(f"  - Optimized portfolio with {len(portfolio_data)} symbols")
            print(
                f"  - Optimization time: {optimization_results['optimization_time']:.3f}s (target: {tester.optimization_target_time:.1f}s)"
            )
            print(
                f"  - Best Sharpe ratio: {characteristics.get('sharpe_ratio', 0):.3f}"
            )
            print(
                f"  - Expected annual return: {characteristics.get('annual_return', 0)*100:.2f}%"
            )
            print("  - Ready for production portfolio management")

            return True
        else:
            print("[FAILED] Real Data Portfolio Optimization: SOME ISSUES")
            print("  - Review failed criteria above")
            print("  - Consider parameter tuning or data quality improvement")

            return False

    except Exception as e:
        print(f"Real data portfolio optimization test error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n{'='*60}")
        print("REAL DATA PORTFOLIO OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("Next: Real data backtesting and stability test")
        print(f"{'='*60}")

    exit(0 if success else 1)
