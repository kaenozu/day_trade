#!/usr/bin/env python3
"""
ポートフォリオ最適化エンジン

Modern Portfolio Theory、Monte Carlo シミュレーション、
相関分析を用いた最適ポートフォリオ構築システム
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

# scikit-learn依存チェック
try:
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn利用可能")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn未インストール - 基本機能のみ利用可能")

# 警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)


class PortfolioOptimizer:
    """
    ポートフォリオ最適化クラス

    Modern Portfolio Theoryに基づく最適ポートフォリオ構築
    """

    def __init__(
        self,
        risk_tolerance: float = 0.5,
        max_position_size: float = 0.3,
        min_position_size: float = 0.01,
        sector_constraints: Optional[Dict[str, float]] = None,
    ):
        """
        初期化

        Args:
            risk_tolerance: リスク許容度 (0-1, 1がリスク選好)
            max_position_size: 最大ポジションサイズ (比率)
            min_position_size: 最小ポジションサイズ (比率)
            sector_constraints: セクター別制約 {sector: max_weight}
        """
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.sector_constraints = sector_constraints or {}

        logger.info("ポートフォリオオプティマイザー初期化:")
        logger.info(f"  - リスク許容度: {risk_tolerance:.2f}")
        logger.info(f"  - 最大ポジション: {max_position_size:.1%}")
        logger.info(f"  - 最小ポジション: {min_position_size:.1%}")

    def calculate_returns_and_risks(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        リターンとリスク（相関行列）を計算

        Args:
            price_data: {symbol: price_dataframe}

        Returns:
            Tuple[expected_returns, covariance_matrix]
        """
        logger.info(f"リターン・リスク計算開始: {len(price_data)}銘柄")

        # 価格データから収益率を計算
        returns_data = {}
        for symbol, data in price_data.items():
            if data.empty or "Close" not in data.columns:
                continue

            # 日次収益率を計算
            daily_returns = data["Close"].pct_change().dropna()
            if len(daily_returns) > 5:  # 最低5日のデータが必要
                returns_data[symbol] = daily_returns

        if len(returns_data) < 2:
            logger.error("リターン計算に必要なデータが不足")
            raise ValueError("最低2銘柄のデータが必要です")

        # DataFrameに変換（インデックスのタイムゾーン情報を統一）
        returns_df = pd.DataFrame(returns_data)

        # インデックスのタイムゾーン情報を除去して統一
        if hasattr(returns_df.index, "tz") and returns_df.index.tz is not None:
            returns_df.index = returns_df.index.tz_localize(None)

        # 欠損値を前方補完
        returns_df = returns_df.fillna(method="ffill").fillna(method="bfill")

        # 期待リターンを計算（年率換算）
        expected_returns = returns_df.mean() * 252

        # 共分散行列を計算（年率換算、Ledoit-Wolf推定使用）
        if SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            covariance_matrix = pd.DataFrame(
                lw.fit(returns_df).covariance_ * 252,
                index=returns_df.columns,
                columns=returns_df.columns,
            )
        else:
            # 標準的な共分散計算
            covariance_matrix = returns_df.cov() * 252

        logger.info("期待リターン統計:")
        logger.info(f"  - 平均: {expected_returns.mean():.2%}")
        logger.info(f"  - 最大: {expected_returns.max():.2%}")
        logger.info(f"  - 最小: {expected_returns.min():.2%}")

        return expected_returns, covariance_matrix

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        investment_amount: float = 1000000,
    ) -> Dict:
        """
        ポートフォリオ最適化実行

        Args:
            expected_returns: 期待リターン
            covariance_matrix: 共分散行列
            investment_amount: 投資金額

        Returns:
            最適化結果辞書
        """
        logger.info("ポートフォリオ最適化開始")

        n_assets = len(expected_returns)
        symbols = expected_returns.index

        # 目的関数：効用関数 (リターン - リスク許容度 * リスク)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            # 効用 = リターン - リスクペナルティ
            utility = portfolio_return - (1 - self.risk_tolerance) * portfolio_risk
            return -utility  # 最大化のため負値

        # 制約条件
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]  # 重みの合計 = 1

        # 範囲制約
        bounds = [(self.min_position_size, self.max_position_size) for _ in range(n_assets)]

        # 初期値：等ウェイト
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # 最適化実行
        try:
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000},
            )

            if not result.success:
                logger.warning(f"最適化警告: {result.message}")

            optimal_weights = result.x

        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            # フォールバック：等ウェイト
            optimal_weights = initial_weights

        # 結果計算
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_risk = np.sqrt(
            np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
        )
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

        # 投資金額配分
        allocations = {}
        for i, symbol in enumerate(symbols):
            weight = optimal_weights[i]
            if weight >= self.min_position_size:
                allocations[symbol] = {
                    "weight": weight,
                    "amount": weight * investment_amount,
                    "expected_return": expected_returns[symbol],
                }

        optimization_result = {
            "allocations": allocations,
            "portfolio_metrics": {
                "expected_return": portfolio_return,
                "volatility": portfolio_risk,
                "sharpe_ratio": sharpe_ratio,
                "total_amount": investment_amount,
            },
            "optimization_success": result.success if "result" in locals() else False,
            "n_positions": len(allocations),
        }

        logger.info("最適化完了:")
        logger.info(f"  - 期待リターン: {portfolio_return:.2%}")
        logger.info(f"  - ボラティリティ: {portfolio_risk:.2%}")
        logger.info(f"  - シャープレシオ: {sharpe_ratio:.2f}")
        logger.info(f"  - ポジション数: {len(allocations)}")

        return optimization_result

    def monte_carlo_simulation(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_simulations: int = 1000,
        time_horizon: int = 252,
    ) -> Dict:
        """
        Monte Carloシミュレーション

        Args:
            expected_returns: 期待リターン
            covariance_matrix: 共分散行列
            n_simulations: シミュレーション回数
            time_horizon: 時間軸（日数）

        Returns:
            シミュレーション結果
        """
        logger.info(f"Monte Carloシミュレーション開始: {n_simulations}回")

        n_assets = len(expected_returns)
        symbols = expected_returns.index

        # ランダムウェイト生成とシミュレーション
        portfolio_results = []

        for _ in range(n_simulations):
            # ランダムウェイト生成（制約付き）
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()  # 正規化

            # 制約チェックと調整
            weights = np.clip(weights, self.min_position_size, self.max_position_size)
            weights = weights / weights.sum()  # 再正規化

            # ポートフォリオ指標計算
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

            # 時間軸での価値変動シミュレーション
            daily_returns = np.random.multivariate_normal(
                expected_returns / 252, covariance_matrix / 252, time_horizon
            )

            portfolio_daily_returns = np.dot(daily_returns, weights)
            final_value = np.prod(1 + portfolio_daily_returns)

            portfolio_results.append(
                {
                    "weights": weights,
                    "expected_return": portfolio_return,
                    "volatility": portfolio_risk,
                    "sharpe_ratio": portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
                    "final_value": final_value,
                }
            )

        # 統計分析
        returns = [r["expected_return"] for r in portfolio_results]
        volatilities = [r["volatility"] for r in portfolio_results]
        sharpe_ratios = [r["sharpe_ratio"] for r in portfolio_results]
        final_values = [r["final_value"] for r in portfolio_results]

        # 効率的フロンティアポイント
        max_sharpe_idx = np.argmax(sharpe_ratios)
        min_vol_idx = np.argmin(volatilities)

        simulation_result = {
            "n_simulations": n_simulations,
            "time_horizon_days": time_horizon,
            "statistics": {
                "return_mean": np.mean(returns),
                "return_std": np.std(returns),
                "volatility_mean": np.mean(volatilities),
                "volatility_std": np.std(volatilities),
                "sharpe_mean": np.mean(sharpe_ratios),
                "final_value_mean": np.mean(final_values),
                "final_value_std": np.std(final_values),
            },
            "efficient_frontier": {
                "max_sharpe": {
                    "weights": dict(zip(symbols, portfolio_results[max_sharpe_idx]["weights"])),
                    "expected_return": portfolio_results[max_sharpe_idx]["expected_return"],
                    "volatility": portfolio_results[max_sharpe_idx]["volatility"],
                    "sharpe_ratio": portfolio_results[max_sharpe_idx]["sharpe_ratio"],
                },
                "min_volatility": {
                    "weights": dict(zip(symbols, portfolio_results[min_vol_idx]["weights"])),
                    "expected_return": portfolio_results[min_vol_idx]["expected_return"],
                    "volatility": portfolio_results[min_vol_idx]["volatility"],
                    "sharpe_ratio": portfolio_results[min_vol_idx]["sharpe_ratio"],
                },
            },
            "percentiles": {
                "final_value_5th": np.percentile(final_values, 5),
                "final_value_50th": np.percentile(final_values, 50),
                "final_value_95th": np.percentile(final_values, 95),
            },
        }

        logger.info("Monte Carloシミュレーション完了:")
        logger.info(
            f"  - 最大シャープレシオ: {simulation_result['efficient_frontier']['max_sharpe']['sharpe_ratio']:.2f}"
        )
        logger.info(
            f"  - 最小ボラティリティ: {simulation_result['efficient_frontier']['min_volatility']['volatility']:.2%}"
        )

        return simulation_result

    def generate_portfolio_recommendation(
        self, price_data: Dict[str, pd.DataFrame], investment_amount: float = 1000000
    ) -> Dict:
        """
        総合的なポートフォリオ推奨を生成

        Args:
            price_data: 価格データ
            investment_amount: 投資金額

        Returns:
            推奨ポートフォリオ結果
        """
        logger.info("ポートフォリオ推奨生成開始")

        try:
            # 1. リターンとリスク計算
            expected_returns, covariance_matrix = self.calculate_returns_and_risks(price_data)

            # 2. ポートフォリオ最適化
            optimization_result = self.optimize_portfolio(
                expected_returns, covariance_matrix, investment_amount
            )

            # 3. Monte Carloシミュレーション
            monte_carlo_result = self.monte_carlo_simulation(expected_returns, covariance_matrix)

            # 4. 総合推奨生成
            recommendation = {
                "timestamp": pd.Timestamp.now(),
                "investment_amount": investment_amount,
                "optimal_portfolio": optimization_result,
                "simulation_analysis": monte_carlo_result,
                "risk_analysis": {
                    "portfolio_beta": self._calculate_portfolio_beta(
                        optimization_result["allocations"], covariance_matrix
                    ),
                    "diversification_ratio": self._calculate_diversification_ratio(
                        optimization_result["allocations"], covariance_matrix
                    ),
                },
                "recommendations": self._generate_text_recommendations(
                    optimization_result, monte_carlo_result
                ),
            }

            logger.info("ポートフォリオ推奨生成完了")
            return recommendation

        except Exception as e:
            logger.error(f"ポートフォリオ推奨生成エラー: {e}")
            return self._get_fallback_recommendation(investment_amount)

    def _calculate_portfolio_beta(
        self, allocations: Dict, covariance_matrix: pd.DataFrame
    ) -> float:
        """ポートフォリオベータ計算"""
        try:
            symbols = list(allocations.keys())
            weights = np.array([allocations[s]["weight"] for s in symbols])

            # 市場ベンチマーク（等ウェイト）との相関
            market_weights = np.ones(len(symbols)) / len(symbols)
            market_var = np.dot(
                market_weights.T,
                np.dot(covariance_matrix.loc[symbols, symbols], market_weights),
            )

            covariance = np.dot(
                weights.T,
                np.dot(covariance_matrix.loc[symbols, symbols], market_weights),
            )
            beta = covariance / market_var if market_var > 0 else 1.0

            return float(beta)
        except Exception:
            return 1.0

    def _calculate_diversification_ratio(
        self, allocations: Dict, covariance_matrix: pd.DataFrame
    ) -> float:
        """分散度合い計算"""
        try:
            symbols = list(allocations.keys())
            weights = np.array([allocations[s]["weight"] for s in symbols])

            # 加重平均個別リスク
            individual_risks = np.sqrt(np.diag(covariance_matrix.loc[symbols, symbols]))
            weighted_avg_risk = np.dot(weights, individual_risks)

            # ポートフォリオリスク
            portfolio_risk = np.sqrt(
                np.dot(weights.T, np.dot(covariance_matrix.loc[symbols, symbols], weights))
            )

            diversification_ratio = (
                weighted_avg_risk / portfolio_risk if portfolio_risk > 0 else 1.0
            )
            return float(diversification_ratio)
        except Exception:
            return 1.0

    def _generate_text_recommendations(
        self, optimization_result: Dict, monte_carlo_result: Dict
    ) -> List[str]:
        """テキスト形式の推奨事項生成"""
        recommendations = []

        # ポートフォリオ特徴分析
        n_positions = optimization_result["n_positions"]
        volatility = optimization_result["portfolio_metrics"]["volatility"]
        sharpe_ratio = optimization_result["portfolio_metrics"]["sharpe_ratio"]

        # 基本評価
        if sharpe_ratio > 1.0:
            recommendations.append("優れたリスク調整後リターンが期待できるポートフォリオです")
        elif sharpe_ratio > 0.5:
            recommendations.append("バランスの取れたリスク・リターン特性のポートフォリオです")
        else:
            recommendations.append("リスクに対するリターンが限定的なポートフォリオです")

        # 分散性評価
        if n_positions >= 10:
            recommendations.append("適切に分散されたポートフォリオ構成です")
        elif n_positions >= 5:
            recommendations.append("中程度に分散されたポートフォリオです")
        else:
            recommendations.append("集中投資型のポートフォリオです。分散を検討してください")

        # リスク評価
        if volatility < 0.15:
            recommendations.append("低リスクポートフォリオです")
        elif volatility < 0.25:
            recommendations.append("中程度のリスクポートフォリオです")
        else:
            recommendations.append("高リスクポートフォリオです。リスク許容度を確認してください")

        # モンテカルロ分析
        final_value_mean = monte_carlo_result["statistics"]["final_value_mean"]
        if final_value_mean > 1.1:
            recommendations.append("1年後の期待価値は投資元本を10%以上上回る見込みです")
        elif final_value_mean > 1.0:
            recommendations.append("1年後の期待価値は投資元本をやや上回る見込みです")
        else:
            recommendations.append("1年後の期待価値は投資元本を下回るリスクがあります")

        return recommendations

    def _get_fallback_recommendation(self, investment_amount: float) -> Dict:
        """エラー時のフォールバック推奨"""
        return {
            "timestamp": pd.Timestamp.now(),
            "investment_amount": investment_amount,
            "error": "ポートフォリオ最適化に失敗しました",
            "fallback_recommendation": "等ウェイト分散投資を推奨します",
            "optimal_portfolio": {
                "allocations": {},
                "portfolio_metrics": {
                    "expected_return": 0.05,
                    "volatility": 0.20,
                    "sharpe_ratio": 0.25,
                },
            },
        }


if __name__ == "__main__":
    # 使用例
    print("ポートフォリオオプティマイザーテスト")

    # サンプルデータ生成

    sample_data = {}
    symbols = ["7203", "8306", "9984", "6758", "4563"]
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    np.random.seed(42)
    for symbol in symbols:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        sample_data[symbol] = pd.DataFrame({"Close": prices}, index=dates)

    # オプティマイザー初期化
    optimizer = PortfolioOptimizer(
        risk_tolerance=0.6, max_position_size=0.4, min_position_size=0.05
    )

    try:
        # ポートフォリオ推奨生成
        recommendation = optimizer.generate_portfolio_recommendation(
            sample_data, investment_amount=1000000
        )

        print("\n=== 推奨ポートフォリオ ===")
        for symbol, allocation in recommendation["optimal_portfolio"]["allocations"].items():
            print(f"{symbol}: {allocation['weight']:.1%} ({allocation['amount']:,.0f}円)")

        print(
            f"\n期待リターン: {recommendation['optimal_portfolio']['portfolio_metrics']['expected_return']:.2%}"
        )
        print(
            f"ボラティリティ: {recommendation['optimal_portfolio']['portfolio_metrics']['volatility']:.2%}"
        )
        print(
            f"シャープレシオ: {recommendation['optimal_portfolio']['portfolio_metrics']['sharpe_ratio']:.2f}"
        )

        print("\n=== 推奨事項 ===")
        for rec in recommendation["recommendations"]:
            print(f"• {rec}")

    except Exception as e:
        print(f"エラー: {e}")
