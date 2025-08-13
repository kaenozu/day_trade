#!/usr/bin/env python3
"""
リスク管理システム

ポートフォリオのリスク計測、制御、監視を行う
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RiskManager:
    """
    ポートフォリオリスク管理クラス

    VaR計算、ストレステスト、相関分析等を提供
    """

    def __init__(
        self,
        confidence_level: float = 0.05,
        time_horizon: int = 1,
        max_correlation: float = 0.8,
        max_sector_weight: float = 0.4,
    ):
        """
        初期化

        Args:
            confidence_level: VaR信頼水準 (0.05 = 5%)
            time_horizon: 時間軸（日数）
            max_correlation: 最大相関係数
            max_sector_weight: セクター最大ウェイト
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.max_correlation = max_correlation
        self.max_sector_weight = max_sector_weight

        logger.info("リスクマネージャー初期化:")
        logger.info(f"  - VaR信頼水準: {(1 - confidence_level) * 100:.0f}%")
        logger.info(f"  - 時間軸: {time_horizon}日")
        logger.info(f"  - 最大相関係数: {max_correlation:.1f}")

    def calculate_value_at_risk(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> Dict:
        """
        Value at Risk (VaR) 計算

        Args:
            portfolio_weights: ポートフォリオウェイト
            returns_data: 収益率データ
            portfolio_value: ポートフォリオ価値

        Returns:
            VaR計算結果
        """
        logger.info("VaR計算開始")

        try:
            # ポートフォリオ収益率計算
            symbols = list(portfolio_weights.keys())

            # データの整合性チェック
            available_symbols = [s for s in symbols if s in returns_data.columns]
            if len(available_symbols) != len(symbols):
                missing = set(symbols) - set(available_symbols)
                logger.warning(f"データ未存在銘柄: {missing}")

            # 利用可能なデータで再計算
            available_weights = np.array(
                [portfolio_weights[s] for s in available_symbols]
            )
            available_weights = available_weights / available_weights.sum()  # 正規化

            portfolio_returns = (
                returns_data[available_symbols] * available_weights
            ).sum(axis=1)

            # ヒストリカル VaR
            historical_var = np.percentile(
                portfolio_returns, self.confidence_level * 100
            )

            # パラメトリック VaR (正規分布仮定)
            returns_mean = portfolio_returns.mean()
            returns_std = portfolio_returns.std()
            parametric_var = returns_mean - returns_std * 1.645  # 95%信頼区間

            # 修正 VaR (歪度・尖度考慮)
            from scipy import stats

            skewness = stats.skew(portfolio_returns)
            kurtosis = stats.kurtosis(portfolio_returns)

            # Cornish-Fisher展開
            z_alpha = stats.norm.ppf(self.confidence_level)
            cf_correction = (
                z_alpha
                + (z_alpha**2 - 1) * skewness / 6
                + (z_alpha**3 - 3 * z_alpha) * kurtosis / 24
                - (2 * z_alpha**3 - 5 * z_alpha) * skewness**2 / 36
            )
            modified_var = returns_mean + returns_std * cf_correction

            # 時間軸調整
            time_adjustment = np.sqrt(self.time_horizon)
            historical_var_adjusted = historical_var * time_adjustment
            parametric_var_adjusted = parametric_var * time_adjustment
            modified_var_adjusted = modified_var * time_adjustment

            # 金額換算
            var_results = {
                "confidence_level": 1 - self.confidence_level,
                "time_horizon": self.time_horizon,
                "portfolio_value": portfolio_value,
                "returns_statistics": {
                    "mean": returns_mean,
                    "std": returns_std,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                },
                "var_estimates": {
                    "historical": {
                        "return_var": historical_var_adjusted,
                        "value_var": abs(historical_var_adjusted * portfolio_value),
                    },
                    "parametric": {
                        "return_var": parametric_var_adjusted,
                        "value_var": abs(parametric_var_adjusted * portfolio_value),
                    },
                    "modified": {
                        "return_var": modified_var_adjusted,
                        "value_var": abs(modified_var_adjusted * portfolio_value),
                    },
                },
                "data_quality": {
                    "n_observations": len(portfolio_returns),
                    "available_symbols": len(available_symbols),
                    "missing_symbols": len(symbols) - len(available_symbols),
                },
            }

            logger.info("VaR計算完了:")
            logger.info(
                f"  - ヒストリカルVaR: {var_results['var_estimates']['historical']['value_var']:,.0f}円"
            )
            logger.info(
                f"  - パラメトリックVaR: {var_results['var_estimates']['parametric']['value_var']:,.0f}円"
            )

            return var_results

        except Exception as e:
            logger.error(f"VaR計算エラー: {e}")
            return self._get_fallback_var_result(portfolio_value)

    def calculate_expected_shortfall(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> Dict:
        """
        Expected Shortfall (ES) / Conditional VaR 計算

        Args:
            portfolio_weights: ポートフォリオウェイト
            returns_data: 収益率データ
            portfolio_value: ポートフォリオ価値

        Returns:
            ES計算結果
        """
        logger.info("Expected Shortfall計算開始")

        try:
            # ポートフォリオ収益率計算
            symbols = list(portfolio_weights.keys())
            available_symbols = [s for s in symbols if s in returns_data.columns]
            available_weights = np.array(
                [portfolio_weights[s] for s in available_symbols]
            )
            available_weights = available_weights / available_weights.sum()

            portfolio_returns = (
                returns_data[available_symbols] * available_weights
            ).sum(axis=1)

            # VaR閾値計算
            var_threshold = np.percentile(
                portfolio_returns, self.confidence_level * 100
            )

            # Expected Shortfall = VaR以下の収益率の平均
            tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
            expected_shortfall = (
                tail_losses.mean() if len(tail_losses) > 0 else var_threshold
            )

            # 時間軸・金額調整
            time_adjustment = np.sqrt(self.time_horizon)
            es_adjusted = expected_shortfall * time_adjustment
            es_value = abs(es_adjusted * portfolio_value)

            es_results = {
                "confidence_level": 1 - self.confidence_level,
                "time_horizon": self.time_horizon,
                "var_threshold": var_threshold * time_adjustment,
                "expected_shortfall": {
                    "return_es": es_adjusted,
                    "value_es": es_value,
                },
                "tail_statistics": {
                    "n_tail_observations": len(tail_losses),
                    "tail_probability": len(tail_losses) / len(portfolio_returns),
                    "worst_loss": portfolio_returns.min() * time_adjustment,
                },
            }

            logger.info(f"Expected Shortfall: {es_value:,.0f}円")
            return es_results

        except Exception as e:
            logger.error(f"ES計算エラー: {e}")
            return {"error": str(e)}

    def analyze_correlations(
        self, returns_data: pd.DataFrame, portfolio_weights: Dict[str, float]
    ) -> Dict:
        """
        相関分析

        Args:
            returns_data: 収益率データ
            portfolio_weights: ポートフォリオウェイト

        Returns:
            相関分析結果
        """
        logger.info("相関分析開始")

        try:
            symbols = list(portfolio_weights.keys())
            available_symbols = [s for s in symbols if s in returns_data.columns]

            if len(available_symbols) < 2:
                return {"error": "相関分析には最低2銘柄必要"}

            # 相関行列計算
            correlation_matrix = returns_data[available_symbols].corr()

            # 高相関ペア検出
            high_correlations = []
            for i, symbol1 in enumerate(available_symbols):
                for symbol2 in available_symbols[i + 1 :]:
                    corr_value = correlation_matrix.loc[symbol1, symbol2]
                    if abs(corr_value) > self.max_correlation:
                        high_correlations.append(
                            {
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": corr_value,
                                "weight1": portfolio_weights.get(symbol1, 0),
                                "weight2": portfolio_weights.get(symbol2, 0),
                            }
                        )

            # ポートフォリオ集中度分析
            weights_array = np.array(
                [portfolio_weights.get(s, 0) for s in available_symbols]
            )
            concentration_metrics = {
                "herfindahl_index": np.sum(weights_array**2),
                "effective_positions": (
                    1 / np.sum(weights_array**2) if np.sum(weights_array**2) > 0 else 0
                ),
                "max_weight": weights_array.max(),
                "top3_weight": np.sort(weights_array)[-3:].sum(),
            }

            correlation_results = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlations": high_correlations,
                "concentration_metrics": concentration_metrics,
                "correlation_statistics": {
                    "avg_correlation": correlation_matrix.values[
                        np.triu_indices_from(correlation_matrix.values, k=1)
                    ].mean(),
                    "max_correlation": correlation_matrix.values[
                        np.triu_indices_from(correlation_matrix.values, k=1)
                    ].max(),
                    "min_correlation": correlation_matrix.values[
                        np.triu_indices_from(correlation_matrix.values, k=1)
                    ].min(),
                },
                "risk_warnings": self._generate_correlation_warnings(
                    high_correlations, concentration_metrics
                ),
            }

            logger.info("相関分析完了:")
            logger.info(
                f"  - 平均相関: {correlation_results['correlation_statistics']['avg_correlation']:.3f}"
            )
            logger.info(f"  - 高相関ペア数: {len(high_correlations)}")
            logger.info(
                f"  - 実効ポジション数: {concentration_metrics['effective_positions']:.1f}"
            )

            return correlation_results

        except Exception as e:
            logger.error(f"相関分析エラー: {e}")
            return {"error": str(e)}

    def stress_test_portfolio(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
        scenarios: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        ストレステスト実行

        Args:
            portfolio_weights: ポートフォリオウェイト
            returns_data: 収益率データ
            portfolio_value: ポートフォリオ価値
            scenarios: カスタムシナリオ

        Returns:
            ストレステスト結果
        """
        logger.info("ストレステスト開始")

        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()

        try:
            symbols = list(portfolio_weights.keys())
            available_symbols = [s for s in symbols if s in returns_data.columns]
            available_weights = np.array(
                [portfolio_weights[s] for s in available_symbols]
            )
            available_weights = available_weights / available_weights.sum()

            stress_results = []

            for scenario in scenarios:
                scenario_name = scenario["name"]
                shock_type = scenario["type"]

                if shock_type == "market_crash":
                    # 市場全体下落シナリオ
                    shock_returns = np.full(
                        len(available_symbols), scenario["shock_value"]
                    )

                elif shock_type == "sector_shock":
                    # セクター別ショック（簡易実装）
                    shock_returns = np.random.normal(
                        scenario["shock_value"], 0.05, len(available_symbols)
                    )

                elif shock_type == "correlation_breakdown":
                    # 相関構造崩壊シナリオ
                    shock_returns = np.random.normal(
                        scenario["shock_value"], 0.10, len(available_symbols)
                    )

                else:
                    # デフォルト：一律ショック
                    shock_returns = np.full(
                        len(available_symbols), scenario["shock_value"]
                    )

                # ポートフォリオへの影響計算
                portfolio_shock = np.dot(available_weights, shock_returns)
                value_impact = portfolio_shock * portfolio_value

                stress_results.append(
                    {
                        "scenario": scenario_name,
                        "shock_type": shock_type,
                        "portfolio_return": portfolio_shock,
                        "value_impact": value_impact,
                        "final_value": portfolio_value * (1 + portfolio_shock),
                        "loss_percentage": abs(portfolio_shock) * 100,
                    }
                )

            # 最悪ケースシナリオ特定
            worst_scenario = min(stress_results, key=lambda x: x["portfolio_return"])

            stress_test_results = {
                "portfolio_value": portfolio_value,
                "scenarios": stress_results,
                "worst_case": worst_scenario,
                "stress_summary": {
                    "avg_loss": np.mean(
                        [
                            s["portfolio_return"]
                            for s in stress_results
                            if s["portfolio_return"] < 0
                        ]
                    ),
                    "max_loss": worst_scenario["portfolio_return"],
                    "scenarios_with_loss": len(
                        [s for s in stress_results if s["portfolio_return"] < 0]
                    ),
                },
                "risk_assessment": self._assess_stress_risk(stress_results),
            }

            logger.info("ストレステスト完了:")
            logger.info(f"  - 最大損失シナリオ: {worst_scenario['scenario']}")
            logger.info(f"  - 最大損失: {abs(worst_scenario['value_impact']):,.0f}円")

            return stress_test_results

        except Exception as e:
            logger.error(f"ストレステストエラー: {e}")
            return {"error": str(e)}

    def _get_default_stress_scenarios(self) -> List[Dict]:
        """デフォルトストレスシナリオ"""
        return [
            {"name": "軽微な市場調整", "type": "market_crash", "shock_value": -0.05},
            {"name": "中程度の市場下落", "type": "market_crash", "shock_value": -0.15},
            {
                "name": "深刻な市場クラッシュ",
                "type": "market_crash",
                "shock_value": -0.30,
            },
            {
                "name": "セクターローテーション",
                "type": "sector_shock",
                "shock_value": -0.10,
            },
            {
                "name": "相関構造崩壊",
                "type": "correlation_breakdown",
                "shock_value": -0.20,
            },
        ]

    def _generate_correlation_warnings(
        self, high_correlations: List[Dict], concentration_metrics: Dict
    ) -> List[str]:
        """相関リスク警告生成"""
        warnings = []

        if len(high_correlations) > 0:
            warnings.append(f"{len(high_correlations)}組の高相関ペアが検出されました")

        if concentration_metrics["herfindahl_index"] > 0.25:
            warnings.append("ポートフォリオが集中しすぎています")

        if concentration_metrics["effective_positions"] < 3:
            warnings.append("実効的な分散が不十分です")

        if concentration_metrics["max_weight"] > 0.3:
            warnings.append("単一銘柄の比重が過大です")

        return warnings

    def _assess_stress_risk(self, stress_results: List[Dict]) -> str:
        """ストレスリスク評価"""
        max_loss = abs(min(s["portfolio_return"] for s in stress_results))

        if max_loss > 0.30:
            return "HIGH_RISK"
        elif max_loss > 0.15:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"

    def _get_fallback_var_result(self, portfolio_value: float) -> Dict:
        """VaRフォールバック結果"""
        return {
            "error": "VaR計算に失敗",
            "fallback_estimate": {
                "value_var": portfolio_value * 0.05,  # 5%固定推定
                "confidence_level": 0.95,
            },
        }

    def generate_risk_report(
        self,
        portfolio_weights: Dict[str, float],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> Dict:
        """
        包括的リスクレポート生成

        Args:
            portfolio_weights: ポートフォリオウェイト
            returns_data: 収益率データ
            portfolio_value: ポートフォリオ価値

        Returns:
            包括的リスクレポート
        """
        logger.info("包括的リスクレポート生成開始")

        try:
            # 各種リスク指標計算
            var_results = self.calculate_value_at_risk(
                portfolio_weights, returns_data, portfolio_value
            )
            es_results = self.calculate_expected_shortfall(
                portfolio_weights, returns_data, portfolio_value
            )
            correlation_results = self.analyze_correlations(
                returns_data, portfolio_weights
            )
            stress_results = self.stress_test_portfolio(
                portfolio_weights, returns_data, portfolio_value
            )

            # リスクレベル総合評価
            risk_factors = []

            # VaRベースリスク評価
            if "var_estimates" in var_results:
                var_ratio = (
                    var_results["var_estimates"]["historical"]["value_var"]
                    / portfolio_value
                )
                if var_ratio > 0.15:
                    risk_factors.append("HIGH_VAR")
                elif var_ratio > 0.08:
                    risk_factors.append("MEDIUM_VAR")

            # 相関リスク評価
            if (
                "risk_warnings" in correlation_results
                and len(correlation_results["risk_warnings"]) > 0
            ):
                risk_factors.append("CORRELATION_RISK")

            # ストレスリスク評価
            if "risk_assessment" in stress_results:
                risk_factors.append(stress_results["risk_assessment"])

            overall_risk = self._determine_overall_risk(risk_factors)

            risk_report = {
                "timestamp": pd.Timestamp.now(),
                "portfolio_value": portfolio_value,
                "overall_risk_level": overall_risk,
                "risk_components": {
                    "value_at_risk": var_results,
                    "expected_shortfall": es_results,
                    "correlation_analysis": correlation_results,
                    "stress_testing": stress_results,
                },
                "risk_summary": {
                    "key_risks": risk_factors,
                    "risk_recommendations": self._generate_risk_recommendations(
                        var_results, correlation_results, stress_results
                    ),
                },
            }

            logger.info(f"リスクレポート生成完了: {overall_risk}")
            return risk_report

        except Exception as e:
            logger.error(f"リスクレポート生成エラー: {e}")
            return {"error": str(e)}

    def _determine_overall_risk(self, risk_factors: List[str]) -> str:
        """総合リスクレベル決定"""
        high_risk_count = len([f for f in risk_factors if "HIGH" in f])
        medium_risk_count = len([f for f in risk_factors if "MEDIUM" in f])

        if high_risk_count >= 2:
            return "HIGH_RISK"
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            return "MEDIUM_RISK"
        else:
            return "LOW_RISK"

    def _generate_risk_recommendations(
        self, var_results: Dict, correlation_results: Dict, stress_results: Dict
    ) -> List[str]:
        """リスク対策推奨事項生成"""
        recommendations = []

        # VaR推奨
        if "var_estimates" in var_results:
            var_value = var_results["var_estimates"]["historical"]["value_var"]
            if var_value > 100000:  # 10万円以上
                recommendations.append(
                    "VaRが高いため、ポジションサイズの削減を検討してください"
                )

        # 相関推奨
        if (
            "high_correlations" in correlation_results
            and len(correlation_results["high_correlations"]) > 0
        ):
            recommendations.append(
                "高相関銘柄の配分見直しで分散効果を向上させてください"
            )

        # ストレス推奨
        if "worst_case" in stress_results:
            worst_loss = abs(stress_results["worst_case"]["portfolio_return"])
            if worst_loss > 0.20:
                recommendations.append(
                    "ストレスシナリオでの損失が大きいため、防御的ポジションを検討してください"
                )

        if len(recommendations) == 0:
            recommendations.append("現在のリスク水準は適切です")

        return recommendations


if __name__ == "__main__":
    # 使用例
    print("リスクマネージャーテスト")

    # サンプルデータ生成

    symbols = ["7203", "8306", "9984", "6758", "4563"]
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")

    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )

    # サンプルポートフォリオ
    portfolio_weights = {
        "7203": 0.25,
        "8306": 0.20,
        "9984": 0.20,
        "6758": 0.20,
        "4563": 0.15,
    }

    # リスクマネージャー初期化
    risk_manager = RiskManager()

    try:
        # 包括的リスクレポート
        risk_report = risk_manager.generate_risk_report(
            portfolio_weights, returns_data, portfolio_value=1000000
        )

        print("\n=== リスクレポート ===")
        print(f"総合リスクレベル: {risk_report['overall_risk_level']}")

        if "value_at_risk" in risk_report["risk_components"]:
            var_result = risk_report["risk_components"]["value_at_risk"]
            if "var_estimates" in var_result:
                hist_var = var_result["var_estimates"]["historical"]["value_var"]
                print(f"ヒストリカルVaR: {hist_var:,.0f}円")

        print("\n=== リスク推奨事項 ===")
        for rec in risk_report["risk_summary"]["risk_recommendations"]:
            print(f"• {rec}")

    except Exception as e:
        print(f"エラー: {e}")
