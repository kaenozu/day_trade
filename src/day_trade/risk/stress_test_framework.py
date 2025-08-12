#!/usr/bin/env python3
"""
ストレステストフレームワーク

Issue #316: 高優先：リスク管理機能強化
金融危機シナリオでのポートフォリオ耐性評価
"""

import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 必要に応じてsklearnをインポート
# from sklearn.preprocessing import StandardScaler


class StressScenario(Enum):
    """ストレスシナリオ"""

    MARKET_CRASH = "market_crash"  # 市場暴落
    SECTOR_ROTATION = "sector_rotation"  # セクター回転
    INTEREST_RATE_SHOCK = "rate_shock"  # 金利ショック
    CURRENCY_CRISIS = "currency_crisis"  # 通貨危機
    LIQUIDITY_CRISIS = "liquidity_crisis"  # 流動性危機
    INFLATION_SPIKE = "inflation_spike"  # インフレ急騰
    GEOPOLITICAL_CRISIS = "geopolitical"  # 地政学リスク
    BLACK_SWAN = "black_swan"  # ブラックスワン
    CUSTOM = "custom"  # カスタムシナリオ


@dataclass
class StressTestResult:
    """ストレステスト結果"""

    scenario_name: str
    scenario_type: StressScenario
    test_date: str

    # ポートフォリオインパクト
    initial_value: float
    stressed_value: float
    absolute_loss: float
    percentage_loss: float

    # リスクメトリクス
    stressed_var_95: float
    stressed_cvar_95: float
    stressed_volatility: float
    max_drawdown: float

    # 銘柄別インパクト
    individual_impacts: Dict[str, float]
    worst_performers: List[Tuple[str, float]]
    best_performers: List[Tuple[str, float]]

    # 回復シミュレーション
    recovery_time_estimate: Optional[int] = None
    recovery_probability: Optional[float] = None

    # その他
    confidence_level: float = 0.95
    simulation_runs: int = 1000


@dataclass
class ScenarioParameters:
    """シナリオパラメータ"""

    name: str
    duration_days: int
    severity: float  # 0-1の厳しさ
    correlation_increase: float  # 相関係数増加
    volatility_multiplier: float
    return_shock: Dict[str, float]  # 銘柄別リターンショック
    sector_impacts: Dict[str, float]  # セクター別インパクト


class AdvancedStressTestFramework:
    """高度ストレステストフレームワーク"""

    def __init__(self, confidence_level: float = 0.95, simulation_runs: int = 1000):
        """
        初期化

        Args:
            confidence_level: 信頼水準
            simulation_runs: シミュレーション回数
        """
        self.confidence_level = confidence_level
        self.simulation_runs = simulation_runs

        # 履歴データ保存用
        self.stress_test_history = []

        # 予定義シナリオ
        self.predefined_scenarios = self._create_predefined_scenarios()

        print("高度ストレステストフレームワーク初期化完了")

    def _create_predefined_scenarios(self) -> Dict[StressScenario, ScenarioParameters]:
        """予定義シナリオ作成"""
        scenarios = {}

        # 2008年リーマンショック風
        scenarios[StressScenario.MARKET_CRASH] = ScenarioParameters(
            name="市場暴落シナリオ",
            duration_days=60,
            severity=0.9,
            correlation_increase=0.4,
            volatility_multiplier=3.0,
            return_shock={"default": -0.40},  # 40%下落
            sector_impacts={
                "finance": -0.50,
                "technology": -0.30,
                "healthcare": -0.20,
                "utilities": -0.15,
            },
        )

        # 金利急上昇
        scenarios[StressScenario.INTEREST_RATE_SHOCK] = ScenarioParameters(
            name="金利ショックシナリオ",
            duration_days=30,
            severity=0.7,
            correlation_increase=0.2,
            volatility_multiplier=2.0,
            return_shock={"default": -0.15},  # 15%下落
            sector_impacts={
                "finance": 0.05,  # 金融は若干プラス
                "utilities": -0.25,  # 公益は大幅マイナス
                "real_estate": -0.30,  # 不動産は大幅マイナス
                "technology": -0.10,
            },
        )

        # 流動性危機
        scenarios[StressScenario.LIQUIDITY_CRISIS] = ScenarioParameters(
            name="流動性危機シナリオ",
            duration_days=21,
            severity=0.8,
            correlation_increase=0.6,  # 相関大幅上昇
            volatility_multiplier=4.0,
            return_shock={"default": -0.25},
            sector_impacts={
                "small_cap": -0.40,  # 小型株は大きな影響
                "large_cap": -0.20,  # 大型株は比較的軽微
                "finance": -0.35,
            },
        )

        # インフレ急騰
        scenarios[StressScenario.INFLATION_SPIKE] = ScenarioParameters(
            name="インフレ急騰シナリオ",
            duration_days=90,
            severity=0.6,
            correlation_increase=0.3,
            volatility_multiplier=1.8,
            return_shock={"default": -0.12},
            sector_impacts={
                "materials": 0.10,  # 資源株はプラス
                "energy": 0.15,  # エネルギーはプラス
                "technology": -0.20,  # テクノロジーはマイナス
                "utilities": -0.15,
            },
        )

        # 地政学リスク
        scenarios[StressScenario.GEOPOLITICAL_CRISIS] = ScenarioParameters(
            name="地政学危機シナリオ",
            duration_days=45,
            severity=0.75,
            correlation_increase=0.35,
            volatility_multiplier=2.5,
            return_shock={"default": -0.20},
            sector_impacts={
                "defense": 0.05,  # 防衛関連はプラス
                "energy": 0.08,  # エネルギーはプラス
                "airlines": -0.35,  # 航空は大幅マイナス
                "tourism": -0.40,  # 観光は大幅マイナス
            },
        )

        # ブラックスワン（極端な事象）
        scenarios[StressScenario.BLACK_SWAN] = ScenarioParameters(
            name="ブラックスワンシナリオ",
            duration_days=14,
            severity=1.0,
            correlation_increase=0.8,  # 相関ほぼ1.0に
            volatility_multiplier=5.0,
            return_shock={"default": -0.60},  # 60%暴落
            sector_impacts={"default": -0.60},
        )

        return scenarios

    def run_stress_test(
        self,
        portfolio_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        scenario: StressScenario,
        custom_parameters: Optional[ScenarioParameters] = None,
    ) -> StressTestResult:
        """
        ストレステスト実行

        Args:
            portfolio_weights: ポートフォリオウェイト
            price_data: 価格データ
            scenario: ストレスシナリオ
            custom_parameters: カスタムパラメータ

        Returns:
            ストレステスト結果
        """
        try:
            # シナリオパラメータ取得
            if custom_parameters:
                params = custom_parameters
            elif scenario in self.predefined_scenarios:
                params = self.predefined_scenarios[scenario]
            else:
                raise ValueError(f"未定義のシナリオ: {scenario}")

            # 初期ポートフォリオ価値計算
            initial_value = self._calculate_portfolio_value(
                portfolio_weights, price_data
            )

            # Monte Carloシミュレーション実行
            simulation_results = self._run_monte_carlo_simulation(
                portfolio_weights, price_data, params
            )

            # 結果分析
            stressed_values = simulation_results["portfolio_values"]
            individual_impacts = simulation_results["individual_impacts"]

            # 統計計算
            stressed_value = np.mean(stressed_values)
            absolute_loss = initial_value - stressed_value
            percentage_loss = absolute_loss / initial_value

            # VaRとCVaR計算
            sorted_values = np.sort(stressed_values)
            var_index = int((1 - self.confidence_level) * len(sorted_values))
            stressed_var_95 = initial_value - sorted_values[var_index]

            # CVaR（Expected Shortfall）
            tail_values = sorted_values[:var_index]
            stressed_cvar_95 = (
                initial_value - np.mean(tail_values)
                if len(tail_values) > 0
                else stressed_var_95
            )

            # ボラティリティ
            returns = (np.array(stressed_values) - initial_value) / initial_value
            stressed_volatility = np.std(returns)

            # 最大ドローダウン推定
            max_drawdown = percentage_loss * 1.2  # 保守的見積もり

            # 最悪・最良パフォーマー特定
            avg_individual_impacts = {
                k: np.mean(v) for k, v in individual_impacts.items()
            }
            sorted_impacts = sorted(avg_individual_impacts.items(), key=lambda x: x[1])

            worst_performers = sorted_impacts[:3]  # 下位3つ
            best_performers = sorted_impacts[-3:][::-1]  # 上位3つ

            # 回復時間推定
            recovery_time = self._estimate_recovery_time(params, percentage_loss)
            recovery_probability = self._estimate_recovery_probability(
                params, percentage_loss
            )

            result = StressTestResult(
                scenario_name=params.name,
                scenario_type=scenario,
                test_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                initial_value=initial_value,
                stressed_value=stressed_value,
                absolute_loss=absolute_loss,
                percentage_loss=percentage_loss,
                stressed_var_95=stressed_var_95,
                stressed_cvar_95=stressed_cvar_95,
                stressed_volatility=stressed_volatility,
                max_drawdown=max_drawdown,
                individual_impacts=avg_individual_impacts,
                worst_performers=worst_performers,
                best_performers=best_performers,
                recovery_time_estimate=recovery_time,
                recovery_probability=recovery_probability,
                confidence_level=self.confidence_level,
                simulation_runs=self.simulation_runs,
            )

            # 履歴に保存
            self.stress_test_history.append(result)

            return result

        except Exception as e:
            print(f"ストレステスト実行エラー: {e}")
            raise

    def _calculate_portfolio_value(
        self,
        weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        base_value: float = 1000000,
    ) -> float:
        """ポートフォリオ価値計算"""
        if not weights:
            return base_value

        # 最新価格でポートフォリオ価値計算
        total_weight = sum(weights.values())
        if total_weight == 0:
            return base_value

        return base_value  # 簡略化: 実際は最新価格で計算

    def _run_monte_carlo_simulation(
        self,
        portfolio_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        params: ScenarioParameters,
    ) -> Dict[str, Any]:
        """Monte Carloシミュレーション実行"""

        portfolio_values = []
        individual_impacts = {symbol: [] for symbol in portfolio_weights}

        # 各銘柄の基本統計計算
        base_stats = self._calculate_base_statistics(price_data)

        for _ in range(self.simulation_runs):
            # ストレスシナリオ下での銘柄リターン生成
            stressed_returns = self._generate_stressed_returns(
                base_stats, params, portfolio_weights.keys()
            )

            # ポートフォリオ価値計算
            portfolio_return = sum(
                portfolio_weights.get(symbol, 0) * ret
                for symbol, ret in stressed_returns.items()
            )

            initial_value = 1000000  # 基準値
            stressed_value = initial_value * (1 + portfolio_return)
            portfolio_values.append(stressed_value)

            # 個別インパクト記録
            for symbol in portfolio_weights:
                individual_impacts[symbol].append(stressed_returns.get(symbol, 0))

        return {
            "portfolio_values": portfolio_values,
            "individual_impacts": individual_impacts,
        }

    def _calculate_base_statistics(
        self, price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """基本統計量計算"""
        base_stats = {}

        for symbol, data in price_data.items():
            if len(data) < 30:
                continue

            returns = data["Close"].pct_change().dropna()

            base_stats[symbol] = {
                "mean_return": returns.mean(),
                "volatility": returns.std(),
                "skewness": returns.skew() if len(returns) > 3 else 0,
                "kurtosis": returns.kurtosis() if len(returns) > 4 else 3,
            }

        return base_stats

    def _generate_stressed_returns(
        self,
        base_stats: Dict[str, Dict[str, float]],
        params: ScenarioParameters,
        symbols: List[str],
    ) -> Dict[str, float]:
        """ストレス下でのリターン生成"""
        stressed_returns = {}

        for symbol in symbols:
            if symbol not in base_stats:
                # デフォルト値
                base_return = 0
                base_volatility = 0.2
            else:
                stats = base_stats[symbol]
                base_return = stats["mean_return"]
                base_volatility = stats["volatility"]

            # ストレスパラメータ適用
            stress_multiplier = params.volatility_multiplier
            stressed_volatility = base_volatility * stress_multiplier

            # リターンショック適用
            return_shock = params.return_shock.get(
                symbol, params.return_shock.get("default", 0)
            )

            # 正規分布からサンプリング（簡略版）
            random_component = np.random.normal(0, stressed_volatility)
            stressed_return = base_return + return_shock + random_component

            stressed_returns[symbol] = stressed_return

        return stressed_returns

    def _estimate_recovery_time(
        self, params: ScenarioParameters, loss_percentage: float
    ) -> int:
        """回復時間推定"""
        # 簡単な経験則ベースの推定
        base_recovery_days = params.duration_days * 3  # 3倍の時間
        severity_multiplier = 1 + params.severity
        loss_multiplier = 1 + abs(loss_percentage)

        estimated_days = int(base_recovery_days * severity_multiplier * loss_multiplier)
        return min(estimated_days, 1000)  # 最大約3年

    def _estimate_recovery_probability(
        self, params: ScenarioParameters, loss_percentage: float
    ) -> float:
        """回復確率推定"""
        # 損失が大きいほど回復確率は低下
        base_probability = 0.8
        loss_penalty = abs(loss_percentage) * 0.5
        severity_penalty = params.severity * 0.2

        probability = base_probability - loss_penalty - severity_penalty
        return max(0.1, min(0.95, probability))

    def run_comprehensive_stress_test(
        self, portfolio_weights: Dict[str, float], price_data: Dict[str, pd.DataFrame]
    ) -> Dict[StressScenario, StressTestResult]:
        """包括的ストレステスト実行"""
        print("包括的ストレステスト実行中...")

        results = {}

        # 主要シナリオでテスト実行
        key_scenarios = [
            StressScenario.MARKET_CRASH,
            StressScenario.INTEREST_RATE_SHOCK,
            StressScenario.LIQUIDITY_CRISIS,
            StressScenario.GEOPOLITICAL_CRISIS,
        ]

        for scenario in key_scenarios:
            try:
                print(f"シナリオ実行: {scenario.value}")
                result = self.run_stress_test(portfolio_weights, price_data, scenario)
                results[scenario] = result
                print(f"完了: 予想損失 {result.percentage_loss:.1%}")
            except Exception as e:
                print(f"シナリオ {scenario.value} でエラー: {e}")

        return results

    def generate_stress_test_report(
        self, results: Dict[StressScenario, StressTestResult]
    ) -> str:
        """ストレステストレポート生成"""
        if not results:
            return "ストレステスト結果がありません。"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("包括的ストレステスト レポート")
        report_lines.append("=" * 80)

        report_lines.append(f"実施日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"信頼水準: {self.confidence_level:.0%}")
        report_lines.append(f"シミュレーション回数: {self.simulation_runs:,}回")

        # シナリオ別結果
        report_lines.append("\n【シナリオ別結果サマリー】")
        report_lines.append(
            f"{'シナリオ':<20} {'予想損失':<12} {'VaR(95%)':<12} {'CVaR(95%)':<12} {'回復時間'}"
        )
        report_lines.append("-" * 80)

        for scenario, result in results.items():
            recovery_time = (
                f"{result.recovery_time_estimate}日"
                if result.recovery_time_estimate
                else "N/A"
            )

            report_lines.append(
                f"{scenario.value:<20} "
                f"{result.percentage_loss:<12.1%} "
                f"{result.stressed_var_95 / result.initial_value:<12.1%} "
                f"{result.stressed_cvar_95 / result.initial_value:<12.1%} "
                f"{recovery_time}"
            )

        # 最悪シナリオ分析
        worst_scenario = max(results.values(), key=lambda x: x.percentage_loss)

        report_lines.append(f"\n【最悪シナリオ詳細: {worst_scenario.scenario_name}】")
        report_lines.append(
            f"予想損失: {worst_scenario.percentage_loss:.1%} ({worst_scenario.absolute_loss:,.0f}円)"
        )
        report_lines.append(f"VaR(95%): {worst_scenario.stressed_var_95:,.0f}円")
        report_lines.append(f"CVaR(95%): {worst_scenario.stressed_cvar_95:,.0f}円")
        report_lines.append(
            f"ストレス下ボラティリティ: {worst_scenario.stressed_volatility:.1%}"
        )

        if worst_scenario.worst_performers:
            report_lines.append("\n最悪パフォーマンス銘柄:")
            for symbol, impact in worst_scenario.worst_performers:
                report_lines.append(f"  {symbol}: {impact:.1%}")

        if worst_scenario.best_performers:
            report_lines.append("\n最良パフォーマンス銘柄:")
            for symbol, impact in worst_scenario.best_performers:
                report_lines.append(f"  {symbol}: {impact:.1%}")

        # 総合リスク評価
        avg_loss = np.mean([r.percentage_loss for r in results.values()])
        max_loss = max([r.percentage_loss for r in results.values()])

        report_lines.append("\n【総合リスク評価】")
        report_lines.append(f"平均予想損失: {avg_loss:.1%}")
        report_lines.append(f"最大予想損失: {max_loss:.1%}")

        if max_loss < 0.2:
            risk_level = "低リスク"
            recommendation = "現在のポートフォリオは比較的安全です。"
        elif max_loss < 0.4:
            risk_level = "中リスク"
            recommendation = "適度なリスク分散を検討してください。"
        else:
            risk_level = "高リスク"
            recommendation = "ポートフォリオの大幅な見直しが必要です。"

        report_lines.append(f"リスクレベル: {risk_level}")
        report_lines.append(f"推奨事項: {recommendation}")

        return "\n".join(report_lines)

    def save_stress_test_results(
        self, results: Dict[StressScenario, StressTestResult], filepath: str
    ):
        """ストレステスト結果保存"""
        try:
            # JSON形式で保存
            results_data = {}

            for scenario, result in results.items():
                results_data[scenario.value] = asdict(result)

            save_data = {
                "test_metadata": {
                    "test_date": datetime.now().isoformat(),
                    "confidence_level": self.confidence_level,
                    "simulation_runs": self.simulation_runs,
                    "framework_version": "1.0",
                },
                "results": results_data,
            }

            import json

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            print(f"ストレステスト結果保存完了: {filepath}")

        except Exception as e:
            print(f"結果保存エラー: {e}")


if __name__ == "__main__":
    # テスト実行
    print("ストレステストフレームワークテスト")
    print("=" * 50)

    # サンプルデータでテスト
    try:
        framework = AdvancedStressTestFramework(simulation_runs=100)  # 軽量版

        # サンプルポートフォリオ
        portfolio_weights = {
            "7203.T": 0.3,  # トヨタ
            "8306.T": 0.3,  # MUFG
            "9984.T": 0.4,  # ソフトバンクG
        }

        # サンプル価格データ生成
        import yfinance as yf

        print("実データ取得中...")
        price_data = {}

        for symbol in portfolio_weights:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")
                if not data.empty:
                    price_data[symbol] = data
                    print(f"取得成功: {symbol}")
            except Exception as e:
                print(f"データ取得エラー {symbol}: {e}")

        if price_data:
            # 市場暴落シナリオテスト
            print("\n市場暴落シナリオテスト実行中...")
            result = framework.run_stress_test(
                portfolio_weights, price_data, StressScenario.MARKET_CRASH
            )

            print("\n=== テスト結果 ===")
            print(f"シナリオ: {result.scenario_name}")
            print(f"予想損失: {result.percentage_loss:.1%}")
            print(f"VaR(95%): {result.stressed_var_95:,.0f}円")
            print(f"CVaR(95%): {result.stressed_cvar_95:,.0f}円")
            print(f"回復時間推定: {result.recovery_time_estimate}日")
            print(f"回復確率: {result.recovery_probability:.1%}")

            # 包括的テスト
            print("\n包括的ストレステスト実行中...")
            comprehensive_results = framework.run_comprehensive_stress_test(
                portfolio_weights, price_data
            )

            # レポート生成
            report = framework.generate_stress_test_report(comprehensive_results)
            print("\n" + report)

            # 結果保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            framework.save_stress_test_results(
                comprehensive_results, f"stress_test_results_{timestamp}.json"
            )

        else:
            print("価格データの取得に失敗しました")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()
