#!/usr/bin/env python3
"""
統合リスク管理システム

Issue #316: 高優先：リスク管理機能強化
動的リバランシング・ストレステスト・VaR計算の統合システム
"""

import warnings
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .dynamic_rebalancing import DynamicRebalancingEngine, MarketRegime, RiskMetrics
from .stress_test_framework import (
    AdvancedStressTestFramework,
    StressScenario,
    StressTestResult,
)

warnings.filterwarnings("ignore")


class RiskLevel(Enum):
    """リスクレベル"""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class AlertType(Enum):
    """アラートタイプ"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """リスクアラート"""

    timestamp: datetime
    alert_type: AlertType
    risk_category: str
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str
    urgency: int  # 1-10の緊急度


@dataclass
class PortfolioRiskProfile:
    """ポートフォリオリスクプロファイル"""

    timestamp: datetime
    overall_risk_level: RiskLevel

    # 基本リスクメトリクス
    portfolio_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    var_95_daily: float
    cvar_95_daily: float

    # ストレステスト結果
    worst_case_loss: float
    stress_var_95: float
    stress_recovery_time: Optional[int]

    # 市場環境
    market_regime: MarketRegime
    correlation_level: float

    # リバランシング状況
    rebalancing_needed: bool
    target_weights: Dict[str, float]
    rebalancing_urgency: float

    # アクティブアラート
    active_alerts: List[RiskAlert]


class IntegratedRiskManagementSystem:
    """統合リスク管理システム"""

    def __init__(
        self, risk_tolerance: str = "moderate", monitoring_frequency: int = 1440
    ):  # 分単位（1日）
        """
        初期化

        Args:
            risk_tolerance: リスク許容度 (conservative, moderate, aggressive)
            monitoring_frequency: 監視頻度（分）
        """
        self.risk_tolerance = risk_tolerance
        self.monitoring_frequency = monitoring_frequency

        # コンポーネント初期化
        self.rebalancing_engine = DynamicRebalancingEngine()
        self.stress_test_framework = AdvancedStressTestFramework(simulation_runs=500)

        # リスク閾値設定
        self.risk_thresholds = self._set_risk_thresholds(risk_tolerance)

        # 履歴データ保存
        self.risk_history = []
        self.alert_history = []

        print("統合リスク管理システム初期化完了")

    def _set_risk_thresholds(self, risk_tolerance: str) -> Dict[str, float]:
        """リスク許容度に応じた閾値設定"""

        thresholds = {
            "conservative": {
                "max_portfolio_volatility": 0.12,  # 12%
                "max_single_position": 0.20,  # 20%
                "max_drawdown_alert": 0.05,  # 5%
                "var_95_daily_alert": 0.02,  # 2%
                "stress_loss_alert": 0.15,  # 15%
                "correlation_alert": 0.7,  # 70%
                "rebalancing_threshold": 0.03,  # 3%
            },
            "moderate": {
                "max_portfolio_volatility": 0.18,  # 18%
                "max_single_position": 0.30,  # 30%
                "max_drawdown_alert": 0.10,  # 10%
                "var_95_daily_alert": 0.03,  # 3%
                "stress_loss_alert": 0.25,  # 25%
                "correlation_alert": 0.8,  # 80%
                "rebalancing_threshold": 0.05,  # 5%
            },
            "aggressive": {
                "max_portfolio_volatility": 0.25,  # 25%
                "max_single_position": 0.40,  # 40%
                "max_drawdown_alert": 0.15,  # 15%
                "var_95_daily_alert": 0.05,  # 5%
                "stress_loss_alert": 0.35,  # 35%
                "correlation_alert": 0.85,  # 85%
                "rebalancing_threshold": 0.07,  # 7%
            },
        }

        return thresholds.get(risk_tolerance, thresholds["moderate"])

    def assess_portfolio_risk(
        self,
        portfolio_weights: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        market_index_data: Optional[pd.DataFrame] = None,
    ) -> PortfolioRiskProfile:
        """
        ポートフォリオ総合リスク評価

        Args:
            portfolio_weights: ポートフォリオウェイト
            price_data: 価格データ
            market_index_data: 市場インデックスデータ

        Returns:
            リスクプロファイル
        """
        try:
            print("総合リスク評価実行中...")

            # 1. 基本リスクメトリクス計算
            risk_metrics = self.rebalancing_engine.calculate_portfolio_risk_metrics(
                portfolio_weights, price_data, market_index_data
            )

            # 2. 市場レジーム検出
            market_regime = self.rebalancing_engine.detect_market_regime(
                price_data, market_index_data
            )

            # 3. リバランシング評価
            rebalancing_signal = self.rebalancing_engine.generate_rebalancing_signal(
                portfolio_weights, price_data, market_index_data
            )

            # 4. ストレステスト実行（主要シナリオのみ）
            stress_results = self._run_key_stress_tests(portfolio_weights, price_data)

            # 5. 相関分析
            correlation_level = self._calculate_portfolio_correlation(
                price_data, portfolio_weights
            )

            # 6. 総合リスクレベル判定
            overall_risk_level = self._determine_overall_risk_level(
                risk_metrics, stress_results, correlation_level
            )

            # 7. アラート生成
            active_alerts = self._generate_risk_alerts(
                risk_metrics, stress_results, correlation_level, portfolio_weights
            )

            # リスクプロファイル作成
            risk_profile = PortfolioRiskProfile(
                timestamp=datetime.now(),
                overall_risk_level=overall_risk_level,
                portfolio_volatility=risk_metrics.portfolio_volatility,
                max_drawdown=risk_metrics.max_drawdown,
                sharpe_ratio=risk_metrics.sharpe_ratio,
                var_95_daily=risk_metrics.value_at_risk_95,
                cvar_95_daily=risk_metrics.expected_shortfall,
                worst_case_loss=max(
                    [r.percentage_loss for r in stress_results.values()]
                )
                if stress_results
                else 0,
                stress_var_95=max([r.stressed_var_95 for r in stress_results.values()])
                if stress_results
                else 0,
                stress_recovery_time=max(
                    [r.recovery_time_estimate or 0 for r in stress_results.values()]
                )
                if stress_results
                else None,
                market_regime=market_regime,
                correlation_level=correlation_level,
                rebalancing_needed=rebalancing_signal is not None,
                target_weights=rebalancing_signal.target_weights
                if rebalancing_signal
                else portfolio_weights,
                rebalancing_urgency=rebalancing_signal.rebalancing_strength
                if rebalancing_signal
                else 0,
                active_alerts=active_alerts,
            )

            # 履歴に保存
            self.risk_history.append(risk_profile)

            return risk_profile

        except Exception as e:
            print(f"リスク評価エラー: {e}")
            # エラー時はデフォルト値を返す
            return self._create_default_risk_profile(portfolio_weights)

    def _run_key_stress_tests(
        self, portfolio_weights: Dict[str, float], price_data: Dict[str, pd.DataFrame]
    ) -> Dict[StressScenario, StressTestResult]:
        """主要ストレステスト実行"""
        try:
            key_scenarios = [
                StressScenario.MARKET_CRASH,
                StressScenario.LIQUIDITY_CRISIS,
            ]
            results = {}

            for scenario in key_scenarios:
                try:
                    result = self.stress_test_framework.run_stress_test(
                        portfolio_weights, price_data, scenario
                    )
                    results[scenario] = result
                except Exception as e:
                    print(f"ストレステスト {scenario.value} エラー: {e}")

            return results

        except Exception as e:
            print(f"ストレステスト実行エラー: {e}")
            return {}

    def _calculate_portfolio_correlation(
        self, price_data: Dict[str, pd.DataFrame], weights: Dict[str, float]
    ) -> float:
        """ポートフォリオ内相関計算"""
        try:
            # 各銘柄のリターン計算
            returns_data = {}
            for symbol, data in price_data.items():
                if symbol in weights and len(data) > 30:
                    returns = data["Close"].pct_change().dropna()
                    returns_data[symbol] = returns.tail(252)  # 直近1年

            if len(returns_data) < 2:
                return 0.0

            # データフレーム作成
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()

            if returns_df.empty:
                return 0.0

            # 相関行列計算
            correlation_matrix = returns_df.corr()

            # 重み付き平均相関計算
            weighted_correlations = []
            symbols = list(returns_data.keys())

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol_i, symbol_j = symbols[i], symbols[j]
                    weight_i = weights.get(symbol_i, 0)
                    weight_j = weights.get(symbol_j, 0)

                    if weight_i > 0 and weight_j > 0:
                        correlation = correlation_matrix.loc[symbol_i, symbol_j]
                        if not np.isnan(correlation):
                            weighted_correlation = correlation * weight_i * weight_j
                            weighted_correlations.append(weighted_correlation)

            return np.mean(weighted_correlations) if weighted_correlations else 0.0

        except Exception as e:
            print(f"相関計算エラー: {e}")
            return 0.0

    def _determine_overall_risk_level(
        self,
        risk_metrics: RiskMetrics,
        stress_results: Dict[StressScenario, StressTestResult],
        correlation_level: float,
    ) -> RiskLevel:
        """総合リスクレベル判定"""
        risk_scores = []

        # ボラティリティスコア
        volatility = risk_metrics.portfolio_volatility
        if volatility > 0.30:
            risk_scores.append(5)  # 非常に高い
        elif volatility > 0.20:
            risk_scores.append(4)  # 高い
        elif volatility > 0.15:
            risk_scores.append(3)  # 中程度
        elif volatility > 0.10:
            risk_scores.append(2)  # 低い
        else:
            risk_scores.append(1)  # 非常に低い

        # ドローダウンスコア
        max_dd = abs(risk_metrics.max_drawdown)
        if max_dd > 0.25:
            risk_scores.append(5)
        elif max_dd > 0.15:
            risk_scores.append(4)
        elif max_dd > 0.10:
            risk_scores.append(3)
        elif max_dd > 0.05:
            risk_scores.append(2)
        else:
            risk_scores.append(1)

        # ストレステストスコア
        if stress_results:
            max_stress_loss = max([r.percentage_loss for r in stress_results.values()])
            if max_stress_loss > 0.50:
                risk_scores.append(6)  # 極端
            elif max_stress_loss > 0.35:
                risk_scores.append(5)
            elif max_stress_loss > 0.25:
                risk_scores.append(4)
            elif max_stress_loss > 0.15:
                risk_scores.append(3)
            elif max_stress_loss > 0.08:
                risk_scores.append(2)
            else:
                risk_scores.append(1)

        # 相関スコア
        if correlation_level > 0.85:
            risk_scores.append(4)
        elif correlation_level > 0.70:
            risk_scores.append(3)
        elif correlation_level > 0.50:
            risk_scores.append(2)
        else:
            risk_scores.append(1)

        # 平均スコアでレベル判定
        avg_score = np.mean(risk_scores)

        if avg_score >= 5.5:
            return RiskLevel.EXTREME
        elif avg_score >= 4.5:
            return RiskLevel.VERY_HIGH
        elif avg_score >= 3.5:
            return RiskLevel.HIGH
        elif avg_score >= 2.5:
            return RiskLevel.MODERATE
        elif avg_score >= 1.5:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW

    def _generate_risk_alerts(
        self,
        risk_metrics: RiskMetrics,
        stress_results: Dict[StressScenario, StressTestResult],
        correlation_level: float,
        portfolio_weights: Dict[str, float],
    ) -> List[RiskAlert]:
        """リスクアラート生成"""
        alerts = []
        current_time = datetime.now()

        # ボラティリティアラート
        if (
            risk_metrics.portfolio_volatility
            > self.risk_thresholds["max_portfolio_volatility"]
        ):
            alert = RiskAlert(
                timestamp=current_time,
                alert_type=AlertType.WARNING,
                risk_category="Portfolio Volatility",
                message="ポートフォリオボラティリティが閾値を超過",
                current_value=risk_metrics.portfolio_volatility,
                threshold_value=self.risk_thresholds["max_portfolio_volatility"],
                recommended_action="リスク分散の見直しまたはポジションサイズの削減",
                urgency=6,
            )
            alerts.append(alert)

        # ドローダウンアラート
        if abs(risk_metrics.max_drawdown) > self.risk_thresholds["max_drawdown_alert"]:
            alert = RiskAlert(
                timestamp=current_time,
                alert_type=AlertType.CRITICAL,
                risk_category="Maximum Drawdown",
                message="最大ドローダウンが警告レベルに到達",
                current_value=abs(risk_metrics.max_drawdown),
                threshold_value=self.risk_thresholds["max_drawdown_alert"],
                recommended_action="損切りルールの見直し・ポートフォリオ再構築",
                urgency=8,
            )
            alerts.append(alert)

        # VaRアラート
        if (
            abs(risk_metrics.value_at_risk_95)
            > self.risk_thresholds["var_95_daily_alert"]
        ):
            alert = RiskAlert(
                timestamp=current_time,
                alert_type=AlertType.WARNING,
                risk_category="Value at Risk",
                message="VaR(95%)が警告レベルを超過",
                current_value=abs(risk_metrics.value_at_risk_95),
                threshold_value=self.risk_thresholds["var_95_daily_alert"],
                recommended_action="リスクエクスポージャーの削減",
                urgency=7,
            )
            alerts.append(alert)

        # ストレステストアラート
        if stress_results:
            max_stress_loss = max([r.percentage_loss for r in stress_results.values()])
            if max_stress_loss > self.risk_thresholds["stress_loss_alert"]:
                alert = RiskAlert(
                    timestamp=current_time,
                    alert_type=AlertType.CRITICAL,
                    risk_category="Stress Test",
                    message="ストレステストで許容レベルを超過する損失",
                    current_value=max_stress_loss,
                    threshold_value=self.risk_thresholds["stress_loss_alert"],
                    recommended_action="危機耐性の向上・ヘッジ戦略の検討",
                    urgency=9,
                )
                alerts.append(alert)

        # 相関アラート
        if correlation_level > self.risk_thresholds["correlation_alert"]:
            alert = RiskAlert(
                timestamp=current_time,
                alert_type=AlertType.WARNING,
                risk_category="Portfolio Correlation",
                message="ポートフォリオ内相関が過度に高い",
                current_value=correlation_level,
                threshold_value=self.risk_thresholds["correlation_alert"],
                recommended_action="異なるセクター・地域への分散投資",
                urgency=5,
            )
            alerts.append(alert)

        # 集中リスクアラート
        max_weight = max(portfolio_weights.values()) if portfolio_weights else 0
        if max_weight > self.risk_thresholds["max_single_position"]:
            alert = RiskAlert(
                timestamp=current_time,
                alert_type=AlertType.WARNING,
                risk_category="Concentration Risk",
                message="単一銘柄の比重が過度に高い",
                current_value=max_weight,
                threshold_value=self.risk_thresholds["max_single_position"],
                recommended_action="ポジションサイズの分散",
                urgency=6,
            )
            alerts.append(alert)

        # アラートを緊急度順にソート
        alerts.sort(key=lambda x: x.urgency, reverse=True)

        # 履歴に保存
        self.alert_history.extend(alerts)

        return alerts

    def _create_default_risk_profile(
        self, portfolio_weights: Dict[str, float]
    ) -> PortfolioRiskProfile:
        """デフォルトリスクプロファイル作成"""
        return PortfolioRiskProfile(
            timestamp=datetime.now(),
            overall_risk_level=RiskLevel.MODERATE,
            portfolio_volatility=0.15,
            max_drawdown=-0.08,
            sharpe_ratio=0.8,
            var_95_daily=-0.02,
            cvar_95_daily=-0.03,
            worst_case_loss=0.20,
            stress_var_95=0,
            stress_recovery_time=None,
            market_regime=MarketRegime.SIDEWAYS,
            correlation_level=0.6,
            rebalancing_needed=False,
            target_weights=portfolio_weights,
            rebalancing_urgency=0,
            active_alerts=[],
        )

    def generate_risk_management_report(
        self, risk_profile: PortfolioRiskProfile
    ) -> str:
        """リスク管理レポート生成"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("統合リスク管理レポート")
        report_lines.append("=" * 80)

        report_lines.append(
            f"評価日時: {risk_profile.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"リスク許容度設定: {self.risk_tolerance.upper()}")

        # 総合リスクレベル
        risk_level_descriptions = {
            RiskLevel.VERY_LOW: "非常に低い - 安全性重視",
            RiskLevel.LOW: "低い - 保守的",
            RiskLevel.MODERATE: "中程度 - バランス型",
            RiskLevel.HIGH: "高い - 積極的",
            RiskLevel.VERY_HIGH: "非常に高い - 要注意",
            RiskLevel.EXTREME: "極端 - 緊急対応必要",
        }

        report_lines.append("\n【総合リスクレベル】")
        report_lines.append(
            f"{risk_level_descriptions[risk_profile.overall_risk_level]}"
        )

        # 基本リスクメトリクス
        report_lines.append("\n【基本リスクメトリクス】")
        report_lines.append(
            f"ポートフォリオボラティリティ: {risk_profile.portfolio_volatility:.2%}"
        )
        report_lines.append(f"最大ドローダウン: {risk_profile.max_drawdown:.2%}")
        report_lines.append(f"シャープレシオ: {risk_profile.sharpe_ratio:.3f}")
        report_lines.append(f"VaR(95%, 日次): {risk_profile.var_95_daily:.2%}")
        report_lines.append(f"CVaR(95%, 日次): {risk_profile.cvar_95_daily:.2%}")

        # 市場環境
        report_lines.append("\n【市場環境】")
        report_lines.append(f"市場レジーム: {risk_profile.market_regime.value}")
        report_lines.append(
            f"ポートフォリオ内相関: {risk_profile.correlation_level:.2%}"
        )

        # ストレステスト結果
        report_lines.append("\n【ストレステスト結果】")
        report_lines.append(f"最悪ケース損失: {risk_profile.worst_case_loss:.2%}")
        if risk_profile.stress_recovery_time:
            report_lines.append(f"推定回復時間: {risk_profile.stress_recovery_time}日")

        # リバランシング推奨
        if risk_profile.rebalancing_needed:
            report_lines.append("\n【リバランシング推奨】")
            report_lines.append(f"緊急度: {risk_profile.rebalancing_urgency:.1%}")
            report_lines.append("目標ウェイト:")
            for symbol, weight in risk_profile.target_weights.items():
                report_lines.append(f"  {symbol}: {weight:.1%}")

        # アクティブアラート
        if risk_profile.active_alerts:
            report_lines.append("\n【アクティブアラート】")
            for alert in risk_profile.active_alerts:
                alert_symbols = {
                    AlertType.INFO: "[INFO]",
                    AlertType.WARNING: "[WARNING]",
                    AlertType.CRITICAL: "[CRITICAL]",
                    AlertType.EMERGENCY: "[EMERGENCY]",
                }
                symbol = alert_symbols.get(alert.alert_type, "[UNKNOWN]")
                report_lines.append(f"{symbol} {alert.risk_category}")
                report_lines.append(f"  {alert.message}")
                report_lines.append(
                    f"  現在値: {alert.current_value:.2%}, 閾値: {alert.threshold_value:.2%}"
                )
                report_lines.append(f"  推奨対応: {alert.recommended_action}")
                report_lines.append("")

        # 総合評価・推奨事項
        report_lines.append("\n【総合評価・推奨事項】")

        if risk_profile.overall_risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            recommendation = (
                "現在のポートフォリオは低リスクです。リターン向上の余地があります。"
            )
        elif risk_profile.overall_risk_level == RiskLevel.MODERATE:
            recommendation = (
                "バランスの取れたポートフォリオです。定期的な見直しを継続してください。"
            )
        elif risk_profile.overall_risk_level == RiskLevel.HIGH:
            recommendation = "リスクが高めです。分散投資の強化を検討してください。"
        else:
            recommendation = "高リスク状態です。早急なリスク軽減策が必要です。"

        report_lines.append(recommendation)

        return "\n".join(report_lines)

    def get_risk_summary_dashboard(self) -> Dict[str, Any]:
        """リスクサマリーダッシュボード用データ"""
        if not self.risk_history:
            return {"error": "リスク履歴データがありません"}

        latest_profile = self.risk_history[-1]

        return {
            "overall_risk_level": latest_profile.overall_risk_level.value,
            "key_metrics": {
                "portfolio_volatility": latest_profile.portfolio_volatility,
                "max_drawdown": latest_profile.max_drawdown,
                "sharpe_ratio": latest_profile.sharpe_ratio,
                "var_95": latest_profile.var_95_daily,
                "worst_case_loss": latest_profile.worst_case_loss,
            },
            "market_environment": {
                "regime": latest_profile.market_regime.value,
                "correlation": latest_profile.correlation_level,
            },
            "alerts_count": len(latest_profile.active_alerts),
            "critical_alerts": len(
                [
                    a
                    for a in latest_profile.active_alerts
                    if a.alert_type in [AlertType.CRITICAL, AlertType.EMERGENCY]
                ]
            ),
            "rebalancing_needed": latest_profile.rebalancing_needed,
            "last_updated": latest_profile.timestamp.isoformat(),
        }


if __name__ == "__main__":
    # テスト実行
    print("統合リスク管理システムテスト")
    print("=" * 50)

    try:
        # システム初期化
        risk_system = IntegratedRiskManagementSystem(risk_tolerance="moderate")

        # サンプルポートフォリオ
        portfolio_weights = {
            "7203.T": 0.25,  # トヨタ
            "8306.T": 0.25,  # MUFG
            "9984.T": 0.30,  # ソフトバンクG
            "6758.T": 0.20,  # ソニー
        }

        # 実データ取得
        import yfinance as yf

        print("実データ取得中...")
        price_data = {}

        for symbol in portfolio_weights:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                if not data.empty:
                    price_data[symbol] = data
                    print(f"取得成功: {symbol}")
            except Exception as e:
                print(f"データ取得エラー {symbol}: {e}")

        if price_data:
            # 総合リスク評価実行
            print("\n総合リスク評価実行中...")
            risk_profile = risk_system.assess_portfolio_risk(
                portfolio_weights, price_data
            )

            # レポート生成・表示
            report = risk_system.generate_risk_management_report(risk_profile)
            print(report)

            # ダッシュボード用データ
            dashboard_data = risk_system.get_risk_summary_dashboard()
            print("\n=== ダッシュボードサマリー ===")
            print(f"総合リスクレベル: {dashboard_data['overall_risk_level']}")
            print(f"アラート数: {dashboard_data['alerts_count']}")
            print(f"緊急アラート: {dashboard_data['critical_alerts']}")
            print(
                f"リバランシング必要: {'はい' if dashboard_data['rebalancing_needed'] else 'いいえ'}"
            )

            print("\n統合リスク管理システムテスト完了")

        else:
            print("価格データの取得に失敗しました")

    except Exception as e:
        print(f"テストエラー: {e}")
        import traceback

        traceback.print_exc()
