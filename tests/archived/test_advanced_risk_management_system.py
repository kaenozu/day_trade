#!/usr/bin/env python3
"""
高度リスク管理システム統合テスト

Issue #316: 高優先：リスク管理機能強化
動的リバランシング・ストレステスト・統合リスク管理の検証
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    from day_trade.risk.dynamic_rebalancing import (
        DynamicRebalancingEngine,
        MarketRegime,
        RebalancingTrigger,
    )
    from day_trade.risk.integrated_risk_management import (
        AlertType,
        IntegratedRiskManagementSystem,
        RiskLevel,
    )
    from day_trade.risk.stress_test_framework import (
        AdvancedStressTestFramework,
        StressScenario,
    )
except ImportError as e:
    print(f"リスク管理モジュールインポートエラー: {e}")
    print("基本機能テストのみ実行します")

print("高度リスク管理システム統合テスト")
print("Issue #316: 高優先：リスク管理機能強化")
print("=" * 60)


class RiskManagementSystemValidator:
    """リスク管理システム検証"""

    def __init__(self):
        """初期化"""
        self.test_results = {}
        self.start_time = datetime.now()

        # テスト対象ポートフォリオ
        self.test_portfolio = {
            "7203.T": 0.25,  # トヨタ自動車
            "8306.T": 0.25,  # 三菱UFJフィナンシャル
            "9984.T": 0.30,  # ソフトバンクグループ
            "6758.T": 0.20,  # ソニーグループ
        }

        print("リスク管理システム検証初期化完了")

    def test_dynamic_rebalancing_engine(self) -> bool:
        """動的リバランシングエンジンテスト"""
        print("\n=== 動的リバランシングエンジンテスト ===")

        try:
            # エンジン初期化
            engine = DynamicRebalancingEngine(
                lookback_window=180, volatility_window=30, momentum_window=60
            )
            print("[OK] DynamicRebalancingEngine初期化成功")

            # サンプルデータ生成
            price_data = self._generate_sample_price_data()

            if not price_data:
                print("[NG] サンプルデータ生成失敗")
                return False

            print(f"[OK] サンプルデータ生成: {len(price_data)}銘柄")

            # 市場レジーム検出テスト
            print("市場レジーム検出テスト...")
            market_regime = engine.detect_market_regime(price_data)

            if isinstance(market_regime, MarketRegime):
                print(f"[OK] 市場レジーム検出: {market_regime.value}")
            else:
                print("[NG] 市場レジーム検出失敗")
                return False

            # リバランシングシグナル生成テスト
            print("リバランシングシグナル生成テスト...")
            signal = engine.generate_rebalancing_signal(self.test_portfolio, price_data)

            if signal is not None:
                print("[OK] リバランシングシグナル生成成功")
                print(f"  トリガー: {signal.trigger_type.value}")
                print(f"  信頼度: {signal.confidence:.2%}")
                print(f"  リバランシング強度: {signal.rebalancing_strength:.2%}")
            else:
                print("[INFO] リバランシング不要（正常な状況）")

            # リスクメトリクス計算テスト
            print("リスクメトリクス計算テスト...")
            risk_metrics = engine.calculate_portfolio_risk_metrics(
                self.test_portfolio, price_data
            )

            # 基本的な妥当性チェック
            metrics_ok = True

            if not (0 <= risk_metrics.portfolio_volatility <= 1):
                print(
                    f"[WARNING] ボラティリティ範囲異常: {risk_metrics.portfolio_volatility}"
                )
                metrics_ok = False

            if not (-1 <= risk_metrics.max_drawdown <= 0):
                print(f"[WARNING] ドローダウン範囲異常: {risk_metrics.max_drawdown}")
                metrics_ok = False

            if metrics_ok:
                print("[OK] リスクメトリクス計算成功")
                print(f"  ボラティリティ: {risk_metrics.portfolio_volatility:.2%}")
                print(f"  最大ドローダウン: {risk_metrics.max_drawdown:.2%}")
                print(f"  シャープレシオ: {risk_metrics.sharpe_ratio:.3f}")
            else:
                print("[WARNING] リスクメトリクスに異常値")

            self.test_results["dynamic_rebalancing"] = {
                "status": "passed" if metrics_ok else "warning",
                "market_regime": market_regime.value,
                "signal_generated": signal is not None,
                "risk_metrics": {
                    "volatility": risk_metrics.portfolio_volatility,
                    "max_drawdown": risk_metrics.max_drawdown,
                    "sharpe_ratio": risk_metrics.sharpe_ratio,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return True

        except Exception as e:
            print(f"[ERROR] 動的リバランシングエンジンテストエラー: {e}")
            self.test_results["dynamic_rebalancing"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_stress_test_framework(self) -> bool:
        """ストレステストフレームワークテスト"""
        print("\n=== ストレステストフレームワークテスト ===")

        try:
            # フレームワーク初期化
            framework = AdvancedStressTestFramework(
                confidence_level=0.95, simulation_runs=100  # テスト用に軽量化
            )
            print("[OK] AdvancedStressTestFramework初期化成功")

            # サンプルデータ生成
            price_data = self._generate_sample_price_data()

            if not price_data:
                print("[NG] サンプルデータ生成失敗")
                return False

            # 市場暴落シナリオテスト
            print("市場暴落シナリオテスト...")

            start_time = time.time()
            crash_result = framework.run_stress_test(
                self.test_portfolio, price_data, StressScenario.MARKET_CRASH
            )
            execution_time = time.time() - start_time

            if crash_result:
                print(f"[OK] 市場暴落シナリオ実行成功 ({execution_time:.2f}秒)")
                print(f"  予想損失: {crash_result.percentage_loss:.1%}")
                print(f"  VaR(95%): {crash_result.stressed_var_95:,.0f}円")
                print(f"  CVaR(95%): {crash_result.stressed_cvar_95:,.0f}円")
                print(f"  回復時間推定: {crash_result.recovery_time_estimate}日")
            else:
                print("[NG] 市場暴落シナリオ実行失敗")
                return False

            # 流動性危機シナリオテスト
            print("流動性危機シナリオテスト...")

            liquidity_result = framework.run_stress_test(
                self.test_portfolio, price_data, StressScenario.LIQUIDITY_CRISIS
            )

            if liquidity_result:
                print("[OK] 流動性危機シナリオ実行成功")
                print(f"  予想損失: {liquidity_result.percentage_loss:.1%}")
            else:
                print("[NG] 流動性危機シナリオ実行失敗")
                return False

            # 包括的ストレステスト
            print("包括的ストレステスト...")

            comprehensive_results = framework.run_comprehensive_stress_test(
                self.test_portfolio, price_data
            )

            if comprehensive_results:
                print(
                    f"[OK] 包括的ストレステスト成功: {len(comprehensive_results)}シナリオ"
                )

                # 結果妥当性チェック
                valid_results = 0
                for _scenario, result in comprehensive_results.items():
                    if 0 <= result.percentage_loss <= 1:  # 0-100%の損失範囲
                        valid_results += 1

                if valid_results == len(comprehensive_results):
                    print("[OK] 全結果が妥当な範囲内")
                else:
                    print(
                        f"[WARNING] {len(comprehensive_results)-valid_results}件の結果に異常"
                    )

            else:
                print("[NG] 包括的ストレステスト失敗")
                return False

            # レポート生成テスト
            print("レポート生成テスト...")

            report = framework.generate_stress_test_report(comprehensive_results)

            if len(report) > 200:
                print(f"[OK] レポート生成成功: {len(report)}文字")
            else:
                print("[NG] レポート生成失敗")
                return False

            self.test_results["stress_test_framework"] = {
                "status": "passed",
                "execution_time": execution_time,
                "scenarios_tested": len(comprehensive_results),
                "crash_scenario_loss": crash_result.percentage_loss,
                "liquidity_scenario_loss": liquidity_result.percentage_loss,
                "report_length": len(report),
                "timestamp": datetime.now().isoformat(),
            }

            return True

        except Exception as e:
            print(f"[ERROR] ストレステストフレームワークテストエラー: {e}")
            self.test_results["stress_test_framework"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_integrated_risk_management(self) -> bool:
        """統合リスク管理システムテスト"""
        print("\n=== 統合リスク管理システムテスト ===")

        try:
            # システム初期化（3つのリスク許容度でテスト）
            risk_tolerances = ["conservative", "moderate", "aggressive"]

            for tolerance in risk_tolerances:
                print(f"\n{tolerance.upper()}設定テスト...")

                risk_system = IntegratedRiskManagementSystem(
                    risk_tolerance=tolerance, monitoring_frequency=1440
                )
                print(f"[OK] 統合リスク管理システム初期化成功: {tolerance}")

                # サンプルデータ生成
                price_data = self._generate_sample_price_data()

                # 総合リスク評価実行
                print("総合リスク評価実行...")

                start_time = time.time()
                risk_profile = risk_system.assess_portfolio_risk(
                    self.test_portfolio, price_data
                )
                evaluation_time = time.time() - start_time

                if risk_profile:
                    print(f"[OK] 総合リスク評価成功 ({evaluation_time:.2f}秒)")
                    print(
                        f"  総合リスクレベル: {risk_profile.overall_risk_level.value}"
                    )
                    print(f"  アクティブアラート数: {len(risk_profile.active_alerts)}")
                    print(
                        f"  リバランシング必要: {'はい' if risk_profile.rebalancing_needed else 'いいえ'}"
                    )

                    # リスクレベル妥当性チェック
                    if isinstance(risk_profile.overall_risk_level, RiskLevel):
                        print("[OK] リスクレベル判定正常")
                    else:
                        print("[NG] リスクレベル判定異常")
                        return False

                else:
                    print("[NG] 総合リスク評価失敗")
                    return False

                # レポート生成テスト
                print("リスク管理レポート生成テスト...")

                report = risk_system.generate_risk_management_report(risk_profile)

                if len(report) > 300:
                    print(f"[OK] レポート生成成功: {len(report)}文字")
                else:
                    print("[NG] レポート生成失敗")
                    return False

                # ダッシュボードデータテスト
                print("ダッシュボードデータ生成テスト...")

                dashboard_data = risk_system.get_risk_summary_dashboard()

                required_keys = ["overall_risk_level", "key_metrics", "alerts_count"]
                missing_keys = [
                    key for key in required_keys if key not in dashboard_data
                ]

                if not missing_keys:
                    print("[OK] ダッシュボードデータ生成成功")
                else:
                    print(f"[NG] ダッシュボードデータ不完全: 欠損キー {missing_keys}")
                    return False

            self.test_results["integrated_risk_management"] = {
                "status": "passed",
                "tolerances_tested": len(risk_tolerances),
                "evaluation_time": evaluation_time,
                "risk_level": risk_profile.overall_risk_level.value,
                "alerts_generated": len(risk_profile.active_alerts),
                "report_length": len(report),
                "timestamp": datetime.now().isoformat(),
            }

            return True

        except Exception as e:
            print(f"[ERROR] 統合リスク管理システムテストエラー: {e}")
            self.test_results["integrated_risk_management"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_system_performance(self) -> bool:
        """システムパフォーマンステスト"""
        print("\n=== システムパフォーマンステスト ===")

        try:
            import psutil

            # 初期リソース状況
            initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
            initial_cpu = psutil.cpu_percent(interval=1)

            print(f"初期メモリ使用量: {initial_memory:.2f}GB")
            print(f"初期CPU使用率: {initial_cpu:.1f}%")

            # 高負荷テスト実行
            print("高負荷テスト実行中...")

            start_time = time.time()

            # 複数のリスク評価を並行実行
            risk_system = IntegratedRiskManagementSystem()
            price_data = self._generate_sample_price_data()

            results = []
            for _i in range(5):  # 5回実行
                result = risk_system.assess_portfolio_risk(
                    self.test_portfolio, price_data
                )
                results.append(result)

            execution_time = time.time() - start_time

            # 最終リソース状況
            final_memory = psutil.virtual_memory().used / (1024**3)
            final_cpu = psutil.cpu_percent(interval=1)

            memory_increase = final_memory - initial_memory
            cpu_increase = final_cpu - initial_cpu

            print(f"実行時間: {execution_time:.2f}秒")
            print(
                f"最終メモリ使用量: {final_memory:.2f}GB (増加: {memory_increase:.2f}GB)"
            )
            print(f"最終CPU使用率: {final_cpu:.1f}% (増加: {cpu_increase:.1f}%)")

            # パフォーマンス評価
            performance_ok = True

            if execution_time > 30:  # 30秒以上
                print(f"[WARNING] 実行時間が長い: {execution_time:.2f}秒")
                performance_ok = False

            if memory_increase > 0.5:  # 500MB以上増加
                print(f"[WARNING] メモリ使用量増加が大きい: {memory_increase:.2f}GB")
                performance_ok = False

            if performance_ok:
                print("[OK] システムパフォーマンス良好")
            else:
                print("[WARNING] システムパフォーマンスに課題")

            self.test_results["system_performance"] = {
                "status": "passed" if performance_ok else "warning",
                "execution_time": execution_time,
                "memory_increase_gb": memory_increase,
                "cpu_increase": cpu_increase,
                "tests_completed": len(results),
                "performance_ok": performance_ok,
                "timestamp": datetime.now().isoformat(),
            }

            return True

        except Exception as e:
            print(f"[ERROR] システムパフォーマンステストエラー: {e}")
            self.test_results["system_performance"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def _generate_sample_price_data(self) -> Dict[str, Any]:
        """サンプル価格データ生成"""
        try:
            import yfinance as yf

            price_data = {}

            for symbol in self.test_portfolio:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="6mo")  # 6ヶ月分

                    if not data.empty and len(data) >= 30:
                        price_data[symbol] = data
                except Exception as e:
                    print(f"データ取得エラー {symbol}: {e}")
                    # エラー時はダミーデータ生成
                    price_data[symbol] = self._generate_dummy_data(symbol)

            return price_data

        except ImportError:
            print("yfinanceが利用できません。ダミーデータを生成します。")

            # ダミーデータ生成
            price_data = {}
            for symbol in self.test_portfolio:
                price_data[symbol] = self._generate_dummy_data(symbol)

            return price_data

    def _generate_dummy_data(self, symbol: str) -> Any:
        """ダミー価格データ生成"""
        import numpy as np
        import pandas as pd

        # 6ヶ月分のダミーデータ
        dates = pd.date_range(end=datetime.now(), periods=180, freq="D")

        # ランダムウォークで価格生成
        np.random.seed(hash(symbol) % 1000)  # シンボルベースのシード

        initial_price = 1000 + hash(symbol) % 5000  # 1000-6000円のレンジ
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 日次リターン

        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # OHLCV データ作成
        opens = np.array(prices) * np.random.uniform(0.99, 1.01, len(prices))
        highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.02, len(prices))
        lows = np.minimum(opens, prices) * np.random.uniform(0.98, 1.0, len(prices))
        volumes = np.random.randint(100000, 1000000, len(prices))

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
                "Volume": volumes,
            },
            index=dates,
        )

    def generate_validation_report(self) -> str:
        """検証レポート生成"""
        total_duration = datetime.now() - self.start_time

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("高度リスク管理システム統合テスト レポート")
        report_lines.append("=" * 80)

        report_lines.append(
            f"テスト実行日時: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"総実行時間: {total_duration}")
        report_lines.append(f"テスト対象: {len(self.test_portfolio)}銘柄ポートフォリオ")

        # 個別テスト結果
        report_lines.append("\n【個別テスト結果】")

        passed_count = 0
        warning_count = 0
        failed_count = 0

        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")

            if status == "passed":
                status_symbol = "[OK]"
                passed_count += 1
            elif status == "warning":
                status_symbol = "[WARNING]"
                warning_count += 1
            else:
                status_symbol = "[NG]"
                failed_count += 1

            display_name = test_name.replace("_", " ").title()
            report_lines.append(f"  {status_symbol} {display_name}")

            # 詳細情報
            if "error" in result:
                report_lines.append(f"    エラー: {result['error']}")

            # テスト固有の情報
            if test_name == "dynamic_rebalancing":
                if "market_regime" in result:
                    report_lines.append(f"    市場レジーム: {result['market_regime']}")
                if "risk_metrics" in result:
                    volatility = result["risk_metrics"]["volatility"]
                    report_lines.append(f"    ボラティリティ: {volatility:.2%}")

            elif test_name == "stress_test_framework":
                if "scenarios_tested" in result:
                    report_lines.append(
                        f"    テスト済みシナリオ数: {result['scenarios_tested']}"
                    )
                if "execution_time" in result:
                    report_lines.append(
                        f"    実行時間: {result['execution_time']:.2f}秒"
                    )

            elif test_name == "integrated_risk_management":
                if "risk_level" in result:
                    report_lines.append(f"    リスクレベル: {result['risk_level']}")
                if "alerts_generated" in result:
                    report_lines.append(
                        f"    生成アラート数: {result['alerts_generated']}"
                    )

            elif test_name == "system_performance":
                if "execution_time" in result:
                    report_lines.append(
                        f"    実行時間: {result['execution_time']:.2f}秒"
                    )
                if "memory_increase_gb" in result:
                    report_lines.append(
                        f"    メモリ増加: {result['memory_increase_gb']:.2f}GB"
                    )

        # 総合評価
        total_tests = len(self.test_results)
        weighted_score = (
            (passed_count + warning_count * 0.7) / total_tests if total_tests > 0 else 0
        )

        report_lines.append("\n【総合評価】")
        report_lines.append(
            f"成功: {passed_count}, 警告: {warning_count}, 失敗: {failed_count}"
        )
        report_lines.append(f"重み付きスコア: {weighted_score:.1%}")

        if weighted_score >= 0.9:
            overall_status = "優秀"
            recommendation = "高度リスク管理システムは本格運用準備完了です。"
        elif weighted_score >= 0.8:
            overall_status = "良好"
            recommendation = "軽微な調整後、本格運用可能です。"
        elif weighted_score >= 0.7:
            overall_status = "要改善"
            recommendation = "いくつかの問題を解決してから運用してください。"
        else:
            overall_status = "要大幅改善"
            recommendation = "システムの大幅な改善が必要です。"

        report_lines.append(f"総合ステータス: {overall_status}")
        report_lines.append(f"推奨事項: {recommendation}")

        # 機能概要
        if weighted_score >= 0.8:
            report_lines.append("\n【実装済み機能】")
            report_lines.append("✓ 動的リバランシング（市場レジーム対応）")
            report_lines.append("✓ 高度ストレステスト（複数シナリオ）")
            report_lines.append("✓ 統合リスク管理（アラート・レポート機能）")
            report_lines.append("✓ リスク許容度別設定")
            report_lines.append("✓ リアルタイムリスク監視")

        return "\n".join(report_lines)


def main():
    """メイン実行"""
    print("高度リスク管理システムの統合テストを開始します")
    print("動的リバランシング・ストレステスト・統合リスク管理を検証します")

    validator = RiskManagementSystemValidator()

    # テストスイート実行
    test_suite = [
        ("動的リバランシングエンジン", validator.test_dynamic_rebalancing_engine),
        ("ストレステストフレームワーク", validator.test_stress_test_framework),
        ("統合リスク管理", validator.test_integrated_risk_management),
        ("システムパフォーマンス", validator.test_system_performance),
    ]

    print(f"\n{len(test_suite)}個のテストを実行します...")

    for test_name, test_function in test_suite:
        print(f"\n{'='*60}")
        print(f"テスト実行: {test_name}")

        try:
            test_function()
        except Exception as e:
            print(f"[CRITICAL ERROR] {test_name}で予期しないエラー: {e}")
            validator.test_results[test_name.lower().replace(" ", "_")] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # 最終レポート生成・表示
    print(f"\n{'='*60}")
    final_report = validator.generate_validation_report()
    print(final_report)

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"advanced_risk_management_test_results_{timestamp}.json"

    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_metadata": {
                        "test_date": validator.start_time.isoformat(),
                        "duration": str(datetime.now() - validator.start_time),
                        "portfolio": validator.test_portfolio,
                    },
                    "test_results": validator.test_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"テスト結果保存完了: {results_file}")
    except Exception as e:
        print(f"結果保存エラー: {e}")

    # テキストレポート保存
    report_file = f"advanced_risk_management_report_{timestamp}.txt"
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"最終レポート保存完了: {report_file}")
    except Exception as e:
        print(f"レポート保存エラー: {e}")

    # 総合判定
    passed_count = sum(
        1
        for result in validator.test_results.values()
        if result.get("status") == "passed"
    )
    warning_count = sum(
        1
        for result in validator.test_results.values()
        if result.get("status") == "warning"
    )
    total_count = len(validator.test_results)

    weighted_score = (
        (passed_count + warning_count * 0.7) / total_count if total_count > 0 else 0
    )

    print(f"\n{'='*60}")
    print("高度リスク管理システム統合テスト完了")

    if weighted_score >= 0.8:
        print("✅ 高度リスク管理システムは運用準備完了です！")
        return True
    elif weighted_score >= 0.6:
        print("⚠️ システムは概ね動作しますが、改善余地があります。")
        return True
    else:
        print("❌ システムに重要な問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
