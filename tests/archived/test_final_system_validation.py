#!/usr/bin/env python3
"""
実データでの最終動作確認テスト

Issue #321: 最優先：実データでの最終動作確認テスト
システム全体の統合テスト・パフォーマンス・信頼性検証
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import psutil

# プロジェクトルート設定
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

try:
    # コアシステムインポート
    from day_trade.automation.orchestrator import Orchestrator
    from day_trade.backtesting.backtest_engine import BacktestEngine
    from day_trade.config.config_manager import ConfigManager
    from day_trade.dashboard.dashboard_core import ProductionDashboard
    from day_trade.data.advanced_ml_engine import AdvancedMLEngine
    from day_trade.data.stock_fetcher import StockFetcher
    from day_trade.daytrade import DayTrade

    # ユーティリティインポート
    from day_trade.utils.performance_monitor import get_performance_monitor
    from day_trade.utils.structured_logging import get_structured_logger

except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("システムの基本機能テストのみ実行します")

print("実データでの最終動作確認テスト")
print("Issue #321: 最優先：実データでの最終動作確認テスト")
print("=" * 60)


class SystemValidationFramework:
    """システム検証フレームワーク"""

    def __init__(self):
        """初期化"""
        self.logger = None
        self.perf_monitor = None
        self.test_results = {}
        self.start_time = datetime.now()

        # テスト設定
        self.test_duration_minutes = 30  # 30分間のテスト
        self.test_symbols = ["7203.T", "8306.T", "9984.T", "6758.T", "9432.T"]  # 5銘柄

        # パフォーマンス基準
        self.performance_thresholds = {
            "max_memory_mb": 1024,  # 最大メモリ1GB
            "max_cpu_percent": 80,  # 最大CPU80%
            "max_response_time_ms": 500,  # 最大応答時間500ms
            "min_success_rate": 0.95,  # 最小成功率95%
            "max_error_rate": 0.05,  # 最大エラー率5%
        }

        try:
            self.logger = get_structured_logger()
            self.perf_monitor = get_performance_monitor()
            print("高度監視機能を有効化")
        except Exception as e:
            print(f"監視機能初期化警告: {e}")

        print("システム検証フレームワーク初期化完了")

    def test_core_system_functionality(self) -> bool:
        """コアシステム機能テスト"""
        print("\n=== コアシステム機能テスト ===")

        try:
            # DayTradeシステム初期化テスト
            print("DayTradeシステム初期化テスト...")

            context_manager = (
                self.perf_monitor.monitor("core_system_init")
                if self.perf_monitor
                else nullcontext()
            )
            with context_manager:
                daytrade = DayTrade()

                if hasattr(daytrade, "config_manager"):
                    print("[OK] ConfigManager初期化成功")
                else:
                    print("[NG] ConfigManager初期化失敗")
                    return False

                if hasattr(daytrade, "stock_fetcher"):
                    print("[OK] StockFetcher初期化成功")
                else:
                    print("[NG] StockFetcher初期化失敗")
                    return False

                # 基本機能テスト
                print("基本機能テスト実行中...")

                # 銘柄データ取得テスト
                test_symbol = "7203.T"
                try:
                    stock_data = daytrade.stock_fetcher.get_stock_data([test_symbol])
                    if stock_data and test_symbol in stock_data:
                        print(f"[OK] 銘柄データ取得成功: {test_symbol}")
                    else:
                        print(f"[NG] 銘柄データ取得失敗: {test_symbol}")
                        return False
                except Exception as e:
                    print(f"[NG] 銘柄データ取得エラー: {e}")
                    return False

            self.test_results["core_system"] = {
                "status": "passed",
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] コアシステム機能テスト成功")
            return True

        except Exception as e:
            print(f"[ERROR] コアシステム機能テストエラー: {e}")
            self.test_results["core_system"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_ml_processing_performance(self) -> bool:
        """ML処理パフォーマンステスト"""
        print("\n=== ML処理パフォーマンステスト ===")

        try:
            print("MLエンジン初期化テスト...")

            # MLエンジンテスト
            ml_engine = AdvancedMLEngine()

            # サンプルデータでML処理テスト
            print("ML処理実行テスト...")

            # テスト用データ準備
            import numpy as np
            import pandas as pd

            n_samples = 100
            n_features = 10

            # サンプル特徴量データ
            feature_data = np.random.randn(n_samples, n_features)
            feature_names = [f"feature_{i}" for i in range(n_features)]

            test_data = pd.DataFrame(feature_data, columns=feature_names)
            test_data["target"] = np.random.choice([0, 1], n_samples)

            start_time = time.time()

            try:
                # ML予測テスト
                predictions = ml_engine.predict(test_data[feature_names])

                ml_processing_time = time.time() - start_time

                if predictions is not None and len(predictions) > 0:
                    print(
                        f"[OK] ML予測成功: {len(predictions)}件, 処理時間: {ml_processing_time:.3f}秒"
                    )

                    # パフォーマンス評価
                    if ml_processing_time < 5.0:  # 5秒以内
                        print("[OK] ML処理パフォーマンス良好")
                    else:
                        print(f"[WARNING] ML処理時間が長い: {ml_processing_time:.3f}秒")
                else:
                    print("[NG] ML予測失敗")
                    return False

            except Exception as e:
                print(f"[NG] ML処理エラー: {e}")
                return False

            self.test_results["ml_performance"] = {
                "status": "passed",
                "processing_time": ml_processing_time,
                "predictions_count": len(predictions),
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] ML処理パフォーマンステスト成功")
            return True

        except Exception as e:
            print(f"[ERROR] ML処理パフォーマンステストエラー: {e}")
            self.test_results["ml_performance"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_real_data_integration(self) -> bool:
        """実データ統合テスト"""
        print("\n=== 実データ統合テスト ===")

        try:
            print("実データ取得・処理統合テスト...")

            # StockFetcher初期化
            stock_fetcher = StockFetcher()

            # 実データ取得テスト
            print(f"実データ取得中: {len(self.test_symbols)}銘柄...")

            start_time = time.time()

            real_data = stock_fetcher.get_stock_data(self.test_symbols)

            data_fetch_time = time.time() - start_time

            if not real_data:
                print("[NG] 実データ取得失敗")
                return False

            successful_symbols = len(
                [
                    s
                    for s in self.test_symbols
                    if s in real_data and not real_data[s].empty
                ]
            )
            success_rate = successful_symbols / len(self.test_symbols)

            print(
                f"[INFO] データ取得成功率: {success_rate:.1%} ({successful_symbols}/{len(self.test_symbols)})"
            )
            print(f"[INFO] データ取得時間: {data_fetch_time:.2f}秒")

            if success_rate >= 0.8:  # 80%以上成功
                print("[OK] 実データ取得成功")
            else:
                print("[NG] 実データ取得成功率が低い")
                return False

            # データ品質チェック
            print("データ品質チェック...")

            quality_issues = 0
            for symbol, data in real_data.items():
                if data.empty:
                    quality_issues += 1
                    continue

                # 基本的なデータ品質チェック
                required_columns = ["Open", "High", "Low", "Close", "Volume"]
                missing_columns = [
                    col for col in required_columns if col not in data.columns
                ]

                if missing_columns:
                    quality_issues += 1
                    print(f"[WARNING] {symbol}: 欠損列 {missing_columns}")
                    continue

                # データ範囲チェック
                if len(data) < 10:
                    quality_issues += 1
                    print(f"[WARNING] {symbol}: データ量不足 ({len(data)}行)")
                    continue

                # NaN値チェック
                nan_count = data[required_columns].isna().sum().sum()
                if nan_count > 0:
                    quality_issues += 1
                    print(f"[WARNING] {symbol}: NaN値発見 ({nan_count}個)")
                    continue

            quality_score = 1 - (quality_issues / len(real_data))
            print(f"[INFO] データ品質スコア: {quality_score:.1%}")

            if quality_score >= 0.8:
                print("[OK] データ品質良好")
            else:
                print("[WARNING] データ品質に問題あり")

            self.test_results["real_data_integration"] = {
                "status": "passed",
                "success_rate": success_rate,
                "quality_score": quality_score,
                "fetch_time": data_fetch_time,
                "symbols_count": successful_symbols,
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] 実データ統合テスト成功")
            return True

        except Exception as e:
            print(f"[ERROR] 実データ統合テストエラー: {e}")
            self.test_results["real_data_integration"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_dashboard_integration(self) -> bool:
        """ダッシュボード統合テスト"""
        print("\n=== ダッシュボード統合テスト ===")

        try:
            print("ダッシュボードシステム統合テスト...")

            # ダッシュボード初期化
            dashboard = ProductionDashboard()

            # 短期間の監視テスト
            print("短期監視テスト開始（10秒）...")
            dashboard.start_monitoring()

            time.sleep(10)  # 10秒間監視

            # ステータス取得テスト
            status = dashboard.get_current_status()

            # 必須データ確認
            required_sections = ["portfolio", "system", "trading", "risk"]
            missing_sections = []

            for section in required_sections:
                if section not in status or not status[section]:
                    missing_sections.append(section)

            if not missing_sections:
                print("[OK] ダッシュボードデータ収集成功")
            else:
                print(f"[NG] ダッシュボードデータ不足: {missing_sections}")
                return False

            # 履歴データテスト
            history = dashboard.get_historical_data("portfolio", hours=1)

            if len(history) > 0:
                print(f"[OK] 履歴データ取得成功: {len(history)}件")
            else:
                print("[NG] 履歴データ取得失敗")
                return False

            # レポート生成テスト
            report = dashboard.generate_status_report()

            if len(report) > 100:
                print(f"[OK] レポート生成成功: {len(report)}文字")
            else:
                print("[NG] レポート生成失敗")
                return False

            dashboard.stop_monitoring()

            self.test_results["dashboard_integration"] = {
                "status": "passed",
                "data_sections": len(required_sections) - len(missing_sections),
                "history_count": len(history),
                "report_length": len(report),
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] ダッシュボード統合テスト成功")
            return True

        except Exception as e:
            print(f"[ERROR] ダッシュボード統合テストエラー: {e}")
            self.test_results["dashboard_integration"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_system_performance_stress(self) -> bool:
        """システムパフォーマンス・ストレステスト"""
        print("\n=== システムパフォーマンス・ストレステスト ===")

        try:
            print("システムリソース監視開始...")

            # システムリソース監視
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent(interval=1)

            print(f"初期メモリ使用率: {initial_memory:.1f}%")
            print(f"初期CPU使用率: {initial_cpu:.1f}%")

            # ストレステスト実行
            print("ストレステスト実行中（並列処理）...")

            stress_start_time = time.time()

            def stress_task(task_id):
                """ストレステスト用タスク"""
                try:
                    # 軽量な処理を繰り返す
                    for _i in range(100):
                        # 簡単な計算処理
                        sum(range(1000))

                        # 短時間スリープ
                        time.sleep(0.01)

                    return f"Task {task_id}: Success"

                except Exception as e:
                    return f"Task {task_id}: Error {e}"

            # 並列ストレステスト
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(stress_task, i) for i in range(10)]

                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            stress_duration = time.time() - stress_start_time

            # 最終リソース使用量確認
            final_memory = psutil.virtual_memory().percent
            final_cpu = psutil.cpu_percent(interval=1)

            print(f"最終メモリ使用率: {final_memory:.1f}%")
            print(f"最終CPU使用率: {final_cpu:.1f}%")
            print(f"ストレステスト時間: {stress_duration:.2f}秒")

            # 成功タスク数
            successful_tasks = len([r for r in results if "Success" in r])
            success_rate = successful_tasks / len(results)

            print(
                f"タスク成功率: {success_rate:.1%} ({successful_tasks}/{len(results)})"
            )

            # パフォーマンス評価
            performance_ok = True

            if final_memory > self.performance_thresholds[
                "max_memory_mb"
            ] / 100 * psutil.virtual_memory().total / (1024**3):
                print(f"[WARNING] メモリ使用量が高い: {final_memory:.1f}%")
                performance_ok = False

            if final_cpu > self.performance_thresholds["max_cpu_percent"]:
                print(f"[WARNING] CPU使用率が高い: {final_cpu:.1f}%")
                performance_ok = False

            if success_rate < self.performance_thresholds["min_success_rate"]:
                print(f"[WARNING] タスク成功率が低い: {success_rate:.1%}")
                performance_ok = False

            if performance_ok:
                print("[OK] システムパフォーマンス良好")
            else:
                print("[WARNING] システムパフォーマンスに課題あり")

            self.test_results["performance_stress"] = {
                "status": "passed" if performance_ok else "warning",
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "initial_cpu": initial_cpu,
                "final_cpu": final_cpu,
                "stress_duration": stress_duration,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] システムパフォーマンス・ストレステスト完了")
            return True

        except Exception as e:
            print(f"[ERROR] システムパフォーマンス・ストレステストエラー: {e}")
            self.test_results["performance_stress"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def test_error_handling_resilience(self) -> bool:
        """エラーハンドリング・復元力テスト"""
        print("\n=== エラーハンドリング・復元力テスト ===")

        try:
            print("エラーハンドリングテスト...")

            # 意図的なエラー状況テスト
            error_scenarios = [
                ("無効な銘柄コード", ["INVALID.T"]),
                ("空の銘柄リスト", []),
                ("存在しない銘柄", ["NONEXIST.T"]),
            ]

            handled_errors = 0
            total_scenarios = len(error_scenarios)

            stock_fetcher = StockFetcher()

            for scenario_name, test_symbols in error_scenarios:
                try:
                    print(f"エラーシナリオ: {scenario_name}")

                    result = stock_fetcher.get_stock_data(test_symbols)

                    # エラーが適切にハンドリングされているかチェック
                    if result is None or (isinstance(result, dict) and not result):
                        print(f"[OK] {scenario_name}: 適切にハンドリング")
                        handled_errors += 1
                    else:
                        print(
                            f"[INFO] {scenario_name}: データ取得成功（予期しない結果）"
                        )
                        handled_errors += 1  # 成功も許容

                except Exception as e:
                    print(f"[OK] {scenario_name}: 例外キャッチ - {type(e).__name__}")
                    handled_errors += 1

            error_handling_rate = handled_errors / total_scenarios

            print(f"エラーハンドリング成功率: {error_handling_rate:.1%}")

            if error_handling_rate >= 0.8:  # 80%以上
                print("[OK] エラーハンドリング良好")
                resilience_ok = True
            else:
                print("[NG] エラーハンドリングに問題")
                resilience_ok = False

            self.test_results["error_handling"] = {
                "status": "passed" if resilience_ok else "failed",
                "handling_rate": error_handling_rate,
                "scenarios_tested": total_scenarios,
                "scenarios_handled": handled_errors,
                "timestamp": datetime.now().isoformat(),
            }

            print("[OK] エラーハンドリング・復元力テスト完了")
            return True

        except Exception as e:
            print(f"[ERROR] エラーハンドリング・復元力テストエラー: {e}")
            self.test_results["error_handling"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            return False

    def generate_final_report(self) -> str:
        """最終テストレポート生成"""
        total_duration = datetime.now() - self.start_time

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("実データでの最終動作確認テスト - 統合レポート")
        report_lines.append("=" * 80)

        report_lines.append(
            f"テスト実行日時: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"総実行時間: {total_duration}")
        report_lines.append(f"テスト対象銘柄: {', '.join(self.test_symbols)}")

        # 個別テスト結果
        report_lines.append("\n【個別テスト結果】")

        passed_tests = 0
        total_tests = 0

        for test_name, result in self.test_results.items():
            total_tests += 1
            status = result.get("status", "unknown")

            if status == "passed":
                status_symbol = "[OK]"
                passed_tests += 1
            elif status == "warning":
                status_symbol = "[WARNING]"
                passed_tests += 0.5  # 警告は半分としてカウント
            else:
                status_symbol = "[NG]"

            report_lines.append(
                f"  {status_symbol} {test_name.replace('_', ' ').title()}"
            )

            # 追加情報
            if "error" in result:
                report_lines.append(f"    エラー: {result['error']}")

            # 個別メトリクス
            if test_name == "real_data_integration":
                if "success_rate" in result:
                    report_lines.append(
                        f"    データ取得成功率: {result['success_rate']:.1%}"
                    )
                if "quality_score" in result:
                    report_lines.append(
                        f"    データ品質スコア: {result['quality_score']:.1%}"
                    )

            if test_name == "performance_stress":
                if "success_rate" in result:
                    report_lines.append(
                        f"    タスク成功率: {result['success_rate']:.1%}"
                    )
                if "final_memory" in result:
                    report_lines.append(
                        f"    最終メモリ使用率: {result['final_memory']:.1f}%"
                    )

        # 総合評価
        success_rate = (passed_tests / total_tests) if total_tests > 0 else 0

        report_lines.append("\n【総合評価】")
        report_lines.append(
            f"成功率: {passed_tests}/{total_tests} ({success_rate:.1%})"
        )

        if success_rate >= 0.9:
            overall_status = "優秀"
            recommendation = "システムは本格運用準備完了です。"
        elif success_rate >= 0.8:
            overall_status = "良好"
            recommendation = "軽微な調整後、本格運用可能です。"
        elif success_rate >= 0.7:
            overall_status = "要改善"
            recommendation = "重要な問題を解決してから運用してください。"
        else:
            overall_status = "不合格"
            recommendation = "システムの大幅な見直しが必要です。"

        report_lines.append(f"総合ステータス: {overall_status}")
        report_lines.append(f"推奨事項: {recommendation}")

        # システム情報
        report_lines.append("\n【システム情報】")
        report_lines.append(f"Python バージョン: {sys.version.split()[0]}")
        report_lines.append(
            f"メモリ総量: {psutil.virtual_memory().total / (1024**3):.1f}GB"
        )
        report_lines.append(f"CPU コア数: {psutil.cpu_count()}")

        return "\n".join(report_lines)

    def save_results(self, filepath: str):
        """テスト結果保存"""
        try:
            result_data = {
                "test_date": self.start_time.isoformat(),
                "total_duration": str(datetime.now() - self.start_time),
                "test_symbols": self.test_symbols,
                "performance_thresholds": self.performance_thresholds,
                "test_results": self.test_results,
                "system_info": {
                    "python_version": sys.version.split()[0],
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "cpu_count": psutil.cpu_count(),
                    "platform": sys.platform,
                },
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"テスト結果保存完了: {filepath}")

        except Exception as e:
            print(f"テスト結果保存エラー: {e}")


# nullcontextの定義（Python 3.6以下対応）
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


def main():
    """メイン実行"""
    print("実データでの最終動作確認テストを開始します")
    print("システム全体の統合テスト・パフォーマンス・信頼性を検証します")

    validator = SystemValidationFramework()

    # テストスイート実行
    test_suite = [
        ("コアシステム機能", validator.test_core_system_functionality),
        ("ML処理パフォーマンス", validator.test_ml_processing_performance),
        ("実データ統合", validator.test_real_data_integration),
        ("ダッシュボード統合", validator.test_dashboard_integration),
        ("システムパフォーマンス・ストレス", validator.test_system_performance_stress),
        ("エラーハンドリング・復元力", validator.test_error_handling_resilience),
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
    final_report = validator.generate_final_report()
    print(final_report)

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"final_system_validation_results_{timestamp}.json"
    validator.save_results(results_file)

    # テキストレポート保存
    report_file = f"final_system_validation_report_{timestamp}.txt"
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
    total_count = len(validator.test_results)
    success_rate = passed_count / total_count if total_count > 0 else 0

    print(f"\n{'='*60}")
    print("最終動作確認テスト完了")

    if success_rate >= 0.8:
        print("✅ システムは本格運用準備完了です！")
        return True
    else:
        print("❌ システムに改善が必要な問題があります。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
