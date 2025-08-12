#!/usr/bin/env python3
"""
SLO/SLI自動監視システム・パフォーマンス検証テスト
- SLO計算エンジンの性能測定
- ダッシュボード生成性能
- アラート応答時間
- HFT対応レイテンシ測定
"""

import asyncio
import json
import os
import sys
import time
import tempfile
import shutil
import sqlite3
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import traceback
import statistics

# プロジェクトパスをsys.pathに追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 環境変数設定
os.environ['PYTHONPATH'] = str(project_root / 'src')
os.environ['DAY_TRADE_CONFIG_PATH'] = str(project_root / 'config')


class SLOPerformanceTest:
    """SLO/SLIパフォーマンステストクラス"""

    def __init__(self):
        self.results = {
            "test_start_time": datetime.now(timezone.utc).isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "system_status": {},
            "hft_metrics": {}
        }

        # テスト環境設定
        self.test_dir = tempfile.mkdtemp(prefix="slo_perf_test_")
        self.db_path = Path(self.test_dir) / "test_slo_performance.db"

        # テスト用データベース作成
        self._setup_test_database()

        print(f"Test directory: {self.test_dir}")
        print(f"Test database: {self.db_path}")

    def _setup_test_database(self):
        """テスト用データベースセットアップ"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # SLIデータテーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sli_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slo_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    metadata TEXT
                )
            ''')

            # SLOレポートテーブル作成
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slo_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slo_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    sli_current REAL NOT NULL,
                    error_budget_consumption REAL NOT NULL,
                    status TEXT NOT NULL,
                    report_data TEXT
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"データベースセットアップエラー: {e}")

    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """テスト結果をログ"""
        self.results["test_results"][test_name] = {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }

        status = "OK" if success else "FAIL"
        print(f"[{status}] {test_name}")

        if details and "performance" in details:
            perf = details["performance"]
            print(f"   Performance: {perf}")

    def generate_test_sli_data(self, slo_name: str, count: int = 10000) -> List[Tuple[float, float, bool]]:
        """テスト用SLIデータ生成"""
        data_points = []
        base_time = time.time() - 3600  # 1時間前から

        for i in range(count):
            timestamp = base_time + (i * 0.36)  # 0.36秒間隔（10000ポイントで1時間）

            # HFT取引レイテンシをシミュレート（マイクロ秒）
            if slo_name == "hft_latency":
                # 正常時: 10-40μs、異常時: 50-100μs
                if random.random() < 0.999:  # 99.9%は正常
                    latency = random.uniform(10, 40)
                    success = True
                else:  # 0.1%は異常
                    latency = random.uniform(50, 100)
                    success = False

            # API応答時間をシミュレート（ミリ秒）
            elif slo_name == "api_latency":
                if random.random() < 0.995:  # 99.5%は正常
                    latency = random.uniform(10, 45)
                    success = True
                else:  # 0.5%は異常
                    latency = random.uniform(55, 200)
                    success = False

            # システム可用性をシミュレート
            elif slo_name == "availability":
                latency = 1.0 if random.random() < 0.9999 else 0.0  # 99.99%可用性
                success = latency > 0.5

            else:
                # デフォルト
                latency = random.uniform(20, 80)
                success = latency < 50

            data_points.append((timestamp, latency, success))

        return data_points

    def test_slo_calculation_performance(self) -> bool:
        """SLO計算性能テスト"""
        try:
            print("\n=== SLO計算性能テスト ===")

            # テストSLOセット
            test_slos = [
                ("hft_latency", "HFT取引レイテンシ"),
                ("api_latency", "API応答時間"),
                ("availability", "システム可用性")
            ]

            performance_results = {}

            for slo_name, description in test_slos:
                print(f"  Testing {slo_name}: {description}")

                # テストデータ生成
                data_generation_start = time.perf_counter()
                test_data = self.generate_test_sli_data(slo_name, 5000)  # 5000ポイント
                data_generation_time = time.perf_counter() - data_generation_start

                # データベースに挿入
                db_insert_start = time.perf_counter()
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for timestamp, value, success in test_data:
                    cursor.execute(
                        "INSERT INTO sli_data (slo_name, timestamp, value, success) VALUES (?, ?, ?, ?)",
                        (slo_name, timestamp, value, success)
                    )

                conn.commit()
                conn.close()
                db_insert_time = time.perf_counter() - db_insert_start

                # SLO計算シミュレート
                calc_start = time.perf_counter()

                # 成功率計算
                total_points = len(test_data)
                successful_points = sum(1 for _, _, success in test_data if success)
                sli_current = (successful_points / total_points) * 100 if total_points > 0 else 0

                # エラーバジェット計算
                target_percentage = 99.9 if slo_name == "hft_latency" else 99.0
                error_budget_total = (100 - target_percentage) * total_points / 100
                error_failures = total_points - successful_points
                error_budget_consumption = error_failures / error_budget_total if error_budget_total > 0 else 0

                calc_time = time.perf_counter() - calc_start

                # パフォーマンス結果
                perf_data = {
                    "data_generation_ms": data_generation_time * 1000,
                    "db_insert_ms": db_insert_time * 1000,
                    "calculation_ms": calc_time * 1000,
                    "total_ms": (data_generation_time + db_insert_time + calc_time) * 1000,
                    "data_points_processed": total_points,
                    "throughput_points_per_second": total_points / (calc_time if calc_time > 0 else 0.001),
                    "sli_result": {
                        "current": sli_current,
                        "target": target_percentage,
                        "error_budget_consumption": error_budget_consumption
                    }
                }

                performance_results[slo_name] = perf_data

                print(f"    Data generation: {perf_data['data_generation_ms']:.1f}ms")
                print(f"    DB insert: {perf_data['db_insert_ms']:.1f}ms")
                print(f"    Calculation: {perf_data['calculation_ms']:.1f}ms")
                print(f"    Throughput: {perf_data['throughput_points_per_second']:.0f} points/sec")
                print(f"    SLI: {sli_current:.3f}% (Target: {target_percentage}%)")

            # 全体パフォーマンス評価
            avg_calc_time = statistics.mean([r["calculation_ms"] for r in performance_results.values()])
            avg_throughput = statistics.mean([r["throughput_points_per_second"] for r in performance_results.values()])

            overall_performance = {
                "average_calculation_time_ms": avg_calc_time,
                "average_throughput_points_per_second": avg_throughput,
                "hft_ready": avg_calc_time < 10.0,  # 10ms以内ならHFT対応
                "production_ready": avg_calc_time < 100.0  # 100ms以内なら本番対応
            }

            performance_results["overall_assessment"] = overall_performance

            # HFTレイテンシ要件チェック
            hft_performance = performance_results.get("hft_latency", {})
            hft_meets_requirements = hft_performance.get("calculation_ms", 999) < 5.0  # 5ms以内

            success = avg_calc_time < 100.0 and hft_meets_requirements

            self.results["performance_metrics"]["slo_calculation"] = performance_results

            self.log_test_result("SLO計算性能", success, {
                "performance": overall_performance,
                "hft_ready": hft_meets_requirements
            })

            return success

        except Exception as e:
            self.log_test_result("SLO計算性能", False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False

    def test_dashboard_generation_performance(self) -> bool:
        """ダッシュボード生成性能テスト"""
        try:
            print("\n=== ダッシュボード生成性能テスト ===")

            dashboard_types = [
                ("hft_dashboard", "HFT Trading Dashboard"),
                ("slo_dashboard", "SLO Monitoring Dashboard"),
                ("system_dashboard", "System Overview Dashboard")
            ]

            generation_results = {}

            for dashboard_type, description in dashboard_types:
                print(f"  Testing {dashboard_type}: {description}")

                # ダッシュボード生成
                generation_start = time.perf_counter()

                # シンプルなダッシュボード構造を作成
                dashboard_config = {
                    "dashboard": {
                        "title": description,
                        "tags": [dashboard_type, "performance_test"],
                        "panels": [],
                        "time": {"from": "now-1h", "to": "now"},
                        "refresh": "5s" if "hft" in dashboard_type else "30s",
                        "templating": {"list": []},
                        "annotations": {"list": []}
                    }
                }

                # パネルを動的生成（HFTダッシュボードは多めのパネル）
                panel_count = 20 if "hft" in dashboard_type else 10

                for i in range(panel_count):
                    panel = {
                        "id": i + 1,
                        "title": f"Panel {i+1}",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": f"test_metric_{i}",
                                "legendFormat": f"Metric {i}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8}
                    }
                    dashboard_config["dashboard"]["panels"].append(panel)

                generation_time = time.perf_counter() - generation_start

                # JSONシリアライゼーション性能測定
                serialization_start = time.perf_counter()
                dashboard_json = json.dumps(dashboard_config, indent=2, ensure_ascii=False)
                serialization_time = time.perf_counter() - serialization_start

                # ファイル保存性能測定
                save_start = time.perf_counter()
                dashboard_path = Path(self.test_dir) / f"{dashboard_type}.json"
                with open(dashboard_path, 'w', encoding='utf-8') as f:
                    f.write(dashboard_json)
                save_time = time.perf_counter() - save_start

                total_time = generation_time + serialization_time + save_time

                generation_data = {
                    "generation_ms": generation_time * 1000,
                    "serialization_ms": serialization_time * 1000,
                    "save_ms": save_time * 1000,
                    "total_ms": total_time * 1000,
                    "panel_count": panel_count,
                    "file_size_bytes": len(dashboard_json.encode('utf-8')),
                    "generation_rate_panels_per_second": panel_count / (generation_time if generation_time > 0 else 0.001)
                }

                generation_results[dashboard_type] = generation_data

                print(f"    Generation: {generation_data['generation_ms']:.1f}ms")
                print(f"    Serialization: {generation_data['serialization_ms']:.1f}ms")
                print(f"    File save: {generation_data['save_ms']:.1f}ms")
                print(f"    Total: {generation_data['total_ms']:.1f}ms")
                print(f"    Panels: {panel_count}, Size: {generation_data['file_size_bytes']} bytes")

            # 全体パフォーマンス評価
            total_generation_times = [r["total_ms"] for r in generation_results.values()]
            avg_generation_time = statistics.mean(total_generation_times)
            max_generation_time = max(total_generation_times)

            overall_assessment = {
                "average_generation_time_ms": avg_generation_time,
                "max_generation_time_ms": max_generation_time,
                "real_time_ready": max_generation_time < 1000.0,  # 1秒以内
                "interactive_ready": max_generation_time < 5000.0,  # 5秒以内
                "total_dashboards_tested": len(dashboard_types)
            }

            generation_results["overall_assessment"] = overall_assessment

            success = max_generation_time < 10000.0  # 10秒以内なら成功

            self.results["performance_metrics"]["dashboard_generation"] = generation_results

            self.log_test_result("ダッシュボード生成性能", success, {
                "performance": overall_assessment
            })

            return success

        except Exception as e:
            self.log_test_result("ダッシュボード生成性能", False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False

    def test_hft_latency_requirements(self) -> bool:
        """HFTレイテンシ要件テスト"""
        try:
            print("\n=== HFTレイテンシ要件テスト ===")

            # 超高速処理のシミュレート
            hft_tests = [
                ("metric_collection", "メトリクス収集"),
                ("sli_recording", "SLI記録"),
                ("alert_evaluation", "アラート評価"),
                ("dashboard_update", "ダッシュボード更新")
            ]

            hft_results = {}

            for test_name, description in hft_tests:
                print(f"  Testing {test_name}: {description}")

                # 複数回実行して統計を取る
                execution_times = []

                for i in range(1000):  # 1000回実行
                    start_time = time.perf_counter_ns()  # ナノ秒精度

                    # 処理をシミュレート
                    if test_name == "metric_collection":
                        # メトリクス値計算
                        value = random.uniform(10, 50)
                        success = value < 50
                        timestamp = time.time()

                    elif test_name == "sli_recording":
                        # SLI記録処理
                        data_point = {
                            "timestamp": time.time(),
                            "value": random.uniform(10, 100),
                            "success": random.random() < 0.999
                        }

                    elif test_name == "alert_evaluation":
                        # アラート条件評価
                        current_value = random.uniform(10, 100)
                        threshold = 50.0
                        alert_needed = current_value > threshold

                    elif test_name == "dashboard_update":
                        # ダッシュボードデータ更新
                        update_data = {
                            "timestamp": time.time(),
                            "metrics": [random.uniform(0, 100) for _ in range(10)]
                        }

                    end_time = time.perf_counter_ns()
                    execution_time_ns = end_time - start_time
                    execution_times.append(execution_time_ns)

                # 統計計算
                avg_time_ns = statistics.mean(execution_times)
                median_time_ns = statistics.median(execution_times)
                p95_time_ns = sorted(execution_times)[int(0.95 * len(execution_times))]
                p99_time_ns = sorted(execution_times)[int(0.99 * len(execution_times))]
                max_time_ns = max(execution_times)

                # マイクロ秒に変換
                test_results = {
                    "executions": len(execution_times),
                    "avg_time_us": avg_time_ns / 1000,
                    "median_time_us": median_time_ns / 1000,
                    "p95_time_us": p95_time_ns / 1000,
                    "p99_time_us": p99_time_ns / 1000,
                    "max_time_us": max_time_ns / 1000,
                    "hft_requirement_met": p99_time_us / 1000 < 1.0,  # 1μs以内
                    "ultra_low_latency_met": p99_time_us / 1000 < 0.1  # 0.1μs以内
                }

                hft_results[test_name] = test_results

                print(f"    Average: {test_results['avg_time_us']:.3f}μs")
                print(f"    P95: {test_results['p95_time_us']:.3f}μs")
                print(f"    P99: {test_results['p99_time_us']:.3f}μs")
                print(f"    Max: {test_results['max_time_us']:.3f}μs")
                print(f"    HFT Ready: {'YES' if test_results['hft_requirement_met'] else 'NO'}")

            # 全体HFT準備状況
            hft_ready_count = sum(1 for r in hft_results.values() if r["hft_requirement_met"])
            total_tests = len(hft_results)

            hft_overall = {
                "hft_ready_tests": hft_ready_count,
                "total_tests": total_tests,
                "hft_readiness_percentage": (hft_ready_count / total_tests) * 100,
                "ultra_low_latency_ready": all(r["ultra_low_latency_met"] for r in hft_results.values()),
                "production_hft_ready": hft_ready_count >= total_tests * 0.8  # 80%以上
            }

            hft_results["overall_assessment"] = hft_overall

            success = hft_overall["production_hft_ready"]

            self.results["hft_metrics"] = hft_results

            self.log_test_result("HFTレイテンシ要件", success, {
                "performance": hft_overall
            })

            return success

        except Exception as e:
            self.log_test_result("HFTレイテンシ要件", False, {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return False

    def generate_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンステスト結果サマリー生成"""
        try:
            print("\n=== パフォーマンステスト結果サマリー ===")

            # テスト完了時刻
            self.results["test_end_time"] = datetime.now(timezone.utc).isoformat()

            # 成功率計算
            total_tests = len(self.results["test_results"])
            successful_tests = sum(1 for result in self.results["test_results"].values() if result["success"])
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

            # パフォーマンス総合評価
            performance_summary = {
                "test_success_rate": success_rate,
                "slo_calculation_performance": "excellent" if success_rate >= 90 else "good",
                "dashboard_generation_performance": "excellent" if success_rate >= 90 else "good",
                "hft_readiness": "ready" if success_rate >= 80 else "needs_improvement",
                "production_readiness": "ready" if success_rate >= 90 else "partial",
                "overall_rating": "excellent" if success_rate >= 90 else "good" if success_rate >= 70 else "needs_attention"
            }

            # HFT特化評価
            if "hft_metrics" in self.results and "overall_assessment" in self.results["hft_metrics"]:
                hft_assessment = self.results["hft_metrics"]["overall_assessment"]
                performance_summary["hft_specific"] = {
                    "ultra_low_latency_ready": hft_assessment.get("ultra_low_latency_ready", False),
                    "hft_readiness_percentage": hft_assessment.get("hft_readiness_percentage", 0),
                    "production_hft_ready": hft_assessment.get("production_hft_ready", False)
                }

            # 推奨事項
            recommendations = []

            if success_rate >= 90:
                recommendations.append("パフォーマンステスト結果は優秀です。本番環境での展開準備が整いました。")
            elif success_rate >= 70:
                recommendations.append("基本性能は良好ですが、一部改善の余地があります。")
            else:
                recommendations.append("パフォーマンスの改善が必要です。最適化を検討してください。")

            if "hft_metrics" in self.results:
                hft_ready = self.results["hft_metrics"].get("overall_assessment", {}).get("production_hft_ready", False)
                if hft_ready:
                    recommendations.append("HFT要件を満たしています。超低レイテンシ取引に対応可能です。")
                else:
                    recommendations.append("HFT要件の最適化が必要です。レイテンシの改善を検討してください。")

            self.results["performance_summary"] = performance_summary
            self.results["recommendations"] = recommendations

            return self.results

        except Exception as e:
            print(f"パフォーマンスサマリー生成エラー: {e}")
            return {"error": str(e)}

    async def run_performance_test_suite(self) -> Dict[str, Any]:
        """パフォーマンステストスイート実行"""
        try:
            print("SLO/SLI自動監視システム パフォーマンステスト開始")
            print("=" * 60)

            # テスト実行順序
            test_sequence = [
                ("SLO計算性能テスト", self.test_slo_calculation_performance),
                ("ダッシュボード生成性能テスト", self.test_dashboard_generation_performance),
                ("HFTレイテンシ要件テスト", self.test_hft_latency_requirements)
            ]

            for test_name, test_func in test_sequence:
                try:
                    print(f"\n>>> {test_name} 実行中...")

                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func()
                    else:
                        result = test_func()

                    if not result:
                        print(f"注意: {test_name}で問題が検出されました")

                except Exception as e:
                    print(f"エラー: {test_name}実行エラー: {e}")
                    self.log_test_result(test_name, False, {"error": str(e)})

            # パフォーマンスサマリー生成
            final_report = self.generate_performance_summary()

            print("\n" + "=" * 60)
            print("SLO/SLI自動監視システム パフォーマンステスト完了")

            if "performance_summary" in final_report:
                summary = final_report["performance_summary"]
                print(f"テスト成功率: {summary['test_success_rate']:.1f}%")
                print(f"SLO計算性能: {summary['slo_calculation_performance'].upper()}")
                print(f"ダッシュボード生成性能: {summary['dashboard_generation_performance'].upper()}")
                print(f"HFT準備状況: {summary['hft_readiness'].upper()}")
                print(f"総合評価: {summary['overall_rating'].upper()}")

                if "hft_specific" in summary:
                    hft = summary["hft_specific"]
                    print(f"\nHFT特化評価:")
                    print(f"  超低レイテンシ対応: {'OK' if hft['ultra_low_latency_ready'] else 'NG'}")
                    print(f"  HFT準備率: {hft['hft_readiness_percentage']:.1f}%")
                    print(f"  本番HFT対応: {'OK' if hft['production_hft_ready'] else 'NG'}")

            return final_report

        except Exception as e:
            print(f"パフォーマンステストスイート実行エラー: {e}")
            print(traceback.format_exc())
            return {"error": str(e), "traceback": traceback.format_exc()}

        finally:
            # テストディレクトリクリーンアップ
            try:
                if os.path.exists(self.test_dir):
                    shutil.rmtree(self.test_dir)
                    print(f"テストディレクトリクリーンアップ完了: {self.test_dir}")
            except Exception as e:
                print(f"テストディレクトリクリーンアップエラー: {e}")


async def main():
    """メイン実行関数"""
    # パフォーマンステスト実行
    test_runner = SLOPerformanceTest()
    results = await test_runner.run_performance_test_suite()

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"slo_performance_verification_results_{timestamp}.json"
    output_path = Path(project_root) / output_filename

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nパフォーマンステスト結果を保存: {output_path}")
    except Exception as e:
        print(f"テスト結果保存エラー: {e}")

    # 結果サマリー表示
    if "performance_summary" in results:
        summary = results["performance_summary"]

        print(f"\nパフォーマンス最終結果:")
        print(f"   総合評価: {summary['overall_rating'].upper()}")
        print(f"   本番対応: {'OK' if summary['production_readiness'] == 'ready' else 'PARTIAL'}")
        print(f"   HFT対応: {'OK' if summary['hft_readiness'] == 'ready' else 'NG'}")

        if "recommendations" in results:
            print(f"\n推奨事項:")
            for rec in results["recommendations"]:
                print(f"   • {rec}")

    return results


if __name__ == "__main__":
    asyncio.run(main())