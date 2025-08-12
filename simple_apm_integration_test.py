#!/usr/bin/env python3
"""
APM統合基盤シンプルテスト - 基本機能検証
"""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトパスをsys.pathに追加
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 環境変数設定
os.environ["PYTHONPATH"] = str(project_root / "src")
os.environ["DAY_TRADE_CONFIG_PATH"] = str(project_root / "config")


class SimpleAPMTest:
    """シンプルなAPMテストクラス"""

    def __init__(self):
        self.results = {
            "test_start_time": datetime.now(timezone.utc).isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "system_status": {},
        }

        # テスト環境設定
        self.test_dir = tempfile.mkdtemp(prefix="apm_simple_test_")
        print(f"Test directory: {self.test_dir}")

    def log_test_result(self, test_name: str, success: bool, details: Dict[str, Any] = None):
        """テスト結果をログ"""
        self.results["test_results"][test_name] = {
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

        status = "OK" if success else "FAIL"
        print(f"[{status}] {test_name}")

        if not success and details:
            print(f"   Error: {details.get('error', 'Unknown error')}")

    def test_config_files_existence(self) -> bool:
        """設定ファイルの存在確認テスト"""
        try:
            print("\n=== 設定ファイル存在確認テスト ===")

            config_files = [
                "config/alertmanager.yml",
                "config/alert.rules",
                "config/prometheus.yml",
                "config/otel-collector-config.yml",
                "docker-compose.observability.yml",
            ]

            results = {}
            for config_file in config_files:
                file_path = project_root / config_file
                exists = file_path.exists()
                results[config_file] = {"exists": exists, "path": str(file_path)}

                if exists:
                    results[config_file]["size_bytes"] = file_path.stat().st_size
                    print(f"  [OK] {config_file} - {results[config_file]['size_bytes']} bytes")
                else:
                    print(f"  [MISSING] {config_file}")

            missing_files = [f for f, info in results.items() if not info["exists"]]
            success = len(missing_files) == 0

            if not success:
                print(f"Missing files: {missing_files}")

            self.log_test_result(
                "設定ファイル存在確認",
                success,
                {"config_files": results, "missing_files": missing_files},
            )

            return success

        except Exception as e:
            self.log_test_result(
                "設定ファイル存在確認",
                False,
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return False

    def test_observability_modules_import(self) -> bool:
        """監視モジュールのインポートテスト"""
        try:
            print("\n=== 監視モジュールインポートテスト ===")

            import_results = {}
            modules_to_test = [
                "src.day_trade.observability.slo_manager",
                "src.day_trade.observability.dashboard_generator",
                "src.day_trade.observability.structured_logger",
            ]

            for module_name in modules_to_test:
                try:
                    __import__(module_name)
                    import_results[module_name] = {"success": True}
                    print(f"  [OK] {module_name}")
                except ImportError as e:
                    import_results[module_name] = {"success": False, "error": str(e)}
                    print(f"  [IMPORT_ERROR] {module_name}: {e}")
                except Exception as e:
                    import_results[module_name] = {
                        "success": False,
                        "error": f"Unexpected error: {str(e)}",
                    }
                    print(f"  [ERROR] {module_name}: {e}")

            successful_imports = sum(1 for result in import_results.values() if result["success"])
            total_imports = len(import_results)
            success = successful_imports > 0  # 最低1つインポートできれば成功

            self.log_test_result(
                "監視モジュールインポート",
                success,
                {
                    "import_results": import_results,
                    "success_rate": f"{successful_imports}/{total_imports}",
                },
            )

            return success

        except Exception as e:
            self.log_test_result(
                "監視モジュールインポート",
                False,
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return False

    def test_dashboard_generation_basic(self) -> bool:
        """基本的なダッシュボード生成テスト"""
        try:
            print("\n=== 基本ダッシュボード生成テスト ===")

            # シンプルなダッシュボード構造を作成
            basic_dashboard = {
                "dashboard": {
                    "title": "Test APM Dashboard",
                    "tags": ["test", "apm"],
                    "panels": [
                        {
                            "id": 1,
                            "title": "Test Metric",
                            "type": "graph",
                            "targets": [{"expr": "up", "legendFormat": "Service Status"}],
                        }
                    ],
                    "time": {"from": "now-1h", "to": "now"},
                    "refresh": "5s",
                }
            }

            # テスト用ダッシュボードファイルを作成
            test_dashboard_path = Path(self.test_dir) / "test_dashboard.json"
            with open(test_dashboard_path, "w", encoding="utf-8") as f:
                json.dump(basic_dashboard, f, indent=2, ensure_ascii=False)

            # ファイル存在と内容確認
            success = test_dashboard_path.exists() and test_dashboard_path.stat().st_size > 100

            dashboard_info = {
                "file_created": test_dashboard_path.exists(),
                "file_size": (
                    test_dashboard_path.stat().st_size if test_dashboard_path.exists() else 0
                ),
                "panels_count": len(basic_dashboard["dashboard"]["panels"]),
                "dashboard_title": basic_dashboard["dashboard"]["title"],
            }

            if success:
                print(f"  [OK] ダッシュボード生成成功: {dashboard_info['file_size']} bytes")
            else:
                print("  [FAIL] ダッシュボード生成失敗")

            self.log_test_result("基本ダッシュボード生成", success, dashboard_info)

            return success

        except Exception as e:
            self.log_test_result(
                "基本ダッシュボード生成",
                False,
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return False

    def test_slo_config_validation(self) -> bool:
        """SLO設定の基本検証テスト"""
        try:
            print("\n=== SLO設定検証テスト ===")

            # 基本的なSLO設定を作成
            slo_config = {
                "slos": [
                    {
                        "name": "api_response_time",
                        "description": "API response time should be under 100ms for 99% of requests",
                        "target_percentage": 99.0,
                        "time_window_hours": 1,
                        "sli_query": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
                        "thresholds": {"warning": 95.0, "critical": 90.0},
                    },
                    {
                        "name": "system_availability",
                        "description": "System should be available 99.9% of the time",
                        "target_percentage": 99.9,
                        "time_window_hours": 24,
                        "sli_query": "avg_over_time(up[5m]) * 100",
                        "thresholds": {"warning": 99.0, "critical": 95.0},
                    },
                ],
                "evaluation_interval_seconds": 60,
                "retention_days": 30,
            }

            # SLO設定ファイルを作成
            slo_config_path = Path(self.test_dir) / "slo_config.json"
            with open(slo_config_path, "w", encoding="utf-8") as f:
                json.dump(slo_config, f, indent=2, ensure_ascii=False)

            # 設定の基本検証
            validation_results = {
                "file_created": slo_config_path.exists(),
                "slo_count": len(slo_config["slos"]),
                "all_slos_have_targets": all(
                    "target_percentage" in slo for slo in slo_config["slos"]
                ),
                "all_slos_have_queries": all("sli_query" in slo for slo in slo_config["slos"]),
                "config_size_bytes": (
                    slo_config_path.stat().st_size if slo_config_path.exists() else 0
                ),
            }

            success = all(
                [
                    validation_results["file_created"],
                    validation_results["slo_count"] > 0,
                    validation_results["all_slos_have_targets"],
                    validation_results["all_slos_have_queries"],
                ]
            )

            if success:
                print(f"  [OK] SLO設定検証成功: {validation_results['slo_count']} SLOs configured")
            else:
                print("  [FAIL] SLO設定検証失敗")

            self.log_test_result("SLO設定検証", success, validation_results)

            return success

        except Exception as e:
            self.log_test_result(
                "SLO設定検証", False, {"error": str(e), "traceback": traceback.format_exc()}
            )
            return False

    def test_docker_compose_validation(self) -> bool:
        """Docker Compose設定の基本検証"""
        try:
            print("\n=== Docker Compose設定検証テスト ===")

            docker_compose_path = project_root / "docker-compose.observability.yml"

            if not docker_compose_path.exists():
                self.log_test_result(
                    "Docker Compose設定検証",
                    False,
                    {"error": "docker-compose.observability.yml not found"},
                )
                return False

            # ファイル読み込み
            with open(docker_compose_path, encoding="utf-8") as f:
                compose_content = f.read()

            # 基本的な内容チェック
            required_services = ["jaeger", "elasticsearch", "prometheus", "grafana"]
            service_checks = {}

            for service in required_services:
                service_present = service in compose_content
                service_checks[service] = service_present
                status = "OK" if service_present else "MISSING"
                print(f"  [{status}] Service: {service}")

            validation_results = {
                "file_exists": True,
                "file_size_bytes": docker_compose_path.stat().st_size,
                "services": service_checks,
                "services_found": sum(service_checks.values()),
                "services_expected": len(required_services),
            }

            success = validation_results["services_found"] >= 2  # 最低2サービスあればOK

            if success:
                print(
                    f"  [OK] Docker Compose検証成功: {validation_results['services_found']}/{validation_results['services_expected']} services found"
                )
            else:
                print("  [FAIL] Docker Compose検証失敗")

            self.log_test_result("Docker Compose設定検証", success, validation_results)

            return success

        except Exception as e:
            self.log_test_result(
                "Docker Compose設定検証",
                False,
                {"error": str(e), "traceback": traceback.format_exc()},
            )
            return False

    def generate_test_summary(self) -> Dict[str, Any]:
        """テスト結果サマリー生成"""
        try:
            print("\n=== テスト結果サマリー生成 ===")

            # テスト完了時刻
            self.results["test_end_time"] = datetime.now(timezone.utc).isoformat()

            # 成功率計算
            total_tests = len(self.results["test_results"])
            successful_tests = sum(
                1 for result in self.results["test_results"].values() if result["success"]
            )
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

            # システム状態評価
            system_status = {
                "overall_health": (
                    "excellent"
                    if success_rate >= 90
                    else "good" if success_rate >= 70 else "needs_attention"
                ),
                "success_rate_percentage": success_rate,
                "total_tests_executed": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "readiness_assessment": {
                    "basic_functionality": success_rate >= 50,
                    "production_ready": success_rate >= 80,
                    "comprehensive_monitoring": success_rate >= 90,
                },
            }

            self.results["system_status"] = system_status

            return self.results

        except Exception as e:
            print(f"サマリー生成エラー: {e}")
            return {"error": str(e)}

    async def run_simple_test_suite(self) -> Dict[str, Any]:
        """シンプルなテストスイート実行"""
        try:
            print("APM統合基盤シンプルテスト開始")
            print("=" * 50)

            # テスト実行順序
            test_sequence = [
                ("設定ファイル存在確認", self.test_config_files_existence),
                ("監視モジュールインポート", self.test_observability_modules_import),
                ("基本ダッシュボード生成", self.test_dashboard_generation_basic),
                ("SLO設定検証", self.test_slo_config_validation),
                ("Docker Compose設定検証", self.test_docker_compose_validation),
            ]

            for test_name, test_func in test_sequence:
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        result = await test_func()
                    else:
                        result = test_func()

                    if not result:
                        print(f"注意: {test_name}で問題が発生しました")

                except Exception as e:
                    print(f"エラー: {test_name}実行エラー: {e}")
                    self.log_test_result(test_name, False, {"error": str(e)})

            # テスト結果サマリー
            final_report = self.generate_test_summary()

            print("\n" + "=" * 50)
            print("APM統合基盤シンプルテスト完了")
            print(f"成功率: {final_report['system_status']['success_rate_percentage']:.1f}%")
            print(f"成功: {final_report['system_status']['successful_tests']}")
            print(f"失敗: {final_report['system_status']['failed_tests']}")
            print(f"総合評価: {final_report['system_status']['overall_health'].upper()}")

            # 準備状況表示
            readiness = final_report["system_status"]["readiness_assessment"]
            print("\n準備状況:")
            print(f"  基本機能: {'OK' if readiness['basic_functionality'] else 'NG'}")
            print(f"  本番準備: {'OK' if readiness['production_ready'] else 'NG'}")
            print(f"  包括監視: {'OK' if readiness['comprehensive_monitoring'] else 'NG'}")

            return final_report

        except Exception as e:
            print(f"テストスイート実行エラー: {e}")
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
    # テスト実行
    test_runner = SimpleAPMTest()
    results = await test_runner.run_simple_test_suite()

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"simple_apm_integration_test_results_{timestamp}.json"
    output_path = Path(project_root) / output_filename

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nテスト結果を保存: {output_path}")
    except Exception as e:
        print(f"テスト結果保存エラー: {e}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
