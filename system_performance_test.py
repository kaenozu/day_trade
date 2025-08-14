#!/usr/bin/env python3
"""
Day Trade Personal - システム総合パフォーマンステスト

システムの安定性・高速化・メモリ使用量を詳細に分析
"""

import asyncio
import time
import psutil
import gc
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class SystemPerformanceTest:
    """システム総合パフォーマンステスト"""

    def __init__(self):
        self.results = {}
        self.process = psutil.Process()

    def start_monitoring(self):
        """監視開始"""
        tracemalloc.start()
        gc.collect()  # ガベージコレクション実行
        self.start_time = time.time()
        self.start_memory = self.process.memory_info()
        self.start_cpu_times = self.process.cpu_times()

    def stop_monitoring(self) -> Dict[str, Any]:
        """監視終了・結果取得"""
        end_time = time.time()
        end_memory = self.process.memory_info()
        end_cpu_times = self.process.cpu_times()

        # メモリ使用量詳細
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "execution_time": end_time - self.start_time,
            "memory_usage": {
                "rss_mb": end_memory.rss / 1024 / 1024,
                "vms_mb": end_memory.vms / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024,
                "current_mb": current / 1024 / 1024,
                "memory_delta_mb": (end_memory.rss - self.start_memory.rss) / 1024 / 1024
            },
            "cpu_usage": {
                "user_time": end_cpu_times.user - self.start_cpu_times.user,
                "system_time": end_cpu_times.system - self.start_cpu_times.system,
                "cpu_percent": self.process.cpu_percent()
            }
        }

    async def test_basic_functionality(self) -> Dict[str, Any]:
        """基本機能パフォーマンステスト"""
        print("[TEST] 基本機能パフォーマンステスト開始...")

        self.start_monitoring()

        try:
            # 個人版システムの基本機能テスト
            from daytrade import PersonalAnalysisEngine

            engine = PersonalAnalysisEngine()
            recommendations = await engine.get_personal_recommendations(limit=3)

            performance_data = self.stop_monitoring()

            result = {
                "test_name": "basic_functionality",
                "success": True,
                "recommendations_count": len(recommendations),
                "performance": performance_data
            }

            print(f"[OK] 基本機能テスト完了: {performance_data['execution_time']:.3f}秒")
            print(f"   メモリ使用量: {performance_data['memory_usage']['current_mb']:.1f}MB")

            return result

        except Exception as e:
            performance_data = self.stop_monitoring()
            print(f"[ERROR] 基本機能テストエラー: {e}")
            return {
                "test_name": "basic_functionality",
                "success": False,
                "error": str(e),
                "performance": performance_data
            }

    async def test_chart_generation_performance(self) -> Dict[str, Any]:
        """チャート生成パフォーマンステスト"""
        print("📊 チャート生成パフォーマンステスト開始...")

        self.start_monitoring()

        try:
            # チャート生成機能テスト
            from src.day_trade.visualization.personal_charts import PersonalChartGenerator
            from daytrade import PersonalAnalysisEngine

            # サンプルデータ生成
            engine = PersonalAnalysisEngine()
            recommendations = await engine.get_personal_recommendations(limit=5)

            # チャート生成
            chart_gen = PersonalChartGenerator()
            analysis_chart = chart_gen.generate_analysis_chart(recommendations)
            summary_chart = chart_gen.generate_simple_summary(recommendations)

            performance_data = self.stop_monitoring()

            result = {
                "test_name": "chart_generation",
                "success": True,
                "charts_generated": 2,
                "analysis_chart_path": analysis_chart,
                "summary_chart_path": summary_chart,
                "performance": performance_data
            }

            print(f"[OK] チャート生成テスト完了: {performance_data['execution_time']:.3f}秒")
            print(f"   メモリ使用量: {performance_data['memory_usage']['current_mb']:.1f}MB")

            return result

        except Exception as e:
            performance_data = self.stop_monitoring()
            print(f"[ERROR] チャート生成テストエラー: {e}")
            return {
                "test_name": "chart_generation",
                "success": False,
                "error": str(e),
                "performance": performance_data
            }

    async def test_multiple_symbol_analysis(self) -> Dict[str, Any]:
        """複数銘柄分析パフォーマンステスト（将来機能のベンチマーク）"""
        print("📈 複数銘柄分析パフォーマンステスト開始...")

        self.start_monitoring()

        try:
            from daytrade import PersonalAnalysisEngine

            engine = PersonalAnalysisEngine()

            # 段階的に銘柄数を増やしてテスト
            results = {}
            for count in [3, 5, 8, 10]:
                test_start = time.time()
                recommendations = await engine.get_personal_recommendations(limit=count)
                test_time = time.time() - test_start

                results[f"{count}_symbols"] = {
                    "time": test_time,
                    "count": len(recommendations)
                }

                print(f"   {count}銘柄分析: {test_time:.3f}秒")

            performance_data = self.stop_monitoring()

            result = {
                "test_name": "multiple_symbol_analysis",
                "success": True,
                "scalability_results": results,
                "performance": performance_data
            }

            print(f"[OK] 複数銘柄分析テスト完了: {performance_data['execution_time']:.3f}秒")

            return result

        except Exception as e:
            performance_data = self.stop_monitoring()
            print(f"[ERROR] 複数銘柄分析テストエラー: {e}")
            return {
                "test_name": "multiple_symbol_analysis",
                "success": False,
                "error": str(e),
                "performance": performance_data
            }

    async def test_memory_leak_detection(self) -> Dict[str, Any]:
        """メモリリーク検出テスト"""
        print("🔍 メモリリーク検出テスト開始...")

        self.start_monitoring()
        memory_samples = []

        try:
            from daytrade import PersonalAnalysisEngine

            engine = PersonalAnalysisEngine()

            # 100回連続実行してメモリリークチェック
            for i in range(100):
                recommendations = await engine.get_personal_recommendations(limit=3)

                # 10回ごとにメモリ使用量サンプリング
                if i % 10 == 0:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_samples.append({
                        "iteration": i,
                        "memory_mb": current_memory
                    })

                # ガベージコレクション（定期的）
                if i % 50 == 0:
                    gc.collect()

            performance_data = self.stop_monitoring()

            # メモリリーク判定
            memory_trend = memory_samples[-1]["memory_mb"] - memory_samples[0]["memory_mb"]
            has_memory_leak = memory_trend > 50  # 50MB以上の増加でリーク疑い

            result = {
                "test_name": "memory_leak_detection",
                "success": True,
                "iterations": 100,
                "memory_samples": memory_samples,
                "memory_trend_mb": memory_trend,
                "has_memory_leak": has_memory_leak,
                "performance": performance_data
            }

            if has_memory_leak:
                print(f"[WARN]  メモリリーク検出: {memory_trend:.1f}MB増加")
            else:
                print(f"[OK] メモリリーク無し: {memory_trend:.1f}MB変化")

            return result

        except Exception as e:
            performance_data = self.stop_monitoring()
            print(f"[ERROR] メモリリーク検出テストエラー: {e}")
            return {
                "test_name": "memory_leak_detection",
                "success": False,
                "error": str(e),
                "performance": performance_data
            }

    async def test_error_handling(self) -> Dict[str, Any]:
        """エラーハンドリングテスト"""
        print("🛡️  エラーハンドリングテスト開始...")

        error_scenarios = []

        try:
            # シナリオ1: 不正な銘柄コード
            self.start_monitoring()

            from daytrade import PersonalAnalysisEngine
            engine = PersonalAnalysisEngine()

            try:
                # 存在しない銘柄での分析（エラーが適切に処理されるか）
                recommendations = await engine.get_personal_recommendations(limit=0)
                error_scenarios.append({
                    "scenario": "invalid_symbol_count",
                    "handled_correctly": True,
                    "error": None
                })
            except Exception as e:
                error_scenarios.append({
                    "scenario": "invalid_symbol_count",
                    "handled_correctly": False,
                    "error": str(e)
                })

            # シナリオ2: メモリ不足シミュレーション
            try:
                # 大量のデータで処理限界をテスト
                recommendations = await engine.get_personal_recommendations(limit=100)
                error_scenarios.append({
                    "scenario": "large_data_processing",
                    "handled_correctly": True,
                    "error": None
                })
            except Exception as e:
                error_scenarios.append({
                    "scenario": "large_data_processing",
                    "handled_correctly": True if "limit" in str(e).lower() else False,
                    "error": str(e)
                })

            performance_data = self.stop_monitoring()

            result = {
                "test_name": "error_handling",
                "success": True,
                "error_scenarios": error_scenarios,
                "performance": performance_data
            }

            handled_count = sum(1 for s in error_scenarios if s["handled_correctly"])
            print(f"[OK] エラーハンドリングテスト完了: {handled_count}/{len(error_scenarios)}シナリオ適切処理")

            return result

        except Exception as e:
            print(f"[ERROR] エラーハンドリングテストエラー: {e}")
            return {
                "test_name": "error_handling",
                "success": False,
                "error": str(e),
                "performance": {}
            }

    def generate_performance_report(self, all_results: List[Dict[str, Any]]):
        """パフォーマンスレポート生成"""
        report = {
            "test_date": datetime.now().isoformat(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            "test_results": all_results,
            "summary": {
                "total_tests": len(all_results),
                "passed_tests": sum(1 for r in all_results if r.get("success", False)),
                "failed_tests": sum(1 for r in all_results if not r.get("success", False))
            }
        }

        # レポート保存
        report_path = f"performance_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[REPORT] パフォーマンスレポート保存: {report_path}")

        # サマリー表示
        print("\n" + "="*60)
        print("システム総合パフォーマンステスト結果")
        print("="*60)

        for result in all_results:
            status = "[OK] PASS" if result.get("success") else "[ERROR] FAIL"
            test_name = result.get("test_name", "Unknown")

            if result.get("success") and "performance" in result:
                perf = result["performance"]
                exec_time = perf.get("execution_time", 0)
                memory_mb = perf.get("memory_usage", {}).get("current_mb", 0)
                print(f"{status} {test_name}: {exec_time:.3f}s, {memory_mb:.1f}MB")
            else:
                print(f"{status} {test_name}: {result.get('error', 'Unknown error')}")

        print(f"\n総合結果: {report['summary']['passed_tests']}/{report['summary']['total_tests']} テスト成功")
        print("="*60)

async def main():
    """メインテスト実行"""
    print("[START] Day Trade Personal システム総合パフォーマンステスト開始")
    print("="*60)

    test_runner = SystemPerformanceTest()
    all_results = []

    # 各テスト実行
    test_methods = [
        test_runner.test_basic_functionality,
        test_runner.test_chart_generation_performance,
        test_runner.test_multiple_symbol_analysis,
        test_runner.test_memory_leak_detection,
        test_runner.test_error_handling
    ]

    for test_method in test_methods:
        try:
            result = await test_method()
            all_results.append(result)

            # テスト間の小休止
            await asyncio.sleep(1)

        except Exception as e:
            print(f"[ERROR] テスト実行エラー: {e}")
            all_results.append({
                "test_name": test_method.__name__,
                "success": False,
                "error": str(e)
            })

    # 最終レポート生成
    test_runner.generate_performance_report(all_results)

    print("\n[COMPLETE] システム総合パフォーマンステスト完了！")

if __name__ == "__main__":
    asyncio.run(main())