#!/usr/bin/env python3
"""
Issue #454統合テスト - DayTrade全自動システム

全機能の統合テストとパフォーマンス検証
"""

import asyncio
import time
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.day_trade.automation.auto_pipeline_manager import run_auto_pipeline
from src.day_trade.recommendation.recommendation_engine import get_daily_recommendations


class IntegrationTestSuite:
    """統合テストスイート"""

    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()

    def log_test_result(self, test_name: str, success: bool, duration: float, details: str = ""):
        """テスト結果記録"""
        self.test_results[test_name] = {
            'success': success,
            'duration': duration,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }

        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name} - {duration:.2f}秒")
        if details:
            print(f"    詳細: {details}")
        print()

    async def test_recommendation_engine_basic(self):
        """基本推奨エンジンテスト"""
        test_start = time.time()

        try:
            print("=== 推奨エンジン基本テスト ===")
            recommendations = await get_daily_recommendations(3)

            success = (
                isinstance(recommendations, list) and
                len(recommendations) <= 3 and
                all(hasattr(rec, 'symbol') for rec in recommendations) and
                all(hasattr(rec, 'composite_score') for rec in recommendations)
            )

            details = f"推奨銘柄数: {len(recommendations)}"
            if recommendations:
                details += f", トップスコア: {recommendations[0].composite_score:.1f}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("推奨エンジン基本機能", success, duration, details)
        return success

    async def test_auto_pipeline_quick(self):
        """自動パイプライン高速テスト"""
        test_start = time.time()

        try:
            print("=== 自動パイプライン高速テスト ===")
            test_symbols = ["7203", "8306", "9984"]  # 3銘柄のみ

            result = await run_auto_pipeline(test_symbols)

            success = (
                result is not None and
                hasattr(result, 'success') and
                result.success and
                len(result.data_collection.collected_symbols) > 0
            )

            details = f"処理銘柄: {len(result.data_collection.collected_symbols)}/{len(test_symbols)}"
            details += f", 品質スコア: {result.quality_report.overall_score:.2f}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("自動パイプライン高速", success, duration, details)
        return success

    def test_simple_interface_help(self):
        """シンプルインターフェースヘルプテスト"""
        test_start = time.time()

        try:
            print("=== シンプルインターフェースヘルプテスト ===")
            result = subprocess.run(
                [sys.executable, "daytrade_simple.py", "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = (
                result.returncode == 0 and
                "DayTrade" in result.stdout and
                "--quick" in result.stdout and
                "--full" in result.stdout
            )

            details = f"終了コード: {result.returncode}, 出力行数: {len(result.stdout.splitlines())}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("シンプルインターフェースヘルプ", success, duration, details)
        return success

    def test_simple_interface_version(self):
        """バージョン情報テスト"""
        test_start = time.time()

        try:
            print("=== バージョン情報テスト ===")
            result = subprocess.run(
                [sys.executable, "daytrade_simple.py", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = (
                result.returncode == 0 and
                "DayTrade Simple Interface" in result.stdout
            )

            details = f"バージョン: {result.stdout.strip()}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("バージョン情報表示", success, duration, details)
        return success

    async def test_performance_benchmark(self):
        """パフォーマンステスト"""
        test_start = time.time()

        try:
            print("=== パフォーマンスベンチマーク ===")

            # 推奨エンジンの速度テスト
            bench_start = time.time()
            recommendations = await get_daily_recommendations(5)
            recommendation_time = time.time() - bench_start

            # 目標: 5分以内（300秒）で完了
            target_time = 300
            success = (
                recommendation_time < target_time and
                len(recommendations) > 0
            )

            details = f"推奨生成時間: {recommendation_time:.2f}秒 (目標: {target_time}秒以内)"
            details += f", 生成数: {len(recommendations)}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("パフォーマンスベンチマーク", success, duration, details)
        return success

    def test_file_structure(self):
        """ファイル構造テスト"""
        test_start = time.time()

        try:
            print("=== ファイル構造テスト ===")

            required_files = [
                "daytrade_simple.py",
                "src/day_trade/automation/auto_pipeline_manager.py",
                "src/day_trade/recommendation/recommendation_engine.py",
                "config/settings.json"
            ]

            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)

            success = len(missing_files) == 0

            if success:
                details = f"全{len(required_files)}ファイルが存在"
            else:
                details = f"不足ファイル: {', '.join(missing_files)}"

        except Exception as e:
            success = False
            details = f"エラー: {e}"

        duration = time.time() - test_start
        self.log_test_result("ファイル構造確認", success, duration, details)
        return success

    def generate_test_report(self):
        """テストレポート生成"""
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        total_tests = len(self.test_results)

        print("\n" + "=" * 60)
        print("         統合テスト結果レポート")
        print("=" * 60)
        print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {total_time:.2f}秒")
        print(f"テスト結果: {passed_tests}/{total_tests} 合格 ({passed_tests/total_tests*100:.1f}%)")
        print()

        print("詳細結果:")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"  {status} {test_name} ({result['duration']:.2f}s)")
            if result['details']:
                print(f"      {result['details']}")

        print("\n" + "=" * 60)

        if passed_tests == total_tests:
            print("🎉 全テストが合格しました！Issue #454の統合テストは成功です。")
        else:
            print("⚠️ 一部のテストが失敗しました。修正が必要です。")

        print("=" * 60)

        return passed_tests == total_tests


async def run_integration_tests():
    """統合テスト実行"""
    print("=" * 60)
    print("    Issue #454 DayTrade全自動システム")
    print("         統合テスト開始")
    print("=" * 60)
    print()

    test_suite = IntegrationTestSuite()

    # テスト実行順序（重要度順）
    tests = [
        ("ファイル構造確認", test_suite.test_file_structure),
        ("バージョン情報", test_suite.test_simple_interface_version),
        ("ヘルプ表示", test_suite.test_simple_interface_help),
        ("推奨エンジン基本", test_suite.test_recommendation_engine_basic),
        ("自動パイプライン", test_suite.test_auto_pipeline_quick),
        ("パフォーマンス", test_suite.test_performance_benchmark),
    ]

    overall_success = True

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            if not success:
                overall_success = False

        except Exception as e:
            print(f"[ERROR] {test_name}でエラー: {e}")
            test_suite.log_test_result(test_name, False, 0, f"例外: {e}")
            overall_success = False

    # テストレポート生成
    final_success = test_suite.generate_test_report()

    return final_success and overall_success


if __name__ == "__main__":
    print("Issue #454統合テスト開始...")

    try:
        success = asyncio.run(run_integration_tests())
        exit_code = 0 if success else 1

        print(f"\n統合テスト{'成功' if success else '失敗'} (終了コード: {exit_code})")

    except KeyboardInterrupt:
        print("\n\nテストが中断されました。")
        exit_code = 130
    except Exception as e:
        print(f"\n予期しないエラー: {e}")
        exit_code = 1

    sys.exit(exit_code)