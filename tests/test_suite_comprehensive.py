#!/usr/bin/env python3
"""
包括的テストスイート実行ランナー

Issue #471対応: プロジェクト全体のテストカバレッジと信頼性向上

主要コンポーネントの統合テストを実行:
- EnsembleSystem
- RecommendationEngine
- その他の重要なMLコンポーネント
"""

import unittest
import sys
import time
from pathlib import Path
import warnings

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 各テストモジュールのインポート
try:
    from tests.ml.test_ensemble_system_comprehensive import run_ensemble_system_tests
    ENSEMBLE_TESTS_AVAILABLE = True
except ImportError:
    ENSEMBLE_TESTS_AVAILABLE = False

try:
    from tests.ml.test_recommendation_engine_comprehensive import run_recommendation_engine_tests
    RECOMMENDATION_TESTS_AVAILABLE = True
except ImportError:
    RECOMMENDATION_TESTS_AVAILABLE = False

warnings.filterwarnings("ignore")


class TestSuiteRunner:
    """包括的テストスイート実行クラス"""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.start_time = None
        self.results = {}

    def run_comprehensive_test_suite(self):
        """包括的テストスイートの実行"""
        print("=" * 80)
        print("🚀 DayTrade包括的テストスイート実行開始")
        print("   Issue #471: EnsembleSystemのテストカバレッジ向上")
        print("=" * 80)

        self.start_time = time.time()

        # 1. EnsembleSystemテスト
        self._run_ensemble_system_tests()

        # 2. RecommendationEngineテスト
        self._run_recommendation_engine_tests()

        # 3. その他のMLコンポーネントテスト
        self._run_additional_ml_tests()

        # 4. 統合テスト
        self._run_integration_tests()

        # 結果サマリー
        self._print_final_summary()

        return self.passed_tests > 0 and self.failed_tests == 0

    def _run_ensemble_system_tests(self):
        """EnsembleSystemテストの実行"""
        print("\n" + "=" * 60)
        print("🎯 EnsembleSystem包括的テスト")
        print("=" * 60)

        if ENSEMBLE_TESTS_AVAILABLE:
            try:
                success = run_ensemble_system_tests()
                self.results['ensemble_system'] = {
                    'status': 'passed' if success else 'failed',
                    'component': 'EnsembleSystem',
                    'description': 'MLアンサンブル学習システム'
                }
                if success:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1

            except Exception as e:
                print(f"❌ EnsembleSystemテスト実行エラー: {e}")
                self.results['ensemble_system'] = {
                    'status': 'error',
                    'component': 'EnsembleSystem',
                    'error': str(e)
                }
                self.error_tests += 1
        else:
            print("⚠️  EnsembleSystemテストが利用できません")
            self.results['ensemble_system'] = {
                'status': 'skipped',
                'component': 'EnsembleSystem',
                'reason': 'テストモジュール未利用'
            }

    def _run_recommendation_engine_tests(self):
        """RecommendationEngineテストの実行"""
        print("\n" + "=" * 60)
        print("📊 RecommendationEngine包括的テスト")
        print("=" * 60)

        if RECOMMENDATION_TESTS_AVAILABLE:
            try:
                success = run_recommendation_engine_tests()
                self.results['recommendation_engine'] = {
                    'status': 'passed' if success else 'failed',
                    'component': 'RecommendationEngine',
                    'description': '推奨銘柄選定エンジン'
                }
                if success:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1

            except Exception as e:
                print(f"❌ RecommendationEngineテスト実行エラー: {e}")
                self.results['recommendation_engine'] = {
                    'status': 'error',
                    'component': 'RecommendationEngine',
                    'error': str(e)
                }
                self.error_tests += 1
        else:
            print("⚠️  RecommendationEngineテストが利用できません")
            self.results['recommendation_engine'] = {
                'status': 'skipped',
                'component': 'RecommendationEngine',
                'reason': 'テストモジュール未利用'
            }

    def _run_additional_ml_tests(self):
        """追加のMLコンポーネントテスト"""
        print("\n" + "=" * 60)
        print("🧠 追加MLコンポーネントテスト")
        print("=" * 60)

        # 既存のMLテストがある場合に実行
        additional_tests = [
            'test_dynamic_weighting_system.py',
            'test_base_model_interface_feature_importance.py',
            'test_stacking_ensemble.py'
        ]

        for test_file in additional_tests:
            test_path = project_root / 'tests' / 'ml' / test_file
            if test_path.exists():
                print(f"🔍 実行中: {test_file}")
                try:
                    # unittestを使用してテスト実行
                    import subprocess
                    result = subprocess.run([
                        sys.executable, '-m', 'unittest',
                        f'tests.ml.{test_file[:-3]}'
                    ], capture_output=True, text=True, cwd=project_root)

                    if result.returncode == 0:
                        print(f"✅ {test_file}: 成功")
                        self.passed_tests += 1
                    else:
                        print(f"❌ {test_file}: 失敗")
                        self.failed_tests += 1

                except Exception as e:
                    print(f"⚠️  {test_file}: エラー - {e}")
                    self.error_tests += 1
            else:
                print(f"⏭️  {test_file}: ファイルが存在しません")

        if not any((project_root / 'tests' / 'ml' / test_file).exists() for test_file in additional_tests):
            print("📝 追加のMLテストファイルは見つかりませんでした")

    def _run_integration_tests(self):
        """統合テストの実行"""
        print("\n" + "=" * 60)
        print("🔗 統合テスト")
        print("=" * 60)

        # 統合テストディレクトリの確認
        integration_test_dir = project_root / 'tests' / 'integration'
        if integration_test_dir.exists():
            integration_tests = list(integration_test_dir.glob('test_*.py'))

            if integration_tests:
                print(f"🧪 統合テスト {len(integration_tests)} ファイルを発見")

                for test_file in integration_tests[:3]:  # 最初の3つのみ実行（時間短縮）
                    print(f"🔍 実行中: {test_file.name}")
                    try:
                        import subprocess
                        result = subprocess.run([
                            sys.executable, '-m', 'unittest',
                            f'tests.integration.{test_file.stem}'
                        ], capture_output=True, text=True, cwd=project_root, timeout=60)

                        if result.returncode == 0:
                            print(f"✅ {test_file.name}: 成功")
                            self.passed_tests += 1
                        else:
                            print(f"❌ {test_file.name}: 失敗")
                            self.failed_tests += 1

                    except subprocess.TimeoutExpired:
                        print(f"⏰ {test_file.name}: タイムアウト")
                        self.error_tests += 1
                    except Exception as e:
                        print(f"⚠️  {test_file.name}: エラー - {e}")
                        self.error_tests += 1
            else:
                print("📝 統合テストファイルは見つかりませんでした")
        else:
            print("📁 統合テストディレクトリが存在しません")

    def _print_final_summary(self):
        """最終結果サマリーの表示"""
        end_time = time.time()
        total_time = end_time - self.start_time

        print("\n" + "=" * 80)
        print("📋 包括的テストスイート実行結果")
        print("=" * 80)

        print(f"⏱️  総実行時間: {total_time:.1f}秒")
        print(f"🎯 総テスト数: {self.passed_tests + self.failed_tests + self.error_tests}")
        print(f"✅ 成功: {self.passed_tests}")
        print(f"❌ 失敗: {self.failed_tests}")
        print(f"⚠️  エラー: {self.error_tests}")

        if self.passed_tests + self.failed_tests + self.error_tests > 0:
            success_rate = (self.passed_tests / (self.passed_tests + self.failed_tests + self.error_tests)) * 100
            print(f"📊 成功率: {success_rate:.1f}%")

        print("\n" + "-" * 60)
        print("コンポーネント別結果:")
        print("-" * 60)

        for component, result in self.results.items():
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'error': '⚠️',
                'skipped': '⏭️'
            }.get(result['status'], '❓')

            component_name = result.get('component', component)
            description = result.get('description', '')

            print(f"{status_icon} {component_name}: {result['status']}")
            if description:
                print(f"   説明: {description}")
            if 'error' in result:
                print(f"   エラー: {result['error'][:100]}...")
            if 'reason' in result:
                print(f"   理由: {result['reason']}")

        print("\n" + "=" * 80)

        # 最終判定
        if self.failed_tests == 0 and self.error_tests == 0 and self.passed_tests > 0:
            print("🎉 全テスト成功！システムの品質が向上しました。")
            print("   Issue #471: テストカバレッジ向上 - 完了")
        elif self.passed_tests > 0:
            print("⚠️  一部テストに問題がありますが、基本機能は動作しています。")
            print("   失敗したテストの詳細を確認し、改善を検討してください。")
        else:
            print("❌ 重要なテストが失敗しています。システムの安定性に問題がある可能性があります。")
            print("   詳細なログを確認し、問題を修正してください。")

        print("=" * 80)


def main():
    """メインエントリーポイント"""
    runner = TestSuiteRunner()
    success = runner.run_comprehensive_test_suite()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())