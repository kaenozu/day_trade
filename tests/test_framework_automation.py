#!/usr/bin/env python3
"""
Issue #760: 包括的テスト自動化と検証フレームワークの構築

高度なテスト自動化システム:
- 自動テスト発見・実行
- カバレッジ監視・レポート
- パフォーマンステスト統合
- CI/CD統合支援
"""

import asyncio
import unittest
import sys
import time
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# プロジェクトルート設定
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'test_automation.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """テスト結果データクラス"""
    name: str
    status: str  # passed, failed, error, skipped
    duration: float
    coverage: Optional[float] = None
    error_message: Optional[str] = None
    output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuite:
    """テストスイートデータクラス"""
    name: str
    test_files: List[Path]
    category: str  # unit, integration, performance, e2e
    priority: int = 1  # 1=高, 2=中, 3=低
    timeout: int = 300  # 秒
    dependencies: List[str] = field(default_factory=list)

class TestDiscovery:
    """テスト自動発見システム"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_patterns = [
            'test_*.py',
            '*_test.py'
        ]

    def discover_all_tests(self) -> Dict[str, TestSuite]:
        """全テストファイルの自動発見"""
        test_suites = {}

        # ユニットテスト
        unit_tests = self._discover_unit_tests()
        if unit_tests:
            test_suites['unit'] = TestSuite(
                name="Unit Tests",
                test_files=unit_tests,
                category="unit",
                priority=1,
                timeout=60
            )

        # 統合テスト
        integration_tests = self._discover_integration_tests()
        if integration_tests:
            test_suites['integration'] = TestSuite(
                name="Integration Tests",
                test_files=integration_tests,
                category="integration",
                priority=2,
                timeout=180
            )

        # MLテスト
        ml_tests = self._discover_ml_tests()
        if ml_tests:
            test_suites['ml'] = TestSuite(
                name="ML Tests",
                test_files=ml_tests,
                category="ml",
                priority=1,
                timeout=120
            )

        # パフォーマンステスト
        performance_tests = self._discover_performance_tests()
        if performance_tests:
            test_suites['performance'] = TestSuite(
                name="Performance Tests",
                test_files=performance_tests,
                category="performance",
                priority=3,
                timeout=600
            )

        logger.info(f"発見したテストスイート: {list(test_suites.keys())}")
        return test_suites

    def _discover_unit_tests(self) -> List[Path]:
        """ユニットテストの発見"""
        unit_tests = []
        tests_dir = self.project_root / 'tests'

        # 直接のテストファイル
        for pattern in self.test_patterns:
            unit_tests.extend(tests_dir.glob(pattern))

        # MLディレクトリのテスト
        ml_dir = tests_dir / 'ml'
        if ml_dir.exists():
            for pattern in self.test_patterns:
                unit_tests.extend(ml_dir.glob(pattern))

        return [t for t in unit_tests if self._is_unit_test(t)]

    def _discover_integration_tests(self) -> List[Path]:
        """統合テストの発見"""
        integration_dir = self.project_root / 'tests' / 'integration'
        if not integration_dir.exists():
            return []

        integration_tests = []
        for pattern in self.test_patterns:
            integration_tests.extend(integration_dir.glob(pattern))

        return integration_tests

    def _discover_ml_tests(self) -> List[Path]:
        """MLテストの発見"""
        ml_tests = []
        ml_dirs = [
            self.project_root / 'tests' / 'ml',
            self.project_root / 'tests' / 'ml' / 'base_models',
            self.project_root / 'tests' / 'ml' / 'stacking'
        ]

        for ml_dir in ml_dirs:
            if ml_dir.exists():
                for pattern in self.test_patterns:
                    ml_tests.extend(ml_dir.glob(pattern))

        return ml_tests

    def _discover_performance_tests(self) -> List[Path]:
        """パフォーマンステストの発見"""
        performance_dir = self.project_root / 'tests' / 'performance'
        if not performance_dir.exists():
            return []

        performance_tests = []
        for pattern in self.test_patterns:
            performance_tests.extend(performance_dir.glob(pattern))

        return performance_tests

    def _is_unit_test(self, test_path: Path) -> bool:
        """ユニットテストかどうかの判定"""
        # 統合テストディレクトリ以外のテスト
        return 'integration' not in str(test_path) and 'performance' not in str(test_path)

class TestExecutor:
    """テスト実行エンジン"""

    def __init__(self, project_root: Path, max_workers: int = 4):
        self.project_root = project_root
        self.max_workers = max_workers

    async def execute_test_suite(self, test_suite: TestSuite) -> List[TestResult]:
        """テストスイートの実行"""
        logger.info(f"テストスイート実行開始: {test_suite.name}")

        results = []

        # 並列実行可能なテストの場合
        if test_suite.category in ['unit', 'ml'] and len(test_suite.test_files) > 1:
            results = await self._execute_parallel(test_suite)
        else:
            results = await self._execute_sequential(test_suite)

        logger.info(f"テストスイート完了: {test_suite.name} - {len(results)} テスト実行")
        return results

    async def _execute_parallel(self, test_suite: TestSuite) -> List[TestResult]:
        """並列テスト実行"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # テストファイルごとにタスク作成
            future_to_test = {
                executor.submit(
                    self._run_single_test,
                    test_file,
                    test_suite.timeout
                ): test_file for test_file in test_suite.test_files
            }

            # 結果収集
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                try:
                    result = future.result(timeout=test_suite.timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"テスト実行エラー {test_file}: {e}")
                    results.append(TestResult(
                        name=test_file.name,
                        status="error",
                        duration=0.0,
                        error_message=str(e)
                    ))

        return results

    async def _execute_sequential(self, test_suite: TestSuite) -> List[TestResult]:
        """順次テスト実行"""
        results = []

        for test_file in test_suite.test_files:
            try:
                result = self._run_single_test(test_file, test_suite.timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"テスト実行エラー {test_file}: {e}")
                results.append(TestResult(
                    name=test_file.name,
                    status="error",
                    duration=0.0,
                    error_message=str(e)
                ))

        return results

    def _run_single_test(self, test_file: Path, timeout: int) -> TestResult:
        """単一テストファイルの実行"""
        start_time = time.time()

        try:
            # テストモジュールパスの生成
            relative_path = test_file.relative_to(self.project_root)
            module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')

            # pytest実行
            cmd = [
                sys.executable, '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=short',
                '--disable-warnings'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                status = "passed"
            else:
                status = "failed"

            return TestResult(
                name=test_file.name,
                status=status,
                duration=duration,
                output=result.stdout,
                error_message=result.stderr if result.stderr else None
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                name=test_file.name,
                status="error",
                duration=time.time() - start_time,
                error_message=f"テストタイムアウト ({timeout}秒)"
            )
        except Exception as e:
            return TestResult(
                name=test_file.name,
                status="error",
                duration=time.time() - start_time,
                error_message=str(e)
            )

class CoverageAnalyzer:
    """カバレッジ分析システム"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def measure_coverage(self, test_files: List[Path]) -> Dict[str, Any]:
        """カバレッジ測定"""
        try:
            # カバレッジ設定ファイル作成
            coverage_config = self._create_coverage_config()

            # カバレッジ付きでテスト実行
            cmd = [
                sys.executable, '-m', 'pytest',
                '--cov=src',
                '--cov-report=json',
                '--cov-report=term-missing',
            ] + [str(f) for f in test_files]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            # カバレッジデータの読み込み
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                return self._parse_coverage_data(coverage_data)

            return {'total_coverage': 0.0, 'files': {}}

        except Exception as e:
            logger.error(f"カバレッジ測定エラー: {e}")
            return {'total_coverage': 0.0, 'files': {}, 'error': str(e)}

    def _create_coverage_config(self) -> Path:
        """カバレッジ設定ファイルの作成"""
        config_content = """
[run]
source = src
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */env/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
"""
        config_file = self.project_root / '.coveragerc'
        with open(config_file, 'w') as f:
            f.write(config_content.strip())

        return config_file

    def _parse_coverage_data(self, coverage_data: Dict) -> Dict[str, Any]:
        """カバレッジデータの解析"""
        total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0.0)

        files_coverage = {}
        for filepath, file_data in coverage_data.get('files', {}).items():
            files_coverage[filepath] = {
                'coverage': file_data.get('summary', {}).get('percent_covered', 0.0),
                'lines_covered': file_data.get('summary', {}).get('covered_lines', 0),
                'lines_total': file_data.get('summary', {}).get('num_statements', 0)
            }

        return {
            'total_coverage': total_coverage,
            'files': files_coverage,
            'timestamp': time.time()
        }

class TestReporter:
    """テスト結果レポートシステム"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def generate_report(self,
                       test_results: Dict[str, List[TestResult]],
                       coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """包括的テストレポートの生成"""

        # 統計計算
        total_tests = sum(len(results) for results in test_results.values())
        passed_tests = sum(
            len([r for r in results if r.status == 'passed'])
            for results in test_results.values()
        )
        failed_tests = sum(
            len([r for r in results if r.status == 'failed'])
            for results in test_results.values()
        )
        error_tests = sum(
            len([r for r in results if r.status == 'error'])
            for results in test_results.values()
        )

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # レポートデータ構築
        report = {
            'timestamp': time.time(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'errors': error_tests,
                'success_rate': success_rate
            },
            'coverage': coverage_data,
            'test_suites': {},
            'recommendations': self._generate_recommendations(
                test_results, coverage_data, success_rate
            )
        }

        # テストスイート別詳細
        for suite_name, results in test_results.items():
            suite_passed = len([r for r in results if r.status == 'passed'])
            suite_total = len(results)
            suite_success_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0

            report['test_suites'][suite_name] = {
                'total': suite_total,
                'passed': suite_passed,
                'failed': len([r for r in results if r.status == 'failed']),
                'errors': len([r for r in results if r.status == 'error']),
                'success_rate': suite_success_rate,
                'avg_duration': sum(r.duration for r in results) / len(results) if results else 0,
                'details': [
                    {
                        'name': r.name,
                        'status': r.status,
                        'duration': r.duration,
                        'error': r.error_message
                    } for r in results
                ]
            }

        # レポートファイル保存
        self._save_report(report)

        return report

    def _generate_recommendations(self,
                                test_results: Dict[str, List[TestResult]],
                                coverage_data: Dict[str, Any],
                                success_rate: float) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []

        # カバレッジ改善
        total_coverage = coverage_data.get('total_coverage', 0.0)
        if total_coverage < 50:
            recommendations.append(
                f"カバレッジが{total_coverage:.1f}%と低いです。テストケースの追加を推奨します。"
            )

        # 失敗テストの分析
        failed_suites = []
        for suite_name, results in test_results.items():
            failed_count = len([r for r in results if r.status in ['failed', 'error']])
            if failed_count > 0:
                failed_suites.append(f"{suite_name} ({failed_count}件)")

        if failed_suites:
            recommendations.append(
                f"失敗したテストスイート: {', '.join(failed_suites)}。詳細を確認してください。"
            )

        # 成功率改善
        if success_rate < 80:
            recommendations.append(
                f"成功率が{success_rate:.1f}%です。テストの安定性向上を推奨します。"
            )

        # パフォーマンス改善
        slow_tests = []
        for results in test_results.values():
            for result in results:
                if result.duration > 30:  # 30秒以上のテスト
                    slow_tests.append(f"{result.name} ({result.duration:.1f}s)")

        if slow_tests:
            recommendations.append(
                f"実行時間の長いテスト: {', '.join(slow_tests[:3])}など。最適化を検討してください。"
            )

        return recommendations

    def _save_report(self, report: Dict[str, Any]) -> None:
        """レポートファイルの保存"""
        # JSON形式で保存
        report_file = self.project_root / 'test_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # 人間向けテキストレポート
        self._save_text_report(report)

    def _save_text_report(self, report: Dict[str, Any]) -> None:
        """テキストレポートの保存"""
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')

        content = f"""
# Day Trade テスト自動化レポート
生成日時: {timestamp_str}

## 📊 概要
- 総テスト数: {report['summary']['total_tests']}
- 成功: {report['summary']['passed']}
- 失敗: {report['summary']['failed']}
- エラー: {report['summary']['errors']}
- 成功率: {report['summary']['success_rate']:.1f}%
- カバレッジ: {report['coverage'].get('total_coverage', 0):.1f}%

## 🎯 テストスイート別結果
"""

        for suite_name, suite_data in report['test_suites'].items():
            content += f"""
### {suite_name}
- テスト数: {suite_data['total']}
- 成功率: {suite_data['success_rate']:.1f}%
- 平均実行時間: {suite_data['avg_duration']:.2f}秒
"""

        if report['recommendations']:
            content += "\n## 💡 改善推奨事項\n"
            for i, rec in enumerate(report['recommendations'], 1):
                content += f"{i}. {rec}\n"

        # ファイル保存
        report_file = self.project_root / 'test_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)

class TestAutomationFramework:
    """包括的テスト自動化フレームワーク"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.discovery = TestDiscovery(self.project_root)
        self.executor = TestExecutor(self.project_root)
        self.coverage_analyzer = CoverageAnalyzer(self.project_root)
        self.reporter = TestReporter(self.project_root)

    async def run_comprehensive_tests(self,
                                    suite_filter: Optional[List[str]] = None,
                                    include_coverage: bool = True) -> Dict[str, Any]:
        """包括的テスト実行"""
        logger.info("🚀 包括的テスト自動化開始")
        start_time = time.time()

        # テスト発見
        test_suites = self.discovery.discover_all_tests()
        logger.info(f"発見したテストスイート: {list(test_suites.keys())}")

        # フィルタ適用
        if suite_filter:
            test_suites = {
                name: suite for name, suite in test_suites.items()
                if name in suite_filter
            }

        # テスト実行
        test_results = {}
        all_test_files = []

        # 優先度順でテスト実行
        sorted_suites = sorted(
            test_suites.items(),
            key=lambda x: x[1].priority
        )

        for suite_name, test_suite in sorted_suites:
            logger.info(f"実行中: {suite_name}")
            results = await self.executor.execute_test_suite(test_suite)
            test_results[suite_name] = results
            all_test_files.extend(test_suite.test_files)

        # カバレッジ測定
        coverage_data = {}
        if include_coverage and all_test_files:
            logger.info("カバレッジ測定中...")
            coverage_data = self.coverage_analyzer.measure_coverage(all_test_files)

        # レポート生成
        logger.info("レポート生成中...")
        report = self.reporter.generate_report(test_results, coverage_data)

        # 実行時間
        total_time = time.time() - start_time
        report['execution_time'] = total_time

        logger.info(f"✅ テスト自動化完了 (実行時間: {total_time:.1f}秒)")

        return report

    def print_summary(self, report: Dict[str, Any]) -> None:
        """結果サマリーの表示"""
        print("\n" + "="*80)
        print("🎯 Day Trade テスト自動化結果")
        print("="*80)

        summary = report['summary']
        print(f"📊 総テスト数: {summary['total_tests']}")
        print(f"✅ 成功: {summary['passed']}")
        print(f"❌ 失敗: {summary['failed']}")
        print(f"⚠️  エラー: {summary['errors']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")

        if 'total_coverage' in report['coverage']:
            print(f"🎯 カバレッジ: {report['coverage']['total_coverage']:.1f}%")

        print(f"⏱️  実行時間: {report.get('execution_time', 0):.1f}秒")

        # 推奨事項
        if report['recommendations']:
            print("\n💡 改善推奨事項:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")

        # 評価
        if summary['success_rate'] >= 90:
            print("\n🎉 優秀！システムの品質は非常に高いです。")
        elif summary['success_rate'] >= 70:
            print("\n👍 良好！いくつかの改善点があります。")
        else:
            print("\n⚠️  注意が必要！システムの安定性向上が推奨されます。")

        print("="*80)

async def main():
    """メインエントリーポイント"""
    # フレームワーク初期化
    framework = TestAutomationFramework()

    # 包括的テスト実行
    report = await framework.run_comprehensive_tests()

    # 結果表示
    framework.print_summary(report)

    # 成功判定
    success_rate = report['summary']['success_rate']
    return 0 if success_rate >= 50 else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))