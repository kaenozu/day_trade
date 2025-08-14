#!/usr/bin/env python3
"""
デプロイ準備チェックスクリプト
Deployment Readiness Check Script

Issue #760: 包括的テスト自動化と検証フレームワークの構築
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import os

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentReadinessChecker:
    """デプロイ準備チェッカー"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checks_passed = 0
        self.checks_total = 0
        self.failure_reasons = []

    def check_test_success_rate(self, test_results_dir: str) -> bool:
        """テスト成功率チェック"""
        self.checks_total += 1
        logger.info("Checking test success rate...")

        try:
            # JUnit XML ファイルを探す
            test_files = list(Path(test_results_dir).glob("**/junit/*.xml"))
            if not test_files:
                # JSON形式のテスト結果も探す
                json_files = list(Path(test_results_dir).glob("**/*test-results*.json"))
                if json_files:
                    return self._check_json_test_results(json_files)
                else:
                    self.failure_reasons.append("No test result files found")
                    return False

            return self._check_junit_test_results(test_files)

        except Exception as e:
            self.failure_reasons.append(f"Test success rate check failed: {e}")
            return False

    def _check_junit_test_results(self, test_files: List[Path]) -> bool:
        """JUnit XML結果チェック"""
        import xml.etree.ElementTree as ET

        total_tests = 0
        failed_tests = 0
        error_tests = 0

        for test_file in test_files:
            try:
                tree = ET.parse(test_file)
                root = tree.getroot()

                # testsuite要素を探す
                for testsuite in root.findall('.//testsuite'):
                    tests = int(testsuite.get('tests', 0))
                    failures = int(testsuite.get('failures', 0))
                    errors = int(testsuite.get('errors', 0))

                    total_tests += tests
                    failed_tests += failures
                    error_tests += errors

            except Exception as e:
                logger.warning(f"Failed to parse {test_file}: {e}")
                continue

        if total_tests == 0:
            self.failure_reasons.append("No tests found in result files")
            return False

        success_rate = ((total_tests - failed_tests - error_tests) / total_tests) * 100
        min_success_rate = self.config.get('min_success_rate', 95.0)
        max_failed_tests = self.config.get('max_failed_tests', 5)

        logger.info(f"Test results: {total_tests} total, {failed_tests + error_tests} failed, {success_rate:.1f}% success")

        if success_rate < min_success_rate:
            self.failure_reasons.append(
                f"Test success rate {success_rate:.1f}% below minimum {min_success_rate}%"
            )
            return False

        if failed_tests + error_tests > max_failed_tests:
            self.failure_reasons.append(
                f"Failed tests {failed_tests + error_tests} exceeds maximum {max_failed_tests}"
            )
            return False

        self.checks_passed += 1
        return True

    def _check_json_test_results(self, json_files: List[Path]) -> bool:
        """JSON形式テスト結果チェック"""
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                summary = data.get('summary', {})
                success_rate = summary.get('success_rate', 0)
                failed_tests = summary.get('failed', 0) + summary.get('errors', 0)

                min_success_rate = self.config.get('min_success_rate', 95.0)
                max_failed_tests = self.config.get('max_failed_tests', 5)

                logger.info(f"Test success rate: {success_rate:.1f}%, failed tests: {failed_tests}")

                if success_rate < min_success_rate:
                    self.failure_reasons.append(
                        f"Test success rate {success_rate:.1f}% below minimum {min_success_rate}%"
                    )
                    return False

                if failed_tests > max_failed_tests:
                    self.failure_reasons.append(
                        f"Failed tests {failed_tests} exceeds maximum {max_failed_tests}"
                    )
                    return False

                self.checks_passed += 1
                return True

            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
                continue

        self.failure_reasons.append("No valid JSON test results found")
        return False

    def check_performance_benchmarks(self) -> bool:
        """パフォーマンスベンチマークチェック"""
        self.checks_total += 1
        logger.info("Checking performance benchmarks...")

        if not self.config.get('require_performance_pass', False):
            logger.info("Performance check not required")
            self.checks_passed += 1
            return True

        try:
            # パフォーマンス結果ファイルを探す
            perf_files = list(Path('.').glob("**/benchmark-*.json"))
            if not perf_files:
                self.failure_reasons.append("No performance benchmark results found")
                return False

            return self._validate_performance_results(perf_files)

        except Exception as e:
            self.failure_reasons.append(f"Performance check failed: {e}")
            return False

    def _validate_performance_results(self, perf_files: List[Path]) -> bool:
        """パフォーマンス結果検証"""
        for perf_file in perf_files:
            try:
                with open(perf_file, 'r') as f:
                    data = json.load(f)

                # Issue #761 目標値チェック
                if 'inference' in str(perf_file):
                    if not self._check_inference_targets(data):
                        return False

                # 一般的なパフォーマンス回帰チェック
                if not self._check_performance_regression(data):
                    return False

            except Exception as e:
                logger.warning(f"Failed to validate {perf_file}: {e}")
                continue

        self.checks_passed += 1
        return True

    def _check_inference_targets(self, data: Dict[str, Any]) -> bool:
        """推論システム目標値チェック"""
        # Issue #761 目標値
        targets = {
            'avg_latency_ms': 5.0,  # <5ms
            'throughput_per_sec': 10000.0,  # >10,000/sec
            'memory_efficiency': 0.5,  # 50%削減
            'accuracy_retention': 0.97  # 97%以上
        }

        for metric, target in targets.items():
            if metric in data:
                value = data[metric]

                # 閾値チェック
                if metric == 'avg_latency_ms' and value > target:
                    self.failure_reasons.append(
                        f"Inference latency {value:.2f}ms exceeds target {target}ms"
                    )
                    return False
                elif metric == 'throughput_per_sec' and value < target:
                    self.failure_reasons.append(
                        f"Throughput {value:.1f}/sec below target {target}/sec"
                    )
                    return False
                elif metric in ['memory_efficiency', 'accuracy_retention'] and value < target:
                    self.failure_reasons.append(
                        f"{metric} {value:.3f} below target {target}"
                    )
                    return False

        return True

    def _check_performance_regression(self, data: Dict[str, Any]) -> bool:
        """パフォーマンス回帰チェック"""
        # ベースラインとの比較
        baseline_file = self.config.get('performance_baseline', 'tests/baselines/performance.json')

        if not Path(baseline_file).exists():
            logger.info("No performance baseline found, skipping regression check")
            return True

        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)

            # 主要メトリクスの回帰チェック
            regression_threshold = self.config.get('performance_regression_threshold', 10.0)

            for metric in ['avg_latency_ms', 'throughput_per_sec', 'memory_usage_mb']:
                if metric in data and metric in baseline:
                    current = data[metric]
                    baseline_val = baseline[metric]

                    if baseline_val > 0:
                        change = ((current - baseline_val) / baseline_val) * 100

                        # レイテンシとメモリは増加が悪い、スループットは減少が悪い
                        if metric in ['avg_latency_ms', 'memory_usage_mb']:
                            if change > regression_threshold:
                                self.failure_reasons.append(
                                    f"{metric} regression: {change:.1f}% increase"
                                )
                                return False
                        elif metric == 'throughput_per_sec':
                            if change < -regression_threshold:
                                self.failure_reasons.append(
                                    f"{metric} regression: {abs(change):.1f}% decrease"
                                )
                                return False

        except Exception as e:
            logger.warning(f"Performance regression check failed: {e}")

        return True

    def check_security_scans(self) -> bool:
        """セキュリティスキャンチェック"""
        self.checks_total += 1
        logger.info("Checking security scan results...")

        try:
            # セキュリティスキャン結果を確認
            security_files = list(Path('.').glob("**/*security*.json")) + \
                           list(Path('.').glob("**/*bandit*.json")) + \
                           list(Path('.').glob("**/*trivy*.json"))

            if not security_files:
                if self.config.get('require_security_pass', False):
                    self.failure_reasons.append("No security scan results found")
                    return False
                else:
                    logger.info("Security scans not required")
                    self.checks_passed += 1
                    return True

            return self._validate_security_results(security_files)

        except Exception as e:
            self.failure_reasons.append(f"Security check failed: {e}")
            return False

    def _validate_security_results(self, security_files: List[Path]) -> bool:
        """セキュリティ結果検証"""
        max_high_issues = self.config.get('max_security_high_issues', 0)
        max_medium_issues = self.config.get('max_security_medium_issues', 5)

        for security_file in security_files:
            try:
                with open(security_file, 'r') as f:
                    data = json.load(f)

                # Bandit形式
                if 'results' in data:
                    high_issues = len([r for r in data['results'] if r.get('issue_severity') == 'HIGH'])
                    medium_issues = len([r for r in data['results'] if r.get('issue_severity') == 'MEDIUM'])

                    if high_issues > max_high_issues:
                        self.failure_reasons.append(
                            f"High security issues: {high_issues} (max: {max_high_issues})"
                        )
                        return False

                    if medium_issues > max_medium_issues:
                        self.failure_reasons.append(
                            f"Medium security issues: {medium_issues} (max: {max_medium_issues})"
                        )
                        return False

            except Exception as e:
                logger.warning(f"Failed to validate {security_file}: {e}")
                continue

        self.checks_passed += 1
        return True

    def check_code_coverage(self) -> bool:
        """コードカバレッジチェック"""
        self.checks_total += 1
        logger.info("Checking code coverage...")

        try:
            # カバレッジファイルを探す
            coverage_files = list(Path('.').glob("**/coverage.xml")) + \
                           list(Path('.').glob("**/.coverage"))

            if not coverage_files:
                if self.config.get('require_coverage_check', False):
                    self.failure_reasons.append("No coverage results found")
                    return False
                else:
                    logger.info("Coverage check not required")
                    self.checks_passed += 1
                    return True

            return self._validate_coverage_results(coverage_files)

        except Exception as e:
            self.failure_reasons.append(f"Coverage check failed: {e}")
            return False

    def _validate_coverage_results(self, coverage_files: List[Path]) -> bool:
        """カバレッジ結果検証"""
        min_coverage = self.config.get('min_code_coverage', 80.0)

        for coverage_file in coverage_files:
            try:
                if coverage_file.suffix == '.xml':
                    # XML形式のカバレッジ
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(coverage_file)
                    root = tree.getroot()

                    # カバレッジ率取得
                    coverage_elem = root.find('.//coverage')
                    if coverage_elem is not None:
                        line_rate = float(coverage_elem.get('line-rate', 0)) * 100

                        if line_rate < min_coverage:
                            self.failure_reasons.append(
                                f"Code coverage {line_rate:.1f}% below minimum {min_coverage}%"
                            )
                            return False

                elif coverage_file.name == '.coverage':
                    # coverage.py形式
                    try:
                        import coverage
                        cov = coverage.Coverage(data_file=str(coverage_file))
                        cov.load()
                        total_lines = cov.get_data().lines_covered()
                        if total_lines:
                            coverage_percent = (sum(len(lines) for lines in total_lines.values()) /
                                             sum(len(lines) for lines in cov.get_data().lines().values())) * 100

                            if coverage_percent < min_coverage:
                                self.failure_reasons.append(
                                    f"Code coverage {coverage_percent:.1f}% below minimum {min_coverage}%"
                                )
                                return False
                    except ImportError:
                        logger.warning("Coverage module not available")

            except Exception as e:
                logger.warning(f"Failed to validate {coverage_file}: {e}")
                continue

        self.checks_passed += 1
        return True

    def run_all_checks(self, test_results_dir: str = ".") -> bool:
        """全チェック実行"""
        logger.info("Starting deployment readiness checks...")

        checks = [
            lambda: self.check_test_success_rate(test_results_dir),
            self.check_performance_benchmarks,
            self.check_security_scans,
            self.check_code_coverage
        ]

        all_passed = True

        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                self.failure_reasons.append(f"Check failed: {e}")
                all_passed = False

        self._print_summary()
        return all_passed

    def _print_summary(self):
        """結果サマリー出力"""
        logger.info(f"\n=== Deployment Readiness Check Summary ===")
        logger.info(f"Checks passed: {self.checks_passed}/{self.checks_total}")

        if self.failure_reasons:
            logger.error("❌ Deployment readiness check FAILED")
            logger.error("Failure reasons:")
            for reason in self.failure_reasons:
                logger.error(f"  - {reason}")
        else:
            logger.info("✅ Deployment readiness check PASSED")

        # CI環境向けの出力
        if os.getenv('GITHUB_ACTIONS'):
            if self.failure_reasons:
                print(f"::error::Deployment readiness check failed: {'; '.join(self.failure_reasons)}")
            else:
                print("::notice::Deployment readiness check passed")


def main():
    parser = argparse.ArgumentParser(description='Check deployment readiness')
    parser.add_argument('--min-success-rate', type=float, default=95.0,
                       help='Minimum test success rate (%)')
    parser.add_argument('--max-failed-tests', type=int, default=5,
                       help='Maximum number of failed tests')
    parser.add_argument('--require-performance-pass', type=bool, default=False,
                       help='Require performance tests to pass')
    parser.add_argument('--require-security-pass', type=bool, default=False,
                       help='Require security scans to pass')
    parser.add_argument('--require-coverage-check', type=bool, default=False,
                       help='Require code coverage check')
    parser.add_argument('--min-code-coverage', type=float, default=80.0,
                       help='Minimum code coverage (%)')
    parser.add_argument('--max-security-high-issues', type=int, default=0,
                       help='Maximum high severity security issues')
    parser.add_argument('--max-security-medium-issues', type=int, default=5,
                       help='Maximum medium severity security issues')
    parser.add_argument('--performance-baseline', type=str,
                       default='tests/baselines/performance.json',
                       help='Performance baseline file')
    parser.add_argument('--performance-regression-threshold', type=float, default=10.0,
                       help='Performance regression threshold (%)')
    parser.add_argument('--test-results-dir', type=str, default='.',
                       help='Directory containing test results')

    args = parser.parse_args()

    config = {
        'min_success_rate': args.min_success_rate,
        'max_failed_tests': args.max_failed_tests,
        'require_performance_pass': args.require_performance_pass,
        'require_security_pass': args.require_security_pass,
        'require_coverage_check': args.require_coverage_check,
        'min_code_coverage': args.min_code_coverage,
        'max_security_high_issues': args.max_security_high_issues,
        'max_security_medium_issues': args.max_security_medium_issues,
        'performance_baseline': args.performance_baseline,
        'performance_regression_threshold': args.performance_regression_threshold
    }

    checker = DeploymentReadinessChecker(config)
    success = checker.run_all_checks(args.test_results_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()