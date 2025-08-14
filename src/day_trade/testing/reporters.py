#!/usr/bin/env python3
"""
テスト結果レポート生成
Test Result Report Generation

Issue #760: 包括的テスト自動化と検証フレームワークの構築
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

# ログ設定
logger = logging.getLogger(__name__)


@dataclass
class TestMetrics:
    """テストメトリクス"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 0.0
    performance_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class CoverageMetrics:
    """カバレッジメトリクス"""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    file_coverage_details: Dict[str, float] = None
    missing_lines: Dict[str, List[int]] = None

    def __post_init__(self):
        if self.file_coverage_details is None:
            self.file_coverage_details = {}
        if self.missing_lines is None:
            self.missing_lines = {}


class BaseReporter:
    """レポート基底クラス"""

    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_report(self, data: Dict[str, Any]) -> str:
        """レポート生成"""
        raise NotImplementedError

    def save_report(self, content: str, filename: str) -> str:
        """レポート保存"""
        file_path = self.output_dir / f"{self.timestamp}_{filename}"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Report saved: {file_path}")
        return str(file_path)


class TestReporter(BaseReporter):
    """基本テストレポート"""

    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """テスト結果レポート生成"""
        summary = test_results.get("summary", {})
        suite_results = test_results.get("suite_results", {})

        report_lines = [
            "# Test Execution Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total Tests: {summary.get('total_tests', 0)}",
            f"- Passed: {summary.get('passed', 0)}",
            f"- Failed: {summary.get('failed', 0)}",
            f"- Errors: {summary.get('errors', 0)}",
            f"- Skipped: {summary.get('skipped', 0)}",
            f"- Success Rate: {summary.get('success_rate', 0):.1f}%",
            f"- Total Duration: {summary.get('total_duration', 0):.2f}s",
            f"- Average Test Duration: {summary.get('average_test_duration', 0):.2f}s",
            ""
        ]

        # スイート別詳細
        report_lines.append("## Test Suite Details")

        for suite_name, results in suite_results.items():
            report_lines.append(f"### {suite_name}")

            passed = sum(1 for r in results if getattr(r, 'status', 'unknown') == 'passed')
            failed = sum(1 for r in results if getattr(r, 'status', 'unknown') == 'failed')
            errors = sum(1 for r in results if getattr(r, 'status', 'unknown') == 'error')

            report_lines.extend([
                f"- Tests: {len(results)}",
                f"- Passed: {passed}",
                f"- Failed: {failed}",
                f"- Errors: {errors}",
                ""
            ])

            # 失敗したテストの詳細
            failed_tests = [r for r in results if getattr(r, 'status', 'unknown') in ['failed', 'error']]
            if failed_tests:
                report_lines.append("#### Failed Tests")
                for test in failed_tests:
                    test_name = getattr(test, 'test_name', 'Unknown')
                    error_msg = getattr(test, 'error_message', 'No error message')
                    report_lines.extend([
                        f"- **{test_name}**",
                        f"  - Error: {error_msg}",
                        ""
                    ])

        return "\n".join(report_lines)

    def generate_json_report(self, test_results: Dict[str, Any]) -> str:
        """JSON形式レポート生成"""
        return json.dumps(test_results, indent=2, default=str)

    def generate_junit_xml(self, test_results: Dict[str, Any]) -> str:
        """JUnit XML形式レポート生成"""
        suite_results = test_results.get("suite_results", {})
        summary = test_results.get("summary", {})

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuites tests="{summary.get("total_tests", 0)}" '
            f'failures="{summary.get("failed", 0)}" '
            f'errors="{summary.get("errors", 0)}" '
            f'time="{summary.get("total_duration", 0):.2f}">'
        ]

        for suite_name, results in suite_results.items():
            suite_tests = len(results)
            suite_failures = sum(1 for r in results if getattr(r, 'status', 'unknown') == 'failed')
            suite_errors = sum(1 for r in results if getattr(r, 'status', 'unknown') == 'error')
            suite_time = sum(getattr(r, 'duration_seconds', 0) for r in results)

            xml_lines.append(
                f'  <testsuite name="{suite_name}" tests="{suite_tests}" '
                f'failures="{suite_failures}" errors="{suite_errors}" time="{suite_time:.2f}">'
            )

            for result in results:
                test_name = getattr(result, 'test_name', 'Unknown')
                duration = getattr(result, 'duration_seconds', 0)
                status = getattr(result, 'status', 'unknown')

                xml_lines.append(f'    <testcase name="{test_name}" time="{duration:.2f}">')

                if status == 'failed':
                    error_msg = getattr(result, 'error_message', 'Test failed')
                    xml_lines.append(f'      <failure message="{error_msg}"/>')
                elif status == 'error':
                    error_msg = getattr(result, 'error_message', 'Test error')
                    xml_lines.append(f'      <error message="{error_msg}"/>')
                elif status == 'skipped':
                    xml_lines.append('      <skipped/>')

                xml_lines.append('    </testcase>')

            xml_lines.append('  </testsuite>')

        xml_lines.append('</testsuites>')
        return "\n".join(xml_lines)


class PerformanceReporter(BaseReporter):
    """パフォーマンステストレポート"""

    def __init__(self, output_dir: str = "test_reports", baseline_file: Optional[str] = None):
        super().__init__(output_dir)
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline() if baseline_file else {}

    def _load_baseline(self) -> Dict[str, Any]:
        """ベースライン データ読み込み"""
        try:
            if os.path.exists(self.baseline_file):
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load baseline: {e}")
        return {}

    def generate_report(self, performance_data: Dict[str, Any]) -> str:
        """パフォーマンスレポート生成"""
        report_lines = [
            "# Performance Test Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # 実行時間分析
        if "execution_times" in performance_data:
            times = performance_data["execution_times"]
            report_lines.extend([
                "## Execution Time Analysis",
                f"- Average: {np.mean(times):.2f}ms",
                f"- Median: {np.median(times):.2f}ms",
                f"- 95th Percentile: {np.percentile(times, 95):.2f}ms",
                f"- Maximum: {np.max(times):.2f}ms",
                f"- Standard Deviation: {np.std(times):.2f}ms",
                ""
            ])

            # ベースラインとの比較
            if "execution_times" in self.baseline_data:
                baseline_avg = np.mean(self.baseline_data["execution_times"])
                current_avg = np.mean(times)
                improvement = (baseline_avg - current_avg) / baseline_avg * 100

                report_lines.extend([
                    "### Baseline Comparison",
                    f"- Baseline Average: {baseline_avg:.2f}ms",
                    f"- Current Average: {current_avg:.2f}ms",
                    f"- Performance Change: {improvement:+.1f}%",
                    ""
                ])

        # メモリ使用量分析
        if "memory_usage" in performance_data:
            memory = performance_data["memory_usage"]
            report_lines.extend([
                "## Memory Usage Analysis",
                f"- Peak Memory: {memory.get('peak_mb', 0):.1f}MB",
                f"- Average Memory: {memory.get('average_mb', 0):.1f}MB",
                f"- Memory Efficiency Score: {memory.get('efficiency_score', 0):.2f}",
                ""
            ])

        # スループット分析
        if "throughput" in performance_data:
            throughput = performance_data["throughput"]
            report_lines.extend([
                "## Throughput Analysis",
                f"- Throughput: {throughput.get('items_per_second', 0):.1f} items/sec",
                f"- Total Items Processed: {throughput.get('total_items', 0)}",
                f"- Processing Duration: {throughput.get('duration_seconds', 0):.2f}s",
                ""
            ])

        return "\n".join(report_lines)

    def generate_performance_charts(self, performance_data: Dict[str, Any]) -> List[str]:
        """パフォーマンスチャート生成"""
        chart_files = []

        # 実行時間ヒストグラム
        if "execution_times" in performance_data:
            plt.figure(figsize=(10, 6))
            times = performance_data["execution_times"]

            plt.hist(times, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}ms')
            plt.axvline(np.median(times), color='green', linestyle='--', label=f'Median: {np.median(times):.2f}ms')

            plt.xlabel('Execution Time (ms)')
            plt.ylabel('Frequency')
            plt.title('Execution Time Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            chart_file = self.output_dir / f"{self.timestamp}_execution_time_histogram.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()

            chart_files.append(str(chart_file))

        # メモリ使用量時系列
        if "memory_timeline" in performance_data:
            plt.figure(figsize=(12, 6))
            timeline = performance_data["memory_timeline"]

            if isinstance(timeline, dict) and "timestamps" in timeline and "memory_mb" in timeline:
                plt.plot(timeline["timestamps"], timeline["memory_mb"], linewidth=2)
                plt.xlabel('Time')
                plt.ylabel('Memory Usage (MB)')
                plt.title('Memory Usage Over Time')
                plt.grid(True, alpha=0.3)

                chart_file = self.output_dir / f"{self.timestamp}_memory_timeline.png"
                plt.savefig(chart_file, dpi=150, bbox_inches='tight')
                plt.close()

                chart_files.append(str(chart_file))

        return chart_files

    def save_baseline(self, performance_data: Dict[str, Any]) -> None:
        """ベースライン データ保存"""
        if self.baseline_file:
            try:
                baseline_dir = os.path.dirname(self.baseline_file)
                os.makedirs(baseline_dir, exist_ok=True)

                with open(self.baseline_file, 'w') as f:
                    json.dump(performance_data, f, indent=2, default=str)

                logger.info(f"Baseline saved: {self.baseline_file}")
            except Exception as e:
                logger.error(f"Failed to save baseline: {e}")


class CoverageReporter(BaseReporter):
    """コードカバレッジレポート"""

    def generate_report(self, coverage_data: Dict[str, Any]) -> str:
        """カバレッジレポート生成"""
        report_lines = [
            "# Code Coverage Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # 総合カバレッジ
        overall = coverage_data.get("overall", {})
        report_lines.extend([
            "## Overall Coverage",
            f"- Line Coverage: {overall.get('line_coverage', 0):.1f}%",
            f"- Branch Coverage: {overall.get('branch_coverage', 0):.1f}%",
            f"- Function Coverage: {overall.get('function_coverage', 0):.1f}%",
            ""
        ])

        # ファイル別カバレッジ
        files = coverage_data.get("files", {})
        if files:
            report_lines.append("## File Coverage Details")

            # カバレッジ順でソート
            sorted_files = sorted(files.items(), key=lambda x: x[1].get('coverage', 0))

            for file_path, file_data in sorted_files:
                coverage_pct = file_data.get('coverage', 0)
                lines_covered = file_data.get('lines_covered', 0)
                lines_total = file_data.get('lines_total', 0)
                missing_lines = file_data.get('missing_lines', [])

                status_icon = "✅" if coverage_pct >= 90 else "⚠️" if coverage_pct >= 70 else "❌"

                report_lines.extend([
                    f"### {status_icon} {file_path}",
                    f"- Coverage: {coverage_pct:.1f}% ({lines_covered}/{lines_total} lines)",
                ])

                if missing_lines:
                    missing_str = ", ".join(map(str, missing_lines[:10]))
                    if len(missing_lines) > 10:
                        missing_str += f", ... ({len(missing_lines) - 10} more)"
                    report_lines.append(f"- Missing Lines: {missing_str}")

                report_lines.append("")

        return "\n".join(report_lines)

    def generate_coverage_badge(self, coverage_percentage: float) -> str:
        """カバレッジバッジSVG生成"""
        if coverage_percentage >= 90:
            color = "brightgreen"
        elif coverage_percentage >= 70:
            color = "yellow"
        elif coverage_percentage >= 50:
            color = "orange"
        else:
            color = "red"

        svg_template = '''
        <svg xmlns="http://www.w3.org/2000/svg" width="104" height="20">
            <linearGradient id="b" x2="0" y2="100%">
                <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
                <stop offset="1" stop-opacity=".1"/>
            </linearGradient>
            <mask id="a">
                <rect width="104" height="20" rx="3" fill="#fff"/>
            </mask>
            <g mask="url(#a)">
                <path fill="#555" d="M0 0h63v20H0z"/>
                <path fill="{color}" d="M63 0h41v20H63z"/>
                <path fill="url(#b)" d="M0 0h104v20H0z"/>
            </g>
            <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
                <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
                <text x="31.5" y="14">coverage</text>
                <text x="82.5" y="15" fill="#010101" fill-opacity=".3">{percentage}%</text>
                <text x="82.5" y="14">{percentage}%</text>
            </g>
        </svg>
        '''.format(color=color, percentage=int(coverage_percentage))

        return svg_template.strip()


class HTMLReporter(BaseReporter):
    """HTML総合レポート"""

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Report - {{ timestamp }}</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
            .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
            .metric-label { color: #666; margin-top: 5px; }
            .passed { color: #28a745; }
            .failed { color: #dc3545; }
            .warning { color: #ffc107; }
            .section { margin-bottom: 30px; }
            .chart-container { text-align: center; margin: 20px 0; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: bold; }
            .status-passed { color: #28a745; font-weight: bold; }
            .status-failed { color: #dc3545; font-weight: bold; }
            .status-error { color: #fd7e14; font-weight: bold; }
            .status-skipped { color: #6c757d; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Test Execution Report</h1>
                <p>Generated: {{ timestamp }}</p>
            </div>

            <div class="section">
                <h2>📈 Summary</h2>
                <div class="summary">
                    <div class="metric-card">
                        <div class="metric-value">{{ summary.total_tests }}</div>
                        <div class="metric-label">Total Tests</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value passed">{{ summary.passed }}</div>
                        <div class="metric-label">Passed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value failed">{{ summary.failed }}</div>
                        <div class="metric-label">Failed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value warning">{{ summary.errors }}</div>
                        <div class="metric-label">Errors</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(summary.success_rate) }}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(summary.total_duration) }}s</div>
                        <div class="metric-label">Total Duration</div>
                    </div>
                </div>
            </div>

            {% if coverage %}
            <div class="section">
                <h2>📊 Code Coverage</h2>
                <div class="summary">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(coverage.line_coverage) }}%</div>
                        <div class="metric-label">Line Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(coverage.branch_coverage) }}%</div>
                        <div class="metric-label">Branch Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f"|format(coverage.function_coverage) }}%</div>
                        <div class="metric-label">Function Coverage</div>
                    </div>
                </div>
            </div>
            {% endif %}

            {% if performance %}
            <div class="section">
                <h2>⚡ Performance Metrics</h2>
                <div class="summary">
                    {% for metric, value in performance.items() %}
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(value) if value is number else value }}</div>
                        <div class="metric-label">{{ metric.replace('_', ' ').title() }}</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}

            {% if charts %}
            <div class="section">
                <h2>📈 Charts</h2>
                {% for chart in charts %}
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ chart }}" alt="Performance Chart" style="max-width: 100%; height: auto;">
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="section">
                <h2>🧪 Test Details</h2>
                {% for suite_name, results in suite_results.items() %}
                <h3>{{ suite_name }}</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Error Message</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in results %}
                        <tr>
                            <td>{{ result.test_name }}</td>
                            <td class="status-{{ result.status }}">{{ result.status.upper() }}</td>
                            <td>{{ "%.3f"|format(result.duration_seconds) }}s</td>
                            <td>{{ result.error_message or '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

    def generate_report(self, report_data: Dict[str, Any]) -> str:
        """HTML レポート生成"""
        template = Template(self.HTML_TEMPLATE)

        # チャートをBase64エンコード
        charts = []
        if "chart_files" in report_data:
            for chart_file in report_data["chart_files"]:
                if os.path.exists(chart_file):
                    with open(chart_file, 'rb') as f:
                        chart_data = base64.b64encode(f.read()).decode('utf-8')
                        charts.append(chart_data)

        return template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            summary=report_data.get("summary", {}),
            coverage=report_data.get("coverage", {}),
            performance=report_data.get("performance", {}),
            suite_results=report_data.get("suite_results", {}),
            charts=charts
        )


# 統合レポート生成器
class ComprehensiveReporter:
    """包括的レポート生成器"""

    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.test_reporter = TestReporter(output_dir)
        self.performance_reporter = PerformanceReporter(output_dir)
        self.coverage_reporter = CoverageReporter(output_dir)
        self.html_reporter = HTMLReporter(output_dir)

    def generate_all_reports(
        self,
        test_results: Dict[str, Any],
        performance_data: Optional[Dict[str, Any]] = None,
        coverage_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """全レポート生成"""
        report_files = {}

        # 基本テストレポート
        test_report = self.test_reporter.generate_report(test_results)
        report_files["test_markdown"] = self.test_reporter.save_report(test_report, "test_report.md")

        # JSON レポート
        json_report = self.test_reporter.generate_json_report(test_results)
        report_files["test_json"] = self.test_reporter.save_report(json_report, "test_report.json")

        # JUnit XML
        junit_xml = self.test_reporter.generate_junit_xml(test_results)
        report_files["junit_xml"] = self.test_reporter.save_report(junit_xml, "junit_report.xml")

        # パフォーマンスレポート
        if performance_data:
            perf_report = self.performance_reporter.generate_report(performance_data)
            report_files["performance"] = self.performance_reporter.save_report(perf_report, "performance_report.md")

            # パフォーマンスチャート
            chart_files = self.performance_reporter.generate_performance_charts(performance_data)
            if chart_files:
                report_files["charts"] = chart_files

        # カバレッジレポート
        if coverage_data:
            coverage_report = self.coverage_reporter.generate_report(coverage_data)
            report_files["coverage"] = self.coverage_reporter.save_report(coverage_report, "coverage_report.md")

        # HTML 統合レポート
        html_data = {
            "summary": test_results.get("summary", {}),
            "suite_results": test_results.get("suite_results", {}),
            "performance": performance_data,
            "coverage": coverage_data,
            "chart_files": report_files.get("charts", [])
        }

        html_report = self.html_reporter.generate_report(html_data)
        report_files["html"] = self.html_reporter.save_report(html_report, "comprehensive_report.html")

        logger.info(f"Generated {len(report_files)} report files")
        return report_files


# 使用例
if __name__ == "__main__":
    # サンプルデータでテスト
    sample_test_results = {
        "summary": {
            "total_tests": 100,
            "passed": 85,
            "failed": 10,
            "errors": 3,
            "skipped": 2,
            "success_rate": 85.0,
            "total_duration": 120.5,
            "average_test_duration": 1.205
        },
        "suite_results": {
            "unit_tests": [],
            "integration_tests": []
        }
    }

    # レポート生成
    reporter = ComprehensiveReporter()
    report_files = reporter.generate_all_reports(sample_test_results)

    print("Generated report files:")
    for report_type, file_path in report_files.items():
        print(f"  {report_type}: {file_path}")