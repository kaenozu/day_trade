#!/usr/bin/env python3
"""
セキュリティテストフレームワーク - Framework Module
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

メインフレームワーククラス、テスト実行、レポート生成
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .authentication import PasswordSecurityTest, SessionSecurityTest
from .compliance import ComplianceTest
from .core import (
    SecurityTest,
    SecurityTestResult,
    TestCategory,
    TestSeverity,
    TestStatus,
)
from .encryption import EncryptionTest
from .network import NetworkSecurityTest
from .validation import InputValidationTest

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class SecurityTestFramework:
    """
    セキュリティテストフレームワーク

    各種セキュリティテストを統合実行し、
    包括的なセキュリティ評価レポートを生成
    """

    def __init__(self, output_path: str = "security/test_results"):
        """
        初期化

        Args:
            output_path: テスト結果出力パス
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # テスト登録
        self.tests: List[SecurityTest] = [
            PasswordSecurityTest(),
            SessionSecurityTest(),
            InputValidationTest(),
            EncryptionTest(),
            NetworkSecurityTest(),
            ComplianceTest(),
        ]

        # テスト結果
        self.test_results: List[SecurityTestResult] = []

        logger.info("SecurityTestFramework初期化完了")

    async def run_all_tests(self, **test_context) -> Dict[str, Any]:
        """全テスト実行"""
        logger.info("セキュリティテスト開始")
        start_time = datetime.utcnow()

        self.test_results = []

        # テスト並列実行
        tasks = []
        for test in self.tests:
            task = self._run_single_test(test, **test_context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果処理
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = self.tests[i].create_result(
                    TestStatus.ERROR,
                    error_message=str(result),
                    remediation="テスト実行環境を確認してください",
                )
                error_result.end_time = datetime.utcnow()
                self.test_results.append(error_result)
            else:
                self.test_results.append(result)

        end_time = datetime.utcnow()

        # 統合レポート生成
        report = self._generate_comprehensive_report(start_time, end_time)

        # 結果保存
        await self._save_test_results(report)

        logger.info(f"セキュリティテスト完了: {len(self.test_results)}テスト実行")

        return report

    async def run_specific_tests(
        self, categories: List[TestCategory], **test_context
    ) -> Dict[str, Any]:
        """特定カテゴリのテスト実行"""
        logger.info(f"セキュリティテスト開始: {[c.value for c in categories]}")
        start_time = datetime.utcnow()

        # カテゴリフィルタリング
        filtered_tests = [test for test in self.tests if test.category in categories]

        self.test_results = []

        # テスト実行
        tasks = []
        for test in filtered_tests:
            task = self._run_single_test(test, **test_context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果処理
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = filtered_tests[i].create_result(
                    TestStatus.ERROR, error_message=str(result)
                )
                error_result.end_time = datetime.utcnow()
                self.test_results.append(error_result)
            else:
                self.test_results.append(result)

        end_time = datetime.utcnow()

        # レポート生成
        report = self._generate_comprehensive_report(start_time, end_time)
        await self._save_test_results(report)

        logger.info(f"セキュリティテスト完了: {len(self.test_results)}テスト実行")

        return report

    async def _run_single_test(
        self, test: SecurityTest, **test_context
    ) -> SecurityTestResult:
        """単一テスト実行"""
        logger.info(f"テスト実行開始: {test.test_name}")
        start_time = datetime.utcnow()

        try:
            result = await test.execute(**test_context)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

            logger.info(f"テスト実行完了: {test.test_name} ({result.status.value})")
            return result

        except Exception as e:
            logger.error(f"テスト実行エラー: {test.test_name} - {e}")
            error_result = test.create_result(TestStatus.ERROR, error_message=str(e))
            error_result.end_time = datetime.utcnow()
            error_result.duration_seconds = (
                error_result.end_time - start_time
            ).total_seconds()
            return error_result

    def _generate_comprehensive_report(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """包括的レポート生成"""
        # 統計情報
        stats = {
            "total_tests": len(self.test_results),
            "passed": sum(
                1 for r in self.test_results if r.status == TestStatus.PASSED
            ),
            "failed": sum(
                1 for r in self.test_results if r.status == TestStatus.FAILED
            ),
            "skipped": sum(
                1 for r in self.test_results if r.status == TestStatus.SKIPPED
            ),
            "errors": sum(1 for r in self.test_results if r.status == TestStatus.ERROR),
        }

        # 重要度別統計
        severity_stats = {}
        for severity in TestSeverity:
            severity_results = [r for r in self.test_results if r.severity == severity]
            severity_stats[severity.value] = {
                "total": len(severity_results),
                "failed": sum(
                    1 for r in severity_results if r.status == TestStatus.FAILED
                ),
                "passed": sum(
                    1 for r in severity_results if r.status == TestStatus.PASSED
                ),
            }

        # カテゴリ別統計
        category_stats = {}
        for category in TestCategory:
            category_results = [r for r in self.test_results if r.category == category]
            if category_results:
                category_stats[category.value] = {
                    "total": len(category_results),
                    "failed": sum(
                        1 for r in category_results if r.status == TestStatus.FAILED
                    ),
                    "passed": sum(
                        1 for r in category_results if r.status == TestStatus.PASSED
                    ),
                }

        # 失敗したテストの詳細
        failed_tests = [
            {
                "test_id": r.test_id,
                "test_name": r.test_name,
                "category": r.category.value,
                "severity": r.severity.value,
                "description": r.description,
                "remediation": r.remediation,
                "evidence": r.evidence,
            }
            for r in self.test_results
            if r.status == TestStatus.FAILED
        ]

        # セキュリティスコア計算
        security_score = self._calculate_security_score()

        report = {
            "report_id": f"security-test-report-{int(start_time.timestamp())}",
            "generated_at": end_time.isoformat(),
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "security_score": security_score,
            "statistics": stats,
            "severity_breakdown": severity_stats,
            "category_breakdown": category_stats,
            "failed_tests": failed_tests,
            "detailed_results": [r.to_dict() for r in self.test_results],
            "recommendations": self._generate_recommendations(),
            "executive_summary": self._generate_executive_summary(
                stats, security_score
            ),
        }

        return report

    def _calculate_security_score(self) -> float:
        """セキュリティスコア計算"""
        if not self.test_results:
            return 0.0

        # 重要度による重み付け
        severity_weights = {
            TestSeverity.CRITICAL: 4.0,
            TestSeverity.HIGH: 3.0,
            TestSeverity.MEDIUM: 2.0,
            TestSeverity.LOW: 1.0,
            TestSeverity.INFO: 0.5,
        }

        total_score = 0.0
        max_possible_score = 0.0

        for result in self.test_results:
            weight = severity_weights[result.severity]
            max_possible_score += weight

            if result.status == TestStatus.PASSED:
                total_score += weight
            elif result.status == TestStatus.FAILED:
                total_score += 0  # 失敗は0点
            elif result.status == TestStatus.SKIPPED:
                max_possible_score -= weight  # スキップは計算から除外

        return (
            (total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
        )

    def _generate_recommendations(self) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        failed_results = [r for r in self.test_results if r.status == TestStatus.FAILED]

        # 重要度別推奨事項
        critical_failures = [
            r for r in failed_results if r.severity == TestSeverity.CRITICAL
        ]
        if critical_failures:
            recommendations.append(
                f"🚨 {len(critical_failures)}件の重大なセキュリティ問題を直ちに修正してください"
            )

        high_failures = [r for r in failed_results if r.severity == TestSeverity.HIGH]
        if high_failures:
            recommendations.append(
                f"⚠️ {len(high_failures)}件の高リスクセキュリティ問題を優先的に修正してください"
            )

        # カテゴリ別推奨事項
        category_failures = {}
        for result in failed_results:
            category = result.category.value
            category_failures[category] = category_failures.get(category, 0) + 1

        for category, count in category_failures.items():
            if count >= 2:
                recommendations.append(
                    f"🔧 {category}カテゴリで{count}件の問題があります。包括的な見直しを検討してください"
                )

        if not failed_results:
            recommendations.append(
                "✅ 全てのセキュリティテストが通過しています。定期的な再評価を継続してください"
            )

        return recommendations

    def _generate_executive_summary(
        self, stats: Dict[str, int], security_score: float
    ) -> str:
        """エグゼクティブサマリー生成"""
        pass_rate = (
            (stats["passed"] / stats["total_tests"] * 100)
            if stats["total_tests"] > 0
            else 0
        )

        summary = f"""セキュリティテスト実行結果サマリー

総テスト数: {stats["total_tests"]}
合格率: {pass_rate:.1f}%
セキュリティスコア: {security_score:.1f}/100

結果詳細:
- 合格: {stats["passed"]}
- 失敗: {stats["failed"]}
- スキップ: {stats["skipped"]}
- エラー: {stats["errors"]}

"""

        if security_score >= 90:
            summary += "🟢 セキュリティ状況: 良好\n優秀なセキュリティ実装です。現在の対策を維持してください。"
        elif security_score >= 70:
            summary += "🟡 セキュリティ状況: 注意\nいくつかの改善点があります。失敗したテストを確認し修正してください。"
        elif security_score >= 50:
            summary += "🟠 セキュリティ状況: 要改善\n重要なセキュリティ問題があります。優先的に対応が必要です。"
        else:
            summary += "🔴 セキュリティ状況: 危険\n深刻なセキュリティ脆弱性があります。直ちに対応してください。"

        return summary

    async def _save_test_results(self, report: Dict[str, Any]):
        """テスト結果保存"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON詳細レポート
        json_report_file = self.output_path / f"security_test_report_{timestamp}.json"
        with open(json_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # HTML要約レポート
        html_report = self._generate_html_report(report)
        html_report_file = self.output_path / f"security_test_summary_{timestamp}.html"
        with open(html_report_file, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"テスト結果保存完了: {json_report_file}, {html_report_file}")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """HTML要約レポート生成"""
        stats = report["statistics"]
        security_score = report["security_score"]

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>セキュリティテストレポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {"green" if security_score >= 70 else "orange" if security_score >= 50 else "red"}; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .failed-tests {{ margin-top: 30px; }}
        .test-item {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #ff4444; }}
        .recommendations {{ background: #e8f4fd; padding: 20px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>セキュリティテストレポート</h1>
        <p>生成日時: {report["generated_at"]}</p>
        <div class="score">{security_score:.1f}/100</div>
    </div>

    <div class="stats">
        <div class="stat">
            <h3>総テスト数</h3>
            <div>{stats["total_tests"]}</div>
        </div>
        <div class="stat">
            <h3>合格</h3>
            <div style="color: green;">{stats["passed"]}</div>
        </div>
        <div class="stat">
            <h3>失敗</h3>
            <div style="color: red;">{stats["failed"]}</div>
        </div>
        <div class="stat">
            <h3>スキップ</h3>
            <div>{stats["skipped"]}</div>
        </div>
    </div>

    <div class="recommendations">
        <h2>推奨事項</h2>
        <ul>
"""

        for rec in report["recommendations"]:
            html += f"            <li>{rec}</li>\n"

        html += """        </ul>
    </div>

    <div class="failed-tests">
        <h2>失敗したテスト</h2>
"""

        for test in report["failed_tests"]:
            html += f"""        <div class="test-item">
            <h3>{test["test_name"]} ({test["severity"].upper()})</h3>
            <p><strong>問題:</strong> {test["description"]}</p>
            <p><strong>修正方法:</strong> {test["remediation"]}</p>
        </div>
"""

        html += """    </div>
</body>
</html>"""

        return html


# Factory function
def create_security_test_framework(
    output_path: str = "security/test_results",
) -> SecurityTestFramework:
    """SecurityTestFrameworkファクトリ関数"""
    return SecurityTestFramework(output_path=output_path)