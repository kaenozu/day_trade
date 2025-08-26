#!/usr/bin/env python3
"""
セキュリティ監査システム - CLIインターフェース
"""

import asyncio

from .enums import AuditConfig, AuditScope, ComplianceFramework
from .main_auditor import SecurityAuditor


async def main():
    """セキュリティ監査デモ"""
    print("=== 統合セキュリティ監査システム ===")

    config = AuditConfig(
        project_root=".",
        target_urls=["https://httpbin.org"],
        audit_scopes=[
            AuditScope.CODE_ANALYSIS,
            AuditScope.APPLICATION,
            AuditScope.INFRASTRUCTURE,
        ],
        compliance_frameworks=[
            ComplianceFramework.OWASP_TOP10,
            ComplianceFramework.NIST_CSF,
        ],
        enable_penetration_testing=False,  # デモ用に無効化
    )

    auditor = SecurityAuditor(config)

    print("セキュリティ監査実行中...")
    report = await auditor.run_comprehensive_audit()

    print("\n=== 監査結果 ===")
    print(f"レポートID: {report.report_id}")
    print(f"プロジェクト: {report.project_name}")
    print(f"総発見事項: {report.total_findings}件")
    print(f"  Critical: {report.critical_findings}件")
    print(f"  High: {report.high_findings}件")
    print(f"  Medium: {report.medium_findings}件")
    print(f"  Low: {report.low_findings}件")

    print("\n=== リスク評価 ===")
    risk = report.risk_assessment
    print(f"総合リスクスコア: {risk['overall_risk_score']}/100")
    print(f"リスクレベル: {risk['risk_level']}")
    print(f"ビジネス影響度: {risk['business_impact']}")

    print("\n=== コンプライアンス評価 ===")
    for framework, result in report.compliance_results.items():
        compliant = "✅" if result["compliant"] else "❌"
        print(f"{framework.upper()}: {compliant} (スコア: {result['score']})")

    print("\n=== 推奨事項 ===")
    for i, rec in enumerate(report.recommendations[:5], 1):
        print(f"{i}. {rec}")

    print("\n=== 修復ロードマップ ===")
    for phase in report.remediation_roadmap:
        print(f"\n{phase['phase']} (優先度: {phase['priority']})")
        print(f"  作業量: {phase['estimated_effort']}")
        print(f"  主要タスク: {', '.join(phase['tasks'][:3])}...")


if __name__ == "__main__":
    # 実行
    asyncio.run(main())