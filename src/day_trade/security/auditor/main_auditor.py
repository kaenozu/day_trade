#!/usr/bin/env python3
"""
統合セキュリティ監査システム - メインオーケストレーター
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from .code_analyzer import CodeSecurityAnalyzer
from .compliance_assessor import ComplianceAssessor
from .enums import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    AuditScope,
    ComplianceFramework,
    SecurityReport,
)
from .infrastructure_analyzer import InfrastructureSecurityAnalyzer

try:
    from ...utils.logging_config import get_context_logger
    from ..penetration_tester import (
        PenetrationTester,
        PenTestConfig,
    )
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    # フォールバック
    class PenetrationTester:
        pass

    class PenTestConfig:
        pass


logger = get_context_logger(__name__)


class SecurityAuditor:
    """統合セキュリティ監査システム"""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.code_analyzer = CodeSecurityAnalyzer(config.project_root)
        self.infra_analyzer = InfrastructureSecurityAnalyzer(config)
        self.compliance_assessor = ComplianceAssessor(config.compliance_frameworks)

        logger.info(f"セキュリティ監査システム初期化: {config.project_root}")

    async def run_comprehensive_audit(self) -> SecurityReport:
        """包括的セキュリティ監査実行"""
        logger.info("包括的セキュリティ監査開始")
        start_time = time.time()

        all_findings = []

        # 監査タスクの並列実行
        tasks = []

        if AuditScope.CODE_ANALYSIS in self.config.audit_scopes:
            tasks.append(self.code_analyzer.analyze_code_security())

        if AuditScope.INFRASTRUCTURE in self.config.audit_scopes:
            tasks.append(self.infra_analyzer.analyze_infrastructure_security())

        if (
            AuditScope.APPLICATION in self.config.audit_scopes
            and self.config.enable_penetration_testing
        ):
            pen_test_tasks = []
            for url in self.config.target_urls:
                try:
                    pen_config = PenTestConfig(
                        target_base_url=url,
                        test_timeout=300,
                        include_aggressive_tests=False,
                    )
                    pen_tester = PenetrationTester(pen_config)
                    pen_test_tasks.append(pen_tester.run_comprehensive_pentest())
                except Exception:
                    # PenetrationTesterが利用できない場合はスキップ
                    pass

            if pen_test_tasks:
                tasks.extend(pen_test_tasks)

        # 監査実行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果の統合
        for result in results:
            if isinstance(result, list):
                all_findings.extend(result)
            elif isinstance(result, dict) and "vulnerabilities" in result:
                # ペネトレーションテスト結果の変換
                pen_test_findings = self._convert_pentest_results(result)
                all_findings.extend(pen_test_findings)
            elif isinstance(result, Exception):
                logger.error(f"監査タスクエラー: {result}")

        # 重複除去と優先度ソート
        unique_findings = self._deduplicate_findings(all_findings)
        sorted_findings = sorted(
            unique_findings,
            key=lambda f: (self._get_severity_priority(f.severity), -f.risk_rating),
            reverse=True,
        )

        # コンプライアンス評価
        compliance_results = self.compliance_assessor.assess_compliance(sorted_findings)

        # リスク評価
        risk_assessment = self._perform_risk_assessment(sorted_findings)

        # レポート生成
        report = self._generate_security_report(
            sorted_findings,
            compliance_results,
            risk_assessment,
            time.time() - start_time,
        )

        logger.info(f"セキュリティ監査完了: {len(sorted_findings)}件の発見事項")

        return report

    def _convert_pentest_results(
        self, pentest_result: Dict[str, Any]
    ) -> List[AuditFinding]:
        """ペネトレーションテスト結果を監査発見事項に変換"""
        findings = []

        for vuln_data in pentest_result.get("vulnerabilities", []):
            finding = AuditFinding(
                finding_id=f"PENTEST-{vuln_data.get('vulnerability_type', 'UNKNOWN')}",
                title=vuln_data.get("title", "Security Vulnerability"),
                description=vuln_data.get("description", ""),
                severity=vuln_data.get("severity", "medium"),
                category="penetration_testing",
                audit_scope=AuditScope.APPLICATION,
                compliance_frameworks=[ComplianceFramework.OWASP_TOP10],
                result=AuditResult.FAIL,
                evidence=[
                    vuln_data.get("evidence", ""),
                    vuln_data.get("attack_vector", ""),
                ],
                remediation_steps=[vuln_data.get("remediation", "")],
                references=[],
                risk_rating=vuln_data.get("cvss_score", 5.0),
                discovered_at=datetime.now().isoformat(),
            )

            findings.append(finding)

        return findings

    def _deduplicate_findings(self, findings: List[AuditFinding]) -> List[AuditFinding]:
        """発見事項の重複除去"""
        unique_findings = {}

        for finding in findings:
            # 重複キーの生成
            key = f"{finding.title}:{finding.category}:{finding.audit_scope.value}"
            key_hash = hash(key)

            if key_hash not in unique_findings:
                unique_findings[key_hash] = finding
            elif finding.risk_rating > unique_findings[key_hash].risk_rating:
                unique_findings[key_hash] = finding

        return list(unique_findings.values())

    def _get_severity_priority(self, severity: str) -> int:
        """深刻度の優先度値取得"""
        priority_map = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "informational": 1,
        }
        return priority_map.get(severity, 1)

    def _perform_risk_assessment(self, findings: List[AuditFinding]) -> Dict[str, Any]:
        """リスク評価実行"""
        if not findings:
            return {
                "overall_risk_score": 0.0,
                "risk_level": "MINIMAL",
                "business_impact": "LOW",
                "likelihood": "LOW",
            }

        # リスクスコア計算
        severity_weights = {
            "critical": 10,
            "high": 7,
            "medium": 4,
            "low": 2,
            "informational": 1,
        }
        total_weighted_score = sum(
            severity_weights.get(f.severity, 1) * f.risk_rating for f in findings
        )

        max_possible_score = len(findings) * 10 * 10  # max severity * max risk rating
        risk_score = (
            (total_weighted_score / max_possible_score) * 100
            if max_possible_score > 0
            else 0
        )

        # リスクレベル判定
        if risk_score >= 80:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        return {
            "overall_risk_score": round(risk_score, 1),
            "risk_level": risk_level,
            "total_findings": len(findings),
            "critical_findings": len([f for f in findings if f.severity == "critical"]),
            "high_findings": len([f for f in findings if f.severity == "high"]),
            "business_impact": self._assess_business_impact(findings),
            "technical_debt": self._assess_technical_debt(findings),
        }

    def _assess_business_impact(self, findings: List[AuditFinding]) -> str:
        """ビジネス影響度評価"""
        critical_count = len([f for f in findings if f.severity == "critical"])
        high_count = len([f for f in findings if f.severity == "high"])

        if critical_count > 0:
            return "HIGH"
        elif high_count > 5:
            return "MEDIUM"
        elif high_count > 0:
            return "LOW"
        else:
            return "MINIMAL"

    def _assess_technical_debt(self, findings: List[AuditFinding]) -> str:
        """技術的負債評価"""
        total_findings = len(findings)

        if total_findings > 50:
            return "HIGH"
        elif total_findings > 20:
            return "MEDIUM"
        elif total_findings > 5:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_security_report(
        self,
        findings: List[AuditFinding],
        compliance_results: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        execution_time: float,
    ) -> SecurityReport:
        """セキュリティレポート生成"""

        # 発見事項のカウント
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "informational": 0,
        }
        for finding in findings:
            severity_counts[finding.severity] = (
                severity_counts.get(finding.severity, 0) + 1
            )

        # エグゼクティブサマリー生成
        executive_summary = self._generate_executive_summary(findings, risk_assessment)

        # 推奨事項生成
        recommendations = self._generate_recommendations(findings)

        # 修復ロードマップ生成
        remediation_roadmap = self._generate_remediation_roadmap(findings)

        report = SecurityReport(
            report_id=f"AUDIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            audit_timestamp=datetime.now().isoformat(),
            project_name=os.path.basename(self.config.project_root),
            audit_config=self.config,
            total_findings=len(findings),
            critical_findings=severity_counts["critical"],
            high_findings=severity_counts["high"],
            medium_findings=severity_counts["medium"],
            low_findings=severity_counts["low"],
            findings=findings,
            compliance_results=compliance_results,
            risk_assessment=risk_assessment,
            executive_summary=executive_summary,
            recommendations=recommendations,
            remediation_roadmap=remediation_roadmap,
        )

        return report

    def _generate_executive_summary(
        self, findings: List[AuditFinding], risk_assessment: Dict[str, Any]
    ) -> str:
        """エグゼクティブサマリー生成"""
        total_findings = len(findings)
        critical_count = len([f for f in findings if f.severity == "critical"])
        high_count = len([f for f in findings if f.severity == "high"])
        risk_level = risk_assessment.get("risk_level", "UNKNOWN")

        summary = f"""
        セキュリティ監査により{total_findings}件の発見事項を特定しました。
        うち重要度の高いもの（Critical: {critical_count}件, High: {high_count}件）は
        優先的な対応が必要です。

        総合リスクレベル: {risk_level}
        ビジネス影響度: {risk_assessment.get("business_impact", "UNKNOWN")}

        主な課題領域は{self._get_top_categories(findings)}です。
        適切なセキュリティ対策の実装により、リスクを大幅に軽減できます。
        """.strip()

        return summary

    def _get_top_categories(self, findings: List[AuditFinding]) -> str:
        """主要なカテゴリ取得"""
        category_counts = {}
        for finding in findings:
            category_counts[finding.category] = (
                category_counts.get(finding.category, 0) + 1
            )

        top_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]
        return "、".join([cat for cat, _ in top_categories])

    def _generate_recommendations(self, findings: List[AuditFinding]) -> List[str]:
        """推奨事項生成"""
        recommendations = set()

        # 発見事項に基づく推奨事項
        for finding in findings:
            recommendations.update(
                finding.remediation_steps[:2]
            )  # 上位2つの修復ステップ

        # 一般的な推奨事項
        if findings:
            recommendations.add("定期的なセキュリティ監査の実施")
            recommendations.add("セキュリティ教育とトレーニングプログラムの実施")
            recommendations.add("インシデントレスポンス計画の策定と訓練")
            recommendations.add("セキュリティツールの自動化とCI/CD統合")

        return list(recommendations)[:10]  # 上位10件

    def _generate_remediation_roadmap(
        self, findings: List[AuditFinding]
    ) -> List[Dict[str, Any]]:
        """修復ロードマップ生成"""
        roadmap = []

        # 緊急対応（Critical/High）
        urgent_findings = [f for f in findings if f.severity in ["critical", "high"]]
        if urgent_findings:
            roadmap.append(
                {
                    "phase": "緊急対応 (0-2週間)",
                    "priority": "CRITICAL",
                    "tasks": [f.title for f in urgent_findings[:5]],
                    "estimated_effort": "高",
                    "business_justification": "セキュリティリスクの即座の軽減",
                }
            )

        # 短期対応（Medium）
        medium_findings = [f for f in findings if f.severity == "medium"]
        if medium_findings:
            roadmap.append(
                {
                    "phase": "短期対応 (2-8週間)",
                    "priority": "HIGH",
                    "tasks": [f.title for f in medium_findings[:8]],
                    "estimated_effort": "中",
                    "business_justification": "セキュリティ態勢の向上",
                }
            )

        # 長期対応（Low/Informational）
        low_findings = [f for f in findings if f.severity in ["low", "informational"]]
        if low_findings:
            roadmap.append(
                {
                    "phase": "長期対応 (8週間以降)",
                    "priority": "MEDIUM",
                    "tasks": [f.title for f in low_findings[:10]],
                    "estimated_effort": "低",
                    "business_justification": "セキュリティ成熟度の向上",
                }
            )

        return roadmap