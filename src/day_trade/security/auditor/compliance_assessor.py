#!/usr/bin/env python3
"""
コンプライアンス評価
"""

from typing import Any, Dict, List

from .enums import AuditFinding, ComplianceFramework

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class ComplianceAssessor:
    """コンプライアンス評価"""

    def __init__(self, frameworks: List[ComplianceFramework]):
        self.frameworks = frameworks

    def assess_compliance(self, findings: List[AuditFinding]) -> Dict[str, Any]:
        """コンプライアンス評価実行"""
        compliance_results = {}

        for framework in self.frameworks:
            assessment = self._assess_framework_compliance(framework, findings)
            compliance_results[framework.value] = assessment

        return compliance_results

    def _assess_framework_compliance(
        self, framework: ComplianceFramework, findings: List[AuditFinding]
    ) -> Dict[str, Any]:
        """フレームワーク別コンプライアンス評価"""

        relevant_findings = [
            f for f in findings if framework in f.compliance_frameworks
        ]

        critical_count = len([f for f in relevant_findings if f.severity == "critical"])
        high_count = len([f for f in relevant_findings if f.severity == "high"])
        medium_count = len([f for f in relevant_findings if f.severity == "medium"])
        low_count = len([f for f in relevant_findings if f.severity == "low"])

        # フレームワーク別のコンプライアンス基準
        compliance_criteria = self._get_compliance_criteria(framework)

        # コンプライアンススコア計算
        score = self._calculate_compliance_score(
            critical_count, high_count, medium_count, low_count, compliance_criteria
        )

        compliant = score >= compliance_criteria.get("minimum_score", 70)

        return {
            "framework": framework.value,
            "compliant": compliant,
            "score": score,
            "findings_breakdown": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "total": len(relevant_findings),
            },
            "requirements_met": self._get_met_requirements(framework, findings),
            "recommendations": self._get_framework_recommendations(framework, findings),
        }

    def _get_compliance_criteria(
        self, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """コンプライアンス基準取得"""
        criteria = {
            ComplianceFramework.NIST_CSF: {
                "minimum_score": 75,
                "max_critical": 0,
                "max_high": 5,
                "description": "NIST Cybersecurity Framework",
            },
            ComplianceFramework.ISO27001: {
                "minimum_score": 80,
                "max_critical": 0,
                "max_high": 3,
                "description": "ISO/IEC 27001 Information Security Management",
            },
            ComplianceFramework.OWASP_TOP10: {
                "minimum_score": 70,
                "max_critical": 0,
                "max_high": 10,
                "description": "OWASP Top 10 Web Application Security Risks",
            },
            ComplianceFramework.PCI_DSS: {
                "minimum_score": 85,
                "max_critical": 0,
                "max_high": 2,
                "description": "Payment Card Industry Data Security Standard",
            },
            ComplianceFramework.SOC2: {
                "minimum_score": 75,
                "max_critical": 0,
                "max_high": 5,
                "description": "Service Organization Control 2",
            },
        }

        return criteria.get(
            framework, {"minimum_score": 70, "max_critical": 0, "max_high": 10}
        )

    def _calculate_compliance_score(
        self, critical: int, high: int, medium: int, low: int, criteria: Dict[str, Any]
    ) -> float:
        """コンプライアンススコア計算"""

        # 基本スコア
        base_score = 100.0

        # 減点
        critical_penalty = critical * 25  # クリティカル1件あたり25点減点
        high_penalty = high * 10  # 高1件あたり10点減点
        medium_penalty = medium * 3  # 中1件あたり3点減点
        low_penalty = low * 1  # 低1件あたり1点減点

        total_penalty = critical_penalty + high_penalty + medium_penalty + low_penalty
        score = max(0.0, base_score - total_penalty)

        return round(score, 1)

    def _get_met_requirements(
        self, framework: ComplianceFramework, findings: List[AuditFinding]
    ) -> List[str]:
        """満たされた要件リスト"""
        # 簡略実装 - 実際はより詳細な要件マッピングが必要
        met_requirements = []

        finding_categories = set(
            f.category for f in findings if framework in f.compliance_frameworks
        )

        if "static_analysis" not in finding_categories:
            met_requirements.append("Static Code Analysis Implemented")
        if "dependency_vulnerability" not in finding_categories:
            met_requirements.append("Dependency Vulnerability Management")

        return met_requirements

    def _get_framework_recommendations(
        self, framework: ComplianceFramework, findings: List[AuditFinding]
    ) -> List[str]:
        """フレームワーク固有の推奨事項"""
        recommendations = []

        relevant_findings = [
            f for f in findings if framework in f.compliance_frameworks
        ]
        critical_high = [
            f for f in relevant_findings if f.severity in ["critical", "high"]
        ]

        if critical_high:
            recommendations.append(
                f"Address {len(critical_high)} critical/high severity findings immediately"
            )

        if framework == ComplianceFramework.OWASP_TOP10:
            recommendations.extend(
                [
                    "Implement secure coding practices",
                    "Regular security training for development team",
                    "Automated security testing in CI/CD pipeline",
                ]
            )
        elif framework == ComplianceFramework.NIST_CSF:
            recommendations.extend(
                [
                    "Establish cybersecurity governance framework",
                    "Implement continuous monitoring capabilities",
                    "Develop incident response procedures",
                ]
            )
        elif framework == ComplianceFramework.ISO27001:
            recommendations.extend(
                [
                    "Document information security management system",
                    "Conduct regular risk assessments",
                    "Implement security awareness program",
                ]
            )

        return recommendations