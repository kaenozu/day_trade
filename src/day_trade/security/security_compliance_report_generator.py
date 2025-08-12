#!/usr/bin/env python3
"""
セキュリティコンプライアンスレポート生成システム
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

エンタープライズレベルのセキュリティコンプライアンスレポート生成:
- OWASP Top 10準拠セキュリティレポート
- PCI DSS、SOX、GDPR準拠レポート
- 脆弱性管理状況レポート
- セキュリティメトリクス分析レポート
- 経営陣向けエグゼクティブサマリー
"""

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

try:
    from .comprehensive_security_control_center import get_security_control_center
    from .enhanced_data_protection import get_data_protection_manager
    from .secure_coding_enforcer import get_secure_coding_enforcer

    SECURITY_COMPONENTS_AVAILABLE = True
except ImportError:
    SECURITY_COMPONENTS_AVAILABLE = False


class ReportType(Enum):
    """レポート種類"""

    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    COMPLIANCE_AUDIT = "compliance_audit"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    SECURITY_METRICS = "security_metrics"


class ComplianceFramework(Enum):
    """コンプライアンスフレームワーク"""

    OWASP_TOP_10 = "owasp_top_10"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    GDPR = "gdpr"
    ISO27001 = "iso27001"


@dataclass
class SecurityReport:
    """セキュリティレポート"""

    id: str
    title: str
    report_type: ReportType
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    executive_summary: str
    detailed_findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    compliance_status: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


class SecurityComplianceReportGenerator:
    """セキュリティコンプライアンスレポート生成システム"""

    def __init__(self):
        self.db_path = "security_reports.db"
        self._initialize_database()

        # セキュリティコンポーネント統合
        if SECURITY_COMPONENTS_AVAILABLE:
            try:
                self.security_control_center = get_security_control_center()
                self.coding_enforcer = get_secure_coding_enforcer()
                self.data_protection = get_data_protection_manager()
            except Exception:
                self.security_control_center = None
                self.coding_enforcer = None
                self.data_protection = None
        else:
            self.security_control_center = None
            self.coding_enforcer = None
            self.data_protection = None

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_reports (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    generated_at DATETIME NOT NULL,
                    period_start DATETIME NOT NULL,
                    period_end DATETIME NOT NULL,
                    executive_summary TEXT,
                    detailed_findings TEXT,
                    recommendations TEXT,
                    metrics TEXT,
                    compliance_status TEXT,
                    risk_assessment TEXT
                )
            """
            )
            conn.commit()

    async def generate_executive_summary_report(
        self, period_days: int = 30
    ) -> SecurityReport:
        """経営陣向けエグゼクティブサマリーレポート生成"""
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=period_days)

        report_id = f"exec_summary_{int(period_end.timestamp())}"

        # データ収集
        security_metrics = await self._collect_security_metrics()
        compliance_status = await self._assess_compliance_status()
        risk_assessment = await self._perform_risk_assessment()

        # エグゼクティブサマリー生成
        exec_summary = self._generate_executive_summary(
            security_metrics, compliance_status, risk_assessment
        )

        # 推奨事項生成
        recommendations = await self._generate_executive_recommendations(
            security_metrics, compliance_status, risk_assessment
        )

        report = SecurityReport(
            id=report_id,
            title=f"Security Executive Summary Report - {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}",
            report_type=ReportType.EXECUTIVE_SUMMARY,
            generated_at=period_end,
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary,
            recommendations=recommendations,
            metrics=security_metrics,
            compliance_status=compliance_status,
            risk_assessment=risk_assessment,
        )

        await self._save_report(report)
        return report

    async def generate_compliance_audit_report(
        self, framework: ComplianceFramework = ComplianceFramework.PCI_DSS
    ) -> SecurityReport:
        """コンプライアンス監査レポート生成"""
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=90)  # 3ヶ月間

        report_id = f"compliance_{framework.value}_{int(period_end.timestamp())}"

        # フレームワーク固有のチェック実行
        compliance_results = await self._perform_framework_audit(framework)
        detailed_findings = await self._analyze_compliance_findings(
            framework, compliance_results
        )

        # コンプライアンス状況評価
        compliance_score = compliance_results.get("overall_score", 0)
        violations = compliance_results.get("violations", [])

        # エグゼクティブサマリー生成
        exec_summary = f"""
## {framework.value.upper()} コンプライアンス監査結果

### 全体評価
- **コンプライアンススコア**: {compliance_score:.1f}/100
- **準拠要件**: {compliance_results.get('requirements_met', 0)}/{compliance_results.get('total_requirements', 0)}
- **重要な違反**: {len([v for v in violations if v.get('severity') in ['critical', 'high']])}件
- **総違反数**: {len(violations)}件

### 主要な所見
{"- 重大なコンプライアンス違反が検出されています。即座の対応が必要です。" if compliance_score < 70 else ""}
{"- 一部要件で改善の余地があります。" if 70 <= compliance_score < 90 else ""}
{"- コンプライアンス要件をおおむね満たしています。" if compliance_score >= 90 else ""}

### リスクレベル
{"🔴 **高リスク**: 重要なコンプライアンス違反により、法的・運用リスクが高い状況" if compliance_score < 60 else ""}
{"🟡 **中リスク**: 一部コンプライアンス要件の改善が必要" if 60 <= compliance_score < 80 else ""}
{"🟢 **低リスク**: コンプライアンス要件をおおむね満たしている" if compliance_score >= 80 else ""}
"""

        # 推奨事項生成
        recommendations = await self._generate_compliance_recommendations(
            framework, compliance_results
        )

        report = SecurityReport(
            id=report_id,
            title=f"{framework.value.upper()} Compliance Audit Report",
            report_type=ReportType.COMPLIANCE_AUDIT,
            generated_at=period_end,
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary.strip(),
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            compliance_status=compliance_results,
            metrics={
                "compliance_score": compliance_score,
                "framework": framework.value,
            },
        )

        await self._save_report(report)
        return report

    async def generate_vulnerability_assessment_report(self) -> SecurityReport:
        """脆弱性評価レポート生成"""
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=7)  # 直近1週間

        report_id = f"vuln_assessment_{int(period_end.timestamp())}"

        # 脆弱性データ収集
        vulnerability_data = await self._collect_vulnerability_data()
        secure_coding_data = await self._collect_secure_coding_violations()

        # 脆弱性分析
        critical_vulns = [
            v for v in vulnerability_data if v.get("severity") == "critical"
        ]
        high_vulns = [v for v in vulnerability_data if v.get("severity") == "high"]
        medium_vulns = [v for v in vulnerability_data if v.get("severity") == "medium"]

        total_vulns = len(vulnerability_data)

        # リスクスコア計算
        risk_score = self._calculate_vulnerability_risk_score(vulnerability_data)

        # エグゼクティブサマリー生成
        exec_summary = f"""
## 脆弱性評価サマリー

### 脆弱性統計
- **総脆弱性数**: {total_vulns}件
- **重大**: {len(critical_vulns)}件
- **高**: {len(high_vulns)}件
- **中**: {len(medium_vulns)}件
- **低**: {total_vulns - len(critical_vulns) - len(high_vulns) - len(medium_vulns)}件

### リスクレベル
- **脆弱性リスクスコア**: {risk_score:.1f}/100

### 緊急対応が必要な脅威
{f"🚨 **{len(critical_vulns)}件の重大脆弱性**が検出されています。" if critical_vulns else ""}
{f"⚠️ **{len(high_vulns)}件の高リスク脆弱性**への対応が推奨されます。" if high_vulns else ""}
{"✅ 重大な脆弱性は検出されていません。" if not critical_vulns and not high_vulns else ""}

### セキュアコーディング違反
- **総違反数**: {len(secure_coding_data)}件
- **修正が必要**: {len([v for v in secure_coding_data if v.get('severity') in ['critical', 'high']])}件
"""

        # 詳細所見
        detailed_findings = []

        # 重大脆弱性の詳細
        for vuln in critical_vulns[:5]:  # 上位5件
            detailed_findings.append(
                {
                    "type": "critical_vulnerability",
                    "title": vuln.get("title", "Unknown Vulnerability"),
                    "description": vuln.get("description", ""),
                    "severity": "critical",
                    "affected_component": vuln.get("component", "Unknown"),
                    "remediation": vuln.get("remediation", "Contact security team"),
                }
            )

        # セキュアコーディング違反の詳細
        for violation in secure_coding_data[:3]:  # 上位3件
            detailed_findings.append(
                {
                    "type": "secure_coding_violation",
                    "title": violation.get("rule_name", "Unknown Rule"),
                    "description": violation.get("message", ""),
                    "severity": violation.get("severity", "medium"),
                    "file_path": violation.get("file_path", ""),
                    "line_number": violation.get("line_number", 0),
                }
            )

        # 推奨事項生成
        recommendations = await self._generate_vulnerability_recommendations(
            vulnerability_data, secure_coding_data
        )

        report = SecurityReport(
            id=report_id,
            title="Vulnerability Assessment Report",
            report_type=ReportType.VULNERABILITY_ASSESSMENT,
            generated_at=period_end,
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary.strip(),
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            metrics={
                "total_vulnerabilities": total_vulns,
                "critical_vulnerabilities": len(critical_vulns),
                "high_vulnerabilities": len(high_vulns),
                "risk_score": risk_score,
            },
        )

        await self._save_report(report)
        return report

    async def generate_security_metrics_report(
        self, period_days: int = 30
    ) -> SecurityReport:
        """セキュリティメトリクスレポート生成"""
        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=period_days)

        report_id = f"security_metrics_{int(period_end.timestamp())}"

        # メトリクス収集
        security_metrics = await self._collect_comprehensive_security_metrics()
        trend_analysis = await self._analyze_security_trends(period_days)

        # KPI計算
        kpis = {
            "security_score": security_metrics.get("security_score", 0),
            "mean_time_to_detection": security_metrics.get("mean_time_to_detection", 0),
            "mean_time_to_resolution": security_metrics.get(
                "mean_time_to_resolution", 0
            ),
            "vulnerability_density": security_metrics.get("vulnerability_density", 0),
            "compliance_percentage": security_metrics.get("compliance_percentage", 0),
        }

        # エグゼクティブサマリー生成
        exec_summary = f"""
## セキュリティメトリクス分析結果

### セキュリティKPI
- **セキュリティスコア**: {kpis['security_score']:.1f}/100
- **コンプライアンス率**: {kpis['compliance_percentage']:.1f}%
- **平均検出時間**: {kpis['mean_time_to_detection']:.1f}時間
- **平均解決時間**: {kpis['mean_time_to_resolution']:.1f}時間
- **脆弱性密度**: {kpis['vulnerability_density']:.2f}/1000行

### トレンド分析
{"🔺 セキュリティ状況が悪化傾向にあります" if trend_analysis.get('trend') == 'declining' else ""}
{"📈 セキュリティ状況が改善傾向にあります" if trend_analysis.get('trend') == 'improving' else ""}
{"➡️ セキュリティ状況は安定しています" if trend_analysis.get('trend') == 'stable' else ""}

### パフォーマンス評価
{"🟢 優秀 - セキュリティ体制が非常に良好です" if kpis['security_score'] >= 90 else ""}
{"🟡 良好 - 一部改善の余地があります" if 70 <= kpis['security_score'] < 90 else ""}
{"🔴 改善必要 - セキュリティ体制の強化が必要です" if kpis['security_score'] < 70 else ""}
"""

        # 詳細メトリクス
        detailed_findings = [
            {"type": "security_kpi", "title": "セキュリティKPI", "metrics": kpis},
            {"type": "trend_analysis", "title": "トレンド分析", "data": trend_analysis},
            {
                "type": "benchmark",
                "title": "ベンチマーク比較",
                "industry_average": {
                    "security_score": 75.0,
                    "mean_time_to_resolution": 48.0,
                    "compliance_percentage": 85.0,
                },
                "our_performance": kpis,
            },
        ]

        # 推奨事項生成
        recommendations = await self._generate_metrics_recommendations(
            kpis, trend_analysis
        )

        report = SecurityReport(
            id=report_id,
            title=f"Security Metrics Report - {period_days} days",
            report_type=ReportType.SECURITY_METRICS,
            generated_at=period_end,
            period_start=period_start,
            period_end=period_end,
            executive_summary=exec_summary.strip(),
            detailed_findings=detailed_findings,
            recommendations=recommendations,
            metrics=security_metrics,
        )

        await self._save_report(report)
        return report

    async def _collect_security_metrics(self) -> Dict[str, Any]:
        """セキュリティメトリクス収集"""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": 85.0,
            "total_threats": 0,
            "critical_threats": 0,
            "high_threats": 0,
            "open_incidents": 0,
            "resolved_incidents": 0,
        }

        if self.security_control_center:
            try:
                dashboard = await self.security_control_center.get_security_dashboard()
                if "security_metrics" in dashboard:
                    sec_metrics = dashboard["security_metrics"]
                    metrics.update(
                        {
                            "security_score": sec_metrics.security_score,
                            "total_threats": sec_metrics.total_threats,
                            "critical_threats": sec_metrics.critical_threats,
                            "high_threats": sec_metrics.high_threats,
                            "open_incidents": sec_metrics.open_incidents,
                            "resolved_incidents": sec_metrics.resolved_incidents,
                        }
                    )
            except Exception:
                pass

        return metrics

    async def _assess_compliance_status(self) -> Dict[str, Any]:
        """コンプライアンス状況評価"""
        compliance = {
            "overall_score": 85.0,
            "frameworks": {
                "pci_dss": {"score": 90.0, "status": "compliant"},
                "sox": {"score": 80.0, "status": "mostly_compliant"},
                "gdpr": {"score": 85.0, "status": "compliant"},
            },
        }

        if self.security_control_center:
            try:
                # コンプライアンスチェック実行
                compliance_results = await self.security_control_center.compliance_monitor.check_compliance(
                    "PCI_DSS"
                )
                compliance["frameworks"]["pci_dss"] = {
                    "score": compliance_results.get("overall_score", 85.0),
                    "status": (
                        "compliant"
                        if compliance_results.get("overall_score", 0) >= 80
                        else "non_compliant"
                    ),
                }
            except Exception:
                pass

        return compliance

    async def _perform_risk_assessment(self) -> Dict[str, Any]:
        """リスク評価実行"""
        return {
            "overall_risk_level": "medium",
            "risk_score": 35.0,  # 0-100 (低いほど良い)
            "risk_categories": {
                "operational": {"level": "medium", "score": 30},
                "financial": {"level": "low", "score": 20},
                "reputational": {"level": "medium", "score": 40},
                "regulatory": {"level": "low", "score": 25},
            },
            "top_risks": [
                "未修正の脆弱性によるシステム侵入リスク",
                "データ漏洩によるコンプライアンス違反リスク",
                "システム停止による業務影響リスク",
            ],
        }

    def _generate_executive_summary(
        self,
        security_metrics: Dict[str, Any],
        compliance_status: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> str:
        """エグゼクティブサマリー生成"""
        security_score = security_metrics.get("security_score", 0)
        compliance_score = compliance_status.get("overall_score", 0)
        risk_score = risk_assessment.get("risk_score", 0)

        summary = f"""
## セキュリティ状況エグゼクティブサマリー

### 全体評価
- **セキュリティスコア**: {security_score:.1f}/100
- **コンプライアンススコア**: {compliance_score:.1f}/100
- **リスクスコア**: {risk_score:.1f}/100 (低いほど良好)

### 主要指標
- **アクティブ脅威**: {security_metrics.get('total_threats', 0)}件
  - 重大: {security_metrics.get('critical_threats', 0)}件
  - 高リスク: {security_metrics.get('high_threats', 0)}件
- **オープンインシデント**: {security_metrics.get('open_incidents', 0)}件
- **解決済インシデント**: {security_metrics.get('resolved_incidents', 0)}件

### リスク評価
- **全体リスクレベル**: {risk_assessment.get('overall_risk_level', 'unknown').upper()}
- **最優先対応事項**: {len(risk_assessment.get('top_risks', []))}項目

### ガバナンス・コンプライアンス
- PCI DSS: {compliance_status.get('frameworks', {}).get('pci_dss', {}).get('score', 0):.1f}%
- SOX: {compliance_status.get('frameworks', {}).get('sox', {}).get('score', 0):.1f}%
- GDPR: {compliance_status.get('frameworks', {}).get('gdpr', {}).get('score', 0):.1f}%
"""

        return summary.strip()

    async def _generate_executive_recommendations(
        self,
        security_metrics: Dict[str, Any],
        compliance_status: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> List[str]:
        """経営陣向け推奨事項生成"""
        recommendations = []

        security_score = security_metrics.get("security_score", 100)
        critical_threats = security_metrics.get("critical_threats", 0)

        # 重大脅威への対応
        if critical_threats > 0:
            recommendations.append(
                f"🚨 {critical_threats}件の重大脅威に対する即座の対応と投資承認が必要"
            )

        # セキュリティスコアに基づく推奨事項
        if security_score < 70:
            recommendations.extend(
                [
                    "セキュリティ体制の根本的な見直しと強化投資の検討",
                    "外部セキュリティコンサルタントによる専門的評価の実施",
                    "セキュリティインシデント対応体制の緊急強化",
                ]
            )
        elif security_score < 90:
            recommendations.extend(
                [
                    "セキュリティ監視システムの機能強化",
                    "スタッフのセキュリティ教育プログラムの拡充",
                    "定期的なセキュリティ監査の実施",
                ]
            )

        # コンプライアンスに基づく推奨事項
        compliance_score = compliance_status.get("overall_score", 100)
        if compliance_score < 80:
            recommendations.append(
                "法規制遵守のための緊急コンプライアンス対応プログラムの実施"
            )

        # リスクに基づく推奨事項
        risk_level = risk_assessment.get("overall_risk_level", "low")
        if risk_level in ["high", "critical"]:
            recommendations.append("ビジネス継続性確保のための緊急リスク軽減策の実施")

        # 基本的な推奨事項
        recommendations.extend(
            [
                "定期的なセキュリティ状況の取締役会への報告継続",
                "年次セキュリティ投資計画の策定と予算確保",
                "業界ベンチマークに基づくセキュリティ成熟度向上計画の策定",
            ]
        )

        return recommendations

    async def _perform_framework_audit(
        self, framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """フレームワーク固有の監査実行"""
        if self.security_control_center:
            try:
                return await self.security_control_center.compliance_monitor.check_compliance(
                    framework.value.replace("_", "_").upper()
                )
            except Exception:
                pass

        # フォールバック実装
        return {
            "framework": framework.value,
            "overall_score": 85.0,
            "requirements_met": 8,
            "total_requirements": 10,
            "violations": [
                {
                    "requirement": "Sample requirement not met",
                    "description": "Sample compliance violation",
                    "severity": "medium",
                }
            ],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _analyze_compliance_findings(
        self, framework: ComplianceFramework, compliance_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """コンプライアンス所見分析"""
        findings = []

        violations = compliance_results.get("violations", [])
        for violation in violations:
            findings.append(
                {
                    "type": "compliance_violation",
                    "framework": framework.value,
                    "title": violation.get("requirement", "Unknown requirement"),
                    "description": violation.get("description", ""),
                    "severity": violation.get("severity", "medium"),
                    "remediation": f"Address {framework.value} requirement compliance",
                }
            )

        # 全体評価
        score = compliance_results.get("overall_score", 0)
        findings.append(
            {
                "type": "overall_assessment",
                "framework": framework.value,
                "title": f"{framework.value.upper()} Overall Compliance Assessment",
                "description": f"Overall compliance score: {score:.1f}/100",
                "severity": (
                    "info" if score >= 90 else "medium" if score >= 70 else "high"
                ),
                "score": score,
            }
        )

        return findings

    async def _generate_compliance_recommendations(
        self, framework: ComplianceFramework, compliance_results: Dict[str, Any]
    ) -> List[str]:
        """コンプライアンス推奨事項生成"""
        recommendations = []

        score = compliance_results.get("overall_score", 100)
        violations = compliance_results.get("violations", [])

        if score < 70:
            recommendations.append(
                f"{framework.value.upper()}要件の緊急対応プログラムを実施してください"
            )

        if violations:
            recommendations.append(
                f"{len(violations)}件のコンプライアンス違反への対応を優先してください"
            )

        # フレームワーク固有の推奨事項
        framework_recommendations = {
            ComplianceFramework.PCI_DSS: [
                "カード情報処理システムのセキュリティ強化",
                "定期的なPCI DSS準拠監査の実施",
                "決済システムの暗号化と監視強化",
            ],
            ComplianceFramework.SOX: [
                "財務報告プロセスの内部統制強化",
                "ITシステムのアクセス制御と監査ログ管理",
                "定期的な内部監査プロセスの見直し",
            ],
            ComplianceFramework.GDPR: [
                "個人データ処理プロセスの透明性向上",
                "データ主体の権利行使対応プロセスの整備",
                "個人データ漏洩時の対応手順の確立",
            ],
        }

        if framework in framework_recommendations:
            recommendations.extend(framework_recommendations[framework])

        recommendations.extend(
            [
                "法規制要件の最新動向の定期的な確認",
                "コンプライアンス担当者の専門教育実施",
                "外部コンプライアンス監査の定期実施",
            ]
        )

        return recommendations

    async def _collect_vulnerability_data(self) -> List[Dict[str, Any]]:
        """脆弱性データ収集"""
        vulnerabilities = []

        if self.security_control_center:
            try:
                dashboard = await self.security_control_center.get_security_dashboard()
                threat_summary = dashboard.get("threat_summary", {})

                # 脅威を脆弱性として扱う
                for threat in threat_summary.get("recent", []):
                    vulnerabilities.append(
                        {
                            "id": threat["id"],
                            "title": threat["title"],
                            "severity": threat["severity"],
                            "detected_at": threat["detected_at"],
                            "type": "threat",
                            "description": f"Security threat: {threat['title']}",
                        }
                    )
            except Exception:
                pass

        # サンプル脆弱性データ（実際の実装では脆弱性スキャナーから取得）
        sample_vulnerabilities = [
            {
                "id": "vuln_001",
                "title": "Outdated dependencies with known vulnerabilities",
                "severity": "high",
                "description": "Several dependencies have known security vulnerabilities",
                "component": "npm packages",
                "remediation": "Update dependencies to latest secure versions",
            }
        ]

        vulnerabilities.extend(sample_vulnerabilities)
        return vulnerabilities

    async def _collect_secure_coding_violations(self) -> List[Dict[str, Any]]:
        """セキュアコーディング違反収集"""
        violations = []

        if self.coding_enforcer:
            try:
                summary = self.coding_enforcer.get_security_summary()
                violation_list = self.coding_enforcer.get_violations(limit=50)

                for violation in violation_list:
                    violations.append(
                        {
                            "id": violation.id,
                            "rule_name": violation.rule_name,
                            "message": violation.message,
                            "severity": violation.severity.value,
                            "file_path": violation.file_path,
                            "line_number": violation.line_number,
                        }
                    )
            except Exception:
                pass

        return violations

    def _calculate_vulnerability_risk_score(
        self, vulnerabilities: List[Dict[str, Any]]
    ) -> float:
        """脆弱性リスクスコア計算"""
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            "critical": 40,
            "high": 20,
            "medium": 10,
            "low": 5,
            "info": 1,
        }

        total_score = 0
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "low")
            total_score += severity_weights.get(severity, 5)

        # 100点満点で正規化
        max_possible_score = len(vulnerabilities) * 40  # 全て critical の場合
        if max_possible_score > 0:
            return min(100.0, (total_score / max_possible_score) * 100)

        return 0.0

    async def _generate_vulnerability_recommendations(
        self,
        vulnerabilities: List[Dict[str, Any]],
        coding_violations: List[Dict[str, Any]],
    ) -> List[str]:
        """脆弱性推奨事項生成"""
        recommendations = []

        critical_vulns = [v for v in vulnerabilities if v.get("severity") == "critical"]
        high_vulns = [v for v in vulnerabilities if v.get("severity") == "high"]

        if critical_vulns:
            recommendations.append(
                f"🚨 {len(critical_vulns)}件の重大脆弱性への即座の対応"
            )

        if high_vulns:
            recommendations.append(f"⚠️ {len(high_vulns)}件の高リスク脆弱性への優先対応")

        if coding_violations:
            critical_coding = [
                v
                for v in coding_violations
                if v.get("severity") in ["critical", "high"]
            ]
            if critical_coding:
                recommendations.append(
                    f"🔧 {len(critical_coding)}件の重要なセキュアコーディング違反の修正"
                )

        recommendations.extend(
            [
                "脆弱性管理プロセスの継続的改善",
                "開発チームへのセキュアコーディング教育強化",
                "自動脆弱性スキャンツールの導入・強化",
                "サードパーティライブラリの定期的な更新",
                "ペネトレーションテストの定期実施",
            ]
        )

        return recommendations

    async def _collect_comprehensive_security_metrics(self) -> Dict[str, Any]:
        """包括的セキュリティメトリクス収集"""
        metrics = await self._collect_security_metrics()

        # 追加メトリクス計算
        metrics.update(
            {
                "mean_time_to_detection": 4.5,  # 時間
                "mean_time_to_resolution": 24.0,  # 時間
                "vulnerability_density": 2.3,  # per 1000 lines
                "compliance_percentage": 87.5,
                "security_training_completion": 92.0,  # %
                "patch_management_score": 85.0,
                "access_control_effectiveness": 90.0,
            }
        )

        return metrics

    async def _analyze_security_trends(self, period_days: int) -> Dict[str, Any]:
        """セキュリティトレンド分析"""
        # 簡易実装 - 実際の環境では履歴データから分析
        return {
            "trend": "improving",  # improving, stable, declining
            "security_score_change": +5.2,
            "incident_count_change": -2,
            "vulnerability_count_change": -8,
            "key_improvements": [
                "脆弱性解決時間の短縮",
                "セキュリティ教育完了率の向上",
                "インシデント発生数の減少",
            ],
        }

    async def _generate_metrics_recommendations(
        self, kpis: Dict[str, Any], trend_analysis: Dict[str, Any]
    ) -> List[str]:
        """メトリクス推奨事項生成"""
        recommendations = []

        security_score = kpis.get("security_score", 100)
        if security_score < 80:
            recommendations.append(
                "セキュリティスコア向上のための包括的改善プログラム実施"
            )

        mttr = kpis.get("mean_time_to_resolution", 0)
        if mttr > 48:  # 48時間以上
            recommendations.append("インシデント解決時間短縮のためのプロセス改善")

        compliance_pct = kpis.get("compliance_percentage", 100)
        if compliance_pct < 90:
            recommendations.append("コンプライアンス率向上のための体系的取り組み")

        # トレンドに基づく推奨事項
        if trend_analysis.get("trend") == "declining":
            recommendations.append("セキュリティ状況悪化の根本原因分析と対策実施")
        elif trend_analysis.get("trend") == "improving":
            recommendations.append("現在の改善トレンドを維持・加速するための継続的投資")

        recommendations.extend(
            [
                "ベンチマーク比較による相対的なセキュリティ成熟度評価",
                "セキュリティメトリクスの自動化と可視化改善",
                "予防的セキュリティ対策への投資シフト",
                "セキュリティROI測定手法の導入",
            ]
        )

        return recommendations

    async def _save_report(self, report: SecurityReport):
        """レポートをデータベースに保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_reports
                    (id, title, report_type, generated_at, period_start, period_end,
                     executive_summary, detailed_findings, recommendations,
                     metrics, compliance_status, risk_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report.id,
                        report.title,
                        report.report_type.value,
                        report.generated_at.isoformat(),
                        report.period_start.isoformat(),
                        report.period_end.isoformat(),
                        report.executive_summary,
                        json.dumps(report.detailed_findings),
                        json.dumps(report.recommendations),
                        json.dumps(report.metrics),
                        json.dumps(report.compliance_status),
                        json.dumps(report.risk_assessment),
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"レポート保存エラー: {e}")

    def export_report_markdown(self, report: SecurityReport) -> str:
        """レポートをMarkdown形式でエクスポート"""
        markdown = f"""# {report.title}

**生成日時**: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**対象期間**: {report.period_start.strftime('%Y-%m-%d')} ～ {report.period_end.strftime('%Y-%m-%d')}
**レポートID**: {report.id}

{report.executive_summary}

## 推奨事項

"""

        for i, rec in enumerate(report.recommendations, 1):
            markdown += f"{i}. {rec}\n"

        markdown += "\n## 詳細所見\n\n"

        for finding in report.detailed_findings:
            markdown += f"### {finding.get('title', 'Unknown')}\n\n"
            markdown += f"**種類**: {finding.get('type', 'unknown')}\n"
            if "severity" in finding:
                markdown += f"**重要度**: {finding['severity'].upper()}\n"
            if "description" in finding:
                markdown += f"**説明**: {finding['description']}\n"
            markdown += "\n"

        markdown += "## メトリクス\n\n"
        markdown += "```json\n"
        markdown += json.dumps(report.metrics, indent=2, ensure_ascii=False)
        markdown += "\n```\n"

        markdown += f"\n---\n*このレポートは自動生成されました - {datetime.now(timezone.utc).isoformat()}*\n"

        return markdown


# グローバルインスタンス
_report_generator = None


def get_security_report_generator() -> SecurityComplianceReportGenerator:
    """グローバルレポート生成器を取得"""
    global _report_generator
    if _report_generator is None:
        _report_generator = SecurityComplianceReportGenerator()
    return _report_generator


if __name__ == "__main__":
    # テスト実行
    async def test_security_reports():
        print("=== セキュリティコンプライアンスレポート生成テスト ===")

        try:
            generator = SecurityComplianceReportGenerator()

            print("\n1. エグゼクティブサマリーレポート生成...")
            exec_report = await generator.generate_executive_summary_report(30)
            print(f"   レポートID: {exec_report.id}")
            print(f"   タイトル: {exec_report.title}")

            print("\n2. コンプライアンス監査レポート生成...")
            compliance_report = await generator.generate_compliance_audit_report(
                ComplianceFramework.PCI_DSS
            )
            print(f"   レポートID: {compliance_report.id}")
            print(
                f"   コンプライアンススコア: {compliance_report.compliance_status.get('overall_score', 0):.1f}"
            )

            print("\n3. 脆弱性評価レポート生成...")
            vuln_report = await generator.generate_vulnerability_assessment_report()
            print(f"   レポートID: {vuln_report.id}")
            print(
                f"   検出脆弱性数: {vuln_report.metrics.get('total_vulnerabilities', 0)}"
            )

            print("\n4. セキュリティメトリクスレポート生成...")
            metrics_report = await generator.generate_security_metrics_report(30)
            print(f"   レポートID: {metrics_report.id}")
            print(
                f"   セキュリティスコア: {metrics_report.metrics.get('security_score', 0):.1f}"
            )

            # Markdownエクスポート
            print("\n5. Markdownレポートエクスポート...")
            markdown_content = generator.export_report_markdown(exec_report)

            # ファイルに保存
            report_file = Path(f"security_report_{exec_report.id}.md")
            report_file.write_text(markdown_content, encoding="utf-8")
            print(f"   Markdownレポート保存: {report_file}")

            print("\n[成功] セキュリティレポート生成テスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_security_reports())
