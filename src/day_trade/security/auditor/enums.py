#!/usr/bin/env python3
"""
セキュリティ監査システムの列挙型とデータクラス
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List


class ComplianceFramework(Enum):
    """コンプライアンスフレームワーク"""

    NIST_CSF = "nist_csf"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    FINRA = "finra"
    JSOX = "jsox"
    CIS_CONTROLS = "cis_controls"
    OWASP_TOP10 = "owasp_top10"


class AuditScope(Enum):
    """監査スコープ"""

    CODE_ANALYSIS = "code_analysis"
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    NETWORK = "network"
    DATA_PROTECTION = "data_protection"
    ACCESS_CONTROL = "access_control"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    THIRD_PARTY_RISK = "third_party_risk"
    COMPLIANCE = "compliance"


class AuditResult(Enum):
    """監査結果"""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class AuditConfig:
    """セキュリティ監査設定"""

    # 監査対象
    project_root: str
    target_urls: List[str] = None

    # 監査スコープ
    audit_scopes: List[AuditScope] = None
    compliance_frameworks: List[ComplianceFramework] = None

    # 実行設定
    enable_penetration_testing: bool = True
    enable_code_analysis: bool = True
    enable_dependency_scanning: bool = True
    enable_container_scanning: bool = True
    enable_infrastructure_scanning: bool = False

    # 出力設定
    output_format: str = "json"  # json, html, pdf
    detailed_report: bool = True
    include_remediation: bool = True

    # 並列処理
    max_concurrent_scans: int = 3
    scan_timeout: int = 1800  # 30分

    def __post_init__(self):
        if self.target_urls is None:
            self.target_urls = ["http://localhost:8000"]
        if self.audit_scopes is None:
            self.audit_scopes = [
                AuditScope.CODE_ANALYSIS,
                AuditScope.APPLICATION,
                AuditScope.NETWORK,
                AuditScope.DATA_PROTECTION,
            ]
        if self.compliance_frameworks is None:
            self.compliance_frameworks = [
                ComplianceFramework.NIST_CSF,
                ComplianceFramework.OWASP_TOP10,
                ComplianceFramework.ISO27001,
            ]


@dataclass
class AuditFinding:
    """監査発見事項"""

    finding_id: str
    title: str
    description: str
    severity: str  # critical, high, medium, low, informational
    category: str
    audit_scope: AuditScope
    compliance_frameworks: List[ComplianceFramework]
    result: AuditResult
    evidence: List[str]
    remediation_steps: List[str]
    references: List[str]
    risk_rating: float  # 0-10
    discovered_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityReport:
    """セキュリティレポート"""

    report_id: str
    audit_timestamp: str
    project_name: str
    audit_config: AuditConfig

    # サマリー
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int

    # 結果詳細
    findings: List[AuditFinding]
    compliance_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]

    # 推奨事項
    executive_summary: str
    recommendations: List[str]
    remediation_roadmap: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)