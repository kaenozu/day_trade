#!/usr/bin/env python3
"""
統合セキュリティ監査システム
Issue #435: 本番環境セキュリティ最終監査 - エンタープライズレベル保証

包括的セキュリティ評価・コンプライアンス監査・リスク分析
"""

import asyncio
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
    from .penetration_tester import (
        PenetrationTester,
        PenTestConfig,
        SecurityVulnerability,
    )
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)

    # フォールバック
    class SecurityVulnerability:
        pass

    class PenetrationTester:
        pass


logger = get_context_logger(__name__)


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


class CodeSecurityAnalyzer:
    """コードセキュリティ分析"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)

    async def analyze_code_security(self) -> List[AuditFinding]:
        """コードセキュリティ分析"""
        findings = []

        # Bandit (Python セキュリティ分析)
        bandit_findings = await self._run_bandit_analysis()
        findings.extend(bandit_findings)

        # Safety (依存関係脆弱性チェック)
        safety_findings = await self._run_safety_analysis()
        findings.extend(safety_findings)

        # カスタムセキュリティルール
        custom_findings = await self._run_custom_security_rules()
        findings.extend(custom_findings)

        return findings

    async def _run_bandit_analysis(self) -> List[AuditFinding]:
        """Bandit 静的解析実行"""
        findings = []

        try:
            cmd = [
                "bandit",
                "-r",
                str(self.project_root),
                "-f",
                "json",
                "--skip",
                "B101,B601",  # assert_used, shell=True の一部除外
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode in [0, 1]:  # 0=OK, 1=問題発見
                try:
                    bandit_data = json.loads(result.stdout)

                    for issue in bandit_data.get("results", []):
                        severity_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}

                        severity = severity_map.get(issue.get("issue_severity", "MEDIUM"), "medium")
                        confidence = issue.get("issue_confidence", "MEDIUM")

                        # 信頼度が低い場合は重要度を下げる
                        if confidence == "LOW" and severity != "low":
                            severity = "medium" if severity == "high" else "low"

                        finding = AuditFinding(
                            finding_id=f"BANDIT-{issue.get('test_id', 'UNKNOWN')}",
                            title=issue.get("test_name", "Security Issue"),
                            description=issue.get("issue_text", "Security vulnerability detected"),
                            severity=severity,
                            category="static_analysis",
                            audit_scope=AuditScope.CODE_ANALYSIS,
                            compliance_frameworks=[ComplianceFramework.OWASP_TOP10],
                            result=AuditResult.FAIL,
                            evidence=[
                                f"File: {issue.get('filename', 'unknown')}",
                                f"Line: {issue.get('line_number', 0)}",
                                f"Code: {issue.get('code', '')[:200]}...",
                            ],
                            remediation_steps=[
                                "Review the flagged code for security vulnerabilities",
                                "Apply secure coding practices",
                                "Consider using safer alternatives",
                            ],
                            references=[
                                f"https://bandit.readthedocs.io/en/latest/plugins/{issue.get('test_id', '').lower()}.html"
                            ],
                            risk_rating=self._calculate_bandit_risk_rating(severity, confidence),
                            discovered_at=datetime.now().isoformat(),
                        )

                        findings.append(finding)

                except json.JSONDecodeError:
                    logger.warning("Bandit JSON出力の解析に失敗")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Bandit実行エラー: {e}")

        return findings

    async def _run_safety_analysis(self) -> List[AuditFinding]:
        """Safety 依存関係脆弱性チェック"""
        findings = []

        try:
            cmd = ["safety", "check", "--json", "--full-report"]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, cwd=self.project_root
            )

            if result.returncode == 255:  # 脆弱性発見
                try:
                    safety_data = json.loads(result.stdout)

                    for vuln in safety_data.get("vulnerabilities", []):
                        package_name = vuln.get("package", "unknown")
                        vulnerability_id = vuln.get("vulnerability_id", "")

                        finding = AuditFinding(
                            finding_id=f"SAFETY-{vulnerability_id}",
                            title=f"Vulnerable dependency: {package_name}",
                            description=vuln.get(
                                "advisory", "Vulnerable package dependency detected"
                            ),
                            severity="high",  # 依存関係の脆弱性は基本的に重要
                            category="dependency_vulnerability",
                            audit_scope=AuditScope.CODE_ANALYSIS,
                            compliance_frameworks=[
                                ComplianceFramework.NIST_CSF,
                                ComplianceFramework.OWASP_TOP10,
                            ],
                            result=AuditResult.FAIL,
                            evidence=[
                                f"Package: {package_name}",
                                f"Installed version: {vuln.get('installed_version', 'unknown')}",
                                f"Vulnerability ID: {vulnerability_id}",
                                f"Advisory: {vuln.get('advisory', '')[:200]}...",
                            ],
                            remediation_steps=[
                                f"Update {package_name} to a safe version",
                                "Review security advisories for the package",
                                "Consider alternative packages if updates are not available",
                            ],
                            references=[f"https://pyup.io/vulnerabilities/{vulnerability_id}/"],
                            risk_rating=8.0,  # 依存関係の脆弱性は高リスク
                            discovered_at=datetime.now().isoformat(),
                        )

                        findings.append(finding)

                except json.JSONDecodeError:
                    logger.warning("Safety JSON出力の解析に失敗")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Safety実行エラー: {e}")

        return findings

    async def _run_custom_security_rules(self) -> List[AuditFinding]:
        """カスタムセキュリティルールチェック"""
        findings = []

        # セキュリティ関連パターンチェック
        security_patterns = [
            {
                "pattern": r"password\s*=\s*['\"][^'\"]*['\"]",
                "title": "Hardcoded Password",
                "severity": "high",
                "description": "Hardcoded password found in source code",
            },
            {
                "pattern": r"api[_-]?key\s*=\s*['\"][^'\"]*['\"]",
                "title": "Hardcoded API Key",
                "severity": "high",
                "description": "Hardcoded API key found in source code",
            },
            {
                "pattern": r"secret\s*=\s*['\"][^'\"]*['\"]",
                "title": "Hardcoded Secret",
                "severity": "medium",
                "description": "Hardcoded secret found in source code",
            },
            {
                "pattern": r"eval\s*\(",
                "title": "Use of eval()",
                "severity": "high",
                "description": "Dangerous use of eval() function",
            },
            {
                "pattern": r"exec\s*\(",
                "title": "Use of exec()",
                "severity": "high",
                "description": "Dangerous use of exec() function",
            },
        ]

        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")

                for rule in security_patterns:
                    import re

                    matches = list(re.finditer(rule["pattern"], content, re.IGNORECASE))

                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1

                        finding = AuditFinding(
                            finding_id=f"CUSTOM-{rule['title'].replace(' ', '_').upper()}-{hash(str(py_file))}",
                            title=rule["title"],
                            description=rule["description"],
                            severity=rule["severity"],
                            category="custom_security_rule",
                            audit_scope=AuditScope.CODE_ANALYSIS,
                            compliance_frameworks=[ComplianceFramework.OWASP_TOP10],
                            result=AuditResult.FAIL,
                            evidence=[
                                f"File: {py_file}",
                                f"Line: {line_num}",
                                f"Pattern: {rule['pattern']}",
                                f"Match: {match.group()[:100]}...",
                            ],
                            remediation_steps=[
                                "Remove or secure the identified security issue",
                                "Use environment variables or secure vaults for secrets",
                                "Avoid dangerous functions like eval() and exec()",
                            ],
                            references=[],
                            risk_rating=self._get_custom_rule_risk_rating(rule["severity"]),
                            discovered_at=datetime.now().isoformat(),
                        )

                        findings.append(finding)

            except Exception as e:
                logger.debug(f"カスタムルールチェックエラー {py_file}: {e}")

        return findings

    def _calculate_bandit_risk_rating(self, severity: str, confidence: str) -> float:
        """Bandit リスクレーティング計算"""
        severity_scores = {"high": 8.0, "medium": 5.0, "low": 2.0}
        confidence_multipliers = {"HIGH": 1.0, "MEDIUM": 0.8, "LOW": 0.5}

        base_score = severity_scores.get(severity, 2.0)
        multiplier = confidence_multipliers.get(confidence, 0.8)

        return min(10.0, base_score * multiplier)

    def _get_custom_rule_risk_rating(self, severity: str) -> float:
        """カスタムルール リスクレーティング"""
        severity_scores = {"critical": 10.0, "high": 8.0, "medium": 5.0, "low": 2.0}
        return severity_scores.get(severity, 2.0)


class InfrastructureSecurityAnalyzer:
    """インフラストラクチャセキュリティ分析"""

    def __init__(self, config: AuditConfig):
        self.config = config

    async def analyze_infrastructure_security(self) -> List[AuditFinding]:
        """インフラストラクチャセキュリティ分析"""
        findings = []

        # Dockerコンテナセキュリティ
        if DOCKER_AVAILABLE and self.config.enable_container_scanning:
            container_findings = await self._analyze_docker_security()
            findings.extend(container_findings)

        # ファイルシステムパーミッション
        filesystem_findings = await self._analyze_filesystem_security()
        findings.extend(filesystem_findings)

        # 環境変数セキュリティ
        env_findings = await self._analyze_environment_security()
        findings.extend(env_findings)

        return findings

    async def _analyze_docker_security(self) -> List[AuditFinding]:
        """Dockerセキュリティ分析"""
        findings = []

        try:
            client = docker.from_env()
            containers = client.containers.list(all=True)

            for container in containers:
                # 特権モードチェック
                if container.attrs.get("HostConfig", {}).get("Privileged", False):
                    findings.append(
                        AuditFinding(
                            finding_id=f"DOCKER-PRIVILEGED-{container.short_id}",
                            title="Docker Container Running in Privileged Mode",
                            description=f"Container {container.name} is running in privileged mode",
                            severity="high",
                            category="container_security",
                            audit_scope=AuditScope.INFRASTRUCTURE,
                            compliance_frameworks=[ComplianceFramework.CIS_CONTROLS],
                            result=AuditResult.FAIL,
                            evidence=[f"Container: {container.name}", "Privileged: true"],
                            remediation_steps=[
                                "Remove --privileged flag from container",
                                "Use specific capabilities instead of privileged mode",
                                "Review container security requirements",
                            ],
                            references=[],
                            risk_rating=8.0,
                            discovered_at=datetime.now().isoformat(),
                        )
                    )

                # ルートユーザーチェック
                config = container.attrs.get("Config", {})
                if config.get("User", "") in ["", "0", "root"]:
                    findings.append(
                        AuditFinding(
                            finding_id=f"DOCKER-ROOT-USER-{container.short_id}",
                            title="Docker Container Running as Root",
                            description=f"Container {container.name} is running as root user",
                            severity="medium",
                            category="container_security",
                            audit_scope=AuditScope.INFRASTRUCTURE,
                            compliance_frameworks=[ComplianceFramework.CIS_CONTROLS],
                            result=AuditResult.FAIL,
                            evidence=[
                                f"Container: {container.name}",
                                f"User: {config.get('User', 'root')}",
                            ],
                            remediation_steps=[
                                "Create and use non-root user in container",
                                "Update Dockerfile to use USER directive",
                                "Test application functionality with non-root user",
                            ],
                            references=[],
                            risk_rating=5.0,
                            discovered_at=datetime.now().isoformat(),
                        )
                    )

        except Exception as e:
            logger.warning(f"Docker分析エラー: {e}")

        return findings

    async def _analyze_filesystem_security(self) -> List[AuditFinding]:
        """ファイルシステムセキュリティ分析"""
        findings = []

        # 危険なファイルパーミッション
        sensitive_files = [
            ".env",
            "config.json",
            "settings.py",
            "secrets.json",
            "database.conf",
        ]

        project_root = Path(self.config.project_root)

        for pattern in sensitive_files:
            for file_path in project_root.rglob(pattern):
                try:
                    stat_info = file_path.stat()
                    permissions = oct(stat_info.st_mode)[-3:]

                    # 777, 666 などの危険なパーミッション
                    if permissions in ["777", "776", "775", "666"]:
                        findings.append(
                            AuditFinding(
                                finding_id=f"FS-PERMS-{hash(str(file_path))}",
                                title="Insecure File Permissions",
                                description=f"File {file_path.name} has insecure permissions",
                                severity="medium",
                                category="filesystem_security",
                                audit_scope=AuditScope.INFRASTRUCTURE,
                                compliance_frameworks=[ComplianceFramework.NIST_CSF],
                                result=AuditResult.FAIL,
                                evidence=[f"File: {file_path}", f"Permissions: {permissions}"],
                                remediation_steps=[
                                    "Set appropriate file permissions (e.g., 640 or 600)",
                                    "Remove world-writable permissions",
                                    "Review file access requirements",
                                ],
                                references=[],
                                risk_rating=4.0,
                                discovered_at=datetime.now().isoformat(),
                            )
                        )

                except Exception as e:
                    logger.debug(f"ファイルパーミッションチェックエラー {file_path}: {e}")

        return findings

    async def _analyze_environment_security(self) -> List[AuditFinding]:
        """環境変数セキュリティ分析"""
        findings = []

        # 危険な環境変数パターン
        dangerous_env_patterns = [
            "DEBUG=True",
            "DEBUG=1",
            "DEVELOPMENT=true",
            "TEST_MODE=true",
        ]

        for var_name, var_value in os.environ.items():
            # 本番環境での危険な設定
            if any(pattern in f"{var_name}={var_value}" for pattern in dangerous_env_patterns):
                findings.append(
                    AuditFinding(
                        finding_id=f"ENV-{var_name}",
                        title="Insecure Environment Configuration",
                        description=f"Potentially insecure environment variable: {var_name}",
                        severity="medium",
                        category="environment_security",
                        audit_scope=AuditScope.INFRASTRUCTURE,
                        compliance_frameworks=[ComplianceFramework.NIST_CSF],
                        result=AuditResult.FAIL,
                        evidence=[f"Variable: {var_name}={var_value}"],
                        remediation_steps=[
                            "Review environment variable settings for production",
                            "Disable debug mode in production",
                            "Use separate configurations for different environments",
                        ],
                        references=[],
                        risk_rating=3.0,
                        discovered_at=datetime.now().isoformat(),
                    )
                )

        return findings


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

        relevant_findings = [f for f in findings if framework in f.compliance_frameworks]

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

    def _get_compliance_criteria(self, framework: ComplianceFramework) -> Dict[str, Any]:
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

        return criteria.get(framework, {"minimum_score": 70, "max_critical": 0, "max_high": 10})

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

        relevant_findings = [f for f in findings if framework in f.compliance_frameworks]
        critical_high = [f for f in relevant_findings if f.severity in ["critical", "high"]]

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
                pen_config = PenTestConfig(
                    target_base_url=url,
                    test_timeout=300,
                    include_aggressive_tests=False,
                )
                pen_tester = PenetrationTester(pen_config)
                pen_test_tasks.append(pen_tester.run_comprehensive_pentest())

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
            sorted_findings, compliance_results, risk_assessment, time.time() - start_time
        )

        logger.info(f"セキュリティ監査完了: {len(sorted_findings)}件の発見事項")

        return report

    def _convert_pentest_results(self, pentest_result: Dict[str, Any]) -> List[AuditFinding]:
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
                evidence=[vuln_data.get("evidence", ""), vuln_data.get("attack_vector", "")],
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
        priority_map = {"critical": 5, "high": 4, "medium": 3, "low": 2, "informational": 1}
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
        severity_weights = {"critical": 10, "high": 7, "medium": 4, "low": 2, "informational": 1}
        total_weighted_score = sum(
            severity_weights.get(f.severity, 1) * f.risk_rating for f in findings
        )

        max_possible_score = len(findings) * 10 * 10  # max severity * max risk rating
        risk_score = (
            (total_weighted_score / max_possible_score) * 100 if max_possible_score > 0 else 0
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
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
        for finding in findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

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
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1

        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        return "、".join([cat for cat, _ in top_categories])

    def _generate_recommendations(self, findings: List[AuditFinding]) -> List[str]:
        """推奨事項生成"""
        recommendations = set()

        # 発見事項に基づく推奨事項
        for finding in findings:
            recommendations.update(finding.remediation_steps[:2])  # 上位2つの修復ステップ

        # 一般的な推奨事項
        if findings:
            recommendations.add("定期的なセキュリティ監査の実施")
            recommendations.add("セキュリティ教育とトレーニングプログラムの実施")
            recommendations.add("インシデントレスポンス計画の策定と訓練")
            recommendations.add("セキュリティツールの自動化とCI/CD統合")

        return list(recommendations)[:10]  # 上位10件

    def _generate_remediation_roadmap(self, findings: List[AuditFinding]) -> List[Dict[str, Any]]:
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


if __name__ == "__main__":
    # セキュリティ監査デモ
    async def main():
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

    # 実行
    asyncio.run(main())
