#!/usr/bin/env python3
"""
コードセキュリティ分析
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from .enums import (
    AuditFinding,
    AuditResult,
    AuditScope,
    ComplianceFramework,
)

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


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
                        severity_map = {
                            "HIGH": "high",
                            "MEDIUM": "medium",
                            "LOW": "low",
                        }

                        severity = severity_map.get(
                            issue.get("issue_severity", "MEDIUM"), "medium"
                        )
                        confidence = issue.get("issue_confidence", "MEDIUM")

                        # 信頼度が低い場合は重要度を下げる
                        if confidence == "LOW" and severity != "low":
                            severity = "medium" if severity == "high" else "low"

                        finding = AuditFinding(
                            finding_id=f"BANDIT-{issue.get('test_id', 'UNKNOWN')}",
                            title=issue.get("test_name", "Security Issue"),
                            description=issue.get(
                                "issue_text", "Security vulnerability detected"
                            ),
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
                            risk_rating=self._calculate_bandit_risk_rating(
                                severity, confidence
                            ),
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
                            references=[
                                f"https://pyup.io/vulnerabilities/{vulnerability_id}/"
                            ],
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
                            risk_rating=self._get_custom_rule_risk_rating(
                                rule["severity"]
                            ),
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