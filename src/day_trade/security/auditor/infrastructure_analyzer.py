#!/usr/bin/env python3
"""
インフラストラクチャセキュリティ分析
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List

from .enums import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    AuditScope,
    ComplianceFramework,
)

try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


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
                            evidence=[
                                f"Container: {container.name}",
                                "Privileged: true",
                            ],
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
                                evidence=[
                                    f"File: {file_path}",
                                    f"Permissions: {permissions}",
                                ],
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
                    logger.debug(
                        f"ファイルパーミッションチェックエラー {file_path}: {e}"
                    )

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
            if any(
                pattern in f"{var_name}={var_value}"
                for pattern in dangerous_env_patterns
            ):
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