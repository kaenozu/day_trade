#!/usr/bin/env python3
"""
セキュリティ管制センター - セキュリティスキャナー
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from .enums import SecurityLevel, ThreatCategory
from .models import SecurityThreat

try:
    import psutil
    MONITORING_LIBS_AVAILABLE = True
except ImportError:
    MONITORING_LIBS_AVAILABLE = False


class SecurityScanner:
    """セキュリティスキャナー"""

    def __init__(self, threat_intelligence_engine):
        self.logger = logging.getLogger(__name__)
        self.threat_intelligence = threat_intelligence_engine

    async def run_comprehensive_scan(
        self, vulnerability_manager=None, coding_enforcer=None
    ) -> Dict[str, Any]:
        """包括的セキュリティスキャン"""
        scan_start = time.time()
        scan_results = {
            "scan_id": f"security_scan_{int(scan_start)}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "components_scanned": [],
            "threats_detected": [],
            "scan_successful": True,
            "threats": [],
        }

        try:
            self.logger.info("包括的セキュリティスキャン開始")

            # 1. 脆弱性スキャン
            if vulnerability_manager:
                try:
                    vuln_results = (
                        await vulnerability_manager.run_comprehensive_scan(".")
                    )
                    scan_results["components_scanned"].append("vulnerability_scan")

                    # 脆弱性を脅威として登録
                    for vuln in vuln_results.get("vulnerabilities", []):
                        threat = SecurityThreat(
                            id=f"vuln_{vuln.get('id', int(time.time()))}",
                            title=f"脆弱性検出: {vuln.get('title', 'Unknown')}",
                            description=vuln.get("description", ""),
                            category=ThreatCategory.VULNERABILITY,
                            severity=SecurityLevel(
                                vuln.get("severity", "medium").lower()
                            ),
                            source="vulnerability_manager",
                            indicators=vuln,
                        )
                        scan_results["threats"].append(threat)
                        scan_results["threats_detected"].append(threat.id)

                except Exception as e:
                    self.logger.error(f"脆弱性スキャンエラー: {e}")

            # 2. セキュアコーディングチェック
            if coding_enforcer:
                try:
                    coding_results = coding_enforcer.scan_directory("src", [".py"])
                    scan_results["components_scanned"].append("secure_coding_check")

                    # セキュリティ違反を脅威として登録
                    for violation in coding_results.get("violations", []):
                        if violation.severity.value in ["critical", "high"]:
                            threat = SecurityThreat(
                                id=f"coding_{violation.id}",
                                title=f"コーディング違反: {violation.rule_name}",
                                description=violation.message,
                                category=ThreatCategory.POLICY_VIOLATION,
                                severity=SecurityLevel(violation.severity.value),
                                source="coding_enforcer",
                                affected_assets=[violation.file_path],
                                indicators={
                                    "line": violation.line_number,
                                    "code": violation.code_snippet,
                                },
                            )
                            scan_results["threats"].append(threat)
                            scan_results["threats_detected"].append(threat.id)

                except Exception as e:
                    self.logger.error(f"セキュアコーディングチェックエラー: {e}")

            # 3. システム状態監視
            try:
                system_threats = await self._scan_system_state()
                scan_results["components_scanned"].append("system_monitoring")

                for threat in system_threats:
                    scan_results["threats"].append(threat)
                    scan_results["threats_detected"].append(threat.id)

            except Exception as e:
                self.logger.error(f"システム監視エラー: {e}")

            scan_duration = time.time() - scan_start
            scan_results["completed_at"] = datetime.now(timezone.utc).isoformat()
            scan_results["scan_duration"] = scan_duration

            self.logger.info(f"包括的セキュリティスキャン完了 ({scan_duration:.2f}秒)")

            return scan_results

        except Exception as e:
            self.logger.error(f"セキュリティスキャンエラー: {e}")
            scan_results["scan_successful"] = False
            scan_results["error"] = str(e)
            return scan_results

    async def _scan_system_state(self) -> List[SecurityThreat]:
        """システム状態スキャン"""
        threats = []

        try:
            if MONITORING_LIBS_AVAILABLE:
                import psutil

                # CPU使用率チェック
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    threat = SecurityThreat(
                        id=f"high_cpu_{int(time.time())}",
                        title="異常なCPU使用率",
                        description=f"CPU使用率が異常に高い状態: {cpu_percent}%",
                        category=ThreatCategory.ANOMALY,
                        severity=SecurityLevel.MEDIUM,
                        source="system_monitor",
                        indicators={"cpu_percent": cpu_percent},
                    )
                    threats.append(threat)

                # メモリ使用率チェック
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    threat = SecurityThreat(
                        id=f"high_memory_{int(time.time())}",
                        title="異常なメモリ使用率",
                        description=f"メモリ使用率が異常に高い状態: {memory.percent}%",
                        category=ThreatCategory.ANOMALY,
                        severity=SecurityLevel.MEDIUM,
                        source="system_monitor",
                        indicators={"memory_percent": memory.percent},
                    )
                    threats.append(threat)

                # 不審なプロセス検出
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    try:
                        process_info = proc.info
                        process_threats = (
                            await self.threat_intelligence._analyze_process_patterns(
                                [process_info]
                            )
                        )
                        threats.extend(process_threats)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

        except Exception as e:
            self.logger.error(f"システム状態スキャンエラー: {e}")

        return threats