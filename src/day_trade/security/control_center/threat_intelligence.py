#!/usr/bin/env python3
"""
セキュリティ管制センター - 脅威インテリジェンス分析エンジン
"""

import logging
import time
from typing import Any, Dict, List

from .enums import SecurityLevel, ThreatCategory
from .models import SecurityThreat


class ThreatIntelligenceEngine:
    """脅威インテリジェンス分析エンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 脅威パターンデータベース
        self.threat_patterns = {
            "suspicious_file_operations": {
                "pattern": r"(rm -rf|del /s|format|fdisk)",
                "severity": SecurityLevel.HIGH,
                "category": ThreatCategory.MALWARE,
            },
            "network_scanning": {
                "pattern": r"(nmap|masscan|zmap|unicornscan)",
                "severity": SecurityLevel.MEDIUM,
                "category": ThreatCategory.INTRUSION,
            },
            "credential_harvesting": {
                "pattern": r"(mimikatz|lazagne|password.*dump|credential.*extract)",
                "severity": SecurityLevel.CRITICAL,
                "category": ThreatCategory.DATA_BREACH,
            },
            "sql_injection_attempt": {
                "pattern": r"(union.*select|drop.*table|exec.*xp_cmdshell)",
                "severity": SecurityLevel.HIGH,
                "category": ThreatCategory.INTRUSION,
            },
        }

    async def analyze_threat_indicators(
        self, data: Dict[str, Any]
    ) -> List[SecurityThreat]:
        """脅威インジケーターを分析"""
        threats = []

        try:
            # ログデータの分析
            if "logs" in data:
                log_threats = await self._analyze_log_patterns(data["logs"])
                threats.extend(log_threats)

            # ネットワークトラフィックの分析
            if "network_traffic" in data:
                network_threats = await self._analyze_network_patterns(
                    data["network_traffic"]
                )
                threats.extend(network_threats)

            # ファイルシステム変更の分析
            if "file_changes" in data:
                file_threats = await self._analyze_file_patterns(data["file_changes"])
                threats.extend(file_threats)

            # プロセス活動の分析
            if "processes" in data:
                process_threats = await self._analyze_process_patterns(
                    data["processes"]
                )
                threats.extend(process_threats)

        except Exception as e:
            self.logger.error(f"脅威分析エラー: {e}")

        return threats

    async def _analyze_log_patterns(self, logs: List[str]) -> List[SecurityThreat]:
        """ログパターンを分析"""
        threats = []

        for log_entry in logs:
            for pattern_name, pattern_info in self.threat_patterns.items():
                import re

                if re.search(pattern_info["pattern"], log_entry, re.IGNORECASE):
                    threat = SecurityThreat(
                        id=f"log_threat_{int(time.time())}_{hash(log_entry) % 10000}",
                        title=f"ログで検出された脅威: {pattern_name}",
                        description=f"ログエントリで suspicious pattern detected: {log_entry[:100]}",
                        category=pattern_info["category"],
                        severity=pattern_info["severity"],
                        source="log_analysis",
                        indicators={"log_entry": log_entry, "pattern": pattern_name},
                        metadata={"detection_method": "pattern_matching"},
                    )
                    threats.append(threat)

        return threats

    async def _analyze_network_patterns(
        self, network_data: Dict[str, Any]
    ) -> List[SecurityThreat]:
        """ネットワークパターンを分析"""
        threats = []

        # 異常な接続数
        if network_data.get("connection_count", 0) > 1000:
            threat = SecurityThreat(
                id=f"network_anomaly_{int(time.time())}",
                title="異常なネットワーク接続数",
                description=f"通常以上の接続数を検出: {network_data['connection_count']}",
                category=ThreatCategory.ANOMALY,
                severity=SecurityLevel.MEDIUM,
                source="network_monitor",
                indicators=network_data,
            )
            threats.append(threat)

        # 不審なポートスキャン
        if network_data.get("port_scan_detected", False):
            threat = SecurityThreat(
                id=f"port_scan_{int(time.time())}",
                title="ポートスキャン検出",
                description="システムに対するポートスキャンが検出されました",
                category=ThreatCategory.INTRUSION,
                severity=SecurityLevel.HIGH,
                source="network_monitor",
                indicators=network_data,
            )
            threats.append(threat)

        return threats

    async def _analyze_file_patterns(
        self, file_changes: List[Dict[str, Any]]
    ) -> List[SecurityThreat]:
        """ファイル変更パターンを分析"""
        threats = []

        # 重要ファイルの変更
        critical_paths = [
            "/etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32",
            "/bin",
            "/sbin",
        ]

        for change in file_changes:
            file_path = change.get("path", "")

            if any(critical_path in file_path for critical_path in critical_paths):
                threat = SecurityThreat(
                    id=f"file_threat_{int(time.time())}_{hash(file_path) % 10000}",
                    title="重要ファイルの変更検出",
                    description=f"システム重要ファイルが変更されました: {file_path}",
                    category=ThreatCategory.MALWARE,
                    severity=SecurityLevel.HIGH,
                    source="file_monitor",
                    affected_assets=[file_path],
                    indicators=change,
                )
                threats.append(threat)

        return threats

    async def _analyze_process_patterns(
        self, processes: List[Dict[str, Any]]
    ) -> List[SecurityThreat]:
        """プロセス活動パターンを分析"""
        threats = []

        # 不審なプロセス名
        suspicious_processes = [
            "mimikatz",
            "psexec",
            "nc.exe",
            "netcat",
            "powershell -enc",
        ]

        for process in processes:
            process_name = process.get("name", "").lower()
            command_line = process.get("cmdline", "").lower()

            for suspicious in suspicious_processes:
                if suspicious in process_name or suspicious in command_line:
                    threat = SecurityThreat(
                        id=f"process_threat_{process.get('pid', int(time.time()))}",
                        title="不審なプロセス検出",
                        description=f"不審なプロセスが検出されました: {process_name}",
                        category=ThreatCategory.MALWARE,
                        severity=SecurityLevel.CRITICAL,
                        source="process_monitor",
                        indicators=process,
                    )
                    threats.append(threat)

        return threats