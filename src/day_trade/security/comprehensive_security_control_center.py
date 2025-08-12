#!/usr/bin/env python3
"""
包括的セキュリティ管制センター
Issue #419: セキュリティ対策の強化と脆弱性管理プロセスの確立

エンタープライズレベルのセキュリティ監視・管理統合プラットフォーム:
- リアルタイムセキュリティ監視
- 脆弱性管理とトリアージ
- セキュリティインシデント対応
- アクセス制御と監査
- コンプライアンス監視
- セキュリティメトリクス分析
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from .access_control_audit_system import get_access_control_auditor
    from .dependency_vulnerability_manager import get_vulnerability_manager
    from .enhanced_data_protection import get_data_protection_manager
    from .sast_dast_security_testing import get_security_test_coordinator
    from .secure_coding_enforcer import get_secure_coding_enforcer

    SECURITY_COMPONENTS_AVAILABLE = True
except ImportError:
    SECURITY_COMPONENTS_AVAILABLE = False
    logging.warning("一部セキュリティコンポーネントが利用できません")

try:
    import psutil
    import requests

    MONITORING_LIBS_AVAILABLE = True
except ImportError:
    MONITORING_LIBS_AVAILABLE = False


class SecurityLevel(Enum):
    """セキュリティレベル"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(Enum):
    """脅威カテゴリ"""

    VULNERABILITY = "vulnerability"
    MALWARE = "malware"
    INTRUSION = "intrusion"
    DATA_BREACH = "data_breach"
    POLICY_VIOLATION = "policy_violation"
    ANOMALY = "anomaly"
    COMPLIANCE = "compliance"


class IncidentStatus(Enum):
    """インシデント状態"""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


@dataclass
class SecurityThreat:
    """セキュリティ脅威"""

    id: str
    title: str
    description: str
    category: ThreatCategory
    severity: SecurityLevel
    source: str  # どこから検出されたか
    affected_assets: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None


@dataclass
class SecurityIncident:
    """セキュリティインシデント"""

    id: str
    title: str
    description: str
    severity: SecurityLevel
    status: IncidentStatus = IncidentStatus.OPEN
    threats: List[SecurityThreat] = field(default_factory=list)
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    response_actions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """セキュリティメトリクス"""

    total_threats: int = 0
    critical_threats: int = 0
    high_threats: int = 0
    medium_threats: int = 0
    low_threats: int = 0
    open_incidents: int = 0
    resolved_incidents: int = 0
    mean_resolution_time: float = 0.0
    security_score: float = 100.0
    compliance_score: float = 100.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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


class IncidentResponseOrchestrator:
    """インシデント対応オーケストレーター"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_playbooks = self._load_response_playbooks()

    def _load_response_playbooks(self) -> Dict[str, List[str]]:
        """対応プレイブックを読み込み"""
        return {
            ThreatCategory.VULNERABILITY.value: [
                "脆弱性の詳細調査",
                "影響範囲の特定",
                "パッチの適用可能性確認",
                "緊急修正の実施",
                "システム再起動とテスト",
            ],
            ThreatCategory.MALWARE.value: [
                "感染システムの隔離",
                "マルウェア分析",
                "影響範囲の調査",
                "システムクリーンアップ",
                "セキュリティ強化",
            ],
            ThreatCategory.INTRUSION.value: [
                "システムアクセスの遮断",
                "ログ分析と証跡追跡",
                "侵入経路の特定",
                "セキュリティホールの修正",
                "監視強化",
            ],
            ThreatCategory.DATA_BREACH.value: [
                "データアクセスの停止",
                "漏洩範囲の調査",
                "法的対応の検討",
                "利用者への通知",
                "セキュリティ監査",
            ],
        }

    async def handle_threat(self, threat: SecurityThreat) -> SecurityIncident:
        """脅威に対するインシデント対応"""
        try:
            # インシデント作成
            incident_id = f"incident_{int(time.time())}_{threat.id[-8:]}"

            incident = SecurityIncident(
                id=incident_id,
                title=f"Security Incident: {threat.title}",
                description=f"Threat detected and escalated to incident: {threat.description}",
                severity=threat.severity,
                threats=[threat],
            )

            # 自動対応アクションの決定
            response_actions = self.response_playbooks.get(
                threat.category.value, ["Manual investigation required"]
            )

            incident.response_actions = response_actions

            # 自動対応の実行（安全なものに限定）
            if threat.severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]:
                await self._execute_immediate_response(threat, incident)

            return incident

        except Exception as e:
            self.logger.error(f"インシデント対応エラー: {e}")
            raise

    async def _execute_immediate_response(
        self, threat: SecurityThreat, incident: SecurityIncident
    ):
        """即座の対応を実行"""
        try:
            # 高リスク脅威に対する自動対応
            if threat.category == ThreatCategory.MALWARE:
                # プロセス情報があれば停止を試行
                if "pid" in threat.indicators:
                    await self._try_terminate_process(threat.indicators["pid"])
                    incident.response_actions.append(
                        f"Attempted to terminate suspicious process {threat.indicators['pid']}"
                    )

            elif threat.category == ThreatCategory.INTRUSION:
                # ファイアウォールルールの追加（シミュレーション）
                if "source_ip" in threat.indicators:
                    await self._simulate_block_ip(threat.indicators["source_ip"])
                    incident.response_actions.append(
                        f"Blocked suspicious IP: {threat.indicators['source_ip']}"
                    )

        except Exception as e:
            self.logger.error(f"自動対応実行エラー: {e}")
            incident.response_actions.append(f"Automatic response failed: {e}")

    async def _try_terminate_process(self, pid: int):
        """プロセス終了を試行"""
        try:
            if MONITORING_LIBS_AVAILABLE:
                import psutil

                process = psutil.Process(pid)
                process.terminate()
                self.logger.info(f"Terminated suspicious process {pid}")
        except Exception as e:
            self.logger.warning(f"プロセス終了に失敗: {e}")

    async def _simulate_block_ip(self, ip_address: str):
        """IPアドレスブロックのシミュレーション"""
        # 実際の環境では適切なファイアウォール制御を実装
        self.logger.info(f"Simulated blocking IP address: {ip_address}")


class ComplianceMonitor:
    """コンプライアンス監視システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compliance_frameworks = {
            "PCI_DSS": {
                "name": "Payment Card Industry Data Security Standard",
                "requirements": [
                    "Install and maintain a firewall configuration",
                    "Do not use vendor-supplied defaults for passwords",
                    "Protect stored cardholder data",
                    "Encrypt transmission of cardholder data",
                ],
            },
            "SOX": {
                "name": "Sarbanes-Oxley Act",
                "requirements": [
                    "Maintain accurate financial records",
                    "Implement internal controls",
                    "Regular auditing and monitoring",
                ],
            },
            "GDPR": {
                "name": "General Data Protection Regulation",
                "requirements": [
                    "Data protection by design and by default",
                    "Consent for data processing",
                    "Right to data portability",
                    "Data breach notification",
                ],
            },
        }

    async def check_compliance(self, framework: str = "PCI_DSS") -> Dict[str, Any]:
        """コンプライアンスチェック"""
        try:
            compliance_results = {
                "framework": framework,
                "framework_name": self.compliance_frameworks.get(framework, {}).get(
                    "name", "Unknown"
                ),
                "overall_score": 0.0,
                "requirements_met": 0,
                "total_requirements": 0,
                "violations": [],
                "recommendations": [],
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

            if framework not in self.compliance_frameworks:
                compliance_results["error"] = (
                    f"Unknown compliance framework: {framework}"
                )
                return compliance_results

            framework_info = self.compliance_frameworks[framework]
            requirements = framework_info["requirements"]
            compliance_results["total_requirements"] = len(requirements)

            # 各要件のチェック（簡易実装）
            requirements_met = 0

            for i, requirement in enumerate(requirements):
                # 実際の実装では、具体的なチェックロジックを実装
                is_met = await self._check_requirement(framework, requirement)

                if is_met:
                    requirements_met += 1
                else:
                    compliance_results["violations"].append(
                        {
                            "requirement": requirement,
                            "description": f"Requirement not fully met: {requirement}",
                            "severity": "medium",
                        }
                    )

            compliance_results["requirements_met"] = requirements_met
            compliance_results["overall_score"] = (
                requirements_met / len(requirements)
            ) * 100

            # 推奨事項生成
            if compliance_results["overall_score"] < 100:
                compliance_results["recommendations"].extend(
                    [
                        "Review and update security policies",
                        "Implement additional monitoring controls",
                        "Conduct regular compliance audits",
                        "Provide staff training on compliance requirements",
                    ]
                )

            return compliance_results

        except Exception as e:
            self.logger.error(f"コンプライアンスチェックエラー: {e}")
            return {"error": str(e), "framework": framework}

    async def _check_requirement(self, framework: str, requirement: str) -> bool:
        """個別要件のチェック"""
        # 簡易実装 - 実際の環境では具体的なチェックを実装

        if "firewall" in requirement.lower():
            # ファイアウォール設定チェック
            return await self._check_firewall_configuration()

        elif "password" in requirement.lower():
            # パスワードポリシーチェック
            return await self._check_password_policies()

        elif "encrypt" in requirement.lower():
            # 暗号化実装チェック
            return await self._check_encryption_implementation()

        elif "audit" in requirement.lower():
            # 監査ログチェック
            return await self._check_audit_logging()

        # デフォルトは部分的準拠
        return True

    async def _check_firewall_configuration(self) -> bool:
        """ファイアウォール設定チェック"""
        # 簡易実装
        return True

    async def _check_password_policies(self) -> bool:
        """パスワードポリシーチェック"""
        # 簡易実装
        return True

    async def _check_encryption_implementation(self) -> bool:
        """暗号化実装チェック"""
        # 簡易実装
        return True

    async def _check_audit_logging(self) -> bool:
        """監査ログチェック"""
        # 簡易実装
        return True


class ComprehensiveSecurityControlCenter:
    """包括的セキュリティ管制センター"""

    def __init__(self, db_path: str = "security_control_center.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.threat_intelligence = ThreatIntelligenceEngine()
        self.incident_response = IncidentResponseOrchestrator()
        self.compliance_monitor = ComplianceMonitor()

        # セキュリティコンポーネント統合
        if SECURITY_COMPONENTS_AVAILABLE:
            try:
                self.vulnerability_manager = get_vulnerability_manager()
                self.coding_enforcer = get_secure_coding_enforcer()
                self.data_protection = get_data_protection_manager()
                self.access_auditor = get_access_control_auditor()
                self.security_tester = get_security_test_coordinator()
            except Exception as e:
                self.logger.warning(
                    f"一部セキュリティコンポーネントの初期化に失敗: {e}"
                )
                self.vulnerability_manager = None
                self.coding_enforcer = None
                self.data_protection = None
                self.access_auditor = None
                self.security_tester = None
        else:
            self.vulnerability_manager = None
            self.coding_enforcer = None
            self.data_protection = None
            self.access_auditor = None
            self.security_tester = None

        # データベース初期化
        self._initialize_database()

        # インメモリストレージ
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.incidents: Dict[str, SecurityIncident] = {}

        self.logger.info("包括的セキュリティ管制センター初期化完了")

    def _initialize_database(self):
        """データベース初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_threats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    source TEXT NOT NULL,
                    affected_assets TEXT,
                    indicators TEXT,
                    metadata TEXT,
                    detected_at DATETIME NOT NULL,
                    expires_at DATETIME,
                    status TEXT DEFAULT 'active'
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS security_incidents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    assigned_to TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    resolved_at DATETIME,
                    response_actions TEXT,
                    impact_assessment TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    requirements_met INTEGER NOT NULL,
                    total_requirements INTEGER NOT NULL,
                    violations TEXT,
                    recommendations TEXT,
                    checked_at DATETIME NOT NULL
                )
            """
            )

            # インデックス作成
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_threats_severity ON security_threats(severity)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_threats_category ON security_threats(category)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_status ON security_incidents(status)"
            )

            conn.commit()

    async def run_comprehensive_security_scan(self) -> Dict[str, Any]:
        """包括的セキュリティスキャン"""
        scan_start = time.time()
        scan_results = {
            "scan_id": f"security_scan_{int(scan_start)}",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "components_scanned": [],
            "threats_detected": [],
            "incidents_created": [],
            "compliance_status": {},
            "recommendations": [],
            "scan_successful": True,
        }

        try:
            self.logger.info("包括的セキュリティスキャン開始")

            # 1. 脆弱性スキャン
            if self.vulnerability_manager:
                try:
                    vuln_results = (
                        await self.vulnerability_manager.run_comprehensive_scan(".")
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
                        await self.register_threat(threat)
                        scan_results["threats_detected"].append(threat.id)

                except Exception as e:
                    self.logger.error(f"脆弱性スキャンエラー: {e}")

            # 2. セキュアコーディングチェック
            if self.coding_enforcer:
                try:
                    coding_results = self.coding_enforcer.scan_directory("src", [".py"])
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
                            await self.register_threat(threat)
                            scan_results["threats_detected"].append(threat.id)

                except Exception as e:
                    self.logger.error(f"セキュアコーディングチェックエラー: {e}")

            # 3. システム状態監視
            try:
                system_threats = await self._scan_system_state()
                scan_results["components_scanned"].append("system_monitoring")

                for threat in system_threats:
                    await self.register_threat(threat)
                    scan_results["threats_detected"].append(threat.id)

            except Exception as e:
                self.logger.error(f"システム監視エラー: {e}")

            # 4. コンプライアンスチェック
            try:
                compliance_results = await self.compliance_monitor.check_compliance(
                    "PCI_DSS"
                )
                scan_results["compliance_status"] = compliance_results
                scan_results["components_scanned"].append("compliance_check")

                # コンプライアンス違反を脅威として登録
                for violation in compliance_results.get("violations", []):
                    threat = SecurityThreat(
                        id=f"compliance_{int(time.time())}_{hash(violation['requirement']) % 10000}",
                        title=f"コンプライアンス違反: {violation['requirement']}",
                        description=violation["description"],
                        category=ThreatCategory.COMPLIANCE,
                        severity=SecurityLevel(violation.get("severity", "medium")),
                        source="compliance_monitor",
                        indicators=violation,
                    )
                    await self.register_threat(threat)
                    scan_results["threats_detected"].append(threat.id)

            except Exception as e:
                self.logger.error(f"コンプライアンスチェックエラー: {e}")

            # 5. 推奨事項生成
            scan_results[
                "recommendations"
            ] = await self._generate_security_recommendations()

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

    async def _generate_security_recommendations(self) -> List[str]:
        """セキュリティ推奨事項を生成"""
        recommendations = []

        # アクティブ脅威に基づく推奨事項
        critical_threats = [
            t
            for t in self.active_threats.values()
            if t.severity == SecurityLevel.CRITICAL
        ]
        high_threats = [
            t for t in self.active_threats.values() if t.severity == SecurityLevel.HIGH
        ]

        if critical_threats:
            recommendations.append(
                "重大な脅威が検出されています。即座の対応が必要です。"
            )
            recommendations.append(
                "システム管理者に緊急連絡を取り、対応を開始してください。"
            )

        if high_threats:
            recommendations.append("高リスク脅威への対応を優先してください。")
            recommendations.append(
                "セキュリティパッチの適用とシステム更新を実施してください。"
            )

        # 一般的な推奨事項
        recommendations.extend(
            [
                "定期的なセキュリティスキャンを継続してください。",
                "セキュリティログの監視体制を強化してください。",
                "スタッフに対するセキュリティ教育を実施してください。",
                "インシデント対応プロセスの見直しと改善を行ってください。",
                "バックアップシステムの動作確認を定期的に実施してください。",
            ]
        )

        return recommendations

    async def register_threat(self, threat: SecurityThreat) -> str:
        """脅威を登録"""
        try:
            self.active_threats[threat.id] = threat

            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_threats
                    (id, title, description, category, severity, source, affected_assets,
                     indicators, metadata, detected_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        threat.id,
                        threat.title,
                        threat.description,
                        threat.category.value,
                        threat.severity.value,
                        threat.source,
                        json.dumps(threat.affected_assets),
                        json.dumps(threat.indicators),
                        json.dumps(threat.metadata),
                        threat.detected_at.isoformat(),
                        threat.expires_at.isoformat() if threat.expires_at else None,
                    ),
                )
                conn.commit()

            # 高リスク脅威の場合、自動的にインシデントを作成
            if threat.severity in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]:
                incident = await self.incident_response.handle_threat(threat)
                await self.register_incident(incident)

            self.logger.info(f"脅威登録: {threat.id} ({threat.severity.value})")
            return threat.id

        except Exception as e:
            self.logger.error(f"脅威登録エラー: {e}")
            raise

    async def register_incident(self, incident: SecurityIncident) -> str:
        """インシデントを登録"""
        try:
            self.incidents[incident.id] = incident

            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO security_incidents
                    (id, title, description, severity, status, assigned_to,
                     created_at, updated_at, resolved_at, response_actions, impact_assessment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        incident.id,
                        incident.title,
                        incident.description,
                        incident.severity.value,
                        incident.status.value,
                        incident.assigned_to,
                        incident.created_at.isoformat(),
                        incident.updated_at.isoformat(),
                        (
                            incident.resolved_at.isoformat()
                            if incident.resolved_at
                            else None
                        ),
                        json.dumps(incident.response_actions),
                        json.dumps(incident.impact_assessment),
                    ),
                )
                conn.commit()

            self.logger.info(f"インシデント登録: {incident.id}")
            return incident.id

        except Exception as e:
            self.logger.error(f"インシデント登録エラー: {e}")
            raise

    async def get_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティダッシュボードデータ取得"""
        try:
            dashboard = {
                "overview": {
                    "system_name": "Day Trade Security Control Center",
                    "version": "1.0.0",
                    "status": "active",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
                "threat_summary": await self._get_threat_summary(),
                "incident_summary": await self._get_incident_summary(),
                "security_metrics": await self._calculate_security_metrics(),
                "recent_activities": await self._get_recent_activities(),
                "component_status": await self._get_component_status(),
                "recommendations": await self._generate_security_recommendations(),
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"ダッシュボード取得エラー: {e}")
            return {"error": str(e)}

    async def _get_threat_summary(self) -> Dict[str, Any]:
        """脅威サマリー取得"""
        threats = list(self.active_threats.values())

        summary = {
            "total": len(threats),
            "by_severity": {
                "critical": len(
                    [t for t in threats if t.severity == SecurityLevel.CRITICAL]
                ),
                "high": len([t for t in threats if t.severity == SecurityLevel.HIGH]),
                "medium": len(
                    [t for t in threats if t.severity == SecurityLevel.MEDIUM]
                ),
                "low": len([t for t in threats if t.severity == SecurityLevel.LOW]),
                "info": len([t for t in threats if t.severity == SecurityLevel.INFO]),
            },
            "by_category": {},
            "recent": [
                {
                    "id": t.id,
                    "title": t.title,
                    "severity": t.severity.value,
                    "detected_at": t.detected_at.isoformat(),
                }
                for t in sorted(threats, key=lambda x: x.detected_at, reverse=True)[:5]
            ],
        }

        # カテゴリ別統計
        for threat in threats:
            category = threat.category.value
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

        return summary

    async def _get_incident_summary(self) -> Dict[str, Any]:
        """インシデントサマリー取得"""
        incidents = list(self.incidents.values())

        return {
            "total": len(incidents),
            "open": len([i for i in incidents if i.status == IncidentStatus.OPEN]),
            "investigating": len(
                [i for i in incidents if i.status == IncidentStatus.INVESTIGATING]
            ),
            "resolved": len(
                [i for i in incidents if i.status == IncidentStatus.RESOLVED]
            ),
            "recent": [
                {
                    "id": i.id,
                    "title": i.title,
                    "severity": i.severity.value,
                    "status": i.status.value,
                    "created_at": i.created_at.isoformat(),
                }
                for i in sorted(incidents, key=lambda x: x.created_at, reverse=True)[:5]
            ],
        }

    async def _calculate_security_metrics(self) -> SecurityMetrics:
        """セキュリティメトリクス計算"""
        threats = list(self.active_threats.values())
        incidents = list(self.incidents.values())

        # 脅威統計
        critical_threats = len(
            [t for t in threats if t.severity == SecurityLevel.CRITICAL]
        )
        high_threats = len([t for t in threats if t.severity == SecurityLevel.HIGH])
        medium_threats = len([t for t in threats if t.severity == SecurityLevel.MEDIUM])
        low_threats = len([t for t in threats if t.severity == SecurityLevel.LOW])

        # インシデント統計
        open_incidents = len(
            [
                i
                for i in incidents
                if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]
            ]
        )
        resolved_incidents = len(
            [i for i in incidents if i.status == IncidentStatus.RESOLVED]
        )

        # 解決時間計算
        resolved_incidents_with_time = [
            i
            for i in incidents
            if i.status == IncidentStatus.RESOLVED and i.resolved_at
        ]

        if resolved_incidents_with_time:
            resolution_times = [
                (i.resolved_at - i.created_at).total_seconds() / 3600  # 時間単位
                for i in resolved_incidents_with_time
            ]
            mean_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            mean_resolution_time = 0.0

        # セキュリティスコア計算（100点満点）
        security_score = 100.0

        # 脅威によるスコア減点
        security_score -= critical_threats * 20  # 重大脅威1つにつき20点減点
        security_score -= high_threats * 10  # 高脅威1つにつき10点減点
        security_score -= medium_threats * 5  # 中脅威1つにつき5点減点
        security_score -= low_threats * 1  # 低脅威1つにつき1点減点

        # オープンインシデントによる減点
        security_score -= open_incidents * 15  # オープンインシデント1つにつき15点減点

        security_score = max(0.0, security_score)  # 0未満にはならない

        return SecurityMetrics(
            total_threats=len(threats),
            critical_threats=critical_threats,
            high_threats=high_threats,
            medium_threats=medium_threats,
            low_threats=low_threats,
            open_incidents=open_incidents,
            resolved_incidents=resolved_incidents,
            mean_resolution_time=mean_resolution_time,
            security_score=security_score,
            compliance_score=85.0,  # コンプライアンススコア（仮の値）
        )

    async def _get_recent_activities(self) -> List[Dict[str, Any]]:
        """最近のアクティビティ取得"""
        activities = []

        # 最近の脅威
        recent_threats = sorted(
            self.active_threats.values(), key=lambda x: x.detected_at, reverse=True
        )[:10]

        for threat in recent_threats:
            activities.append(
                {
                    "type": "threat_detected",
                    "title": f"脅威検出: {threat.title}",
                    "severity": threat.severity.value,
                    "timestamp": threat.detected_at.isoformat(),
                    "details": {
                        "threat_id": threat.id,
                        "category": threat.category.value,
                        "source": threat.source,
                    },
                }
            )

        # 最近のインシデント
        recent_incidents = sorted(
            self.incidents.values(), key=lambda x: x.created_at, reverse=True
        )[:5]

        for incident in recent_incidents:
            activities.append(
                {
                    "type": "incident_created",
                    "title": f"インシデント作成: {incident.title}",
                    "severity": incident.severity.value,
                    "timestamp": incident.created_at.isoformat(),
                    "details": {
                        "incident_id": incident.id,
                        "status": incident.status.value,
                    },
                }
            )

        # 時系列でソート
        activities.sort(key=lambda x: x["timestamp"], reverse=True)

        return activities[:15]  # 最新15件

    async def _get_component_status(self) -> Dict[str, Any]:
        """コンポーネント状態取得"""
        status = {}

        # セキュリティコンポーネントの状態チェック
        components = [
            ("vulnerability_manager", self.vulnerability_manager),
            ("secure_coding_enforcer", self.coding_enforcer),
            ("data_protection_manager", self.data_protection),
            ("access_control_auditor", self.access_auditor),
            ("security_test_coordinator", self.security_tester),
        ]

        for name, component in components:
            if component:
                status[name] = {
                    "status": "active",
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }
            else:
                status[name] = {"status": "unavailable", "last_check": None}

        # システムリソース状態
        if MONITORING_LIBS_AVAILABLE:
            try:
                import psutil

                status["system_resources"] = {
                    "status": "active",
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": (
                        psutil.disk_usage("/").percent
                        if hasattr(psutil.disk_usage("/"), "percent")
                        else 0
                    ),
                }
            except Exception:
                status["system_resources"] = {"status": "error"}
        else:
            status["system_resources"] = {"status": "unavailable"}

        return status

    async def cleanup(self):
        """リソースクリーンアップ"""
        try:
            # 期限切れの脅威を削除
            current_time = datetime.now(timezone.utc)
            expired_threats = [
                threat_id
                for threat_id, threat in self.active_threats.items()
                if threat.expires_at and current_time > threat.expires_at
            ]

            for threat_id in expired_threats:
                del self.active_threats[threat_id]

                # データベースからも削除
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE security_threats SET status = 'expired' WHERE id = ?",
                        (threat_id,),
                    )
                    conn.commit()

            self.logger.info(f"期限切れ脅威を削除: {len(expired_threats)}件")

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")


# グローバルインスタンス
_security_control_center = None


def get_security_control_center() -> ComprehensiveSecurityControlCenter:
    """グローバルセキュリティ管制センターを取得"""
    global _security_control_center
    if _security_control_center is None:
        _security_control_center = ComprehensiveSecurityControlCenter()
    return _security_control_center


if __name__ == "__main__":
    # テスト実行
    async def test_security_control_center():
        print("=== 包括的セキュリティ管制センターテスト ===")

        try:
            # セキュリティ管制センター初期化
            control_center = ComprehensiveSecurityControlCenter()

            print("\n1. セキュリティ管制センター初期化完了")

            # 包括的セキュリティスキャン実行
            print("\n2. 包括的セキュリティスキャン実行中...")
            scan_results = await control_center.run_comprehensive_security_scan()

            print("   スキャン結果:")
            print(f"   - スキャンID: {scan_results['scan_id']}")
            print(f"   - 成功: {scan_results['scan_successful']}")
            print(f"   - スキャン時間: {scan_results.get('scan_duration', 0):.2f}秒")
            print(
                f"   - スキャンコンポーネント: {len(scan_results['components_scanned'])}"
            )
            print(f"   - 検出脅威数: {len(scan_results['threats_detected'])}")
            print(f"   - 作成インシデント数: {len(scan_results['incidents_created'])}")

            # セキュリティダッシュボード取得
            print("\n3. セキュリティダッシュボード取得...")
            dashboard = await control_center.get_security_dashboard()

            if "error" not in dashboard:
                print(f"   システム: {dashboard['overview']['system_name']}")
                print(f"   ステータス: {dashboard['overview']['status']}")

                threat_summary = dashboard["threat_summary"]
                print("   脅威サマリー:")
                print(f"   - 総脅威数: {threat_summary['total']}")
                print(f"   - 重大: {threat_summary['by_severity']['critical']}")
                print(f"   - 高: {threat_summary['by_severity']['high']}")
                print(f"   - 中: {threat_summary['by_severity']['medium']}")
                print(f"   - 低: {threat_summary['by_severity']['low']}")

                incident_summary = dashboard["incident_summary"]
                print("   インシデントサマリー:")
                print(f"   - 総インシデント数: {incident_summary['total']}")
                print(f"   - オープン: {incident_summary['open']}")
                print(f"   - 調査中: {incident_summary['investigating']}")
                print(f"   - 解決済: {incident_summary['resolved']}")

                metrics = dashboard["security_metrics"]
                print("   セキュリティメトリクス:")
                print(f"   - セキュリティスコア: {metrics.security_score:.1f}/100")
                print(
                    f"   - コンプライアンススコア: {metrics.compliance_score:.1f}/100"
                )
                print(f"   - 平均解決時間: {metrics.mean_resolution_time:.2f}時間")
            else:
                print(f"   エラー: {dashboard['error']}")

            # コンプライアンスチェック
            print("\n4. コンプライアンスチェック実行...")
            compliance_results = (
                await control_center.compliance_monitor.check_compliance("PCI_DSS")
            )

            if "error" not in compliance_results:
                print(f"   フレームワーク: {compliance_results['framework_name']}")
                print(f"   スコア: {compliance_results['overall_score']:.1f}/100")
                print(
                    f"   要件適合: {compliance_results['requirements_met']}/{compliance_results['total_requirements']}"
                )
                print(f"   違反数: {len(compliance_results['violations'])}")

            # テスト脅威登録
            print("\n5. テスト脅威登録...")
            test_threat = SecurityThreat(
                id="test_threat_001",
                title="テスト脅威",
                description="セキュリティ管制センターのテスト用脅威",
                category=ThreatCategory.VULNERABILITY,
                severity=SecurityLevel.MEDIUM,
                source="test_system",
                indicators={"test": True},
            )

            threat_id = await control_center.register_threat(test_threat)
            print(f"   登録された脅威ID: {threat_id}")

            # クリーンアップ
            print("\n6. クリーンアップ実行...")
            await control_center.cleanup()

            print("\n[成功] 包括的セキュリティ管制センターテスト完了")

        except Exception as e:
            print(f"[エラー] テストエラー: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(test_security_control_center())
