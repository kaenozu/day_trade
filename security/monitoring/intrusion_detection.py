#!/usr/bin/env python3
"""
Issue #800 Phase 5: セキュリティ監視・侵入検知システム
Day Trade ML System サイバーセキュリティ・脅威検知
"""

import os
import json
import logging
import time
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import re
import requests
import subprocess
from collections import defaultdict, deque

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """脅威レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """攻撃タイプ"""
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    DDOS = "ddos"
    PORT_SCAN = "port_scan"
    MALWARE = "malware"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class ActionType(Enum):
    """対応アクション"""
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    ALERT_ONLY = "alert_only"
    QUARANTINE = "quarantine"
    KILL_SESSION = "kill_session"
    FORCE_LOGOUT = "force_logout"

@dataclass
class SecurityEvent:
    """セキュリティイベント"""
    event_id: str
    timestamp: datetime
    source_ip: str
    target_service: str
    attack_type: AttackType
    threat_level: ThreatLevel
    description: str
    evidence: Dict
    action_taken: Optional[ActionType] = None
    resolved: bool = False
    user_agent: str = ""
    request_path: str = ""
    payload: str = ""

@dataclass
class ThreatIntelligence:
    """脅威インテリジェンス"""
    ip_address: str
    threat_type: str
    confidence: float  # 0.0 - 1.0
    source: str
    first_seen: datetime
    last_updated: datetime
    description: str

@dataclass
class SecurityRule:
    """セキュリティルール"""
    rule_id: str
    name: str
    description: str
    pattern: str
    attack_type: AttackType
    threat_level: ThreatLevel
    action: ActionType
    enabled: bool = True
    false_positive_rate: float = 0.0

class IntrusionDetectionSystem:
    """侵入検知システム"""

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.rules: Dict[str, SecurityRule] = {}
        self.threat_intel: Dict[str, ThreatIntelligence] = {}
        self.blocked_ips: Set[str] = set()
        self.monitoring_active = False

        # 統計情報
        self.ip_request_counts = defaultdict(lambda: deque(maxlen=1000))
        self.failed_login_attempts = defaultdict(lambda: deque(maxlen=100))
        self.suspicious_patterns = defaultdict(int)

        # ML ベースの異常検知
        self.baseline_metrics = {}
        self.anomaly_threshold = 2.0  # 標準偏差

        # 初期化
        self._load_security_rules()
        self._load_threat_intelligence()

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            logger.warning("Security monitoring already active")
            return

        self.monitoring_active = True

        # ログ監視スレッド
        log_monitor_thread = threading.Thread(
            target=self._log_monitor_loop,
            name="security-log-monitor"
        )
        log_monitor_thread.daemon = True
        log_monitor_thread.start()

        # ネットワーク監視スレッド
        network_monitor_thread = threading.Thread(
            target=self._network_monitor_loop,
            name="security-network-monitor"
        )
        network_monitor_thread.daemon = True
        network_monitor_thread.start()

        # 異常検知スレッド
        anomaly_thread = threading.Thread(
            target=self._anomaly_detection_loop,
            name="security-anomaly-detector"
        )
        anomaly_thread.daemon = True
        anomaly_thread.start()

        logger.info("Security monitoring started")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        logger.info("Security monitoring stopped")

    def analyze_request(self, request_data: Dict) -> Optional[SecurityEvent]:
        """リクエスト分析"""
        try:
            source_ip = request_data.get('source_ip', '')
            user_agent = request_data.get('user_agent', '')
            request_path = request_data.get('path', '')
            payload = request_data.get('payload', '')
            timestamp = datetime.utcnow()

            # 基本的な脅威チェック
            threats = []

            # 1. 既知の悪意のあるIP確認
            if self._is_malicious_ip(source_ip):
                threats.append((AttackType.UNAUTHORIZED_ACCESS, ThreatLevel.HIGH, "Known malicious IP"))

            # 2. SQL インジェクション検知
            if self._detect_sql_injection(payload + request_path):
                threats.append((AttackType.SQL_INJECTION, ThreatLevel.HIGH, "SQL injection pattern detected"))

            # 3. XSS 検知
            if self._detect_xss(payload + request_path):
                threats.append((AttackType.XSS, ThreatLevel.MEDIUM, "XSS pattern detected"))

            # 4. ブルートフォース検知
            if self._detect_brute_force(source_ip, request_path):
                threats.append((AttackType.BRUTE_FORCE, ThreatLevel.HIGH, "Brute force attack detected"))

            # 5. 異常なユーザーエージェント
            if self._detect_suspicious_user_agent(user_agent):
                threats.append((AttackType.ANOMALOUS_BEHAVIOR, ThreatLevel.LOW, "Suspicious user agent"))

            # 6. レート制限チェック
            if self._check_rate_limit_violation(source_ip):
                threats.append((AttackType.DDOS, ThreatLevel.MEDIUM, "Rate limit exceeded"))

            # 最も深刻な脅威を選択
            if threats:
                attack_type, threat_level, description = max(threats, key=lambda x: x[1].value)

                event = SecurityEvent(
                    event_id=self._generate_event_id(),
                    timestamp=timestamp,
                    source_ip=source_ip,
                    target_service=request_data.get('service', 'unknown'),
                    attack_type=attack_type,
                    threat_level=threat_level,
                    description=description,
                    evidence=request_data,
                    user_agent=user_agent,
                    request_path=request_path,
                    payload=payload
                )

                # 自動対応実行
                self._execute_security_action(event)

                # イベント記録
                self.events.append(event)

                return event

            return None

        except Exception as e:
            logger.error(f"Request analysis failed: {str(e)}")
            return None

    def get_security_dashboard(self) -> Dict:
        """セキュリティダッシュボード情報"""
        recent_events = [e for e in self.events if e.timestamp > datetime.utcnow() - timedelta(hours=24)]

        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1

        attack_counts = defaultdict(int)
        for event in recent_events:
            attack_counts[event.attack_type.value] += 1

        top_attackers = defaultdict(int)
        for event in recent_events:
            top_attackers[event.source_ip] += 1

        return {
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'total_events_24h': len(recent_events),
            'blocked_ips_count': len(self.blocked_ips),
            'threat_levels': dict(threat_counts),
            'attack_types': dict(attack_counts),
            'top_attackers': dict(sorted(top_attackers.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_events': [
                {
                    'event_id': e.event_id,
                    'timestamp': e.timestamp.isoformat(),
                    'source_ip': e.source_ip,
                    'attack_type': e.attack_type.value,
                    'threat_level': e.threat_level.value,
                    'description': e.description
                }
                for e in sorted(recent_events, key=lambda x: x.timestamp, reverse=True)[:20]
            ],
            'last_updated': datetime.utcnow().isoformat()
        }

    def block_ip_address(self, ip_address: str, reason: str = "Manual block") -> bool:
        """IP アドレスブロック"""
        try:
            # IP アドレス検証
            ipaddress.ip_address(ip_address)

            self.blocked_ips.add(ip_address)

            # ファイアウォールルール追加（実際の実装）
            self._add_firewall_rule(ip_address, "DENY")

            logger.warning(f"IP blocked: {ip_address} - {reason}")

            # 通知送信
            self._send_security_alert(f"IP Address Blocked: {ip_address}", f"Reason: {reason}", ThreatLevel.HIGH)

            return True

        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {str(e)}")
            return False

    def unblock_ip_address(self, ip_address: str) -> bool:
        """IP アドレスブロック解除"""
        try:
            if ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)

                # ファイアウォールルール削除
                self._remove_firewall_rule(ip_address)

                logger.info(f"IP unblocked: {ip_address}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to unblock IP {ip_address}: {str(e)}")
            return False

    def add_threat_intelligence(self, intel: ThreatIntelligence):
        """脅威インテリジェンス追加"""
        self.threat_intel[intel.ip_address] = intel
        logger.info(f"Threat intelligence added: {intel.ip_address} - {intel.threat_type}")

    def _load_security_rules(self):
        """セキュリティルール読み込み"""
        rules = [
            SecurityRule(
                rule_id="sql_injection_basic",
                name="Basic SQL Injection Detection",
                description="Detects common SQL injection patterns",
                pattern=r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+.*\s+(from|into|table|database)",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                action=ActionType.BLOCK_IP
            ),
            SecurityRule(
                rule_id="xss_basic",
                name="Basic XSS Detection",
                description="Detects common XSS patterns",
                pattern=r"(?i)<script|javascript:|vbscript:|onload=|onerror=",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.MEDIUM,
                action=ActionType.ALERT_ONLY
            ),
            SecurityRule(
                rule_id="brute_force_login",
                name="Login Brute Force Detection",
                description="Detects brute force login attempts",
                pattern=r"/login|/auth|/signin",
                attack_type=AttackType.BRUTE_FORCE,
                threat_level=ThreatLevel.HIGH,
                action=ActionType.RATE_LIMIT
            ),
            SecurityRule(
                rule_id="port_scan",
                name="Port Scan Detection",
                description="Detects port scanning activities",
                pattern=r"rapid_connection_attempts",
                attack_type=AttackType.PORT_SCAN,
                threat_level=ThreatLevel.MEDIUM,
                action=ActionType.BLOCK_IP
            ),
            SecurityRule(
                rule_id="suspicious_user_agent",
                name="Suspicious User Agent",
                description="Detects suspicious user agents",
                pattern=r"(?i)(bot|crawler|scanner|sqlmap|nikto|burp|nmap)",
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.LOW,
                action=ActionType.ALERT_ONLY
            )
        ]

        self.rules = {rule.rule_id: rule for rule in rules}

    def _load_threat_intelligence(self):
        """脅威インテリジェンス読み込み"""
        # 既知の悪意のあるIPリスト（サンプル）
        malicious_ips = [
            "192.0.2.1",    # RFC5737 documentation IP
            "198.51.100.1", # RFC5737 documentation IP
            "203.0.113.1"   # RFC5737 documentation IP
        ]

        for ip in malicious_ips:
            intel = ThreatIntelligence(
                ip_address=ip,
                threat_type="malware_c2",
                confidence=0.9,
                source="threat_feed",
                first_seen=datetime.utcnow() - timedelta(days=30),
                last_updated=datetime.utcnow(),
                description="Known malicious IP from threat intelligence feed"
            )
            self.threat_intel[ip] = intel

    def _log_monitor_loop(self):
        """ログ監視ループ"""
        while self.monitoring_active:
            try:
                # アプリケーションログファイル監視
                self._monitor_application_logs()

                # システムログ監視
                self._monitor_system_logs()

                # 認証ログ監視
                self._monitor_auth_logs()

            except Exception as e:
                logger.error(f"Log monitoring error: {str(e)}")

            time.sleep(30)

    def _network_monitor_loop(self):
        """ネットワーク監視ループ"""
        while self.monitoring_active:
            try:
                # ネットワーク接続監視
                self._monitor_network_connections()

                # 異常なトラフィック検知
                self._detect_abnormal_traffic()

            except Exception as e:
                logger.error(f"Network monitoring error: {str(e)}")

            time.sleep(60)

    def _anomaly_detection_loop(self):
        """異常検知ループ"""
        while self.monitoring_active:
            try:
                # ベースライン更新
                self._update_baseline_metrics()

                # 異常検知実行
                self._detect_anomalies()

            except Exception as e:
                logger.error(f"Anomaly detection error: {str(e)}")

            time.sleep(300)  # 5分間隔

    def _is_malicious_ip(self, ip_address: str) -> bool:
        """悪意のあるIP判定"""
        return ip_address in self.threat_intel

    def _detect_sql_injection(self, text: str) -> bool:
        """SQL インジェクション検知"""
        rule = self.rules.get("sql_injection_basic")
        if rule and rule.enabled:
            return bool(re.search(rule.pattern, text))
        return False

    def _detect_xss(self, text: str) -> bool:
        """XSS 検知"""
        rule = self.rules.get("xss_basic")
        if rule and rule.enabled:
            return bool(re.search(rule.pattern, text))
        return False

    def _detect_brute_force(self, source_ip: str, path: str) -> bool:
        """ブルートフォース検知"""
        # ログイン関連パス確認
        if not re.search(r"/login|/auth|/signin", path):
            return False

        # 失敗試行回数確認
        now = datetime.utcnow()
        recent_attempts = self.failed_login_attempts[source_ip]

        # 5分以内の試行回数
        recent_count = sum(1 for timestamp in recent_attempts if now - timestamp < timedelta(minutes=5))

        return recent_count > 10  # 5分間で10回以上

    def _detect_suspicious_user_agent(self, user_agent: str) -> bool:
        """疑わしいユーザーエージェント検知"""
        rule = self.rules.get("suspicious_user_agent")
        if rule and rule.enabled:
            return bool(re.search(rule.pattern, user_agent))
        return False

    def _check_rate_limit_violation(self, source_ip: str) -> bool:
        """レート制限違反確認"""
        now = datetime.utcnow()
        requests = self.ip_request_counts[source_ip]

        # 1分以内のリクエスト数
        recent_count = sum(1 for timestamp in requests if now - timestamp < timedelta(minutes=1))

        return recent_count > 100  # 1分間で100リクエスト以上

    def _execute_security_action(self, event: SecurityEvent):
        """セキュリティアクション実行"""
        rule = next((r for r in self.rules.values() if r.attack_type == event.attack_type), None)

        if not rule:
            return

        action = rule.action

        try:
            if action == ActionType.BLOCK_IP:
                self.block_ip_address(event.source_ip, f"Auto-block: {event.description}")
                event.action_taken = ActionType.BLOCK_IP

            elif action == ActionType.RATE_LIMIT:
                self._apply_rate_limit(event.source_ip)
                event.action_taken = ActionType.RATE_LIMIT

            elif action == ActionType.ALERT_ONLY:
                self._send_security_alert(
                    f"Security Alert: {event.attack_type.value}",
                    event.description,
                    event.threat_level
                )
                event.action_taken = ActionType.ALERT_ONLY

        except Exception as e:
            logger.error(f"Failed to execute security action: {str(e)}")

    def _monitor_application_logs(self):
        """アプリケーションログ監視"""
        # 実装省略（ログファイル読み込み・解析）
        pass

    def _monitor_system_logs(self):
        """システムログ監視"""
        # 実装省略（syslog監視）
        pass

    def _monitor_auth_logs(self):
        """認証ログ監視"""
        # 実装省略（認証失敗ログ監視）
        pass

    def _monitor_network_connections(self):
        """ネットワーク接続監視"""
        # 実装省略（netstat、ss コマンド）
        pass

    def _detect_abnormal_traffic(self):
        """異常トラフィック検知"""
        # 実装省略（ネットワーク統計分析）
        pass

    def _update_baseline_metrics(self):
        """ベースラインメトリクス更新"""
        # 実装省略（正常時の統計情報更新）
        pass

    def _detect_anomalies(self):
        """異常検知"""
        # 実装省略（統計的異常検知）
        pass

    def _add_firewall_rule(self, ip_address: str, action: str):
        """ファイアウォールルール追加"""
        # 実装省略（iptables、ufw等）
        pass

    def _remove_firewall_rule(self, ip_address: str):
        """ファイアウォールルール削除"""
        # 実装省略
        pass

    def _apply_rate_limit(self, ip_address: str):
        """レート制限適用"""
        # 実装省略（nginx rate limiting等）
        pass

    def _send_security_alert(self, title: str, message: str, threat_level: ThreatLevel):
        """セキュリティアラート送信"""
        alert_data = {
            'title': title,
            'message': message,
            'threat_level': threat_level.value,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Slack 通知
        # 実装省略

        logger.warning(f"Security Alert [{threat_level.value.upper()}]: {title} - {message}")

    def _generate_event_id(self) -> str:
        """イベントID生成"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"SEC_{timestamp}_{random_suffix}"

if __name__ == '__main__':
    # テスト用
    ids = IntrusionDetectionSystem()

    # 監視開始
    ids.start_monitoring()

    # サンプルリクエスト分析
    test_requests = [
        {
            'source_ip': '192.168.1.100',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'path': '/api/predict',
            'payload': '{"data": "normal request"}',
            'service': 'ml-service'
        },
        {
            'source_ip': '203.0.113.1',  # 既知の悪意のあるIP
            'user_agent': 'sqlmap/1.0',
            'path': '/login',
            'payload': "' OR 1=1--",
            'service': 'auth-service'
        }
    ]

    for request in test_requests:
        event = ids.analyze_request(request)
        if event:
            print(f"Security Event Detected: {event.attack_type.value} - {event.description}")

    # ダッシュボード情報表示
    dashboard = ids.get_security_dashboard()
    print(f"Security Dashboard: {json.dumps(dashboard, indent=2)}")

    time.sleep(5)
    ids.stop_monitoring()