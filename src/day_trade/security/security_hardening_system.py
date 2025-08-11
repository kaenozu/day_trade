#!/usr/bin/env python3
"""
セキュリティ強化・脆弱性対策システム
Phase G: 本番運用最適化フェーズ

包括的なセキュリティ監視・脅威検知・防御システム
"""

import ipaddress
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


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
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALWARE = "malware"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class SecurityEvent(Enum):
    """セキュリティイベント"""

    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_REQUEST = "suspicious_request"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    SYSTEM_INTRUSION = "system_intrusion"


@dataclass
class ThreatAlert:
    """脅威アラート"""

    alert_id: str
    threat_level: ThreatLevel
    attack_type: AttackType
    source_ip: Optional[str]
    target_resource: str
    timestamp: datetime
    description: str
    evidence: List[str]
    mitigation_actions: List[str]
    resolved: bool = False


@dataclass
class SecurityRule:
    """セキュリティルール"""

    rule_id: str
    name: str
    pattern: str
    threat_level: ThreatLevel
    attack_type: AttackType
    enabled: bool = True
    false_positive_count: int = 0


@dataclass
class NetworkConnection:
    """ネットワーク接続情報"""

    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    timestamp: datetime
    data_size: int


class IPBlocklist:
    """IPブロックリスト管理"""

    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.blocked_networks: List[ipaddress.IPv4Network] = []
        self.temporary_blocks: Dict[str, datetime] = {}
        self.block_lock = threading.Lock()

        # デフォルトブロックリスト読み込み
        self._load_default_blocklist()

    def _load_default_blocklist(self):
        """デフォルトブロックリスト読み込み"""
        # 既知の悪意のあるIPレンジ（例）
        default_blocks = [
            "10.0.0.0/8",  # プライベートIPの一部（設定による）
            "192.168.0.0/16",  # プライベートIP
            "127.0.0.0/8",  # ローカルホスト
        ]

        # 実際の運用では外部のIP脅威フィードを使用
        known_malicious_ips = [
            "198.51.100.1",  # RFC5737 テスト用IP（例）
            "203.0.113.1",  # RFC5737 テスト用IP（例）
        ]

        with self.block_lock:
            for ip in known_malicious_ips:
                self.blocked_ips.add(ip)

    def add_ip_block(self, ip: str, duration_hours: int = None):
        """IP ブロック追加"""
        with self.block_lock:
            if duration_hours:
                # 一時ブロック
                expiry = datetime.now() + timedelta(hours=duration_hours)
                self.temporary_blocks[ip] = expiry
            else:
                # 永続ブロック
                self.blocked_ips.add(ip)

    def remove_ip_block(self, ip: str):
        """IP ブロック削除"""
        with self.block_lock:
            self.blocked_ips.discard(ip)
            self.temporary_blocks.pop(ip, None)

    def is_blocked(self, ip: str) -> bool:
        """IP ブロック状態確認"""
        with self.block_lock:
            # 永続ブロック確認
            if ip in self.blocked_ips:
                return True

            # 一時ブロック確認
            if ip in self.temporary_blocks:
                if datetime.now() < self.temporary_blocks[ip]:
                    return True
                else:
                    # 期限切れブロック削除
                    del self.temporary_blocks[ip]

            # ネットワークブロック確認
            try:
                ip_obj = ipaddress.IPv4Address(ip)
                for network in self.blocked_networks:
                    if ip_obj in network:
                        return True
            except ipaddress.AddressValueError:
                pass

            return False

    def cleanup_expired_blocks(self):
        """期限切れブロック削除"""
        with self.block_lock:
            current_time = datetime.now()
            expired_ips = [
                ip
                for ip, expiry in self.temporary_blocks.items()
                if current_time >= expiry
            ]
            for ip in expired_ips:
                del self.temporary_blocks[ip]


class IntrusionDetectionSystem:
    """侵入検知システム"""

    def __init__(self):
        self.security_rules: List[SecurityRule] = []
        self.connection_history: deque = deque(maxlen=10000)
        self.failed_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.request_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        self._initialize_security_rules()

    def _initialize_security_rules(self):
        """セキュリティルール初期化"""
        default_rules = [
            SecurityRule(
                rule_id="sql_injection",
                name="SQL インジェクション検知",
                pattern=r".*(union|select|insert|update|delete|drop|alter|exec|execute).*\s*(from|into|where|order\s+by).*",
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.SQL_INJECTION,
            ),
            SecurityRule(
                rule_id="xss_attempt",
                name="XSS 攻撃検知",
                pattern=r".*(<script|javascript:|on\w+\s*=|<iframe|<object|<embed).*",
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.XSS,
            ),
            SecurityRule(
                rule_id="path_traversal",
                name="パストラバーサル検知",
                pattern=r".*(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c).*",
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.UNAUTHORIZED_ACCESS,
            ),
            SecurityRule(
                rule_id="command_injection",
                name="コマンドインジェクション検知",
                pattern=r".*(\||;|&|\$\(|`|nc\s|wget\s|curl\s|/bin/|cmd\.exe).*",
                threat_level=ThreatLevel.CRITICAL,
                attack_type=AttackType.PRIVILEGE_ESCALATION,
            ),
            SecurityRule(
                rule_id="credential_stuffing",
                name="クレデンシャルスタッフィング検知",
                pattern=r".*(admin|administrator|root|password|123456|qwerty|login).*",
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.BRUTE_FORCE,
            ),
        ]

        self.security_rules.extend(default_rules)

    def analyze_request(self, request_data: Dict[str, Any]) -> List[ThreatAlert]:
        """リクエスト分析"""
        alerts = []

        source_ip = request_data.get("source_ip", "")
        request_path = request_data.get("path", "")
        request_method = request_data.get("method", "")
        request_body = request_data.get("body", "")
        user_agent = request_data.get("user_agent", "")

        # 全体のリクエスト内容
        full_request = f"{request_method} {request_path} {request_body} {user_agent}"

        # セキュリティルールチェック
        for rule in self.security_rules:
            if not rule.enabled:
                continue

            if re.search(rule.pattern, full_request, re.IGNORECASE):
                alert = ThreatAlert(
                    alert_id=f"{rule.rule_id}_{int(time.time())}",
                    threat_level=rule.threat_level,
                    attack_type=rule.attack_type,
                    source_ip=source_ip,
                    target_resource=request_path,
                    timestamp=datetime.now(),
                    description=f"{rule.name}: 疑わしいパターンが検出されました",
                    evidence=[
                        f"ルール: {rule.name}",
                        f"パターン: {rule.pattern}",
                        f"リクエスト: {full_request[:500]}",
                    ],
                    mitigation_actions=[
                        f"ソースIP {source_ip} を一時ブロック検討",
                        "リクエストの詳細分析",
                        "ログの継続監視",
                    ],
                )
                alerts.append(alert)

        # レート制限チェック
        rate_alert = self._check_rate_limiting(source_ip)
        if rate_alert:
            alerts.append(rate_alert)

        return alerts

    def _check_rate_limiting(self, source_ip: str) -> Optional[ThreatAlert]:
        """レート制限チェック"""
        current_time = datetime.now()

        # 過去1分間のリクエスト数
        self.request_rates[source_ip].append(current_time)

        # 過去1分より古いエントリを削除
        cutoff_time = current_time - timedelta(minutes=1)
        while (
            self.request_rates[source_ip]
            and self.request_rates[source_ip][0] < cutoff_time
        ):
            self.request_rates[source_ip].popleft()

        request_count = len(self.request_rates[source_ip])

        # 1分間に100リクエスト以上でDDoS疑い
        if request_count > 100:
            return ThreatAlert(
                alert_id=f"ddos_{source_ip}_{int(time.time())}",
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.DDOS,
                source_ip=source_ip,
                target_resource="*",
                timestamp=current_time,
                description=f"DDoS 攻撃の疑い: {source_ip} から {request_count} req/min",
                evidence=[f"リクエスト数: {request_count}/分"],
                mitigation_actions=[
                    f"IP {source_ip} の即座ブロック",
                    "ネットワークレベルでの制限",
                    "トラフィック分散の確認",
                ],
            )

        return None

    def record_login_attempt(
        self, source_ip: str, username: str, success: bool
    ) -> Optional[ThreatAlert]:
        """ログイン試行記録"""
        current_time = datetime.now()

        if not success:
            # ログイン失敗記録
            self.failed_login_attempts[source_ip].append(current_time)

            # 過去10分間の失敗数チェック
            cutoff_time = current_time - timedelta(minutes=10)
            recent_failures = [
                t for t in self.failed_login_attempts[source_ip] if t > cutoff_time
            ]
            self.failed_login_attempts[source_ip] = recent_failures

            # 10分間に5回以上の失敗でブルートフォース攻撃疑い
            if len(recent_failures) >= 5:
                return ThreatAlert(
                    alert_id=f"brute_force_{source_ip}_{int(time.time())}",
                    threat_level=ThreatLevel.HIGH,
                    attack_type=AttackType.BRUTE_FORCE,
                    source_ip=source_ip,
                    target_resource="/login",
                    timestamp=current_time,
                    description=f"ブルートフォース攻撃の疑い: {source_ip} からの連続ログイン失敗",
                    evidence=[
                        f"失敗回数: {len(recent_failures)}回/10分",
                        f"対象ユーザー: {username}",
                    ],
                    mitigation_actions=[
                        f"IP {source_ip} の24時間ブロック",
                        f"ユーザー {username} のアカウント保護",
                        "追加認証の要求",
                    ],
                )

        return None


class SecurityHardeningSystem:
    """セキュリティ強化システム"""

    def __init__(self):
        self.ip_blocklist = IPBlocklist()
        self.ids = IntrusionDetectionSystem()
        self.threat_alerts: List[ThreatAlert] = []
        self.security_metrics: Dict[str, int] = defaultdict(int)
        self.alert_lock = threading.Lock()

        print("=" * 80)
        print("[SECURITY] セキュリティ強化・脅威検知システム")
        print("Phase G: 本番運用最適化フェーズ")
        print("=" * 80)

    def process_security_event(
        self, event_type: SecurityEvent, event_data: Dict[str, Any]
    ) -> List[ThreatAlert]:
        """セキュリティイベント処理"""
        alerts = []

        with self.alert_lock:
            self.security_metrics[f"{event_type.value}_total"] += 1

            if event_type == SecurityEvent.SUSPICIOUS_REQUEST:
                # リクエスト分析
                request_alerts = self.ids.analyze_request(event_data)
                alerts.extend(request_alerts)

                # IPブロック確認
                source_ip = event_data.get("source_ip", "")
                if source_ip and self.ip_blocklist.is_blocked(source_ip):
                    block_alert = ThreatAlert(
                        alert_id=f"blocked_ip_{source_ip}_{int(time.time())}",
                        threat_level=ThreatLevel.MEDIUM,
                        attack_type=AttackType.UNAUTHORIZED_ACCESS,
                        source_ip=source_ip,
                        target_resource=event_data.get("path", ""),
                        timestamp=datetime.now(),
                        description=f"ブロック済みIP からのアクセス試行: {source_ip}",
                        evidence=[f"ブロック済みIP: {source_ip}"],
                        mitigation_actions=["接続拒否", "ログ記録"],
                    )
                    alerts.append(block_alert)

            elif event_type == SecurityEvent.LOGIN_FAILURE:
                # ログイン失敗分析
                source_ip = event_data.get("source_ip", "")
                username = event_data.get("username", "")

                login_alert = self.ids.record_login_attempt(source_ip, username, False)
                if login_alert:
                    alerts.append(login_alert)
                    # ブルートフォース攻撃の場合、IP を一時ブロック
                    if login_alert.attack_type == AttackType.BRUTE_FORCE:
                        self.ip_blocklist.add_ip_block(source_ip, duration_hours=24)

            elif event_type == SecurityEvent.PERMISSION_DENIED:
                # 権限エラー分析
                source_ip = event_data.get("source_ip", "")
                resource = event_data.get("resource", "")

                alerts.append(
                    ThreatAlert(
                        alert_id=f"permission_denied_{int(time.time())}",
                        threat_level=ThreatLevel.LOW,
                        attack_type=AttackType.UNAUTHORIZED_ACCESS,
                        source_ip=source_ip,
                        target_resource=resource,
                        timestamp=datetime.now(),
                        description=f"認可されていないリソースへのアクセス試行: {resource}",
                        evidence=[f"ソースIP: {source_ip}", f"リソース: {resource}"],
                        mitigation_actions=["アクセス拒否", "監視継続"],
                    )
                )

            # アラート保存
            for alert in alerts:
                self.threat_alerts.append(alert)
                self.security_metrics[f"{alert.attack_type.value}_alerts"] += 1

        return alerts

    def get_security_dashboard(self) -> Dict[str, Any]:
        """セキュリティダッシュボード"""
        with self.alert_lock:
            active_alerts = [
                alert for alert in self.threat_alerts if not alert.resolved
            ]

            # 脅威レベル別集計
            threat_counts = defaultdict(int)
            for alert in active_alerts:
                threat_counts[alert.threat_level.value] += 1

            # 攻撃タイプ別集計
            attack_counts = defaultdict(int)
            for alert in active_alerts:
                attack_counts[alert.attack_type.value] += 1

            # 最近の重要なアラート
            recent_critical = [
                alert
                for alert in active_alerts
                if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
                and alert.timestamp > datetime.now() - timedelta(hours=24)
            ]

            return {
                "total_alerts": len(active_alerts),
                "threat_level_distribution": dict(threat_counts),
                "attack_type_distribution": dict(attack_counts),
                "recent_critical_alerts": len(recent_critical),
                "blocked_ips_count": len(self.ip_blocklist.blocked_ips)
                + len(self.ip_blocklist.temporary_blocks),
                "security_metrics": dict(self.security_metrics),
                "system_status": "PROTECTED"
                if len(active_alerts) < 10
                else "UNDER_ATTACK",
            }

    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self.alert_lock:
            recent_alerts = [
                alert for alert in self.threat_alerts if alert.timestamp > cutoff_time
            ]

            # 統計計算
            total_alerts = len(recent_alerts)
            resolved_alerts = len([a for a in recent_alerts if a.resolved])
            resolution_rate = (
                (resolved_alerts / total_alerts) * 100 if total_alerts > 0 else 0
            )

            # 最も多い攻撃タイプ
            attack_counter = defaultdict(int)
            for alert in recent_alerts:
                attack_counter[alert.attack_type.value] += 1

            top_attacks = sorted(
                attack_counter.items(), key=lambda x: x[1], reverse=True
            )[:5]

            # 最も多いソースIP
            ip_counter = defaultdict(int)
            for alert in recent_alerts:
                if alert.source_ip:
                    ip_counter[alert.source_ip] += 1

            top_source_ips = sorted(
                ip_counter.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "report_period_hours": hours,
                "timestamp": datetime.now().isoformat(),
                "total_alerts": total_alerts,
                "resolved_alerts": resolved_alerts,
                "resolution_rate_percent": round(resolution_rate, 2),
                "top_attack_types": top_attacks,
                "top_source_ips": top_source_ips,
                "security_recommendations": self._generate_security_recommendations(
                    recent_alerts
                ),
            }

    def _generate_security_recommendations(
        self, alerts: List[ThreatAlert]
    ) -> List[str]:
        """セキュリティ推奨事項生成"""
        recommendations = []

        # 攻撃タイプ別の推奨事項
        attack_types = set(alert.attack_type for alert in alerts)

        if AttackType.SQL_INJECTION in attack_types:
            recommendations.append(
                "SQLインジェクション対策: パラメータ化クエリの徹底使用"
            )

        if AttackType.XSS in attack_types:
            recommendations.append("XSS対策: 入力値のサニタイズ強化")

        if AttackType.BRUTE_FORCE in attack_types:
            recommendations.append(
                "ブルートフォース対策: アカウントロックアウト機能の実装"
            )

        if AttackType.DDOS in attack_types:
            recommendations.append("DDoS対策: レート制限とCDN利用の検討")

        # 一般的な推奨事項
        if len(alerts) > 50:
            recommendations.append(
                "高い攻撃頻度: WAF（Web Application Firewall）の導入検討"
            )

        recommendations.extend(
            [
                "定期的なセキュリティパッチ適用",
                "ログ監視の継続実施",
                "セキュリティ意識向上研修の実施",
                "侵入テストの定期実行",
            ]
        )

        return recommendations[:10]  # 最大10個の推奨事項

    def resolve_threat_alert(self, alert_id: str):
        """脅威アラート解決"""
        with self.alert_lock:
            for alert in self.threat_alerts:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    break

    def add_custom_security_rule(self, rule: SecurityRule):
        """カスタムセキュリティルール追加"""
        self.ids.security_rules.append(rule)

    def save_security_report(self, filename: str = None) -> str:
        """セキュリティレポート保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_report_{timestamp}.json"

        report = self.generate_security_report()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return f"セキュリティレポート保存完了: {filename}"


def main():
    """メイン実行"""
    security_system = SecurityHardeningSystem()

    print("\n[DEMO] セキュリティシステム デモンストレーション")

    # テスト用セキュリティイベント
    test_events = [
        # 通常のリクエスト
        (
            SecurityEvent.SUSPICIOUS_REQUEST,
            {
                "source_ip": "192.168.1.100",
                "method": "GET",
                "path": "/api/data",
                "body": "",
                "user_agent": "Mozilla/5.0",
            },
        ),
        # SQL インジェクション試行
        (
            SecurityEvent.SUSPICIOUS_REQUEST,
            {
                "source_ip": "203.0.113.1",
                "method": "POST",
                "path": "/login",
                "body": "username=admin&password=' OR '1'='1",
                "user_agent": "sqlmap/1.0",
            },
        ),
        # XSS 試行
        (
            SecurityEvent.SUSPICIOUS_REQUEST,
            {
                "source_ip": "198.51.100.1",
                "method": "GET",
                "path": "/search",
                "body": 'q=<script>alert("xss")</script>',
                "user_agent": "AttackBot",
            },
        ),
        # ブルートフォース攻撃
        (
            SecurityEvent.LOGIN_FAILURE,
            {"source_ip": "203.0.113.1", "username": "admin"},
        ),
        # 権限エラー
        (
            SecurityEvent.PERMISSION_DENIED,
            {"source_ip": "192.168.1.200", "resource": "/admin/config"},
        ),
    ]

    print(f"\n[TEST] {len(test_events)} のセキュリティイベントを処理中...")

    all_alerts = []
    for event_type, event_data in test_events:
        alerts = security_system.process_security_event(event_type, event_data)
        all_alerts.extend(alerts)

        for alert in alerts:
            print(f"[ALERT] {alert.threat_level.value.upper()}: {alert.description}")

    # ダッシュボード表示
    dashboard = security_system.get_security_dashboard()
    print(f"\n[DASHBOARD] システム状態: {dashboard['system_status']}")
    print(f"アクティブアラート: {dashboard['total_alerts']}件")
    print(f"ブロック済みIP: {dashboard['blocked_ips_count']}件")

    # レポート生成・保存
    report_file = security_system.save_security_report()
    print(f"\n[REPORT] {report_file}")

    print("\n[COMPLETE] セキュリティシステム デモ完了")


if __name__ == "__main__":
    main()
