#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Monitoring System - セキュリティモニタリングシステム
セキュリティイベント監視、脅威検出、インシデント対応
"""

import time
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
from collections import defaultdict, deque


class ThreatLevel(Enum):
    """脅威レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """セキュリティイベントタイプ"""
    AUTHENTICATION_FAILURE = "auth_failure"
    SUSPICIOUS_ACCESS = "suspicious_access"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SQL_INJECTION_ATTEMPT = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    DATA_BREACH_ATTEMPT = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALWARE_DETECTED = "malware_detected"


@dataclass
class SecurityEvent:
    """セキュリティイベント"""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    user_agent: str
    endpoint: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False


@dataclass
class ThreatIndicator:
    """脅威インジケータ"""
    indicator_type: str
    value: str
    threat_level: ThreatLevel
    description: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1


class SecurityMonitoringSystem:
    """セキュリティモニタリングシステム"""

    def __init__(self, config_file: str = "config/security.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)

        # 設定読み込み
        self.config = self._load_security_config()

        # イベント保存
        self.events = deque(maxlen=10000)  # 最大10,000イベント
        self.threat_indicators = {}

        # レート制限追跡
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))

        # ブラックリスト/ホワイトリスト
        self.blacklisted_ips = set()
        self.whitelisted_ips = set()

        # 監視統計
        self.stats = {
            'total_events': 0,
            'critical_events': 0,
            'blocked_attempts': 0,
            'last_update': datetime.now()
        }

        # ログ設定
        from daytrade_logging import get_logger
        self.logger = get_logger("security_monitoring")

        # 初期化
        self._load_threat_indicators()
        self._init_default_rules()

        self.logger.info("Security Monitoring System initialized")

    def _load_security_config(self) -> Dict[str, Any]:
        """セキュリティ設定を読み込み"""
        default_config = {
            'rate_limits': {
                'api_requests_per_minute': 60,
                'login_attempts_per_hour': 5,
                'data_requests_per_hour': 1000
            },
            'detection_rules': {
                'enable_sql_injection_detection': True,
                'enable_xss_detection': True,
                'enable_suspicious_user_agent_detection': True,
                'enable_geo_blocking': False
            },
            'blocked_countries': [],
            'alert_thresholds': {
                'critical_events_per_hour': 5,
                'failed_logins_per_ip': 3,
                'suspicious_patterns_per_day': 10
            }
        }

        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # デフォルト設定とマージ
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # デフォルト設定を保存
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load security config: {e}")
            return default_config

    def _load_threat_indicators(self):
        """脅威インジケータを読み込み"""
        # 既知の悪意あるIPアドレスやパターン
        malicious_patterns = [
            "union select",
            "drop table",
            "script>alert",
            "<iframe",
            "javascript:",
            "eval(",
            "base64_decode"
        ]

        # パターンを脅威インジケータとして登録
        for pattern in malicious_patterns:
            self.threat_indicators[pattern] = ThreatIndicator(
                indicator_type="malicious_pattern",
                value=pattern,
                threat_level=ThreatLevel.HIGH,
                description=f"Malicious pattern: {pattern}",
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )

    def _init_default_rules(self):
        """デフォルトセキュリティルールの初期化"""
        # ローカルネットワークをホワイトリストに追加
        local_networks = [
            "127.0.0.1",
            "::1",
            "10.0.0.0/8",
            "172.16.0.0/12",
            "192.168.0.0/16"
        ]

        for network in local_networks:
            try:
                if '/' in network:
                    # CIDR記法の場合
                    self.whitelisted_ips.add(network)
                else:
                    # 単一IPの場合
                    self.whitelisted_ips.add(network)
            except Exception as e:
                self.logger.warning(f"Failed to add whitelisted IP {network}: {e}")

    def generate_event_id(self) -> str:
        """イベントIDを生成"""
        timestamp = str(int(time.time() * 1000))
        random_data = str(hash(time.time()))
        return hashlib.md5(f"{timestamp}{random_data}".encode()).hexdigest()[:16]

    def is_ip_whitelisted(self, ip_address: str) -> bool:
        """IPアドレスがホワイトリストに含まれているか確認"""
        try:
            ip = ipaddress.ip_address(ip_address)
            for whitelisted in self.whitelisted_ips:
                try:
                    if '/' in whitelisted:
                        network = ipaddress.ip_network(whitelisted, strict=False)
                        if ip in network:
                            return True
                    else:
                        if str(ip) == whitelisted:
                            return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def is_ip_blacklisted(self, ip_address: str) -> bool:
        """IPアドレスがブラックリストに含まれているか確認"""
        return ip_address in self.blacklisted_ips

    def check_rate_limit(self, identifier: str, limit_type: str) -> bool:
        """レート制限チェック"""
        now = datetime.now()
        limit_key = f"{identifier}:{limit_type}"

        # 制限設定を取得
        if limit_type == "api_requests":
            max_requests = self.config['rate_limits']['api_requests_per_minute']
            time_window = 60  # 1分
        elif limit_type == "login_attempts":
            max_requests = self.config['rate_limits']['login_attempts_per_hour']
            time_window = 3600  # 1時間
        elif limit_type == "data_requests":
            max_requests = self.config['rate_limits']['data_requests_per_hour']
            time_window = 3600  # 1時間
        else:
            return True  # 不明な制限タイプは許可

        # 古いエントリを削除
        cutoff_time = now - timedelta(seconds=time_window)
        requests = self.rate_limits[limit_key]

        while requests and requests[0] < cutoff_time:
            requests.popleft()

        # 制限チェック
        if len(requests) >= max_requests:
            return False

        # 新しいリクエストを記録
        requests.append(now)
        return True

    def detect_malicious_patterns(self, input_data: str) -> List[str]:
        """悪意のあるパターンを検出"""
        detected_patterns = []
        input_lower = input_data.lower()

        for pattern, indicator in self.threat_indicators.items():
            if indicator.indicator_type == "malicious_pattern" and pattern in input_lower:
                detected_patterns.append(pattern)
                # カウントと最終確認時間を更新
                indicator.count += 1
                indicator.last_seen = datetime.now()

        return detected_patterns

    def analyze_user_agent(self, user_agent: str) -> ThreatLevel:
        """User-Agentを分析して脅威レベルを判定"""
        if not user_agent:
            return ThreatLevel.MEDIUM

        user_agent_lower = user_agent.lower()

        # 疑わしいUser-Agentパターン
        suspicious_patterns = [
            "sqlmap",
            "nikto",
            "burp",
            "acunetix",
            "nessus",
            "openvas",
            "curl",
            "wget",
            "python-requests",
            "bot",
            "crawler",
            "spider"
        ]

        for pattern in suspicious_patterns:
            if pattern in user_agent_lower:
                return ThreatLevel.HIGH

        # 一般的でないUser-Agent
        common_browsers = ["chrome", "firefox", "safari", "edge", "opera"]
        if not any(browser in user_agent_lower for browser in common_browsers):
            return ThreatLevel.MEDIUM

        return ThreatLevel.LOW

    def log_security_event(self, event_type: SecurityEventType, source_ip: str,
                         user_agent: str = "", endpoint: str = "",
                         details: Dict[str, Any] = None) -> SecurityEvent:
        """セキュリティイベントを記録"""
        if details is None:
            details = {}

        # 脅威レベルを判定
        threat_level = ThreatLevel.LOW

        if event_type in [SecurityEventType.SQL_INJECTION_ATTEMPT,
                         SecurityEventType.DATA_BREACH_ATTEMPT]:
            threat_level = ThreatLevel.CRITICAL
        elif event_type in [SecurityEventType.XSS_ATTEMPT,
                           SecurityEventType.PRIVILEGE_ESCALATION]:
            threat_level = ThreatLevel.HIGH
        elif event_type in [SecurityEventType.RATE_LIMIT_EXCEEDED,
                           SecurityEventType.SUSPICIOUS_ACCESS]:
            threat_level = ThreatLevel.MEDIUM

        # User-Agent分析を追加
        ua_threat = self.analyze_user_agent(user_agent)
        if ua_threat.value == "high" and threat_level.value != "critical":
            threat_level = ThreatLevel.HIGH

        # イベント作成
        event = SecurityEvent(
            event_id=self.generate_event_id(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            timestamp=datetime.now(),
            details=details
        )

        # イベントを保存
        self.events.append(event)

        # 統計更新
        self.stats['total_events'] += 1
        if threat_level == ThreatLevel.CRITICAL:
            self.stats['critical_events'] += 1
        self.stats['last_update'] = datetime.now()

        # ログ出力
        self.logger.warning(
            f"Security Event: {event_type.value} from {source_ip} "
            f"[{threat_level.value}] - {endpoint}"
        )

        # 自動対応
        self._auto_response(event)

        return event

    def _auto_response(self, event: SecurityEvent):
        """自動対応処理"""
        # クリティカルレベルの場合、IPをブラックリストに追加
        if event.threat_level == ThreatLevel.CRITICAL:
            if not self.is_ip_whitelisted(event.source_ip):
                self.blacklisted_ips.add(event.source_ip)
                self.stats['blocked_attempts'] += 1
                self.logger.critical(f"Auto-blocked IP: {event.source_ip}")

        # 繰り返し攻撃の検出
        recent_events = [e for e in self.events
                        if e.source_ip == event.source_ip
                        and e.timestamp > datetime.now() - timedelta(hours=1)]

        if len(recent_events) >= 5:  # 1時間に5回以上
            if not self.is_ip_whitelisted(event.source_ip):
                self.blacklisted_ips.add(event.source_ip)
                self.stats['blocked_attempts'] += 1
                self.logger.warning(f"Auto-blocked repeat attacker: {event.source_ip}")

    def validate_request(self, source_ip: str, user_agent: str = "",
                        endpoint: str = "", request_data: str = "") -> Dict[str, Any]:
        """リクエストの検証"""
        validation_result = {
            'allowed': True,
            'threat_level': ThreatLevel.LOW,
            'issues': [],
            'blocked_reason': None
        }

        # IPブラックリストチェック
        if self.is_ip_blacklisted(source_ip):
            validation_result['allowed'] = False
            validation_result['blocked_reason'] = 'IP blacklisted'
            validation_result['threat_level'] = ThreatLevel.CRITICAL
            return validation_result

        # 悪意のあるパターン検出
        if request_data:
            malicious_patterns = self.detect_malicious_patterns(request_data)
            if malicious_patterns:
                validation_result['allowed'] = False
                validation_result['blocked_reason'] = f'Malicious patterns detected: {malicious_patterns}'
                validation_result['threat_level'] = ThreatLevel.HIGH
                validation_result['issues'].extend(malicious_patterns)

                # セキュリティイベントを記録
                if 'union select' in malicious_patterns or 'drop table' in malicious_patterns:
                    self.log_security_event(
                        SecurityEventType.SQL_INJECTION_ATTEMPT,
                        source_ip, user_agent, endpoint,
                        {'patterns': malicious_patterns, 'data': request_data[:200]}
                    )
                elif any(p in malicious_patterns for p in ['script>', '<iframe', 'javascript:']):
                    self.log_security_event(
                        SecurityEventType.XSS_ATTEMPT,
                        source_ip, user_agent, endpoint,
                        {'patterns': malicious_patterns, 'data': request_data[:200]}
                    )

        # User-Agent分析
        ua_threat = self.analyze_user_agent(user_agent)
        if ua_threat in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            validation_result['threat_level'] = ua_threat
            validation_result['issues'].append('Suspicious user agent')

            if ua_threat == ThreatLevel.HIGH:
                self.log_security_event(
                    SecurityEventType.SUSPICIOUS_ACCESS,
                    source_ip, user_agent, endpoint,
                    {'reason': 'Suspicious user agent'}
                )

        return validation_result

    def get_security_status(self) -> Dict[str, Any]:
        """セキュリティ状況を取得"""
        now = datetime.now()

        # 最近のイベント分析
        recent_events = [e for e in self.events
                        if e.timestamp > now - timedelta(hours=24)]

        critical_events_24h = len([e for e in recent_events
                                  if e.threat_level == ThreatLevel.CRITICAL])

        high_events_24h = len([e for e in recent_events
                              if e.threat_level == ThreatLevel.HIGH])

        # 最も活発な攻撃者IP
        ip_counts = defaultdict(int)
        for event in recent_events:
            ip_counts[event.source_ip] += 1

        top_attackers = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # 全体的な脅威レベル判定
        overall_threat = ThreatLevel.LOW
        if critical_events_24h > 10:
            overall_threat = ThreatLevel.CRITICAL
        elif critical_events_24h > 3 or high_events_24h > 20:
            overall_threat = ThreatLevel.HIGH
        elif high_events_24h > 5:
            overall_threat = ThreatLevel.MEDIUM

        return {
            'overall_threat_level': overall_threat.value,
            'stats': {
                'total_events': self.stats['total_events'],
                'critical_events_24h': critical_events_24h,
                'high_events_24h': high_events_24h,
                'blocked_ips_count': len(self.blacklisted_ips),
                'active_threats': len([e for e in recent_events if not e.resolved])
            },
            'top_attackers': top_attackers,
            'recent_events': [
                {
                    'event_id': e.event_id,
                    'type': e.event_type.value,
                    'threat_level': e.threat_level.value,
                    'source_ip': e.source_ip,
                    'timestamp': e.timestamp.isoformat(),
                    'resolved': e.resolved
                }
                for e in list(self.events)[-10:]  # 最新10件
            ],
            'last_update': self.stats['last_update'].isoformat()
        }

    def resolve_event(self, event_id: str) -> bool:
        """イベントを解決済みにマーク"""
        for event in self.events:
            if event.event_id == event_id:
                event.resolved = True
                self.logger.info(f"Security event {event_id} resolved")
                return True
        return False

    def unblock_ip(self, ip_address: str) -> bool:
        """IPアドレスのブロックを解除"""
        if ip_address in self.blacklisted_ips:
            self.blacklisted_ips.remove(ip_address)
            self.logger.info(f"Unblocked IP: {ip_address}")
            return True
        return False

    def add_to_whitelist(self, ip_address: str) -> bool:
        """IPアドレスをホワイトリストに追加"""
        try:
            # IPアドレスの妥当性をチェック
            ipaddress.ip_address(ip_address)
            self.whitelisted_ips.add(ip_address)

            # ブラックリストからも削除
            if ip_address in self.blacklisted_ips:
                self.blacklisted_ips.remove(ip_address)

            self.logger.info(f"Added to whitelist: {ip_address}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add {ip_address} to whitelist: {e}")
            return False


# グローバルインスタンス
_security_monitor = None


def get_security_monitor() -> SecurityMonitoringSystem:
    """グローバルセキュリティモニターを取得"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitoringSystem()
    return _security_monitor


def validate_request(source_ip: str, user_agent: str = "",
                    endpoint: str = "", request_data: str = "") -> Dict[str, Any]:
    """リクエスト検証（便利関数）"""
    return get_security_monitor().validate_request(source_ip, user_agent, endpoint, request_data)


def log_security_event(event_type: SecurityEventType, source_ip: str,
                      user_agent: str = "", endpoint: str = "",
                      details: Dict[str, Any] = None) -> SecurityEvent:
    """セキュリティイベント記録（便利関数）"""
    return get_security_monitor().log_security_event(event_type, source_ip, user_agent, endpoint, details)


if __name__ == "__main__":
    print("🔒 セキュリティモニタリングシステムテスト")
    print("=" * 50)

    monitor = SecurityMonitoringSystem()

    # テストイベント生成
    print("テストイベント生成中...")

    # 通常のアクセス
    result = monitor.validate_request("192.168.1.100", "Mozilla/5.0", "/api/data", "")
    print(f"通常アクセス: 許可={result['allowed']}, 脅威レベル={result['threat_level'].value}")

    # SQLインジェクション試行
    result = monitor.validate_request("203.0.113.1", "sqlmap/1.0", "/api/data", "id=1' UNION SELECT * FROM users--")
    print(f"SQLインジェクション: 許可={result['allowed']}, 理由={result['blocked_reason']}")

    # XSS試行
    result = monitor.validate_request("198.51.100.1", "Chrome/90.0", "/search", "<script>alert('xss')</script>")
    print(f"XSS試行: 許可={result['allowed']}, 理由={result['blocked_reason']}")

    # セキュリティ状況
    status = monitor.get_security_status()
    print(f"\nセキュリティ状況:")
    print(f"  全体脅威レベル: {status['overall_threat_level']}")
    print(f"  総イベント数: {status['stats']['total_events']}")
    print(f"  ブロック済みIP数: {status['stats']['blocked_ips_count']}")

    print("\nテスト完了")