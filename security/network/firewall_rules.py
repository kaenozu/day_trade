#!/usr/bin/env python3
"""
Issue #800 Phase 5: ネットワークセキュリティ・ファイアウォール管理
Day Trade ML System ネットワーク保護
"""

import os
import json
import logging
import ipaddress
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FirewallAction(Enum):
    """ファイアウォールアクション"""
    ALLOW = "ALLOW"
    DENY = "DENY"
    DROP = "DROP"
    LOG = "LOG"

class Protocol(Enum):
    """プロトコル"""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ALL = "all"

@dataclass
class FirewallRule:
    """ファイアウォールルール"""
    id: str
    name: str
    action: FirewallAction
    protocol: Protocol
    source_ip: str  # CIDR notation or "any"
    dest_ip: str    # CIDR notation or "any"
    source_port: Union[int, str]  # port number or "any"
    dest_port: Union[int, str]    # port number or "any"
    description: str
    priority: int = 100
    enabled: bool = True
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

@dataclass
class NetworkZone:
    """ネットワークゾーン定義"""
    name: str
    description: str
    cidr_blocks: List[str]
    security_level: str  # high, medium, low
    allowed_services: List[str]
    default_action: FirewallAction

class NetworkSecurityManager:
    """ネットワークセキュリティ管理"""

    def __init__(self):
        self.rules: Dict[str, FirewallRule] = {}
        self.zones: Dict[str, NetworkZone] = {}
        self.blocked_ips: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, Dict] = {}

        # セキュリティゾーン定義
        self._initialize_security_zones()

        # デフォルトルール作成
        self._create_default_rules()

    def add_firewall_rule(self, rule: FirewallRule) -> bool:
        """ファイアウォールルール追加"""
        try:
            # バリデーション
            if not self._validate_rule(rule):
                return False

            # タイムスタンプ設定
            rule.created_at = datetime.utcnow()
            rule.modified_at = datetime.utcnow()

            # ルール追加
            self.rules[rule.id] = rule

            logger.info(f"Firewall rule added: {rule.name} ({rule.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add firewall rule: {str(e)}")
            return False

    def remove_firewall_rule(self, rule_id: str) -> bool:
        """ファイアウォールルール削除"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            del self.rules[rule_id]
            logger.info(f"Firewall rule removed: {rule.name} ({rule_id})")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """ルール有効化"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.rules[rule_id].modified_at = datetime.utcnow()
            logger.info(f"Firewall rule enabled: {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """ルール無効化"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.rules[rule_id].modified_at = datetime.utcnow()
            logger.info(f"Firewall rule disabled: {rule_id}")
            return True
        return False

    def check_access(self, source_ip: str, dest_ip: str, dest_port: int, protocol: str) -> Tuple[bool, str]:
        """アクセス許可確認"""

        # ブロックIPチェック
        if self._is_ip_blocked(source_ip):
            return False, f"IP blocked: {source_ip}"

        # レート制限チェック
        if not self._check_rate_limit(source_ip):
            return False, f"Rate limit exceeded: {source_ip}"

        # ファイアウォールルール評価
        matching_rules = self._find_matching_rules(source_ip, dest_ip, dest_port, protocol)

        if not matching_rules:
            # デフォルト動作（ゾーンベース）
            zone = self._get_zone_for_ip(source_ip)
            if zone and zone.default_action == FirewallAction.ALLOW:
                return True, "Default zone allow"
            else:
                return False, "Default deny"

        # 優先度順でルール評価
        sorted_rules = sorted(matching_rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            if rule.action == FirewallAction.ALLOW:
                return True, f"Allowed by rule: {rule.name}"
            elif rule.action in [FirewallAction.DENY, FirewallAction.DROP]:
                return False, f"Denied by rule: {rule.name}"

        return False, "No matching allow rule"

    def block_ip(self, ip_address: str, reason: str, duration_hours: int = 24) -> bool:
        """IP アドレスブロック"""
        try:
            # IP アドレス検証
            ipaddress.ip_address(ip_address)

            block_info = {
                'reason': reason,
                'blocked_at': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat(),
                'duration_hours': duration_hours
            }

            self.blocked_ips[ip_address] = block_info

            logger.warning(f"IP blocked: {ip_address} - {reason} (Duration: {duration_hours}h)")
            return True

        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {str(e)}")
            return False

    def unblock_ip(self, ip_address: str) -> bool:
        """IP アドレスブロック解除"""
        if ip_address in self.blocked_ips:
            del self.blocked_ips[ip_address]
            logger.info(f"IP unblocked: {ip_address}")
            return True
        return False

    def set_rate_limit(self, ip_or_subnet: str, requests_per_minute: int, burst_limit: int = None) -> bool:
        """レート制限設定"""
        try:
            if burst_limit is None:
                burst_limit = requests_per_minute * 2

            limit_config = {
                'requests_per_minute': requests_per_minute,
                'burst_limit': burst_limit,
                'current_requests': 0,
                'window_start': time.time(),
                'burst_used': 0
            }

            self.rate_limits[ip_or_subnet] = limit_config

            logger.info(f"Rate limit set: {ip_or_subnet} - {requests_per_minute}/min (burst: {burst_limit})")
            return True

        except Exception as e:
            logger.error(f"Failed to set rate limit for {ip_or_subnet}: {str(e)}")
            return False

    def get_security_status(self) -> Dict:
        """セキュリティ状況取得"""
        active_rules = len([r for r in self.rules.values() if r.enabled])
        blocked_ips_count = len(self.blocked_ips)
        rate_limited_ips = len(self.rate_limits)

        # 期限切れブロックIP確認
        expired_blocks = []
        current_time = datetime.utcnow()

        for ip, block_info in self.blocked_ips.items():
            expires_at = datetime.fromisoformat(block_info['expires_at'])
            if current_time > expires_at:
                expired_blocks.append(ip)

        return {
            'active_firewall_rules': active_rules,
            'total_firewall_rules': len(self.rules),
            'blocked_ips': blocked_ips_count,
            'expired_blocks': len(expired_blocks),
            'rate_limited_endpoints': rate_limited_ips,
            'security_zones': len(self.zones),
            'last_updated': datetime.utcnow().isoformat()
        }

    def export_rules(self) -> List[Dict]:
        """ルール設定エクスポート"""
        exported_rules = []
        for rule in self.rules.values():
            rule_dict = asdict(rule)
            # datetime オブジェクトを文字列に変換
            if rule_dict['created_at']:
                rule_dict['created_at'] = rule_dict['created_at'].isoformat()
            if rule_dict['modified_at']:
                rule_dict['modified_at'] = rule_dict['modified_at'].isoformat()
            # Enum を値に変換
            rule_dict['action'] = rule_dict['action'].value
            rule_dict['protocol'] = rule_dict['protocol'].value
            exported_rules.append(rule_dict)

        return exported_rules

    def import_rules(self, rules_data: List[Dict]) -> bool:
        """ルール設定インポート"""
        try:
            imported_count = 0

            for rule_data in rules_data:
                # Enum 復元
                rule_data['action'] = FirewallAction(rule_data['action'])
                rule_data['protocol'] = Protocol(rule_data['protocol'])

                # datetime 復元
                if rule_data['created_at']:
                    rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                if rule_data['modified_at']:
                    rule_data['modified_at'] = datetime.fromisoformat(rule_data['modified_at'])

                rule = FirewallRule(**rule_data)

                if self.add_firewall_rule(rule):
                    imported_count += 1

            logger.info(f"Imported {imported_count} firewall rules")
            return True

        except Exception as e:
            logger.error(f"Failed to import rules: {str(e)}")
            return False

    def _initialize_security_zones(self):
        """セキュリティゾーン初期化"""
        self.zones = {
            'public': NetworkZone(
                name='public',
                description='Public internet zone',
                cidr_blocks=['0.0.0.0/0'],
                security_level='low',
                allowed_services=['http', 'https'],
                default_action=FirewallAction.DENY
            ),
            'internal': NetworkZone(
                name='internal',
                description='Internal private network',
                cidr_blocks=['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'],
                security_level='medium',
                allowed_services=['all'],
                default_action=FirewallAction.ALLOW
            ),
            'ml_services': NetworkZone(
                name='ml_services',
                description='ML services network',
                cidr_blocks=['10.100.0.0/16'],
                security_level='high',
                allowed_services=['ml-api', 'data-api', 'monitoring'],
                default_action=FirewallAction.ALLOW
            ),
            'data_services': NetworkZone(
                name='data_services',
                description='Data services network',
                cidr_blocks=['10.200.0.0/16'],
                security_level='high',
                allowed_services=['database', 'cache', 'storage'],
                default_action=FirewallAction.ALLOW
            ),
            'management': NetworkZone(
                name='management',
                description='Management and monitoring network',
                cidr_blocks=['10.255.0.0/16'],
                security_level='high',
                allowed_services=['ssh', 'monitoring', 'logging'],
                default_action=FirewallAction.ALLOW
            )
        }

    def _create_default_rules(self):
        """デフォルトファイアウォールルール作成"""
        default_rules = [
            # SSH アクセス（管理ネットワークのみ）
            FirewallRule(
                id='ssh_management_only',
                name='SSH Access - Management Network Only',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='10.255.0.0/16',
                dest_ip='any',
                source_port='any',
                dest_port=22,
                description='Allow SSH from management network only',
                priority=10
            ),

            # HTTP/HTTPS アクセス
            FirewallRule(
                id='web_access_public',
                name='Web Access - Public',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='any',
                dest_ip='any',
                source_port='any',
                dest_port=80,
                description='Allow HTTP access',
                priority=20
            ),

            FirewallRule(
                id='https_access_public',
                name='HTTPS Access - Public',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='any',
                dest_ip='any',
                source_port='any',
                dest_port=443,
                description='Allow HTTPS access',
                priority=20
            ),

            # MLサービスAPI（内部ネットワークのみ）
            FirewallRule(
                id='ml_api_internal',
                name='ML API - Internal Network',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='10.0.0.0/8',
                dest_ip='10.100.0.0/16',
                source_port='any',
                dest_port=8000,
                description='Allow ML API access from internal network',
                priority=30
            ),

            # データベースアクセス（データサービスのみ）
            FirewallRule(
                id='database_data_services',
                name='Database - Data Services Only',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='10.100.0.0/16',  # ML services
                dest_ip='10.200.0.0/16',    # Data services
                source_port='any',
                dest_port=5432,
                description='Allow database access from ML services',
                priority=40
            ),

            # Redis キャッシュアクセス
            FirewallRule(
                id='redis_internal',
                name='Redis Cache - Internal Services',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='10.100.0.0/16',
                dest_ip='10.200.0.0/16',
                source_port='any',
                dest_port=6379,
                description='Allow Redis access from ML services',
                priority=40
            ),

            # 監視サービス
            FirewallRule(
                id='monitoring_access',
                name='Monitoring Services',
                action=FirewallAction.ALLOW,
                protocol=Protocol.TCP,
                source_ip='10.255.0.0/16',  # Management network
                dest_ip='any',
                source_port='any',
                dest_port=9090,  # Prometheus
                description='Allow monitoring access from management network',
                priority=50
            ),

            # ICMP（ping）
            FirewallRule(
                id='icmp_internal',
                name='ICMP - Internal Network',
                action=FirewallAction.ALLOW,
                protocol=Protocol.ICMP,
                source_ip='10.0.0.0/8',
                dest_ip='10.0.0.0/8',
                source_port='any',
                dest_port='any',
                description='Allow ICMP within internal network',
                priority=60
            ),

            # 外部からの不正アクセス拒否
            FirewallRule(
                id='deny_external_database',
                name='Deny External Database Access',
                action=FirewallAction.DENY,
                protocol=Protocol.TCP,
                source_ip='0.0.0.0/0',
                dest_ip='any',
                source_port='any',
                dest_port=5432,
                description='Deny external database access',
                priority=90
            ),

            FirewallRule(
                id='deny_external_redis',
                name='Deny External Redis Access',
                action=FirewallAction.DENY,
                protocol=Protocol.TCP,
                source_ip='0.0.0.0/0',
                dest_ip='any',
                source_port='any',
                dest_port=6379,
                description='Deny external Redis access',
                priority=90
            )
        ]

        for rule in default_rules:
            self.add_firewall_rule(rule)

    def _validate_rule(self, rule: FirewallRule) -> bool:
        """ルールバリデーション"""
        try:
            # IP アドレス形式確認
            if rule.source_ip != 'any':
                ipaddress.ip_network(rule.source_ip, strict=False)

            if rule.dest_ip != 'any':
                ipaddress.ip_network(rule.dest_ip, strict=False)

            # ポート番号確認
            if rule.source_port != 'any':
                port = int(rule.source_port)
                if not (1 <= port <= 65535):
                    raise ValueError(f"Invalid source port: {port}")

            if rule.dest_port != 'any':
                port = int(rule.dest_port)
                if not (1 <= port <= 65535):
                    raise ValueError(f"Invalid destination port: {port}")

            return True

        except Exception as e:
            logger.error(f"Rule validation failed: {str(e)}")
            return False

    def _find_matching_rules(self, source_ip: str, dest_ip: str, dest_port: int, protocol: str) -> List[FirewallRule]:
        """マッチするルール検索"""
        matching_rules = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # プロトコル確認
            if rule.protocol != Protocol.ALL and rule.protocol.value != protocol.lower():
                continue

            # IP アドレス確認
            if not self._ip_matches(source_ip, rule.source_ip):
                continue

            if not self._ip_matches(dest_ip, rule.dest_ip):
                continue

            # ポート確認
            if rule.dest_port != 'any' and int(rule.dest_port) != dest_port:
                continue

            matching_rules.append(rule)

        return matching_rules

    def _ip_matches(self, ip_addr: str, rule_ip: str) -> bool:
        """IP アドレスマッチング"""
        if rule_ip == 'any':
            return True

        try:
            ip = ipaddress.ip_address(ip_addr)
            network = ipaddress.ip_network(rule_ip, strict=False)
            return ip in network
        except:
            return False

    def _is_ip_blocked(self, ip_address: str) -> bool:
        """IP ブロック状態確認"""
        if ip_address not in self.blocked_ips:
            return False

        block_info = self.blocked_ips[ip_address]
        expires_at = datetime.fromisoformat(block_info['expires_at'])

        if datetime.utcnow() > expires_at:
            # 期限切れブロック削除
            del self.blocked_ips[ip_address]
            return False

        return True

    def _check_rate_limit(self, ip_address: str) -> bool:
        """レート制限確認"""
        # 最も具体的なマッチを検索
        matching_limit = None
        best_match_bits = -1

        for ip_or_subnet, limit_config in self.rate_limits.items():
            try:
                if ip_or_subnet == ip_address:
                    # 完全一致
                    matching_limit = limit_config
                    break

                network = ipaddress.ip_network(ip_or_subnet, strict=False)
                if ipaddress.ip_address(ip_address) in network:
                    if network.prefixlen > best_match_bits:
                        matching_limit = limit_config
                        best_match_bits = network.prefixlen

            except:
                continue

        if not matching_limit:
            return True  # 制限なし

        # 時間窓確認
        current_time = time.time()
        window_duration = 60  # 1分

        if current_time - matching_limit['window_start'] > window_duration:
            # 新しい時間窓
            matching_limit['window_start'] = current_time
            matching_limit['current_requests'] = 0
            matching_limit['burst_used'] = 0

        # レート制限確認
        if matching_limit['current_requests'] >= matching_limit['requests_per_minute']:
            # バースト制限確認
            if matching_limit['burst_used'] >= matching_limit['burst_limit']:
                return False
            else:
                matching_limit['burst_used'] += 1

        matching_limit['current_requests'] += 1
        return True

    def _get_zone_for_ip(self, ip_address: str) -> Optional[NetworkZone]:
        """IP アドレスのゾーン取得"""
        try:
            ip = ipaddress.ip_address(ip_address)

            for zone in self.zones.values():
                for cidr_block in zone.cidr_blocks:
                    network = ipaddress.ip_network(cidr_block, strict=False)
                    if ip in network:
                        return zone

            return None

        except:
            return None

if __name__ == '__main__':
    # テスト用
    security_manager = NetworkSecurityManager()

    # アクセステスト
    test_cases = [
        ('192.168.1.100', '10.100.1.5', 8000, 'tcp'),    # ML API アクセス
        ('203.0.113.1', '10.200.1.5', 5432, 'tcp'),      # 外部からDB アクセス
        ('10.255.1.10', '10.100.1.5', 22, 'tcp'),        # 管理ネットワークからSSH
    ]

    for source, dest, port, proto in test_cases:
        allowed, reason = security_manager.check_access(source, dest, port, proto)
        print(f"Access {source} -> {dest}:{port}/{proto}: {'ALLOWED' if allowed else 'DENIED'} - {reason}")

    # セキュリティ状況表示
    status = security_manager.get_security_status()
    print(f"\nSecurity Status: {json.dumps(status, indent=2)}")