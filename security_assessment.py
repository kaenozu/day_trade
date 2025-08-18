#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Assessment - 高度なセキュリティ強化・脆弱性評価システム
Issue #950対応: セキュリティ監査 + 脆弱性検出 + 自動化防御
"""

import os
import re
import json
import hashlib
import hmac
import secrets
import ssl
import socket
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, Counter
import sqlite3
from pathlib import Path

# 暗号化関連
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# ネットワークセキュリティ
try:
    import requests
    import urllib3
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class VulnerabilityLevel(Enum):
    """脆弱性レベル"""
    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class SecurityCategory(Enum):
    """セキュリティカテゴリ"""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_PROTECTION = "DATA_PROTECTION"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    CONFIGURATION = "CONFIGURATION"
    NETWORK_SECURITY = "NETWORK_SECURITY"
    FILE_SYSTEM = "FILE_SYSTEM"
    LOGGING_MONITORING = "LOGGING_MONITORING"
    DEPENDENCY = "DEPENDENCY"
    CODE_QUALITY = "CODE_QUALITY"


class ThreatType(Enum):
    """脅威タイプ"""
    SQL_INJECTION = "SQL_INJECTION"
    XSS = "XSS"
    CSRF = "CSRF"
    DIRECTORY_TRAVERSAL = "DIRECTORY_TRAVERSAL"
    COMMAND_INJECTION = "COMMAND_INJECTION"
    BUFFER_OVERFLOW = "BUFFER_OVERFLOW"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    DATA_LEAK = "DATA_LEAK"
    WEAK_ENCRYPTION = "WEAK_ENCRYPTION"
    INSECURE_DEFAULTS = "INSECURE_DEFAULTS"


@dataclass
class SecurityVulnerability:
    """セキュリティ脆弱性"""
    vuln_id: str
    title: str
    description: str
    category: SecurityCategory
    threat_type: ThreatType
    severity: VulnerabilityLevel
    affected_files: List[str]
    location: str  # ファイル:行番号
    remediation: str
    references: List[str]
    detected_at: datetime
    status: str = "OPEN"  # OPEN, FIXED, ACCEPTED, FALSE_POSITIVE


@dataclass
class SecurityScanResult:
    """セキュリティスキャン結果"""
    scan_id: str
    scan_type: str
    started_at: datetime
    completed_at: datetime
    vulnerabilities: List[SecurityVulnerability]
    files_scanned: int
    duration_ms: float
    summary: Dict[str, int]


class FileSecurityScanner:
    """ファイルセキュリティスキャナー"""
    
    def __init__(self):
        self.dangerous_functions = {
            'python': [
                'eval', 'exec', 'compile', '__import__',
                'open', 'file', 'input', 'raw_input',
                'subprocess.call', 'subprocess.run', 'os.system',
                'pickle.load', 'pickle.loads'
            ],
            'javascript': [
                'eval', 'Function', 'setTimeout', 'setInterval',
                'document.write', 'innerHTML', 'outerHTML'
            ],
            'sql': [
                'DROP', 'DELETE', 'UPDATE', 'INSERT',
                'EXEC', 'EXECUTE', 'xp_', 'sp_'
            ]
        }
        
        self.security_patterns = {
            'hardcoded_secrets': [
                r'(?i)(api[_-]?key|password|secret|token)\s*[=:]\s*["\']([a-z0-9]{20,})["\']',
                r'(?i)(aws[_-]?access[_-]?key)["\']?\s*[=:]\s*["\']([a-z0-9]{20,})["\']',
                r'(?i)(private[_-]?key)["\']?\s*[=:]\s*["\'][^"\']{20,}["\']'
            ],
            'sql_injection': [
                r'(?i)select\s+.*\s+from\s+.*\s+where\s+.*["\']?\s*\+\s*["\']?',
                r'(?i)(query|sql)\s*[=:]\s*["\'].*["\']?\s*\+\s*.*["\']?',
                r'(?i)execute\s*\(\s*["\'].*["\']?\s*\+\s*.*["\']?\)'
            ],
            'xss_vulnerable': [
                r'(?i)document\.write\s*\(\s*[^)]*user',
                r'(?i)innerHTML\s*=\s*[^;]*user',
                r'(?i)eval\s*\(\s*[^)]*user'
            ],
            'weak_crypto': [
                r'(?i)(md5|sha1)\s*\(',
                r'(?i)des\s*\(',
                r'(?i)random\.random\s*\('
            ],
            'insecure_network': [
                r'(?i)http://[^/]*\.(com|org|net)',
                r'(?i)verify\s*=\s*False',
                r'(?i)ssl[_-]?verify\s*=\s*False'
            ]
        }
    
    def scan_file(self, file_path: str) -> List[SecurityVulnerability]:
        """ファイルスキャン"""
        vulnerabilities = []
        
        if not os.path.exists(file_path):
            return vulnerabilities
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
            
            # パターンマッチング検査
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            vuln = self._create_vulnerability(
                                category, pattern, file_path, line_num, line
                            )
                            if vuln:
                                vulnerabilities.append(vuln)
            
            # 危険関数検査
            file_ext = os.path.splitext(file_path)[1].lower()
            lang = self._get_language(file_ext)
            
            if lang in self.dangerous_functions:
                for func in self.dangerous_functions[lang]:
                    for line_num, line in enumerate(lines, 1):
                        if func in line:
                            vuln = self._create_function_vulnerability(
                                func, file_path, line_num, line, lang
                            )
                            if vuln:
                                vulnerabilities.append(vuln)
            
        except Exception as e:
            logging.error(f"File scan error for {file_path}: {e}")
        
        return vulnerabilities
    
    def _get_language(self, file_ext: str) -> str:
        """ファイル拡張子から言語判定"""
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'javascript',
            '.sql': 'sql',
            '.html': 'javascript',
            '.php': 'php'
        }
        return lang_map.get(file_ext, 'unknown')
    
    def _create_vulnerability(self, category: str, pattern: str, file_path: str,
                            line_num: int, line_content: str) -> Optional[SecurityVulnerability]:
        """脆弱性オブジェクト作成"""
        vuln_map = {
            'hardcoded_secrets': {
                'title': 'ハードコードされた機密情報',
                'threat_type': ThreatType.DATA_LEAK,
                'severity': VulnerabilityLevel.HIGH,
                'category': SecurityCategory.DATA_PROTECTION,
                'description': 'コード内に機密情報（パスワード、APIキーなど）がハードコードされています。',
                'remediation': '環境変数または設定ファイルを使用して機密情報を管理してください。'
            },
            'sql_injection': {
                'title': 'SQLインジェクションの可能性',
                'threat_type': ThreatType.SQL_INJECTION,
                'severity': VulnerabilityLevel.CRITICAL,
                'category': SecurityCategory.INPUT_VALIDATION,
                'description': 'ユーザー入力をSQLクエリに直接結合している可能性があります。',
                'remediation': 'パラメータ化クエリまたはプリペアドステートメントを使用してください。'
            },
            'xss_vulnerable': {
                'title': 'XSS脆弱性の可能性',
                'threat_type': ThreatType.XSS,
                'severity': VulnerabilityLevel.HIGH,
                'category': SecurityCategory.INPUT_VALIDATION,
                'description': 'ユーザー入力をサニタイズせずにDOMに挿入している可能性があります。',
                'remediation': '入力値をエスケープまたはサニタイズしてください。'
            },
            'weak_crypto': {
                'title': '弱い暗号化アルゴリズム',
                'threat_type': ThreatType.WEAK_ENCRYPTION,
                'severity': VulnerabilityLevel.MEDIUM,
                'category': SecurityCategory.DATA_PROTECTION,
                'description': '非推奨または弱い暗号化アルゴリズムが使用されています。',
                'remediation': 'SHA-256以上またはAES等の強固な暗号化を使用してください。'
            },
            'insecure_network': {
                'title': '安全でないネットワーク通信',
                'threat_type': ThreatType.DATA_LEAK,
                'severity': VulnerabilityLevel.MEDIUM,
                'category': SecurityCategory.NETWORK_SECURITY,
                'description': 'HTTP通信またはSSL証明書検証が無効化されています。',
                'remediation': 'HTTPS通信を使用し、SSL証明書を適切に検証してください。'
            }
        }
        
        if category not in vuln_map:
            return None
        
        vuln_info = vuln_map[category]
        vuln_id = self._generate_vuln_id(file_path, line_num, category)
        
        return SecurityVulnerability(
            vuln_id=vuln_id,
            title=vuln_info['title'],
            description=vuln_info['description'],
            category=vuln_info['category'],
            threat_type=vuln_info['threat_type'],
            severity=vuln_info['severity'],
            affected_files=[file_path],
            location=f"{file_path}:{line_num}",
            remediation=vuln_info['remediation'],
            references=[],
            detected_at=datetime.now()
        )
    
    def _create_function_vulnerability(self, func_name: str, file_path: str,
                                     line_num: int, line_content: str,
                                     language: str) -> Optional[SecurityVulnerability]:
        """危険関数脆弱性作成"""
        risk_functions = {
            'eval': {
                'title': '危険関数: eval()',
                'severity': VulnerabilityLevel.HIGH,
                'threat_type': ThreatType.COMMAND_INJECTION,
                'description': 'eval()関数はコードインジェクションの原因となり得ます。'
            },
            'exec': {
                'title': '危険関数: exec()',
                'severity': VulnerabilityLevel.HIGH,
                'threat_type': ThreatType.COMMAND_INJECTION,
                'description': 'exec()関数はコードインジェクションの原因となり得ます。'
            },
            'os.system': {
                'title': '危険関数: os.system()',
                'severity': VulnerabilityLevel.HIGH,
                'threat_type': ThreatType.COMMAND_INJECTION,
                'description': 'os.system()はコマンドインジェクションの原因となり得ます。'
            },
            'pickle.load': {
                'title': '危険関数: pickle.load()',
                'severity': VulnerabilityLevel.MEDIUM,
                'threat_type': ThreatType.COMMAND_INJECTION,
                'description': 'pickle.load()は信頼できないデータで実行すると危険です。'
            }
        }
        
        if func_name not in risk_functions:
            # 一般的な危険関数として処理
            func_info = {
                'title': f'潜在的な危険関数: {func_name}',
                'severity': VulnerabilityLevel.LOW,
                'threat_type': ThreatType.COMMAND_INJECTION,
                'description': f'{func_name}関数の使用が検出されました。適切な検証を確認してください。'
            }
        else:
            func_info = risk_functions[func_name]
        
        vuln_id = self._generate_vuln_id(file_path, line_num, f"func_{func_name}")
        
        return SecurityVulnerability(
            vuln_id=vuln_id,
            title=func_info['title'],
            description=func_info['description'],
            category=SecurityCategory.CODE_QUALITY,
            threat_type=func_info['threat_type'],
            severity=func_info['severity'],
            affected_files=[file_path],
            location=f"{file_path}:{line_num}",
            remediation="関数の使用箇所を見直し、より安全な代替手段を検討してください。",
            references=[],
            detected_at=datetime.now()
        )
    
    def _generate_vuln_id(self, file_path: str, line_num: int, category: str) -> str:
        """脆弱性ID生成"""
        content = f"{file_path}:{line_num}:{category}:{datetime.now()}"
        return f"VULN-{hashlib.md5(content.encode()).hexdigest()[:8].upper()}"


class DependencyScanner:
    """依存関係スキャナー"""
    
    def __init__(self):
        self.known_vulnerabilities = {
            # 既知の脆弱性データベース（簡易版）
            'requests': {
                '2.25.1': ['CVE-2021-33503'],
                '2.24.0': ['CVE-2021-33503']
            },
            'flask': {
                '1.0.0': ['CVE-2019-1010083'],
                '0.12.0': ['CVE-2018-1000656']
            }
        }
    
    def scan_requirements(self, requirements_file: str = "requirements.txt") -> List[SecurityVulnerability]:
        """requirements.txtスキャン"""
        vulnerabilities = []
        
        if not os.path.exists(requirements_file):
            return vulnerabilities
        
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements = f.readlines()
            
            for line_num, line in enumerate(requirements, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # パッケージ名とバージョン解析
                if '==' in line:
                    package, version = line.split('==')
                    package = package.strip()
                    version = version.strip()
                    
                    # 既知の脆弱性チェック
                    if package in self.known_vulnerabilities:
                        if version in self.known_vulnerabilities[package]:
                            cves = self.known_vulnerabilities[package][version]
                            
                            vuln = SecurityVulnerability(
                                vuln_id=f"DEP-{hashlib.md5(f'{package}{version}'.encode()).hexdigest()[:8].upper()}",
                                title=f"脆弱性のある依存関係: {package} {version}",
                                description=f"パッケージ {package} バージョン {version} には既知の脆弱性があります: {', '.join(cves)}",
                                category=SecurityCategory.DEPENDENCY,
                                threat_type=ThreatType.PRIVILEGE_ESCALATION,
                                severity=VulnerabilityLevel.HIGH,
                                affected_files=[requirements_file],
                                location=f"{requirements_file}:{line_num}",
                                remediation=f"{package}を最新の安全なバージョンにアップデートしてください。",
                                references=[f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={cve}" for cve in cves],
                                detected_at=datetime.now()
                            )
                            vulnerabilities.append(vuln)
        
        except Exception as e:
            logging.error(f"Requirements scan error: {e}")
        
        return vulnerabilities


class ConfigurationScanner:
    """設定ファイルスキャナー"""
    
    def __init__(self):
        self.insecure_configs = {
            'debug': ['true', 'True', '1', 'on'],
            'ssl_verify': ['false', 'False', '0', 'off'],
            'allow_origins': ['*'],
            'cors_origins': ['*'],
            'secret_key': ['debug', 'development', 'test', '123456']
        }
    
    def scan_config_files(self, config_dir: str = ".") -> List[SecurityVulnerability]:
        """設定ファイルスキャン"""
        vulnerabilities = []
        config_files = []
        
        # 設定ファイル検索
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                if any(file.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.ini', '.cfg', '.conf', '.env']):
                    config_files.append(os.path.join(root, file))
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # key=value 形式の設定チェック
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip().strip('"\'')
                        
                        for insecure_key, insecure_values in self.insecure_configs.items():
                            if insecure_key in key and value.lower() in [v.lower() for v in insecure_values]:
                                vuln = SecurityVulnerability(
                                    vuln_id=f"CFG-{hashlib.md5(f'{config_file}{line_num}{key}'.encode()).hexdigest()[:8].upper()}",
                                    title=f"安全でない設定: {key}",
                                    description=f"設定 '{key}' に安全でない値 '{value}' が設定されています。",
                                    category=SecurityCategory.CONFIGURATION,
                                    threat_type=ThreatType.INSECURE_DEFAULTS,
                                    severity=VulnerabilityLevel.MEDIUM,
                                    affected_files=[config_file],
                                    location=f"{config_file}:{line_num}",
                                    remediation=f"'{key}' の値をより安全な設定に変更してください。",
                                    references=[],
                                    detected_at=datetime.now()
                                )
                                vulnerabilities.append(vuln)
            
            except Exception as e:
                logging.error(f"Config file scan error for {config_file}: {e}")
        
        return vulnerabilities


class NetworkSecurityScanner:
    """ネットワークセキュリティスキャナー"""
    
    def __init__(self):
        self.common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]
        self.dangerous_ports = [21, 23, 25, 110, 143]  # 暗号化されていないプロトコル
    
    def scan_open_ports(self, target: str = "localhost") -> List[SecurityVulnerability]:
        """オープンポートスキャン"""
        vulnerabilities = []
        
        try:
            for port in self.common_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                
                result = sock.connect_ex((target, port))
                if result == 0:  # ポートが開いている
                    if port in self.dangerous_ports:
                        vuln = SecurityVulnerability(
                            vuln_id=f"NET-{target}-{port}",
                            title=f"危険なポートが開いています: {port}",
                            description=f"暗号化されていないプロトコルのポート {port} が開いています。",
                            category=SecurityCategory.NETWORK_SECURITY,
                            threat_type=ThreatType.DATA_LEAK,
                            severity=VulnerabilityLevel.MEDIUM,
                            affected_files=[],
                            location=f"{target}:{port}",
                            remediation="暗号化されたプロトコル（SSH、HTTPS等）の使用を検討してください。",
                            references=[],
                            detected_at=datetime.now()
                        )
                        vulnerabilities.append(vuln)
                
                sock.close()
        
        except Exception as e:
            logging.error(f"Network scan error: {e}")
        
        return vulnerabilities
    
    def check_ssl_configuration(self, hostname: str, port: int = 443) -> List[SecurityVulnerability]:
        """SSL設定チェック"""
        vulnerabilities = []
        
        if not HAS_REQUESTS:
            return vulnerabilities
        
        try:
            # SSL証明書情報取得
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    
                    # 弱い暗号化スイートチェック
                    if cipher and len(cipher) >= 3:
                        cipher_name = cipher[0]
                        if any(weak in cipher_name.upper() for weak in ['DES', 'RC4', 'MD5']):
                            vuln = SecurityVulnerability(
                                vuln_id=f"SSL-WEAK-{hostname}",
                                title="弱いSSL暗号化スイート",
                                description=f"弱い暗号化スイート {cipher_name} が使用されています。",
                                category=SecurityCategory.NETWORK_SECURITY,
                                threat_type=ThreatType.WEAK_ENCRYPTION,
                                severity=VulnerabilityLevel.MEDIUM,
                                affected_files=[],
                                location=f"{hostname}:{port}",
                                remediation="より強固な暗号化スイートを使用してください。",
                                references=[],
                                detected_at=datetime.now()
                            )
                            vulnerabilities.append(vuln)
                    
                    # 証明書有効期限チェック
                    if cert:
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.now()).days
                        
                        if days_until_expiry < 30:
                            severity = VulnerabilityLevel.HIGH if days_until_expiry < 7 else VulnerabilityLevel.MEDIUM
                            
                            vuln = SecurityVulnerability(
                                vuln_id=f"SSL-EXPIRY-{hostname}",
                                title="SSL証明書の期限切れが近い",
                                description=f"SSL証明書が {days_until_expiry} 日後に期限切れになります。",
                                category=SecurityCategory.NETWORK_SECURITY,
                                threat_type=ThreatType.DATA_LEAK,
                                severity=severity,
                                affected_files=[],
                                location=f"{hostname}:{port}",
                                remediation="SSL証明書を更新してください。",
                                references=[],
                                detected_at=datetime.now()
                            )
                            vulnerabilities.append(vuln)
        
        except Exception as e:
            logging.error(f"SSL check error for {hostname}: {e}")
        
        return vulnerabilities


class SecurityAssessmentEngine:
    """セキュリティ評価エンジン"""
    
    def __init__(self):
        self.file_scanner = FileSecurityScanner()
        self.dependency_scanner = DependencyScanner()
        self.config_scanner = ConfigurationScanner()
        self.network_scanner = NetworkSecurityScanner()
        
        # データベース初期化
        self._init_database()
        
        # スキャン結果履歴
        self.scan_history: List[SecurityScanResult] = []
    
    def _init_database(self):
        """データベース初期化"""
        self.db_path = "data/security_assessment.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                vuln_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                category TEXT,
                threat_type TEXT,
                severity INTEGER,
                affected_files TEXT,
                location TEXT,
                remediation TEXT,
                reference_links TEXT,
                detected_at DATETIME,
                status TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                scan_id TEXT PRIMARY KEY,
                scan_type TEXT,
                started_at DATETIME,
                completed_at DATETIME,
                files_scanned INTEGER,
                duration_ms REAL,
                total_vulns INTEGER,
                critical_vulns INTEGER,
                high_vulns INTEGER,
                medium_vulns INTEGER,
                low_vulns INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def comprehensive_security_scan(self, target_dir: str = ".") -> SecurityScanResult:
        """包括的セキュリティスキャン"""
        scan_id = f"SCAN-{int(time.time())}-{secrets.token_hex(4)}"
        started_at = datetime.now()
        
        logging.info(f"Starting comprehensive security scan: {scan_id}")
        
        all_vulnerabilities = []
        files_scanned = 0
        
        try:
            # 1. ファイルスキャン
            logging.info("Scanning files for security vulnerabilities...")
            for root, dirs, files in os.walk(target_dir):
                # システムディレクトリを除外
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv']]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.endswith(ext) for ext in ['.py', '.js', '.ts', '.sql', '.html', '.php']):
                        file_vulns = self.file_scanner.scan_file(file_path)
                        all_vulnerabilities.extend(file_vulns)
                        files_scanned += 1
            
            # 2. 依存関係スキャン
            logging.info("Scanning dependencies...")
            dep_vulns = self.dependency_scanner.scan_requirements()
            all_vulnerabilities.extend(dep_vulns)
            
            # 3. 設定ファイルスキャン
            logging.info("Scanning configuration files...")
            config_vulns = self.config_scanner.scan_config_files(target_dir)
            all_vulnerabilities.extend(config_vulns)
            
            # 4. ネットワークスキャン（オプション）
            logging.info("Performing network security checks...")
            network_vulns = self.network_scanner.scan_open_ports()
            all_vulnerabilities.extend(network_vulns)
            
        except Exception as e:
            logging.error(f"Security scan error: {e}")
        
        completed_at = datetime.now()
        duration_ms = (completed_at - started_at).total_seconds() * 1000
        
        # 結果サマリー生成
        severity_counts = Counter([v.severity for v in all_vulnerabilities])
        summary = {
            'total': len(all_vulnerabilities),
            'critical': severity_counts.get(VulnerabilityLevel.CRITICAL, 0),
            'high': severity_counts.get(VulnerabilityLevel.HIGH, 0),
            'medium': severity_counts.get(VulnerabilityLevel.MEDIUM, 0),
            'low': severity_counts.get(VulnerabilityLevel.LOW, 0),
            'info': severity_counts.get(VulnerabilityLevel.INFO, 0)
        }
        
        # スキャン結果作成
        scan_result = SecurityScanResult(
            scan_id=scan_id,
            scan_type="COMPREHENSIVE",
            started_at=started_at,
            completed_at=completed_at,
            vulnerabilities=all_vulnerabilities,
            files_scanned=files_scanned,
            duration_ms=duration_ms,
            summary=summary
        )
        
        # データベース保存
        self._save_scan_result(scan_result)
        self.scan_history.append(scan_result)
        
        logging.info(f"Security scan completed: {len(all_vulnerabilities)} vulnerabilities found")
        return scan_result
    
    def _save_scan_result(self, scan_result: SecurityScanResult):
        """スキャン結果保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # スキャン結果保存
        cursor.execute("""
            INSERT INTO scan_results 
            (scan_id, scan_type, started_at, completed_at, files_scanned, duration_ms,
             total_vulns, critical_vulns, high_vulns, medium_vulns, low_vulns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scan_result.scan_id,
            scan_result.scan_type,
            scan_result.started_at,
            scan_result.completed_at,
            scan_result.files_scanned,
            scan_result.duration_ms,
            scan_result.summary['total'],
            scan_result.summary['critical'],
            scan_result.summary['high'],
            scan_result.summary['medium'],
            scan_result.summary['low']
        ))
        
        # 脆弱性詳細保存
        for vuln in scan_result.vulnerabilities:
            cursor.execute("""
                INSERT OR REPLACE INTO vulnerabilities 
                (vuln_id, title, description, category, threat_type, severity,
                 affected_files, location, remediation, reference_links, detected_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vuln.vuln_id,
                vuln.title,
                vuln.description,
                vuln.category.value,
                vuln.threat_type.value,
                vuln.severity.value,
                json.dumps(vuln.affected_files),
                vuln.location,
                vuln.remediation,
                json.dumps(vuln.references),
                vuln.detected_at,
                vuln.status
            ))
        
        conn.commit()
        conn.close()
    
    def generate_security_report(self, scan_id: str) -> str:
        """セキュリティレポート生成"""
        scan_result = None
        for scan in self.scan_history:
            if scan.scan_id == scan_id:
                scan_result = scan
                break
        
        if not scan_result:
            return "スキャン結果が見つかりません。"
        
        # HTMLレポート生成
        report_html = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>セキュリティ評価レポート - {scan_result.scan_id}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #d32f2f; border-bottom: 3px solid #d32f2f; padding-bottom: 10px; }}
                h2 {{ color: #1976d2; margin-top: 30px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .summary-card {{ padding: 20px; border-radius: 8px; text-align: center; color: white; }}
                .critical {{ background: #d32f2f; }}
                .high {{ background: #ff5722; }}
                .medium {{ background: #ff9800; }}
                .low {{ background: #4caf50; }}
                .info {{ background: #2196f3; }}
                .vuln {{ margin: 15px 0; padding: 15px; border-left: 4px solid; background: #f9f9f9; }}
                .vuln-critical {{ border-left-color: #d32f2f; }}
                .vuln-high {{ border-left-color: #ff5722; }}
                .vuln-medium {{ border-left-color: #ff9800; }}
                .vuln-low {{ border-left-color: #4caf50; }}
                .location {{ font-family: monospace; background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }}
                .remediation {{ background: #e8f5e8; padding: 10px; border-radius: 4px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 セキュリティ評価レポート</h1>
                
                <div class="summary">
                    <div class="summary-card critical">
                        <h3>Critical</h3>
                        <div style="font-size: 2em; font-weight: bold;">{scan_result.summary['critical']}</div>
                    </div>
                    <div class="summary-card high">
                        <h3>High</h3>
                        <div style="font-size: 2em; font-weight: bold;">{scan_result.summary['high']}</div>
                    </div>
                    <div class="summary-card medium">
                        <h3>Medium</h3>
                        <div style="font-size: 2em; font-weight: bold;">{scan_result.summary['medium']}</div>
                    </div>
                    <div class="summary-card low">
                        <h3>Low</h3>
                        <div style="font-size: 2em; font-weight: bold;">{scan_result.summary['low']}</div>
                    </div>
                    <div class="summary-card info">
                        <h3>Info</h3>
                        <div style="font-size: 2em; font-weight: bold;">{scan_result.summary['info']}</div>
                    </div>
                </div>
                
                <h2>📊 スキャン詳細</h2>
                <ul>
                    <li><strong>スキャンID:</strong> {scan_result.scan_id}</li>
                    <li><strong>実行時間:</strong> {scan_result.started_at.strftime('%Y-%m-%d %H:%M:%S')} - {scan_result.completed_at.strftime('%H:%M:%S')}</li>
                    <li><strong>スキャン時間:</strong> {scan_result.duration_ms:.1f} ms</li>
                    <li><strong>対象ファイル数:</strong> {scan_result.files_scanned}</li>
                    <li><strong>検出脆弱性:</strong> {scan_result.summary['total']}件</li>
                </ul>
                
                <h2>🚨 検出された脆弱性</h2>
        """
        
        # 脆弱性を重要度別にソート
        sorted_vulns = sorted(scan_result.vulnerabilities, key=lambda v: v.severity.value, reverse=True)
        
        for vuln in sorted_vulns:
            severity_class = vuln.severity.name.lower()
            severity_name = {
                'CRITICAL': '緊急',
                'HIGH': '高',
                'MEDIUM': '中',
                'LOW': '低',
                'INFO': '情報'
            }.get(vuln.severity.name, vuln.severity.name)
            
            report_html += f"""
                <div class="vuln vuln-{severity_class}">
                    <h3>{vuln.title} <span style="background: #{severity_class}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">{severity_name}</span></h3>
                    <p><strong>脅威タイプ:</strong> {vuln.threat_type.value}</p>
                    <p><strong>カテゴリ:</strong> {vuln.category.value}</p>
                    <p><strong>場所:</strong> <span class="location">{vuln.location}</span></p>
                    <p>{vuln.description}</p>
                    <div class="remediation">
                        <strong>📋 対処方法:</strong><br>
                        {vuln.remediation}
                    </div>
                </div>
            """
        
        report_html += """
            </div>
        </body>
        </html>
        """
        
        return report_html
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """セキュリティメトリクス取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 最新スキャン結果
        cursor.execute("""
            SELECT * FROM scan_results ORDER BY completed_at DESC LIMIT 1
        """)
        latest_scan = cursor.fetchone()
        
        # トレンド分析
        cursor.execute("""
            SELECT scan_type, AVG(total_vulns), AVG(critical_vulns), AVG(high_vulns)
            FROM scan_results 
            WHERE completed_at >= datetime('now', '-30 days')
            GROUP BY scan_type
        """)
        trends = cursor.fetchall()
        
        # 脆弱性分布
        cursor.execute("""
            SELECT category, COUNT(*) FROM vulnerabilities 
            WHERE status = 'OPEN'
            GROUP BY category
        """)
        category_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'latest_scan': latest_scan,
            'monthly_trends': trends,
            'open_vulnerabilities_by_category': category_distribution,
            'total_scans': len(self.scan_history)
        }


# グローバルインスタンス
security_assessment_engine = SecurityAssessmentEngine()


def run_security_scan(target_dir: str = ".") -> SecurityScanResult:
    """セキュリティスキャン実行"""
    return security_assessment_engine.comprehensive_security_scan(target_dir)


def generate_security_report(scan_id: str) -> str:
    """セキュリティレポート生成"""
    return security_assessment_engine.generate_security_report(scan_id)


def get_security_metrics() -> Dict[str, Any]:
    """セキュリティメトリクス取得"""
    return security_assessment_engine.get_security_metrics()


if __name__ == "__main__":
    print("=== Security Assessment Test ===")
    
    # セキュリティスキャン実行
    print("Running comprehensive security scan...")
    scan_result = run_security_scan(".")
    
    print(f"Scan completed: {scan_result.scan_id}")
    print(f"Duration: {scan_result.duration_ms:.1f}ms")
    print(f"Files scanned: {scan_result.files_scanned}")
    print(f"Vulnerabilities found: {scan_result.summary['total']}")
    print(f"  Critical: {scan_result.summary['critical']}")
    print(f"  High: {scan_result.summary['high']}")
    print(f"  Medium: {scan_result.summary['medium']}")
    print(f"  Low: {scan_result.summary['low']}")
    
    # レポート生成
    print("\nGenerating security report...")
    report_html = generate_security_report(scan_result.scan_id)
    
    # レポートファイル保存
    report_file = f"data/security_report_{scan_result.scan_id}.html"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_html)
    
    print(f"Security report saved: {report_file}")
    
    # メトリクス確認
    metrics = get_security_metrics()
    print(f"\nSecurity metrics:")
    print(f"Total scans performed: {metrics['total_scans']}")
    print(f"Open vulnerabilities by category: {metrics['open_vulnerabilities_by_category']}")
    
    print("\nSecurity assessment test completed!")