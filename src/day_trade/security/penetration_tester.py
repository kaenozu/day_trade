#!/usr/bin/env python3
"""
自動ペネトレーションテストシステム
Issue #435: 本番環境セキュリティ最終監査 - ペネトレーションテスト自動化

エンタープライズレベルの自動セキュリティテスト・脆弱性評価
"""

import asyncio
import hashlib
import json
import random
import re
import socket
import ssl
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class VulnerabilityType(Enum):
    """脆弱性種別"""

    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    CSRF = "cross_site_request_forgery"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_FLAW = "authorization_flaw"
    SENSITIVE_DATA_EXPOSURE = "sensitive_data_exposure"
    XML_EXTERNAL_ENTITIES = "xml_external_entities"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    USING_COMPONENTS_WITH_KNOWN_VULNERABILITIES = "known_vulnerable_components"
    INSUFFICIENT_LOGGING = "insufficient_logging"
    INJECTION = "injection"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    BUSINESS_LOGIC_FLAW = "business_logic_flaw"
    DDOS_VULNERABILITY = "ddos_vulnerability"
    INFORMATION_DISCLOSURE = "information_disclosure"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class SeverityLevel(Enum):
    """深刻度レベル"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class SecurityVulnerability:
    """セキュリティ脆弱性情報"""

    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    title: str
    description: str
    affected_url: str
    attack_vector: str
    evidence: str
    remediation: str
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    discovered_at: float = None

    def __post_init__(self):
        if self.discovered_at is None:
            self.discovered_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability_type": self.vulnerability_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_url": self.affected_url,
            "attack_vector": self.attack_vector,
            "evidence": self.evidence,
            "remediation": self.remediation,
            "cvss_score": self.cvss_score,
            "cve_id": self.cve_id,
            "discovered_at": self.discovered_at,
        }


@dataclass
class PenTestConfig:
    """ペネトレーションテスト設定"""

    target_base_url: str
    test_timeout: int = 300  # 5分
    max_concurrent_tests: int = 5
    include_aggressive_tests: bool = False
    authentication_enabled: bool = True
    test_credentials: Dict[str, str] = None
    excluded_paths: List[str] = None
    custom_headers: Dict[str, str] = None
    proxy_config: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.test_credentials is None:
            self.test_credentials = {}
        if self.excluded_paths is None:
            self.excluded_paths = []
        if self.custom_headers is None:
            self.custom_headers = {}


class WebSecurityTester:
    """Webアプリケーションセキュリティテスター"""

    def __init__(self, config: PenTestConfig):
        self.config = config
        self.session = requests.Session()
        self.vulnerabilities = []

        # HTTPセッション設定
        retry_strategy = Retry(
            total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # カスタムヘッダー設定
        self.session.headers.update(
            {
                "User-Agent": "PenTester/1.0 (Security Scanner)",
                **self.config.custom_headers,
            }
        )

        # プロキシ設定
        if self.config.proxy_config:
            self.session.proxies.update(self.config.proxy_config)

    async def test_sql_injection(
        self, target_url: str, parameters: Dict[str, str]
    ) -> List[SecurityVulnerability]:
        """SQLインジェクションテスト"""
        vulnerabilities = []

        sql_payloads = [
            "' OR '1'='1",
            "' OR '1'='1' --",
            "' OR '1'='1' #",
            "admin'--",
            "admin'/*",
            "' OR 1=1--",
            "'; WAITFOR DELAY '00:00:05'--",
            "'; SELECT SLEEP(5)--",
            "1' UNION SELECT null,null,version()--",
            "1' AND (SELECT SUBSTRING(@@version,1,1))='M'--",
        ]

        for param_name, param_value in parameters.items():
            for payload in sql_payloads:
                try:
                    test_params = parameters.copy()
                    test_params[param_name] = payload

                    start_time = time.time()
                    response = self.session.get(
                        target_url, params=test_params, timeout=10
                    )
                    response_time = time.time() - start_time

                    # SQLエラーパターン検出
                    error_patterns = [
                        r"SQL syntax.*MySQL",
                        r"Warning.*mysql_.*",
                        r"MySQLSyntaxErrorException",
                        r"valid MySQL result",
                        r"PostgreSQL.*ERROR",
                        r"Warning.*pg_.*",
                        r"valid PostgreSQL result",
                        r"SQLServer JDBC Driver",
                        r"SqlException",
                        r"Oracle error",
                        r"Oracle.*Driver",
                        r"Warning.*oci_.*",
                        r"Microsoft.*ODBC.*SQL Server.*Driver",
                    ]

                    for pattern in error_patterns:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            vulnerabilities.append(
                                SecurityVulnerability(
                                    vulnerability_type=VulnerabilityType.SQL_INJECTION,
                                    severity=SeverityLevel.HIGH,
                                    title="SQL Injection Vulnerability Detected",
                                    description=f"Parameter '{param_name}' appears vulnerable to SQL injection",
                                    affected_url=target_url,
                                    attack_vector=f"GET parameter: {param_name}={payload}",
                                    evidence=f"Database error pattern found: {pattern}",
                                    remediation="Use parameterized queries and input validation",
                                    cvss_score=8.5,
                                )
                            )
                            break

                    # タイムベース SQLインジェクション検出
                    if response_time > 5 and "SLEEP" in payload.upper():
                        vulnerabilities.append(
                            SecurityVulnerability(
                                vulnerability_type=VulnerabilityType.SQL_INJECTION,
                                severity=SeverityLevel.HIGH,
                                title="Time-based SQL Injection",
                                description=f"Parameter '{param_name}' vulnerable to time-based SQL injection",
                                affected_url=target_url,
                                attack_vector=f"Time-based payload: {payload}",
                                evidence=f"Response delayed by {response_time:.2f}s",
                                remediation="Use parameterized queries and input validation",
                                cvss_score=8.0,
                            )
                        )

                except Exception as e:
                    logger.debug(f"SQLインジェクションテストエラー: {e}")

        return vulnerabilities

    async def test_xss(
        self, target_url: str, parameters: Dict[str, str]
    ) -> List[SecurityVulnerability]:
        """XSS（クロスサイトスクリプティング）テスト"""
        vulnerabilities = []

        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<details open ontoggle=alert('XSS')>",
        ]

        for param_name, param_value in parameters.items():
            for payload in xss_payloads:
                try:
                    test_params = parameters.copy()
                    test_params[param_name] = payload

                    response = self.session.get(
                        target_url, params=test_params, timeout=10
                    )

                    # ペイロードが反映されているかチェック
                    if payload in response.text and response.headers.get(
                        "content-type", ""
                    ).startswith("text/html"):
                        # CSPヘッダーチェック
                        csp_header = response.headers.get("Content-Security-Policy", "")
                        severity = (
                            SeverityLevel.LOW if csp_header else SeverityLevel.HIGH
                        )

                        vulnerabilities.append(
                            SecurityVulnerability(
                                vulnerability_type=VulnerabilityType.XSS,
                                severity=severity,
                                title="Cross-Site Scripting (XSS) Vulnerability",
                                description=f"Parameter '{param_name}' vulnerable to XSS attacks",
                                affected_url=target_url,
                                attack_vector=f"GET parameter: {param_name}={payload}",
                                evidence="Payload reflected in response without proper encoding",
                                remediation="Implement proper input validation and output encoding",
                                cvss_score=(
                                    6.1 if severity == SeverityLevel.HIGH else 3.5
                                ),
                            )
                        )

                except Exception as e:
                    logger.debug(f"XSSテストエラー: {e}")

        return vulnerabilities

    async def test_authentication_bypass(
        self, target_url: str
    ) -> List[SecurityVulnerability]:
        """認証バイパステスト"""
        vulnerabilities = []

        bypass_attempts = [
            # パラメータ操作
            {"admin": "true"},
            {"user_id": "1"},
            {"role": "admin"},
            {"authenticated": "true"},
            {"login": "true"},
            # 共通認証バイパス
            {"username": "admin", "password": "admin"},
            {"username": "administrator", "password": "password"},
            {"username": "root", "password": "root"},
            {"username": "admin", "password": ""},
            {"username": "", "password": ""},
        ]

        for attempt in bypass_attempts:
            try:
                response = self.session.post(
                    urllib.parse.urljoin(target_url, "/login"),
                    data=attempt,
                    timeout=10,
                    allow_redirects=False,
                )

                # 成功の兆候をチェック
                success_indicators = [
                    response.status_code == 302,  # リダイレクト
                    "dashboard" in response.text.lower(),
                    "welcome" in response.text.lower(),
                    "logout" in response.text.lower(),
                    "profile" in response.text.lower(),
                ]

                if any(success_indicators):
                    vulnerabilities.append(
                        SecurityVulnerability(
                            vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                            severity=SeverityLevel.CRITICAL,
                            title="Authentication Bypass Vulnerability",
                            description="Weak authentication mechanism allows bypass",
                            affected_url=urllib.parse.urljoin(target_url, "/login"),
                            attack_vector=f"POST data: {attempt}",
                            evidence=f"Authentication bypassed with status {response.status_code}",
                            remediation="Implement strong authentication mechanisms",
                            cvss_score=9.8,
                        )
                    )

            except Exception as e:
                logger.debug(f"認証バイパステストエラー: {e}")

        return vulnerabilities

    async def test_sensitive_data_exposure(
        self, target_url: str
    ) -> List[SecurityVulnerability]:
        """機密データ露出テスト"""
        vulnerabilities = []

        sensitive_paths = [
            "/.env",
            "/config.php",
            "/config.json",
            "/web.config",
            "/app.config",
            "/.git/config",
            "/admin/config.php",
            "/backup.sql",
            "/database.sql",
            "/users.csv",
            "/passwords.txt",
            "/.htpasswd",
            "/phpinfo.php",
            "/info.php",
            "/server-status",
            "/server-info",
        ]

        for path in sensitive_paths:
            try:
                test_url = urllib.parse.urljoin(target_url, path)
                response = self.session.get(test_url, timeout=10)

                if response.status_code == 200:
                    # 機密データパターン検出
                    sensitive_patterns = [
                        r"password\s*[=:]\s*['\"]?[\w@#$%]+",
                        r"api[_-]?key\s*[=:]\s*['\"]?[\w-]+",
                        r"secret\s*[=:]\s*['\"]?[\w-]+",
                        r"token\s*[=:]\s*['\"]?[\w.-]+",
                        r"database.*host.*=.*",
                        r"DB_PASSWORD\s*=\s*['\"]?[\w@#$%]+",
                        r"mysql://.*:.*@",
                        r"postgresql://.*:.*@",
                        r"-----BEGIN (PRIVATE|RSA) KEY-----",
                    ]

                    for pattern in sensitive_patterns:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            vulnerabilities.append(
                                SecurityVulnerability(
                                    vulnerability_type=VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
                                    severity=SeverityLevel.HIGH,
                                    title="Sensitive Data Exposure",
                                    description=f"Sensitive information exposed at {path}",
                                    affected_url=test_url,
                                    attack_vector=f"Direct access to {path}",
                                    evidence=f"Sensitive data pattern found: {pattern}",
                                    remediation="Remove sensitive files from web-accessible directories",
                                    cvss_score=7.5,
                                )
                            )
                            break

            except Exception as e:
                logger.debug(f"機密データ露出テストエラー: {e}")

        return vulnerabilities

    async def test_security_headers(
        self, target_url: str
    ) -> List[SecurityVulnerability]:
        """セキュリティヘッダーテスト"""
        vulnerabilities = []

        try:
            response = self.session.get(target_url, timeout=10)
            headers = response.headers

            # 必要なセキュリティヘッダーチェック
            required_headers = {
                "X-Frame-Options": {
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Clickjacking protection missing",
                    "remediation": "Add X-Frame-Options header",
                },
                "X-Content-Type-Options": {
                    "severity": SeverityLevel.MEDIUM,
                    "description": "MIME type sniffing protection missing",
                    "remediation": "Add X-Content-Type-Options: nosniff",
                },
                "X-XSS-Protection": {
                    "severity": SeverityLevel.LOW,
                    "description": "XSS protection header missing",
                    "remediation": "Add X-XSS-Protection: 1; mode=block",
                },
                "Strict-Transport-Security": {
                    "severity": SeverityLevel.HIGH,
                    "description": "HTTPS enforcement missing",
                    "remediation": "Add Strict-Transport-Security header",
                },
                "Content-Security-Policy": {
                    "severity": SeverityLevel.MEDIUM,
                    "description": "Content Security Policy missing",
                    "remediation": "Implement Content-Security-Policy header",
                },
            }

            for header_name, config in required_headers.items():
                if header_name not in headers:
                    vulnerabilities.append(
                        SecurityVulnerability(
                            vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                            severity=config["severity"],
                            title=f"Missing Security Header: {header_name}",
                            description=config["description"],
                            affected_url=target_url,
                            attack_vector="Missing HTTP security header",
                            evidence=f"Response lacks {header_name} header",
                            remediation=config["remediation"],
                            cvss_score=self._get_cvss_for_severity(config["severity"]),
                        )
                    )

            # 危険なヘッダーチェック
            if "Server" in headers:
                vulnerabilities.append(
                    SecurityVulnerability(
                        vulnerability_type=VulnerabilityType.INFORMATION_DISCLOSURE,
                        severity=SeverityLevel.LOW,
                        title="Server Information Disclosure",
                        description="Server header reveals server information",
                        affected_url=target_url,
                        attack_vector="HTTP response headers",
                        evidence=f"Server: {headers['Server']}",
                        remediation="Remove or obfuscate Server header",
                        cvss_score=2.0,
                    )
                )

        except Exception as e:
            logger.debug(f"セキュリティヘッダーテストエラー: {e}")

        return vulnerabilities

    def _get_cvss_for_severity(self, severity: SeverityLevel) -> float:
        """深刻度からCVSSスコア算出"""
        cvss_mapping = {
            SeverityLevel.CRITICAL: 9.5,
            SeverityLevel.HIGH: 7.5,
            SeverityLevel.MEDIUM: 5.0,
            SeverityLevel.LOW: 2.5,
            SeverityLevel.INFORMATIONAL: 1.0,
        }
        return cvss_mapping.get(severity, 0.0)


class NetworkSecurityTester:
    """ネットワークセキュリティテスター"""

    def __init__(self, config: PenTestConfig):
        self.config = config

    async def test_ssl_tls_configuration(
        self, hostname: str, port: int = 443
    ) -> List[SecurityVulnerability]:
        """SSL/TLS設定テスト"""
        vulnerabilities = []

        try:
            context = ssl.create_default_context()

            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    # SSL/TLS情報取得
                    cipher = ssock.cipher()
                    version = ssock.version()
                    cert = ssock.getpeercert()

                    # 弱い暗号スイートチェック
                    weak_ciphers = ["RC4", "DES", "MD5", "NULL"]
                    if cipher and any(
                        weak_cipher in str(cipher) for weak_cipher in weak_ciphers
                    ):
                        vulnerabilities.append(
                            SecurityVulnerability(
                                vulnerability_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                                severity=SeverityLevel.HIGH,
                                title="Weak Cipher Suite",
                                description=f"Weak cipher suite in use: {cipher}",
                                affected_url=f"{hostname}:{port}",
                                attack_vector="SSL/TLS cipher negotiation",
                                evidence=f"Cipher: {cipher}",
                                remediation="Configure strong cipher suites only",
                                cvss_score=7.4,
                            )
                        )

                    # 古いTLSバージョンチェック
                    if version in ["TLSv1", "TLSv1.1", "SSLv2", "SSLv3"]:
                        vulnerabilities.append(
                            SecurityVulnerability(
                                vulnerability_type=VulnerabilityType.WEAK_CRYPTOGRAPHY,
                                severity=SeverityLevel.HIGH,
                                title="Outdated TLS Version",
                                description=f"Outdated TLS/SSL version: {version}",
                                affected_url=f"{hostname}:{port}",
                                attack_vector="SSL/TLS version negotiation",
                                evidence=f"Version: {version}",
                                remediation="Upgrade to TLS 1.2 or higher",
                                cvss_score=7.4,
                            )
                        )

                    # 証明書有効期限チェック
                    if cert:
                        not_after = cert.get("notAfter")
                        if not_after:
                            # 証明書の有効期限をチェック（簡略実装）
                            import datetime

                            try:
                                expiry_date = datetime.datetime.strptime(
                                    not_after, "%b %d %H:%M:%S %Y %Z"
                                )
                                days_until_expiry = (
                                    expiry_date - datetime.datetime.now()
                                ).days

                                if days_until_expiry < 30:
                                    severity = (
                                        SeverityLevel.HIGH
                                        if days_until_expiry < 7
                                        else SeverityLevel.MEDIUM
                                    )
                                    vulnerabilities.append(
                                        SecurityVulnerability(
                                            vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                                            severity=severity,
                                            title="Certificate Expiring Soon",
                                            description=f"SSL certificate expires in {days_until_expiry} days",
                                            affected_url=f"{hostname}:{port}",
                                            attack_vector="Certificate validation",
                                            evidence=f"Expiry: {not_after}",
                                            remediation="Renew SSL certificate",
                                            cvss_score=(
                                                5.0
                                                if severity == SeverityLevel.HIGH
                                                else 3.0
                                            ),
                                        )
                                    )
                            except Exception:
                                pass

        except Exception as e:
            logger.debug(f"SSL/TLSテストエラー: {e}")

        return vulnerabilities

    async def test_open_ports(self, hostname: str) -> List[SecurityVulnerability]:
        """オープンポートスキャン"""
        vulnerabilities = []

        common_ports = [
            21,
            22,
            23,
            25,
            53,
            80,
            110,
            135,
            139,
            143,
            443,
            993,
            995,
            1433,
            3306,
            3389,
            5432,
            6379,
            27017,
        ]

        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((hostname, port))
                sock.close()

                if result == 0:  # ポートが開いている
                    service_name = self._get_service_name(port)

                    # 危険なサービスのチェック
                    dangerous_services = {
                        21: ("FTP", SeverityLevel.MEDIUM),
                        23: ("Telnet", SeverityLevel.HIGH),
                        135: ("RPC", SeverityLevel.MEDIUM),
                        139: ("NetBIOS", SeverityLevel.MEDIUM),
                        1433: ("MSSQL", SeverityLevel.HIGH),
                        3306: ("MySQL", SeverityLevel.HIGH),
                        3389: ("RDP", SeverityLevel.HIGH),
                        5432: ("PostgreSQL", SeverityLevel.HIGH),
                        6379: ("Redis", SeverityLevel.HIGH),
                        27017: ("MongoDB", SeverityLevel.HIGH),
                    }

                    if port in dangerous_services:
                        service, severity = dangerous_services[port]
                        vulnerabilities.append(
                            SecurityVulnerability(
                                vulnerability_type=VulnerabilityType.SECURITY_MISCONFIGURATION,
                                severity=severity,
                                title=f"Exposed {service} Service",
                                description=f"{service} service exposed on port {port}",
                                affected_url=f"{hostname}:{port}",
                                attack_vector=f"Network connection to port {port}",
                                evidence=f"Port {port} ({service}) is accessible",
                                remediation=f"Secure {service} service or restrict access",
                                cvss_score=self._get_cvss_for_severity(severity),
                            )
                        )

            except Exception as e:
                logger.debug(f"ポートスキャンエラー {port}: {e}")

        return vulnerabilities

    def _get_service_name(self, port: int) -> str:
        """ポート番号からサービス名を取得"""
        service_map = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            135: "RPC",
            139: "NetBIOS",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            1433: "MSSQL",
            3306: "MySQL",
            3389: "RDP",
            5432: "PostgreSQL",
            6379: "Redis",
            27017: "MongoDB",
        }
        return service_map.get(port, f"Unknown-{port}")

    def _get_cvss_for_severity(self, severity: SeverityLevel) -> float:
        """深刻度からCVSSスコア算出"""
        cvss_mapping = {
            SeverityLevel.CRITICAL: 9.5,
            SeverityLevel.HIGH: 7.5,
            SeverityLevel.MEDIUM: 5.0,
            SeverityLevel.LOW: 2.5,
            SeverityLevel.INFORMATIONAL: 1.0,
        }
        return cvss_mapping.get(severity, 0.0)


class PenetrationTester:
    """統合ペネトレーションテストシステム"""

    def __init__(self, config: PenTestConfig):
        self.config = config
        self.web_tester = WebSecurityTester(config)
        self.network_tester = NetworkSecurityTester(config)
        self.vulnerabilities = []

        logger.info(f"ペネトレーションテスト初期化: {config.target_base_url}")

    async def run_comprehensive_pentest(self) -> Dict[str, Any]:
        """包括的ペネトレーションテスト実行"""
        logger.info("包括的ペネトレーションテスト開始")
        start_time = time.time()

        # テストの実行
        test_tasks = []

        # Webアプリケーションテスト
        test_tasks.append(self._run_web_security_tests())

        # ネットワークセキュリティテスト
        hostname = urllib.parse.urlparse(self.config.target_base_url).hostname
        if hostname:
            test_tasks.append(self._run_network_security_tests(hostname))

        # 並列実行
        test_results = await asyncio.gather(*test_tasks, return_exceptions=True)

        # 結果の集約
        all_vulnerabilities = []
        for result in test_results:
            if isinstance(result, list):
                all_vulnerabilities.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"テスト実行エラー: {result}")

        # 重複除去と優先度ソート
        unique_vulnerabilities = self._deduplicate_vulnerabilities(all_vulnerabilities)
        sorted_vulnerabilities = sorted(
            unique_vulnerabilities,
            key=lambda v: (v.severity.value, -v.cvss_score or 0),
            reverse=True,
        )

        execution_time = time.time() - start_time

        # レポート生成
        report = self._generate_pentest_report(sorted_vulnerabilities, execution_time)

        logger.info(
            f"ペネトレーションテスト完了: {len(sorted_vulnerabilities)}件の脆弱性発見"
        )

        return report

    async def _run_web_security_tests(self) -> List[SecurityVulnerability]:
        """Webセキュリティテスト実行"""
        vulnerabilities = []

        try:
            # 基本パラメータテスト
            test_params = {"id": "1", "user": "test", "search": "query"}

            # SQLインジェクションテスト
            sql_vulns = await self.web_tester.test_sql_injection(
                self.config.target_base_url, test_params
            )
            vulnerabilities.extend(sql_vulns)

            # XSSテスト
            xss_vulns = await self.web_tester.test_xss(
                self.config.target_base_url, test_params
            )
            vulnerabilities.extend(xss_vulns)

            # 認証バイパステスト
            if self.config.authentication_enabled:
                auth_vulns = await self.web_tester.test_authentication_bypass(
                    self.config.target_base_url
                )
                vulnerabilities.extend(auth_vulns)

            # 機密データ露出テスト
            sensitive_vulns = await self.web_tester.test_sensitive_data_exposure(
                self.config.target_base_url
            )
            vulnerabilities.extend(sensitive_vulns)

            # セキュリティヘッダーテスト
            header_vulns = await self.web_tester.test_security_headers(
                self.config.target_base_url
            )
            vulnerabilities.extend(header_vulns)

        except Exception as e:
            logger.error(f"Webセキュリティテストエラー: {e}")

        return vulnerabilities

    async def _run_network_security_tests(
        self, hostname: str
    ) -> List[SecurityVulnerability]:
        """ネットワークセキュリティテスト実行"""
        vulnerabilities = []

        try:
            # SSL/TLSテスト
            if self.config.target_base_url.startswith("https"):
                ssl_vulns = await self.network_tester.test_ssl_tls_configuration(
                    hostname
                )
                vulnerabilities.extend(ssl_vulns)

            # ポートスキャン（非攻撃的）
            port_vulns = await self.network_tester.test_open_ports(hostname)
            vulnerabilities.extend(port_vulns)

        except Exception as e:
            logger.error(f"ネットワークセキュリティテストエラー: {e}")

        return vulnerabilities

    def _deduplicate_vulnerabilities(
        self, vulnerabilities: List[SecurityVulnerability]
    ) -> List[SecurityVulnerability]:
        """脆弱性の重複除去"""
        unique_vulns = {}

        for vuln in vulnerabilities:
            # 重複キー生成
            key = f"{vuln.vulnerability_type.value}:{vuln.affected_url}:{vuln.attack_vector}"
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]

            if key_hash not in unique_vulns:
                unique_vulns[key_hash] = vuln
            elif vuln.severity.value < unique_vulns[key_hash].severity.value:
                # より深刻な脆弱性で上書き
                unique_vulns[key_hash] = vuln

        return list(unique_vulns.values())

    def _generate_pentest_report(
        self, vulnerabilities: List[SecurityVulnerability], execution_time: float
    ) -> Dict[str, Any]:
        """ペネトレーションテストレポート生成"""

        # 深刻度別カウント
        severity_counts = {}
        for severity in SeverityLevel:
            severity_counts[severity.value] = 0

        for vuln in vulnerabilities:
            severity_counts[vuln.severity.value] += 1

        # 脆弱性タイプ別カウント
        type_counts = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.vulnerability_type.value
            type_counts[vuln_type] = type_counts.get(vuln_type, 0) + 1

        # リスクスコア計算
        risk_score = self._calculate_risk_score(vulnerabilities)

        return {
            "target": self.config.target_base_url,
            "test_timestamp": time.time(),
            "execution_time_seconds": execution_time,
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "severity_breakdown": severity_counts,
                "vulnerability_types": type_counts,
                "risk_score": risk_score,
                "risk_level": self._get_risk_level(risk_score),
            },
            "vulnerabilities": [vuln.to_dict() for vuln in vulnerabilities],
            "recommendations": self._generate_recommendations(vulnerabilities),
            "compliance_status": self._assess_compliance(vulnerabilities),
        }

    def _calculate_risk_score(
        self, vulnerabilities: List[SecurityVulnerability]
    ) -> float:
        """リスクスコア計算"""
        if not vulnerabilities:
            return 0.0

        severity_weights = {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 7.0,
            SeverityLevel.MEDIUM: 4.0,
            SeverityLevel.LOW: 2.0,
            SeverityLevel.INFORMATIONAL: 0.5,
        }

        total_score = 0.0
        for vuln in vulnerabilities:
            weight = severity_weights.get(vuln.severity, 1.0)
            cvss_score = vuln.cvss_score or 5.0
            total_score += weight * (cvss_score / 10.0)

        # 0-100の範囲に正規化
        max_possible_score = len(vulnerabilities) * 10.0
        return min(100.0, (total_score / max_possible_score) * 100.0)

    def _get_risk_level(self, risk_score: float) -> str:
        """リスクレベル判定"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_recommendations(
        self, vulnerabilities: List[SecurityVulnerability]
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = set()

        vuln_types_found = {vuln.vulnerability_type for vuln in vulnerabilities}

        recommendation_map = {
            VulnerabilityType.SQL_INJECTION: "SQLインジェクション対策: パラメータ化クエリと入力検証を実装",
            VulnerabilityType.XSS: "XSS対策: 出力エンコーディングとCSPヘッダーを実装",
            VulnerabilityType.AUTHENTICATION_BYPASS: "認証強化: 多要素認証と強固な認証メカニズムを実装",
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE: "データ保護: 機密ファイルのアクセス制限と暗号化を実装",
            VulnerabilityType.SECURITY_MISCONFIGURATION: "セキュリティ設定: セキュリティヘッダーとサーバー設定を強化",
            VulnerabilityType.WEAK_CRYPTOGRAPHY: "暗号化強化: 強力な暗号化アルゴリズムとTLS1.2以上を使用",
        }

        for vuln_type in vuln_types_found:
            if vuln_type in recommendation_map:
                recommendations.add(recommendation_map[vuln_type])

        # 一般的な推奨事項
        if vulnerabilities:
            recommendations.add(
                "定期的なセキュリティ監査とペネトレーションテストの実施"
            )
            recommendations.add("セキュリティ教育とセキュアコーディング研修の実施")
            recommendations.add("インシデントレスポンス計画の策定と訓練")

        return list(recommendations)

    def _assess_compliance(
        self, vulnerabilities: List[SecurityVulnerability]
    ) -> Dict[str, Any]:
        """コンプライアンス評価"""
        critical_count = sum(
            1 for v in vulnerabilities if v.severity == SeverityLevel.CRITICAL
        )
        high_count = sum(1 for v in vulnerabilities if v.severity == SeverityLevel.HIGH)

        # 簡単なコンプライアンス評価
        pci_dss_compliant = critical_count == 0 and high_count <= 2
        iso27001_compliant = critical_count == 0 and high_count <= 5
        nist_compliant = critical_count == 0 and high_count <= 3

        return {
            "pci_dss": {
                "compliant": pci_dss_compliant,
                "critical_issues": critical_count,
                "high_issues": high_count,
            },
            "iso27001": {
                "compliant": iso27001_compliant,
                "critical_issues": critical_count,
                "high_issues": high_count,
            },
            "nist_csf": {
                "compliant": nist_compliant,
                "critical_issues": critical_count,
                "high_issues": high_count,
            },
        }


if __name__ == "__main__":
    # ペネトレーションテストデモ
    async def main():
        print("=== 自動ペネトレーションテストシステム ===")

        config = PenTestConfig(
            target_base_url="https://httpbin.org",
            test_timeout=60,
            max_concurrent_tests=3,
            include_aggressive_tests=False,
        )

        tester = PenetrationTester(config)

        print("ペネトレーションテスト実行中...")
        report = await tester.run_comprehensive_pentest()

        print("\n=== テスト結果 ===")
        print(f"対象: {report['target']}")
        print(f"実行時間: {report['execution_time_seconds']:.2f}秒")
        print(f"発見脆弱性: {report['summary']['total_vulnerabilities']}件")
        print(f"リスクスコア: {report['summary']['risk_score']:.1f}/100")
        print(f"リスクレベル: {report['summary']['risk_level']}")

        print("\n=== 深刻度別内訳 ===")
        for severity, count in report["summary"]["severity_breakdown"].items():
            if count > 0:
                print(f"{severity.upper()}: {count}件")

        print("\n=== 推奨事項 ===")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")

        print("\n=== コンプライアンス評価 ===")
        compliance = report["compliance_status"]
        for framework, status in compliance.items():
            compliant = "✅" if status["compliant"] else "❌"
            print(
                f"{framework.upper()}: {compliant} (Critical: {status['critical_issues']}, High: {status['high_issues']})"
            )

    # 実行
    asyncio.run(main())
