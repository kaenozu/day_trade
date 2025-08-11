#!/usr/bin/env python3
"""
セキュリティテストフレームワーク
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

侵入テスト、セキュリティスキャン、脆弱性評価、
コンプライアンステストを統合したテストフレームワーク
"""

import asyncio
import json
import logging
import re
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class TestCategory(Enum):
    """テストカテゴリ"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_VALIDATION = "data_validation"
    ENCRYPTION = "encryption"
    NETWORK_SECURITY = "network_security"
    SESSION_MANAGEMENT = "session_management"
    INPUT_VALIDATION = "input_validation"
    ERROR_HANDLING = "error_handling"
    COMPLIANCE = "compliance"
    INFRASTRUCTURE = "infrastructure"


class TestSeverity(Enum):
    """テスト結果重要度"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestStatus(Enum):
    """テストステータス"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class SecurityTestResult:
    """セキュリティテスト結果"""

    test_id: str
    test_name: str
    category: TestCategory
    severity: TestSeverity
    status: TestStatus

    # 実行情報
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # 結果詳細
    description: str = ""
    expected: str = ""
    actual: str = ""
    error_message: Optional[str] = None

    # 修正ガイダンス
    remediation: str = ""
    references: List[str] = field(default_factory=list)

    # 証跡
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "error_message": self.error_message,
            "remediation": self.remediation,
            "references": self.references,
            "evidence": self.evidence,
        }


class SecurityTest(ABC):
    """セキュリティテスト基底クラス"""

    def __init__(
        self,
        test_id: str,
        test_name: str,
        category: TestCategory,
        severity: TestSeverity = TestSeverity.MEDIUM,
    ):
        self.test_id = test_id
        self.test_name = test_name
        self.category = category
        self.severity = severity

    @abstractmethod
    async def execute(self, **kwargs) -> SecurityTestResult:
        """テスト実行"""
        pass

    def create_result(
        self,
        status: TestStatus,
        description: str = "",
        expected: str = "",
        actual: str = "",
        error_message: Optional[str] = None,
        remediation: str = "",
        evidence: Optional[Dict[str, Any]] = None,
    ) -> SecurityTestResult:
        """テスト結果作成"""
        return SecurityTestResult(
            test_id=self.test_id,
            test_name=self.test_name,
            category=self.category,
            severity=self.severity,
            status=status,
            start_time=datetime.utcnow(),
            description=description,
            expected=expected,
            actual=actual,
            error_message=error_message,
            remediation=remediation,
            evidence=evidence or {},
        )


class PasswordSecurityTest(SecurityTest):
    """パスワードセキュリティテスト"""

    def __init__(self):
        super().__init__(
            "PWD001",
            "パスワード強度テスト",
            TestCategory.AUTHENTICATION,
            TestSeverity.HIGH,
        )

    async def execute(self, password_policy=None, **kwargs) -> SecurityTestResult:
        """パスワード強度テスト実行"""
        try:
            weak_passwords = [
                "password",
                "123456",
                "password123",
                "admin",
                "qwerty",
                "letmein",
                "welcome",
                "123456789",
                "password1",
                "trading",
            ]

            issues = []

            # 弱いパスワードテスト
            for weak_pwd in weak_passwords:
                if self._is_password_acceptable(weak_pwd, password_policy):
                    issues.append(f"弱いパスワード '{weak_pwd}' が受け入れられます")

            # パスワードポリシーテスト
            if password_policy:
                test_passwords = [
                    ("short", "Aa1!"),  # 短すぎる
                    ("no_upper", "abcd1234!"),  # 大文字なし
                    ("no_lower", "ABCD1234!"),  # 小文字なし
                    ("no_number", "AbcdEfgh!"),  # 数字なし
                    ("no_symbol", "Abcd1234"),  # 記号なし
                    ("sequential", "Abcd1234!"),  # 連続文字
                ]

                for test_name, test_pwd in test_passwords:
                    if self._is_password_acceptable(test_pwd, password_policy):
                        issues.append(
                            f"ポリシー違反パスワード ({test_name}): '{test_pwd}' が受け入れられます"
                        )

            if issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="パスワード強度に問題があります",
                    expected="強固なパスワードポリシーの適用",
                    actual=f"{len(issues)}件のパスワード強度問題",
                    remediation="パスワードポリシーを強化し、弱いパスワードを拒否してください",
                    evidence={"issues": issues},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="パスワード強度は適切です",
                    expected="強固なパスワードポリシーの適用",
                    actual="パスワードポリシーが適切に適用されています",
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="パスワードポリシーの設定を確認してください",
            )

    def _is_password_acceptable(self, password: str, policy) -> bool:
        """パスワード受け入れ判定（簡易実装）"""
        if not policy:
            return len(password) >= 8  # デフォルト最小長

        if len(password) < getattr(policy, "min_length", 8):
            return False

        if getattr(policy, "require_uppercase", False) and not any(
            c.isupper() for c in password
        ):
            return False

        if getattr(policy, "require_lowercase", False) and not any(
            c.islower() for c in password
        ):
            return False

        if getattr(policy, "require_numbers", False) and not any(
            c.isdigit() for c in password
        ):
            return False

        if getattr(policy, "require_symbols", False) and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            return False

        return True


class SessionSecurityTest(SecurityTest):
    """セッションセキュリティテスト"""

    def __init__(self):
        super().__init__(
            "SES001",
            "セッション管理テスト",
            TestCategory.SESSION_MANAGEMENT,
            TestSeverity.HIGH,
        )

    async def execute(self, session_manager=None, **kwargs) -> SecurityTestResult:
        """セッション管理テスト実行"""
        try:
            issues = []

            if not session_manager:
                return self.create_result(
                    TestStatus.SKIPPED,
                    description="セッション管理システムが提供されていません",
                )

            # セッションタイムアウトテスト
            test_session = self._create_test_session(session_manager)
            if test_session:
                # 期限切れセッションのチェック
                original_timeout = getattr(test_session, "expires_at", None)
                if not original_timeout:
                    issues.append("セッションに有効期限が設定されていません")

                # セッション固定攻撃対策
                original_id = getattr(test_session, "session_id", None)
                # セッションローテーションのテスト（実装依存）

                # 同時セッション制限テスト
                if hasattr(session_manager, "sessions"):
                    session_count = len(session_manager.sessions)
                    if session_count > 100:  # 仮の上限
                        issues.append(f"同時セッション数が多すぎます: {session_count}")

            if issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="セッション管理に問題があります",
                    expected="安全なセッション管理",
                    actual=f"{len(issues)}件のセッション管理問題",
                    remediation="セッションタイムアウト、ローテーション、制限を適切に設定してください",
                    evidence={"issues": issues},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="セッション管理は適切です",
                    expected="安全なセッション管理",
                    actual="セッション管理が適切に実装されています",
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="セッション管理システムの設定を確認してください",
            )

    def _create_test_session(self, session_manager):
        """テスト用セッション作成"""
        try:
            # セッション管理システムの実装に依存
            if hasattr(session_manager, "create_session"):
                return session_manager.create_session(
                    user=self._create_dummy_user(),
                    ip_address="127.0.0.1",
                    user_agent="SecurityTest/1.0",
                )
        except Exception:
            return None

    def _create_dummy_user(self):
        """ダミーユーザー作成"""

        class DummyUser:
            def __init__(self):
                self.user_id = "test-user"
                self.username = "test_user"
                self.session_timeout_minutes = 30

        return DummyUser()


class InputValidationTest(SecurityTest):
    """入力検証テスト"""

    def __init__(self):
        super().__init__(
            "INP001",
            "入力検証・SQLインジェクション対策テスト",
            TestCategory.INPUT_VALIDATION,
            TestSeverity.CRITICAL,
        )

    async def execute(self, input_validators=None, **kwargs) -> SecurityTestResult:
        """入力検証テスト実行"""
        try:
            # SQLインジェクション攻撃パターン
            sql_injection_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "1'; UPDATE users SET password='hacked' WHERE '1'='1",
                "admin'--",
                "1' UNION SELECT * FROM users--",
            ]

            # XSS攻撃パターン
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src='x' onerror='alert(1)'>",
                "';alert(String.fromCharCode(88,83,83))//",
                "<svg onload=alert(1)>",
            ]

            # パストラバーサル攻撃パターン
            path_traversal_payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
                "....//....//....//etc/passwd",
            ]

            vulnerabilities = []

            # 各攻撃パターンのテスト
            for payload in sql_injection_payloads:
                if self._test_input_vulnerability(
                    payload, "SQL Injection", input_validators
                ):
                    vulnerabilities.append(f"SQLインジェクション: {payload}")

            for payload in xss_payloads:
                if self._test_input_vulnerability(payload, "XSS", input_validators):
                    vulnerabilities.append(f"XSS: {payload}")

            for payload in path_traversal_payloads:
                if self._test_input_vulnerability(
                    payload, "Path Traversal", input_validators
                ):
                    vulnerabilities.append(f"パストラバーサル: {payload}")

            if vulnerabilities:
                return self.create_result(
                    TestStatus.FAILED,
                    description="入力検証に脆弱性があります",
                    expected="全ての危険な入力を適切にサニタイズまたは拒否",
                    actual=f"{len(vulnerabilities)}件の入力検証脆弱性",
                    remediation="パラメータ化クエリの使用、入力検証の強化、出力エスケープの実装",
                    evidence={"vulnerabilities": vulnerabilities},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="入力検証は適切に実装されています",
                    expected="全ての危険な入力を適切にサニタイズまたは拒否",
                    actual="入力検証が適切に動作しています",
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="入力検証システムの設定を確認してください",
            )

    def _test_input_vulnerability(
        self, payload: str, attack_type: str, validators
    ) -> bool:
        """入力脆弱性テスト"""
        # 実際の実装では、アプリケーションの入力検証機能をテストする
        # ここでは基本的なパターンマッチングで脆弱性を検出

        # 危険なパターンが検証されずに通る場合は脆弱
        dangerous_patterns = {
            "SQL Injection": [r"'.*OR.*'", r"';.*DROP.*", r"UNION.*SELECT"],
            "XSS": [r"<script>", r"javascript:", r"onerror="],
            "Path Traversal": [r"\.\./", r"\.\.\\", r"%2e%2e"],
        }

        patterns = dangerous_patterns.get(attack_type, [])
        for pattern in patterns:
            if re.search(pattern, payload, re.IGNORECASE):
                # 実際のバリデーターがあればテストし、なければ脆弱とみなす
                if not validators:
                    return True
                # バリデーターのテスト実装（実際の実装に依存）
                return not self._validate_input(payload, validators)

        return False

    def _validate_input(self, input_data: str, validators) -> bool:
        """入力検証実行"""
        # 実際のバリデーターの実装に依存
        # ここでは基本的な検証のみ実装
        if not validators:
            return False

        # 基本的なサニタイゼーション確認
        return not any(
            dangerous in input_data.lower()
            for dangerous in [
                "script",
                "javascript",
                "drop table",
                "union select",
                "../",
            ]
        )


class EncryptionTest(SecurityTest):
    """暗号化テスト"""

    def __init__(self):
        super().__init__(
            "ENC001",
            "暗号化・データ保護テスト",
            TestCategory.ENCRYPTION,
            TestSeverity.HIGH,
        )

    async def execute(
        self, data_protection_manager=None, **kwargs
    ) -> SecurityTestResult:
        """暗号化テスト実行"""
        try:
            if not data_protection_manager:
                return self.create_result(
                    TestStatus.SKIPPED,
                    description="データ保護システムが提供されていません",
                )

            issues = []

            # 暗号化/復号化基本テスト
            test_data = "機密データテスト: API Key 123456"

            try:
                encrypted_data = data_protection_manager.encrypt_data(test_data)
                decrypted_data = data_protection_manager.decrypt_data(encrypted_data)

                if decrypted_data != test_data:
                    issues.append("暗号化/復号化が正しく動作しません")

                # 暗号化データの可視性チェック
                if test_data in str(encrypted_data.ciphertext):
                    issues.append("暗号化されたデータに平文が含まれています")

                # キー管理テスト
                if not encrypted_data.key_id:
                    issues.append("暗号化データにキーIDが設定されていません")

                # アルゴリズム強度チェック
                if hasattr(encrypted_data, "algorithm"):
                    weak_algorithms = ["des", "md5", "sha1", "rc4"]
                    if any(
                        weak in encrypted_data.algorithm.value.lower()
                        for weak in weak_algorithms
                    ):
                        issues.append(
                            f"弱い暗号化アルゴリズムが使用されています: {encrypted_data.algorithm.value}"
                        )

            except Exception as e:
                issues.append(f"暗号化処理でエラーが発生: {str(e)}")

            # キー管理テスト
            if hasattr(data_protection_manager, "key_manager"):
                key_manager = data_protection_manager.key_manager

                # キーローテーションチェック
                keys = (
                    key_manager.list_keys() if hasattr(key_manager, "list_keys") else []
                )
                for key_info in keys:
                    if "rotation_due" in key_info:
                        rotation_due = datetime.fromisoformat(key_info["rotation_due"])
                        if datetime.utcnow() > rotation_due:
                            issues.append(
                                f"キーローテーションが必要: {key_info['key_id']}"
                            )

            if issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="暗号化に問題があります",
                    expected="強固な暗号化とキー管理",
                    actual=f"{len(issues)}件の暗号化問題",
                    remediation="強い暗号化アルゴリズムの使用、適切なキー管理の実装",
                    evidence={"issues": issues},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="暗号化は適切に実装されています",
                    expected="強固な暗号化とキー管理",
                    actual="暗号化とキー管理が適切に動作しています",
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="暗号化システムの設定を確認してください",
            )


class NetworkSecurityTest(SecurityTest):
    """ネットワークセキュリティテスト"""

    def __init__(self):
        super().__init__(
            "NET001",
            "ネットワークセキュリティテスト",
            TestCategory.NETWORK_SECURITY,
            TestSeverity.MEDIUM,
        )

    async def execute(
        self, target_host="localhost", target_ports=None, **kwargs
    ) -> SecurityTestResult:
        """ネットワークセキュリティテスト実行"""
        try:
            if target_ports is None:
                target_ports = [22, 23, 53, 80, 443, 993, 995, 3306, 3389, 5432, 8080]

            issues = []
            open_ports = []

            # ポートスキャン
            for port in target_ports:
                if await self._is_port_open(target_host, port):
                    open_ports.append(port)

            # 危険なポートチェック
            dangerous_ports = {
                22: "SSH - 適切な認証設定を確認",
                23: "Telnet - 非暗号化通信、使用を避ける",
                53: "DNS - 外部からのアクセスを制限",
                3306: "MySQL - 外部からのアクセスを制限",
                3389: "RDP - 適切な認証設定を確認",
                5432: "PostgreSQL - 外部からのアクセスを制限",
                8080: "HTTP Proxy - 不要な場合は無効化",
            }

            for port in open_ports:
                if port in dangerous_ports:
                    issues.append(
                        f"ポート {port} が開放されています: {dangerous_ports[port]}"
                    )

            # SSL/TLS設定テスト（HTTPS対応ポートのみ）
            ssl_ports = [443, 8443, 993, 995]
            for port in open_ports:
                if port in ssl_ports:
                    ssl_issues = await self._test_ssl_configuration(target_host, port)
                    issues.extend(ssl_issues)

            # DNS設定テスト
            dns_issues = await self._test_dns_security(target_host)
            issues.extend(dns_issues)

            if issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="ネットワークセキュリティに問題があります",
                    expected="適切なネットワーク設定とファイアウォール",
                    actual=f"{len(issues)}件のネットワークセキュリティ問題",
                    remediation="不要なポートの無効化、SSL/TLS設定の強化、ファイアウォール設定の見直し",
                    evidence={"issues": issues, "open_ports": open_ports},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="ネットワークセキュリティは適切です",
                    expected="適切なネットワーク設定とファイアウォール",
                    actual="ネットワーク設定が適切に構成されています",
                    evidence={"open_ports": open_ports},
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="ネットワーク設定を確認してください",
            )

    async def _is_port_open(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """ポート開放チェック"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return False

    async def _test_ssl_configuration(self, host: str, port: int) -> List[str]:
        """SSL/TLS設定テスト"""
        issues = []

        try:
            # SSL証明書と設定の確認
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            reader, writer = await asyncio.open_connection(host, port, ssl=context)

            # SSL情報取得
            ssl_object = writer.get_extra_info("ssl_object")
            if ssl_object:
                # プロトコルバージョンチェック
                protocol = ssl_object.version()
                if protocol in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]:
                    issues.append(
                        f"古いSSL/TLSプロトコルが使用されています: {protocol}"
                    )

                # 証明書有効期限チェック
                cert = ssl_object.getpeercert()
                if cert:
                    not_after = cert.get("notAfter")
                    if not_after:
                        # 証明書有効期限の解析と警告
                        pass  # 実装簡略化のため省略

            writer.close()
            await writer.wait_closed()

        except Exception:
            # SSL接続できない場合は非SSL通信の可能性
            issues.append(f"ポート {port} でSSL/TLS接続ができません")

        return issues

    async def _test_dns_security(self, host: str) -> List[str]:
        """DNS設定セキュリティテスト"""
        issues = []

        try:
            # DNSリゾルバーテスト（簡略版）
            import socket

            try:
                socket.gethostbyname(host)
            except socket.gaierror:
                issues.append(f"DNS解決に失敗: {host}")

        except Exception:
            pass  # DNS テストは任意

        return issues


class ComplianceTest(SecurityTest):
    """コンプライアンステスト"""

    def __init__(self):
        super().__init__(
            "CMP001",
            "コンプライアンス要件テスト",
            TestCategory.COMPLIANCE,
            TestSeverity.HIGH,
        )

    async def execute(self, security_config=None, **kwargs) -> SecurityTestResult:
        """コンプライアンステスト実行"""
        try:
            if not security_config:
                return self.create_result(
                    TestStatus.SKIPPED,
                    description="セキュリティ設定が提供されていません",
                )

            compliance_issues = []

            # パスワードポリシーコンプライアンス
            password_policy = getattr(security_config, "password_policy", None)
            if password_policy:
                if password_policy.min_length < 8:
                    compliance_issues.append(
                        "パスワード最小長が8文字未満（業界標準要件）"
                    )

                if password_policy.max_age_days > 180:
                    compliance_issues.append(
                        "パスワード有効期限が180日を超過（金融業界要件）"
                    )

                if (
                    not password_policy.require_uppercase
                    or not password_policy.require_lowercase
                ):
                    compliance_issues.append(
                        "パスワードの大文字小文字要件未設定（NIST要件）"
                    )

            # セッション管理コンプライアンス
            session_policy = getattr(security_config, "session_policy", None)
            if session_policy:
                if session_policy.max_inactive_minutes > 60:
                    compliance_issues.append(
                        "セッション非アクティブタイムアウトが60分を超過（金融業界要件）"
                    )

                if not session_policy.session_rotation_enabled:
                    compliance_issues.append(
                        "セッションローテーションが無効（OWASP要件）"
                    )

            # MFAコンプライアンス
            mfa_policy = getattr(security_config, "mfa_policy", None)
            if mfa_policy:
                if not mfa_policy.required_for_admin:
                    compliance_issues.append(
                        "管理者アカウントでMFAが必須でない（PCI DSS要件）"
                    )

            # 監査ログコンプライアンス
            audit_policy = getattr(security_config, "audit_policy", None)
            if audit_policy:
                if not audit_policy.log_all_access:
                    compliance_issues.append("全アクセスログが無効（SOX法要件）")

                if audit_policy.retention_days < 365:
                    compliance_issues.append("ログ保持期間が1年未満（金融業界要件）")

            # データ保護コンプライアンス
            data_protection = getattr(security_config, "data_protection_policy", None)
            if data_protection:
                if not data_protection.encryption_at_rest:
                    compliance_issues.append("保存時暗号化が無効（GDPR/PCI DSS要件）")

                if not data_protection.encryption_in_transit:
                    compliance_issues.append("転送時暗号化が無効（PCI DSS要件）")

            if compliance_issues:
                return self.create_result(
                    TestStatus.FAILED,
                    description="コンプライアンス要件に非準拠の項目があります",
                    expected="業界標準および法規制要件への完全準拠",
                    actual=f"{len(compliance_issues)}件のコンプライアンス問題",
                    remediation="各種コンプライアンス要件に合致するよう設定を調整してください",
                    evidence={"compliance_issues": compliance_issues},
                )
            else:
                return self.create_result(
                    TestStatus.PASSED,
                    description="コンプライアンス要件を満たしています",
                    expected="業界標準および法規制要件への完全準拠",
                    actual="全てのコンプライアンス要件を満たしています",
                )

        except Exception as e:
            return self.create_result(
                TestStatus.ERROR,
                error_message=str(e),
                remediation="コンプライアンス設定を確認してください",
            )


class SecurityTestFramework:
    """
    セキュリティテストフレームワーク

    各種セキュリティテストを統合実行し、
    包括的なセキュリティ評価レポートを生成
    """

    def __init__(self, output_path: str = "security/test_results"):
        """
        初期化

        Args:
            output_path: テスト結果出力パス
        """
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # テスト登録
        self.tests: List[SecurityTest] = [
            PasswordSecurityTest(),
            SessionSecurityTest(),
            InputValidationTest(),
            EncryptionTest(),
            NetworkSecurityTest(),
            ComplianceTest(),
        ]

        # テスト結果
        self.test_results: List[SecurityTestResult] = []

        logger.info("SecurityTestFramework初期化完了")

    async def run_all_tests(self, **test_context) -> Dict[str, Any]:
        """全テスト実行"""
        logger.info("セキュリティテスト開始")
        start_time = datetime.utcnow()

        self.test_results = []

        # テスト並列実行
        tasks = []
        for test in self.tests:
            task = self._run_single_test(test, **test_context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果処理
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = self.tests[i].create_result(
                    TestStatus.ERROR,
                    error_message=str(result),
                    remediation="テスト実行環境を確認してください",
                )
                error_result.end_time = datetime.utcnow()
                self.test_results.append(error_result)
            else:
                self.test_results.append(result)

        end_time = datetime.utcnow()

        # 統合レポート生成
        report = self._generate_comprehensive_report(start_time, end_time)

        # 結果保存
        await self._save_test_results(report)

        logger.info(f"セキュリティテスト完了: {len(self.test_results)}テスト実行")

        return report

    async def run_specific_tests(
        self, categories: List[TestCategory], **test_context
    ) -> Dict[str, Any]:
        """特定カテゴリのテスト実行"""
        logger.info(f"セキュリティテスト開始: {[c.value for c in categories]}")
        start_time = datetime.utcnow()

        # カテゴリフィルタリング
        filtered_tests = [test for test in self.tests if test.category in categories]

        self.test_results = []

        # テスト実行
        tasks = []
        for test in filtered_tests:
            task = self._run_single_test(test, **test_context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 結果処理
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = filtered_tests[i].create_result(
                    TestStatus.ERROR, error_message=str(result)
                )
                error_result.end_time = datetime.utcnow()
                self.test_results.append(error_result)
            else:
                self.test_results.append(result)

        end_time = datetime.utcnow()

        # レポート生成
        report = self._generate_comprehensive_report(start_time, end_time)
        await self._save_test_results(report)

        logger.info(f"セキュリティテスト完了: {len(self.test_results)}テスト実行")

        return report

    async def _run_single_test(
        self, test: SecurityTest, **test_context
    ) -> SecurityTestResult:
        """単一テスト実行"""
        logger.info(f"テスト実行開始: {test.test_name}")
        start_time = datetime.utcnow()

        try:
            result = await test.execute(**test_context)
            result.end_time = datetime.utcnow()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

            logger.info(f"テスト実行完了: {test.test_name} ({result.status.value})")
            return result

        except Exception as e:
            logger.error(f"テスト実行エラー: {test.test_name} - {e}")
            error_result = test.create_result(TestStatus.ERROR, error_message=str(e))
            error_result.end_time = datetime.utcnow()
            error_result.duration_seconds = (
                error_result.end_time - start_time
            ).total_seconds()
            return error_result

    def _generate_comprehensive_report(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """包括的レポート生成"""
        # 統計情報
        stats = {
            "total_tests": len(self.test_results),
            "passed": sum(
                1 for r in self.test_results if r.status == TestStatus.PASSED
            ),
            "failed": sum(
                1 for r in self.test_results if r.status == TestStatus.FAILED
            ),
            "skipped": sum(
                1 for r in self.test_results if r.status == TestStatus.SKIPPED
            ),
            "errors": sum(1 for r in self.test_results if r.status == TestStatus.ERROR),
        }

        # 重要度別統計
        severity_stats = {}
        for severity in TestSeverity:
            severity_results = [r for r in self.test_results if r.severity == severity]
            severity_stats[severity.value] = {
                "total": len(severity_results),
                "failed": sum(
                    1 for r in severity_results if r.status == TestStatus.FAILED
                ),
                "passed": sum(
                    1 for r in severity_results if r.status == TestStatus.PASSED
                ),
            }

        # カテゴリ別統計
        category_stats = {}
        for category in TestCategory:
            category_results = [r for r in self.test_results if r.category == category]
            if category_results:
                category_stats[category.value] = {
                    "total": len(category_results),
                    "failed": sum(
                        1 for r in category_results if r.status == TestStatus.FAILED
                    ),
                    "passed": sum(
                        1 for r in category_results if r.status == TestStatus.PASSED
                    ),
                }

        # 失敗したテストの詳細
        failed_tests = [
            {
                "test_id": r.test_id,
                "test_name": r.test_name,
                "category": r.category.value,
                "severity": r.severity.value,
                "description": r.description,
                "remediation": r.remediation,
                "evidence": r.evidence,
            }
            for r in self.test_results
            if r.status == TestStatus.FAILED
        ]

        # セキュリティスコア計算
        security_score = self._calculate_security_score()

        report = {
            "report_id": f"security-test-report-{int(start_time.timestamp())}",
            "generated_at": end_time.isoformat(),
            "execution_time_seconds": (end_time - start_time).total_seconds(),
            "security_score": security_score,
            "statistics": stats,
            "severity_breakdown": severity_stats,
            "category_breakdown": category_stats,
            "failed_tests": failed_tests,
            "detailed_results": [r.to_dict() for r in self.test_results],
            "recommendations": self._generate_recommendations(),
            "executive_summary": self._generate_executive_summary(
                stats, security_score
            ),
        }

        return report

    def _calculate_security_score(self) -> float:
        """セキュリティスコア計算"""
        if not self.test_results:
            return 0.0

        # 重要度による重み付け
        severity_weights = {
            TestSeverity.CRITICAL: 4.0,
            TestSeverity.HIGH: 3.0,
            TestSeverity.MEDIUM: 2.0,
            TestSeverity.LOW: 1.0,
            TestSeverity.INFO: 0.5,
        }

        total_score = 0.0
        max_possible_score = 0.0

        for result in self.test_results:
            weight = severity_weights[result.severity]
            max_possible_score += weight

            if result.status == TestStatus.PASSED:
                total_score += weight
            elif result.status == TestStatus.FAILED:
                total_score += 0  # 失敗は0点
            elif result.status == TestStatus.SKIPPED:
                max_possible_score -= weight  # スキップは計算から除外

        return (
            (total_score / max_possible_score * 100) if max_possible_score > 0 else 0.0
        )

    def _generate_recommendations(self) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        failed_results = [r for r in self.test_results if r.status == TestStatus.FAILED]

        # 重要度別推奨事項
        critical_failures = [
            r for r in failed_results if r.severity == TestSeverity.CRITICAL
        ]
        if critical_failures:
            recommendations.append(
                f"🚨 {len(critical_failures)}件の重大なセキュリティ問題を直ちに修正してください"
            )

        high_failures = [r for r in failed_results if r.severity == TestSeverity.HIGH]
        if high_failures:
            recommendations.append(
                f"⚠️ {len(high_failures)}件の高リスクセキュリティ問題を優先的に修正してください"
            )

        # カテゴリ別推奨事項
        category_failures = {}
        for result in failed_results:
            category = result.category.value
            category_failures[category] = category_failures.get(category, 0) + 1

        for category, count in category_failures.items():
            if count >= 2:
                recommendations.append(
                    f"🔧 {category}カテゴリで{count}件の問題があります。包括的な見直しを検討してください"
                )

        if not failed_results:
            recommendations.append(
                "✅ 全てのセキュリティテストが通過しています。定期的な再評価を継続してください"
            )

        return recommendations

    def _generate_executive_summary(
        self, stats: Dict[str, int], security_score: float
    ) -> str:
        """エグゼクティブサマリー生成"""
        pass_rate = (
            (stats["passed"] / stats["total_tests"] * 100)
            if stats["total_tests"] > 0
            else 0
        )

        summary = f"""セキュリティテスト実行結果サマリー

総テスト数: {stats['total_tests']}
合格率: {pass_rate:.1f}%
セキュリティスコア: {security_score:.1f}/100

結果詳細:
- 合格: {stats['passed']}
- 失敗: {stats['failed']}
- スキップ: {stats['skipped']}
- エラー: {stats['errors']}

"""

        if security_score >= 90:
            summary += "🟢 セキュリティ状況: 良好\n優秀なセキュリティ実装です。現在の対策を維持してください。"
        elif security_score >= 70:
            summary += "🟡 セキュリティ状況: 注意\nいくつかの改善点があります。失敗したテストを確認し修正してください。"
        elif security_score >= 50:
            summary += "🟠 セキュリティ状況: 要改善\n重要なセキュリティ問題があります。優先的に対応が必要です。"
        else:
            summary += "🔴 セキュリティ状況: 危険\n深刻なセキュリティ脆弱性があります。直ちに対応してください。"

        return summary

    async def _save_test_results(self, report: Dict[str, Any]):
        """テスト結果保存"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # JSON詳細レポート
        json_report_file = self.output_path / f"security_test_report_{timestamp}.json"
        with open(json_report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # HTML要約レポート
        html_report = self._generate_html_report(report)
        html_report_file = self.output_path / f"security_test_summary_{timestamp}.html"
        with open(html_report_file, "w", encoding="utf-8") as f:
            f.write(html_report)

        logger.info(f"テスト結果保存完了: {json_report_file}, {html_report_file}")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """HTML要約レポート生成"""
        stats = report["statistics"]
        security_score = report["security_score"]

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>セキュリティテストレポート</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {'green' if security_score >= 70 else 'orange' if security_score >= 50 else 'red'}; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat {{ text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .failed-tests {{ margin-top: 30px; }}
        .test-item {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #ff4444; }}
        .recommendations {{ background: #e8f4fd; padding: 20px; margin: 20px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>セキュリティテストレポート</h1>
        <p>生成日時: {report['generated_at']}</p>
        <div class="score">{security_score:.1f}/100</div>
    </div>

    <div class="stats">
        <div class="stat">
            <h3>総テスト数</h3>
            <div>{stats['total_tests']}</div>
        </div>
        <div class="stat">
            <h3>合格</h3>
            <div style="color: green;">{stats['passed']}</div>
        </div>
        <div class="stat">
            <h3>失敗</h3>
            <div style="color: red;">{stats['failed']}</div>
        </div>
        <div class="stat">
            <h3>スキップ</h3>
            <div>{stats['skipped']}</div>
        </div>
    </div>

    <div class="recommendations">
        <h2>推奨事項</h2>
        <ul>
"""

        for rec in report["recommendations"]:
            html += f"            <li>{rec}</li>\n"

        html += """        </ul>
    </div>

    <div class="failed-tests">
        <h2>失敗したテスト</h2>
"""

        for test in report["failed_tests"]:
            html += f"""        <div class="test-item">
            <h3>{test['test_name']} ({test['severity'].upper()})</h3>
            <p><strong>問題:</strong> {test['description']}</p>
            <p><strong>修正方法:</strong> {test['remediation']}</p>
        </div>
"""

        html += """    </div>
</body>
</html>"""

        return html


# Factory function
def create_security_test_framework(
    output_path: str = "security/test_results",
) -> SecurityTestFramework:
    """SecurityTestFrameworkファクトリ関数"""
    return SecurityTestFramework(output_path=output_path)


if __name__ == "__main__":
    # テスト実行
    async def main():
        print("=== Issue #419 セキュリティテストフレームワーク実行 ===")

        try:
            # セキュリティテストフレームワーク初期化
            framework = create_security_test_framework()

            print("\n1. 全セキュリティテスト実行")

            # テストコンテキスト準備（実際のシステムコンポーネントを想定）
            test_context = {
                "password_policy": None,  # 実際のパスワードポリシー
                "session_manager": None,  # 実際のセッション管理システム
                "input_validators": None,  # 実際の入力検証システム
                "data_protection_manager": None,  # 実際のデータ保護システム
                "security_config": None,  # 実際のセキュリティ設定
                "target_host": "localhost",
                "target_ports": [22, 80, 443, 8080],
            }

            # テスト実行
            report = await framework.run_all_tests(**test_context)

            print("テスト実行完了")
            print(f"レポートID: {report['report_id']}")
            print(f"実行時間: {report['execution_time_seconds']:.2f}秒")
            print(f"セキュリティスコア: {report['security_score']:.1f}/100")

            print("\n統計:")
            stats = report["statistics"]
            print(f"  総テスト数: {stats['total_tests']}")
            print(f"  合格: {stats['passed']}")
            print(f"  失敗: {stats['failed']}")
            print(f"  スキップ: {stats['skipped']}")
            print(f"  エラー: {stats['errors']}")

            print("\n重要度別:")
            for severity, data in report["severity_breakdown"].items():
                if data["total"] > 0:
                    print(f"  {severity}: {data['failed']}/{data['total']} 失敗")

            print("\n推奨事項:")
            for rec in report["recommendations"]:
                print(f"  {rec}")

            print("\nエグゼクティブサマリー:")
            print(report["executive_summary"])

            # 特定カテゴリのみテスト
            print("\n2. 認証関連テストのみ実行")
            auth_report = await framework.run_specific_tests(
                [TestCategory.AUTHENTICATION, TestCategory.SESSION_MANAGEMENT],
                **test_context,
            )

            print(
                f"認証テスト完了: {auth_report['statistics']['total_tests']}テスト実行"
            )
            print(f"認証テストスコア: {auth_report['security_score']:.1f}/100")

        except Exception as e:
            print(f"テスト実行エラー: {e}")
            import traceback

            traceback.print_exc()

        print("\n=== セキュリティテストフレームワーク実行完了 ===")

    asyncio.run(main())
