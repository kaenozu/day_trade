#!/usr/bin/env python3
"""
セキュリティ設定管理
Issue #419: セキュリティ強化 - セキュリティポリシーとガイドライン

セキュリティポリシー、設定管理、コンプライアンスチェック、
セキュリティ監視を統合した設定管理システム
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


class SecurityLevel(Enum):
    """セキュリティレベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """コンプライアンスフレームワーク"""

    NIST_CSF = "nist_csf"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    FINRA = "finra"
    JSOX = "jsox"  # 日本版SOX法


@dataclass
class PasswordPolicy:
    """パスワードポリシー"""

    min_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_symbols: bool = True
    max_age_days: int = 90
    history_count: int = 10
    lockout_threshold: int = 5
    lockout_duration_minutes: int = 30
    require_change_on_first_login: bool = True


@dataclass
class SessionPolicy:
    """セッションポリシー"""

    max_inactive_minutes: int = 30
    max_session_duration_hours: int = 8
    require_reauthentication_for_sensitive_ops: bool = True
    concurrent_session_limit: int = 3
    session_rotation_enabled: bool = True
    ip_binding_enabled: bool = True


@dataclass
class MFAPolicy:
    """多要素認証ポリシー"""

    required_for_admin: bool = True
    required_for_trading: bool = True
    required_for_high_risk_ops: bool = True
    totp_window_seconds: int = 30
    backup_codes_count: int = 10
    recovery_code_length: int = 8


@dataclass
class AuditPolicy:
    """監査ポリシー"""

    log_all_access: bool = True
    log_failed_attempts: bool = True
    log_privilege_escalation: bool = True
    log_data_access: bool = True
    retention_days: int = 365
    real_time_alerting: bool = True
    export_format: str = "json"


@dataclass
class DataProtectionPolicy:
    """データ保護ポリシー"""

    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_rotation_days: int = 90
    backup_encryption: bool = True
    data_classification_required: bool = True
    anonymization_for_analytics: bool = True
    retention_policies: Dict[str, int] = field(
        default_factory=lambda: {
            "trading_data": 2555,  # 7年
            "audit_logs": 2555,
            "user_data": 365,
            "system_logs": 90,
        }
    )


@dataclass
class NetworkSecurityPolicy:
    """ネットワークセキュリティポリシー"""

    firewall_enabled: bool = True
    intrusion_detection: bool = True
    rate_limiting: bool = True
    ip_whitelisting: bool = False
    vpn_required_for_admin: bool = True
    api_rate_limits: Dict[str, int] = field(
        default_factory=lambda: {"default": 100, "trading": 1000, "data_access": 500}
    )


@dataclass
class ComplianceRequirement:
    """コンプライアンス要件"""

    framework: ComplianceFramework
    requirements: List[str] = field(default_factory=list)
    implementation_status: str = "planning"  # planning, implementing, compliant, non_compliant
    last_assessment: Optional[datetime] = None
    next_assessment: Optional[datetime] = None
    evidence_documents: List[str] = field(default_factory=list)


class SecurityConfigManager:
    """
    セキュリティ設定管理システム

    セキュリティポリシー、コンプライアンス要件、
    設定検証、監視設定を統合管理
    """

    def __init__(
        self,
        config_path: str = "security/config",
        security_level: SecurityLevel = SecurityLevel.HIGH,
    ):
        """
        初期化

        Args:
            config_path: 設定保存パス
            security_level: セキュリティレベル
        """
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

        self.security_level = security_level

        # ポリシー初期化
        self.password_policy = PasswordPolicy()
        self.session_policy = SessionPolicy()
        self.mfa_policy = MFAPolicy()
        self.audit_policy = AuditPolicy()
        self.data_protection_policy = DataProtectionPolicy()
        self.network_security_policy = NetworkSecurityPolicy()

        # コンプライアンス要件
        self.compliance_requirements: Dict[ComplianceFramework, ComplianceRequirement] = {}

        # 設定読み込み
        self._load_configuration()
        self._initialize_compliance_requirements()

        # セキュリティレベル適用
        self._apply_security_level()

        logger.info(f"SecurityConfigManager初期化完了 (レベル: {security_level.value})")

    def _load_configuration(self):
        """設定読み込み"""
        config_file = self.config_path / "security_config.json"
        if config_file.exists():
            try:
                with open(config_file, encoding="utf-8") as f:
                    data = json.load(f)

                # ポリシー読み込み
                if "password_policy" in data:
                    self._load_password_policy(data["password_policy"])

                if "session_policy" in data:
                    self._load_session_policy(data["session_policy"])

                if "mfa_policy" in data:
                    self._load_mfa_policy(data["mfa_policy"])

                if "audit_policy" in data:
                    self._load_audit_policy(data["audit_policy"])

                if "data_protection_policy" in data:
                    self._load_data_protection_policy(data["data_protection_policy"])

                if "network_security_policy" in data:
                    self._load_network_security_policy(data["network_security_policy"])

                logger.info("セキュリティ設定読み込み完了")

            except Exception as e:
                logger.error(f"設定読み込みエラー: {e}")

    def _load_password_policy(self, data: Dict[str, Any]):
        """パスワードポリシー読み込み"""
        self.password_policy = PasswordPolicy(
            min_length=data.get("min_length", 12),
            require_uppercase=data.get("require_uppercase", True),
            require_lowercase=data.get("require_lowercase", True),
            require_numbers=data.get("require_numbers", True),
            require_symbols=data.get("require_symbols", True),
            max_age_days=data.get("max_age_days", 90),
            history_count=data.get("history_count", 10),
            lockout_threshold=data.get("lockout_threshold", 5),
            lockout_duration_minutes=data.get("lockout_duration_minutes", 30),
            require_change_on_first_login=data.get("require_change_on_first_login", True),
        )

    def _load_session_policy(self, data: Dict[str, Any]):
        """セッションポリシー読み込み"""
        self.session_policy = SessionPolicy(
            max_inactive_minutes=data.get("max_inactive_minutes", 30),
            max_session_duration_hours=data.get("max_session_duration_hours", 8),
            require_reauthentication_for_sensitive_ops=data.get(
                "require_reauthentication_for_sensitive_ops", True
            ),
            concurrent_session_limit=data.get("concurrent_session_limit", 3),
            session_rotation_enabled=data.get("session_rotation_enabled", True),
            ip_binding_enabled=data.get("ip_binding_enabled", True),
        )

    def _load_mfa_policy(self, data: Dict[str, Any]):
        """MFAポリシー読み込み"""
        self.mfa_policy = MFAPolicy(
            required_for_admin=data.get("required_for_admin", True),
            required_for_trading=data.get("required_for_trading", True),
            required_for_high_risk_ops=data.get("required_for_high_risk_ops", True),
            totp_window_seconds=data.get("totp_window_seconds", 30),
            backup_codes_count=data.get("backup_codes_count", 10),
            recovery_code_length=data.get("recovery_code_length", 8),
        )

    def _load_audit_policy(self, data: Dict[str, Any]):
        """監査ポリシー読み込み"""
        self.audit_policy = AuditPolicy(
            log_all_access=data.get("log_all_access", True),
            log_failed_attempts=data.get("log_failed_attempts", True),
            log_privilege_escalation=data.get("log_privilege_escalation", True),
            log_data_access=data.get("log_data_access", True),
            retention_days=data.get("retention_days", 365),
            real_time_alerting=data.get("real_time_alerting", True),
            export_format=data.get("export_format", "json"),
        )

    def _load_data_protection_policy(self, data: Dict[str, Any]):
        """データ保護ポリシー読み込み"""
        self.data_protection_policy = DataProtectionPolicy(
            encryption_at_rest=data.get("encryption_at_rest", True),
            encryption_in_transit=data.get("encryption_in_transit", True),
            key_rotation_days=data.get("key_rotation_days", 90),
            backup_encryption=data.get("backup_encryption", True),
            data_classification_required=data.get("data_classification_required", True),
            anonymization_for_analytics=data.get("anonymization_for_analytics", True),
            retention_policies=data.get(
                "retention_policies",
                {
                    "trading_data": 2555,
                    "audit_logs": 2555,
                    "user_data": 365,
                    "system_logs": 90,
                },
            ),
        )

    def _load_network_security_policy(self, data: Dict[str, Any]):
        """ネットワークセキュリティポリシー読み込み"""
        self.network_security_policy = NetworkSecurityPolicy(
            firewall_enabled=data.get("firewall_enabled", True),
            intrusion_detection=data.get("intrusion_detection", True),
            rate_limiting=data.get("rate_limiting", True),
            ip_whitelisting=data.get("ip_whitelisting", False),
            vpn_required_for_admin=data.get("vpn_required_for_admin", True),
            api_rate_limits=data.get(
                "api_rate_limits", {"default": 100, "trading": 1000, "data_access": 500}
            ),
        )

    def _initialize_compliance_requirements(self):
        """コンプライアンス要件初期化"""
        # 日本の金融システム向けコンプライアンス
        self.compliance_requirements[ComplianceFramework.FINRA] = ComplianceRequirement(
            framework=ComplianceFramework.FINRA,
            requirements=[
                "取引記録の保持（3-6年）",
                "顧客識別・認証",
                "不正取引監視",
                "内部統制システム",
                "監査証跡の確保",
            ],
            implementation_status="implementing",
        )

        self.compliance_requirements[ComplianceFramework.JSOX] = ComplianceRequirement(
            framework=ComplianceFramework.JSOX,
            requirements=[
                "IT全社統制の確立",
                "アプリケーション統制",
                "ITセキュリティ管理",
                "変更管理プロセス",
                "データバックアップ・復旧",
            ],
            implementation_status="implementing",
        )

        self.compliance_requirements[ComplianceFramework.ISO27001] = ComplianceRequirement(
            framework=ComplianceFramework.ISO27001,
            requirements=[
                "情報セキュリティマネジメントシステム(ISMS)",
                "リスクアセスメント",
                "アクセス制御",
                "暗号化管理",
                "インシデント管理",
                "事業継続計画",
            ],
            implementation_status="planning",
        )

    def _apply_security_level(self):
        """セキュリティレベル適用"""
        if self.security_level == SecurityLevel.CRITICAL:
            # 最高レベルのセキュリティ設定
            self.password_policy.min_length = 16
            self.password_policy.max_age_days = 60
            self.password_policy.lockout_threshold = 3

            self.session_policy.max_inactive_minutes = 15
            self.session_policy.max_session_duration_hours = 4
            self.session_policy.concurrent_session_limit = 1

            self.mfa_policy.required_for_admin = True
            self.mfa_policy.required_for_trading = True
            self.mfa_policy.required_for_high_risk_ops = True

        elif self.security_level == SecurityLevel.HIGH:
            # 高レベルのセキュリティ設定（デフォルト）
            pass  # 既にデフォルト値が高セキュリティ

        elif self.security_level == SecurityLevel.MEDIUM:
            # 中レベルのセキュリティ設定
            self.password_policy.min_length = 10
            self.password_policy.max_age_days = 120

            self.session_policy.max_inactive_minutes = 60
            self.session_policy.concurrent_session_limit = 5

            self.mfa_policy.required_for_trading = False

        elif self.security_level == SecurityLevel.LOW:
            # 低レベルのセキュリティ設定（開発・テスト用）
            self.password_policy.min_length = 8
            self.password_policy.require_symbols = False
            self.password_policy.max_age_days = 180

            self.session_policy.max_inactive_minutes = 120
            self.session_policy.require_reauthentication_for_sensitive_ops = False

            self.mfa_policy.required_for_admin = False
            self.mfa_policy.required_for_trading = False
            self.mfa_policy.required_for_high_risk_ops = False

    def save_configuration(self):
        """設定保存"""
        config_file = self.config_path / "security_config.json"
        try:
            data = {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat(),
                "security_level": self.security_level.value,
                "password_policy": {
                    "min_length": self.password_policy.min_length,
                    "require_uppercase": self.password_policy.require_uppercase,
                    "require_lowercase": self.password_policy.require_lowercase,
                    "require_numbers": self.password_policy.require_numbers,
                    "require_symbols": self.password_policy.require_symbols,
                    "max_age_days": self.password_policy.max_age_days,
                    "history_count": self.password_policy.history_count,
                    "lockout_threshold": self.password_policy.lockout_threshold,
                    "lockout_duration_minutes": self.password_policy.lockout_duration_minutes,
                    "require_change_on_first_login": self.password_policy.require_change_on_first_login,
                },
                "session_policy": {
                    "max_inactive_minutes": self.session_policy.max_inactive_minutes,
                    "max_session_duration_hours": self.session_policy.max_session_duration_hours,
                    "require_reauthentication_for_sensitive_ops": self.session_policy.require_reauthentication_for_sensitive_ops,
                    "concurrent_session_limit": self.session_policy.concurrent_session_limit,
                    "session_rotation_enabled": self.session_policy.session_rotation_enabled,
                    "ip_binding_enabled": self.session_policy.ip_binding_enabled,
                },
                "mfa_policy": {
                    "required_for_admin": self.mfa_policy.required_for_admin,
                    "required_for_trading": self.mfa_policy.required_for_trading,
                    "required_for_high_risk_ops": self.mfa_policy.required_for_high_risk_ops,
                    "totp_window_seconds": self.mfa_policy.totp_window_seconds,
                    "backup_codes_count": self.mfa_policy.backup_codes_count,
                    "recovery_code_length": self.mfa_policy.recovery_code_length,
                },
                "audit_policy": {
                    "log_all_access": self.audit_policy.log_all_access,
                    "log_failed_attempts": self.audit_policy.log_failed_attempts,
                    "log_privilege_escalation": self.audit_policy.log_privilege_escalation,
                    "log_data_access": self.audit_policy.log_data_access,
                    "retention_days": self.audit_policy.retention_days,
                    "real_time_alerting": self.audit_policy.real_time_alerting,
                    "export_format": self.audit_policy.export_format,
                },
                "data_protection_policy": {
                    "encryption_at_rest": self.data_protection_policy.encryption_at_rest,
                    "encryption_in_transit": self.data_protection_policy.encryption_in_transit,
                    "key_rotation_days": self.data_protection_policy.key_rotation_days,
                    "backup_encryption": self.data_protection_policy.backup_encryption,
                    "data_classification_required": self.data_protection_policy.data_classification_required,
                    "anonymization_for_analytics": self.data_protection_policy.anonymization_for_analytics,
                    "retention_policies": self.data_protection_policy.retention_policies,
                },
                "network_security_policy": {
                    "firewall_enabled": self.network_security_policy.firewall_enabled,
                    "intrusion_detection": self.network_security_policy.intrusion_detection,
                    "rate_limiting": self.network_security_policy.rate_limiting,
                    "ip_whitelisting": self.network_security_policy.ip_whitelisting,
                    "vpn_required_for_admin": self.network_security_policy.vpn_required_for_admin,
                    "api_rate_limits": self.network_security_policy.api_rate_limits,
                },
            }

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("セキュリティ設定保存完了")

        except Exception as e:
            logger.error(f"設定保存エラー: {e}")

    def validate_configuration(self) -> tuple[bool, List[str]]:
        """設定検証"""
        issues = []

        # パスワードポリシー検証
        if self.password_policy.min_length < 8:
            issues.append("パスワード最小長が8文字未満です")

        if not any(
            [
                self.password_policy.require_uppercase,
                self.password_policy.require_lowercase,
                self.password_policy.require_numbers,
                self.password_policy.require_symbols,
            ]
        ):
            issues.append("パスワードに文字種要件が設定されていません")

        # セッションポリシー検証
        if self.session_policy.max_inactive_minutes > 240:  # 4時間
            issues.append("セッション非アクティブタイムアウトが長すぎます")

        if self.session_policy.concurrent_session_limit > 10:
            issues.append("同時セッション数上限が多すぎます")

        # MFAポリシー検証
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            if not self.mfa_policy.required_for_admin:
                issues.append("高セキュリティレベルでは管理者MFAが必須です")

        # データ保護ポリシー検証
        if not self.data_protection_policy.encryption_at_rest:
            issues.append("保存時暗号化が無効です")

        if not self.data_protection_policy.encryption_in_transit:
            issues.append("転送時暗号化が無効です")

        return len(issues) == 0, issues

    def get_compliance_status(self) -> Dict[str, Any]:
        """コンプライアンス状況取得"""
        status = {}

        for framework, requirement in self.compliance_requirements.items():
            status[framework.value] = {
                "requirements_count": len(requirement.requirements),
                "implementation_status": requirement.implementation_status,
                "last_assessment": (
                    requirement.last_assessment.isoformat() if requirement.last_assessment else None
                ),
                "next_assessment": (
                    requirement.next_assessment.isoformat() if requirement.next_assessment else None
                ),
                "requirements": requirement.requirements,
            }

        return status

    def update_compliance_status(
        self,
        framework: ComplianceFramework,
        status: str,
        assessment_notes: Optional[str] = None,
    ):
        """コンプライアンス状況更新"""
        if framework in self.compliance_requirements:
            requirement = self.compliance_requirements[framework]
            requirement.implementation_status = status
            requirement.last_assessment = datetime.utcnow()

            # 次回評価日設定（年次）
            requirement.next_assessment = datetime.utcnow() + timedelta(days=365)

            logger.info(f"コンプライアンス状況更新: {framework.value} -> {status}")

    def generate_security_policy_document(self) -> str:
        """セキュリティポリシードキュメント生成"""
        doc = f"""# セキュリティポリシー
生成日時: {datetime.utcnow().isoformat()}
セキュリティレベル: {self.security_level.value.upper()}

## 1. パスワードポリシー
- 最小文字数: {self.password_policy.min_length}文字
- 大文字必須: {'はい' if self.password_policy.require_uppercase else 'いいえ'}
- 小文字必須: {'はい' if self.password_policy.require_lowercase else 'いいえ'}
- 数字必須: {'はい' if self.password_policy.require_numbers else 'いいえ'}
- 記号必須: {'はい' if self.password_policy.require_symbols else 'いいえ'}
- パスワード有効期限: {self.password_policy.max_age_days}日
- 履歴保持数: {self.password_policy.history_count}個
- ロックアウト閾値: {self.password_policy.lockout_threshold}回
- ロックアウト継続時間: {self.password_policy.lockout_duration_minutes}分

## 2. セッション管理ポリシー
- 非アクティブタイムアウト: {self.session_policy.max_inactive_minutes}分
- 最大セッション継続時間: {self.session_policy.max_session_duration_hours}時間
- 機密操作での再認証: {'必須' if self.session_policy.require_reauthentication_for_sensitive_ops else '任意'}
- 同時セッション上限: {self.session_policy.concurrent_session_limit}個
- セッションローテーション: {'有効' if self.session_policy.session_rotation_enabled else '無効'}
- IPアドレス固定: {'有効' if self.session_policy.ip_binding_enabled else '無効'}

## 3. 多要素認証(MFA)ポリシー
- 管理者MFA必須: {'はい' if self.mfa_policy.required_for_admin else 'いいえ'}
- 取引MFA必須: {'はい' if self.mfa_policy.required_for_trading else 'いいえ'}
- 高リスク操作MFA必須: {'はい' if self.mfa_policy.required_for_high_risk_ops else 'いいえ'}
- TOTP許容時間: {self.mfa_policy.totp_window_seconds}秒
- バックアップコード数: {self.mfa_policy.backup_codes_count}個

## 4. 監査ポリシー
- 全アクセスログ: {'有効' if self.audit_policy.log_all_access else '無効'}
- 失敗ログ: {'有効' if self.audit_policy.log_failed_attempts else '無効'}
- 権限昇格ログ: {'有効' if self.audit_policy.log_privilege_escalation else '無効'}
- データアクセスログ: {'有効' if self.audit_policy.log_data_access else '無効'}
- ログ保持期間: {self.audit_policy.retention_days}日
- リアルタイムアラート: {'有効' if self.audit_policy.real_time_alerting else '無効'}

## 5. データ保護ポリシー
- 保存時暗号化: {'必須' if self.data_protection_policy.encryption_at_rest else '任意'}
- 転送時暗号化: {'必須' if self.data_protection_policy.encryption_in_transit else '任意'}
- キーローテーション: {self.data_protection_policy.key_rotation_days}日
- バックアップ暗号化: {'有効' if self.data_protection_policy.backup_encryption else '無効'}
- データ分類: {'必須' if self.data_protection_policy.data_classification_required else '任意'}

### データ保持期間
"""

        for data_type, days in self.data_protection_policy.retention_policies.items():
            doc += f"- {data_type}: {days}日\n"

        doc += f"""
## 6. ネットワークセキュリティポリシー
- ファイアウォール: {'有効' if self.network_security_policy.firewall_enabled else '無効'}
- 侵入検知: {'有効' if self.network_security_policy.intrusion_detection else '無効'}
- レート制限: {'有効' if self.network_security_policy.rate_limiting else '無効'}
- IPホワイトリスト: {'有効' if self.network_security_policy.ip_whitelisting else '無効'}
- 管理者VPN必須: {'はい' if self.network_security_policy.vpn_required_for_admin else 'いいえ'}

### APIレート制限
"""

        for api_type, limit in self.network_security_policy.api_rate_limits.items():
            doc += f"- {api_type}: {limit}リクエスト/分\n"

        doc += """
## 7. コンプライアンス要件
"""

        for framework, requirement in self.compliance_requirements.items():
            doc += f"""
### {framework.value.upper()}
実装状況: {requirement.implementation_status}
要件数: {len(requirement.requirements)}個

要件一覧:
"""
            for req in requirement.requirements:
                doc += f"- {req}\n"

        return doc

    def get_security_report(self) -> Dict[str, Any]:
        """セキュリティレポート生成"""
        is_valid, validation_issues = self.validate_configuration()

        report = {
            "report_id": f"security-config-report-{int(datetime.utcnow().timestamp())}",
            "generated_at": datetime.utcnow().isoformat(),
            "security_level": self.security_level.value,
            "configuration_valid": is_valid,
            "validation_issues": validation_issues,
            "compliance_status": self.get_compliance_status(),
            "policy_summary": {
                "password_min_length": self.password_policy.min_length,
                "session_timeout_minutes": self.session_policy.max_inactive_minutes,
                "mfa_required_for_admin": self.mfa_policy.required_for_admin,
                "encryption_at_rest": self.data_protection_policy.encryption_at_rest,
                "audit_logging_enabled": self.audit_policy.log_all_access,
            },
            "recommendations": self._generate_recommendations(validation_issues),
        }

        return report

    def _generate_recommendations(self, validation_issues: List[str]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        if validation_issues:
            recommendations.append(f"🔧 {len(validation_issues)}件の設定問題を修正してください")

        if self.security_level == SecurityLevel.LOW:
            recommendations.append("🔒 本番環境ではセキュリティレベルをHIGH以上に設定してください")

        if not self.mfa_policy.required_for_admin:
            recommendations.append("🔐 管理者アカウントには多要素認証を必須にしてください")

        # コンプライアンス状況チェック
        non_compliant_frameworks = [
            framework.value
            for framework, req in self.compliance_requirements.items()
            if req.implementation_status in ["planning", "non_compliant"]
        ]

        if non_compliant_frameworks:
            recommendations.append(
                f"📋 {', '.join(non_compliant_frameworks)}のコンプライアンス実装を完了してください"
            )

        if not recommendations:
            recommendations.append("✅ セキュリティ設定は適切に構成されています")

        return recommendations


# Factory function
def create_security_config_manager(
    config_path: str = "security/config",
    security_level: SecurityLevel = SecurityLevel.HIGH,
) -> SecurityConfigManager:
    """SecurityConfigManagerファクトリ関数"""
    return SecurityConfigManager(config_path=config_path, security_level=security_level)


if __name__ == "__main__":
    # テスト実行
    def main():
        print("=== Issue #419 セキュリティ設定管理システムテスト ===")

        try:
            print("\n1. セキュリティ設定管理システム初期化")
            config_manager = create_security_config_manager()

            print(f"セキュリティレベル: {config_manager.security_level.value}")
            print(f"パスワード最小長: {config_manager.password_policy.min_length}文字")
            print(f"セッションタイムアウト: {config_manager.session_policy.max_inactive_minutes}分")

            print("\n2. 設定検証テスト")
            is_valid, issues = config_manager.validate_configuration()
            print(f"設定有効性: {is_valid}")
            if issues:
                print("検証問題:")
                for issue in issues:
                    print(f"  - {issue}")

            print("\n3. コンプライアンス状況")
            compliance_status = config_manager.get_compliance_status()
            for framework, status in compliance_status.items():
                print(
                    f"{framework}: {status['implementation_status']} ({status['requirements_count']}要件)"
                )

            print("\n4. セキュリティレベル変更テスト")
            # CRITICALレベルに変更
            config_manager.security_level = SecurityLevel.CRITICAL
            config_manager._apply_security_level()

            print(f"変更後パスワード最小長: {config_manager.password_policy.min_length}文字")
            print(
                f"変更後セッションタイムアウト: {config_manager.session_policy.max_inactive_minutes}分"
            )

            print("\n5. ポリシードキュメント生成")
            policy_doc = config_manager.generate_security_policy_document()
            print(f"ドキュメント長: {len(policy_doc)}文字")
            print("ドキュメントプレビュー:")
            print(policy_doc[:500] + "...")

            print("\n6. セキュリティレポート生成")
            report = config_manager.get_security_report()

            print(f"レポートID: {report['report_id']}")
            print(f"設定有効: {report['configuration_valid']}")
            print(f"問題数: {len(report['validation_issues'])}")
            print("推奨事項:")
            for rec in report["recommendations"]:
                print(f"  {rec}")

            print("\n7. 設定保存テスト")
            config_manager.save_configuration()
            print("設定保存完了")

        except Exception as e:
            print(f"テスト実行エラー: {e}")
            import traceback

            traceback.print_exc()

        print("\n=== セキュリティ設定管理システムテスト完了 ===")

    main()
