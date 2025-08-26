#!/usr/bin/env python3
"""
セキュリティモジュール
Issue #419: セキュリティ強化 + Phase G: 本番運用最適化フェーズ

企業レベルのセキュリティシステムを提供:
- 脆弱性管理・スキャン
- データ保護・暗号化
- アクセス制御・認証
- セキュリティ設定管理
- セキュリティテスト
- 統合管理システム
- セキュリティハードニング
"""

__version__ = "1.0.0"
__author__ = "DayTrade Security Team"

# Phase G: セキュリティハードニングシステム (既存)
from .security_hardening_system import (
    AttackType,
    IntrusionDetectionSystem,
    IPBlocklist,
    SecurityEvent,
    SecurityHardeningSystem,
    SecurityRule,
    ThreatAlert,
    ThreatLevel,
)

# Issue #419: セキュリティ強化コンポーネント
try:
    from .access_control import AccessControlManager, create_access_control_manager
    from .data_protection import DataProtectionManager, create_data_protection_manager
    from .secure_hash_utils import (
        SecureHashUtils,
        replace_md5_hash,
        secure_cache_key_generator,
    )
    from .security_config import SecurityConfigManager, create_security_config_manager
    from .security_manager import SecurityManager, create_security_manager
    # SecurityTestFramework - 新しいモジュール形式と旧形式両方をサポート
    try:
        # 新しいモジュール分割形式を優先
        from .framework import SecurityTestFramework, create_security_test_framework
        from .core import (
            SecurityTest,
            SecurityTestResult,
            TestCategory,
            TestSeverity,
            TestStatus,
        )
        from .authentication import PasswordSecurityTest, SessionSecurityTest
        from .validation import InputValidationTest
        from .encryption import EncryptionTest
        from .network import NetworkSecurityTest
        from .compliance import ComplianceTest
    except ImportError:
        # 旧形式（単一ファイル）からのフォールバック
        from .security_test_framework import (
            SecurityTestFramework,
            create_security_test_framework,
            SecurityTest,
            SecurityTestResult,
            TestCategory,
            TestSeverity,
            TestStatus,
            PasswordSecurityTest,
            SessionSecurityTest,
            InputValidationTest,
            EncryptionTest,
            NetworkSecurityTest,
            ComplianceTest,
        )
    from .vulnerability_manager import (
        VulnerabilityManager,
        create_vulnerability_manager,
    )

    # 便利な統合関数
    def initialize_security_system(
        base_path: str = "security", security_level: str = "high"
    ) -> SecurityManager:
        """
        セキュリティシステム初期化

        Args:
            base_path: セキュリティデータベースパス
            security_level: セキュリティレベル (low, medium, high, critical)

        Returns:
            初期化済みセキュリティマネージャー

        Example:
            >>> security_manager = initialize_security_system()
            >>> report = await security_manager.run_comprehensive_security_assessment()
        """
        return create_security_manager(
            base_path=base_path, security_level=security_level
        )

    def get_security_info():
        """セキュリティシステム情報取得"""
        return {
            "version": __version__,
            "components": [
                "SecurityManager - 統合セキュリティ管理",
                "VulnerabilityManager - 脆弱性管理・スキャン",
                "DataProtectionManager - データ保護・暗号化",
                "AccessControlManager - アクセス制御・認証",
                "SecurityConfigManager - セキュリティ設定管理",
                "SecurityTestFramework - セキュリティテスト",
                "SecurityHardeningSystem - セキュリティハードニング",
            ],
            "features": [
                "包括的脆弱性スキャン (pip-audit, safety, bandit)",
                "AES-256-GCM / Fernet 暗号化システム",
                "ロールベースアクセス制御 (RBAC)",
                "多要素認証 (TOTP)",
                "セッション管理・監査ログ",
                "セキュリティポリシー管理",
                "コンプライアンス要件 (NIST, ISO27001, SOX)",
                "自動セキュリティテスト",
                "セキュリティダッシュボード",
                "リアルタイム脅威検知・対応",
            ],
            "security_levels": ["low", "medium", "high", "critical"],
            "compliance_frameworks": [
                "NIST_CSF",
                "ISO27001",
                "SOC2",
                "GDPR",
                "PCI_DSS",
                "FINRA",
                "JSOX",
            ],
        }

    SECURITY_COMPONENTS_AVAILABLE = True

except ImportError as e:
    # Issue #419コンポーネントが不足している場合の対応
    import logging

    logging.warning(f"一部のセキュリティコンポーネントが利用できません: {e}")

    def initialize_security_system(*args, **kwargs):
        raise ImportError(
            "セキュリティシステムの初期化に失敗しました。依存関係を確認してください。"
        )

    def get_security_info():
        return {
            "version": __version__,
            "status": "limited",
            "error": "Some security components are not available due to missing dependencies",
            "available_components": ["SecurityHardeningSystem"],
        }

    SecurityManager = None
    VulnerabilityManager = None
    DataProtectionManager = None
    AccessControlManager = None
    SecurityConfigManager = None
    SecurityTestFramework = None

    SECURITY_COMPONENTS_AVAILABLE = False

# エクスポートリスト
__all__ = [
    # Phase G: セキュリティハードニング (既存)
    "SecurityHardeningSystem",
    "ThreatLevel",
    "AttackType",
    "SecurityEvent",
    "ThreatAlert",
    "SecurityRule",
    "IPBlocklist",
    "IntrusionDetectionSystem",
    # Issue #419: セキュリティ強化
    "SecurityManager",
    "VulnerabilityManager",
    "DataProtectionManager",
    "AccessControlManager",
    "SecurityConfigManager",
    "SecurityTestFramework",
    # SecurityTestFramework関連クラス
    "SecurityTest",
    "SecurityTestResult", 
    "TestCategory",
    "TestSeverity",
    "TestStatus",
    "PasswordSecurityTest",
    "SessionSecurityTest",
    "InputValidationTest",
    "EncryptionTest",
    "NetworkSecurityTest",
    "ComplianceTest",
    "SecureHashUtils",
    "secure_cache_key_generator",
    "replace_md5_hash",
    "create_security_manager",
    "create_vulnerability_manager",
    "create_data_protection_manager",
    "create_access_control_manager",
    "create_security_config_manager",
    "create_security_test_framework",
    "initialize_security_system",
    "get_security_info",
    "SECURITY_COMPONENTS_AVAILABLE",
]
