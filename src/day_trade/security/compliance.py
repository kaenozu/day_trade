#!/usr/bin/env python3
"""
セキュリティテストフレームワーク - Compliance Module
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

コンプライアンス要件テスト、法規制対応テスト
"""

from .core import (
    SecurityTest,
    SecurityTestResult,
    TestCategory,
    TestSeverity,
    TestStatus,
)


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