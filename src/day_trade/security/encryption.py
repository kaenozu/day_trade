#!/usr/bin/env python3
"""
セキュリティテストフレームワーク - Encryption Module
Issue #419: セキュリティ強化 - セキュリティテストフレームワークの導入

暗号化テスト、データ保護テスト
"""

from datetime import datetime

from .core import (
    SecurityTest,
    SecurityTestResult,
    TestCategory,
    TestSeverity,
    TestStatus,
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