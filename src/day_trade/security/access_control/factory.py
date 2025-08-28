#!/usr/bin/env python3
"""
アクセス制御システム - ファクトリ関数とテストユーティリティ

このモジュールは、AccessControlManagerの作成と
基本的なテスト機能を提供します。
"""

import pyotp

from .access_control_manager import AccessControlManager
from .enums import Permission, UserRole


def create_access_control_manager(
    storage_path: str = "security/access_control", **config
) -> AccessControlManager:
    """
    AccessControlManagerファクトリ関数
    
    Args:
        storage_path: データ保存パス
        **config: 追加設定
        
    Returns:
        AccessControlManager: 初期化済みのマネージャー
    """
    return AccessControlManager(storage_path=storage_path, **config)


def run_access_control_test():
    """
    アクセス制御システムの基本テスト実行
    
    システムの基本機能をテストして、正常に動作することを確認します。
    """
    print("=== Issue #419 アクセス制御・認証システムテスト ===")

    manager = None
    try:
        # アクセス制御システム初期化
        manager = create_access_control_manager()

        print("\n1. システム初期化状態")
        print(f"登録ユーザー数: {len(manager.users)}")
        print(f"アクティブセッション数: {len(manager.sessions)}")

        print("\n2. テストユーザー作成")
        test_user = manager.create_user(
            username="test_trader",
            email="trader@test.com",
            password="SecureTrading2024!",
            role=UserRole.TRADER,
        )

        if test_user:
            print(
                f"テストユーザー作成成功: {test_user.username} ({test_user.role.value})"
            )

            print("\n3. MFA設定テスト")
            success, message, qr_code = manager.setup_mfa(test_user.user_id)
            if success:
                print("MFA設定開始成功")
                print(f"QRコード生成: {len(qr_code) if qr_code else 0} bytes")

                # テスト用のTOTPコード生成
                totp = pyotp.TOTP(test_user.totp_secret)
                test_code = totp.now()

                mfa_success, mfa_message = manager.enable_mfa(
                    test_user.user_id, test_code
                )
                print(f"MFA有効化: {mfa_success} - {mfa_message}")

            print("\n4. 認証テスト")
            auth_success, auth_message, auth_user = manager.authenticate_user(
                username="test_trader",
                password="SecureTrading2024!",
                ip_address="127.0.0.1",
                user_agent="TestClient/1.0",
                totp_code=totp.now() if test_user.mfa_enabled else None,
            )

            print(f"認証結果: {auth_success} - {auth_message}")

            if auth_success and auth_user:
                print("\n5. セッション作成・権限テスト")
                session = manager.create_session(
                    auth_user, "127.0.0.1", "TestClient/1.0"
                )

                print(f"セッション作成: {session.session_id[:16]}...")
                print(f"権限数: {len(session.permissions)}")
                print(f"リスクスコア: {session.risk_score:.2f}")

                # 権限チェックテスト
                can_trade = manager.check_permission(
                    session, Permission.PLACE_ORDERS
                )
                can_manage = manager.check_permission(
                    session, Permission.MANAGE_USERS
                )

                print(f"取引権限: {can_trade}")
                print(f"管理権限: {can_manage}")

        print("\n6. セキュリティレポート生成")
        report = manager.get_security_report()

        print(f"レポートID: {report['report_id']}")
        print("ユーザー統計:")
        user_stats = report["user_statistics"]
        print(f"  総ユーザー数: {user_stats['total_users']}")
        print(f"  アクティブ: {user_stats['active_users']}")
        print(f"  MFA有効: {user_stats['mfa_enabled_users']}")

        print("セッション統計:")
        session_stats = report["session_statistics"]
        print(f"  アクティブセッション: {session_stats['active_sessions']}")
        print(f"  高リスクセッション: {session_stats['high_risk_sessions']}")

        print("推奨事項:")
        for rec in report["recommendations"]:
            print(f"  {rec}")

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== アクセス制御・認証システムテスト完了 ===")


if __name__ == "__main__":
    run_access_control_test()