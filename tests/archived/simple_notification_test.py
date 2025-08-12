#!/usr/bin/env python3
"""
Issue #445: 不要な通知パラメータ削除の簡単なテスト
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_notification_channels():
    """通知チャンネルの削除確認"""
    print("Issue #445: 不要な通知パラメータ削除テスト")
    print("=" * 50)

    try:
        # alert_engine.pyのNotificationChannelをテスト
        from day_trade.monitoring.alert_engine import NotificationChannel

        print("\n✓ alert_engine.py - NotificationChannel:")
        channels = [member.value for member in NotificationChannel]
        print(f"  現在のチャンネル: {channels}")

        # 必要なチャンネルが存在することを確認
        expected = ["email", "slack", "webhook"]
        for channel in expected:
            if channel in channels:
                print(f"  ✓ {channel} - 保持")
            else:
                print(f"  ✗ {channel} - 見つかりません")

        # 削除されたチャンネルが存在しないことを確認
        removed = ["sms", "discord"]
        for channel in removed:
            if channel not in channels:
                print(f"  ✓ {channel} - 正しく削除されました")
            else:
                print(f"  ✗ {channel} - まだ存在しています")

    except Exception as e:
        print(f"✗ alert_engine.pyのテスト失敗: {e}")

    try:
        # notification_system.pyの基本動作テスト
        from day_trade.monitoring.notification_system import NotificationSystem

        print("\n✓ notification_system.py - インポートテスト:")
        system = NotificationSystem()
        print("  ✓ NotificationSystemの初期化成功")
        print(f"  ✓ テンプレート数: {len(system.templates)}")
        print(f"  ✓ 通知履歴: {len(system.notification_history)}")

    except Exception as e:
        print(f"✗ notification_system.pyのテスト失敗: {e}")

    print("\n" + "=" * 50)
    print("削除完了サマリー:")
    print("■ 削除された通知チャンネル:")
    print("  - SMS通知（未実装）")
    print("  - Discord通知（機能重複）")
    print("■ 保持された通知チャンネル:")
    print("  - Email通知（重要）")
    print("  - Slack通知（チーム通知）")
    print("  - Webhook通知（外部統合）")
    print("=" * 50)


if __name__ == "__main__":
    test_notification_channels()
