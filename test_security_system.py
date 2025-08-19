#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
セキュリティシステムのテスト
Issue #918 項目9対応: セキュリティ強化

セキュリティ機能の包括的動作確認テスト
"""

import sys
import os
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# パスの設定
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_security_services_registration():
    """セキュリティサービス登録テスト"""
    print("=== セキュリティサービス登録テスト ===")

    try:
        from src.day_trade.core.dependency_injection import get_container
        from src.day_trade.core.services import register_default_services
        from src.day_trade.core.security_services import (
            register_security_services, get_security_services,
            IInputValidationService, IAuthenticationService, IAuthorizationService,
            IRateLimitService, ISecurityAuditService
        )

        # サービス登録
        register_default_services()
        register_security_services()
        print("OK: セキュリティサービス登録完了")

        # サービス取得テスト
        security_services = get_security_services()

        # 各サービスの存在確認
        assert 'validation' in security_services, "入力検証サービスが見つからない"
        assert 'authentication' in security_services, "認証サービスが見つからない"
        assert 'authorization' in security_services, "認可サービスが見つからない"
        assert 'rate_limit' in security_services, "レート制限サービスが見つからない"
        assert 'audit' in security_services, "監査サービスが見つからない"

        print("OK: 全セキュリティサービス取得成功")

        # 個別サービス解決テスト
        container = get_container()
        validation_service = container.resolve(IInputValidationService)
        auth_service = container.resolve(IAuthenticationService)
        authz_service = container.resolve(IAuthorizationService)
        rate_limit_service = container.resolve(IRateLimitService)
        audit_service = container.resolve(ISecurityAuditService)

        assert validation_service is not None, "入力検証サービスの解決に失敗"
        assert auth_service is not None, "認証サービスの解決に失敗"
        assert authz_service is not None, "認可サービスの解決に失敗"
        assert rate_limit_service is not None, "レート制限サービスの解決に失敗"
        assert audit_service is not None, "監査サービスの解決に失敗"

        print("OK: 個別サービス解決成功")

        return True

    except Exception as e:
        print(f"FAIL: セキュリティサービス登録テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_validation():
    """入力検証テスト"""
    print("\n=== 入力検証テスト ===")

    try:
        from src.day_trade.core.security_services import get_security_services, ThreatLevel

        security_services = get_security_services()
        validation_service = security_services['validation']

        # 1. 文字列検証テスト
        # 正常なケース
        result = validation_service.validate_string("正常な文字列", max_length=100)
        assert result.is_valid, "正常な文字列が無効と判定された"
        assert result.threat_level == ThreatLevel.INFO, "正常な文字列の脅威レベルが不正"
        print("OK: 正常な文字列検証")

        # XSS攻撃パターン
        result = validation_service.validate_string("<script>alert('xss')</script>")
        assert not result.is_valid, "XSS攻撃パターンが有効と判定された"
        assert result.threat_level == ThreatLevel.HIGH, "XSS攻撃の脅威レベルが不正"
        print("OK: XSS攻撃パターン検出")

        # 長すぎる文字列
        long_string = "a" * 2000
        result = validation_service.validate_string(long_string, max_length=1000)
        assert not result.is_valid, "長すぎる文字列が有効と判定された"
        print("OK: 長さ制限検証")

        # 2. 数値検証テスト
        # 正常な数値
        result = validation_service.validate_number("123.45", min_value=0, max_value=1000)
        assert result.is_valid, "正常な数値が無効と判定された"
        assert result.sanitized_value == 123.45, "数値の変換が正しくない"
        print("OK: 正常な数値検証")

        # 範囲外の数値
        result = validation_service.validate_number("2000", min_value=0, max_value=1000)
        assert not result.is_valid, "範囲外の数値が有効と判定された"
        print("OK: 範囲外数値検証")

        # 不正な数値文字列
        result = validation_service.validate_number("abc123")
        assert not result.is_valid, "不正な数値文字列が有効と判定された"
        print("OK: 不正数値文字列検証")

        # 3. メールアドレス検証テスト
        # 正常なメールアドレス
        result = validation_service.validate_email("test@example.com")
        assert result.is_valid, "正常なメールアドレスが無効と判定された"
        assert result.sanitized_value == "test@example.com", "メールアドレスのサニタイズが不正"
        print("OK: 正常なメールアドレス検証")

        # 不正なメールアドレス
        result = validation_service.validate_email("invalid-email")
        assert not result.is_valid, "不正なメールアドレスが有効と判定された"
        print("OK: 不正メールアドレス検証")

        # 4. IPアドレス検証テスト
        # 正常なIPv4アドレス
        result = validation_service.validate_ip_address("192.168.1.1")
        assert result.is_valid, "正常なIPv4アドレスが無効と判定された"
        print("OK: IPv4アドレス検証")

        # 正常なIPv6アドレス
        result = validation_service.validate_ip_address("::1")
        assert result.is_valid, "正常なIPv6アドレスが無効と判定された"
        print("OK: IPv6アドレス検証")

        # 不正なIPアドレス
        result = validation_service.validate_ip_address("999.999.999.999")
        assert not result.is_valid, "不正なIPアドレスが有効と判定された"
        print("OK: 不正IPアドレス検証")

        # 5. SQL入力サニタイゼーションテスト
        # SQLインジェクション攻撃
        result = validation_service.sanitize_sql_input("'; DROP TABLE users; --")
        assert result.threat_level == ThreatLevel.HIGH, "SQLインジェクション脅威レベルが不正"
        print("OK: SQLインジェクション検出")

        # 正常なSQL入力
        result = validation_service.sanitize_sql_input("正常なテキスト")
        assert result.is_valid, "正常なSQL入力が無効と判定された"
        print("OK: 正常SQL入力検証")

        # 6. ファイルパス検証テスト
        # 正常なファイルパス
        result = validation_service.validate_file_path("/home/user/document.txt", {'.txt', '.pdf'})
        assert result.is_valid, "正常なファイルパスが無効と判定された"
        print("OK: 正常ファイルパス検証")

        # パストラバーサル攻撃
        result = validation_service.validate_file_path("../../../etc/passwd")
        assert not result.is_valid, "パストラバーサル攻撃が有効と判定された"
        assert result.threat_level == ThreatLevel.HIGH, "パストラバーサル脅威レベルが不正"
        print("OK: パストラバーサル攻撃検出")

        return True

    except Exception as e:
        print(f"FAIL: 入力検証テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_authentication_service():
    """認証サービステスト"""
    print("\n=== 認証サービステスト ===")

    try:
        from src.day_trade.core.security_services import get_security_services

        security_services = get_security_services()
        auth_service = security_services['authentication']

        # テストユーザー登録
        test_username = "testuser"
        test_password = "TestPass123!"
        permissions = {"data.read", "trade.view"}

        registration_success = auth_service.register_user(test_username, test_password, permissions)
        assert registration_success, "テストユーザーの登録に失敗"
        print("OK: テストユーザー登録成功")

        # 1. パスワードハッシュ化テスト
        password_hash = auth_service.hash_password(test_password)
        assert password_hash != test_password, "パスワードがハッシュ化されていない"
        assert ":" in password_hash, "ハッシュ形式が正しくない"
        print("OK: パスワードハッシュ化")

        # 2. パスワード検証テスト
        # 正しいパスワード
        is_valid = auth_service.verify_password(test_password, password_hash)
        assert is_valid, "正しいパスワードが認証に失敗"
        print("OK: 正しいパスワード検証")

        # 間違ったパスワード
        is_valid = auth_service.verify_password("WrongPassword", password_hash)
        assert not is_valid, "間違ったパスワードが認証に成功"
        print("OK: 間違ったパスワード検証")

        # 3. ユーザー認証テスト
        # 正常なログイン
        auth_result = auth_service.authenticate_user(test_username, test_password)
        assert auth_result.is_authenticated, "正常なログインが失敗"
        assert auth_result.user_id == test_username, "ユーザーIDが正しくない"
        assert auth_result.session_token is not None, "セッショントークンが生成されていない"
        print("OK: 正常なユーザー認証")

        session_token = auth_result.session_token

        # 間違ったパスワードでのログイン
        auth_result = auth_service.authenticate_user(test_username, "WrongPassword")
        assert not auth_result.is_authenticated, "間違ったパスワードでログインが成功"
        print("OK: 間違ったパスワードでのログイン拒否")

        # 存在しないユーザーでのログイン
        auth_result = auth_service.authenticate_user("nonexistent", test_password)
        assert not auth_result.is_authenticated, "存在しないユーザーでログインが成功"
        print("OK: 存在しないユーザーでのログイン拒否")

        # 4. セッション検証テスト
        # 有効なセッション
        session_result = auth_service.validate_session(session_token)
        assert session_result.is_authenticated, "有効なセッションが無効と判定"
        assert session_result.user_id == test_username, "セッションのユーザーIDが不正"
        print("OK: 有効なセッション検証")

        # 無効なセッション
        session_result = auth_service.validate_session("invalid_token")
        assert not session_result.is_authenticated, "無効なセッションが有効と判定"
        print("OK: 無効なセッション検証")

        # 5. セッション無効化テスト
        revoke_success = auth_service.revoke_session(session_token)
        assert revoke_success, "セッション無効化に失敗"
        print("OK: セッション無効化")

        # 無効化されたセッションの検証
        session_result = auth_service.validate_session(session_token)
        assert not session_result.is_authenticated, "無効化されたセッションが有効と判定"
        print("OK: 無効化されたセッション検証")

        return True

    except Exception as e:
        print(f"FAIL: 認証サービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_authorization_service():
    """認可サービステスト"""
    print("\n=== 認可サービステスト ===")

    try:
        from src.day_trade.core.security_services import get_security_services, ActionType

        security_services = get_security_services()
        authz_service = security_services['authorization']

        test_user = "testuser_authz"

        # 1. ロール割り当てテスト
        # 管理者ロール
        assign_success = authz_service.assign_role(test_user, "admin")
        assert assign_success, "管理者ロールの割り当てに失敗"
        print("OK: 管理者ロール割り当て")

        # 管理者権限チェック
        has_permission = authz_service.check_permission(test_user, ActionType.SYSTEM_ADMIN)
        assert has_permission, "管理者の権限チェックに失敗"
        print("OK: 管理者権限チェック")

        # 取引権限チェック
        has_permission = authz_service.check_permission(test_user, ActionType.TRADE_EXECUTION)
        assert has_permission, "管理者の取引権限チェックに失敗"
        print("OK: 管理者取引権限チェック")

        # 2. トレーダーロールテスト
        trader_user = "trader_user"
        assign_success = authz_service.assign_role(trader_user, "trader")
        assert assign_success, "トレーダーロールの割り当てに失敗"
        print("OK: トレーダーロール割り当て")

        # トレーダー取引権限チェック
        has_permission = authz_service.check_permission(trader_user, ActionType.TRADE_EXECUTION)
        assert has_permission, "トレーダーの取引権限チェックに失敗"
        print("OK: トレーダー取引権限チェック")

        # トレーダーシステム管理権限チェック（拒否されるべき）
        has_permission = authz_service.check_permission(trader_user, ActionType.SYSTEM_ADMIN)
        assert not has_permission, "トレーダーがシステム管理権限を持っている"
        print("OK: トレーダーシステム管理権限拒否")

        # 3. ビューアーロールテスト
        viewer_user = "viewer_user"
        assign_success = authz_service.assign_role(viewer_user, "viewer")
        assert assign_success, "ビューアーロールの割り当てに失敗"
        print("OK: ビューアーロール割り当て")

        # ビューアーデータ読み取り権限チェック
        has_permission = authz_service.check_permission(viewer_user, ActionType.DATA_ACCESS)
        assert has_permission, "ビューアーのデータ読み取り権限チェックに失敗"
        print("OK: ビューアーデータ読み取り権限チェック")

        # ビューアー取引権限チェック（拒否されるべき）
        has_permission = authz_service.check_permission(viewer_user, ActionType.TRADE_EXECUTION)
        assert not has_permission, "ビューアーが取引権限を持っている"
        print("OK: ビューアー取引権限拒否")

        # 4. 個別権限操作テスト
        custom_user = "custom_user"

        # 権限付与
        grant_success = authz_service.grant_permission(custom_user, "custom.permission")
        assert grant_success, "権限付与に失敗"
        print("OK: 個別権限付与")

        # 権限確認
        user_permissions = authz_service.get_user_permissions(custom_user)
        assert "custom.permission" in user_permissions, "付与した権限が見つからない"
        print("OK: 個別権限確認")

        # 権限剥奪
        revoke_success = authz_service.revoke_permission(custom_user, "custom.permission")
        assert revoke_success, "権限剥奪に失敗"
        print("OK: 個別権限剥奪")

        # 剥奪確認
        user_permissions = authz_service.get_user_permissions(custom_user)
        assert "custom.permission" not in user_permissions, "剥奪した権限がまだ存在"
        print("OK: 権限剥奪確認")

        return True

    except Exception as e:
        print(f"FAIL: 認可サービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rate_limit_service():
    """レート制限サービステスト"""
    print("\n=== レート制限サービステスト ===")

    try:
        from src.day_trade.core.security_services import get_security_services

        security_services = get_security_services()
        rate_limit_service = security_services['rate_limit']

        test_key = "test_api_user"
        limit = 5
        window_seconds = 10

        # 1. 初期レート制限チェック
        rate_info = rate_limit_service.check_rate_limit(test_key, limit, window_seconds)
        assert not rate_info.is_exceeded, "初期状態でレート制限が超過している"
        assert rate_info.current_count == 0, "初期カウントが0でない"
        assert rate_info.limit == limit, "制限値が正しくない"
        print("OK: 初期レート制限チェック")

        # 2. カウンター増分テスト
        for i in range(3):
            count = rate_limit_service.increment_counter(test_key)
            assert count == i + 1, f"カウンター増分が正しくない: expected {i+1}, got {count}"
        print("OK: カウンター増分")

        # 3. レート制限内でのチェック
        rate_info = rate_limit_service.check_rate_limit(test_key, limit, window_seconds)
        assert not rate_info.is_exceeded, "制限内でレート制限が超過している"
        assert rate_info.current_count == 3, "現在のカウントが正しくない"
        print("OK: レート制限内チェック")

        # 4. レート制限超過テスト
        # 制限を超えるまでカウンターを増分
        while rate_limit_service.increment_counter(test_key) < limit:
            pass

        # 制限超過後のチェック
        rate_info = rate_limit_service.check_rate_limit(test_key, limit, window_seconds)
        assert rate_info.is_exceeded, "レート制限超過が検出されない"
        assert rate_info.current_count >= limit, "カウントが制限値に達していない"
        print("OK: レート制限超過検出")

        # 5. カウンターリセットテスト
        reset_success = rate_limit_service.reset_counter(test_key)
        assert reset_success, "カウンターリセットに失敗"
        print("OK: カウンターリセット")

        # リセット後のチェック
        rate_info = rate_limit_service.check_rate_limit(test_key, limit, window_seconds)
        assert not rate_info.is_exceeded, "リセット後にレート制限が超過している"
        assert rate_info.current_count == 0, "リセット後のカウントが0でない"
        print("OK: リセット後チェック")

        # 6. 複数キーでのテスト
        key2 = "test_api_user2"

        # key2のカウンターを初期化してから増分
        rate_limit_service.check_rate_limit(key2, limit, window_seconds)  # 初期化
        rate_limit_service.increment_counter(key2)
        rate_limit_service.increment_counter(key2)

        # key1とkey2が独立していることを確認
        rate_info1 = rate_limit_service.check_rate_limit(test_key, limit, window_seconds)
        rate_info2 = rate_limit_service.check_rate_limit(key2, limit, window_seconds)

        assert rate_info1.current_count == 0, f"key1のカウントが影響を受けている: {rate_info1.current_count}"
        assert rate_info2.current_count == 2, f"key2のカウントが正しくない: expected 2, got {rate_info2.current_count}"
        print("OK: 複数キー独立性確認")

        return True

    except Exception as e:
        print(f"FAIL: レート制限サービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_audit_service():
    """セキュリティ監査サービステスト"""
    print("\n=== セキュリティ監査サービステスト ===")

    try:
        from src.day_trade.core.security_services import (
            get_security_services, create_security_event,
            ThreatLevel, ActionType
        )

        security_services = get_security_services()
        audit_service = security_services['audit']

        # 1. セキュリティイベントログテスト
        # 複数のイベントを作成
        events = [
            create_security_event(
                "login_attempt",
                ThreatLevel.INFO,
                source_ip="192.168.1.100",
                user_id="user1",
                action=ActionType.LOGIN
            ),
            create_security_event(
                "failed_login",
                ThreatLevel.MEDIUM,
                source_ip="192.168.1.101",
                user_id="user2",
                action=ActionType.LOGIN,
                blocked=True
            ),
            create_security_event(
                "sql_injection_attempt",
                ThreatLevel.HIGH,
                source_ip="10.0.0.1",
                details={"attempted_query": "'; DROP TABLE users; --"}
            ),
            create_security_event(
                "unauthorized_access",
                ThreatLevel.CRITICAL,
                source_ip="203.0.113.1",
                user_id="admin_account",
                action=ActionType.SYSTEM_ADMIN,
                resource="sensitive_config",
                blocked=True
            )
        ]

        # イベントログ
        event_ids = []
        for event in events:
            event_id = audit_service.log_security_event(event)
            assert event_id != "", "イベントIDが生成されていない"
            event_ids.append(event_id)

        print(f"OK: {len(events)}個のセキュリティイベントログ成功")

        # 2. イベント取得テスト
        # 全イベント取得
        all_events = audit_service.get_security_events()
        assert len(all_events) >= len(events), "ログしたイベント数が足りない"
        print("OK: 全イベント取得")

        # 脅威レベルフィルター
        high_threat_events = audit_service.get_security_events(threat_level=ThreatLevel.HIGH)
        assert len(high_threat_events) >= 1, "HIGH脅威レベルのイベントが見つからない"
        print("OK: 脅威レベルフィルター")

        # 時間範囲フィルター
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        recent_events = audit_service.get_security_events(start_time=one_hour_ago)
        assert len(recent_events) >= len(events), "時間範囲フィルターが正しく動作していない"
        print("OK: 時間範囲フィルター")

        # 3. 脅威分析テスト
        threat_analysis = audit_service.analyze_threats(time_window_hours=1)

        assert 'total_events' in threat_analysis, "脅威分析に総イベント数が含まれていない"
        assert 'threat_level_distribution' in threat_analysis, "脅威レベル分布が含まれていない"
        assert 'risk_score' in threat_analysis, "リスクスコアが含まれていない"
        assert 'risk_level' in threat_analysis, "リスクレベルが含まれていない"

        assert threat_analysis['total_events'] >= len(events), "分析対象イベント数が不正"
        print("OK: 脅威分析")

        # リスクスコアの妥当性チェック
        risk_score = threat_analysis['risk_score']
        assert 0 <= risk_score <= 100, f"リスクスコアが範囲外: {risk_score}"
        print(f"OK: リスクスコア算出 ({risk_score:.1f})")

        # 4. セキュリティレポート生成テスト
        security_report = audit_service.generate_security_report()

        assert 'report_generated_at' in security_report, "レポート生成時刻が含まれていない"
        assert 'analysis_24_hours' in security_report, "24時間分析が含まれていない"
        assert 'analysis_7_days' in security_report, "7日分析が含まれていない"
        assert 'analysis_30_days' in security_report, "30日分析が含まれていない"
        assert 'recommendations' in security_report, "推奨事項が含まれていない"

        print("OK: セキュリティレポート生成")

        # 推奨事項の確認
        recommendations = security_report['recommendations']
        assert isinstance(recommendations, list), "推奨事項がリスト形式でない"
        assert len(recommendations) > 0, "推奨事項が生成されていない"
        print(f"OK: {len(recommendations)}件の推奨事項生成")

        # 5. 脅威レベル分布の確認
        threat_dist = threat_analysis['threat_level_distribution']
        expected_levels = ['info', 'medium', 'high', 'critical']

        for level in expected_levels:
            if level in threat_dist:
                print(f"  {level.upper()}: {threat_dist[level]}件")

        return True

    except Exception as e:
        print(f"FAIL: セキュリティ監査サービステスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_security_scenario():
    """統合セキュリティシナリオテスト"""
    print("\n=== 統合セキュリティシナリオテスト ===")

    try:
        from src.day_trade.core.security_services import (
            get_security_services, create_security_event,
            ThreatLevel, ActionType
        )

        security_services = get_security_services()
        validation_service = security_services['validation']
        auth_service = security_services['authentication']
        authz_service = security_services['authorization']
        rate_limit_service = security_services['rate_limit']
        audit_service = security_services['audit']

        # シナリオ: 悪意のあるユーザーによる攻撃シミュレーション
        attacker_ip = "203.0.113.100"

        # 1. 不正な入力による攻撃
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "999.999.999.999"
        ]

        blocked_attempts = 0
        for malicious_input in malicious_inputs:
            # 各種検証メソッドで確認
            string_result = validation_service.validate_string(malicious_input)
            sql_result = validation_service.sanitize_sql_input(malicious_input)
            path_result = validation_service.validate_file_path(malicious_input)
            ip_result = validation_service.validate_ip_address(malicious_input)

            # いずれかで高脅威が検出された場合
            high_threat_detected = False
            threat_level = ThreatLevel.INFO

            for result in [string_result, sql_result, path_result, ip_result]:
                if result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    high_threat_detected = True
                    threat_level = result.threat_level
                    break
                elif not result.is_valid and result.threat_level == ThreatLevel.MEDIUM:
                    high_threat_detected = True
                    threat_level = result.threat_level

            if high_threat_detected:
                blocked_attempts += 1

                # セキュリティイベントログ
                event = create_security_event(
                    "malicious_input_detected",
                    threat_level,
                    source_ip=attacker_ip,
                    details={"input": malicious_input, "threat_type": "input_validation"},
                    blocked=True
                )
                audit_service.log_security_event(event)

        assert blocked_attempts >= 2, "悪意のある入力の検出数が不足"
        print(f"OK: {blocked_attempts}件の悪意ある入力をブロック")

        # 2. ブルートフォース攻撃シミュレーション
        target_user = "admin"
        wrong_passwords = ["password", "123456", "admin", "qwerty", "letmein"]

        failed_logins = 0
        for password in wrong_passwords:
            auth_result = auth_service.authenticate_user(target_user, password)
            if not auth_result.is_authenticated:
                failed_logins += 1

                # レート制限チェック
                rate_info = rate_limit_service.check_rate_limit(
                    f"login_{attacker_ip}", 3, 300  # 5分間で3回まで
                )
                rate_limit_service.increment_counter(f"login_{attacker_ip}")

                # セキュリティイベントログ
                event = create_security_event(
                    "failed_login_attempt",
                    ThreatLevel.MEDIUM if not rate_info.is_exceeded else ThreatLevel.HIGH,
                    source_ip=attacker_ip,
                    user_id=target_user,
                    action=ActionType.LOGIN,
                    details={"attempt_count": rate_info.current_count + 1},
                    blocked=rate_info.is_exceeded
                )
                audit_service.log_security_event(event)

        print(f"OK: {failed_logins}件のログイン失敗を記録")

        # 3. 権限昇格攻撃シミュレーション
        # 通常ユーザーでログイン
        normal_user = "normaluser"
        auth_service.register_user(normal_user, "UserPass123!", {"data.read"})
        authz_service.assign_role(normal_user, "viewer")

        # 管理者権限を要求
        has_admin_permission = authz_service.check_permission(normal_user, ActionType.SYSTEM_ADMIN)
        assert not has_admin_permission, "通常ユーザーが管理者権限を持っている"

        # 権限昇格試行をログ
        event = create_security_event(
            "privilege_escalation_attempt",
            ThreatLevel.HIGH,
            source_ip=attacker_ip,
            user_id=normal_user,
            action=ActionType.SYSTEM_ADMIN,
            details={"requested_permission": "system.admin"},
            blocked=True
        )
        audit_service.log_security_event(event)
        print("OK: 権限昇格攻撃を阻止")

        # 4. 最終的な脅威分析
        final_analysis = audit_service.analyze_threats(time_window_hours=1)

        total_events = final_analysis['total_events']
        blocked_events = final_analysis['blocked_events']
        risk_score = final_analysis['risk_score']

        print(f"OK: 統合テスト完了")
        print(f"  総イベント数: {total_events}")
        print(f"  ブロックされたイベント: {blocked_events}")
        print(f"  リスクスコア: {risk_score:.1f}")
        print(f"  リスクレベル: {final_analysis['risk_level']}")

        # セキュリティシステムが適切に機能していることを確認
        assert total_events > 0, "セキュリティイベントが記録されていない"
        assert blocked_events > 0, "攻撃がブロックされていない"

        return True

    except Exception as e:
        print(f"FAIL: 統合セキュリティシナリオテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト実行"""
    print("セキュリティシステムの包括的テストを開始します...\n")

    results = []

    # 各テストを実行
    results.append(("セキュリティサービス登録", test_security_services_registration()))
    results.append(("入力検証", test_input_validation()))
    results.append(("認証サービス", test_authentication_service()))
    results.append(("認可サービス", test_authorization_service()))
    results.append(("レート制限サービス", test_rate_limit_service()))
    results.append(("セキュリティ監査サービス", test_security_audit_service()))
    results.append(("統合セキュリティシナリオ", test_integrated_security_scenario()))

    # 結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)

    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:<25}: {status}")
        if result:
            passed += 1

    print(f"\n合計: {passed}/{len(results)} テスト通過")

    if passed == len(results):
        print("SUCCESS: 全テストが正常に完了しました！")
        print("セキュリティ強化が成功しました。")
        return 0
    else:
        print("WARNING: 一部のテストが失敗しました。")
        return 1

if __name__ == "__main__":
    sys.exit(main())