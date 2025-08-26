#!/usr/bin/env python3
"""
アクセス制御システム - 認証処理

このモジュールは、ユーザー認証と権限チェック機能を提供します。
"""

import logging
from datetime import datetime
from typing import Optional, Tuple

from .enums import AuthenticationMethod, Permission, Session, User

try:
    from ...utils.logging_config import get_context_logger
except ImportError:
    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


class AuthenticationMixin:
    """
    認証機能のミックスイン
    
    AccessControlManagerに認証関連機能を追加します。
    """

    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        totp_code: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[User]]:
        """
        ユーザー認証
        
        Args:
            username: ユーザー名
            password: パスワード
            ip_address: IPアドレス
            user_agent: ユーザーエージェント
            totp_code: TOTPコード（MFA有効時）
            
        Returns:
            Tuple[bool, Optional[str], Optional[User]]: (認証結果, メッセージ, ユーザー)
        """
        # ユーザー検索
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break

        if not user:
            self._log_access(
                "UNKNOWN_USER",
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "user_not_found", "username": username},
            )
            return False, "認証に失敗しました", None

        # アカウント状態チェック
        if not user.is_active:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "account_inactive"},
            )
            return False, "アカウントが無効です", None

        if user.is_locked:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "account_locked"},
            )
            return False, "アカウントがロックされています", None

        # IP制限チェック
        if user.allowed_ip_addresses and ip_address not in user.allowed_ip_addresses:
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {"reason": "ip_restricted"},
            )
            return False, "許可されていないIPアドレスです", None

        # パスワード検証
        if not self._verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1

            # アカウントロック判定
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.is_locked = True
                logger.warning(
                    f"アカウントロック: {username} (失敗回数: {user.failed_login_attempts})"
                )

            self._save_users()
            self._log_access(
                user.user_id,
                None,
                "authentication",
                "login",
                ip_address,
                user_agent,
                False,
                {
                    "reason": "invalid_password",
                    "failed_attempts": user.failed_login_attempts,
                },
            )
            return False, "認証に失敗しました", None

        # MFA検証（有効な場合）
        if user.mfa_enabled and AuthenticationMethod.TOTP in user.mfa_methods:
            if not totp_code:
                return False, "TOTPコードが必要です", None

            if not self.mfa_manager.verify_totp(user.totp_secret, totp_code):
                self._log_access(
                    user.user_id,
                    None,
                    "authentication",
                    "mfa_failure",
                    ip_address,
                    user_agent,
                    False,
                    {"reason": "invalid_totp"},
                )
                return False, "TOTPコードが正しくありません", None

        # 認証成功
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        self._save_users()

        self._log_access(
            user.user_id,
            None,
            "authentication",
            "login",
            ip_address,
            user_agent,
            True,
            {"mfa_used": user.mfa_enabled},
        )

        logger.info(f"ユーザー認証成功: {username}")
        return True, "認証成功", user

    def check_permission(
        self, session: Session, permission: Permission, resource: Optional[str] = None
    ) -> bool:
        """
        権限チェック
        
        Args:
            session: セッション
            permission: チェックする権限
            resource: リソース名（オプション）
            
        Returns:
            bool: 権限があるかどうか
        """
        has_permission = permission in session.permissions

        self._log_access(
            session.user_id,
            session.session_id,
            "authorization",
            "check_permission",
            session.ip_address,
            session.user_agent,
            has_permission,
            {"permission": permission.value, "resource": resource},
        )

        return has_permission

    def cleanup_expired_sessions(self) -> int:
        """
        期限切れセッションのクリーンアップ
        
        Returns:
            int: 削除されたセッション数
        """
        from .enums import SessionStatus
        
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if not session.is_valid():
                expired_sessions.append(session_id)
                session.status = SessionStatus.EXPIRED

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"期限切れセッションをクリーンアップしました: {len(expired_sessions)}件")

        return len(expired_sessions)