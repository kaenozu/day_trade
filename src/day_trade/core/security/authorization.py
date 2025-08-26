#!/usr/bin/env python3
"""
認可サービス実装
Issue #918 項目9対応: セキュリティ強化

ユーザー権限管理、ロールベースアクセス制御機能
"""

import threading
from typing import Dict, Set

from ..dependency_injection import ILoggingService, injectable, singleton
from .interfaces import IAuthorizationService
from .types import ActionType


@singleton(IAuthorizationService)
@injectable
class AuthorizationService(IAuthorizationService):
    """認可サービス実装"""

    def __init__(self, logging_service: ILoggingService):
        self.logging_service = logging_service
        self.logger = logging_service.get_logger(__name__, "AuthorizationService")

        # ユーザー権限管理
        self._user_permissions: Dict[str, Set[str]] = {}
        self._permission_lock = threading.RLock()

        # デフォルト権限定義
        self._default_permissions = {
            'admin': {
                'system.admin', 'data.read', 'data.write', 'data.delete',
                'trade.execute', 'trade.view', 'config.read', 'config.write'
            },
            'trader': {
                'data.read', 'trade.execute', 'trade.view', 'config.read'
            },
            'viewer': {
                'data.read', 'trade.view'
            }
        }

    def check_permission(self, user_id: str, action: ActionType, resource: str = None) -> bool:
        """権限チェック"""
        try:
            with self._permission_lock:
                user_permissions = self._user_permissions.get(user_id, set())

                # アクション種別に応じた権限マッピング
                required_permissions = self._get_required_permissions(action, resource)

                # 権限チェック
                has_permission = any(perm in user_permissions for perm in required_permissions)

                if not has_permission:
                    self.logger.warning(f"Permission denied for user {user_id}: {action.value}")

                return has_permission

        except Exception as e:
            self.logger.error(f"Permission check error: {e}")
            return False

    def grant_permission(self, user_id: str, permission: str) -> bool:
        """権限付与"""
        try:
            with self._permission_lock:
                if user_id not in self._user_permissions:
                    self._user_permissions[user_id] = set()

                self._user_permissions[user_id].add(permission)
                self.logger.info(f"Permission granted to {user_id}: {permission}")
                return True

        except Exception as e:
            self.logger.error(f"Permission grant error: {e}")
            return False

    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """権限剥奪"""
        try:
            with self._permission_lock:
                if user_id in self._user_permissions:
                    self._user_permissions[user_id].discard(permission)
                    self.logger.info(f"Permission revoked from {user_id}: {permission}")
                    return True
                return False

        except Exception as e:
            self.logger.error(f"Permission revoke error: {e}")
            return False

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """ユーザー権限取得"""
        with self._permission_lock:
            return self._user_permissions.get(user_id, set()).copy()

    def assign_role(self, user_id: str, role: str) -> bool:
        """ロール割り当て"""
        try:
            if role not in self._default_permissions:
                self.logger.error(f"Unknown role: {role}")
                return False

            with self._permission_lock:
                self._user_permissions[user_id] = self._default_permissions[role].copy()
                self.logger.info(f"Role assigned to {user_id}: {role}")
                return True

        except Exception as e:
            self.logger.error(f"Role assignment error: {e}")
            return False

    def _get_required_permissions(self, action: ActionType, resource: str = None) -> Set[str]:
        """必要権限取得"""
        permission_map = {
            ActionType.LOGIN: {'system.login'},
            ActionType.LOGOUT: {'system.logout'},
            ActionType.API_ACCESS: {'api.access'},
            ActionType.DATA_ACCESS: {'data.read'},
            ActionType.CONFIG_CHANGE: {'config.write'},
            ActionType.TRADE_EXECUTION: {'trade.execute'},
            ActionType.FILE_ACCESS: {'file.access'},
            ActionType.SYSTEM_ADMIN: {'system.admin'}
        }

        base_permissions = permission_map.get(action, {'generic.access'})

        # リソース特化権限の追加
        if resource:
            resource_permission = f"{action.value}.{resource}"
            base_permissions.add(resource_permission)

        return base_permissions