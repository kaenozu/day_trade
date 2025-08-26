#!/usr/bin/env python3
"""
アクセス制御システム - ロール権限管理

このモジュールは、ユーザーロールと権限の管理を行います。
RBAC（Role-Based Access Control）に基づいた権限制御を提供します。
"""

from typing import Dict, Set

from .enums import Permission, UserRole


class RolePermissionManager:
    """
    ロール権限管理システム
    
    ユーザーロールに基づいた権限管理を行い、
    セキュアなアクセス制御を提供します。
    """

    def __init__(self):
        """
        ロール権限マッピングの初期化
        """
        self.role_permissions = self._initialize_role_permissions()

    def _initialize_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """
        ロール別権限の初期化
        
        各ユーザーロールに対して適切な権限セットを定義します。
        セキュリティの原則に従い、最小権限の原則を適用しています。
        
        Returns:
            Dict[UserRole, Set[Permission]]: ロールと権限のマッピング
        """
        return {
            UserRole.GUEST: set(),
            
            UserRole.VIEWER: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_REPORTS,
            },
            
            UserRole.TRADER: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.PLACE_ORDERS,
                Permission.MODIFY_ORDERS,
                Permission.CANCEL_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.RUN_ANALYSIS,
            },
            
            UserRole.ANALYST: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.RUN_ANALYSIS,
                Permission.CREATE_STRATEGIES,
                Permission.MODIFY_STRATEGIES,
                Permission.BACKTEST_STRATEGIES,
                Permission.EXPORT_DATA,
            },
            
            UserRole.ADMIN: {
                Permission.VIEW_MARKET_DATA,
                Permission.VIEW_HISTORICAL_DATA,
                Permission.VIEW_REPORTS,
                Permission.PLACE_ORDERS,
                Permission.MODIFY_ORDERS,
                Permission.CANCEL_ORDERS,
                Permission.VIEW_POSITIONS,
                Permission.RUN_ANALYSIS,
                Permission.CREATE_STRATEGIES,
                Permission.MODIFY_STRATEGIES,
                Permission.BACKTEST_STRATEGIES,
                Permission.MANAGE_USERS,
                Permission.MANAGE_SETTINGS,
                Permission.VIEW_LOGS,
                Permission.EXPORT_DATA,
            },
            
            UserRole.SUPER_ADMIN: set(Permission),  # 全権限
        }

    def get_permissions(self, role: UserRole) -> Set[Permission]:
        """
        ロールの権限取得
        
        Args:
            role: ユーザーロール
            
        Returns:
            Set[Permission]: 権限のセット（コピー）
        """
        return self.role_permissions.get(role, set()).copy()

    def has_permission(self, role: UserRole, permission: Permission) -> bool:
        """
        権限チェック
        
        Args:
            role: ユーザーロール
            permission: チェックする権限
            
        Returns:
            bool: 権限を持っているかどうか
        """
        return permission in self.role_permissions.get(role, set())

    def get_role_hierarchy(self) -> Dict[UserRole, int]:
        """
        ロール階層の取得
        
        数値が大きいほど上位のロールを表します。
        
        Returns:
            Dict[UserRole, int]: ロールと階層レベルのマッピング
        """
        return {
            UserRole.GUEST: 0,
            UserRole.VIEWER: 1,
            UserRole.TRADER: 2,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3,
            UserRole.SUPER_ADMIN: 4,
        }

    def is_higher_role(self, role1: UserRole, role2: UserRole) -> bool:
        """
        ロール階層の比較
        
        Args:
            role1: 比較するロール1
            role2: 比較するロール2
            
        Returns:
            bool: role1がrole2より上位のロールかどうか
        """
        hierarchy = self.get_role_hierarchy()
        return hierarchy.get(role1, 0) > hierarchy.get(role2, 0)

    def get_permission_description(self, permission: Permission) -> str:
        """
        権限の説明取得
        
        Args:
            permission: 権限
            
        Returns:
            str: 権限の説明
        """
        descriptions = {
            Permission.VIEW_MARKET_DATA: "市場データの閲覧",
            Permission.VIEW_HISTORICAL_DATA: "履歴データの閲覧",
            Permission.VIEW_REPORTS: "レポートの閲覧",
            Permission.PLACE_ORDERS: "注文の発行",
            Permission.MODIFY_ORDERS: "注文の変更",
            Permission.CANCEL_ORDERS: "注文の取消",
            Permission.VIEW_POSITIONS: "ポジションの閲覧",
            Permission.RUN_ANALYSIS: "分析の実行",
            Permission.CREATE_STRATEGIES: "戦略の作成",
            Permission.MODIFY_STRATEGIES: "戦略の変更",
            Permission.BACKTEST_STRATEGIES: "戦略のバックテスト",
            Permission.MANAGE_USERS: "ユーザー管理",
            Permission.MANAGE_SETTINGS: "設定管理",
            Permission.VIEW_LOGS: "ログの閲覧",
            Permission.MANAGE_SECURITY: "セキュリティ管理",
            Permission.EXECUTE_BULK_OPERATIONS: "一括操作の実行",
            Permission.MODIFY_RISK_LIMITS: "リスク制限の変更",
            Permission.ACCESS_API_KEYS: "APIキーへのアクセス",
            Permission.EXPORT_DATA: "データのエクスポート",
        }
        
        return descriptions.get(permission, permission.value)