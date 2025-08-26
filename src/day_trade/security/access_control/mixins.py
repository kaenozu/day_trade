#!/usr/bin/env python3
"""
アクセス制御システム - ミックスイン統合

このモジュールは、各種ミックスインをAccessControlManagerに統合します。
"""


def integrate_mixins():
    """
    各種ミックスインをAccessControlManagerクラスに統合
    """
    from .access_control_manager import AccessControlManager
    from .authentication import AuthenticationMixin
    from .reporting import ReportingMixin
    
    # AuthenticationMixinのメソッドを追加
    for attr_name in dir(AuthenticationMixin):
        if not attr_name.startswith('__'):
            attr = getattr(AuthenticationMixin, attr_name)
            if callable(attr):
                setattr(AccessControlManager, attr_name, attr)
    
    # ReportingMixinのメソッドを追加
    for attr_name in dir(ReportingMixin):
        if not attr_name.startswith('__'):
            attr = getattr(ReportingMixin, attr_name)
            if callable(attr):
                setattr(AccessControlManager, attr_name, attr)


# モジュール読み込み時に自動統合
integrate_mixins()