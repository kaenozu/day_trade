#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Templates Package
スイングトレード用テンプレートパッケージ

HTMLテンプレート定数とスタイルを管理するパッケージです。

使用可能なテンプレート:
- DASHBOARD_TEMPLATE: メインダッシュボード
- PURCHASE_FORM_TEMPLATE: 購入記録フォーム  
- PARTIAL_SELL_FORM_TEMPLATE: 部分売却フォーム
- MONITORING_TEMPLATE: 監視管理画面
- ALERTS_TEMPLATE: アラート管理画面
"""

from .dashboard import DASHBOARD_TEMPLATE
from .purchase_form import PURCHASE_FORM_TEMPLATE
from .sell_form import PARTIAL_SELL_FORM_TEMPLATE
from .monitoring import MONITORING_TEMPLATE  
from .alerts import ALERTS_TEMPLATE

__all__ = [
    'DASHBOARD_TEMPLATE',
    'PURCHASE_FORM_TEMPLATE', 
    'PARTIAL_SELL_FORM_TEMPLATE',
    'MONITORING_TEMPLATE',
    'ALERTS_TEMPLATE',
]