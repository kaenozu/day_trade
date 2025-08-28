#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Web UI Package
スイングトレード用WebUIパッケージ

This package provides a web-based user interface for swing trading management,
including portfolio monitoring, purchase/sell record management, and alert system.

主要機能:
- ポートフォリオ監視ダッシュボード
- 購入・売却記録管理
- アラートシステム
- リアルタイム価格更新

Usage:
    from web.swing_trading import SwingTradingWebUI
    
    ui = SwingTradingWebUI(host="127.0.0.1", port=5001)
    ui.run(debug=False)
"""

from .app import SwingTradingWebUI, create_app

__version__ = "1.0.0"
__author__ = "Day Trade System"

# パッケージレベルのエクスポート
__all__ = [
    'SwingTradingWebUI',
    'create_app',
]

# グローバルインスタンス（後方互換性のため）
swing_trading_web_ui = SwingTradingWebUI()