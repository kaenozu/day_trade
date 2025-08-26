#!/usr/bin/env python3
"""
Webダッシュボードモジュール

Issue #324: プロダクション運用監視ダッシュボード構築
Flask+WebSocket リアルタイムダッシュボードUI

後方互換性のためのメインインターフェース
"""

from .core import WebDashboard

__all__ = ['WebDashboard']

# メイン実行用の関数も再エクスポート
from .core import main

__version__ = '1.0.0'