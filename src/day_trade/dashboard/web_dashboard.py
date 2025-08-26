#!/usr/bin/env python3
"""
Webベースダッシュボード - 後方互換性レイヤー

Issue #324: プロダクション運用監視ダッシュボード構築
Flask+WebSocket リアルタイムダッシュボードUI

このファイルは後方互換性のために残されています。
実際の実装は web_dashboard/ パッケージに分割されています。
"""

# 後方互換性のためのインポート
from .web_dashboard import WebDashboard, main

# 元のすべての機能を再エクスポート
__all__ = ['WebDashboard', 'main']