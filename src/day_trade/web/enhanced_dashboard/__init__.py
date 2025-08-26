#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Dashboard Package - 拡張ダッシュボードパッケージ

モジュール化されたウェブダッシュボードシステムのエントリーポイント
後方互換性を保ちながら、分割されたモジュールからクラスをインポート
"""

# 主要クラスのインポート
from .dashboard import EnhancedWebDashboard
from .config import (
    DashboardConfig,
    DashboardTheme,
    ChartType,
    UpdateFrequency,
    AlertConfig,
    UserPreferences,
    load_dashboard_config
)
from .real_time_manager import RealTimeDataManager
from .visualization import AdvancedVisualization
from .alert_manager import AlertManager
from .routes import DashboardRoutes
from .socket_handlers import SocketHandlers
from .templates import DashboardTemplates

# 後方互換性のためのエイリアス
from .dashboard import EnhancedWebDashboard as EnhancedWebDashboard

# ファクトリー関数（元のファイルから移行）
def create_enhanced_web_dashboard(config_path=None, port=8080):
    """
    EnhancedWebDashboardインスタンスの作成
    
    Args:
        config_path: 設定ファイルパス
        port: ポート番号
    
    Returns:
        EnhancedWebDashboardインスタンス
    """
    from pathlib import Path
    config_path_obj = Path(config_path) if config_path else None
    return EnhancedWebDashboard(config_path=config_path_obj, port=port)

# パッケージメタデータ
__version__ = "1.0.0"
__author__ = "Enhanced Trading Dashboard Team"
__description__ = "Modular web dashboard for trading analysis and monitoring"

# パブリックAPI
__all__ = [
    # メインクラス
    'EnhancedWebDashboard',
    
    # 設定関連
    'DashboardConfig',
    'DashboardTheme',
    'ChartType',
    'UpdateFrequency',
    'AlertConfig',
    'UserPreferences',
    'load_dashboard_config',
    
    # コンポーネントクラス
    'RealTimeDataManager',
    'AdvancedVisualization',
    'AlertManager',
    'DashboardRoutes',
    'SocketHandlers',
    'DashboardTemplates',
    
    # ファクトリー関数
    'create_enhanced_web_dashboard',
]

# 利便性インポート
try:
    # Web依存関係の確認
    from flask import Flask
    from flask_socketio import SocketIO
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

def check_dependencies():
    """依存関係のチェック"""
    missing_deps = []
    
    if not WEB_AVAILABLE:
        missing_deps.extend(['flask', 'flask-socketio'])
    
    try:
        import plotly
    except ImportError:
        missing_deps.append('plotly')
    
    try:
        import pandas
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    return missing_deps

def get_package_info():
    """パッケージ情報の取得"""
    missing_deps = check_dependencies()
    
    return {
        'version': __version__,
        'description': __description__,
        'web_available': WEB_AVAILABLE,
        'missing_dependencies': missing_deps,
        'modules': {
            'dashboard': 'Main dashboard class',
            'config': 'Configuration and settings',
            'real_time_manager': 'Real-time data management',
            'visualization': 'Chart and graph generation',
            'alert_manager': 'Alert and notification system',
            'routes': 'Flask route definitions',
            'socket_handlers': 'WebSocket event handlers',
            'templates': 'HTML template management'
        }
    }

# モジュール初期化時の依存関係チェック
import warnings

missing_deps = check_dependencies()
if missing_deps:
    warnings.warn(
        f"以下の依存関係が不足しています: {', '.join(missing_deps)}\n"
        f"インストールコマンド: pip install {' '.join(missing_deps)}",
        ImportWarning,
        stacklevel=2
    )