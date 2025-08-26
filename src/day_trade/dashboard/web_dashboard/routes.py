#!/usr/bin/env python3
"""
Webダッシュボード ルートモジュール

HTTPルートの統合インターフェース
"""

from ...utils.logging_config import get_context_logger
from .view_routes import setup_view_routes
from .api_routes import setup_api_routes
from .chart_routes import setup_chart_routes

logger = get_context_logger(__name__)


def setup_routes(app, dashboard_core, visualization_engine, security_manager, 
                socketio, debug=False):
    """HTTPルート設定"""
    
    # ビューレート設定
    setup_view_routes(app)
    
    # APIルート設定
    setup_api_routes(
        app, 
        dashboard_core, 
        visualization_engine, 
        security_manager,
        socketio,
        debug
    )
    
    # チャートAPIルート設定
    setup_chart_routes(
        app,
        dashboard_core,
        visualization_engine,
        security_manager
    )
    
    logger.info("HTTPルートの設定が完了しました")