#!/usr/bin/env python3
"""
Webダッシュボード JavaScript生成モジュール

JavaScript静的ファイル生成機能の統合インターフェース
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger
from .dashboard_js import DashboardJSGenerator
from .analysis_js import AnalysisJSGenerator

logger = get_context_logger(__name__)


class JSGenerator:
    """JavaScript生成クラス"""

    def __init__(self):
        """初期化"""
        self.dashboard_js_generator = DashboardJSGenerator()
        self.analysis_js_generator = AnalysisJSGenerator()

    def create_dashboard_js(self, static_dir: Path, security_manager):
        """メインダッシュボードJavaScript作成"""
        self.dashboard_js_generator.create_dashboard_js(static_dir, security_manager)

    def create_analysis_js(self, static_dir: Path, security_manager):
        """分析ダッシュボードJavaScript作成"""
        self.analysis_js_generator.create_analysis_js(static_dir, security_manager)
        
        logger.info("JavaScriptファイルの生成が完了しました")