#!/usr/bin/env python3
"""
Webダッシュボード 分析ダッシュボードJavaScript生成モジュール

分析ダッシュボード用JavaScript生成機能の統合インターフェース
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger
from .analysis_core_js import AnalysisCoreJSGenerator
from .analysis_ui_js import AnalysisUIJSGenerator

logger = get_context_logger(__name__)


class AnalysisJSGenerator:
    """分析ダッシュボードJavaScript生成クラス"""

    def __init__(self):
        """初期化"""
        self.core_generator = AnalysisCoreJSGenerator()
        self.ui_generator = AnalysisUIJSGenerator()

    def create_analysis_js(self, static_dir: Path, security_manager):
        """分析ダッシュボードJavaScript作成"""
        # コア機能とUI機能を統合
        js_content = (
            self.core_generator.generate_core_js() + 
            "\n\n" + 
            self.ui_generator.generate_ui_js()
        )

        # セキュアなファイル作成
        security_manager.create_secure_file(
            static_dir / "analysis.js", js_content, 0o644
        )
        
        logger.info("分析ダッシュボードJavaScriptの生成が完了しました")