#!/usr/bin/env python3
"""
Webダッシュボード 静的ファイル生成モジュール

CSS、JavaScript静的ファイル生成機能の統合インターフェース
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger
from .css_generator import CSSGenerator
from .js_generator import JSGenerator

logger = get_context_logger(__name__)


class StaticFileGenerator:
    """静的ファイル生成クラス"""

    def __init__(self):
        """初期化"""
        self.css_generator = CSSGenerator()
        self.js_generator = JSGenerator()

    def create_static_files(self, static_dir: Path, security_manager):
        """静的ファイル作成"""
        # CSS ファイル
        self.css_generator.create_css_file(static_dir, security_manager)
        # メインダッシュボード JavaScript
        self.js_generator.create_dashboard_js(static_dir, security_manager)
        # 分析ダッシュボード JavaScript
        self.js_generator.create_analysis_js(static_dir, security_manager)
        
        logger.info("静的ファイルの生成が完了しました")