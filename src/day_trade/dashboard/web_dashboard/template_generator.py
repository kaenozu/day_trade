#!/usr/bin/env python3
"""
Webダッシュボード テンプレート生成モジュール

HTMLテンプレート生成機能の統合インターフェース
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger
from .dashboard_template import DashboardTemplateGenerator
from .analysis_template import AnalysisTemplateGenerator

logger = get_context_logger(__name__)


class TemplateGenerator:
    """HTMLテンプレート生成クラス"""

    def __init__(self):
        """初期化"""
        self.dashboard_generator = DashboardTemplateGenerator()
        self.analysis_generator = AnalysisTemplateGenerator()

    def create_templates(self, templates_dir: Path, security_manager):
        """テンプレート作成"""
        # メインダッシュボードテンプレート
        self.dashboard_generator.create_dashboard_template(templates_dir, security_manager)
        # 分析ダッシュボードテンプレート
        self.analysis_generator.create_analysis_template(templates_dir, security_manager)
        
        logger.info("HTMLテンプレートの生成が完了しました")