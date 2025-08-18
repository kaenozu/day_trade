"""
Day Trade Web Module - Issue #959 リファクタリング対応
Web関連モジュールのパッケージ
"""

from .services.recommendation_service import RecommendationService
from .services.template_service import TemplateService

__all__ = ['RecommendationService', 'TemplateService']
