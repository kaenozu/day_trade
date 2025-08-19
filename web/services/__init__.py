"""
Web Services Package - Issue #959 リファクタリング対応
"""

from .recommendation_service import RecommendationService
from .template_service import TemplateService

__all__ = ['RecommendationService', 'TemplateService']
