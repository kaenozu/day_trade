"""
Web Routes Package - Issue #959 リファクタリング対応
"""

from .main_routes import setup_main_routes
from .api_routes import setup_api_routes

__all__ = ['setup_main_routes', 'setup_api_routes']
