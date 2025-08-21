"""
Boatrace Web UI・ダッシュボードモジュール
"""

from .dashboard import create_app
from .api_routes import api_bp

__all__ = [
    "create_app",
    "api_bp"
]