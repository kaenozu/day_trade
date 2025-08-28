"""
外部APIクライアント パッケージ
後方互換性を保つためのエクスポート定義
"""

import warnings

# 列挙型
from .enums import APIProvider, DataType, RequestMethod

# データモデル
from .models import APIConfig, APIEndpoint, APIRequest, APIResponse, RateLimitState

# メインクライアント
from .api_client import ExternalAPIClient

# コンポーネントクラス（高度な使用向け）
from .auth_manager import AuthenticationManager
from .cache_manager import CacheManager
from .data_normalizers import DataNormalizer
from .error_handlers import ErrorHandler
from .rate_limiter import RateLimiter
from .url_builder import URLBuilder

# 後方互換性のための警告
warnings.warn(
    "新しいモジュラー構造を推奨します。詳細は external_api パッケージのドキュメントを参照してください。",
    DeprecationWarning,
    stacklevel=2
)

# 公開API
__all__ = [
    # 列挙型
    "APIProvider",
    "DataType", 
    "RequestMethod",
    # データモデル
    "APIConfig",
    "APIEndpoint",
    "APIRequest",
    "APIResponse",
    "RateLimitState",
    # メインクラス
    "ExternalAPIClient",
    # コンポーネントクラス
    "AuthenticationManager",
    "CacheManager",
    "DataNormalizer",
    "ErrorHandler",
    "RateLimiter",
    "URLBuilder",
]