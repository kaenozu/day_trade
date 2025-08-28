"""
外部APIクライアント データモデル定義
APIリクエスト・レスポンス・設定のデータクラス
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .enums import APIProvider, DataType, RequestMethod

# セキュア APIクライアント統合（オプショナル）
try:
    from ...api.secure_api_client import (
        SecurityLevel,
        SecureAPIKeyManager,
        SecureURLBuilder,
        URLSecurityPolicy,
    )

    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False
    SecurityLevel = None
    SecureAPIKeyManager = None
    SecureURLBuilder = None
    URLSecurityPolicy = None

# セキュリティマネージャー（オプショナル）
try:
    from ...core.security_manager import SecurityManager
except ImportError:
    SecurityManager = None


@dataclass
class APIEndpoint:
    """APIエンドポイント設定"""

    provider: APIProvider
    data_type: DataType
    endpoint_url: str
    method: RequestMethod = RequestMethod.GET

    # レート制限設定
    rate_limit_per_second: float = 1.0
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # 認証設定
    requires_auth: bool = False
    auth_header_name: Optional[str] = None
    auth_param_name: Optional[str] = None

    # データ変換設定
    response_format: str = "json"  # json, csv, xml
    data_path: Optional[str] = None  # JSONレスポンス内のデータパス

    # エラー回復設定
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: int = 30


@dataclass
class APIRequest:
    """APIリクエスト"""

    endpoint: APIEndpoint
    params: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None

    # リクエストメタデータ
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    priority: int = 1  # 1=高, 2=中, 3=低


@dataclass
class APIResponse:
    """APIレスポンス"""

    request: APIRequest
    status_code: int
    response_data: Any
    headers: Dict[str, str]

    # 実行メタデータ
    response_time_ms: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

    # 処理結果
    normalized_data: Optional[pd.DataFrame] = None
    cache_key: Optional[str] = None


@dataclass
class RateLimitState:
    """レート制限状態"""

    provider: APIProvider
    requests_per_second: int = 0
    requests_per_minute: int = 0
    requests_per_hour: int = 0

    # タイムウィンドウ
    second_window_start: datetime = field(default_factory=datetime.now)
    minute_window_start: datetime = field(default_factory=datetime.now)
    hour_window_start: datetime = field(default_factory=datetime.now)

    # 制限情報
    last_request_time: Optional[datetime] = None
    blocked_until: Optional[datetime] = None


@dataclass
class APIConfig:
    """API設定（セキュリティ強化版）"""

    # 基本設定
    user_agent: str = "DayTrade-API-Client/1.0"
    default_timeout_seconds: int = 30
    max_concurrent_requests: int = 10

    # レート制限設定
    global_rate_limit_per_second: float = 5.0
    respect_server_rate_limits: bool = True
    rate_limit_buffer_factor: float = 0.8  # 80%の余裕を持つ

    # キャッシュ設定
    enable_response_caching: bool = True
    cache_ttl_seconds: int = 300  # 5分
    cache_size_limit: int = 1000

    # エラー回復設定
    default_max_retries: int = 3
    exponential_backoff: bool = True
    max_backoff_seconds: float = 60.0

    # セキュリティ設定（強化版）
    security_manager: Optional[SecurityManager] = None
    secure_key_manager: Optional[SecureAPIKeyManager] = None
    secure_url_builder: Optional[SecureURLBuilder] = None
    security_level: Optional[SecurityLevel] = (
        SecurityLevel.MEDIUM if SECURE_API_AVAILABLE else None
    )

    api_key_prefix_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "yahoo_finance": "YF_" + "API_" + "KEY",
            "alpha_vantage": "AV_" + "API_" + "KEY",
            "iex_cloud": "IEX_" + "API_" + "KEY",
            "finnhub": "FINNHUB_" + "API_" + "KEY",
            "polygon": "POLYGON_" + "API_" + "KEY",
            "twelve_data": "TWELVE_DATA_" + "API_" + "KEY",
        }
    )

    # セキュアURL構築ポリシー
    url_security_policy: Optional[URLSecurityPolicy] = None

    # 廃止予定フィールド（後方互換性のため残存）
    api_keys: Dict[str, str] = field(default_factory=dict)
    oauth_tokens: Dict[str, str] = field(default_factory=dict)