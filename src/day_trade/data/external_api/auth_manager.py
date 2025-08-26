"""
外部APIクライアント 認証管理
セキュアなAPIキー管理と認証機能
"""

from typing import Optional

from ...utils.logging_config import get_context_logger
from .enums import APIProvider
from .models import APIConfig, APIEndpoint

# セキュア APIクライアント統合（オプショナル）
try:
    from ...api.secure_api_client import SecureAPIKeyManager

    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False
    SecureAPIKeyManager = None

# セキュリティマネージャー（オプショナル）
try:
    from ...core.security_manager import SecurityManager
except ImportError:
    SecurityManager = None

logger = get_context_logger(__name__)


class AuthenticationManager:
    """認証管理クラス（セキュリティ強化版）"""

    def __init__(self, config: APIConfig):
        self.config = config

    def get_auth_key(self, endpoint: APIEndpoint) -> Optional[str]:
        """認証キー取得（セキュリティ強化版）"""
        provider_key = endpoint.provider.value

        # 1. セキュアAPIキーマネージャーを使用した取得を優先
        if self.config.secure_key_manager and SECURE_API_AVAILABLE:
            try:
                # ホスト名を取得してセキュリティ検証
                import urllib.parse

                parsed_url = urllib.parse.urlparse(endpoint.endpoint_url)
                host = parsed_url.hostname or "unknown"

                api_key = self.config.secure_key_manager.get_api_key(provider_key, host)
                if api_key:
                    logger.debug(
                        f"セキュアAPIキーマネージャーからキー取得成功: {provider_key}"
                    )
                    return api_key
                else:
                    logger.info(
                        f"セキュアAPIキーマネージャーでキー未登録: {provider_key}"
                    )
            except Exception as e:
                logger.error(f"セキュアAPIキーマネージャーエラー: {e}")

        # 2. 従来のSecurityManagerを使用した安全なキー取得
        if (
            self.config.security_manager
            and provider_key in self.config.api_key_prefix_mapping
        ):
            try:
                env_key_name = self.config.api_key_prefix_mapping[provider_key]
                api_key = self.config.security_manager.get_api_key(env_key_name)
                if api_key:
                    logger.debug(
                        f"セキュリティマネージャーからAPIキー取得: {provider_key}"
                    )
                    return api_key
                else:
                    logger.warning(
                        f"セキュリティマネージャーでAPIキー未設定: {env_key_name}"
                    )
            except Exception as e:
                logger.error(f"セキュリティマネージャーAPIキー取得エラー: {e}")

        # 3. フォールバック: 従来のapi_keys辞書から取得（後方互換性）
        fallback_key = self.config.api_keys.get(provider_key)
        if fallback_key:
            logger.warning(f"従来のAPIキー辞書を使用（非推奨）: {provider_key}")
            return fallback_key

        # 4. APIキーが見つからない場合
        logger.error(f"APIキーが設定されていません: {provider_key}")
        return None

    def add_auth_to_request(
        self, endpoint: APIEndpoint, params: dict, headers: dict
    ) -> bool:
        """リクエストに認証情報を追加"""
        if not endpoint.requires_auth:
            return True

        auth_key = self.get_auth_key(endpoint)
        if not auth_key:
            return False

        # パラメータとして追加
        if endpoint.auth_param_name:
            params[endpoint.auth_param_name] = auth_key

        # ヘッダーとして追加
        if endpoint.auth_header_name:
            headers[endpoint.auth_header_name] = auth_key

        return True