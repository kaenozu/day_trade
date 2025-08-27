"""
監視システム用セキュリティミドルウェア

APIエンドポイントとWebSocket接続のセキュリティ機能を提供。
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass, field
from aiohttp import web, WSMsgType
from aiohttp.web import Request, Response, WebSocketResponse
import logging

from ..security.security_manager import SecurityManager


@dataclass
class RateLimitRule:
    """レート制限ルール"""
    requests_per_minute: int
    burst_limit: int = 0  # 瞬間的な許可数
    window_size: int = 60  # 秒


@dataclass
class ClientInfo:
    """クライアント情報"""
    ip_address: str
    requests: list = field(default_factory=list)  # タイムスタンプのリスト
    blocked_until: Optional[datetime] = None
    total_requests: int = 0


class RateLimiter:
    """レート制限器"""
    
    def __init__(self, default_rule: RateLimitRule = None):
        self.default_rule = default_rule or RateLimitRule(requests_per_minute=1000)
        self.clients: Dict[str, ClientInfo] = {}
        self.rules: Dict[str, RateLimitRule] = {}
        
    def add_rule(self, endpoint: str, rule: RateLimitRule):
        """エンドポイント固有のルール追加"""
        self.rules[endpoint] = rule
        
    def is_allowed(self, client_ip: str, endpoint: str = "") -> bool:
        """リクエスト許可判定"""
        now = datetime.now()
        
        # クライアント情報取得または作成
        if client_ip not in self.clients:
            self.clients[client_ip] = ClientInfo(ip_address=client_ip)
            
        client = self.clients[client_ip]
        
        # ブロック期間チェック
        if client.blocked_until and now < client.blocked_until:
            return False
            
        # 適用ルール決定
        rule = self.rules.get(endpoint, self.default_rule)
        
        # 古いリクエストを削除（ウィンドウサイズ外）
        cutoff_time = now - timedelta(seconds=rule.window_size)
        client.requests = [req_time for req_time in client.requests if req_time > cutoff_time]
        
        # レート制限チェック
        if len(client.requests) >= rule.requests_per_minute:
            # ブロック設定
            client.blocked_until = now + timedelta(minutes=5)
            logging.warning(f"レート制限によりクライアントをブロック: {client_ip}")
            return False
            
        # リクエスト記録
        client.requests.append(now)
        client.total_requests += 1
        return True
        
    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """クライアント統計取得"""
        if client_ip not in self.clients:
            return {"requests_in_window": 0, "total_requests": 0, "blocked": False}
            
        client = self.clients[client_ip]
        now = datetime.now()
        
        return {
            "requests_in_window": len(client.requests),
            "total_requests": client.total_requests,
            "blocked": client.blocked_until and now < client.blocked_until,
            "blocked_until": client.blocked_until.isoformat() if client.blocked_until else None
        }


class AuthenticationMiddleware:
    """認証ミドルウェア"""
    
    def __init__(self, security_manager: Optional[SecurityManager] = None,
                 require_auth: bool = False):
        self.security_manager = security_manager
        self.require_auth = require_auth
        self.api_keys: Set[str] = set()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    def add_api_key(self, api_key: str):
        """APIキー追加"""
        # ハッシュ化して保存
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys.add(hashed_key)
        
    def verify_api_key(self, api_key: str) -> bool:
        """APIキー検証"""
        if not api_key:
            return False
            
        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        return hashed_key in self.api_keys
        
    def create_session(self, user_id: str, permissions: list = None) -> str:
        """セッション作成"""
        session_id = hashlib.sha256(f"{user_id}_{time.time()}".encode()).hexdigest()
        self.sessions[session_id] = {
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
        
    def verify_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """セッション検証"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # セッション有効期限チェック（24時間）
        if datetime.now() - session["created_at"] > timedelta(hours=24):
            del self.sessions[session_id]
            return None
            
        # アクティビティ更新
        session["last_activity"] = datetime.now()
        return session
        
    def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """リクエスト認証"""
        if not self.require_auth:
            return {"user_id": "anonymous", "permissions": ["read"]}
            
        # Authorization ヘッダーチェック
        auth_header = request.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            
            # APIキー認証
            if self.verify_api_key(token):
                return {"user_id": "api_user", "permissions": ["read", "write"]}
                
            # セッション認証
            session = self.verify_session(token)
            if session:
                return session
                
        return None


class SecurityMiddleware:
    """セキュリティミドルウェア"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rate_limiter = RateLimiter(
            RateLimitRule(
                requests_per_minute=self.config.get("rate_limit_requests_per_minute", 1000)
            )
        )
        self.auth = AuthenticationMiddleware(
            require_auth=self.config.get("require_authentication", False)
        )
        
        # セキュリティヘッダー設定
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
    async def middleware_handler(self, request: Request, handler: Callable) -> Response:
        """ミドルウェア処理"""
        client_ip = self._get_client_ip(request)
        endpoint = request.path
        
        # レート制限チェック
        if not self.rate_limiter.is_allowed(client_ip, endpoint):
            logging.warning(f"レート制限違反: {client_ip} -> {endpoint}")
            return web.json_response(
                {"error": "Rate limit exceeded"}, 
                status=429,
                headers={"Retry-After": "300"}
            )
            
        # 認証チェック（認証が必要な場合）
        auth_info = self.auth.authenticate_request(request)
        if self.auth.require_auth and not auth_info:
            return web.json_response(
                {"error": "Authentication required"}, 
                status=401
            )
            
        # リクエスト情報をコンテキストに追加
        request["auth_info"] = auth_info
        request["client_ip"] = client_ip
        
        try:
            # ハンドラー実行
            response = await handler(request)
            
            # セキュリティヘッダー追加
            if isinstance(response, web.Response):
                for header, value in self.security_headers.items():
                    response.headers[header] = value
                    
            return response
            
        except Exception as e:
            logging.error(f"ハンドラーエラー: {e}")
            return web.json_response(
                {"error": "Internal server error"}, 
                status=500
            )
            
    def _get_client_ip(self, request: Request) -> str:
        """クライアントIP取得"""
        # プロキシ経由の場合のヘッダーもチェック
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
            
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
            
        return request.remote or "unknown"
        
    def get_security_stats(self) -> Dict[str, Any]:
        """セキュリティ統計取得"""
        return {
            "total_clients": len(self.rate_limiter.clients),
            "blocked_clients": sum(
                1 for client in self.rate_limiter.clients.values()
                if client.blocked_until and datetime.now() < client.blocked_until
            ),
            "active_sessions": len(self.auth.sessions),
            "total_api_keys": len(self.auth.api_keys)
        }


def create_security_middleware(app: web.Application, config: Dict[str, Any] = None) -> SecurityMiddleware:
    """セキュリティミドルウェア作成とアプリケーションへの適用"""
    middleware = SecurityMiddleware(config)
    
    # ミドルウェアをアプリケーションに追加
    app.middlewares.append(middleware.middleware_handler)
    
    return middleware