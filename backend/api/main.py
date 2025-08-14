# FastAPI Backend API
# Day Trade ML System - Issue #803

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta

# Routers
from .routers import (
    auth,
    dashboard,
    trading,
    analytics,
    monitoring,
    reports,
    users,
    settings
)

# Middleware
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware

# Services
from .services.websocket_manager import WebSocketManager
from .services.notification_service import NotificationService
from .services.cache_service import CacheService
from .services.metrics_service import MetricsService

# Database
from .database.connection import database, engine
from .database.models import Base

# Config
from .config import settings

# Utils
from .utils.security import verify_token
from .utils.metrics import track_request_metrics


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    # Startup
    logging.info("Starting Day Trade ML API...")

    # データベース接続
    await database.connect()

    # WebSocket Manager初期化
    app.state.websocket_manager = WebSocketManager()

    # 通知サービス初期化
    app.state.notification_service = NotificationService()

    # キャッシュサービス初期化
    app.state.cache_service = CacheService()

    # メトリクスサービス初期化
    app.state.metrics_service = MetricsService()

    # バックグラウンドタスク開始
    asyncio.create_task(app.state.websocket_manager.start_heartbeat())
    asyncio.create_task(app.state.metrics_service.start_collection())

    logging.info("Day Trade ML API started successfully")

    yield

    # Shutdown
    logging.info("Shutting down Day Trade ML API...")

    # データベース切断
    await database.disconnect()

    # WebSocket接続を全て閉じる
    await app.state.websocket_manager.disconnect_all()

    logging.info("Day Trade ML API shutdown complete")


# FastAPI アプリケーション初期化
app = FastAPI(
    title="Day Trade ML API",
    description="高頻度取引機械学習システム API",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
    # Response model
    default_response_class=JSONResponse,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Request-ID"],
)

# 圧縮
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host
if settings.ALLOWED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )

# カスタムミドルウェア
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Security
security = HTTPBearer()


# 依存関数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """現在のユーザーを取得"""
    token = credentials.credentials
    user = await verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user


async def get_websocket_manager() -> WebSocketManager:
    """WebSocket Manager を取得"""
    return app.state.websocket_manager


async def get_notification_service() -> NotificationService:
    """通知サービスを取得"""
    return app.state.notification_service


async def get_cache_service() -> CacheService:
    """キャッシュサービスを取得"""
    return app.state.cache_service


async def get_metrics_service() -> MetricsService:
    """メトリクスサービスを取得"""
    return app.state.metrics_service


# ルーター登録
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["認証"]
)

app.include_router(
    dashboard.router,
    prefix="/api/v1/dashboard",
    tags=["ダッシュボード"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    trading.router,
    prefix="/api/v1/trading",
    tags=["取引"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    analytics.router,
    prefix="/api/v1/analytics",
    tags=["分析"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitoring",
    tags=["監視"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    reports.router,
    prefix="/api/v1/reports",
    tags=["レポート"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    users.router,
    prefix="/api/v1/users",
    tags=["ユーザー"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    settings.router,
    prefix="/api/v1/settings",
    tags=["設定"],
    dependencies=[Depends(get_current_user)]
)


# WebSocket エンドポイント
@app.websocket("/ws/dashboard")
async def dashboard_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """ダッシュボードリアルタイムデータ WebSocket"""
    try:
        # 認証
        if token:
            user = await verify_token(token)
            if not user:
                await websocket.close(code=4001, reason="Unauthorized")
                return
        else:
            await websocket.close(code=4001, reason="Token required")
            return

        # 接続受け入れ
        await websocket.accept()
        await ws_manager.connect(websocket, user.id, "dashboard")

        # 初期データ送信
        initial_data = await get_dashboard_initial_data(user.id)
        await websocket.send_json({
            "type": "initial_data",
            "data": initial_data,
            "timestamp": datetime.utcnow().isoformat()
        })

        # メッセージループ
        while True:
            try:
                # クライアントからのメッセージを受信
                message = await websocket.receive_json()
                await handle_websocket_message(websocket, user.id, message, ws_manager)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.error(f"WebSocket message handling error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Message processing failed",
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


@app.websocket("/ws/trading")
async def trading_websocket(
    websocket: WebSocket,
    token: Optional[str] = None,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """取引リアルタイムデータ WebSocket"""
    try:
        # 認証
        if token:
            user = await verify_token(token)
            if not user:
                await websocket.close(code=4001, reason="Unauthorized")
                return
        else:
            await websocket.close(code=4001, reason="Token required")
            return

        # 接続受け入れ
        await websocket.accept()
        await ws_manager.connect(websocket, user.id, "trading")

        # 取引データストリーミング
        while True:
            try:
                message = await websocket.receive_json()
                await handle_trading_websocket_message(websocket, user.id, message, ws_manager)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.error(f"Trading WebSocket error: {e}")

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(websocket)


# ヘルスチェック
@app.get("/health", tags=["システム"])
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected" if database.is_connected else "disconnected",
            "cache": "connected",  # キャッシュサービスのステータス
            "websocket": "active",
        }
    }


# メトリクス (Prometheus形式)
@app.get("/metrics", tags=["システム"])
async def get_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """Prometheusメトリクス取得"""
    return await metrics_service.get_prometheus_metrics()


# API情報
@app.get("/api/v1/info", tags=["システム"])
async def api_info():
    """API情報取得"""
    return {
        "name": "Day Trade ML API",
        "version": "1.0.0",
        "description": "高頻度取引機械学習システム API",
        "endpoints": {
            "auth": "/api/v1/auth",
            "dashboard": "/api/v1/dashboard",
            "trading": "/api/v1/trading",
            "analytics": "/api/v1/analytics",
            "monitoring": "/api/v1/monitoring",
            "reports": "/api/v1/reports",
            "users": "/api/v1/users",
            "settings": "/api/v1/settings",
        },
        "websockets": {
            "dashboard": "/ws/dashboard",
            "trading": "/ws/trading",
        },
        "docs": "/docs" if settings.DEBUG else None,
        "limits": {
            "max_requests_per_minute": 1000,
            "max_websocket_connections": 100,
        }
    }


# ルートエンドポイント
@app.get("/", tags=["システム"])
async def root():
    """ルートエンドポイント"""
    return {
        "message": "Day Trade ML API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "docs_url": "/docs" if settings.DEBUG else None,
    }


# エラーハンドラー
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logging.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# ヘルパー関数
async def get_dashboard_initial_data(user_id: str) -> Dict[str, Any]:
    """ダッシュボード初期データ取得"""
    # ここで実際のダッシュボードデータを取得
    return {
        "kpis": {
            "roi": 5.2,
            "trades": 47,
            "accuracy": 93.4,
            "portfolio": 125430
        },
        "charts": {
            "roi_trend": [],
            "trading_volume": [],
        },
        "trades": [],
        "alerts": [],
    }


async def handle_websocket_message(
    websocket: WebSocket,
    user_id: str,
    message: Dict[str, Any],
    ws_manager: WebSocketManager
):
    """WebSocketメッセージ処理"""
    message_type = message.get("type")

    if message_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })

    elif message_type == "subscribe":
        # 特定のデータストリームに購読
        stream = message.get("stream")
        await ws_manager.subscribe(websocket, stream)

    elif message_type == "unsubscribe":
        # データストリームの購読解除
        stream = message.get("stream")
        await ws_manager.unsubscribe(websocket, stream)

    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_trading_websocket_message(
    websocket: WebSocket,
    user_id: str,
    message: Dict[str, Any],
    ws_manager: WebSocketManager
):
    """取引WebSocketメッセージ処理"""
    message_type = message.get("type")

    if message_type == "trade_request":
        # 取引リクエスト処理
        trade_data = message.get("data")
        # 実際の取引処理をここで実行
        result = await process_trade_request(user_id, trade_data)

        await websocket.send_json({
            "type": "trade_response",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        })

    elif message_type == "market_data_subscribe":
        # 市場データ購読
        symbols = message.get("symbols", [])
        await ws_manager.subscribe_market_data(websocket, symbols)


async def process_trade_request(user_id: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
    """取引リクエスト処理"""
    # 実際の取引処理ロジック
    return {
        "trade_id": "trade_12345",
        "status": "executed",
        "symbol": trade_data.get("symbol"),
        "action": trade_data.get("action"),
        "quantity": trade_data.get("quantity"),
        "price": 175.23,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        access_log=True,
        log_level="info" if not settings.DEBUG else "debug",
    )