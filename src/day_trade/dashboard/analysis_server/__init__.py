"""
分析ダッシュボードサーバー統合パッケージ

元のanalysis_dashboard_server.pyとの下位互換性を提供し、
モジュラーアーキテクチャへの移行を支援します。
"""

from .app_config import (
    create_app,
    get_analysis_engine,
    get_chart_manager,
    get_educational_system,
    get_report_manager,
    get_system_status,
    print_server_info,
)
from .websocket_handlers import (
    ConnectionManager,
    get_connection_manager,
    handle_websocket_connection,
)

# FastAPIアプリケーションの作成
app = create_app()

# ルーターの登録
from .analysis_routes import router as analysis_router
from .chart_routes import router as chart_router
from .education_routes import router as education_router
from .report_routes import router as report_router
from .websocket_handlers import handle_websocket_connection

# WebSocketエンドポイント
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket接続エンドポイント"""
    await handle_websocket_connection(websocket)

# APIルーターの登録
app.include_router(analysis_router)
app.include_router(chart_router)
app.include_router(report_router)
app.include_router(education_router)

# 自動取引・注文実行エンドポイントの明示的禁止
from fastapi import HTTPException

@app.api_route("/api/trading/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def trading_disabled(path: str):
    """取引関連API完全無効化"""
    raise HTTPException(
        status_code=403,
        detail="自動取引機能は完全に無効化されています。このシステムは分析専用です。"
    )

@app.api_route("/api/orders/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def orders_disabled(path: str):
    """注文関連API完全無効化"""
    raise HTTPException(
        status_code=403,
        detail="注文実行機能は完全に無効化されています。このシステムは分析専用です。"
    )

# 下位互換性のためのエクスポート
__all__ = [
    "app",
    "create_app",
    "get_analysis_engine",
    "get_chart_manager",
    "get_educational_system",
    "get_report_manager",
    "get_system_status",
    "get_connection_manager",
    "ConnectionManager",
    "handle_websocket_connection",
    "print_server_info",
]

# メイン実行時の処理
if __name__ == "__main__":
    import uvicorn
    
    print_server_info()
    uvicorn.run(app, host="127.0.0.1", port=8000)