"""
分析専用ダッシュボードサーバー

【重要】自動取引機能は完全に無効化されています
分析・情報提供・教育支援のみを行うセーフモードサーバー
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..automation.analysis_only_engine import AnalysisOnlyEngine
from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

app = FastAPI(
    title="Day Trade 分析専用ダッシュボード",
    description="完全セーフモード - 分析・教育・研究専用システム",
    version="1.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境用、本番では制限
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信（存在する場合のみ）
static_dir = "src/day_trade/dashboard/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# グローバル変数
analysis_engine: Optional[AnalysisOnlyEngine] = None
connected_clients: Dict[str, WebSocket] = {}


class ConnectionManager:
    """WebSocket接続管理"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"新しいクライアント接続: {len(self.active_connections)}台接続中")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"クライアント切断: {len(self.active_connections)}台接続中")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message, ensure_ascii=False))

    async def broadcast(self, message: dict):
        if self.active_connections:
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message, ensure_ascii=False))
                except Exception:
                    # 接続が切れている場合は削除
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """サーバー起動時の初期化"""
    global analysis_engine

    # セーフモード確認
    if not is_safe_mode():
        raise RuntimeError("セーフモードでない場合はサーバーを起動できません")

    logger.info("分析専用ダッシュボードサーバー起動中...")
    logger.info("※ 自動取引機能は完全に無効化されています")
    logger.info("※ 分析・教育・研究専用モードで動作します")

    # 分析エンジンの初期化
    test_symbols = ["7203", "8306", "9984", "6758", "4689"]  # サンプル銘柄
    analysis_engine = AnalysisOnlyEngine(test_symbols, update_interval=10.0)

    logger.info("分析専用ダッシュボードサーバー起動完了")


@app.on_event("shutdown")
async def shutdown_event():
    """サーバー終了時のクリーンアップ"""
    global analysis_engine

    if analysis_engine and analysis_engine.status.value == "running":
        await analysis_engine.stop()

    logger.info("分析専用ダッシュボードサーバー停止")


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """メインダッシュボードページ"""
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Day Trade - 分析専用ダッシュボード</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; padding: 20px; background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; border-radius: 10px;
                text-align: center; margin-bottom: 20px;
            }
            .safe-mode-banner {
                background: #28a745; color: white;
                padding: 10px; text-align: center;
                border-radius: 5px; margin-bottom: 20px;
                font-weight: bold;
            }
            .container {
                display: grid; grid-template-columns: 1fr 1fr;
                gap: 20px; max-width: 1200px; margin: 0 auto;
            }
            .card {
                background: white; padding: 20px; border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .status { font-size: 1.2em; margin: 10px 0; }
            .enabled { color: #28a745; }
            .disabled { color: #dc3545; text-decoration: line-through; }
            button {
                background: #667eea; color: white; border: none;
                padding: 10px 20px; border-radius: 5px; cursor: pointer;
            }
            button:hover { background: #5a6fd8; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            #log {
                background: #000; color: #0f0; padding: 10px;
                height: 200px; overflow-y: scroll; font-family: monospace;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Day Trade 分析専用ダッシュボード</h1>
            <p>教育・研究・分析専用システム</p>
        </div>

        <div class="safe-mode-banner">
            🔒 セーフモード有効 - 自動取引は完全に無効化されています
        </div>

        <div class="container">
            <div class="card">
                <h3>🎯 システム状態</h3>
                <div class="status">📈 市場分析: <span class="enabled">有効</span></div>
                <div class="status">📊 データ取得: <span class="enabled">有効</span></div>
                <div class="status">🎓 教育機能: <span class="enabled">有効</span></div>
                <div class="status">⚡ シグナル生成: <span class="enabled">有効</span></div>
                <div class="status">🚫 自動取引: <span class="disabled">完全無効</span></div>
                <div class="status">🚫 注文実行: <span class="disabled">完全無効</span></div>

                <hr>
                <button onclick="startAnalysis()">分析開始</button>
                <button onclick="stopAnalysis()">分析停止</button>
                <button onclick="getReport()">レポート生成</button>
            </div>

            <div class="card">
                <h3>📋 リアルタイム分析ログ</h3>
                <div id="log">システム待機中...\n</div>
            </div>
        </div>

        <script>
            const ws = new WebSocket(`ws://localhost:8000/ws`);
            const log = document.getElementById('log');

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                log.innerHTML += new Date().toLocaleTimeString() + ' - ' + data.message + '\\n';
                log.scrollTop = log.scrollHeight;
            };

            ws.onopen = function() {
                log.innerHTML += '[接続] ダッシュボード接続完了\\n';
            };

            ws.onclose = function() {
                log.innerHTML += '[切断] サーバー接続終了\\n';
            };

            async function startAnalysis() {
                try {
                    const response = await fetch('/api/analysis/start', { method: 'POST' });
                    const result = await response.json();
                    log.innerHTML += '[分析] ' + result.message + '\\n';
                } catch (e) {
                    log.innerHTML += '[エラー] 分析開始に失敗: ' + e.message + '\\n';
                }
            }

            async function stopAnalysis() {
                try {
                    const response = await fetch('/api/analysis/stop', { method: 'POST' });
                    const result = await response.json();
                    log.innerHTML += '[分析] ' + result.message + '\\n';
                } catch (e) {
                    log.innerHTML += '[エラー] 分析停止に失敗: ' + e.message + '\\n';
                }
            }

            async function getReport() {
                try {
                    const response = await fetch('/api/analysis/report');
                    const result = await response.json();
                    log.innerHTML += '[レポート] ' + JSON.stringify(result, null, 2) + '\\n';
                } catch (e) {
                    log.innerHTML += '[エラー] レポート取得に失敗: ' + e.message + '\\n';
                }
            }
        </script>
    </body>
    </html>
    """
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続エンドポイント"""
    await manager.connect(websocket)

    try:
        # 接続時の初期メッセージ
        await manager.send_personal_message(
            {
                "type": "connection",
                "message": "分析専用ダッシュボードに接続しました",
                "safe_mode": True,
                "trading_disabled": True,
            },
            websocket,
        )

        while True:
            # クライアントからのメッセージを待機
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # セーフモード確認メッセージ
            if message_data.get("type") == "ping":
                await manager.send_personal_message(
                    {
                        "type": "pong",
                        "message": "分析システム稼働中（セーフモード）",
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/system/status")
async def get_system_status():
    """システム状態取得"""
    config = get_current_trading_config()

    if analysis_engine:
        engine_status = analysis_engine.get_status()
    else:
        engine_status = {"status": "not_initialized"}

    return {
        "safe_mode": is_safe_mode(),
        "trading_disabled": not config.enable_automatic_trading,
        "analysis_engine": engine_status,
        "server_time": datetime.now().isoformat(),
        "system_type": "analysis_only",
        "warning": "自動取引機能は完全に無効化されています",
    }


@app.post("/api/analysis/start")
async def start_analysis():
    """分析エンジン開始"""
    global analysis_engine

    if not analysis_engine:
        raise HTTPException(
            status_code=500, detail="分析エンジンが初期化されていません"
        )

    if analysis_engine.status.value == "running":
        return {"message": "分析エンジンは既に実行中です", "status": "already_running"}

    # 非同期で分析開始
    asyncio.create_task(analysis_engine.start())

    # 開始通知を全クライアントにブロードキャスト
    await manager.broadcast(
        {
            "type": "analysis_start",
            "message": "分析エンジンを開始しました（セーフモード）",
            "timestamp": datetime.now().isoformat(),
        }
    )

    return {"message": "分析エンジンを開始しました", "status": "started"}


@app.post("/api/analysis/stop")
async def stop_analysis():
    """分析エンジン停止"""
    global analysis_engine

    if not analysis_engine:
        raise HTTPException(
            status_code=500, detail="分析エンジンが初期化されていません"
        )

    if analysis_engine.status.value == "stopped":
        return {"message": "分析エンジンは既に停止中です", "status": "already_stopped"}

    await analysis_engine.stop()

    # 停止通知を全クライアントにブロードキャスト
    await manager.broadcast(
        {
            "type": "analysis_stop",
            "message": "分析エンジンを停止しました",
            "timestamp": datetime.now().isoformat(),
        }
    )

    return {"message": "分析エンジンを停止しました", "status": "stopped"}


@app.get("/api/analysis/report")
async def get_analysis_report():
    """分析レポート取得"""
    global analysis_engine

    if not analysis_engine:
        raise HTTPException(
            status_code=500, detail="分析エンジンが初期化されていません"
        )

    # 市場サマリー取得
    market_summary = analysis_engine.get_market_summary()

    # 個別銘柄分析取得
    all_analyses = analysis_engine.get_all_analyses()

    # 最新レポート取得
    latest_report = analysis_engine.get_latest_report()

    return {
        "market_summary": market_summary,
        "individual_analyses": {
            symbol: {
                "current_price": float(analysis.current_price),
                "analysis_time": analysis.analysis_timestamp.isoformat(),
                "recommendations": analysis.recommendations[:3],  # 上位3つ
                "volatility": analysis.volatility,
                "price_trend": analysis.price_trend,
                "volume_trend": analysis.volume_trend,
            }
            for symbol, analysis in all_analyses.items()
        },
        "latest_report": latest_report.__dict__ if latest_report else None,
        "generated_at": datetime.now().isoformat(),
        "disclaimer": "これは分析情報です。投資判断は自己責任で行ってください。",
    }


@app.get("/api/analysis/symbols")
async def get_monitored_symbols():
    """監視銘柄一覧取得"""
    global analysis_engine

    if not analysis_engine:
        return {"symbols": [], "count": 0}

    return {
        "symbols": analysis_engine.symbols,
        "count": len(analysis_engine.symbols),
        "status": analysis_engine.status.value,
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🔒 Day Trade 分析専用ダッシュボード")
    print("=" * 80)
    print("⚠️  重要: 自動取引機能は完全に無効化されています")
    print("✅  分析・教育・研究専用システムとして動作します")
    print("🌐  ダッシュボード: http://localhost:8000")
    print("=" * 80)

    uvicorn.run(app, host="127.0.0.1", port=8000)
