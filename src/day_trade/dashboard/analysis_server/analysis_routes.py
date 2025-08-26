"""
基本分析APIルーティング

分析エンジンの開始/停止、システム状態取得、分析レポート生成API
"""

import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from ...utils.logging_config import get_context_logger
from .app_config import get_analysis_engine, get_system_status
from .websocket_handlers import get_connection_manager

logger = get_context_logger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])
manager = get_connection_manager()


@router.get("/system/status")
async def get_system_status_route():
    """システム状態取得"""
    status = get_system_status()
    status["server_time"] = datetime.now().isoformat()
    return status


@router.post("/analysis/start")
async def start_analysis():
    """分析エンジン開始"""
    analysis_engine = get_analysis_engine()

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


@router.post("/analysis/stop")
async def stop_analysis():
    """分析エンジン停止"""
    analysis_engine = get_analysis_engine()

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


@router.get("/analysis/report")
async def get_analysis_report():
    """分析レポート取得"""
    analysis_engine = get_analysis_engine()

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


@router.get("/analysis/symbols")
async def get_monitored_symbols():
    """監視銘柄一覧取得"""
    analysis_engine = get_analysis_engine()

    if not analysis_engine:
        return {"symbols": [], "count": 0}

    return {
        "symbols": analysis_engine.symbols,
        "count": len(analysis_engine.symbols),
        "status": analysis_engine.status.value,
    }


@router.post("/analysis/risk")
async def get_risk_analysis(request: dict):
    """リスク分析データ取得"""
    try:
        symbols = request.get("symbols", ["7203.T"])

        # モックリスクデータ
        risk_data = {}

        for symbol in symbols:
            base_return = 0.08 + (hash(symbol) % 20 - 10) / 100
            base_vol = 0.15 + (hash(symbol) % 10) / 100

            risk_data[f"{symbol} 戦略"] = {
                "return": base_return,
                "volatility": base_vol,
                "sharpe_ratio": base_return / base_vol,
                "max_drawdown": -0.05 - (hash(symbol) % 10) / 100,
                "sortino_ratio": (base_return / base_vol) * 1.2,
            }

        return risk_data

    except Exception as e:
        logger.error(f"リスク分析データ取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"リスク分析エラー: {str(e)}"
        ) from e


@router.get("/", response_class=HTMLResponse)
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