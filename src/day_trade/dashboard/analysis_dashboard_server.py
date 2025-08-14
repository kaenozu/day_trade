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
from .custom_reports import CustomReportManager
from .educational_system import EducationalSystem
from .interactive_charts import InteractiveChartManager

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
chart_manager: Optional[InteractiveChartManager] = None
report_manager: Optional[CustomReportManager] = None
educational_system: Optional[EducationalSystem] = None
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
    global analysis_engine, chart_manager, report_manager, educational_system

    # セーフモード確認
    if not is_safe_mode():
        raise RuntimeError("セーフモードでない場合はサーバーを起動できません")

    logger.info("分析専用ダッシュボードサーバー起動中...")
    logger.info("※ 自動取引機能は完全に無効化されています")
    logger.info("※ 分析・教育・研究専用モードで動作します")

    # 分析エンジンの初期化
    test_symbols = ["7203", "8306", "9984", "6758", "4689"]  # サンプル銘柄
    analysis_engine = AnalysisOnlyEngine(test_symbols, update_interval=10.0)

    # インタラクティブチャートマネージャーの初期化
    chart_manager = InteractiveChartManager()
    logger.info("インタラクティブチャート機能を有効化しました")

    # カスタムレポートマネージャーの初期化
    report_manager = CustomReportManager()
    logger.info("カスタムレポート機能を有効化しました")

    # 教育システムの初期化
    educational_system = EducationalSystem()
    logger.info("教育・学習支援機能を有効化しました")

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


# === インタラクティブチャートAPI ===


@app.post("/api/charts/price")
async def create_price_chart(request: dict):
    """価格チャート作成"""
    global chart_manager, analysis_engine

    if not chart_manager or not analysis_engine:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        # サンプルデータ（実際の実装では分析エンジンからデータ取得）
        import numpy as np
        import pandas as pd

        symbols = request.get("symbols", ["7203.T"])
        chart_type = request.get("chartType", "candlestick")
        indicators = request.get("indicators", [])
        show_volume = request.get("showVolume", True)

        # モックデータ生成（実データに置き換え可能）
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        data = {}

        for symbol in symbols:
            base_price = 2000 + hash(symbol) % 1000
            prices = base_price + np.cumsum(np.random.normal(0, 20, len(dates)))

            data[symbol] = pd.DataFrame(
                {
                    "Open": prices + np.random.normal(0, 10, len(dates)),
                    "High": prices + np.abs(np.random.normal(10, 5, len(dates))),
                    "Low": prices - np.abs(np.random.normal(10, 5, len(dates))),
                    "Close": prices,
                    "Volume": np.random.randint(1000000, 10000000, len(dates)),
                },
                index=dates,
            )

        chart_data = chart_manager.create_price_chart(
            data=data,
            symbols=symbols,
            chart_type=chart_type,
            show_volume=show_volume,
            indicators=indicators,
        )

        return chart_data

    except Exception as e:
        logger.error(f"価格チャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/charts/performance")
async def create_performance_chart(request: dict):
    """パフォーマンスチャート作成"""
    global chart_manager

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        symbols = request.get("symbols", ["7203.T"])

        # モックパフォーマンスデータ
        import numpy as np
        import pandas as pd

        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
        performance_data = {}

        for symbol in symbols:
            # ランダムリターンデータ生成
            returns = np.random.normal(0.001, 0.02, len(dates))
            performance_data[f"{symbol} 戦略"] = returns.tolist()

        # ベンチマークデータ
        benchmark = {"TOPIX": np.random.normal(0.0005, 0.018, len(dates)).tolist()}

        chart_data = chart_manager.create_performance_chart(
            performance_data=performance_data,
            dates=dates.to_pydatetime().tolist(),
            benchmark=benchmark,
            cumulative=True,
        )

        return chart_data

    except Exception as e:
        logger.error(f"パフォーマンスチャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/analysis/risk")
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
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/charts/risk-scatter")
async def create_risk_scatter_chart(risk_data: dict):
    """リスク散布図チャート作成"""
    global chart_manager

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        chart_data = chart_manager.create_risk_analysis_chart(
            risk_data=risk_data, chart_type="scatter"
        )
        return chart_data

    except Exception as e:
        logger.error(f"リスク散布図作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/charts/risk-radar")
async def create_risk_radar_chart(risk_data: dict):
    """リスクレーダーチャート作成"""
    global chart_manager

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        chart_data = chart_manager.create_risk_analysis_chart(
            risk_data=risk_data, chart_type="radar"
        )
        return chart_data

    except Exception as e:
        logger.error(f"リスクレーダーチャート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/charts/market-overview")
async def get_market_overview():
    """市場概要ダッシュボード取得"""
    global chart_manager

    if not chart_manager:
        raise HTTPException(
            status_code=503, detail="チャートマネージャーが初期化されていません"
        )

    try:
        # モック市場データ
        import numpy as np
        import pandas as pd

        market_data = {
            "sector_performance": {
                "テクノロジー": 0.15,
                "金融": 0.08,
                "ヘルスケア": 0.12,
                "消費財": 0.06,
                "エネルギー": -0.03,
            },
            "volume_trend": {
                "dates": pd.date_range(start="2024-12-01", end="2024-12-31", freq="D")
                .strftime("%Y-%m-%d")
                .tolist(),
                "volumes": np.random.randint(1000000, 5000000, 31).tolist(),
            },
            "price_distribution": np.random.normal(2500, 500, 1000).tolist(),
            "correlation_matrix": {
                "symbols": ["7203.T", "8306.T", "9984.T"],
                "values": [[1.0, 0.3, 0.1], [0.3, 1.0, 0.6], [0.1, 0.6, 1.0]],
            },
        }

        chart_data = chart_manager.create_market_overview_dashboard(
            market_data=market_data, layout="grid"
        )

        return chart_data

    except Exception as e:
        logger.error(f"市場概要ダッシュボード作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"ダッシュボード作成エラー: {str(e)}"
        ) from e


# === 強化ダッシュボードページ ===


@app.get("/enhanced", response_class=HTMLResponse)
async def enhanced_dashboard():
    """強化版ダッシュボードページ"""
    template_path = "src/day_trade/dashboard/templates/enhanced_dashboard.html"

    try:
        with open(template_path, encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>強化版ダッシュボード</h1><p>テンプレートファイルが見つかりません</p>",
            status_code=404,
        )


# === カスタムレポートAPI ===


@app.post("/api/reports/create")
async def create_custom_report(request: dict):
    """カスタムレポート作成"""
    global report_manager, analysis_engine

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        template = request.get("template", "standard")
        custom_metrics = request.get("metrics", [])
        title = request.get("title")

        # サンプルデータ（実際の実装では分析エンジンからデータ取得）
        import numpy as np

        sample_data = {
            "portfolio_returns": np.random.normal(0.001, 0.02, 252).tolist(),
            "benchmark_returns": np.random.normal(0.0008, 0.018, 252).tolist(),
            "symbols": ["7203.T", "8306.T", "9984.T", "6758.T", "6861.T"],
            "sector_weights": {
                "technology": 0.35,
                "finance": 0.25,
                "healthcare": 0.20,
                "consumer": 0.15,
                "energy": 0.05,
            },
        }

        report_result = report_manager.create_custom_report(
            data=sample_data,
            template=template,
            custom_metrics=custom_metrics,
            title=title,
        )

        return report_result

    except Exception as e:
        logger.error(f"カスタムレポート作成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/reports/export")
async def export_report(request: dict):
    """レポートエクスポート"""
    global report_manager

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        report_data = request.get("report_data")
        export_format = request.get("format", "json")  # json, excel, pdf
        filename = request.get("filename")

        if not report_data:
            raise HTTPException(
                status_code=400, detail="レポートデータが提供されていません"
            )

        if export_format == "json":
            filepath = report_manager.export_to_json(report_data, filename)
        elif export_format == "excel":
            filepath = report_manager.export_to_excel(report_data, filename)
        elif export_format == "pdf":
            try:
                filepath = report_manager.export_to_pdf(report_data, filename)
            except ImportError as import_error:
                raise HTTPException(
                    status_code=501, detail="PDFエクスポートライブラリが利用できません"
                ) from import_error
        else:
            raise HTTPException(
                status_code=400, detail="サポートされていないエクスポート形式"
            )

        return {
            "success": True,
            "filepath": filepath,
            "format": export_format,
            "message": f"{export_format.upper()}ファイルが正常にエクスポートされました",
        }

    except Exception as e:
        logger.error(f"レポートエクスポートエラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/reports/templates")
async def get_report_templates():
    """利用可能レポートテンプレート一覧取得"""
    global report_manager

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        templates = report_manager.get_available_templates()
        return {"success": True, "templates": templates}

    except Exception as e:
        logger.error(f"テンプレート取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/reports/metrics")
async def get_available_metrics():
    """利用可能指標一覧取得"""
    global report_manager

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        metrics = report_manager.get_available_metrics()
        return {"success": True, "metrics": metrics}

    except Exception as e:
        logger.error(f"指標取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/reports/schedule")
async def schedule_periodic_report(request: dict):
    """定期レポートスケジュール設定"""
    global report_manager

    if not report_manager:
        raise HTTPException(
            status_code=503, detail="レポートマネージャーが初期化されていません"
        )

    try:
        template = request.get("template", "standard")
        frequency = request.get("frequency", "monthly")  # daily, weekly, monthly
        recipients = request.get("recipients", [])

        if frequency not in ["daily", "weekly", "monthly"]:
            raise HTTPException(status_code=400, detail="無効な頻度が指定されました")

        schedule_result = report_manager.schedule_periodic_report(
            template=template, frequency=frequency, recipients=recipients
        )

        return schedule_result

    except Exception as e:
        logger.error(f"定期レポートスケジュール設定エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


# === 教育システムAPI ===


@app.get("/api/education/glossary/{term}")
async def get_term_explanation(term: str):
    """用語解説取得"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        explanation = educational_system.get_term_explanation(term)
        if explanation:
            return {"success": True, "explanation": explanation}
        else:
            return {"success": False, "message": "該当する用語が見つかりませんでした"}

    except Exception as e:
        logger.error(f"用語解説取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/education/glossary")
async def search_glossary(query: str = "", category: str = None):
    """用語集検索"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        if query:
            results = educational_system.search_glossary(query, category)
        else:
            # 全カテゴリ取得
            categories = educational_system.get_categories()
            return {"success": True, "categories": categories}

        return {"success": True, "results": results, "count": len(results)}

    except Exception as e:
        logger.error(f"用語集検索エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/education/strategies/{strategy}")
async def get_strategy_guide(strategy: str):
    """投資戦略ガイド取得"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        guide = educational_system.get_strategy_guide(strategy)
        if guide:
            return {"success": True, "guide": guide}
        else:
            return {
                "success": False,
                "message": "該当する戦略ガイドが見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"戦略ガイド取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/education/case-studies/{case}")
async def get_case_study(case: str):
    """過去事例分析取得"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        study = educational_system.get_case_study(case)
        if study:
            return {"success": True, "case_study": study}
        else:
            return {
                "success": False,
                "message": "該当する事例分析が見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"事例分析取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/education/learning/{module}")
async def get_learning_module(module: str):
    """学習モジュール取得"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        learning_module = educational_system.get_learning_module(module)
        if learning_module:
            return {"success": True, "module": learning_module}
        else:
            return {
                "success": False,
                "message": "該当する学習モジュールが見つかりませんでした",
            }

    except Exception as e:
        logger.error(f"学習モジュール取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"学習モジュール取得エラー: {str(e)}"
        ) from e


@app.post("/api/education/quiz")
async def generate_quiz(request: dict):
    """クイズ問題生成"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        topic = request.get("topic")
        difficulty = request.get("difficulty")
        count = request.get("count", 5)

        if count > 20:
            raise HTTPException(
                status_code=400, detail="問題数は20問以下にしてください"
            )

        questions = educational_system.get_quiz_questions(
            topic=topic, difficulty=difficulty, count=count
        )

        return {
            "success": True,
            "questions": questions,
            "count": len(questions),
            "settings": {"topic": topic, "difficulty": difficulty, "count": count},
        }

    except Exception as e:
        logger.error(f"クイズ生成エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.get("/api/education/progress/{user_id}")
async def get_learning_progress(user_id: str):
    """学習進捗取得"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        progress = educational_system.get_learning_progress(user_id)
        recommendations = educational_system.get_recommendations(user_id)

        return {
            "success": True,
            "progress": progress,
            "recommendations": recommendations,
            "user_id": user_id,
        }

    except Exception as e:
        logger.error(f"学習進捗取得エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


@app.post("/api/education/progress/{user_id}")
async def update_learning_progress(user_id: str, request: dict):
    """学習進捗更新"""
    global educational_system

    if not educational_system:
        raise HTTPException(
            status_code=503, detail="教育システムが初期化されていません"
        )

    try:
        module = request.get("module")
        quiz_score = request.get("quiz_score")
        study_time = request.get("study_time")

        progress = educational_system.update_learning_progress(
            user_id=user_id,
            module=module,
            quiz_score=quiz_score,
            study_time=study_time,
        )

        return {"success": True, "progress": progress, "user_id": user_id}

    except Exception as e:
        logger.error(f"学習進捗更新エラー: {e}")
        raise HTTPException(
            status_code=500, detail=f"チャート作成エラー: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🔒 Day Trade 分析専用ダッシュボード")
    print("=" * 80)
    print("⚠️  重要: 自動取引機能は完全に無効化されています")
    print("✅  分析・教育・研究専用システムとして動作します")
    print("🌐  ダッシュボード: http://localhost:8000")
    print("=" * 80)


# 自動取引・注文実行エンドポイントの明示的禁止
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

    uvicorn.run(app, host="127.0.0.1", port=8000)
