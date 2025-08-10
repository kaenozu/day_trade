"""
Enhanced Realtime Dashboard Server

既存のダッシュボードにリアルタイム監視機能を統合
- Feature Store監視
- システムメトリクス収集
- リアルタイムデータストリーミング
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

from ..config.trading_mode_config import get_current_trading_config, is_safe_mode
from ..utils.logging_config import get_context_logger
from ..ml.feature_store import FeatureStore
from .core.metrics_collector import MetricsCollector
from .core.feature_store_monitor import FeatureStoreMonitor
from .core.realtime_stream import RealtimeStream

logger = get_context_logger(__name__)

app = FastAPI(
    title="Day Trade Enhanced Realtime Dashboard",
    description="リアルタイム監視機能付き分析専用ダッシュボード - 完全セーフモード",
    version="2.0.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境用
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信
static_dir = "src/day_trade/dashboard/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# グローバル変数
metrics_collector: Optional[MetricsCollector] = None
feature_store_monitor: Optional[FeatureStoreMonitor] = None
realtime_stream: Optional[RealtimeStream] = None
feature_store: Optional[FeatureStore] = None


@app.on_event("startup")
async def startup_event():
    """Enhanced Dashboard サーバー起動時の初期化"""
    global metrics_collector, feature_store_monitor, realtime_stream, feature_store

    # セーフモード確認
    if not is_safe_mode():
        raise RuntimeError("セーフモードでない場合はサーバーを起動できません")

    logger.info("🚀 Enhanced Realtime Dashboard 起動中...")
    logger.info("🔒 セーフモード: 自動取引機能は完全に無効化されています")
    logger.info("📊 分析・監視・教育専用システムとして動作します")

    # Feature Store初期化
    feature_store = FeatureStore(cache_size=1000)
    logger.info("✅ Feature Store初期化完了")

    # メトリクス収集器初期化
    metrics_collector = MetricsCollector(collection_interval=1.0)
    await metrics_collector.start_collection()
    logger.info("📈 システムメトリクス収集開始")

    # Feature Store監視器初期化
    feature_store_monitor = FeatureStoreMonitor(update_interval=2.0)
    feature_store_monitor.set_feature_store(feature_store)
    await feature_store_monitor.start_monitoring()
    logger.info("🔍 Feature Store監視開始")

    # リアルタイムストリーム初期化
    realtime_stream = RealtimeStream(broadcast_interval=2.0)
    realtime_stream.set_metrics_collector(metrics_collector)
    realtime_stream.set_feature_store_monitor(feature_store_monitor)
    await realtime_stream.start_streaming()
    logger.info("📡 リアルタイムストリーミング開始")

    logger.info("🎯 Enhanced Dashboard起動完了 - http://localhost:8001")


@app.on_event("shutdown")
async def shutdown_event():
    """サーバー終了時のクリーンアップ"""
    global metrics_collector, feature_store_monitor, realtime_stream

    logger.info("🛑 Enhanced Dashboard停止処理中...")

    if realtime_stream:
        await realtime_stream.stop_streaming()

    if feature_store_monitor:
        await feature_store_monitor.stop_monitoring()

    if metrics_collector:
        await metrics_collector.stop_collection()

    logger.info("✅ Enhanced Dashboard停止完了")


@app.get("/", response_class=HTMLResponse)
async def get_enhanced_dashboard():
    """Enhanced リアルタイムダッシュボードページ"""
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 Day Trade Enhanced Realtime Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333; padding: 20px; min-height: 100vh;
            }

            .header {
                background: rgba(255,255,255,0.95); padding: 20px;
                border-radius: 15px; text-align: center; margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }

            .safe-banner {
                background: linear-gradient(90deg, #28a745, #20c997);
                color: white; padding: 15px; text-align: center;
                border-radius: 10px; margin-bottom: 20px;
                font-weight: bold; box-shadow: 0 4px 15px rgba(40,167,69,0.3);
            }

            .dashboard-grid {
                display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px; max-width: 1400px; margin: 0 auto;
            }

            .card {
                background: rgba(255,255,255,0.95); padding: 25px;
                border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }

            .card h3 {
                color: #495057; margin-bottom: 20px; font-size: 1.4em;
                display: flex; align-items: center; gap: 10px;
            }

            .metric-row {
                display: flex; justify-content: space-between; align-items: center;
                padding: 12px 0; border-bottom: 1px solid #e9ecef;
            }

            .metric-row:last-child { border-bottom: none; }

            .metric-label { font-weight: 500; color: #6c757d; }
            .metric-value { font-weight: bold; font-size: 1.1em; }

            .status-indicator {
                display: inline-block; width: 12px; height: 12px;
                border-radius: 50%; margin-right: 8px;
            }

            .status-excellent { background: #28a745; }
            .status-good { background: #17a2b8; }
            .status-warning { background: #ffc107; }
            .status-critical { background: #dc3545; }

            .progress-bar {
                width: 100%; height: 8px; background: #e9ecef;
                border-radius: 4px; overflow: hidden; margin: 8px 0;
            }

            .progress-fill {
                height: 100%; transition: all 0.3s ease;
                background: linear-gradient(90deg, #28a745, #20c997);
            }

            .log-container {
                background: #1a1a1a; color: #00ff00; padding: 15px;
                border-radius: 8px; height: 300px; overflow-y: auto;
                font-family: 'Courier New', monospace; font-size: 0.9em;
            }

            .controls {
                display: flex; gap: 15px; margin-top: 20px;
                flex-wrap: wrap;
            }

            .btn {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white; border: none; padding: 12px 24px;
                border-radius: 8px; cursor: pointer; font-weight: 500;
                transition: all 0.3s ease;
            }

            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }

            .btn:disabled {
                background: #6c757d; cursor: not-allowed;
                transform: none; box-shadow: none;
            }

            .chart-container {
                width: 100%; height: 200px; margin-top: 15px;
                background: #f8f9fa; border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                color: #6c757d;
            }

            .connection-status {
                position: fixed; top: 20px; right: 20px;
                padding: 10px 15px; border-radius: 25px;
                color: white; font-weight: bold; font-size: 0.9em;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                z-index: 1000;
            }

            .connected { background: linear-gradient(90deg, #28a745, #20c997); }
            .disconnected { background: linear-gradient(90deg, #dc3545, #c82333); }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Enhanced Realtime Dashboard</h1>
            <p>Feature Store監視 × システムメトリクス × リアルタイム分析</p>
        </div>

        <div class="safe-banner">
            🔒 完全セーフモード - 自動取引無効 | 分析・監視・教育専用システム
        </div>

        <div class="connection-status" id="connectionStatus">
            🔌 接続中...
        </div>

        <div class="dashboard-grid">
            <!-- Feature Store 監視カード -->
            <div class="card">
                <h3>🚀 Feature Store パフォーマンス</h3>
                <div class="metric-row">
                    <span class="metric-label">高速化倍率</span>
                    <span class="metric-value" id="speedupRatio">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">キャッシュヒット率</span>
                    <span class="metric-value" id="hitRate">--</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="hitRateProgress" style="width: 0%"></div>
                </div>
                <div class="metric-row">
                    <span class="metric-label">平均応答時間</span>
                    <span class="metric-value" id="responseTime">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">総リクエスト数</span>
                    <span class="metric-value" id="totalRequests">--</span>
                </div>
            </div>

            <!-- システム健全性カード -->
            <div class="card">
                <h3>💊 システム健全性</h3>
                <div class="metric-row">
                    <span class="metric-label">全体ステータス</span>
                    <span class="metric-value" id="systemStatus">
                        <span class="status-indicator" id="statusIndicator"></span>
                        <span id="statusText">--</span>
                    </span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">CPU使用率</span>
                    <span class="metric-value" id="cpuUsage">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">メモリ使用率</span>
                    <span class="metric-value" id="memoryUsage">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ディスク使用率</span>
                    <span class="metric-value" id="diskUsage">--</span>
                </div>
            </div>

            <!-- パフォーマンス統計カード -->
            <div class="card">
                <h3>📈 パフォーマンス統計</h3>
                <div class="metric-row">
                    <span class="metric-label">監視継続時間</span>
                    <span class="metric-value" id="monitoringDuration">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">データポイント数</span>
                    <span class="metric-value" id="dataPoints">--</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">メモリ使用量</span>
                    <span class="metric-value" id="fsMemoryUsage">--</span>
                </div>
                <div class="chart-container">
                    📊 リアルタイムチャート（準備中）
                </div>
            </div>

            <!-- ログ & 制御カード -->
            <div class="card">
                <h3>📝 リアルタイムログ</h3>
                <div class="log-container" id="logContainer">
                    [起動中] Enhanced Dashboard初期化中...<br>
                </div>
                <div class="controls">
                    <button class="btn" onclick="requestHealthReport()">健全性レポート</button>
                    <button class="btn" onclick="requestFeatureStoreReport()">FSレポート</button>
                    <button class="btn" onclick="clearLog()">ログクリア</button>
                    <button class="btn" onclick="downloadReport()">レポートDL</button>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;

            function connectWebSocket() {
                try {
                    ws = new WebSocket('ws://localhost:8001/ws/realtime');

                    ws.onopen = function() {
                        log('[接続] リアルタイムダッシュボード接続完了');
                        updateConnectionStatus(true);
                        reconnectAttempts = 0;
                    };

                    ws.onmessage = function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            handleRealtimeData(data);
                        } catch (e) {
                            console.error('メッセージパース失敗:', e);
                        }
                    };

                    ws.onclose = function() {
                        log('[切断] WebSocket接続終了');
                        updateConnectionStatus(false);

                        if (reconnectAttempts < maxReconnectAttempts) {
                            reconnectAttempts++;
                            log(`[再接続] 試行 ${reconnectAttempts}/${maxReconnectAttempts}`);
                            setTimeout(connectWebSocket, 3000);
                        }
                    };

                    ws.onerror = function(error) {
                        console.error('WebSocketエラー:', error);
                        log('[エラー] WebSocket接続エラー');
                    };

                } catch (e) {
                    console.error('WebSocket接続失敗:', e);
                    updateConnectionStatus(false);
                }
            }

            function handleRealtimeData(data) {
                switch(data.type) {
                    case 'realtime_update':
                        updateDashboardMetrics(data.data);
                        break;
                    case 'alert':
                        log(`[${data.alert_type.toUpperCase()}] ${data.message}`);
                        break;
                    case 'notification':
                        log(`[通知] ${data.title}: ${data.message}`);
                        break;
                    case 'data_response':
                        handleDataResponse(data);
                        break;
                    default:
                        console.log('未処理メッセージ:', data);
                }
            }

            function updateDashboardMetrics(data) {
                // Feature Store メトリクス更新
                if (data.feature_store && data.feature_store.metrics) {
                    const fs = data.feature_store.metrics;
                    updateElement('speedupRatio', `${fs.speedup_ratio}x`);
                    updateElement('hitRate', `${fs.hit_rate}%`);
                    updateElement('responseTime', `${(fs.avg_computation_time * 1000).toFixed(2)}ms`);
                    updateElement('totalRequests', fs.total_requests.toLocaleString());

                    // プログレスバー更新
                    const progressBar = document.getElementById('hitRateProgress');
                    if (progressBar) {
                        progressBar.style.width = `${fs.hit_rate}%`;
                    }
                }

                // システムメトリクス更新
                if (data.system_metrics && data.system_metrics.current) {
                    const sys = data.system_metrics.current;
                    if (sys.cpu) updateElement('cpuUsage', `${sys.cpu.usage_percent.toFixed(1)}%`);
                    if (sys.memory) updateElement('memoryUsage', `${sys.memory.usage_percent.toFixed(1)}%`);
                    if (sys.disk) updateElement('diskUsage', `${sys.disk.usage_percent.toFixed(1)}%`);
                }

                // Feature Store パフォーマンス統計
                if (data.feature_store && data.feature_store.performance_summary) {
                    const perf = data.feature_store.performance_summary;
                    updateElement('monitoringDuration', `${perf.monitoring_period_minutes}分`);
                    updateElement('dataPoints', perf.samples_count.toLocaleString());
                }
            }

            function updateElement(id, value) {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            }

            function updateConnectionStatus(connected) {
                const status = document.getElementById('connectionStatus');
                if (connected) {
                    status.textContent = '🟢 接続中';
                    status.className = 'connection-status connected';
                } else {
                    status.textContent = '🔴 切断';
                    status.className = 'connection-status disconnected';
                }
            }

            function log(message) {
                const container = document.getElementById('logContainer');
                const timestamp = new Date().toLocaleTimeString();
                container.innerHTML += `<div>${timestamp} - ${message}</div>`;
                container.scrollTop = container.scrollHeight;
            }

            function requestHealthReport() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'request_data',
                        data_type: 'system_health'
                    }));
                    log('[リクエスト] システム健全性レポート要求');
                }
            }

            function requestFeatureStoreReport() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'request_data',
                        data_type: 'feature_store_report'
                    }));
                    log('[リクエスト] Feature Storeレポート要求');
                }
            }

            function clearLog() {
                document.getElementById('logContainer').innerHTML = '[ログクリア] 表示をクリアしました<br>';
            }

            function downloadReport() {
                log('[DL] レポートダウンロード機能は実装予定');
            }

            function handleDataResponse(data) {
                if (data.data_type === 'system_health') {
                    const health = data.data;
                    log(`[健全性] スコア: ${health.overall_health}/100`);

                    // ステータス表示更新
                    const indicator = document.getElementById('statusIndicator');
                    const statusText = document.getElementById('statusText');

                    if (health.overall_health >= 80) {
                        indicator.className = 'status-indicator status-excellent';
                        statusText.textContent = '優秀';
                    } else if (health.overall_health >= 60) {
                        indicator.className = 'status-indicator status-good';
                        statusText.textContent = '良好';
                    } else if (health.overall_health >= 40) {
                        indicator.className = 'status-indicator status-warning';
                        statusText.textContent = '注意';
                    } else {
                        indicator.className = 'status-indicator status-critical';
                        statusText.textContent = '危険';
                    }

                } else if (data.data_type === 'feature_store_report') {
                    const report = data.data;
                    log(`[FSレポート] 監視時間: ${report.monitoring_duration_hours}時間`);

                    if (report.key_achievements && report.key_achievements.length > 0) {
                        report.key_achievements.forEach(achievement => {
                            log(`[成果] ${achievement}`);
                        });
                    }
                }
            }

            // 初期化
            window.onload = function() {
                log('[起動] Enhanced Dashboard初期化完了');
                connectWebSocket();

                // 定期的にping送信
                setInterval(() => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({type: 'ping'}));
                    }
                }, 30000);
            };
        </script>
    </body>
    </html>
    """
    )


@app.websocket("/ws/realtime")
async def websocket_realtime_endpoint(websocket: WebSocket):
    """リアルタイムデータ配信WebSocketエンドポイント"""
    global realtime_stream

    if not realtime_stream:
        await websocket.close(code=1011, reason="リアルタイムストリームが利用できません")
        return

    await realtime_stream.connect(websocket)

    try:
        while True:
            # クライアントからのメッセージ処理
            data = await websocket.receive_text()
            message = json.loads(data)
            await realtime_stream.handle_client_message(websocket, message)

    except WebSocketDisconnect:
        realtime_stream.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocketエラー: {e}")
        realtime_stream.disconnect(websocket)


# REST API エンドポイント

@app.get("/api/system/status")
async def get_system_status():
    """システムステータス取得"""
    global metrics_collector, feature_store_monitor, realtime_stream

    status = {
        "safe_mode": is_safe_mode(),
        "server_time": datetime.now().isoformat(),
        "components": {
            "metrics_collector": metrics_collector is not None,
            "feature_store_monitor": feature_store_monitor is not None,
            "realtime_stream": realtime_stream is not None
        }
    }

    if metrics_collector:
        status["system_health"] = metrics_collector.generate_health_report()

    if feature_store_monitor:
        status["feature_store_health"] = feature_store_monitor.get_health_status()

    if realtime_stream:
        status["connection_stats"] = realtime_stream.get_connection_stats()

    return status


@app.get("/api/metrics/current")
async def get_current_metrics():
    """現在のメトリクス取得"""
    global metrics_collector, feature_store_monitor

    result = {"timestamp": datetime.now().isoformat()}

    if metrics_collector:
        result["system_metrics"] = metrics_collector.get_current_metrics()

    if feature_store_monitor:
        result["feature_store_metrics"] = feature_store_monitor.get_current_metrics()

    return result


@app.get("/api/metrics/history")
async def get_metrics_history(minutes: int = 10):
    """メトリクス履歴取得"""
    global metrics_collector, feature_store_monitor

    result = {
        "period_minutes": minutes,
        "timestamp": datetime.now().isoformat()
    }

    if metrics_collector:
        result["system_history"] = metrics_collector.get_metrics_history(minutes)

    if feature_store_monitor:
        result["feature_store_history"] = feature_store_monitor.get_metrics_history(minutes)

    return result


@app.get("/api/reports/comprehensive")
async def get_comprehensive_report():
    """包括的レポート取得"""
    global metrics_collector, feature_store_monitor

    report = {
        "generated_at": datetime.now().isoformat(),
        "report_type": "comprehensive_monitoring_report",
        "safe_mode": is_safe_mode()
    }

    if metrics_collector:
        report["system_health_report"] = metrics_collector.generate_health_report()

    if feature_store_monitor:
        report["feature_store_report"] = feature_store_monitor.generate_report()

    return report


@app.post("/api/feature-store/demo")
async def demo_feature_store():
    """Feature Store デモ実行"""
    global feature_store

    if not feature_store:
        raise HTTPException(status_code=503, detail="Feature Storeが利用できません")

    # デモ用の特徴量生成
    import pandas as pd
    import numpy as np

    # サンプルデータ作成
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'price': np.random.uniform(1000, 2000, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Feature Store経由で特徴量生成
    try:
        from ..ml.feature_pipeline import FeaturePipeline
        pipeline = FeaturePipeline()

        # バッチ処理でデモ実行
        features = pipeline.batch_generate_features(data, batch_size=20)

        return {
            "success": True,
            "message": "Feature Store デモ実行完了",
            "features_generated": len(features) if features else 0,
            "demo_data_size": len(data),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Feature Store デモエラー: {e}")
        raise HTTPException(status_code=500, detail=f"デモ実行エラー: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🚀 Day Trade Enhanced Realtime Dashboard")
    print("=" * 80)
    print("🔒 完全セーフモード - 自動取引機能無効化")
    print("📊 Feature Store監視 + システムメトリクス")
    print("📡 リアルタイムストリーミング対応")
    print("🌐 ダッシュボード: http://localhost:8001")
    print("=" * 80)

    uvicorn.run(app, host="127.0.0.1", port=8001)
