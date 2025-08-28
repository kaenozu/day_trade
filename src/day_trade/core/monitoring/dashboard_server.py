"""
監視ダッシュボードサーバー

リアルタイム監視ダッシュボードのWebサーバー実装。
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from aiohttp import web, WSMsgType
from aiohttp.web import Application, Request, WebSocketResponse
import aiohttp_cors

from .unified_monitoring_system import UnifiedMonitoringSystem, get_global_monitoring_system
from .security_middleware import SecurityMiddleware, create_security_middleware
from ..config.configuration_manager import ConfigurationManager


class DashboardWebSocketManager:
    """WebSocket接続管理器"""
    
    def __init__(self, max_connections: int = 100):
        self.connections: List[WebSocketResponse] = []
        self.max_connections = max_connections
        
    async def add_connection(self, ws: WebSocketResponse):
        """WebSocket接続追加"""
        if len(self.connections) >= self.max_connections:
            logging.warning(f"WebSocket接続数上限達成: {self.max_connections}")
            await ws.close(code=1013, message="Too many connections")
            return False
            
        self.connections.append(ws)
        logging.info(f"WebSocket接続追加: {len(self.connections)}件")
        return True
        
    async def remove_connection(self, ws: WebSocketResponse):
        """WebSocket接続削除"""
        if ws in self.connections:
            self.connections.remove(ws)
            logging.info(f"WebSocket接続削除: {len(self.connections)}件")
            
    async def broadcast(self, data: Dict[str, Any]):
        """全接続にブロードキャスト"""
        if not self.connections:
            return
            
        message = json.dumps(data, ensure_ascii=False, default=str)
        disconnected = []
        
        for ws in self.connections:
            try:
                await ws.send_str(message)
            except Exception as e:
                logging.warning(f"WebSocketブロードキャストエラー: {e}")
                disconnected.append(ws)
                
        # 切断された接続を削除
        for ws in disconnected:
            await self.remove_connection(ws)


class DashboardServer:
    """ダッシュボードサーバー"""
    
    def __init__(self, monitoring_system: Optional[UnifiedMonitoringSystem] = None,
                 config_manager: Optional[ConfigurationManager] = None,
                 security_config: Optional[Dict[str, Any]] = None):
        self.monitoring_system = monitoring_system or get_global_monitoring_system()
        self.config_manager = config_manager
        self.security_config = security_config or {}
        self.ws_manager = DashboardWebSocketManager(
            max_connections=self.security_config.get('max_connections', 100)
        )
        self.app = None
        self.runner = None
        self.site = None
        self.broadcast_task = None
        self.security_middleware: Optional[SecurityMiddleware] = None
        
    async def create_app(self) -> Application:
        """アプリケーション作成"""
        app = web.Application()
        
        # セキュリティミドルウェア設定
        self.security_middleware = create_security_middleware(app, self.security_config)
        
        # CORS設定（セキュアに制限）
        cors = aiohttp_cors.setup(app, defaults={
            "http://localhost:*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers=["Content-Type", "Authorization"],
                allow_headers=["Content-Type", "Authorization"],
                allow_methods=["GET", "POST", "OPTIONS"]
            ),
            "https://localhost:*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers=["Content-Type", "Authorization"],
                allow_headers=["Content-Type", "Authorization"],
                allow_methods=["GET", "POST", "OPTIONS"]
            )
        })
        
        # ルート設定
        app.router.add_get('/', self.index_handler)
        app.router.add_get('/api/status', self.status_handler)
        app.router.add_get('/api/metrics', self.metrics_handler)
        app.router.add_get('/api/alerts', self.alerts_handler)
        app.router.add_get('/api/health', self.health_handler)
        app.router.add_get('/ws', self.websocket_handler)
        
        # CORS適用
        for route in list(app.router.routes()):
            cors.add(route)
            
        return app
        
    async def index_handler(self, request: Request) -> web.Response:
        """インデックスハンドラー"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Day Trade 監視ダッシュボード</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 30px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .alert-card { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 15px; margin: 10px 0; }
                .alert-critical { background: #f8d7da; border-color: #f5c6cb; }
                .alert-warning { background: #fff3cd; border-color: #ffeaa7; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-online { background-color: #28a745; }
                .status-offline { background-color: #dc3545; }
                .refresh-time { color: #666; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Day Trade システム監視ダッシュボード</h1>
                    <div class="refresh-time">最終更新: <span id="lastUpdate">-</span></div>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">システム状態</div>
                        <div id="systemStatus">
                            <span class="status-indicator status-online"></span>オンライン
                        </div>
                        <div>稼働時間: <span id="uptime">-</span></div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">CPU使用率</div>
                        <div class="metric-value" id="cpuUsage">-</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">メモリ使用率</div>
                        <div class="metric-value" id="memoryUsage">-</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">ディスク使用率</div>
                        <div class="metric-value" id="diskUsage">-</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">収集済みメトリクス</div>
                        <div class="metric-value" id="metricsCollected">-</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-title">アクティブアラート</div>
                        <div class="metric-value" id="activeAlerts">-</div>
                    </div>
                </div>
                
                <div id="alertsContainer">
                    <h2>アラート</h2>
                    <div id="alertsList"></div>
                </div>
            </div>
            
            <script>
                let ws = null;
                
                function connectWebSocket() {
                    ws = new WebSocket(`ws://${window.location.host}/ws`);
                    
                    ws.onopen = function() {
                        console.log('WebSocket接続開始');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateDashboard(data);
                    };
                    
                    ws.onclose = function() {
                        console.log('WebSocket接続終了');
                        setTimeout(connectWebSocket, 5000);
                    };
                    
                    ws.onerror = function(error) {
                        console.error('WebSocketエラー:', error);
                    };
                }
                
                function updateDashboard(data) {
                    document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
                    
                    if (data.status) {
                        document.getElementById('uptime').textContent = data.status.uptime || '-';
                        document.getElementById('metricsCollected').textContent = data.status.metrics_collected || '0';
                        document.getElementById('activeAlerts').textContent = data.status.active_alerts || '0';
                    }
                    
                    if (data.system_overview) {
                        document.getElementById('cpuUsage').textContent = 
                            data.system_overview.cpu_usage ? data.system_overview.cpu_usage.toFixed(1) + '%' : '-';
                        document.getElementById('memoryUsage').textContent = 
                            data.system_overview.memory_usage ? data.system_overview.memory_usage.toFixed(1) + '%' : '-';
                        document.getElementById('diskUsage').textContent = 
                            data.system_overview.disk_usage ? data.system_overview.disk_usage.toFixed(1) + '%' : '-';
                    }
                    
                    if (data.alerts) {
                        updateAlerts(data.alerts);
                    }
                }
                
                function updateAlerts(alerts) {
                    const alertsList = document.getElementById('alertsList');
                    
                    if (alerts.length === 0) {
                        alertsList.innerHTML = '<p>アクティブなアラートはありません。</p>';
                        return;
                    }
                    
                    alertsList.innerHTML = alerts.map(alert => `
                        <div class="alert-card alert-${alert.level}">
                            <strong>${alert.name}</strong> (${alert.level})
                            <br>${alert.message}
                            <br><small>発生時刻: ${new Date(alert.created_at).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
                
                // 初期化
                connectWebSocket();
                
                // 定期的にデータ取得
                setInterval(async function() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        console.error('データ取得エラー:', error);
                    }
                }, 10000);
            </script>
        </body>
        </html>
        """
        return web.Response(text=html_content, content_type='text/html')
        
    async def status_handler(self, request: Request) -> web.Response:
        """ステータスハンドラー"""
        # シンプルな認証チェック（実際の実装ではより強固な認証が必要）
        auth_header = request.headers.get('Authorization')
        if not auth_header and self.config_manager:
            # 認証が有効な場合のみチェック
            return web.json_response({"error": "Authentication required"}, status=401)
            
        try:
            status = self.monitoring_system.get_monitoring_status()
            alerts = self.monitoring_system.alert_manager.get_active_alerts()
            
            response_data = {
                "status": status,
                "system_overview": status.get("system_overview", {}),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "name": alert.name,
                        "level": alert.level.value,
                        "message": alert.message,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in alerts
                ]
            }
            
            return web.json_response(response_data)
        except Exception as e:
            logging.error(f"ステータス取得エラー: {e}")
            return web.json_response({"error": str(e)}, status=500)
            
    async def metrics_handler(self, request: Request) -> web.Response:
        """メトリクスハンドラー"""
        try:
            format_param = request.query.get('format', 'json')
            metrics_data = self.monitoring_system.export_metrics(format_param)
            
            if format_param == 'json':
                return web.Response(text=metrics_data, content_type='application/json')
            else:
                return web.Response(text=metrics_data, content_type='text/plain')
        except Exception as e:
            logging.error(f"メトリクス取得エラー: {e}")
            return web.json_response({"error": str(e)}, status=500)
            
    async def alerts_handler(self, request: Request) -> web.Response:
        """アラートハンドラー"""
        try:
            alerts = self.monitoring_system.alert_manager.get_active_alerts()
            alerts_data = [
                {
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "source": alert.source,
                    "created_at": alert.created_at.isoformat(),
                    "metric_name": alert.metric_name,
                    "threshold_value": alert.threshold_value,
                    "current_value": alert.current_value
                }
                for alert in alerts
            ]
            
            return web.json_response(alerts_data)
        except Exception as e:
            logging.error(f"アラート取得エラー: {e}")
            return web.json_response({"error": str(e)}, status=500)
            
    async def health_handler(self, request: Request) -> web.Response:
        """ヘルスハンドラー"""
        try:
            health_checks = self.monitoring_system.health_check_manager.health_checks
            health_data = [
                {
                    "name": hc.name,
                    "endpoint": hc.endpoint,
                    "status": hc.status,
                    "response_time": hc.response_time,
                    "last_check": hc.last_check.isoformat(),
                    "error_message": hc.error_message
                }
                for hc in health_checks.values()
            ]
            
            return web.json_response(health_data)
        except Exception as e:
            logging.error(f"ヘルスチェック取得エラー: {e}")
            return web.json_response({"error": str(e)}, status=500)
            
    async def websocket_handler(self, request: Request) -> WebSocketResponse:
        """WebSocketハンドラー"""
        ws = WebSocketResponse()
        await ws.prepare(request)
        
        connection_added = await self.ws_manager.add_connection(ws)
        if not connection_added:
            return ws
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        # クライアントからのメッセージ処理（将来の拡張用）
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({"error": "Invalid JSON"}))
                elif msg.type == WSMsgType.ERROR:
                    logging.error(f'WebSocketエラー: {ws.exception()}')
        except Exception as e:
            logging.error(f'WebSocket処理エラー: {e}')
        finally:
            await self.ws_manager.remove_connection(ws)
            
        return ws
        
    async def _broadcast_loop(self):
        """ブロードキャストループ"""
        while True:
            try:
                # 監視データ取得
                status = self.monitoring_system.get_monitoring_status()
                alerts = self.monitoring_system.alert_manager.get_active_alerts()
                
                data = {
                    "type": "update",
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "system_overview": status.get("system_overview", {}),
                    "alerts": [
                        {
                            "alert_id": alert.alert_id,
                            "name": alert.name,
                            "level": alert.level.value,
                            "message": alert.message,
                            "created_at": alert.created_at.isoformat()
                        }
                        for alert in alerts
                    ]
                }
                
                await self.ws_manager.broadcast(data)
                
            except Exception as e:
                logging.error(f"ブロードキャストエラー: {e}")
                
            await asyncio.sleep(5)  # 5秒間隔
            
    async def start(self, host: str = 'localhost', port: int = 8080):
        """サーバー開始"""
        self.app = await self.create_app()
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        
        # ブロードキャストタスク開始
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logging.info(f"ダッシュボードサーバーを開始しました: http://{host}:{port}")
        
    async def stop(self):
        """サーバー停止"""
        if self.broadcast_task:
            self.broadcast_task.cancel()
            
        if self.site:
            await self.site.stop()
            
        if self.runner:
            await self.runner.cleanup()
            
        logging.info("ダッシュボードサーバーを停止しました")


# サーバーインスタンス管理
_dashboard_server: Optional[DashboardServer] = None


async def start_dashboard_server(host: str = 'localhost', port: int = 8080,
                                monitoring_system: Optional[UnifiedMonitoringSystem] = None,
                                security_config: Optional[Dict[str, Any]] = None) -> DashboardServer:
    """ダッシュボードサーバー開始"""
    global _dashboard_server
    
    if _dashboard_server is None:
        _dashboard_server = DashboardServer(
            monitoring_system=monitoring_system,
            security_config=security_config
        )
        
    await _dashboard_server.start(host, port)
    return _dashboard_server


async def stop_dashboard_server():
    """ダッシュボードサーバー停止"""
    global _dashboard_server
    
    if _dashboard_server:
        await _dashboard_server.stop()
        _dashboard_server = None