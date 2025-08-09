#!/usr/bin/env python3
"""
強化ダッシュボード UI
Phase E: ユーザーエクスペリエンス強化

リアルタイム更新・レスポンシブデザイン・ダークモード対応
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..core.optimization_strategy import OptimizationConfig, OptimizationLevel
from ..core.enhanced_error_handler import EnhancedErrorHandler, handle_error_gracefully
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class DashboardThemeManager:
    """ダッシュボードテーマ管理"""
    
    def __init__(self):
        self.themes = {
            'light': {
                'name': 'ライト',
                'primary_color': '#2c3e50',
                'secondary_color': '#3498db',
                'background_color': '#ffffff',
                'text_color': '#2c3e50',
                'card_background': '#f8f9fa',
                'border_color': '#dee2e6',
                'success_color': '#28a745',
                'warning_color': '#ffc107',
                'danger_color': '#dc3545',
                'info_color': '#17a2b8'
            },
            'dark': {
                'name': 'ダーク',
                'primary_color': '#ffffff',
                'secondary_color': '#4dabf7',
                'background_color': '#1a1a1a',
                'text_color': '#ffffff',
                'card_background': '#2d3748',
                'border_color': '#4a5568',
                'success_color': '#48bb78',
                'warning_color': '#ed8936',
                'danger_color': '#f56565',
                'info_color': '#4299e1'
            },
            'high_contrast': {
                'name': 'ハイコントラスト',
                'primary_color': '#000000',
                'secondary_color': '#0066cc',
                'background_color': '#ffffff',
                'text_color': '#000000',
                'card_background': '#f0f0f0',
                'border_color': '#000000',
                'success_color': '#006600',
                'warning_color': '#cc6600',
                'danger_color': '#cc0000',
                'info_color': '#0066cc'
            }
        }
        
        self.current_theme = 'light'
    
    def get_theme(self, theme_name: str = None) -> Dict[str, Any]:
        """テーマ取得"""
        theme_name = theme_name or self.current_theme
        return self.themes.get(theme_name, self.themes['light'])
    
    def get_css_variables(self, theme_name: str = None) -> str:
        """CSS変数生成"""
        theme = self.get_theme(theme_name)
        css_vars = []
        
        for key, value in theme.items():
            if key != 'name':
                css_var_name = key.replace('_', '-')
                css_vars.append(f"--{css_var_name}: {value};")
        
        return "\n".join(css_vars)


class RealTimeDataManager:
    """リアルタイムデータ管理"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.data_cache: Dict[str, Any] = {}
        self.last_update = datetime.now()
        
    async def connect(self, websocket: WebSocket):
        """WebSocket接続"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket接続確立: {len(self.active_connections)}個の接続")
    
    def disconnect(self, websocket: WebSocket):
        """WebSocket切断"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket接続切断: {len(self.active_connections)}個の接続")
    
    async def broadcast_data(self, data: Dict[str, Any]):
        """データブロードキャスト"""
        if not self.active_connections:
            return
        
        message = json.dumps({
            'type': 'data_update',
            'timestamp': datetime.now().isoformat(),
            'data': data
        }, ensure_ascii=False)
        
        # 接続中の全クライアントにデータ送信
        disconnected_connections = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"WebSocket送信エラー: {e}")
                disconnected_connections.append(connection)
        
        # 切断されたコネクションを削除
        for connection in disconnected_connections:
            self.disconnect(connection)
    
    async def update_cache_and_broadcast(self, key: str, data: Any):
        """キャッシュ更新とブロードキャスト"""
        self.data_cache[key] = data
        self.last_update = datetime.now()
        
        await self.broadcast_data({
            'key': key,
            'data': data,
            'update_time': self.last_update.isoformat()
        })


class ResponsiveLayoutManager:
    """レスポンシブレイアウト管理"""
    
    def __init__(self):
        self.breakpoints = {
            'xs': 0,      # モバイル
            'sm': 576,    # 小型タブレット
            'md': 768,    # タブレット
            'lg': 992,    # デスクトップ
            'xl': 1200,   # 大型デスクトップ
            'xxl': 1400   # 超大型ディスプレイ
        }
        
        self.layout_configs = {
            'mobile': {
                'columns': 1,
                'chart_height': '300px',
                'sidebar_collapsed': True,
                'show_detailed_metrics': False
            },
            'tablet': {
                'columns': 2,
                'chart_height': '400px',
                'sidebar_collapsed': False,
                'show_detailed_metrics': True
            },
            'desktop': {
                'columns': 3,
                'chart_height': '500px',
                'sidebar_collapsed': False,
                'show_detailed_metrics': True
            }
        }
    
    def get_layout_for_width(self, width: int) -> Dict[str, Any]:
        """画面幅に応じたレイアウト取得"""
        if width < self.breakpoints['md']:
            return self.layout_configs['mobile']
        elif width < self.breakpoints['lg']:
            return self.layout_configs['tablet']
        else:
            return self.layout_configs['desktop']


class EnhancedDashboardServer:
    """強化ダッシュボードサーバー"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI が必要です: pip install fastapi uvicorn")
        
        self.config = config or OptimizationConfig()
        self.app = FastAPI(title="Day Trade Enhanced Dashboard", version="2.0")
        self.error_handler = EnhancedErrorHandler(language='ja')
        self.theme_manager = DashboardThemeManager()
        self.realtime_manager = RealTimeDataManager()
        self.layout_manager = ResponsiveLayoutManager()
        
        # 静的ファイル・テンプレートの設定
        self.setup_static_files()
        self.setup_routes()
        
        logger.info("強化ダッシュボードサーバー初期化完了")
    
    def setup_static_files(self):
        """静的ファイル設定"""
        # 静的ファイルディレクトリ作成
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        
        templates_dir = Path(__file__).parent / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # 静的ファイルマウント
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.templates = Jinja2Templates(directory=str(templates_dir))
        
        # デフォルトテンプレート作成
        self.create_default_templates()
    
    def create_default_templates(self):
        """デフォルトテンプレート作成"""
        templates_dir = Path(__file__).parent / "templates"
        
        # メインダッシュボードテンプレート
        main_template = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade Enhanced Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            {{ theme_css }}
        }
        
        body {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background-color: var(--primary-color) !important;
        }
        
        .card {
            background-color: var(--card-background);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .sidebar {
            background-color: var(--card-background);
            min-height: calc(100vh - 76px);
            border-right: 1px solid var(--border-color);
        }
        
        .chart-container {
            position: relative;
            height: {{ chart_height }};
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        
        .status-online { background-color: var(--success-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-error { background-color: var(--danger-color); }
        
        .loading-spinner {
            border: 4px solid var(--border-color);
            border-top: 4px solid var(--secondary-color);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                transform: translateX(-100%);
                transition: transform 0.3s ease;
            }
            
            .sidebar.show {
                transform: translateX(0);
            }
        }
    </style>
</head>
<body>
    <!-- ナビゲーションバー -->
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">
                <i class="fas fa-chart-line me-2"></i>
                Day Trade Dashboard
            </a>
            
            <div class="navbar-nav ms-auto">
                <!-- テーマ切り替え -->
                <div class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle" href="#" id="themeDropdown" role="button" data-bs-toggle="dropdown">
                        <i class="fas fa-palette me-1"></i>テーマ
                    </a>
                    <ul class="dropdown-menu">
                        <li><a class="dropdown-item" href="#" data-theme="light">ライト</a></li>
                        <li><a class="dropdown-item" href="#" data-theme="dark">ダーク</a></li>
                        <li><a class="dropdown-item" href="#" data-theme="high_contrast">ハイコントラスト</a></li>
                    </ul>
                </div>
                
                <!-- 接続状態 -->
                <div class="nav-item">
                    <span class="navbar-text">
                        <span class="status-indicator status-online" id="connectionStatus"></span>
                        <span id="connectionText">接続中</span>
                    </span>
                </div>
            </div>
        </div>
    </nav>
    
    <div class="container-fluid">
        <div class="row">
            <!-- サイドバー -->
            <div class="col-md-3 col-lg-2 sidebar p-3" id="sidebar">
                <h5><i class="fas fa-tachometer-alt me-2"></i>システム状態</h5>
                
                <!-- システムメトリクス -->
                <div class="card mb-3">
                    <div class="card-body p-3">
                        <h6 class="card-title">パフォーマンス</h6>
                        <div class="mb-2">
                            <small>処理速度</small>
                            <div class="progress">
                                <div class="progress-bar" id="performanceBar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="mb-2">
                            <small>メモリ使用量</small>
                            <div class="progress">
                                <div class="progress-bar bg-info" id="memoryBar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="mb-2">
                            <small>キャッシュ効率</small>
                            <div class="progress">
                                <div class="progress-bar bg-success" id="cacheBar" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 最新エラー -->
                <div class="card mb-3">
                    <div class="card-body p-3">
                        <h6 class="card-title">システム状況</h6>
                        <div id="errorSummary">
                            <div class="loading-spinner mx-auto"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- メインコンテンツ -->
            <div class="col-md-9 col-lg-10 p-4">
                <!-- アラート表示エリア -->
                <div id="alertContainer"></div>
                
                <!-- メトリクスカード -->
                <div class="row mb-4">
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <i class="fas fa-chart-line fa-2x text-success mb-2"></i>
                                <h4 class="card-title" id="totalAnalyses">-</h4>
                                <p class="card-text">総分析数</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <i class="fas fa-clock fa-2x text-info mb-2"></i>
                                <h4 class="card-title" id="avgProcessingTime">-</h4>
                                <p class="card-text">平均処理時間</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <i class="fas fa-percentage fa-2x text-warning mb-2"></i>
                                <h4 class="card-title" id="successRate">-</h4>
                                <p class="card-text">成功率</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 mb-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <i class="fas fa-memory fa-2x text-danger mb-2"></i>
                                <h4 class="card-title" id="memoryUsage">-</h4>
                                <p class="card-text">メモリ使用量</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- チャートエリア -->
                <div class="row">
                    <div class="col-lg-8 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">
                                    <i class="fas fa-chart-area me-2"></i>
                                    リアルタイム分析結果
                                </h5>
                            </div>
                            <div class="card-body">
                                <div class="chart-container" id="mainChart">
                                    <div class="d-flex justify-content-center align-items-center h-100">
                                        <div class="loading-spinner"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-lg-4 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title mb-0">
                                    <i class="fas fa-list me-2"></i>
                                    最新アクティビティ
                                </h5>
                            </div>
                            <div class="card-body">
                                <div id="activityLog" style="height: {{ chart_height }}; overflow-y: auto;">
                                    <!-- アクティビティログがここに表示 -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // WebSocket接続管理
        class DashboardWebSocket {
            constructor() {
                this.ws = null;
                this.reconnectInterval = 5000;
                this.maxReconnectAttempts = 10;
                this.reconnectAttempts = 0;
                this.connect();
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('WebSocket接続確立');
                    this.reconnectAttempts = 0;
                    this.updateConnectionStatus(true);
                };
                
                this.ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.ws.onclose = () => {
                    console.log('WebSocket接続切断');
                    this.updateConnectionStatus(false);
                    this.scheduleReconnect();
                };
                
                this.ws.onerror = (error) => {
                    console.error('WebSocket エラー:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            scheduleReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        console.log(`再接続試行 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                        this.connect();
                    }, this.reconnectInterval);
                }
            }
            
            updateConnectionStatus(connected) {
                const statusIndicator = document.getElementById('connectionStatus');
                const statusText = document.getElementById('connectionText');
                
                if (connected) {
                    statusIndicator.className = 'status-indicator status-online';
                    statusText.textContent = '接続中';
                } else {
                    statusIndicator.className = 'status-indicator status-error';
                    statusText.textContent = '切断';
                }
            }
            
            handleMessage(data) {
                if (data.type === 'data_update') {
                    this.updateDashboard(data.data);
                } else if (data.type === 'error') {
                    this.showError(data.message);
                }
            }
            
            updateDashboard(data) {
                // メトリクス更新
                if (data.metrics) {
                    document.getElementById('totalAnalyses').textContent = data.metrics.total_analyses || '-';
                    document.getElementById('avgProcessingTime').textContent = 
                        data.metrics.avg_processing_time ? `${data.metrics.avg_processing_time}ms` : '-';
                    document.getElementById('successRate').textContent = 
                        data.metrics.success_rate ? `${(data.metrics.success_rate * 100).toFixed(1)}%` : '-';
                    document.getElementById('memoryUsage').textContent = 
                        data.metrics.memory_usage ? `${data.metrics.memory_usage}MB` : '-';
                }
                
                // プログレスバー更新
                if (data.performance) {
                    document.getElementById('performanceBar').style.width = 
                        `${(data.performance.speed_score * 100).toFixed(0)}%`;
                    document.getElementById('memoryBar').style.width = 
                        `${(data.performance.memory_efficiency * 100).toFixed(0)}%`;
                    document.getElementById('cacheBar').style.width = 
                        `${(data.performance.cache_hit_rate * 100).toFixed(0)}%`;
                }
                
                // アクティビティログ追加
                if (data.activity) {
                    this.addActivityLog(data.activity);
                }
            }
            
            addActivityLog(activity) {
                const logContainer = document.getElementById('activityLog');
                const logEntry = document.createElement('div');
                logEntry.className = 'mb-2 p-2 border-start border-3 border-primary';
                logEntry.innerHTML = `
                    <small class="text-muted">${new Date(activity.timestamp).toLocaleTimeString()}</small><br>
                    <span>${activity.message}</span>
                `;
                
                logContainer.insertBefore(logEntry, logContainer.firstChild);
                
                // 古いログエントリを削除（最新20件のみ保持）
                while (logContainer.children.length > 20) {
                    logContainer.removeChild(logContainer.lastChild);
                }
            }
            
            showError(message) {
                const alertContainer = document.getElementById('alertContainer');
                const alert = document.createElement('div');
                alert.className = 'alert alert-warning alert-dismissible fade show';
                alert.innerHTML = `
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;
                alertContainer.appendChild(alert);
                
                // 5秒後に自動削除
                setTimeout(() => {
                    if (alert.parentNode) {
                        alert.parentNode.removeChild(alert);
                    }
                }, 5000);
            }
        }
        
        // テーマ切り替え
        document.addEventListener('DOMContentLoaded', function() {
            // WebSocket接続初期化
            const dashboard = new DashboardWebSocket();
            
            // テーマ切り替えイベント
            document.querySelectorAll('[data-theme]').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    const theme = this.getAttribute('data-theme');
                    
                    // テーマ変更をサーバーに送信
                    fetch('/api/theme', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({theme: theme})
                    }).then(() => {
                        location.reload(); // ページリロード
                    });
                });
            });
        });
    </script>
</body>
</html>
        """
        
        with open(templates_dir / "dashboard.html", "w", encoding="utf-8") as f:
            f.write(main_template)
    
    def setup_routes(self):
        """ルート設定"""
        
        @self.app.get("/", response_class=HTMLResponse)
        @handle_error_gracefully("dashboard_main", "dashboard")
        async def dashboard_main(request: Request):
            """メインダッシュボード"""
            theme = self.theme_manager.get_theme()
            layout = self.layout_manager.get_layout_for_width(1200)  # デフォルト幅
            
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "theme_css": self.theme_manager.get_css_variables(),
                "chart_height": layout['chart_height']
            })
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocketエンドポイント"""
            await self.realtime_manager.connect(websocket)
            try:
                while True:
                    # クライアントからのメッセージ待機
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # メッセージに応じた処理
                    if message.get('type') == 'request_update':
                        await self.send_dashboard_data(websocket)
                    
            except Exception as e:
                logger.error(f"WebSocketエラー: {e}")
            finally:
                self.realtime_manager.disconnect(websocket)
        
        @self.app.post("/api/theme")
        @handle_error_gracefully("theme_change", "dashboard")
        async def change_theme(request: Request):
            """テーマ変更"""
            body = await request.json()
            theme_name = body.get('theme', 'light')
            
            if theme_name in self.theme_manager.themes:
                self.theme_manager.current_theme = theme_name
                return JSONResponse({"status": "success", "theme": theme_name})
            else:
                return JSONResponse({"status": "error", "message": "無効なテーマ"}, status_code=400)
        
        @self.app.get("/api/metrics")
        @handle_error_gracefully("get_metrics", "dashboard")
        async def get_metrics():
            """メトリクス取得"""
            # システムメトリクス取得
            metrics = await self.collect_system_metrics()
            return JSONResponse(metrics)
        
        @self.app.get("/api/health")
        async def health_check():
            """ヘルスチェック"""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",
                "active_connections": len(self.realtime_manager.active_connections)
            })
    
    async def send_dashboard_data(self, websocket: WebSocket):
        """ダッシュボードデータ送信"""
        try:
            metrics = await self.collect_system_metrics()
            
            await websocket.send_text(json.dumps({
                "type": "data_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "metrics": metrics,
                    "performance": await self.get_performance_data(),
                    "activity": {
                        "timestamp": datetime.now().isoformat(),
                        "message": f"システムメトリクス更新: {metrics.get('total_analyses', 0)}件の分析完了"
                    }
                }
            }, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"ダッシュボードデータ送信エラー: {e}")
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """システムメトリクス収集"""
        # 統合最適化システムからメトリクス取得
        try:
            from ..core.optimization_strategy import OptimizationStrategyFactory
            
            # 登録済みコンポーネントのパフォーマンス取得
            components = OptimizationStrategyFactory.get_registered_components()
            total_analyses = 0
            avg_processing_time = 0
            success_rate = 1.0
            
            for component_name in components.keys():
                try:
                    strategy = OptimizationStrategyFactory.get_strategy(component_name, self.config)
                    if hasattr(strategy, 'get_performance_metrics'):
                        metrics = strategy.get_performance_metrics()
                        total_analyses += metrics.get('execution_count', 0)
                        avg_processing_time += metrics.get('average_time', 0)
                        success_rate = min(success_rate, metrics.get('success_rate', 1.0))
                except Exception:
                    continue
            
            # メモリ使用量取得
            import psutil
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.used / 1024 / 1024  # MB
            
            return {
                "total_analyses": total_analyses,
                "avg_processing_time": avg_processing_time / max(len(components), 1),
                "success_rate": success_rate,
                "memory_usage": memory_usage,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"メトリクス収集エラー: {e}")
            return {
                "total_analyses": 0,
                "avg_processing_time": 0,
                "success_rate": 0,
                "memory_usage": 0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_performance_data(self) -> Dict[str, float]:
        """パフォーマンスデータ取得"""
        try:
            import psutil
            
            # CPU・メモリ使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "speed_score": max(0, (100 - cpu_percent) / 100),  # CPU使用率から速度スコア算出
                "memory_efficiency": (memory.available / memory.total),  # 利用可能メモリ比率
                "cache_hit_rate": 0.85  # 仮の値（実際のキャッシュ統計から取得）
            }
            
        except Exception as e:
            logger.error(f"パフォーマンスデータ取得エラー: {e}")
            return {
                "speed_score": 0.0,
                "memory_efficiency": 0.0,
                "cache_hit_rate": 0.0
            }
    
    async def start_background_tasks(self):
        """バックグラウンドタスク開始"""
        async def periodic_update():
            """定期的データ更新"""
            while True:
                try:
                    # 5秒間隔でデータ更新
                    await asyncio.sleep(5)
                    
                    if self.realtime_manager.active_connections:
                        metrics = await self.collect_system_metrics()
                        await self.realtime_manager.broadcast_data({
                            "metrics": metrics,
                            "performance": await self.get_performance_data(),
                            "activity": {
                                "timestamp": datetime.now().isoformat(),
                                "message": f"定期更新: {len(self.realtime_manager.active_connections)}クライアント接続中"
                            }
                        })
                        
                except Exception as e:
                    logger.error(f"定期更新エラー: {e}")
        
        # バックグラウンドタスク開始
        asyncio.create_task(periodic_update())
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """サーバー起動"""
        # バックグラウンドタスク開始
        asyncio.create_task(self.start_background_tasks())
        
        logger.info(f"強化ダッシュボード起動: http://{host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if debug else "warning",
            access_log=debug
        )