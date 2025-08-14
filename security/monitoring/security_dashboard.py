#!/usr/bin/env python3
"""
Security Dashboard - Day Trade ML System
リアルタイムセキュリティ監視ダッシュボード
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum

import psutil
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """セキュリティイベントタイプ"""
    INTRUSION_ATTEMPT = "intrusion_attempt"
    UNUSUAL_TRAFFIC = "unusual_traffic"
    FAILED_LOGIN = "failed_login"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALWARE_DETECTION = "malware_detection"
    CONFIGURATION_CHANGE = "configuration_change"
    NETWORK_ANOMALY = "network_anomaly"


class SeverityLevel(Enum):
    """重要度レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """セキュリティイベント"""
    id: str
    timestamp: datetime
    event_type: SecurityEventType
    severity: SeverityLevel
    source_ip: str
    target: str
    description: str
    details: Dict[str, Any]
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SecurityMetrics:
    """セキュリティメトリクス"""
    timestamp: datetime
    total_events: int
    critical_events: int
    high_events: int
    medium_events: int
    low_events: int
    failed_logins: int
    intrusion_attempts: int
    network_anomalies: int
    system_health: float  # 0-100%
    threat_level: str     # "low", "medium", "high", "critical"


class SecurityEventAnalyzer:
    """セキュリティイベント分析エンジン"""

    def __init__(self):
        self.event_history: List[SecurityEvent] = []
        self.redis_client = self._setup_redis()

    def _setup_redis(self) -> Optional[redis.Redis]:
        """Redis接続設定"""
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                db=2,  # セキュリティ専用DB
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    def add_event(self, event: SecurityEvent) -> None:
        """セキュリティイベント追加"""
        self.event_history.append(event)

        # Redis に保存
        if self.redis_client:
            try:
                event_data = asdict(event)
                event_data['timestamp'] = event.timestamp.isoformat()
                event_data['event_type'] = event.event_type.value
                event_data['severity'] = event.severity.value

                self.redis_client.lpush(
                    'security_events',
                    json.dumps(event_data)
                )

                # 最大1000件まで保持
                self.redis_client.ltrim('security_events', 0, 999)

            except Exception as e:
                logger.error(f"Failed to store event in Redis: {e}")

        # 即座の脅威評価
        self._evaluate_immediate_threat(event)

    def _evaluate_immediate_threat(self, event: SecurityEvent) -> None:
        """即座の脅威評価"""
        if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            # 高優先度アラート
            asyncio.create_task(self._send_immediate_alert(event))

            # 自動対応の実行
            if event.event_type == SecurityEventType.INTRUSION_ATTEMPT:
                asyncio.create_task(self._block_source_ip(event.source_ip))

    async def _send_immediate_alert(self, event: SecurityEvent) -> None:
        """即座のアラート送信"""
        alert_message = {
            "type": "security_alert",
            "severity": event.severity.value,
            "event_type": event.event_type.value,
            "source_ip": event.source_ip,
            "description": event.description,
            "timestamp": event.timestamp.isoformat(),
            "action_required": True
        }

        # Slack通知、メール通知等
        logger.critical(f"SECURITY ALERT: {alert_message}")

    async def _block_source_ip(self, source_ip: str) -> None:
        """送信元IPのブロック"""
        try:
            # ファイアウォールルールの追加 (例)
            import subprocess
            subprocess.run([
                'iptables', '-A', 'INPUT', '-s', source_ip, '-j', 'DROP'
            ], check=True)

            logger.info(f"Blocked IP address: {source_ip}")

        except Exception as e:
            logger.error(f"Failed to block IP {source_ip}: {e}")

    def get_recent_events(self, hours: int = 24) -> List[SecurityEvent]:
        """最近のセキュリティイベント取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.event_history
            if event.timestamp >= cutoff_time
        ]

    def get_metrics(self) -> SecurityMetrics:
        """セキュリティメトリクス計算"""
        recent_events = self.get_recent_events(24)

        # イベント数集計
        total_events = len(recent_events)
        critical_events = len([e for e in recent_events if e.severity == SeverityLevel.CRITICAL])
        high_events = len([e for e in recent_events if e.severity == SeverityLevel.HIGH])
        medium_events = len([e for e in recent_events if e.severity == SeverityLevel.MEDIUM])
        low_events = len([e for e in recent_events if e.severity == SeverityLevel.LOW])

        # 特定イベント数
        failed_logins = len([
            e for e in recent_events
            if e.event_type == SecurityEventType.FAILED_LOGIN
        ])
        intrusion_attempts = len([
            e for e in recent_events
            if e.event_type == SecurityEventType.INTRUSION_ATTEMPT
        ])
        network_anomalies = len([
            e for e in recent_events
            if e.event_type == SecurityEventType.NETWORK_ANOMALY
        ])

        # システム健全性計算
        system_health = self._calculate_system_health()

        # 脅威レベル判定
        threat_level = self._determine_threat_level(critical_events, high_events)

        return SecurityMetrics(
            timestamp=datetime.now(),
            total_events=total_events,
            critical_events=critical_events,
            high_events=high_events,
            medium_events=medium_events,
            low_events=low_events,
            failed_logins=failed_logins,
            intrusion_attempts=intrusion_attempts,
            network_anomalies=network_anomalies,
            system_health=system_health,
            threat_level=threat_level
        )

    def _calculate_system_health(self) -> float:
        """システム健全性計算"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=1)

            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent

            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent

            # ネットワーク統計
            network = psutil.net_io_counters()

            # 健全性スコア計算 (100点満点)
            health_score = 100.0

            # リソース使用率でペナルティ
            if cpu_usage > 80:
                health_score -= (cpu_usage - 80) * 2
            if memory_usage > 80:
                health_score -= (memory_usage - 80) * 2
            if disk_usage > 90:
                health_score -= (disk_usage - 90) * 3

            # セキュリティイベントでペナルティ
            recent_events = self.get_recent_events(1)  # 過去1時間
            critical_events = len([e for e in recent_events if e.severity == SeverityLevel.CRITICAL])
            high_events = len([e for e in recent_events if e.severity == SeverityLevel.HIGH])

            health_score -= critical_events * 20
            health_score -= high_events * 10

            return max(0.0, min(100.0, health_score))

        except Exception as e:
            logger.error(f"Failed to calculate system health: {e}")
            return 50.0  # デフォルト値

    def _determine_threat_level(self, critical: int, high: int) -> str:
        """脅威レベル判定"""
        if critical > 0:
            return "critical"
        elif high >= 3:
            return "high"
        elif high >= 1:
            return "medium"
        else:
            return "low"


class SecurityDashboard:
    """セキュリティダッシュボード"""

    def __init__(self):
        self.app = FastAPI(title="Security Dashboard")
        self.analyzer = SecurityEventAnalyzer()
        self.active_connections: List[WebSocket] = []

        self._setup_routes()
        self._start_monitoring()

    def _setup_routes(self):
        """ルート設定"""

        @self.app.get("/")
        async def dashboard():
            """ダッシュボードHTML"""
            return HTMLResponse(self._get_dashboard_html())

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket接続"""
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # 定期的にメトリクス送信
                    metrics = self.analyzer.get_metrics()
                    await websocket.send_json(asdict(metrics))
                    await asyncio.sleep(5)  # 5秒間隔

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

        @self.app.get("/api/events")
        async def get_events():
            """セキュリティイベント取得API"""
            events = self.analyzer.get_recent_events(24)
            return {
                "events": [
                    {
                        **asdict(event),
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "severity": event.severity.value
                    }
                    for event in events
                ]
            }

        @self.app.get("/api/metrics")
        async def get_metrics():
            """セキュリティメトリクス取得API"""
            metrics = self.analyzer.get_metrics()
            return asdict(metrics)

    def _start_monitoring(self):
        """監視開始"""
        asyncio.create_task(self._system_monitor())
        asyncio.create_task(self._log_analyzer())

    async def _system_monitor(self):
        """システム監視"""
        while True:
            try:
                # システムリソース監視
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()

                # 異常検知
                if cpu_usage > 95:
                    event = SecurityEvent(
                        id=f"sys_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        event_type=SecurityEventType.NETWORK_ANOMALY,
                        severity=SeverityLevel.HIGH,
                        source_ip="localhost",
                        target="system",
                        description=f"High CPU usage: {cpu_usage}%",
                        details={"cpu_usage": cpu_usage, "memory_usage": memory.percent}
                    )
                    self.analyzer.add_event(event)

                if memory.percent > 90:
                    event = SecurityEvent(
                        id=f"mem_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        event_type=SecurityEventType.NETWORK_ANOMALY,
                        severity=SeverityLevel.MEDIUM,
                        source_ip="localhost",
                        target="system",
                        description=f"High memory usage: {memory.percent}%",
                        details={"memory_usage": memory.percent}
                    )
                    self.analyzer.add_event(event)

                await asyncio.sleep(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)

    async def _log_analyzer(self):
        """ログ分析"""
        while True:
            try:
                # ログファイル分析（例：Nginxアクセスログ）
                await self._analyze_access_logs()

                # アプリケーションログ分析
                await self._analyze_application_logs()

                await asyncio.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"Log analysis error: {e}")
                await asyncio.sleep(120)

    async def _analyze_access_logs(self):
        """アクセスログ分析"""
        # 実装例：Nginxアクセスログの分析
        # 短時間での大量リクエスト検知
        # 異常なユーザーエージェント検知
        # SQLインジェクション試行検知
        pass

    async def _analyze_application_logs(self):
        """アプリケーションログ分析"""
        # 実装例：アプリケーションログの分析
        # 認証失敗の集中検知
        # エラー率の異常上昇検知
        # 異常なAPI呼び出しパターン検知
        pass

    def _get_dashboard_html(self) -> str:
        """ダッシュボードHTML生成"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Security Dashboard - Day Trade ML</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric { text-align: center; margin: 10px 0; }
                .metric-value { font-size: 2em; font-weight: bold; }
                .critical { color: #dc3545; }
                .high { color: #fd7e14; }
                .medium { color: #ffc107; }
                .low { color: #28a745; }
                .threat-critical { background: #dc3545; color: white; }
                .threat-high { background: #fd7e14; color: white; }
                .threat-medium { background: #ffc107; color: black; }
                .threat-low { background: #28a745; color: white; }
                .status-indicator { width: 20px; height: 20px; border-radius: 50%; display: inline-block; margin-right: 10px; }
                .live { animation: pulse 2s infinite; }
                @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 Security Dashboard - Day Trade ML System</h1>

                <div class="grid">
                    <div class="card">
                        <h3>システム健全性</h3>
                        <div class="metric">
                            <div class="metric-value" id="system-health">--</div>
                            <div>System Health</div>
                        </div>
                        <canvas id="health-chart" width="200" height="100"></canvas>
                    </div>

                    <div class="card">
                        <h3>脅威レベル</h3>
                        <div class="metric">
                            <div class="status-indicator live" id="threat-indicator"></div>
                            <span id="threat-level">--</span>
                        </div>
                    </div>

                    <div class="card">
                        <h3>セキュリティイベント (24時間)</h3>
                        <div class="metric critical">
                            <div class="metric-value" id="critical-events">--</div>
                            <div>Critical</div>
                        </div>
                        <div class="metric high">
                            <div class="metric-value" id="high-events">--</div>
                            <div>High</div>
                        </div>
                        <div class="metric medium">
                            <div class="metric-value" id="medium-events">--</div>
                            <div>Medium</div>
                        </div>
                        <div class="metric low">
                            <div class="metric-value" id="low-events">--</div>
                            <div>Low</div>
                        </div>
                    </div>

                    <div class="card">
                        <h3>侵入検知</h3>
                        <div class="metric">
                            <div class="metric-value" id="intrusion-attempts">--</div>
                            <div>Intrusion Attempts</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value" id="failed-logins">--</div>
                            <div>Failed Logins</div>
                        </div>
                    </div>
                </div>

                <div class="card" style="margin-top: 20px;">
                    <h3>最近のセキュリティイベント</h3>
                    <div id="recent-events"></div>
                </div>
            </div>

            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };

                function updateDashboard(metrics) {
                    document.getElementById('system-health').textContent = metrics.system_health.toFixed(1) + '%';
                    document.getElementById('threat-level').textContent = metrics.threat_level.toUpperCase();
                    document.getElementById('critical-events').textContent = metrics.critical_events;
                    document.getElementById('high-events').textContent = metrics.high_events;
                    document.getElementById('medium-events').textContent = metrics.medium_events;
                    document.getElementById('low-events').textContent = metrics.low_events;
                    document.getElementById('intrusion-attempts').textContent = metrics.intrusion_attempts;
                    document.getElementById('failed-logins').textContent = metrics.failed_logins;

                    // 脅威レベル表示更新
                    const indicator = document.getElementById('threat-indicator');
                    const threatLevel = document.getElementById('threat-level');
                    indicator.className = 'status-indicator live threat-' + metrics.threat_level;
                    threatLevel.className = 'threat-' + metrics.threat_level;
                }

                // 最近のイベント取得
                fetch('/api/events')
                    .then(response => response.json())
                    .then(data => {
                        const eventsContainer = document.getElementById('recent-events');
                        eventsContainer.innerHTML = data.events.slice(0, 10).map(event => `
                            <div style="padding: 10px; margin: 5px 0; border-left: 4px solid var(--${event.severity}-color); background: #f8f9fa;">
                                <strong>${event.event_type}</strong> - ${event.severity.toUpperCase()}
                                <br><small>${event.timestamp} | ${event.source_ip} → ${event.target}</small>
                                <br>${event.description}
                            </div>
                        `).join('');
                    });
            </script>
        </body>
        </html>
        """


# 使用例とテスト
async def simulate_security_events():
    """セキュリティイベントのシミュレーション"""
    dashboard = SecurityDashboard()

    # テストイベント生成
    test_events = [
        SecurityEvent(
            id="test_001",
            timestamp=datetime.now(),
            event_type=SecurityEventType.FAILED_LOGIN,
            severity=SeverityLevel.MEDIUM,
            source_ip="192.168.1.100",
            target="auth_service",
            description="Multiple failed login attempts",
            details={"attempts": 5, "username": "admin"}
        ),
        SecurityEvent(
            id="test_002",
            timestamp=datetime.now(),
            event_type=SecurityEventType.INTRUSION_ATTEMPT,
            severity=SeverityLevel.HIGH,
            source_ip="10.0.0.50",
            target="api_gateway",
            description="SQL injection attempt detected",
            details={"payload": "'; DROP TABLE users; --", "endpoint": "/api/login"}
        )
    ]

    for event in test_events:
        dashboard.analyzer.add_event(event)

    return dashboard


if __name__ == "__main__":
    import uvicorn

    # ダッシュボード起動
    dashboard = SecurityDashboard()

    # テストイベント追加
    asyncio.create_task(simulate_security_events())

    # サーバー起動
    uvicorn.run(
        dashboard.app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )