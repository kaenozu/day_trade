#!/usr/bin/env python3
"""
統合ダッシュボードシステム - Phase J-1: Issue #901最終統合フェーズ

全システムを統合管理する統合ダッシュボードシステムの実装
リアルタイム監視、予測システム、リスク管理、パフォーマンス分析を統合
"""

import asyncio
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..analysis.enhanced_ensemble import EnhancedEnsembleStrategy
from ..api.realtime_prediction_api import RealtimePredictionAPI
from ..automation.orchestrator import AutomationOrchestrator
from ..core.enhanced_error_handler import EnhancedErrorHandler
from ..data.enhanced_data_version_control import EnhancedDataVersionControl
from ..ml.ensemble_system import EnsembleSystem
from ..monitoring.advanced_monitoring_system import AdvancedMonitoringSystem
from ..realtime.live_prediction_engine import LivePredictionEngine
from ..risk.integrated_risk_management import IntegratedRiskManagement
from ..security.integrated_security_dashboard import IntegratedSecurityDashboard
from ..utils.logging_config import get_context_logger
from ..visualization.dashboard.interactive_dashboard import InteractiveDashboard
from .dashboard_core import ProductionDashboard
from .web_dashboard import WebDashboard

logger = get_context_logger(__name__)


class IntegratedDashboardSystem:
    """統合ダッシュボードシステム"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self.data_dir = self.base_dir / "data" / "integrated_dashboard"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # データベース初期化
        self.db_path = self.data_dir / "integrated_dashboard.db"
        self._init_database()

        # エラーハンドラー
        self.error_handler = EnhancedErrorHandler()

        # コアコンポーネント
        self.web_dashboard = None
        self.production_dashboard = None
        self.interactive_dashboard = None
        self.security_dashboard = None

        # 分析・予測システム
        self.ensemble_system = None
        self.enhanced_ensemble = None
        self.prediction_engine = None
        self.prediction_api = None

        # 監視・リスク管理
        self.monitoring_system = None
        self.risk_management = None
        self.automation_orchestrator = None

        # データ管理
        self.data_version_control = None

        # システム状態
        self.is_running = False
        self.last_update = None
        self.system_metrics = {}
        self.active_alerts = []

        # スレッド制御
        self.update_thread = None
        self.monitoring_thread = None

        logger.info("統合ダッシュボードシステム初期化完了")

    def _init_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # システムメトリクステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        component TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        metric_data TEXT,
                        status TEXT DEFAULT 'normal'
                    )
                ''')

                # アラートテーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        component TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        acknowledged BOOLEAN DEFAULT FALSE,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                ''')

                # システム統計テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_statistics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        total_predictions INTEGER DEFAULT 0,
                        successful_predictions INTEGER DEFAULT 0,
                        prediction_accuracy REAL DEFAULT 0.0,
                        total_trades INTEGER DEFAULT 0,
                        profitable_trades INTEGER DEFAULT 0,
                        total_profit REAL DEFAULT 0.0,
                        system_uptime INTEGER DEFAULT 0,
                        active_components INTEGER DEFAULT 0
                    )
                ''')

                # パフォーマンステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        response_time REAL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        network_io REAL,
                        prediction_latency REAL,
                        throughput REAL
                    )
                ''')

                conn.commit()
                logger.info("統合ダッシュボードデータベース初期化完了")

        except Exception as e:
            error_msg = f"データベース初期化エラー: {e}"
            logger.error(error_msg)
            self.error_handler.handle_error(e, {"context": "database_init"})

    async def initialize_components(self):
        """全コンポーネント初期化"""
        try:
            logger.info("統合ダッシュボードコンポーネント初期化開始")

            # Webダッシュボード
            self.web_dashboard = WebDashboard(port=5000, debug=False)

            # プロダクションダッシュボード
            self.production_dashboard = ProductionDashboard()

            # インタラクティブダッシュボード
            self.interactive_dashboard = InteractiveDashboard()

            # セキュリティダッシュボード
            self.security_dashboard = IntegratedSecurityDashboard()

            # 分析・予測システム
            self.ensemble_system = EnsembleSystem()
            self.enhanced_ensemble = EnhancedEnsembleStrategy()
            self.prediction_engine = LivePredictionEngine()
            self.prediction_api = RealtimePredictionAPI()

            # 監視・リスク管理
            self.monitoring_system = AdvancedMonitoringSystem()
            self.risk_management = IntegratedRiskManagement()
            self.automation_orchestrator = AutomationOrchestrator()

            # データ管理
            self.data_version_control = EnhancedDataVersionControl()

            # 初期統計記録
            await self._record_system_startup()

            logger.info("統合ダッシュボードコンポーネント初期化完了")

        except Exception as e:
            error_msg = f"コンポーネント初期化エラー: {e}"
            logger.error(error_msg)
            self.error_handler.handle_error(e, {"context": "component_init"})
            raise

    async def start_integrated_monitoring(self):
        """統合監視開始"""
        try:
            logger.info("統合監視システム開始")
            self.is_running = True

            # 各コンポーネントの監視開始
            if self.monitoring_system:
                await self.monitoring_system.start_monitoring()

            if self.risk_management:
                await self.risk_management.start_monitoring()

            if self.prediction_engine:
                await self.prediction_engine.start()

            # 統合監視スレッド開始
            self.monitoring_thread = threading.Thread(
                target=self._integrated_monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()

            # メトリクス更新スレッド開始
            self.update_thread = threading.Thread(
                target=self._metrics_update_loop, daemon=True
            )
            self.update_thread.start()

            logger.info("統合監視システム開始完了")

        except Exception as e:
            error_msg = f"統合監視開始エラー: {e}"
            logger.error(error_msg)
            self.error_handler.handle_error(e, {"context": "monitoring_start"})

    def _integrated_monitoring_loop(self):
        """統合監視ループ"""
        while self.is_running:
            try:
                # システム全体のヘルスチェック
                self._perform_health_check()

                # アラート処理
                self._process_alerts()

                # パフォーマンス監視
                self._monitor_performance()

                # 自動最適化
                self._auto_optimization()

                time.sleep(30)  # 30秒間隔

            except Exception as e:
                logger.error(f"統合監視ループエラー: {e}")
                self.error_handler.handle_error(e, {"context": "monitoring_loop"})
                time.sleep(10)

    def _metrics_update_loop(self):
        """メトリクス更新ループ"""
        while self.is_running:
            try:
                # システムメトリクス収集
                metrics = self._collect_system_metrics()

                # データベース保存
                self._save_metrics(metrics)

                # 最新情報更新
                self.last_update = datetime.now()
                self.system_metrics = metrics

                time.sleep(60)  # 1分間隔

            except Exception as e:
                logger.error(f"メトリクス更新ループエラー: {e}")
                time.sleep(30)

    def _perform_health_check(self):
        """システムヘルスチェック"""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'components': {},
                'overall_status': 'healthy'
            }

            # 各コンポーネントのヘルスチェック
            components = {
                'web_dashboard': self.web_dashboard,
                'production_dashboard': self.production_dashboard,
                'ensemble_system': self.ensemble_system,
                'prediction_engine': self.prediction_engine,
                'monitoring_system': self.monitoring_system,
                'risk_management': self.risk_management
            }

            unhealthy_count = 0
            for name, component in components.items():
                if component:
                    try:
                        # 各コンポーネントのヘルスチェック実行
                        if hasattr(component, 'health_check'):
                            status = component.health_check()
                        else:
                            # 基本的な存在チェック
                            status = {'status': 'healthy', 'details': 'Component active'}

                        health_status['components'][name] = status

                        if status.get('status') != 'healthy':
                            unhealthy_count += 1

                    except Exception as e:
                        health_status['components'][name] = {
                            'status': 'unhealthy',
                            'error': str(e)
                        }
                        unhealthy_count += 1
                else:
                    health_status['components'][name] = {
                        'status': 'not_initialized'
                    }
                    unhealthy_count += 1

            # 全体ステータス判定
            if unhealthy_count == 0:
                health_status['overall_status'] = 'healthy'
            elif unhealthy_count <= 2:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'unhealthy'

            # ヘルスステータスをメトリクスに記録
            self._record_metric(
                'system_health',
                'overall_status',
                1 if health_status['overall_status'] == 'healthy' else 0,
                json.dumps(health_status)
            )

        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")

    def _process_alerts(self):
        """アラート処理"""
        try:
            # 新しいアラートを収集
            new_alerts = []

            # 各コンポーネントからアラート収集
            if self.monitoring_system and hasattr(self.monitoring_system, 'get_alerts'):
                new_alerts.extend(self.monitoring_system.get_alerts())

            if self.risk_management and hasattr(self.risk_management, 'get_alerts'):
                new_alerts.extend(self.risk_management.get_alerts())

            # アラートをデータベースに保存
            for alert in new_alerts:
                self._save_alert(alert)

            # アクティブアラート更新
            self.active_alerts = self._get_active_alerts()

        except Exception as e:
            logger.error(f"アラート処理エラー: {e}")

    def _monitor_performance(self):
        """パフォーマンス監視"""
        try:
            import psutil
            import time as time_module

            start_time = time_module.time()

            # システムリソース取得
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            network_info = psutil.net_io_counters()

            # レスポンス時間測定（簡易版）
            response_time = time_module.time() - start_time

            # パフォーマンスメトリクス記録
            performance_data = {
                'response_time': response_time,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_info.percent,
                'disk_usage': disk_info.percent if hasattr(disk_info, 'percent') else 0,
                'network_io': network_info.bytes_sent + network_info.bytes_recv,
                'prediction_latency': 0.0,  # 実際の予測レイテンシは別途測定
                'throughput': 0.0  # 実際のスループットは別途測定
            }

            self._save_performance_metrics(performance_data)

        except Exception as e:
            logger.error(f"パフォーマンス監視エラー: {e}")

    def _auto_optimization(self):
        """自動最適化"""
        try:
            # システム負荷チェック
            if hasattr(self, 'system_metrics') and self.system_metrics:
                cpu_usage = self.system_metrics.get('cpu_usage', 0)
                memory_usage = self.system_metrics.get('memory_usage', 0)

                # 高負荷時の自動最適化
                if cpu_usage > 80 or memory_usage > 90:
                    logger.warning(f"システム高負荷検出 - CPU: {cpu_usage}%, Memory: {memory_usage}%")

                    # キャッシュクリア
                    if self.ensemble_system and hasattr(self.ensemble_system, 'clear_cache'):
                        self.ensemble_system.clear_cache()

                    # ガベージコレクション実行
                    import gc
                    gc.collect()

        except Exception as e:
            logger.error(f"自動最適化エラー: {e}")

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """システムメトリクス収集"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system': {},
                'prediction': {},
                'trading': {},
                'risk': {},
                'performance': {}
            }

            # システムメトリクス
            import psutil
            metrics['system'] = {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
                'uptime': time.time() - psutil.boot_time()
            }

            # 予測システムメトリクス
            if self.prediction_engine:
                if hasattr(self.prediction_engine, 'get_metrics'):
                    metrics['prediction'] = self.prediction_engine.get_metrics()

            # 取引システムメトリクス
            if self.automation_orchestrator:
                if hasattr(self.automation_orchestrator, 'get_trading_metrics'):
                    metrics['trading'] = self.automation_orchestrator.get_trading_metrics()

            # リスクメトリクス
            if self.risk_management:
                if hasattr(self.risk_management, 'get_risk_metrics'):
                    metrics['risk'] = self.risk_management.get_risk_metrics()

            return metrics

        except Exception as e:
            logger.error(f"システムメトリクス収集エラー: {e}")
            return {}

    def _save_metrics(self, metrics: Dict[str, Any]):
        """メトリクス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 各コンポーネントのメトリクスを保存
                for component, component_metrics in metrics.items():
                    if isinstance(component_metrics, dict):
                        for metric_name, metric_value in component_metrics.items():
                            if isinstance(metric_value, (int, float)):
                                cursor.execute('''
                                    INSERT INTO system_metrics
                                    (component, metric_name, metric_value)
                                    VALUES (?, ?, ?)
                                ''', (component, metric_name, metric_value))
                            else:
                                cursor.execute('''
                                    INSERT INTO system_metrics
                                    (component, metric_name, metric_data)
                                    VALUES (?, ?, ?)
                                ''', (component, metric_name, json.dumps(metric_value)))

                conn.commit()

        except Exception as e:
            logger.error(f"メトリクス保存エラー: {e}")

    def _record_metric(self, component: str, metric_name: str,
                      metric_value: Optional[float] = None,
                      metric_data: Optional[str] = None):
        """個別メトリクス記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics
                    (component, metric_name, metric_value, metric_data)
                    VALUES (?, ?, ?, ?)
                ''', (component, metric_name, metric_value, metric_data))
                conn.commit()

        except Exception as e:
            logger.error(f"メトリクス記録エラー: {e}")

    def _save_alert(self, alert: Dict[str, Any]):
        """アラート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts
                    (alert_type, severity, component, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert.get('type', 'unknown'),
                    alert.get('severity', 'info'),
                    alert.get('component', 'system'),
                    alert.get('message', ''),
                    json.dumps(alert.get('details', {}))
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"アラート保存エラー: {e}")

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT alert_type, severity, component, message, details, timestamp
                    FROM alerts
                    WHERE resolved = FALSE
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''')

                alerts = []
                for row in cursor.fetchall():
                    alerts.append({
                        'type': row[0],
                        'severity': row[1],
                        'component': row[2],
                        'message': row[3],
                        'details': json.loads(row[4]) if row[4] else {},
                        'timestamp': row[5]
                    })

                return alerts

        except Exception as e:
            logger.error(f"アクティブアラート取得エラー: {e}")
            return []

    def _save_performance_metrics(self, performance_data: Dict[str, float]):
        """パフォーマンスメトリクス保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics
                    (response_time, cpu_usage, memory_usage, disk_usage,
                     network_io, prediction_latency, throughput)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance_data.get('response_time', 0),
                    performance_data.get('cpu_usage', 0),
                    performance_data.get('memory_usage', 0),
                    performance_data.get('disk_usage', 0),
                    performance_data.get('network_io', 0),
                    performance_data.get('prediction_latency', 0),
                    performance_data.get('throughput', 0)
                ))
                conn.commit()

        except Exception as e:
            logger.error(f"パフォーマンスメトリクス保存エラー: {e}")

    async def _record_system_startup(self):
        """システム起動記録"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_statistics
                    (total_predictions, successful_predictions, prediction_accuracy,
                     total_trades, profitable_trades, total_profit,
                     system_uptime, active_components)
                    VALUES (0, 0, 0.0, 0, 0, 0.0, 0, 8)
                ''')
                conn.commit()

        except Exception as e:
            logger.error(f"システム起動記録エラー: {e}")

    def get_integrated_status(self) -> Dict[str, Any]:
        """統合ステータス取得"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'running' if self.is_running else 'stopped',
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'active_alerts_count': len(self.active_alerts),
                'system_metrics': self.system_metrics,
                'active_alerts': self.active_alerts[:10],  # 最新10件
                'components_status': {
                    'web_dashboard': self.web_dashboard is not None,
                    'production_dashboard': self.production_dashboard is not None,
                    'interactive_dashboard': self.interactive_dashboard is not None,
                    'security_dashboard': self.security_dashboard is not None,
                    'ensemble_system': self.ensemble_system is not None,
                    'prediction_engine': self.prediction_engine is not None,
                    'monitoring_system': self.monitoring_system is not None,
                    'risk_management': self.risk_management is not None
                }
            }

        except Exception as e:
            logger.error(f"統合ステータス取得エラー: {e}")
            return {'error': str(e)}

    def get_historical_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """過去メトリクス取得"""
        try:
            since = datetime.now() - timedelta(hours=hours)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # システムメトリクス
                cursor.execute('''
                    SELECT timestamp, component, metric_name, metric_value
                    FROM system_metrics
                    WHERE timestamp >= ? AND metric_value IS NOT NULL
                    ORDER BY timestamp
                ''', (since.isoformat(),))

                metrics_data = {}
                for row in cursor.fetchall():
                    timestamp, component, metric_name, metric_value = row
                    if component not in metrics_data:
                        metrics_data[component] = {}
                    if metric_name not in metrics_data[component]:
                        metrics_data[component][metric_name] = []

                    metrics_data[component][metric_name].append({
                        'timestamp': timestamp,
                        'value': metric_value
                    })

                # パフォーマンスメトリクス
                cursor.execute('''
                    SELECT timestamp, response_time, cpu_usage, memory_usage
                    FROM performance_metrics
                    WHERE timestamp >= ?
                    ORDER BY timestamp
                ''', (since.isoformat(),))

                performance_data = []
                for row in cursor.fetchall():
                    performance_data.append({
                        'timestamp': row[0],
                        'response_time': row[1],
                        'cpu_usage': row[2],
                        'memory_usage': row[3]
                    })

                return {
                    'metrics': metrics_data,
                    'performance': performance_data,
                    'period_hours': hours
                }

        except Exception as e:
            logger.error(f"過去メトリクス取得エラー: {e}")
            return {}

    async def stop_integrated_monitoring(self):
        """統合監視停止"""
        try:
            logger.info("統合監視システム停止開始")
            self.is_running = False

            # 各コンポーネントの監視停止
            if self.monitoring_system and hasattr(self.monitoring_system, 'stop_monitoring'):
                await self.monitoring_system.stop_monitoring()

            if self.risk_management and hasattr(self.risk_management, 'stop_monitoring'):
                await self.risk_management.stop_monitoring()

            if self.prediction_engine and hasattr(self.prediction_engine, 'stop'):
                await self.prediction_engine.stop()

            # スレッド終了待機
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)

            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)

            logger.info("統合監視システム停止完了")

        except Exception as e:
            logger.error(f"統合監視停止エラー: {e}")

    def run_web_interface(self, port: int = 5000):
        """Web インターフェース起動"""
        try:
            if self.web_dashboard:
                logger.info(f"統合ダッシュボードWebインターフェース起動: http://localhost:{port}")
                self.web_dashboard.run()
            else:
                logger.error("Webダッシュボードが初期化されていません")

        except Exception as e:
            logger.error(f"Webインターフェース起動エラー: {e}")


async def main():
    """メイン実行"""
    logger.info("統合ダッシュボードシステム起動")
    logger.info("=" * 60)

    integrated_dashboard = IntegratedDashboardSystem()

    try:
        # コンポーネント初期化
        await integrated_dashboard.initialize_components()

        # 統合監視開始
        await integrated_dashboard.start_integrated_monitoring()

        # Webインターフェース起動
        integrated_dashboard.run_web_interface()

    except KeyboardInterrupt:
        logger.info("\n統合ダッシュボードシステム停止中...")
        await integrated_dashboard.stop_integrated_monitoring()

    except Exception as e:
        logger.error(f"統合ダッシュボードシステムエラー: {e}")
        await integrated_dashboard.stop_integrated_monitoring()


if __name__ == "__main__":
    asyncio.run(main())