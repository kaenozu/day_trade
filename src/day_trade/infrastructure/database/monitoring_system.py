"""
データベース監視・アラートシステム

本番環境でのデータベースパフォーマンス監視、異常検知、アラート機能
リアルタイム監視、閾値ベースアラート、ダッシュボード連携対応
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import psutil

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from day_trade.core.error_handling.unified_error_system import (
    ApplicationError, DataAccessError, SystemError,
    error_boundary, global_error_handler
)
from day_trade.core.logging.unified_logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class DatabaseMetrics:
    """データベースメトリクス"""
    timestamp: datetime

    # 接続関連
    active_connections: int
    max_connections: int
    connection_pool_usage: float

    # パフォーマンス関連
    queries_per_second: float
    average_query_time: float
    slow_queries_count: int

    # リソース関連
    cpu_usage: float
    memory_usage_mb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float

    # エラー関連
    connection_errors: int
    query_errors: int
    deadlocks: int

    # データベース固有
    database_size_mb: float
    table_count: int
    index_count: int


@dataclass
class AlertRule:
    """アラートルール"""
    name: str
    metric_name: str
    operator: str  # >, <, >=, <=, ==
    threshold: float
    duration_seconds: int
    severity: str  # critical, warning, info
    enabled: bool
    description: str


@dataclass
class Alert:
    """アラート"""
    id: str
    rule_name: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool
    resolved_at: Optional[datetime] = None


class MonitoringError(SystemError):
    """監視システム専用エラー"""

    def __init__(self, message: str, monitor_type: str = None, **kwargs):
        super().__init__(message, operation=f"monitoring_{monitor_type}", **kwargs)


class DatabaseMonitoringSystem:
    """データベース監視システム"""

    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config
        self.monitoring_config = config.get('monitoring', {})

        # 監視設定
        self.enabled = self.monitoring_config.get('enabled', True)
        self.interval_seconds = self.monitoring_config.get('interval_seconds', 30)
        self.metrics_retention_hours = self.monitoring_config.get('metrics_retention_hours', 24)
        self.max_metrics_count = int(self.metrics_retention_hours * 3600 / self.interval_seconds)

        # データ保存
        self.metrics_history: deque = deque(maxlen=self.max_metrics_count)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # アラートルール
        self.alert_rules: List[AlertRule] = []
        self._load_default_alert_rules()

        # アラート通知コールバック
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # 監視状態
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_running = False

        # 統計情報
        self._last_metrics: Optional[DatabaseMetrics] = None
        self._alert_counts = defaultdict(int)

        # データベース種別検出
        self.database_type = self._detect_database_type()

    def _detect_database_type(self) -> str:
        """データベース種別検出"""
        try:
            dialect_name = self.engine.dialect.name
            if dialect_name in ['postgresql', 'sqlite']:
                return dialect_name
            else:
                return 'unknown'
        except Exception:
            return 'unknown'

    def _load_default_alert_rules(self) -> None:
        """デフォルトアラートルール読み込み"""
        default_rules = [
            AlertRule(
                name="high_connection_usage",
                metric_name="connection_pool_usage",
                operator=">=",
                threshold=0.8,
                duration_seconds=300,
                severity="warning",
                enabled=True,
                description="接続プール使用率が80%以上"
            ),
            AlertRule(
                name="critical_connection_usage",
                metric_name="connection_pool_usage",
                operator=">=",
                threshold=0.95,
                duration_seconds=60,
                severity="critical",
                enabled=True,
                description="接続プール使用率が95%以上"
            ),
            AlertRule(
                name="slow_queries",
                metric_name="slow_queries_count",
                operator=">",
                threshold=10,
                duration_seconds=300,
                severity="warning",
                enabled=True,
                description="スロークエリが10件以上"
            ),
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage",
                operator=">=",
                threshold=80.0,
                duration_seconds=600,
                severity="warning",
                enabled=True,
                description="CPU使用率が80%以上"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="disk_usage_percent",
                operator=">=",
                threshold=85.0,
                duration_seconds=300,
                severity="critical",
                enabled=True,
                description="ディスク使用率が85%以上"
            ),
            AlertRule(
                name="connection_errors",
                metric_name="connection_errors",
                operator=">",
                threshold=5,
                duration_seconds=300,
                severity="warning",
                enabled=True,
                description="接続エラーが5件以上"
            ),
            AlertRule(
                name="deadlocks",
                metric_name="deadlocks",
                operator=">",
                threshold=0,
                duration_seconds=60,
                severity="warning",
                enabled=True,
                description="デッドロックが発生"
            )
        ]

        # 設定からカスタムルールを追加
        custom_rules = self.monitoring_config.get('alert_rules', [])
        for rule_config in custom_rules:
            rule = AlertRule(**rule_config)
            default_rules.append(rule)

        self.alert_rules = default_rules
        logger.info(f"アラートルール読み込み完了: {len(self.alert_rules)}件")

    @error_boundary(
        component_name="monitoring_system",
        operation_name="collect_metrics",
        suppress_errors=True
    )
    def collect_metrics(self) -> Optional[DatabaseMetrics]:
        """メトリクス収集"""
        try:
            timestamp = datetime.now()

            # データベース接続情報
            connection_info = self._get_connection_metrics()

            # パフォーマンス情報
            performance_info = self._get_performance_metrics()

            # システムリソース情報
            system_info = self._get_system_metrics()

            # データベース情報
            database_info = self._get_database_metrics()

            metrics = DatabaseMetrics(
                timestamp=timestamp,
                **connection_info,
                **performance_info,
                **system_info,
                **database_info
            )

            # メトリクス履歴に追加
            self.metrics_history.append(metrics)
            self._last_metrics = metrics

            logger.debug(
                "メトリクス収集完了",
                active_connections=metrics.active_connections,
                cpu_usage=metrics.cpu_usage,
                memory_usage_mb=metrics.memory_usage_mb
            )

            return metrics

        except Exception as e:
            logger.error(f"メトリクス収集失敗: {e}")
            return None

    def _get_connection_metrics(self) -> Dict[str, Any]:
        """接続関連メトリクス取得"""
        try:
            pool = self.engine.pool

            # SQLAlchemy版に応じた属性アクセス
            try:
                active_connections = getattr(pool, 'checkedout', lambda: 0)()
                size = getattr(pool, 'size', lambda: 0)()
                overflow = getattr(pool, 'overflow', lambda: 0)()
                max_connections = size + overflow
                connection_pool_usage = active_connections / max_connections if max_connections > 0 else 0
            except Exception:
                # フォールバック値
                active_connections = 0
                max_connections = 0
                connection_pool_usage = 0.0

            return {
                "active_connections": active_connections,
                "max_connections": max_connections,
                "connection_pool_usage": connection_pool_usage
            }

        except Exception as e:
            logger.warning(f"接続メトリクス取得失敗: {e}")
            return {
                "active_connections": 0,
                "max_connections": 0,
                "connection_pool_usage": 0.0
            }

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンスメトリクス取得"""
        try:
            # データベース種別に応じたクエリ実行
            if self.database_type == 'postgresql':
                metrics = self._get_postgresql_performance()
            elif self.database_type == 'sqlite':
                metrics = self._get_sqlite_performance()
            else:
                metrics = self._get_generic_performance()

            return metrics

        except Exception as e:
            logger.warning(f"パフォーマンスメトリクス取得失敗: {e}")
            return {
                "queries_per_second": 0.0,
                "average_query_time": 0.0,
                "slow_queries_count": 0
            }

    def _get_postgresql_performance(self) -> Dict[str, Any]:
        """PostgreSQLパフォーマンスメトリクス"""
        try:
            with self.engine.connect() as conn:
                # アクティブクエリ数
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM pg_stat_activity
                    WHERE state = 'active' AND query != '<IDLE>'
                """))
                active_queries = result.scalar()

                # 実行時間の長いクエリ
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM pg_stat_activity
                    WHERE state = 'active'
                    AND now() - query_start > interval '5 seconds'
                """))
                slow_queries = result.scalar()

                return {
                    "queries_per_second": float(active_queries) / self.interval_seconds,
                    "average_query_time": 0.0,  # 詳細計算は複雑なため簡略化
                    "slow_queries_count": slow_queries
                }

        except Exception as e:
            logger.debug(f"PostgreSQL固有メトリクス取得失敗: {e}")
            return self._get_generic_performance()

    def _get_sqlite_performance(self) -> Dict[str, Any]:
        """SQLiteパフォーマンスメトリクス"""
        # SQLiteは限定的な統計情報のため基本値を返す
        return {
            "queries_per_second": 0.0,
            "average_query_time": 0.0,
            "slow_queries_count": 0
        }

    def _get_generic_performance(self) -> Dict[str, Any]:
        """汎用パフォーマンスメトリクス"""
        return {
            "queries_per_second": 0.0,
            "average_query_time": 0.0,
            "slow_queries_count": 0
        }

    def _get_system_metrics(self) -> Dict[str, Any]:
        """システムリソースメトリクス取得"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=None)

            # メモリ使用量
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024

            # ディスク使用率（データベースファイルのあるディスク）
            disk_usage = psutil.disk_usage('.')
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100

            # ディスクI/O（簡易版）
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / 1024 / 1024 if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / 1024 / 1024 if disk_io else 0

            return {
                "cpu_usage": cpu_usage,
                "memory_usage_mb": memory_usage_mb,
                "disk_usage_percent": disk_usage_percent,
                "disk_io_read_mb": disk_io_read_mb,
                "disk_io_write_mb": disk_io_write_mb,
                "connection_errors": 0,  # 実装簡略化
                "query_errors": 0,      # 実装簡略化
                "deadlocks": 0          # 実装簡略化
            }

        except Exception as e:
            logger.warning(f"システムメトリクス取得失敗: {e}")
            return {
                "cpu_usage": 0.0,
                "memory_usage_mb": 0.0,
                "disk_usage_percent": 0.0,
                "disk_io_read_mb": 0.0,
                "disk_io_write_mb": 0.0,
                "connection_errors": 0,
                "query_errors": 0,
                "deadlocks": 0
            }

    def _get_database_metrics(self) -> Dict[str, Any]:
        """データベース固有メトリクス取得"""
        try:
            if self.database_type == 'postgresql':
                return self._get_postgresql_database_metrics()
            elif self.database_type == 'sqlite':
                return self._get_sqlite_database_metrics()
            else:
                return self._get_generic_database_metrics()

        except Exception as e:
            logger.warning(f"データベースメトリクス取得失敗: {e}")
            return self._get_generic_database_metrics()

    def _get_postgresql_database_metrics(self) -> Dict[str, Any]:
        """PostgreSQLデータベースメトリクス"""
        try:
            with self.engine.connect() as conn:
                # データベースサイズ
                result = conn.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size_str = result.scalar()

                # テーブル数
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = 'public'
                """))
                table_count = result.scalar()

                # インデックス数
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM pg_indexes
                    WHERE schemaname = 'public'
                """))
                index_count = result.scalar()

                # サイズを数値に変換（簡易版）
                database_size_mb = self._parse_size_string(db_size_str)

                return {
                    "database_size_mb": database_size_mb,
                    "table_count": table_count,
                    "index_count": index_count
                }

        except Exception as e:
            logger.debug(f"PostgreSQL固有データベースメトリクス取得失敗: {e}")
            return self._get_generic_database_metrics()

    def _get_sqlite_database_metrics(self) -> Dict[str, Any]:
        """SQLiteデータベースメトリクス"""
        try:
            with self.engine.connect() as conn:
                # データベースサイズ（ファイルサイズから）
                database_url = str(self.engine.url)
                if database_url.startswith('sqlite:///'):
                    db_file = database_url.replace('sqlite:///', '')
                    try:
                        import os
                        size_bytes = os.path.getsize(db_file)
                        database_size_mb = size_bytes / 1024 / 1024
                    except:
                        database_size_mb = 0.0
                else:
                    database_size_mb = 0.0

                # テーブル数
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM sqlite_master
                    WHERE type = 'table' AND name != 'sqlite_sequence'
                """))
                table_count = result.scalar()

                # インデックス数
                result = conn.execute(text("""
                    SELECT COUNT(*) FROM sqlite_master WHERE type = 'index'
                """))
                index_count = result.scalar()

                return {
                    "database_size_mb": database_size_mb,
                    "table_count": table_count,
                    "index_count": index_count
                }

        except Exception as e:
            logger.debug(f"SQLite固有データベースメトリクス取得失敗: {e}")
            return self._get_generic_database_metrics()

    def _get_generic_database_metrics(self) -> Dict[str, Any]:
        """汎用データベースメトリクス"""
        return {
            "database_size_mb": 0.0,
            "table_count": 0,
            "index_count": 0
        }

    def _parse_size_string(self, size_str: str) -> float:
        """サイズ文字列を数値に変換"""
        if not size_str:
            return 0.0

        size_str = size_str.strip().lower()

        try:
            if 'mb' in size_str:
                return float(size_str.replace('mb', '').strip())
            elif 'gb' in size_str:
                return float(size_str.replace('gb', '').strip()) * 1024
            elif 'kb' in size_str:
                return float(size_str.replace('kb', '').strip()) / 1024
            elif 'bytes' in size_str:
                return float(size_str.replace('bytes', '').strip()) / 1024 / 1024
            else:
                # 数値のみの場合はバイト単位と仮定
                return float(size_str) / 1024 / 1024
        except:
            return 0.0

    def check_alerts(self, metrics: DatabaseMetrics) -> List[Alert]:
        """アラートチェック"""
        new_alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # メトリクス値取得
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is None:
                continue

            # 閾値チェック
            is_triggered = self._evaluate_threshold(metric_value, rule.operator, rule.threshold)

            alert_id = f"{rule.name}_{rule.metric_name}"

            if is_triggered:
                # 既存アラートがある場合はスキップ
                if alert_id in self.active_alerts:
                    continue

                # 新しいアラート作成
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    current_value=metric_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"{rule.description}: 現在値={metric_value:.2f}, 閾値={rule.threshold}",
                    timestamp=metrics.timestamp,
                    resolved=False
                )

                self.active_alerts[alert_id] = alert
                new_alerts.append(alert)
                self._alert_counts[rule.severity] += 1

                logger.warning(
                    f"アラート発生: {alert.message}",
                    severity=alert.severity,
                    metric=rule.metric_name,
                    value=metric_value
                )

            else:
                # アラート解決チェック
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = metrics.timestamp

                    self.alert_history.append(alert)
                    del self.active_alerts[alert_id]

                    logger.info(
                        f"アラート解決: {alert.message}",
                        duration_seconds=(alert.resolved_at - alert.timestamp).total_seconds()
                    )

        # アラート通知実行
        for alert in new_alerts:
            self._notify_alert(alert)

        return new_alerts

    def _evaluate_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """閾値評価"""
        if operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.001
        else:
            logger.warning(f"不明な比較演算子: {operator}")
            return False

    def _notify_alert(self, alert: Alert) -> None:
        """アラート通知"""
        try:
            for callback in self.alert_callbacks:
                callback(alert)
        except Exception as e:
            logger.error(f"アラート通知失敗: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """アラート通知コールバック追加"""
        self.alert_callbacks.append(callback)
        logger.info("アラート通知コールバック追加")

    def start_monitoring(self) -> None:
        """監視開始"""
        if not self.enabled:
            logger.info("監視が無効のため開始しません")
            return

        if self._monitoring_running:
            logger.warning("監視は既に実行中です")
            return

        self._monitoring_running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()

        logger.info(f"データベース監視開始: {self.interval_seconds}秒間隔")

    def stop_monitoring(self) -> None:
        """監視停止"""
        self._monitoring_running = False

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.info("監視停止中...")
            self._monitoring_thread.join(timeout=10)

        logger.info("データベース監視停止")

    def _monitoring_loop(self) -> None:
        """監視ループ"""
        logger.info("データベース監視ループ開始")

        while self._monitoring_running:
            try:
                # メトリクス収集
                metrics = self.collect_metrics()

                if metrics:
                    # アラートチェック
                    self.check_alerts(metrics)

                # インターバル待機
                time.sleep(self.interval_seconds)

            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(60)  # エラー時は1分待機

        logger.info("データベース監視ループ終了")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態取得"""
        return {
            "enabled": self.enabled,
            "running": self._monitoring_running,
            "interval_seconds": self.interval_seconds,
            "metrics_count": len(self.metrics_history),
            "active_alerts_count": len(self.active_alerts),
            "alert_history_count": len(self.alert_history),
            "alert_rules_count": len([rule for rule in self.alert_rules if rule.enabled]),
            "last_collection": self._last_metrics.timestamp.isoformat() if self._last_metrics else None
        }

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """現在のメトリクス取得"""
        if self._last_metrics:
            return asdict(self._last_metrics)
        return None

    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """メトリクス履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_metrics = [
            asdict(metrics) for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

        return recent_metrics

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        return [asdict(alert) for alert in self.active_alerts.values()]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """アラート統計取得"""
        total_alerts = len(self.alert_history) + len(self.active_alerts)

        return {
            "total_alerts": total_alerts,
            "active_alerts": len(self.active_alerts),
            "resolved_alerts": len(self.alert_history),
            "critical_count": self._alert_counts.get('critical', 0),
            "warning_count": self._alert_counts.get('warning', 0),
            "info_count": self._alert_counts.get('info', 0)
        }


# グローバルインスタンス管理
_monitoring_system: Optional[DatabaseMonitoringSystem] = None


def get_monitoring_system() -> Optional[DatabaseMonitoringSystem]:
    """監視システム取得"""
    return _monitoring_system


def initialize_monitoring_system(engine: Engine, config: Dict[str, Any]) -> DatabaseMonitoringSystem:
    """監視システム初期化"""
    global _monitoring_system

    _monitoring_system = DatabaseMonitoringSystem(engine, config)

    # 自動監視開始
    if config.get('monitoring', {}).get('auto_start', False):
        _monitoring_system.start_monitoring()

    logger.info("データベース監視システム初期化完了")
    return _monitoring_system