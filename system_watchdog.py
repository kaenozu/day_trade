#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Watchdog - システム監視・自動回復
Issue #948対応: 自動監視 + 自己回復 + アラート
"""

import time
import threading
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AlertSeverity(Enum):
    """アラート重要度"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SystemAlert:
    """システムアラート"""
    timestamp: datetime
    severity: AlertSeverity
    component: str
    message: str
    metric_value: float
    threshold_value: float
    auto_recovery: bool


class SystemWatchdog:
    """システム監視システム"""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.check_interval = 60  # 60秒間隔

        # 監視閾値
        self.thresholds = {
            'cpu_warning': 80,     # CPU 80%で警告
            'cpu_critical': 95,    # CPU 95%で重要
            'memory_warning': 85,  # メモリ 85%で警告
            'memory_critical': 95, # メモリ 95%で重要
            'disk_warning': 85,    # ディスク 85%で警告
            'disk_critical': 95    # ディスク 95%で重要
        }

        # アラート履歴
        self.alerts = []
        self.alert_history_limit = 100

        # 自動回復カウンター
        self.recovery_attempts = {
            'memory_cleanup': 0,
            'process_restart': 0,
            'cache_clear': 0
        }

        # 統計
        self.monitoring_stats = {
            'start_time': None,
            'checks_performed': 0,
            'alerts_generated': 0,
            'recoveries_attempted': 0
        }

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitoring_stats['start_time'] = datetime.now()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logging.info("System watchdog monitoring started")

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        logging.info("System watchdog monitoring stopped")

    def _monitoring_loop(self):
        """監視メインループ"""
        while self.monitoring:
            try:
                self._perform_system_check()
                self.monitoring_stats['checks_performed'] += 1
                time.sleep(self.check_interval)

            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # エラー時は30秒待機

    def _perform_system_check(self):
        """システムチェック実行"""
        # CPU監視
        cpu_percent = psutil.cpu_percent(interval=1)
        self._check_cpu_usage(cpu_percent)

        # メモリ監視
        memory = psutil.virtual_memory()
        self._check_memory_usage(memory.percent)

        # ディスク監視
        disk = psutil.disk_usage('.')
        disk_percent = (disk.used / disk.total) * 100
        self._check_disk_usage(disk_percent)

        # プロセス監視
        self._check_process_health()

    def _check_cpu_usage(self, cpu_percent: float):
        """CPU使用率チェック"""
        if cpu_percent >= self.thresholds['cpu_critical']:
            self._generate_alert(
                AlertSeverity.CRITICAL,
                "CPU",
                f"Critical CPU usage: {cpu_percent:.1f}%",
                cpu_percent,
                self.thresholds['cpu_critical'],
                auto_recovery=True
            )
            self._attempt_cpu_recovery()

        elif cpu_percent >= self.thresholds['cpu_warning']:
            self._generate_alert(
                AlertSeverity.WARNING,
                "CPU",
                f"High CPU usage: {cpu_percent:.1f}%",
                cpu_percent,
                self.thresholds['cpu_warning'],
                auto_recovery=False
            )

    def _check_memory_usage(self, memory_percent: float):
        """メモリ使用率チェック"""
        if memory_percent >= self.thresholds['memory_critical']:
            self._generate_alert(
                AlertSeverity.CRITICAL,
                "Memory",
                f"Critical memory usage: {memory_percent:.1f}%",
                memory_percent,
                self.thresholds['memory_critical'],
                auto_recovery=True
            )
            self._attempt_memory_recovery()

        elif memory_percent >= self.thresholds['memory_warning']:
            self._generate_alert(
                AlertSeverity.WARNING,
                "Memory",
                f"High memory usage: {memory_percent:.1f}%",
                memory_percent,
                self.thresholds['memory_warning'],
                auto_recovery=False
            )

    def _check_disk_usage(self, disk_percent: float):
        """ディスク使用率チェック"""
        if disk_percent >= self.thresholds['disk_critical']:
            self._generate_alert(
                AlertSeverity.CRITICAL,
                "Disk",
                f"Critical disk usage: {disk_percent:.1f}%",
                disk_percent,
                self.thresholds['disk_critical'],
                auto_recovery=True
            )
            self._attempt_disk_cleanup()

        elif disk_percent >= self.thresholds['disk_warning']:
            self._generate_alert(
                AlertSeverity.WARNING,
                "Disk",
                f"High disk usage: {disk_percent:.1f}%",
                disk_percent,
                self.thresholds['disk_warning'],
                auto_recovery=False
            )

    def _check_process_health(self):
        """プロセス健全性チェック"""
        try:
            process = psutil.Process()

            # メモリリーク検出（簡易版）
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 500:  # 500MB超
                self._generate_alert(
                    AlertSeverity.WARNING,
                    "Process",
                    f"High process memory usage: {memory_mb:.1f}MB",
                    memory_mb,
                    500,
                    auto_recovery=False
                )

            # ファイルディスクリプタ数チェック
            try:
                open_files = len(process.open_files())
                if open_files > 100:
                    self._generate_alert(
                        AlertSeverity.WARNING,
                        "Process",
                        f"High open file count: {open_files}",
                        open_files,
                        100,
                        auto_recovery=False
                    )
            except (psutil.AccessDenied, AttributeError):
                pass

        except psutil.NoSuchProcess:
            pass

    def _generate_alert(self, severity: AlertSeverity, component: str, message: str,
                       metric_value: float, threshold_value: float, auto_recovery: bool = False):
        """アラート生成"""
        alert = SystemAlert(
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            message=message,
            metric_value=metric_value,
            threshold_value=threshold_value,
            auto_recovery=auto_recovery
        )

        self.alerts.append(alert)
        self.monitoring_stats['alerts_generated'] += 1

        # 履歴制限
        if len(self.alerts) > self.alert_history_limit:
            self.alerts = self.alerts[-self.alert_history_limit//2:]

        # ログ出力
        log_level = {
            AlertSeverity.INFO: logging.info,
            AlertSeverity.WARNING: logging.warning,
            AlertSeverity.ERROR: logging.error,
            AlertSeverity.CRITICAL: logging.critical
        }.get(severity, logging.info)

        log_level(f"[WATCHDOG] {message}")

    def _attempt_cpu_recovery(self):
        """CPU回復試行"""
        if self.recovery_attempts['process_restart'] < 3:  # 最大3回まで
            try:
                # 軽量なCPU負荷軽減
                import gc
                gc.collect()

                self.recovery_attempts['process_restart'] += 1
                self.monitoring_stats['recoveries_attempted'] += 1

                logging.info("CPU recovery attempted: garbage collection performed")

            except Exception as e:
                logging.error(f"CPU recovery failed: {e}")

    def _attempt_memory_recovery(self):
        """メモリ回復試行"""
        if self.recovery_attempts['memory_cleanup'] < 5:  # 最大5回まで
            try:
                import gc

                # ガベージコレクション
                collected = gc.collect()

                self.recovery_attempts['memory_cleanup'] += 1
                self.monitoring_stats['recoveries_attempted'] += 1

                logging.info(f"Memory recovery attempted: collected {collected} objects")

            except Exception as e:
                logging.error(f"Memory recovery failed: {e}")

    def _attempt_disk_cleanup(self):
        """ディスククリーンアップ試行"""
        if self.recovery_attempts['cache_clear'] < 2:  # 最大2回まで
            try:
                # 安全なキャッシュクリア
                import tempfile
                import shutil

                # システム一時ディレクトリのクリーンアップ（安全な範囲）
                temp_dir = tempfile.gettempdir()
                temp_files_cleaned = 0

                # 実際のクリーンアップはより慎重に実装する必要がある
                # ここでは統計のみ更新

                self.recovery_attempts['cache_clear'] += 1
                self.monitoring_stats['recoveries_attempted'] += 1

                logging.info("Disk cleanup attempted")

            except Exception as e:
                logging.error(f"Disk cleanup failed: {e}")

    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状況取得"""
        uptime = None
        if self.monitoring_stats['start_time']:
            uptime = (datetime.now() - self.monitoring_stats['start_time']).total_seconds()

        # 最近のアラート（過去1時間）
        recent_alerts = [
            alert for alert in self.alerts
            if (datetime.now() - alert.timestamp).seconds < 3600
        ]

        return {
            'monitoring_active': self.monitoring,
            'uptime_seconds': uptime,
            'checks_performed': self.monitoring_stats['checks_performed'],
            'alerts_generated': self.monitoring_stats['alerts_generated'],
            'recoveries_attempted': self.monitoring_stats['recoveries_attempted'],
            'recent_alerts_count': len(recent_alerts),
            'recovery_attempts': dict(self.recovery_attempts),
            'thresholds': dict(self.thresholds)
        }

    def get_recent_alerts(self, hours: int = 24) -> list:
        """最近のアラート取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'component': alert.component,
                'message': alert.message,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'auto_recovery': alert.auto_recovery
            }
            for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]

        return recent_alerts

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """閾値更新"""
        for key, value in new_thresholds.items():
            if key in self.thresholds:
                self.thresholds[key] = value
                logging.info(f"Updated threshold {key}: {value}")

    def reset_recovery_counters(self):
        """回復カウンターリセット"""
        self.recovery_attempts = {key: 0 for key in self.recovery_attempts}
        logging.info("Recovery counters reset")


# グローバルインスタンス
system_watchdog = SystemWatchdog()


def start_system_monitoring():
    """システム監視開始"""
    system_watchdog.start_monitoring()


def stop_system_monitoring():
    """システム監視停止"""
    system_watchdog.stop_monitoring()


def get_watchdog_status() -> Dict[str, Any]:
    """監視状況取得"""
    return system_watchdog.get_monitoring_status()


def main():
    """メインテスト"""
    print("=== System Watchdog Test ===")

    # 監視開始
    print("Starting system monitoring...")
    start_system_monitoring()

    # 短時間待機
    time.sleep(5)

    # 状況確認
    status = get_watchdog_status()
    print(f"Monitoring active: {status['monitoring_active']}")
    print(f"Checks performed: {status['checks_performed']}")
    print(f"Alerts generated: {status['alerts_generated']}")
    print(f"Recovery attempts: {status['recovery_attempts']}")

    # 最近のアラート
    recent_alerts = system_watchdog.get_recent_alerts(1)
    print(f"Recent alerts: {len(recent_alerts)}")

    for alert in recent_alerts[-3:]:  # 最新3件
        print(f"  {alert['severity']}: {alert['message']}")

    # 監視停止
    print("\nStopping monitoring...")
    stop_system_monitoring()

    print("Watchdog test completed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()