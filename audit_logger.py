#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit Logger - 監査・ログ機能強化システム
Issue #933 Phase 4対応: セキュリティ監査ログとエラー追跡システム
"""

import logging
import json
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import hashlib
import uuid

try:
    from data_persistence import data_persistence
    HAS_DATA_PERSISTENCE = True
except ImportError:
    data_persistence = None
    HAS_DATA_PERSISTENCE = False

try:
    from version import get_version_info
    HAS_VERSION_INFO = True
except ImportError:
    HAS_VERSION_INFO = False


class SecurityEventType(Enum):
    """セキュリティイベントタイプ"""
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    API_RATE_LIMIT = "api_rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_ANOMALY = "system_anomaly"


class LogLevel(Enum):
    """カスタムログレベル"""
    AUDIT = "AUDIT"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    BUSINESS = "BUSINESS"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditLogger:
    """拡張監査ログシステム"""

    def __init__(self, log_dir: str = "logs", enable_console: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_console = enable_console
        self.session_id = str(uuid.uuid4())

        # ロック
        self._lock = threading.Lock()

        # ログ統計
        self.log_counts = {level.value: 0 for level in LogLevel}
        self.security_events = []
        self.performance_alerts = []

        # ログ設定
        self._setup_loggers()

        # 自動アラートシステム
        self._alert_thread = None
        self._stop_alerts = False
        self.start_alert_monitoring()

    def _setup_loggers(self):
        """ログシステムの設定"""
        # カスタムフォーマッター
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-12s | SESSION:%(session_id)s | %(message)s'
        )

        # 監査ログ
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)

        audit_handler = logging.FileHandler(self.log_dir / 'audit.log', encoding='utf-8')
        audit_handler.setFormatter(formatter)
        self.audit_logger.addHandler(audit_handler)

        # セキュリティログ
        self.security_logger = logging.getLogger('security')
        self.security_logger.setLevel(logging.INFO)

        security_handler = logging.FileHandler(self.log_dir / 'security.log', encoding='utf-8')
        security_handler.setFormatter(formatter)
        self.security_logger.addHandler(security_handler)

        # パフォーマンスログ
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)

        perf_handler = logging.FileHandler(self.log_dir / 'performance.log', encoding='utf-8')
        perf_handler.setFormatter(formatter)
        self.performance_logger.addHandler(perf_handler)

        # エラーログ
        self.error_logger = logging.getLogger('errors')
        self.error_logger.setLevel(logging.ERROR)

        error_handler = logging.FileHandler(self.log_dir / 'errors.log', encoding='utf-8')
        error_handler.setFormatter(formatter)
        self.error_logger.addHandler(error_handler)

        # コンソール出力（オプション）
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            for logger in [self.audit_logger, self.security_logger,
                          self.performance_logger, self.error_logger]:
                logger.addHandler(console_handler)

    def log_analysis_request(self, symbol: str, source: str, user_id: str = None,
                           request_data: Dict = None):
        """分析リクエストの監査ログ"""
        with self._lock:
            log_data = {
                'event_type': 'analysis_request',
                'symbol': symbol,
                'source': source,
                'user_id': user_id or 'anonymous',
                'session_id': self.session_id,
                'request_data': request_data or {},
                'timestamp': datetime.now().isoformat(),
                'client_hash': self._generate_client_hash(source, user_id)
            }

            self.audit_logger.info(
                f"ANALYSIS_REQUEST | Symbol:{symbol} | Source:{source} | User:{user_id}",
                extra={'session_id': self.session_id}
            )

            # データベースに保存
            if HAS_DATA_PERSISTENCE:
                data_persistence.save_analysis_result(
                    symbol=symbol,
                    analysis_type='audit_request',
                    duration_ms=0,
                    result_data=log_data,
                    session_id=self.session_id
                )

            self.log_counts[LogLevel.AUDIT.value] += 1

    def log_security_event(self, event_type: SecurityEventType, details: Dict[str, Any],
                          severity: str = "medium", user_id: str = None):
        """セキュリティイベントのログ"""
        with self._lock:
            event_data = {
                'event_type': event_type.value,
                'severity': severity,
                'user_id': user_id or 'system',
                'session_id': self.session_id,
                'details': details,
                'timestamp': datetime.now().isoformat(),
                'event_id': str(uuid.uuid4())
            }

            self.security_logger.warning(
                f"SECURITY_EVENT | {event_type.value.upper()} | Severity:{severity} | User:{user_id}",
                extra={'session_id': self.session_id}
            )

            # 重要なセキュリティイベントをメモリに保存
            self.security_events.append(event_data)
            if len(self.security_events) > 1000:  # 最新1000件のみ保持
                self.security_events.pop(0)

            # データベースに保存
            if HAS_DATA_PERSISTENCE:
                data_persistence.save_error_log(
                    error_type=f"security_{event_type.value}",
                    error_message=f"Security event: {event_type.value}",
                    context_data=event_data,
                    session_id=self.session_id
                )

            self.log_counts[LogLevel.SECURITY.value] += 1

            # 緊急レベルのイベントは即座にアラート
            if severity == "critical":
                self._trigger_immediate_alert(event_data)

    def log_performance_anomaly(self, metric: str, value: float, threshold: float,
                              context: Dict[str, Any] = None):
        """パフォーマンス異常のログ"""
        with self._lock:
            anomaly_data = {
                'metric': metric,
                'value': value,
                'threshold': threshold,
                'severity': 'high' if value > threshold * 2 else 'medium',
                'context': context or {},
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'anomaly_id': str(uuid.uuid4())
            }

            self.performance_logger.warning(
                f"PERFORMANCE_ANOMALY | {metric}:{value} > threshold:{threshold}",
                extra={'session_id': self.session_id}
            )

            # パフォーマンスアラートをメモリに保存
            self.performance_alerts.append(anomaly_data)
            if len(self.performance_alerts) > 500:  # 最新500件のみ保持
                self.performance_alerts.pop(0)

            # データベースに保存
            if HAS_DATA_PERSISTENCE:
                data_persistence.save_error_log(
                    error_type="performance_anomaly",
                    error_message=f"Performance anomaly: {metric} = {value} (threshold: {threshold})",
                    context_data=anomaly_data,
                    session_id=self.session_id
                )

            self.log_counts[LogLevel.PERFORMANCE.value] += 1

    def log_business_event(self, event_type: str, details: Dict[str, Any],
                          user_id: str = None):
        """ビジネスロジック関連のイベントログ"""
        with self._lock:
            event_data = {
                'event_type': event_type,
                'user_id': user_id or 'system',
                'session_id': self.session_id,
                'details': details,
                'timestamp': datetime.now().isoformat(),
                'business_event_id': str(uuid.uuid4())
            }

            self.audit_logger.info(
                f"BUSINESS_EVENT | {event_type} | User:{user_id}",
                extra={'session_id': self.session_id}
            )

            # データベースに保存
            if HAS_DATA_PERSISTENCE:
                data_persistence.save_analysis_result(
                    symbol='BUSINESS',
                    analysis_type=event_type,
                    duration_ms=0,
                    result_data=event_data,
                    session_id=self.session_id
                )

            self.log_counts[LogLevel.BUSINESS.value] += 1

    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None,
                             user_id: str = None):
        """コンテキスト付きエラーログ"""
        with self._lock:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc(),
                'context': context or {},
                'user_id': user_id or 'system',
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'error_id': str(uuid.uuid4())
            }

            self.error_logger.error(
                f"ERROR_WITH_CONTEXT | {type(error).__name__}: {error}",
                extra={'session_id': self.session_id}
            )

            # データベースに保存
            if HAS_DATA_PERSISTENCE:
                data_persistence.save_error_log(
                    error_type=type(error).__name__,
                    error_message=str(error),
                    stack_trace=traceback.format_exc(),
                    context_data=error_data,
                    session_id=self.session_id
                )

            self.log_counts[LogLevel.ERROR.value] += 1

    def _generate_client_hash(self, source: str, user_id: str = None) -> str:
        """クライアント識別用のハッシュ生成"""
        client_string = f"{source}_{user_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d')}"
        return hashlib.md5(client_string.encode()).hexdigest()[:8]

    def _trigger_immediate_alert(self, event_data: Dict[str, Any]):
        """緊急アラートの送信"""
        alert_message = f"""
        🚨 CRITICAL SECURITY ALERT 🚨
        Event: {event_data['event_type']}
        Time: {event_data['timestamp']}
        Session: {event_data['session_id']}
        Details: {event_data['details']}
        """

        print(alert_message)
        # 実際のプロダクションでは、メール送信やSlack通知などを実装

    def start_alert_monitoring(self):
        """アラート監視を開始"""
        if self._alert_thread is None or not self._alert_thread.is_alive():
            self._stop_alerts = False
            self._alert_thread = threading.Thread(target=self._alert_monitoring_loop)
            self._alert_thread.daemon = True
            self._alert_thread.start()

    def stop_alert_monitoring(self):
        """アラート監視を停止"""
        self._stop_alerts = True
        if self._alert_thread:
            self._alert_thread.join(timeout=1)

    def _alert_monitoring_loop(self):
        """アラート監視ループ"""
        import time

        while not self._stop_alerts:
            try:
                # 過去1時間のセキュリティイベント数をチェック
                recent_events = [
                    event for event in self.security_events
                    if datetime.fromisoformat(event['timestamp']) > datetime.now() - timedelta(hours=1)
                ]

                if len(recent_events) > 10:  # 1時間に10件以上のセキュリティイベント
                    self._trigger_immediate_alert({
                        'event_type': 'excessive_security_events',
                        'count': len(recent_events),
                        'timestamp': datetime.now().isoformat(),
                        'details': {'recent_events': len(recent_events)}
                    })

                # パフォーマンス異常の集計チェック
                recent_performance_alerts = [
                    alert for alert in self.performance_alerts
                    if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(minutes=30)
                ]

                if len(recent_performance_alerts) > 5:  # 30分間に5件以上のパフォーマンス異常
                    self._trigger_immediate_alert({
                        'event_type': 'performance_degradation',
                        'count': len(recent_performance_alerts),
                        'timestamp': datetime.now().isoformat(),
                        'details': {'recent_alerts': len(recent_performance_alerts)}
                    })

            except Exception as e:
                self.log_error_with_context(e, {'context': 'alert_monitoring_loop'})

            time.sleep(300)  # 5分間隔でチェック

    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """監査サマリーを取得"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # 最近のセキュリティイベント
            recent_security = [
                event for event in self.security_events
                if datetime.fromisoformat(event['timestamp']) > cutoff_time
            ]

            # 最近のパフォーマンス異常
            recent_performance = [
                alert for alert in self.performance_alerts
                if datetime.fromisoformat(alert['timestamp']) > cutoff_time
            ]

            # セキュリティイベント種別の集計
            security_by_type = {}
            for event in recent_security:
                event_type = event['event_type']
                security_by_type[event_type] = security_by_type.get(event_type, 0) + 1

            # パフォーマンス異常メトリックの集計
            performance_by_metric = {}
            for alert in recent_performance:
                metric = alert['metric']
                performance_by_metric[metric] = performance_by_metric.get(metric, 0) + 1

            return {
                'period_hours': hours,
                'session_id': self.session_id,
                'log_counts': dict(self.log_counts),
                'security_events': {
                    'total': len(recent_security),
                    'by_type': security_by_type,
                    'recent_events': recent_security[-10:]  # 最新10件
                },
                'performance_alerts': {
                    'total': len(recent_performance),
                    'by_metric': performance_by_metric,
                    'recent_alerts': recent_performance[-10:]  # 最新10件
                },
                'system_info': {
                    'version': get_version_info() if HAS_VERSION_INFO else {'version': 'unknown'},
                    'data_persistence_enabled': HAS_DATA_PERSISTENCE,
                    'log_directory': str(self.log_dir)
                }
            }

    def export_audit_report(self, hours: int = 24, format: str = 'json') -> str:
        """監査レポートのエクスポート"""
        summary = self.get_audit_summary(hours)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == 'json':
            filename = self.log_dir / f"audit_report_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        elif format.lower() == 'txt':
            filename = self.log_dir / f"audit_report_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Day Trade Personal - 監査レポート\n")
                f.write(f"生成日時: {datetime.now().isoformat()}\n")
                f.write(f"期間: 過去{hours}時間\n")
                f.write(f"セッションID: {summary['session_id']}\n")
                f.write(f"\n=== ログ統計 ===\n")
                for level, count in summary['log_counts'].items():
                    f.write(f"{level}: {count}件\n")

                f.write(f"\n=== セキュリティイベント ===\n")
                f.write(f"総計: {summary['security_events']['total']}件\n")
                for event_type, count in summary['security_events']['by_type'].items():
                    f.write(f"{event_type}: {count}件\n")

                f.write(f"\n=== パフォーマンス異常 ===\n")
                f.write(f"総計: {summary['performance_alerts']['total']}件\n")
                for metric, count in summary['performance_alerts']['by_metric'].items():
                    f.write(f"{metric}: {count}件\n")

        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(filename)

    def cleanup_old_logs(self, days: int = 30):
        """古いログファイルのクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=days)

        log_files = [
            'audit.log', 'security.log', 'performance.log', 'errors.log'
        ]

        for log_file in log_files:
            log_path = self.log_dir / log_file
            if log_path.exists():
                # ログローテーション実装
                if log_path.stat().st_size > 10 * 1024 * 1024:  # 10MB以上
                    backup_path = self.log_dir / f"{log_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    log_path.rename(backup_path)
                    print(f"[ログローテーション] {log_file} -> {backup_path.name}")


# グローバルインスタンス
audit_logger = AuditLogger()


def audit_decorator(event_type: str = None):
    """監査ログデコレーター"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            event_name = event_type or func.__name__

            try:
                # 実行前ログ
                audit_logger.log_business_event(
                    f"{event_name}_start",
                    {'function': func.__name__, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                )

                result = func(*args, **kwargs)

                # 成功ログ
                duration = (datetime.now() - start_time).total_seconds()
                audit_logger.log_business_event(
                    f"{event_name}_success",
                    {'function': func.__name__, 'duration_seconds': duration}
                )

                return result

            except Exception as e:
                # エラーログ
                duration = (datetime.now() - start_time).total_seconds()
                audit_logger.log_error_with_context(
                    e,
                    {'function': func.__name__, 'duration_seconds': duration, 'event_type': event_name}
                )
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # テスト実行
    logger = AuditLogger()

    # テストイベント
    logger.log_analysis_request("7203", "web_ui", "test_user")
    logger.log_security_event(
        SecurityEventType.SUSPICIOUS_ACTIVITY,
        {"ip": "192.168.1.1", "user_agent": "test_agent"},
        severity="medium",
        user_id="test_user"
    )
    logger.log_performance_anomaly("response_time_ms", 1500.0, 500.0)
    logger.log_business_event("stock_analysis", {"symbol": "7203", "result": "BUY"})

    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        logger.log_error_with_context(e, {"test": "error logging"})

    # サマリー表示
    print("=== 監査サマリー ===")
    summary = logger.get_audit_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    # レポートエクスポート
    report_file = logger.export_audit_report(format='txt')
    print(f"\n監査レポートを出力: {report_file}")