"""
データベース監視ダッシュボード

リアルタイム監視データの可視化、レポート生成機能
WebUI対応、JSON API、シンプルなHTMLレポート生成
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading
from collections import defaultdict

from day_trade.core.logging.unified_logging_system import get_logger
from .monitoring_system import DatabaseMonitoringSystem, get_monitoring_system
from .backup_manager import DatabaseBackupManager, get_backup_manager
from .restore_manager import DatabaseRestoreManager, get_restore_manager

logger = get_logger(__name__)


class DatabaseDashboard:
    """データベース監視ダッシュボード"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dashboard_config = config.get('dashboard', {})

        # ダッシュボード設定
        self.enabled = self.dashboard_config.get('enabled', True)
        self.refresh_interval = self.dashboard_config.get('refresh_interval_seconds', 10)
        self.report_path = Path(self.dashboard_config.get('report_path', './reports'))

        # レポート設定
        self.auto_generate_reports = self.dashboard_config.get('auto_generate_reports', True)
        self.report_interval_hours = self.dashboard_config.get('report_interval_hours', 24)

        # データ保存
        self.dashboard_data: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None

        # レポート生成スレッド
        self._report_thread: Optional[threading.Thread] = None
        self._report_running = False

        # レポートディレクトリ作成
        self.report_path.mkdir(parents=True, exist_ok=True)

    def start_dashboard(self) -> None:
        """ダッシュボード開始"""
        if not self.enabled:
            logger.info("ダッシュボードが無効のため開始しません")
            return

        logger.info("データベースダッシュボード開始")

        # 自動レポート生成開始
        if self.auto_generate_reports:
            self._start_report_generation()

    def stop_dashboard(self) -> None:
        """ダッシュボード停止"""
        self._report_running = False

        if self._report_thread and self._report_thread.is_alive():
            logger.info("レポート生成停止中...")
            self._report_thread.join(timeout=10)

        logger.info("データベースダッシュボード停止")

    def _start_report_generation(self) -> None:
        """自動レポート生成開始"""
        self._report_running = True
        self._report_thread = threading.Thread(target=self._report_generation_loop, daemon=True)
        self._report_thread.start()
        logger.info(f"自動レポート生成開始: {self.report_interval_hours}時間間隔")

    def _report_generation_loop(self) -> None:
        """レポート生成ループ"""
        logger.info("自動レポート生成ループ開始")

        while self._report_running:
            try:
                # 日次レポート生成
                self.generate_daily_report()

                # 次の実行まで待機
                time.sleep(self.report_interval_hours * 3600)

            except Exception as e:
                logger.error(f"自動レポート生成エラー: {e}")
                time.sleep(3600)  # エラー時は1時間待機

        logger.info("自動レポート生成ループ終了")

    def refresh_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ更新"""
        try:
            current_time = datetime.now()

            # 各システムからデータ収集
            monitoring_system = get_monitoring_system()
            backup_manager = get_backup_manager()
            restore_manager = get_restore_manager()

            # システム状態
            system_status = self._get_system_status(monitoring_system, backup_manager, restore_manager)

            # メトリクス情報
            metrics_data = self._get_metrics_data(monitoring_system)

            # アラート情報
            alert_data = self._get_alert_data(monitoring_system)

            # バックアップ情報
            backup_data = self._get_backup_data(backup_manager)

            # 復元情報
            restore_data = self._get_restore_data(restore_manager)

            # パフォーマンス統計
            performance_stats = self._get_performance_statistics(monitoring_system)

            self.dashboard_data = {
                'timestamp': current_time.isoformat(),
                'system_status': system_status,
                'metrics': metrics_data,
                'alerts': alert_data,
                'backups': backup_data,
                'restores': restore_data,
                'performance': performance_stats
            }

            self.last_update = current_time

            logger.debug("ダッシュボードデータ更新完了")

            return self.dashboard_data

        except Exception as e:
            logger.error(f"ダッシュボードデータ更新失敗: {e}")
            return {}

    def _get_system_status(
        self,
        monitoring_system: Optional[DatabaseMonitoringSystem],
        backup_manager: Optional[DatabaseBackupManager],
        restore_manager: Optional[DatabaseRestoreManager]
    ) -> Dict[str, Any]:
        """システム状態取得"""

        monitoring_status = "unknown"
        backup_status = "unknown"
        restore_status = "unknown"

        if monitoring_system:
            monitor_status = monitoring_system.get_monitoring_status()
            monitoring_status = "running" if monitor_status.get('running', False) else "stopped"

        if backup_manager:
            backup_stats = backup_manager.get_backup_statistics()
            backup_status = "enabled" if backup_stats.get('scheduler_running', False) else "disabled"

        restore_status = "available" if restore_manager else "unavailable"

        # 全体のヘルス状態判定
        overall_health = "healthy"
        if monitoring_status == "stopped":
            overall_health = "warning"
        if backup_status == "disabled":
            if overall_health == "healthy":
                overall_health = "warning"

        return {
            'overall_health': overall_health,
            'monitoring': {
                'status': monitoring_status,
                'enabled': monitoring_system is not None
            },
            'backup': {
                'status': backup_status,
                'enabled': backup_manager is not None
            },
            'restore': {
                'status': restore_status,
                'enabled': restore_manager is not None
            },
            'uptime_hours': self._calculate_uptime_hours()
        }

    def _get_metrics_data(self, monitoring_system: Optional[DatabaseMonitoringSystem]) -> Dict[str, Any]:
        """メトリクスデータ取得"""
        if not monitoring_system:
            return {}

        current_metrics = monitoring_system.get_current_metrics()
        metrics_history = monitoring_system.get_metrics_history(hours=1)

        # 時系列データの簡易統計
        history_stats = {}
        if metrics_history:
            for metric_name in ['cpu_usage', 'memory_usage_mb', 'connection_pool_usage']:
                values = [m.get(metric_name, 0) for m in metrics_history if metric_name in m]
                if values:
                    history_stats[metric_name] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1] if values else 0
                    }

        return {
            'current': current_metrics,
            'history_count': len(metrics_history),
            'history_stats': history_stats,
            'collection_interval': monitoring_system.interval_seconds
        }

    def _get_alert_data(self, monitoring_system: Optional[DatabaseMonitoringSystem]) -> Dict[str, Any]:
        """アラートデータ取得"""
        if not monitoring_system:
            return {}

        active_alerts = monitoring_system.get_active_alerts()
        alert_stats = monitoring_system.get_alert_statistics()

        # 重要度別集計
        severity_counts = defaultdict(int)
        for alert in active_alerts:
            severity_counts[alert.get('severity', 'unknown')] += 1

        return {
            'active_alerts': active_alerts,
            'active_count': len(active_alerts),
            'severity_breakdown': dict(severity_counts),
            'statistics': alert_stats
        }

    def _get_backup_data(self, backup_manager: Optional[DatabaseBackupManager]) -> Dict[str, Any]:
        """バックアップデータ取得"""
        if not backup_manager:
            return {}

        backup_list = backup_manager.list_backups()
        backup_stats = backup_manager.get_backup_statistics()

        # 最近のバックアップ（最大5件）
        recent_backups = backup_list[:5] if backup_list else []

        return {
            'statistics': backup_stats,
            'recent_backups': recent_backups,
            'total_backups': len(backup_list)
        }

    def _get_restore_data(self, restore_manager: Optional[DatabaseRestoreManager]) -> Dict[str, Any]:
        """復元データ取得"""
        if not restore_manager:
            return {}

        restore_operations = restore_manager.list_restore_operations()

        # 最近の復元操作（最大5件）
        recent_restores = restore_operations[:5] if restore_operations else []

        # ステータス別集計
        status_counts = defaultdict(int)
        for operation in restore_operations:
            status_counts[operation.get('status', 'unknown')] += 1

        return {
            'recent_operations': recent_restores,
            'total_operations': len(restore_operations),
            'status_breakdown': dict(status_counts)
        }

    def _get_performance_statistics(self, monitoring_system: Optional[DatabaseMonitoringSystem]) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        if not monitoring_system:
            return {}

        # 過去24時間のメトリクス
        metrics_24h = monitoring_system.get_metrics_history(hours=24)

        if not metrics_24h:
            return {}

        # パフォーマンス指標計算
        cpu_values = [m.get('cpu_usage', 0) for m in metrics_24h]
        memory_values = [m.get('memory_usage_mb', 0) for m in metrics_24h]
        connection_values = [m.get('connection_pool_usage', 0) for m in metrics_24h]

        performance_stats = {}

        if cpu_values:
            performance_stats['cpu_24h'] = {
                'avg': round(sum(cpu_values) / len(cpu_values), 2),
                'max': round(max(cpu_values), 2),
                'above_80_percent': sum(1 for v in cpu_values if v > 80) / len(cpu_values) * 100
            }

        if memory_values:
            performance_stats['memory_24h'] = {
                'avg_mb': round(sum(memory_values) / len(memory_values), 2),
                'max_mb': round(max(memory_values), 2),
                'growth_rate': self._calculate_growth_rate(memory_values)
            }

        if connection_values:
            performance_stats['connections_24h'] = {
                'avg_usage': round(sum(connection_values) / len(connection_values), 4),
                'max_usage': round(max(connection_values), 4),
                'peak_times': self._find_peak_usage_times(metrics_24h, 'connection_pool_usage')
            }

        return performance_stats

    def _calculate_uptime_hours(self) -> float:
        """稼働時間計算（簡易版）"""
        # 実際の実装では適切な稼働時間計算を行う
        if self.last_update:
            # ダッシュボード開始からの時間を概算
            return 24.0  # プレースホルダー
        return 0.0

    def _calculate_growth_rate(self, values: List[float]) -> float:
        """成長率計算"""
        if len(values) < 2:
            return 0.0

        start_value = values[0] if values[0] > 0 else 1
        end_value = values[-1]

        return round(((end_value - start_value) / start_value) * 100, 2)

    def _find_peak_usage_times(self, metrics: List[Dict[str, Any]], metric_name: str) -> List[str]:
        """ピーク使用時間特定"""
        if len(metrics) < 5:
            return []

        # 上位5つの高使用率時間を取得
        sorted_metrics = sorted(
            metrics,
            key=lambda x: x.get(metric_name, 0),
            reverse=True
        )

        peak_times = []
        for metric in sorted_metrics[:5]:
            timestamp = metric.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    peak_times.append(dt.strftime('%H:%M'))
                except:
                    pass

        return peak_times

    def generate_daily_report(self) -> Dict[str, Any]:
        """日次レポート生成"""
        try:
            report_date = datetime.now().strftime('%Y-%m-%d')

            # 最新データで更新
            dashboard_data = self.refresh_dashboard_data()

            # レポートデータ構築
            report_data = {
                'report_type': 'daily',
                'generated_at': datetime.now().isoformat(),
                'report_date': report_date,
                'summary': self._generate_daily_summary(dashboard_data),
                'details': dashboard_data
            }

            # JSONレポート保存
            json_file = self.report_path / f"daily_report_{report_date}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            # HTMLレポート生成
            html_file = self.report_path / f"daily_report_{report_date}.html"
            self._generate_html_report(report_data, html_file)

            logger.info(f"日次レポート生成完了: {json_file}")

            return {
                'status': 'success',
                'report_date': report_date,
                'json_file': str(json_file),
                'html_file': str(html_file),
                'data': report_data
            }

        except Exception as e:
            logger.error(f"日次レポート生成失敗: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _generate_daily_summary(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """日次サマリー生成"""
        system_status = dashboard_data.get('system_status', {})
        alerts = dashboard_data.get('alerts', {})
        backups = dashboard_data.get('backups', {})
        performance = dashboard_data.get('performance', {})

        # ヘルス状態サマリー
        health_summary = {
            'overall_health': system_status.get('overall_health', 'unknown'),
            'active_alerts': alerts.get('active_count', 0),
            'critical_alerts': alerts.get('severity_breakdown', {}).get('critical', 0),
            'warning_alerts': alerts.get('severity_breakdown', {}).get('warning', 0)
        }

        # バックアップサマリー
        backup_summary = {
            'total_backups': backups.get('statistics', {}).get('total_backups', 0),
            'backup_success_rate': backups.get('statistics', {}).get('success_rate', 0),
            'total_size_mb': backups.get('statistics', {}).get('total_size_mb', 0)
        }

        # パフォーマンスサマリー
        perf_summary = {}
        if performance.get('cpu_24h'):
            perf_summary['avg_cpu_usage'] = performance['cpu_24h'].get('avg', 0)
            perf_summary['max_cpu_usage'] = performance['cpu_24h'].get('max', 0)

        if performance.get('memory_24h'):
            perf_summary['avg_memory_mb'] = performance['memory_24h'].get('avg_mb', 0)
            perf_summary['max_memory_mb'] = performance['memory_24h'].get('max_mb', 0)

        return {
            'health': health_summary,
            'backups': backup_summary,
            'performance': perf_summary,
            'recommendations': self._generate_recommendations(dashboard_data)
        }

    def _generate_recommendations(self, dashboard_data: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        # アラートベースの推奨事項
        alerts = dashboard_data.get('alerts', {})
        if alerts.get('active_count', 0) > 0:
            recommendations.append("アクティブなアラートがあります。システムの確認をお勧めします。")

        # パフォーマンスベースの推奨事項
        performance = dashboard_data.get('performance', {})
        cpu_stats = performance.get('cpu_24h', {})

        if cpu_stats.get('avg', 0) > 70:
            recommendations.append("CPU使用率が高めです。負荷分散を検討してください。")

        if cpu_stats.get('above_80_percent', 0) > 20:
            recommendations.append("CPU使用率が80%を超える時間が多いです。スケールアップを検討してください。")

        # バックアップベースの推奨事項
        backups = dashboard_data.get('backups', {})
        success_rate = backups.get('statistics', {}).get('success_rate', 100)

        if success_rate < 95:
            recommendations.append("バックアップ成功率が低下しています。バックアップ設定を確認してください。")

        if not recommendations:
            recommendations.append("システムは正常に動作しています。")

        return recommendations

    def _generate_html_report(self, report_data: Dict[str, Any], output_file: Path) -> None:
        """HTMLレポート生成"""
        try:
            summary = report_data.get('summary', {})

            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>データベース監視 日次レポート - {report_data.get('report_date', '')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #007bff; }}
        .health-good {{ border-left-color: #28a745; }}
        .health-warning {{ border-left-color: #ffc107; }}
        .health-critical {{ border-left-color: #dc3545; }}
        .metric {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .metric-label {{ font-weight: bold; }}
        .recommendations {{ background: #e9ecef; padding: 20px; border-radius: 6px; margin: 20px 0; }}
        .recommendations ul {{ margin: 10px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>データベース監視 日次レポート</h1>
        <p><strong>レポート日付:</strong> {report_data.get('report_date', '')}</p>
        <p><strong>生成日時:</strong> {report_data.get('generated_at', '')}</p>

        <h2>システム概要</h2>
        <div class="summary-grid">
            <div class="summary-card health-{summary.get('health', {}).get('overall_health', 'unknown')}">
                <h3>システムヘルス</h3>
                <div class="metric">
                    <span class="metric-label">総合状態:</span>
                    <span>{summary.get('health', {}).get('overall_health', 'Unknown')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">アクティブアラート:</span>
                    <span>{summary.get('health', {}).get('active_alerts', 0)}件</span>
                </div>
            </div>

            <div class="summary-card">
                <h3>バックアップ状況</h3>
                <div class="metric">
                    <span class="metric-label">総バックアップ数:</span>
                    <span>{summary.get('backups', {}).get('total_backups', 0)}件</span>
                </div>
                <div class="metric">
                    <span class="metric-label">成功率:</span>
                    <span>{summary.get('backups', {}).get('backup_success_rate', 0):.1f}%</span>
                </div>
            </div>

            <div class="summary-card">
                <h3>パフォーマンス</h3>
                <div class="metric">
                    <span class="metric-label">平均CPU使用率:</span>
                    <span>{summary.get('performance', {}).get('avg_cpu_usage', 0):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">平均メモリ使用量:</span>
                    <span>{summary.get('performance', {}).get('avg_memory_mb', 0):.0f}MB</span>
                </div>
            </div>
        </div>

        <h2>推奨事項</h2>
        <div class="recommendations">
            <ul>
"""

            for recommendation in summary.get('recommendations', []):
                html_content += f"                <li>{recommendation}</li>\n"

            html_content += f"""
            </ul>
        </div>

        <div class="footer">
            <p>このレポートは Day Trading System の自動監視機能により生成されました。</p>
            <p>詳細な情報については、システム管理者にお問い合わせください。</p>
        </div>
    </div>
</body>
</html>
"""

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTMLレポート生成完了: {output_file}")

        except Exception as e:
            logger.error(f"HTMLレポート生成失敗: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ取得（API用）"""
        if not self.dashboard_data or not self.last_update:
            return self.refresh_dashboard_data()

        # データが古い場合は更新
        if datetime.now() - self.last_update > timedelta(seconds=self.refresh_interval):
            return self.refresh_dashboard_data()

        return self.dashboard_data

    def get_recent_reports(self, days: int = 7) -> List[Dict[str, Any]]:
        """最近のレポート一覧取得"""
        reports = []

        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            json_file = self.report_path / f"daily_report_{date}.json"

            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    reports.append({
                        'date': date,
                        'file_path': str(json_file),
                        'generated_at': report_data.get('generated_at', ''),
                        'summary': report_data.get('summary', {})
                    })
                except Exception as e:
                    logger.warning(f"レポートファイル読み込み失敗: {json_file} - {e}")

        return reports


# グローバルインスタンス管理
_dashboard: Optional[DatabaseDashboard] = None


def get_dashboard() -> Optional[DatabaseDashboard]:
    """ダッシュボード取得"""
    return _dashboard


def initialize_dashboard(config: Dict[str, Any]) -> DatabaseDashboard:
    """ダッシュボード初期化"""
    global _dashboard

    _dashboard = DatabaseDashboard(config)
    _dashboard.start_dashboard()

    logger.info("データベースダッシュボード初期化完了")
    return _dashboard