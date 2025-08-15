#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Performance Monitor - システムパフォーマンス監視
リアルタイムパフォーマンス監視と自動最適化の統合システム
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from performance_optimization_system import get_performance_system, PerformanceLevel
from enhanced_data_provider import get_data_provider
from fallback_notification_system import get_notification_system


class MonitoringMode(Enum):
    """監視モード"""
    PASSIVE = "passive"      # 監視のみ
    ACTIVE = "active"        # 自動最適化あり
    AGGRESSIVE = "aggressive" # 積極的最適化


@dataclass
class SystemHealth:
    """システム健全性"""
    overall_status: str
    performance_level: PerformanceLevel
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    last_optimization: Optional[datetime]
    uptime_hours: float


class SystemPerformanceMonitor:
    """システムパフォーマンス監視"""
    
    def __init__(self, monitoring_mode: MonitoringMode = MonitoringMode.ACTIVE):
        self.monitoring_mode = monitoring_mode
        self.performance_system = get_performance_system()
        self.data_provider = get_data_provider()
        self.notification_system = get_notification_system()
        
        # 監視設定
        self.monitoring_interval = 30  # 30秒間隔
        self.health_check_interval = 300  # 5分間隔
        
        # システム開始時間
        self.start_time = datetime.now()
        
        # 監視スレッド
        self.monitoring_thread = None
        self.health_thread = None
        self.running = False
        
        # 健全性履歴
        self.health_history = []
        
        from daytrade_logging import get_logger
        self.logger = get_logger("system_performance_monitor")
        
        self.logger.info(f"System Performance Monitor initialized in {monitoring_mode.value} mode")
    
    def start_monitoring(self):
        """監視開始"""
        if self.running:
            return
        
        self.running = True
        
        # パフォーマンス監視スレッド
        def performance_monitoring_loop():
            while self.running:
                try:
                    self._check_performance()
                    time.sleep(self.monitoring_interval)
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                    time.sleep(60)  # エラー時は1分待機
        
        # ヘルスチェックスレッド  
        def health_monitoring_loop():
            while self.running:
                try:
                    self._check_system_health()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(120)  # エラー時は2分待機
        
        self.monitoring_thread = threading.Thread(target=performance_monitoring_loop, daemon=True)
        self.health_thread = threading.Thread(target=health_monitoring_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.health_thread.start()
        
        # バックグラウンド最適化も開始
        if self.monitoring_mode in [MonitoringMode.ACTIVE, MonitoringMode.AGGRESSIVE]:
            self.performance_system.start_background_optimization()
        
        self.logger.info("System Performance Monitor started")
    
    def stop_monitoring(self):
        """監視停止"""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        if self.health_thread:
            self.health_thread.join(timeout=5)
        
        # バックグラウンド最適化も停止
        self.performance_system.stop_background_optimization()
        
        self.logger.info("System Performance Monitor stopped")
    
    def _check_performance(self):
        """パフォーマンスチェック"""
        metrics = self.performance_system.get_current_metrics()
        performance_level = self.performance_system.assess_performance_level(metrics)
        
        # 監視モードに応じた対応
        if self.monitoring_mode == MonitoringMode.AGGRESSIVE:
            # 積極的最適化
            if performance_level in [PerformanceLevel.DEGRADED, PerformanceLevel.CRITICAL]:
                self.performance_system.auto_optimize()
        elif self.monitoring_mode == MonitoringMode.ACTIVE:
            # クリティカル時のみ最適化
            if performance_level == PerformanceLevel.CRITICAL:
                self.performance_system.auto_optimize()
        
        # ログ出力（レベルに応じて）
        if performance_level == PerformanceLevel.CRITICAL:
            self.logger.error(f"CRITICAL performance: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
        elif performance_level == PerformanceLevel.DEGRADED:
            self.logger.warning(f"DEGRADED performance: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%")
    
    def _check_system_health(self):
        """システム健全性チェック"""
        try:
            metrics = self.performance_system.get_current_metrics()
            performance_level = self.performance_system.assess_performance_level(metrics)
            
            critical_issues = []
            warnings = []
            recommendations = []
            
            # CPU使用率チェック
            if metrics.cpu_percent > 90:
                critical_issues.append(f"CPU使用率が危険レベル: {metrics.cpu_percent:.1f}%")
            elif metrics.cpu_percent > 70:
                warnings.append(f"CPU使用率が高い: {metrics.cpu_percent:.1f}%")
                recommendations.append("CPU負荷の高い処理を見直してください")
            
            # メモリ使用率チェック
            if metrics.memory_percent > 90:
                critical_issues.append(f"メモリ使用率が危険レベル: {metrics.memory_percent:.1f}%")
            elif metrics.memory_percent > 80:
                warnings.append(f"メモリ使用率が高い: {metrics.memory_percent:.1f}%")
                recommendations.append("メモリクリーンアップを実行してください")
            
            # レスポンス時間チェック
            if metrics.response_time_ms > 2000:
                critical_issues.append(f"レスポンス時間が遅い: {metrics.response_time_ms:.1f}ms")
            elif metrics.response_time_ms > 1000:
                warnings.append(f"レスポンス時間が遅い: {metrics.response_time_ms:.1f}ms")
                recommendations.append("処理の最適化を検討してください")
            
            # キャッシュヒット率チェック
            if metrics.cache_hit_rate < 0.5:
                warnings.append(f"キャッシュヒット率が低い: {metrics.cache_hit_rate:.1%}")
                recommendations.append("キャッシュ戦略を見直してください")
            
            # データプロバイダー状態チェック
            try:
                provider_status = self.data_provider.get_provider_status()
                failed_providers = [name for name, status in provider_status.items() 
                                  if status.get('circuit_open', False)]
                
                if failed_providers:
                    critical_issues.append(f"データプロバイダー障害: {', '.join(failed_providers)}")
            except Exception as e:
                warnings.append(f"データプロバイダー状態確認失敗: {e}")
            
            # 通知システム状態チェック
            try:
                notification_summary = self.notification_system.get_session_summary()
                if notification_summary['dummy_data_count'] > 0:
                    warnings.append(f"ダミーデータ使用中: {notification_summary['dummy_data_count']}件")
                    recommendations.append("データ品質を確認してください")
            except Exception as e:
                warnings.append(f"通知システム状態確認失敗: {e}")
            
            # 全体ステータス判定
            if critical_issues:
                overall_status = "CRITICAL"
            elif warnings:
                overall_status = "WARNING"
            else:
                overall_status = "HEALTHY"
            
            # 最後の最適化時間
            last_optimization = getattr(self.performance_system, 'last_optimization', None)
            
            # システム稼働時間
            uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            
            # システム健全性を作成
            health = SystemHealth(
                overall_status=overall_status,
                performance_level=performance_level,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                last_optimization=last_optimization,
                uptime_hours=uptime_hours
            )
            
            # 履歴に追加
            self.health_history.append({
                'timestamp': datetime.now(),
                'health': health
            })
            
            # 履歴の制限（24時間分）
            if len(self.health_history) > 288:  # 5分間隔の24時間分
                self.health_history = self.health_history[-288:]
            
            # ログ出力
            if overall_status == "CRITICAL":
                self.logger.error(f"System health CRITICAL: {len(critical_issues)} critical issues")
            elif overall_status == "WARNING":
                self.logger.warning(f"System health WARNING: {len(warnings)} warnings")
            else:
                self.logger.info(f"System health HEALTHY (uptime: {uptime_hours:.1f}h)")
        
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def get_current_health(self) -> SystemHealth:
        """現在のシステム健全性を取得"""
        if self.health_history:
            return self.health_history[-1]['health']
        else:
            # 初回チェック
            self._check_system_health()
            return self.health_history[-1]['health'] if self.health_history else SystemHealth(
                overall_status="UNKNOWN",
                performance_level=PerformanceLevel.UNKNOWN,
                critical_issues=[],
                warnings=["健全性チェック未実行"],
                recommendations=["システム監視を開始してください"],
                last_optimization=None,
                uptime_hours=0.0
            )
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """監視レポートを生成"""
        current_health = self.get_current_health()
        performance_report = self.performance_system.get_performance_report()
        
        # 健全性履歴の統計
        recent_health = [h['health'] for h in self.health_history[-12:]]  # 最近1時間分
        
        critical_count = sum(1 for h in recent_health if h.overall_status == "CRITICAL")
        warning_count = sum(1 for h in recent_health if h.overall_status == "WARNING")
        healthy_count = sum(1 for h in recent_health if h.overall_status == "HEALTHY")
        
        return {
            'monitoring_status': {
                'mode': self.monitoring_mode.value,
                'running': self.running,
                'uptime_hours': current_health.uptime_hours,
                'monitoring_interval_seconds': self.monitoring_interval
            },
            'current_health': {
                'overall_status': current_health.overall_status,
                'performance_level': current_health.performance_level.value,
                'critical_issues_count': len(current_health.critical_issues),
                'warnings_count': len(current_health.warnings),
                'recommendations_count': len(current_health.recommendations),
                'last_optimization': current_health.last_optimization.isoformat() if current_health.last_optimization else None
            },
            'recent_statistics': {
                'total_checks': len(recent_health),
                'critical_count': critical_count,
                'warning_count': warning_count,
                'healthy_count': healthy_count,
                'health_rate_percent': (healthy_count / len(recent_health) * 100) if recent_health else 0
            },
            'performance_summary': {
                'current_level': performance_report['current_performance']['level'],
                'cpu_percent': performance_report['current_performance']['cpu_percent'],
                'memory_percent': performance_report['current_performance']['memory_percent'],
                'cache_hit_rate': performance_report['current_performance']['cache_hit_rate']
            },
            'issues': {
                'critical': current_health.critical_issues,
                'warnings': current_health.warnings,
                'recommendations': current_health.recommendations
            },
            'timestamp': datetime.now().isoformat()
        }


# グローバルインスタンス
_system_monitor = None


def get_system_monitor() -> SystemPerformanceMonitor:
    """グローバルシステム監視を取得"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemPerformanceMonitor()
    return _system_monitor


def start_system_monitoring(mode: MonitoringMode = MonitoringMode.ACTIVE):
    """システム監視開始（便利関数）"""
    monitor = get_system_monitor()
    monitor.monitoring_mode = mode
    monitor.start_monitoring()


def stop_system_monitoring():
    """システム監視停止（便利関数）"""
    monitor = get_system_monitor()
    monitor.stop_monitoring()


def get_system_health() -> SystemHealth:
    """システム健全性取得（便利関数）"""
    return get_system_monitor().get_current_health()


if __name__ == "__main__":
    print("🔍 システムパフォーマンス監視テスト")
    print("=" * 50)
    
    monitor = SystemPerformanceMonitor(MonitoringMode.ACTIVE)
    
    # 監視開始
    print("監視開始...")
    monitor.start_monitoring()
    
    # 少し待機
    time.sleep(5)
    
    # 現在の健全性
    health = monitor.get_current_health()
    print(f"\nシステム健全性:")
    print(f"  全体ステータス: {health.overall_status}")
    print(f"  パフォーマンスレベル: {health.performance_level.value}")
    print(f"  稼働時間: {health.uptime_hours:.2f}時間")
    
    if health.critical_issues:
        print(f"  重大な問題: {len(health.critical_issues)}件")
        for issue in health.critical_issues:
            print(f"    - {issue}")
    
    if health.warnings:
        print(f"  警告: {len(health.warnings)}件")
        for warning in health.warnings[:3]:  # 最初の3件のみ表示
            print(f"    - {warning}")
    
    # レポート生成
    report = monitor.get_monitoring_report()
    print(f"\n監視統計:")
    print(f"  健全率: {report['recent_statistics']['health_rate_percent']:.1f}%")
    print(f"  CPU使用率: {report['performance_summary']['cpu_percent']:.1f}%")
    print(f"  メモリ使用率: {report['performance_summary']['memory_percent']:.1f}%")
    
    # 監視停止
    print("\n監視停止...")
    monitor.stop_monitoring()
    
    print("テスト完了")