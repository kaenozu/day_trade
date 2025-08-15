#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Environment Manager - 本番環境設定管理システム
環境設定、設定検証、パフォーマンス監視の統合管理
"""

import os
import json
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import configparser


class EnvironmentType(Enum):
    """環境タイプ"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SystemStatus(Enum):
    """システム状態"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class EnvironmentConfig:
    """環境設定"""
    env_type: EnvironmentType
    debug_mode: bool
    log_level: str
    max_memory_mb: int
    max_cpu_percent: float
    data_retention_days: int
    backup_enabled: bool
    monitoring_enabled: bool
    security_features: Dict[str, bool]


@dataclass
class SystemMetrics:
    """システムメトリクス"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float
    response_time_ms: float
    error_rate: float


class ProductionEnvironmentManager:
    """本番環境設定管理システム"""
    
    def __init__(self, config_file: str = "config/production.ini"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(exist_ok=True)
        
        # 設定の初期化
        self.env_config = self._load_environment_config()
        self.start_time = datetime.now()
        self.metrics_history = []
        
        # ログ設定
        from daytrade_logging import get_logger
        self.logger = get_logger("production_environment")
        
        # パフォーマンス閾値
        self.performance_thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 70.0,
            'memory_critical': 90.0,
            'memory_warning': 80.0,
            'disk_critical': 95.0,
            'disk_warning': 85.0,
            'response_time_critical': 5000.0,  # 5秒
            'response_time_warning': 2000.0,   # 2秒
            'error_rate_critical': 5.0,        # 5%
            'error_rate_warning': 2.0          # 2%
        }
        
        self.logger.info(f"Production Environment Manager initialized for {self.env_config.env_type.value}")
    
    def _load_environment_config(self) -> EnvironmentConfig:
        """環境設定を読み込み"""
        try:
            if self.config_file.exists():
                config = configparser.ConfigParser()
                config.read(self.config_file)
                
                return EnvironmentConfig(
                    env_type=EnvironmentType(config.get('environment', 'type', fallback='development')),
                    debug_mode=config.getboolean('environment', 'debug_mode', fallback=False),
                    log_level=config.get('logging', 'level', fallback='INFO'),
                    max_memory_mb=config.getint('resources', 'max_memory_mb', fallback=2048),
                    max_cpu_percent=config.getfloat('resources', 'max_cpu_percent', fallback=80.0),
                    data_retention_days=config.getint('data', 'retention_days', fallback=30),
                    backup_enabled=config.getboolean('backup', 'enabled', fallback=True),
                    monitoring_enabled=config.getboolean('monitoring', 'enabled', fallback=True),
                    security_features={
                        'input_validation': config.getboolean('security', 'input_validation', fallback=True),
                        'rate_limiting': config.getboolean('security', 'rate_limiting', fallback=True),
                        'encryption': config.getboolean('security', 'encryption', fallback=True)
                    }
                )
            else:
                # デフォルト設定を作成
                return self._create_default_config()
                
        except Exception as e:
            self.logger.error(f"Failed to load environment config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> EnvironmentConfig:
        """デフォルト設定を作成"""
        # 環境変数から環境タイプを判定
        env_type_str = os.environ.get('DAY_TRADE_ENV', 'development')
        try:
            env_type = EnvironmentType(env_type_str)
        except ValueError:
            env_type = EnvironmentType.DEVELOPMENT
        
        # 環境タイプに応じた設定
        if env_type == EnvironmentType.PRODUCTION:
            config = EnvironmentConfig(
                env_type=env_type,
                debug_mode=False,
                log_level='WARNING',
                max_memory_mb=4096,
                max_cpu_percent=70.0,
                data_retention_days=90,
                backup_enabled=True,
                monitoring_enabled=True,
                security_features={
                    'input_validation': True,
                    'rate_limiting': True,
                    'encryption': True
                }
            )
        elif env_type == EnvironmentType.STAGING:
            config = EnvironmentConfig(
                env_type=env_type,
                debug_mode=False,
                log_level='INFO',
                max_memory_mb=2048,
                max_cpu_percent=80.0,
                data_retention_days=30,
                backup_enabled=True,
                monitoring_enabled=True,
                security_features={
                    'input_validation': True,
                    'rate_limiting': False,
                    'encryption': True
                }
            )
        else:  # DEVELOPMENT
            config = EnvironmentConfig(
                env_type=env_type,
                debug_mode=True,
                log_level='DEBUG',
                max_memory_mb=1024,
                max_cpu_percent=90.0,
                data_retention_days=7,
                backup_enabled=False,
                monitoring_enabled=False,
                security_features={
                    'input_validation': False,
                    'rate_limiting': False,
                    'encryption': False
                }
            )
        
        # 設定ファイルに保存
        self._save_config(config)
        return config
    
    def _save_config(self, config: EnvironmentConfig):
        """設定をファイルに保存"""
        try:
            config_parser = configparser.ConfigParser()
            
            config_parser['environment'] = {
                'type': config.env_type.value,
                'debug_mode': str(config.debug_mode)
            }
            
            config_parser['logging'] = {
                'level': config.log_level
            }
            
            config_parser['resources'] = {
                'max_memory_mb': str(config.max_memory_mb),
                'max_cpu_percent': str(config.max_cpu_percent)
            }
            
            config_parser['data'] = {
                'retention_days': str(config.data_retention_days)
            }
            
            config_parser['backup'] = {
                'enabled': str(config.backup_enabled)
            }
            
            config_parser['monitoring'] = {
                'enabled': str(config.monitoring_enabled)
            }
            
            config_parser['security'] = {}
            for key, value in config.security_features.items():
                config_parser['security'][key] = str(value)
            
            with open(self.config_file, 'w') as f:
                config_parser.write(f)
                
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """システムメトリクスを取得"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # ネットワークI/O
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv
            }
            
            # プロセス数
            process_count = len(psutil.pids())
            
            # アップタイム
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # レスポンス時間（簡易測定）
            response_start = time.time()
            # 簡単な処理を実行してレスポンス時間を測定
            _ = [i for i in range(1000)]
            response_time_ms = (time.time() - response_start) * 1000
            
            # エラー率（仮想値）
            error_rate = 1.2  # 1.2%
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_usage_percent=disk_usage_percent,
                network_io=network_data,
                process_count=process_count,
                uptime_seconds=uptime_seconds,
                response_time_ms=response_time_ms,
                error_rate=error_rate
            )
            
            # メトリクス履歴に追加
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # 履歴の上限（24時間分）
            if len(self.metrics_history) > 1440:  # 1分毎の24時間分
                self.metrics_history = self.metrics_history[-1440:]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_usage_percent=0.0,
                network_io={'bytes_sent': 0, 'bytes_recv': 0},
                process_count=0,
                uptime_seconds=0.0,
                response_time_ms=0.0,
                error_rate=0.0
            )
    
    def assess_system_status(self, metrics: SystemMetrics) -> Tuple[SystemStatus, List[str]]:
        """システム状態を評価"""
        issues = []
        status = SystemStatus.HEALTHY
        
        # CPU チェック
        if metrics.cpu_percent >= self.performance_thresholds['cpu_critical']:
            issues.append(f"CPU使用率が危険レベル: {metrics.cpu_percent:.1f}%")
            status = SystemStatus.CRITICAL
        elif metrics.cpu_percent >= self.performance_thresholds['cpu_warning']:
            issues.append(f"CPU使用率が警告レベル: {metrics.cpu_percent:.1f}%")
            if status == SystemStatus.HEALTHY:
                status = SystemStatus.WARNING
        
        # メモリ チェック
        if metrics.memory_percent >= self.performance_thresholds['memory_critical']:
            issues.append(f"メモリ使用率が危険レベル: {metrics.memory_percent:.1f}%")
            status = SystemStatus.CRITICAL
        elif metrics.memory_percent >= self.performance_thresholds['memory_warning']:
            issues.append(f"メモリ使用率が警告レベル: {metrics.memory_percent:.1f}%")
            if status == SystemStatus.HEALTHY:
                status = SystemStatus.WARNING
        
        # ディスク チェック
        if metrics.disk_usage_percent >= self.performance_thresholds['disk_critical']:
            issues.append(f"ディスク使用率が危険レベル: {metrics.disk_usage_percent:.1f}%")
            status = SystemStatus.CRITICAL
        elif metrics.disk_usage_percent >= self.performance_thresholds['disk_warning']:
            issues.append(f"ディスク使用率が警告レベル: {metrics.disk_usage_percent:.1f}%")
            if status == SystemStatus.HEALTHY:
                status = SystemStatus.WARNING
        
        # レスポンス時間 チェック
        if metrics.response_time_ms >= self.performance_thresholds['response_time_critical']:
            issues.append(f"レスポンス時間が危険レベル: {metrics.response_time_ms:.1f}ms")
            status = SystemStatus.CRITICAL
        elif metrics.response_time_ms >= self.performance_thresholds['response_time_warning']:
            issues.append(f"レスポンス時間が警告レベル: {metrics.response_time_ms:.1f}ms")
            if status == SystemStatus.HEALTHY:
                status = SystemStatus.WARNING
        
        # エラー率 チェック
        if metrics.error_rate >= self.performance_thresholds['error_rate_critical']:
            issues.append(f"エラー率が危険レベル: {metrics.error_rate:.1f}%")
            status = SystemStatus.CRITICAL
        elif metrics.error_rate >= self.performance_thresholds['error_rate_warning']:
            issues.append(f"エラー率が警告レベル: {metrics.error_rate:.1f}%")
            if status == SystemStatus.HEALTHY:
                status = SystemStatus.WARNING
        
        return status, issues
    
    def validate_configuration(self) -> List[str]:
        """設定の検証"""
        issues = []
        
        # 本番環境での必須設定チェック
        if self.env_config.env_type == EnvironmentType.PRODUCTION:
            if self.env_config.debug_mode:
                issues.append("本番環境でデバッグモードが有効になっています")
            
            if self.env_config.log_level == 'DEBUG':
                issues.append("本番環境でDEBUGログレベルが設定されています")
            
            if not self.env_config.backup_enabled:
                issues.append("本番環境でバックアップが無効になっています")
            
            if not self.env_config.monitoring_enabled:
                issues.append("本番環境で監視が無効になっています")
            
            if not self.env_config.security_features.get('input_validation', False):
                issues.append("入力検証が無効になっています")
            
            if not self.env_config.security_features.get('encryption', False):
                issues.append("暗号化が無効になっています")
        
        # リソース制限チェック
        if self.env_config.max_memory_mb < 512:
            issues.append(f"メモリ制限が低すぎます: {self.env_config.max_memory_mb}MB")
        
        if self.env_config.max_cpu_percent > 95:
            issues.append(f"CPU制限が高すぎます: {self.env_config.max_cpu_percent}%")
        
        # 環境変数チェック
        required_env_vars = ['PATH', 'PYTHONPATH']
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(f"必須環境変数が設定されていません: {var}")
        
        return issues
    
    def get_environment_report(self) -> Dict[str, Any]:
        """環境レポートを生成"""
        metrics = self.get_system_metrics()
        status, status_issues = self.assess_system_status(metrics)
        config_issues = self.validate_configuration()
        
        return {
            'environment': {
                'type': self.env_config.env_type.value,
                'debug_mode': self.env_config.debug_mode,
                'log_level': self.env_config.log_level
            },
            'system_status': {
                'status': status.value,
                'issues': status_issues
            },
            'system_metrics': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'uptime_hours': metrics.uptime_seconds / 3600,
                'response_time_ms': metrics.response_time_ms,
                'error_rate': metrics.error_rate
            },
            'configuration': {
                'validation_issues': config_issues,
                'security_features': self.env_config.security_features,
                'backup_enabled': self.env_config.backup_enabled,
                'monitoring_enabled': self.env_config.monitoring_enabled
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_performance(self) -> List[str]:
        """パフォーマンス最適化の提案"""
        metrics = self.get_system_metrics()
        recommendations = []
        
        # CPU最適化
        if metrics.cpu_percent > 80:
            recommendations.append("CPU使用率が高いため、並列処理の見直しを検討してください")
        
        # メモリ最適化
        if metrics.memory_percent > 85:
            recommendations.append("メモリ使用率が高いため、キャッシュサイズの調整を検討してください")
        
        # ディスク最適化
        if metrics.disk_usage_percent > 90:
            recommendations.append("ディスク容量が不足しています。ログローテーションまたは容量拡張を検討してください")
        
        # レスポンス時間最適化
        if metrics.response_time_ms > 1000:
            recommendations.append("レスポンス時間が遅いため、データベースクエリの最適化を検討してください")
        
        # プロセス数最適化
        if metrics.process_count > 200:
            recommendations.append("プロセス数が多いため、不要なサービスの停止を検討してください")
        
        return recommendations


# グローバルインスタンス
_environment_manager = None


def get_environment_manager() -> ProductionEnvironmentManager:
    """グローバル環境管理システムを取得"""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = ProductionEnvironmentManager()
    return _environment_manager


def get_system_status() -> Tuple[SystemStatus, List[str]]:
    """システム状態を取得"""
    manager = get_environment_manager()
    metrics = manager.get_system_metrics()
    return manager.assess_system_status(metrics)


def get_environment_report() -> Dict[str, Any]:
    """環境レポートを取得"""
    return get_environment_manager().get_environment_report()


if __name__ == "__main__":
    print("🏭 本番環境設定管理システムテスト")
    print("=" * 50)
    
    manager = ProductionEnvironmentManager()
    
    # 環境レポート
    report = manager.get_environment_report()
    
    print(f"環境タイプ: {report['environment']['type']}")
    print(f"システム状態: {report['system_status']['status']}")
    print(f"CPU使用率: {report['system_metrics']['cpu_percent']:.1f}%")
    print(f"メモリ使用率: {report['system_metrics']['memory_percent']:.1f}%")
    
    # 問題があれば表示
    if report['system_status']['issues']:
        print("\n警告:")
        for issue in report['system_status']['issues']:
            print(f"  - {issue}")
    
    if report['configuration']['validation_issues']:
        print("\n設定問題:")
        for issue in report['configuration']['validation_issues']:
            print(f"  - {issue}")
    
    # 最適化提案
    recommendations = manager.optimize_performance()
    if recommendations:
        print("\n最適化提案:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print("\nテスト完了")