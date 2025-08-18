"""
統合データベース管理システム

本番データベース、バックアップ、監視、復元、ダッシュボードの統合管理
シンプルなAPI、設定管理、自動初期化機能
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from day_trade.core.error_handling.unified_error_system import (
    ApplicationError, DataAccessError, SystemError,
    error_boundary, global_error_handler
)
from day_trade.core.logging.unified_logging_system import get_logger

from .production_database_manager import (
    ProductionDatabaseManager, 
    initialize_production_database
)
from .backup_manager import (
    DatabaseBackupManager, 
    initialize_backup_manager,
    get_backup_manager
)
from .monitoring_system import (
    DatabaseMonitoringSystem, 
    initialize_monitoring_system,
    get_monitoring_system
)
from .restore_manager import (
    DatabaseRestoreManager, 
    initialize_restore_manager,
    get_restore_manager
)
from .dashboard import (
    DatabaseDashboard, 
    initialize_dashboard,
    get_dashboard
)

logger = get_logger(__name__)


class UnifiedDatabaseError(SystemError):
    """統合データベース管理エラー"""
    
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, operation=f"unified_db_{component}", **kwargs)


class UnifiedDatabaseManager:
    """統合データベース管理システム"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/database_production.yaml"
        
        # 各コンポーネント
        self.production_db: Optional[ProductionDatabaseManager] = None
        self.backup_manager: Optional[DatabaseBackupManager] = None
        self.monitoring_system: Optional[DatabaseMonitoringSystem] = None
        self.restore_manager: Optional[DatabaseRestoreManager] = None
        self.dashboard: Optional[DatabaseDashboard] = None
        
        # 初期化状態
        self.initialized = False
        self.initialization_time: Optional[datetime] = None
        self.components_status: Dict[str, str] = {}
        
    @error_boundary(
        component_name="unified_database_manager",
        operation_name="initialize",
        suppress_errors=False
    )
    def initialize(self, auto_start: bool = True) -> Dict[str, Any]:
        """統合データベースシステム初期化"""
        try:
            logger.info("統合データベース管理システム初期化開始")
            
            # 本番データベースマネージャー初期化
            self.production_db = initialize_production_database(self.config_path)
            self.components_status['production_db'] = 'initialized'
            
            # 設定読み込み
            config = self.production_db.config
            
            # バックアップマネージャー初期化
            try:
                self.backup_manager = initialize_backup_manager(config)
                self.components_status['backup_manager'] = 'initialized'
                
                if auto_start and config.get('backup', {}).get('auto_start_scheduler', False):
                    self.backup_manager.start_scheduler()
                    self.components_status['backup_manager'] = 'running'
                    
            except Exception as e:
                logger.warning(f"バックアップマネージャー初期化失敗: {e}")
                self.components_status['backup_manager'] = 'failed'
            
            # 監視システム初期化
            try:
                engine = self.production_db.connection_pool.engine
                self.monitoring_system = initialize_monitoring_system(engine, config)
                self.components_status['monitoring_system'] = 'initialized'
                
                if auto_start and config.get('monitoring', {}).get('auto_start', False):
                    self.monitoring_system.start_monitoring()
                    self.components_status['monitoring_system'] = 'running'
                    
            except Exception as e:
                logger.warning(f"監視システム初期化失敗: {e}")
                self.components_status['monitoring_system'] = 'failed'
            
            # 復元マネージャー初期化
            try:
                backup_path = Path(config.get('backup', {}).get('backup_path', './backups'))
                self.restore_manager = initialize_restore_manager(config, backup_path)
                self.components_status['restore_manager'] = 'initialized'
            except Exception as e:
                logger.warning(f"復元マネージャー初期化失敗: {e}")
                self.components_status['restore_manager'] = 'failed'
            
            # ダッシュボード初期化
            try:
                self.dashboard = initialize_dashboard(config)
                self.components_status['dashboard'] = 'running'
            except Exception as e:
                logger.warning(f"ダッシュボード初期化失敗: {e}")
                self.components_status['dashboard'] = 'failed'
            
            self.initialized = True
            self.initialization_time = datetime.now()
            
            # 初期化結果
            success_count = sum(1 for status in self.components_status.values() if status in ['initialized', 'running'])
            total_count = len(self.components_status)
            
            logger.info(
                f"統合データベース管理システム初期化完了",
                success_components=success_count,
                total_components=total_count,
                components_status=self.components_status
            )
            
            return {
                'status': 'success',
                'initialized_at': self.initialization_time.isoformat(),
                'components': self.components_status,
                'success_rate': f"{success_count}/{total_count}",
                'auto_start': auto_start
            }
            
        except Exception as e:
            logger.error(f"統合データベース管理システム初期化失敗: {e}")
            raise UnifiedDatabaseError(f"初期化失敗: {e}", component="initialization")
    
    def shutdown(self) -> Dict[str, Any]:
        """統合データベースシステム停止"""
        try:
            logger.info("統合データベース管理システム停止開始")
            
            shutdown_results = {}
            
            # ダッシュボード停止
            if self.dashboard:
                try:
                    self.dashboard.stop_dashboard()
                    shutdown_results['dashboard'] = 'stopped'
                except Exception as e:
                    logger.warning(f"ダッシュボード停止失敗: {e}")
                    shutdown_results['dashboard'] = 'error'
            
            # 監視システム停止
            if self.monitoring_system:
                try:
                    self.monitoring_system.stop_monitoring()
                    shutdown_results['monitoring_system'] = 'stopped'
                except Exception as e:
                    logger.warning(f"監視システム停止失敗: {e}")
                    shutdown_results['monitoring_system'] = 'error'
            
            # バックアップスケジューラー停止
            if self.backup_manager:
                try:
                    self.backup_manager.stop_scheduler()
                    shutdown_results['backup_manager'] = 'stopped'
                except Exception as e:
                    logger.warning(f"バックアップマネージャー停止失敗: {e}")
                    shutdown_results['backup_manager'] = 'error'
            
            shutdown_results['production_db'] = 'active'  # 本番DBは停止しない
            
            logger.info("統合データベース管理システム停止完了", shutdown_results=shutdown_results)
            
            return {
                'status': 'success',
                'shutdown_at': datetime.now().isoformat(),
                'results': shutdown_results
            }
            
        except Exception as e:
            logger.error(f"統合データベース管理システム停止失敗: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'shutdown_at': datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム全体状態取得"""
        if not self.initialized:
            return {
                'initialized': False,
                'message': 'システムが初期化されていません'
            }
        
        # 各コンポーネントの詳細状態
        detailed_status = {}
        
        # 本番データベース
        if self.production_db:
            health = self.production_db.health_check()
            db_info = self.production_db.get_database_info()
            detailed_status['production_db'] = {
                'status': 'active',
                'health': health,
                'database_info': db_info
            }
        
        # バックアップマネージャー
        if self.backup_manager:
            backup_stats = self.backup_manager.get_backup_statistics()
            detailed_status['backup_manager'] = {
                'status': self.components_status.get('backup_manager', 'unknown'),
                'statistics': backup_stats
            }
        
        # 監視システム
        if self.monitoring_system:
            monitor_status = self.monitoring_system.get_monitoring_status()
            current_metrics = self.monitoring_system.get_current_metrics()
            detailed_status['monitoring_system'] = {
                'status': self.components_status.get('monitoring_system', 'unknown'),
                'monitoring_status': monitor_status,
                'current_metrics': current_metrics
            }
        
        # 復元マネージャー
        if self.restore_manager:
            detailed_status['restore_manager'] = {
                'status': self.components_status.get('restore_manager', 'unknown'),
                'available': True
            }
        
        # ダッシュボード
        if self.dashboard:
            detailed_status['dashboard'] = {
                'status': self.components_status.get('dashboard', 'unknown'),
                'last_update': self.dashboard.last_update.isoformat() if self.dashboard.last_update else None
            }
        
        # 全体の健全性判定
        overall_health = 'healthy'
        critical_components = ['production_db']
        
        for component in critical_components:
            if component not in detailed_status:
                overall_health = 'critical'
                break
            
            component_status = detailed_status[component]
            if component == 'production_db':
                if component_status.get('health', {}).get('status') != 'healthy':
                    overall_health = 'critical'
                    break
        
        # 警告レベルの問題チェック
        if overall_health == 'healthy':
            if self.monitoring_system:
                active_alerts = self.monitoring_system.get_active_alerts()
                critical_alerts = [a for a in active_alerts if a.get('severity') == 'critical']
                if critical_alerts:
                    overall_health = 'warning'
        
        return {
            'initialized': True,
            'initialization_time': self.initialization_time.isoformat(),
            'overall_health': overall_health,
            'components': detailed_status,
            'components_count': {
                'total': len(self.components_status),
                'active': len([s for s in self.components_status.values() if s in ['initialized', 'running']]),
                'failed': len([s for s in self.components_status.values() if s == 'failed'])
            }
        }
    
    # 便利メソッド群
    
    def create_backup(self, backup_type: str = "manual") -> Dict[str, Any]:
        """手動バックアップ作成"""
        if not self.backup_manager:
            raise UnifiedDatabaseError("バックアップマネージャーが利用できません", component="backup")
        
        return self.backup_manager.create_backup(backup_type)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """バックアップ一覧取得"""
        if not self.backup_manager:
            return []
        
        return self.backup_manager.list_backups()
    
    def restore_database(self, backup_filename: str, **kwargs) -> Dict[str, Any]:
        """データベース復元"""
        if not self.restore_manager:
            raise UnifiedDatabaseError("復元マネージャーが利用できません", component="restore")
        
        return self.restore_manager.restore_database(backup_filename, **kwargs)
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """現在のメトリクス取得"""
        if not self.monitoring_system:
            return None
        
        return self.monitoring_system.get_current_metrics()
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """アクティブアラート取得"""
        if not self.monitoring_system:
            return []
        
        return self.monitoring_system.get_active_alerts()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """ダッシュボードデータ取得"""
        if not self.dashboard:
            return {}
        
        return self.dashboard.get_dashboard_data()
    
    def generate_report(self) -> Dict[str, Any]:
        """レポート生成"""
        if not self.dashboard:
            raise UnifiedDatabaseError("ダッシュボードが利用できません", component="dashboard")
        
        return self.dashboard.generate_daily_report()
    
    def run_health_check(self) -> Dict[str, Any]:
        """総合ヘルスチェック実行"""
        health_results = {}
        
        # データベースヘルスチェック
        if self.production_db:
            db_health = self.production_db.health_check()
            health_results['database'] = db_health
        
        # 監視システムから最新メトリクス取得
        if self.monitoring_system:
            metrics = self.monitoring_system.get_current_metrics()
            alerts = self.monitoring_system.get_active_alerts()
            
            health_results['monitoring'] = {
                'metrics_available': metrics is not None,
                'active_alerts_count': len(alerts),
                'critical_alerts': len([a for a in alerts if a.get('severity') == 'critical'])
            }
        
        # バックアップ状態確認
        if self.backup_manager:
            backup_stats = self.backup_manager.get_backup_statistics()
            health_results['backup'] = {
                'total_backups': backup_stats.get('total_backups', 0),
                'success_rate': backup_stats.get('success_rate', 0),
                'scheduler_running': backup_stats.get('scheduler_running', False)
            }
        
        # 全体的な健全性判定
        overall_status = 'healthy'
        issues = []
        
        # データベース接続問題
        if health_results.get('database', {}).get('status') != 'healthy':
            overall_status = 'critical'
            issues.append('データベース接続に問題があります')
        
        # クリティカルアラート
        critical_alerts = health_results.get('monitoring', {}).get('critical_alerts', 0)
        if critical_alerts > 0:
            overall_status = 'warning' if overall_status == 'healthy' else overall_status
            issues.append(f'クリティカルアラートが{critical_alerts}件発生中です')
        
        # バックアップ問題
        backup_success_rate = health_results.get('backup', {}).get('success_rate', 100)
        if backup_success_rate < 95:
            overall_status = 'warning' if overall_status == 'healthy' else overall_status
            issues.append(f'バックアップ成功率が低下しています ({backup_success_rate:.1f}%)')
        
        return {
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'components': health_results,
            'issues': issues,
            'recommendations': self._generate_health_recommendations(health_results, issues)
        }
    
    def _generate_health_recommendations(self, health_results: Dict[str, Any], issues: List[str]) -> List[str]:
        """ヘルスチェック推奨事項生成"""
        recommendations = []
        
        if not issues:
            recommendations.append("システムは正常に動作しています。")
            return recommendations
        
        # データベース関連の推奨事項
        if health_results.get('database', {}).get('status') != 'healthy':
            recommendations.append("データベース接続を確認し、必要に応じて再起動を検討してください。")
        
        # アラート関連の推奨事項
        critical_alerts = health_results.get('monitoring', {}).get('critical_alerts', 0)
        if critical_alerts > 0:
            recommendations.append("クリティカルアラートの原因を調査し、対処してください。")
        
        # バックアップ関連の推奨事項
        backup_success_rate = health_results.get('backup', {}).get('success_rate', 100)
        if backup_success_rate < 95:
            recommendations.append("バックアップ設定とストレージ容量を確認してください。")
        
        return recommendations


# グローバルインスタンス管理
_unified_manager: Optional[UnifiedDatabaseManager] = None


def get_unified_database_manager() -> Optional[UnifiedDatabaseManager]:
    """統合データベースマネージャー取得"""
    return _unified_manager


def initialize_unified_database_manager(config_path: Optional[str] = None, auto_start: bool = True) -> UnifiedDatabaseManager:
    """統合データベースマネージャー初期化"""
    global _unified_manager
    
    _unified_manager = UnifiedDatabaseManager(config_path)
    _unified_manager.initialize(auto_start=auto_start)
    
    logger.info("統合データベース管理システムが利用可能になりました")
    return _unified_manager