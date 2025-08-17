#!/usr/bin/env python3
"""
継続的精度監視サービス
Issue #857: 93%精度維持保証の24/7継続監視

バックグラウンドサービスとして動作し、常時精度を監視
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from enhanced_performance_monitor import (
    EnhancedPerformanceMonitorV2,
    AccuracyGuaranteeConfig,
    AccuracyGuaranteeLevel,
    MonitoringIntensity
)


class ContinuousAccuracyService:
    """継続的精度監視サービス"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_service_config()
        self.monitor = EnhancedPerformanceMonitorV2(config_path)
        self.service_active = True
        self.monitoring_task: Optional[asyncio.Task] = None
        self.status_report_task: Optional[asyncio.Task] = None
        
        # 監視統計
        self.service_stats = {
            "start_time": datetime.now(),
            "monitoring_cycles": 0,
            "accuracy_violations": 0,
            "emergency_triggers": 0,
            "retraining_executions": 0,
            "last_report_time": None
        }
        
        self.setup_logging()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """ログ設定"""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        
        # ログフォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ファイルハンドラー
        log_file = Path(log_config.get("file", "logs/continuous_accuracy_service.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 他のロガーレベル調整
        logging.getLogger("enhanced_performance_monitor").setLevel(log_level)
    
    def setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            self.logger.info(f"シグナル {signum} を受信しました。サービスを停止します。")
            self.service_active = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _load_service_config(self) -> Dict:
        """サービス設定読み込み"""
        default_config = {
            "service": {
                "name": "continuous_accuracy_service",
                "description": "93%精度維持保証サービス",
                "monitoring_symbols": ["7203", "8306", "4751", "9984", "6758"],
                "monitoring_interval_minutes": 5,
                "status_report_interval_minutes": 60,
                "auto_restart_on_error": True,
                "max_restart_attempts": 3
            },
            "accuracy_guarantee": {
                "level": "standard_93",
                "min_accuracy": 93.0,
                "target_accuracy": 95.0,
                "emergency_threshold": 85.0,
                "auto_recovery": True
            },
            "monitoring": {
                "intensity": "high",
                "trend_analysis": True,
                "anomaly_detection": True,
                "predictive_alerts": True
            },
            "notifications": {
                "enabled": True,
                "channels": ["log", "file"],
                "critical_threshold": 85.0,
                "warning_threshold": 90.0
            },
            "logging": {
                "level": "INFO",
                "file": "logs/continuous_accuracy_service.log",
                "rotation": "daily",
                "retention_days": 30
            },
            "performance": {
                "max_memory_mb": 512,
                "max_cpu_percent": 25,
                "monitoring_timeout_minutes": 30
            }
        }
        
        if self.config_path:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
                    self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}")
        
        return default_config
    
    async def start_service(self) -> None:
        """サービス開始"""
        self.logger.info("=== 継続的精度監視サービス開始 ===")
        
        # 設定情報表示
        service_config = self.config["service"]
        self.logger.info(f"サービス名: {service_config['name']}")
        self.logger.info(f"監視銘柄: {service_config['monitoring_symbols']}")
        self.logger.info(f"監視間隔: {service_config['monitoring_interval_minutes']}分")
        self.logger.info(f"精度保証レベル: {self.config['accuracy_guarantee']['min_accuracy']}%")
        
        try:
            # 監視タスク開始
            self.monitoring_task = asyncio.create_task(
                self._run_continuous_monitoring()
            )
            
            # 定期レポートタスク開始
            self.status_report_task = asyncio.create_task(
                self._run_status_reporting()
            )
            
            # タスク実行
            await asyncio.gather(
                self.monitoring_task,
                self.status_report_task,
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"サービス実行エラー: {e}")
        finally:
            await self._cleanup_service()
    
    async def _run_continuous_monitoring(self) -> None:
        """継続監視実行"""
        service_config = self.config["service"]
        symbols = service_config["monitoring_symbols"]
        interval_minutes = service_config["monitoring_interval_minutes"]
        
        restart_attempts = 0
        max_attempts = service_config.get("max_restart_attempts", 3)
        
        while self.service_active:
            try:
                self.logger.info(f"監視サイクル開始: {len(symbols)}銘柄")
                
                # 強化監視実行（タイムアウト付き）
                timeout_minutes = self.config["performance"]["monitoring_timeout_minutes"]
                
                await asyncio.wait_for(
                    self.monitor.start_enhanced_monitoring(symbols),
                    timeout=timeout_minutes * 60
                )
                
                # 統計更新
                self.service_stats["monitoring_cycles"] += 1
                restart_attempts = 0  # 成功時はリセット
                
            except asyncio.TimeoutError:
                self.logger.warning(f"監視タイムアウト ({timeout_minutes}分)")
                self.service_stats["monitoring_cycles"] += 1
                
            except Exception as e:
                self.logger.error(f"監視エラー: {e}")
                restart_attempts += 1
                
                if (service_config.get("auto_restart_on_error", True) and 
                    restart_attempts <= max_attempts):
                    self.logger.info(f"自動再起動 ({restart_attempts}/{max_attempts})")
                    await asyncio.sleep(30)  # 30秒待機
                    continue
                else:
                    self.logger.error("最大再起動回数に達しました。サービスを停止します。")
                    self.service_active = False
                    break
            
            # 次回まで待機
            if self.service_active:
                await asyncio.sleep(interval_minutes * 60)
    
    async def _run_status_reporting(self) -> None:
        """定期ステータスレポート"""
        interval_minutes = self.config["service"]["status_report_interval_minutes"]
        
        while self.service_active:
            try:
                await asyncio.sleep(interval_minutes * 60)
                
                if not self.service_active:
                    break
                
                # ステータスレポート生成
                await self._generate_status_report()
                
            except Exception as e:
                self.logger.error(f"ステータスレポートエラー: {e}")
    
    async def _generate_status_report(self) -> None:
        """ステータスレポート生成"""
        try:
            # 包括レポート取得
            monitor_report = await self.monitor.generate_comprehensive_report()
            
            # サービス統計更新
            current_time = datetime.now()
            uptime = current_time - self.service_stats["start_time"]
            
            # レポート作成
            status_report = {
                "service_status": {
                    "name": self.config["service"]["name"],
                    "status": "active" if self.service_active else "stopped",
                    "uptime_hours": uptime.total_seconds() / 3600,
                    "monitoring_cycles": self.service_stats["monitoring_cycles"],
                    "last_report": current_time.isoformat()
                },
                "accuracy_status": {
                    "guarantee_level": self.config["accuracy_guarantee"]["min_accuracy"],
                    "current_compliance": self._assess_current_compliance(monitor_report),
                    "violations_count": self.service_stats["accuracy_violations"],
                    "emergency_triggers": self.service_stats["emergency_triggers"]
                },
                "system_performance": {
                    "monitoring_active": monitor_report.get("monitoring_status", "unknown"),
                    "resource_usage": monitor_report.get("resource_usage", {}),
                    "continuous_metrics_count": len(monitor_report.get("continuous_metrics", {}))
                },
                "next_actions": self._recommend_next_actions(monitor_report)
            }
            
            # ログ出力
            self.logger.info("=== ステータスレポート ===")
            self.logger.info(f"稼働時間: {uptime.total_seconds()/3600:.1f}時間")
            self.logger.info(f"監視サイクル: {self.service_stats['monitoring_cycles']}回")
            self.logger.info(f"精度保証: {status_report['accuracy_status']['current_compliance']}")
            
            # ファイル保存
            report_file = Path("reports/continuous_service_status.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(status_report, f, ensure_ascii=False, indent=2, default=str)
            
            self.service_stats["last_report_time"] = current_time
            
        except Exception as e:
            self.logger.error(f"ステータスレポート生成エラー: {e}")
    
    def _assess_current_compliance(self, monitor_report: Dict) -> str:
        """現在の精度保証遵守状況評価"""
        try:
            metrics = monitor_report.get("continuous_metrics", {})
            if not metrics:
                return "不明"
            
            min_accuracy = self.config["accuracy_guarantee"]["min_accuracy"]
            
            # 全銘柄の精度チェック
            violations = 0
            total_symbols = len(metrics)
            
            for symbol, data in metrics.items():
                accuracy = data.get("accuracy", 0)
                if accuracy < min_accuracy:
                    violations += 1
            
            compliance_rate = (total_symbols - violations) / total_symbols * 100
            
            if compliance_rate >= 100:
                return "完全遵守"
            elif compliance_rate >= 90:
                return f"概ね遵守 ({compliance_rate:.1f}%)"
            elif compliance_rate >= 80:
                return f"要注意 ({compliance_rate:.1f}%)"
            else:
                return f"違反状態 ({compliance_rate:.1f}%)"
                
        except Exception:
            return "評価エラー"
    
    def _recommend_next_actions(self, monitor_report: Dict) -> List[str]:
        """次回アクション推奨"""
        actions = []
        
        try:
            # 精度状況チェック
            metrics = monitor_report.get("continuous_metrics", {})
            min_accuracy = self.config["accuracy_guarantee"]["min_accuracy"]
            
            low_accuracy_symbols = []
            for symbol, data in metrics.items():
                accuracy = data.get("accuracy", 0)
                if accuracy < min_accuracy:
                    low_accuracy_symbols.append(symbol)
            
            if low_accuracy_symbols:
                actions.append(f"精度低下銘柄の再学習検討: {', '.join(low_accuracy_symbols)}")
            
            # リソース使用状況チェック
            resource_usage = monitor_report.get("resource_usage", {})
            if resource_usage.get("active_training", 0) > 0:
                actions.append("現在再学習中 - 完了まで監視継続")
            
            # 監視状況チェック
            if monitor_report.get("monitoring_status") != "active":
                actions.append("監視システムの再起動を検討")
            
            if not actions:
                actions.append("正常稼働中 - 継続監視")
                
        except Exception:
            actions.append("状況評価エラー - システム確認が必要")
        
        return actions
    
    async def _cleanup_service(self) -> None:
        """サービスクリーンアップ"""
        self.logger.info("サービスクリーンアップ開始")
        
        try:
            # 監視停止
            if self.monitor:
                self.monitor.stop_enhanced_monitoring()
            
            # タスクキャンセル
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self.status_report_task and not self.status_report_task.done():
                self.status_report_task.cancel()
                try:
                    await self.status_report_task
                except asyncio.CancelledError:
                    pass
            
            # 最終レポート生成
            await self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
        
        self.logger.info("=== 継続的精度監視サービス停止 ===")
    
    async def _generate_final_report(self) -> None:
        """最終レポート生成"""
        try:
            end_time = datetime.now()
            uptime = end_time - self.service_stats["start_time"]
            
            final_report = {
                "service_summary": {
                    "name": self.config["service"]["name"],
                    "start_time": self.service_stats["start_time"].isoformat(),
                    "end_time": end_time.isoformat(),
                    "total_uptime_hours": uptime.total_seconds() / 3600,
                    "monitoring_cycles": self.service_stats["monitoring_cycles"],
                    "accuracy_violations": self.service_stats["accuracy_violations"],
                    "emergency_triggers": self.service_stats["emergency_triggers"],
                    "retraining_executions": self.service_stats["retraining_executions"]
                },
                "performance_metrics": {
                    "average_cycle_time": uptime.total_seconds() / max(1, self.service_stats["monitoring_cycles"]),
                    "reliability_score": (
                        self.service_stats["monitoring_cycles"] / 
                        max(1, self.service_stats["monitoring_cycles"] + self.service_stats["accuracy_violations"])
                    ) * 100
                }
            }
            
            # ファイル保存
            report_file = Path("reports/continuous_service_final.json")
            report_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"最終レポート保存: {report_file}")
            self.logger.info(f"総稼働時間: {uptime.total_seconds()/3600:.1f}時間")
            self.logger.info(f"監視サイクル: {self.service_stats['monitoring_cycles']}回")
            
        except Exception as e:
            self.logger.error(f"最終レポート生成エラー: {e}")
    
    def stop_service(self) -> None:
        """サービス停止"""
        self.logger.info("サービス停止要求を受信しました")
        self.service_active = False


async def run_continuous_service(config_path: Optional[str] = None):
    """継続サービス実行"""
    service = ContinuousAccuracyService(config_path)
    
    try:
        await service.start_service()
    except KeyboardInterrupt:
        service.logger.info("手動停止要求")
        service.stop_service()
    except Exception as e:
        service.logger.error(f"サービス実行エラー: {e}")
        raise


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="継続的精度監視サービス")
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="設定ファイルパス"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="デーモンモードで実行"
    )
    
    args = parser.parse_args()
    
    try:
        # デーモンモード（実装は簡略化）
        if args.daemon:
            print("デーモンモードは将来実装予定です")
            sys.exit(1)
        
        # 通常モード
        asyncio.run(run_continuous_service(args.config))
        
    except KeyboardInterrupt:
        print("\n継続的精度監視サービスを停止しました")
    except Exception as e:
        print(f"サービス実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()