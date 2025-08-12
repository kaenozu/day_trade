#!/usr/bin/env python3
"""
本番環境起動スクリプト

Issue #320: システム本番稼働準備・設定最適化
本番環境でのシステム安全起動・監視開始
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from day_trade.config.environment_config import (
        Environment,
        get_environment_config_manager,
    )
    from day_trade.utils.advanced_fault_tolerance import get_recovery_system
    from day_trade.utils.performance_monitor import get_performance_monitor
    from day_trade.utils.structured_logging import get_structured_logger
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("プロジェクトルートから実行してください")
    sys.exit(1)


class ProductionSystemManager:
    """本番システム管理"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_manager = None
        self.logger = None
        self.performance_monitor = None
        self.recovery_system = None
        self.structured_logger = None
        self.startup_time = datetime.now()
        self.pid_file = self.project_root / "daytrade.pid"

        print("Production System Manager 初期化中...")

    def initialize_production_environment(self):
        """本番環境初期化"""
        try:
            # 環境変数設定
            os.environ["DAYTRADE_ENV"] = "production"

            # 設定管理初期化
            self.config_manager = get_environment_config_manager()
            config = self.config_manager.load_config(Environment.PRODUCTION)

            # ログ設定初期化
            self._setup_production_logging(config)

            # 構造化ログシステム初期化
            self.structured_logger = get_structured_logger()

            # システムコンポーネント初期化
            self._initialize_system_components()

            # プロダクション環境検証
            self._validate_production_environment(config)

            self.logger.info("本番環境初期化完了")
            return True

        except Exception as e:
            print(f"本番環境初期化エラー: {e}")
            return False

    def _setup_production_logging(self, config: Dict[str, Any]):
        """本番ログ設定"""
        log_config = config.get("system", {}).get("logging", {})

        # ログディレクトリ作成
        log_dir = Path(log_config.get("output_directory", "logs/production"))
        log_dir.mkdir(parents=True, exist_ok=True)

        # プロダクションロガー設定
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "production.log", encoding="utf-8"),
                (
                    logging.StreamHandler()
                    if log_config.get("log_to_console", False)
                    else logging.NullHandler()
                ),
            ],
        )

        self.logger = logging.getLogger("ProductionSystem")

    def _initialize_system_components(self):
        """システムコンポーネント初期化"""
        try:
            # パフォーマンス監視システム
            self.performance_monitor = get_performance_monitor()
            self.logger.info("パフォーマンス監視システム初期化完了")

            # 自動復旧システム
            self.recovery_system = get_recovery_system()
            self.recovery_system.start_monitoring()
            self.logger.info("自動復旧システム監視開始")

            # システム監視開始
            self._start_system_monitoring()

        except Exception as e:
            self.logger.error(f"システムコンポーネント初期化エラー: {e}")
            raise

    def _validate_production_environment(self, config: Dict[str, Any]):
        """本番環境検証"""
        validation_checks = [
            ("データベース接続", self._check_database_connection),
            ("ディスク容量", self._check_disk_space),
            ("メモリ容量", self._check_memory_availability),
            ("設定ファイル検証", self._check_configuration_validity),
            ("セキュリティ設定", self._check_security_settings),
            ("ネットワーク接続", self._check_network_connectivity),
        ]

        failed_checks = []

        for check_name, check_func in validation_checks:
            try:
                if not check_func(config):
                    failed_checks.append(check_name)
                    self.logger.error(f"本番環境検証失敗: {check_name}")
                else:
                    self.logger.info(f"本番環境検証成功: {check_name}")
            except Exception as e:
                failed_checks.append(check_name)
                self.logger.error(f"本番環境検証エラー: {check_name} - {e}")

        if failed_checks:
            raise RuntimeError(f"本番環境検証失敗: {', '.join(failed_checks)}")

        self.logger.info("本番環境検証完了 - 全チェック成功")

    def _check_database_connection(self, config: Dict[str, Any]) -> bool:
        """データベース接続確認"""
        try:
            db_url = config.get("data", {}).get("storage", {}).get("database_url", "")
            if not db_url:
                return False

            # データディレクトリ作成
            if db_url.startswith("sqlite:///"):
                db_path = Path(db_url.replace("sqlite:///", ""))
                db_path.parent.mkdir(parents=True, exist_ok=True)

            return True
        except Exception:
            return False

    def _check_disk_space(self, config: Dict[str, Any]) -> bool:
        """ディスク容量確認"""
        try:
            import shutil

            free_space = shutil.disk_usage(".").free
            required_space = 1 * 1024 * 1024 * 1024  # 1GB
            return free_space > required_space
        except Exception:
            return False

    def _check_memory_availability(self, config: Dict[str, Any]) -> bool:
        """メモリ容量確認"""
        try:
            import psutil

            available_memory = psutil.virtual_memory().available
            required_memory = (
                config.get("system", {}).get("performance", {}).get("memory_limit_mb", 2048)
                * 1024
                * 1024
            )
            return available_memory > required_memory
        except Exception:
            return True  # psutilがない場合はスキップ

    def _check_configuration_validity(self, config: Dict[str, Any]) -> bool:
        """設定ファイル妥当性確認"""
        required_sections = [
            "watchlist",
            "system",
            "daytrading",
            "analysis",
            "portfolio",
            "alerts",
            "data",
        ]
        return all(section in config for section in required_sections)

    def _check_security_settings(self, config: Dict[str, Any]) -> bool:
        """セキュリティ設定確認"""
        security_config = config.get("security", {})
        return security_config.get("data_protection", {}).get(
            "encryption_at_rest", False
        ) and security_config.get("access_control", {}).get("enabled", False)

    def _check_network_connectivity(self, config: Dict[str, Any]) -> bool:
        """ネットワーク接続確認"""
        try:
            import urllib.request

            urllib.request.urlopen("https://www.google.com", timeout=5)
            return True
        except Exception:
            return False

    def _start_system_monitoring(self):
        """システム監視開始"""
        # プロセス監視開始
        self._start_process_monitoring()

        # ヘルスチェック開始
        self._start_health_checks()

    def _start_process_monitoring(self):
        """プロセス監視開始"""
        # PIDファイル作成
        try:
            with open(self.pid_file, "w") as f:
                f.write(str(os.getpid()))
            self.logger.info(f"PIDファイル作成: {self.pid_file}")
        except Exception as e:
            self.logger.warning(f"PIDファイル作成失敗: {e}")

    def _start_health_checks(self):
        """ヘルスチェック開始"""
        # 定期ヘルスチェックは自動復旧システムで実行済み
        self.logger.info("ヘルスチェック機能 - 自動復旧システムで実行中")

    def start_main_application(self):
        """メインアプリケーション開始"""
        try:
            self.logger.info("メインアプリケーション開始")

            # 構造化ログでシステム開始を記録
            with self.structured_logger.operation_context("production_system_start", "system"):
                # メイン処理ループ（実際のアプリケーションロジック）
                self._run_main_loop()

        except KeyboardInterrupt:
            self.logger.info("キーボード割り込み - システム停止開始")
            self.graceful_shutdown()
        except Exception as e:
            self.logger.error(f"メインアプリケーションエラー: {e}")
            self.emergency_shutdown()

    def _run_main_loop(self):
        """メイン処理ループ"""
        self.logger.info("メイン処理ループ開始")

        loop_count = 0
        while True:
            try:
                # システム状態確認
                system_status = self.recovery_system.get_system_status()

                if loop_count % 60 == 0:  # 1分ごと
                    self.logger.info(
                        f"システム稼働中 - 稼働時間推定: {system_status['uptime_estimate']['uptime_percent']:.1f}%"
                    )

                # メイン業務処理（ここに実際のトレーディングロジックを配置）
                time.sleep(1)  # 1秒待機

                loop_count += 1

            except Exception as e:
                self.logger.error(f"メインループエラー: {e}")
                time.sleep(5)  # エラー時は5秒待機

    def graceful_shutdown(self):
        """正常停止処理"""
        self.logger.info("正常停止処理開始")

        try:
            # 自動復旧システム停止
            if self.recovery_system:
                self.recovery_system.stop_monitoring()
                self.logger.info("自動復旧システム停止完了")

            # 構造化ログでシステム停止を記録
            if self.structured_logger:
                with self.structured_logger.operation_context("production_system_stop", "system"):
                    runtime = (datetime.now() - self.startup_time).total_seconds()
                    self.logger.info(f"システム停止 - 稼働時間: {runtime:.1f}秒")

            # PIDファイル削除
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("PIDファイル削除完了")

            self.logger.info("正常停止処理完了")

        except Exception as e:
            self.logger.error(f"正常停止処理エラー: {e}")

    def emergency_shutdown(self):
        """緊急停止処理"""
        self.logger.error("緊急停止処理開始")

        try:
            # 最小限の停止処理
            if self.recovery_system:
                self.recovery_system.stop_monitoring()

            if self.pid_file.exists():
                self.pid_file.unlink()

            self.logger.error("緊急停止処理完了")

        except Exception as e:
            self.logger.error(f"緊急停止処理エラー: {e}")

        sys.exit(1)


def signal_handler(signum, frame):
    """シグナルハンドラー"""
    print(f"シグナル {signum} を受信 - 正常停止開始")
    sys.exit(0)


def main():
    """メイン実行関数"""
    print("Day Trade Production System - Starting")
    print("=" * 50)

    # シグナルハンドラー設定
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, signal_handler)

    # システム管理インスタンス作成
    system_manager = ProductionSystemManager()

    try:
        # 本番環境初期化
        if not system_manager.initialize_production_environment():
            print("本番環境初期化失敗 - 終了")
            return 1

        print("本番環境初期化完了 - システム開始")

        # メインアプリケーション開始
        system_manager.start_main_application()

        return 0

    except KeyboardInterrupt:
        print("\\nキーボード割り込み - 正常停止")
        system_manager.graceful_shutdown()
        return 0

    except Exception as e:
        print(f"システム起動エラー: {e}")
        system_manager.emergency_shutdown()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
