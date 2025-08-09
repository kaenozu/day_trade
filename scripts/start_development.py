#!/usr/bin/env python3
"""
開発環境起動スクリプト

Issue #320: システム本番稼働準備・設定最適化
開発環境でのシステム起動・デバッグ支援
"""

import logging
import os
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
    print("開発環境では一部機能が制限される可能性があります")


class DevelopmentSystemManager:
    """開発システム管理"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_manager = None
        self.logger = None
        self.startup_time = datetime.now()

        print("Development System Manager 初期化中...")

    def initialize_development_environment(self):
        """開発環境初期化"""
        try:
            # 環境変数設定
            os.environ["DAYTRADE_ENV"] = "development"

            # 設定管理初期化
            self.config_manager = get_environment_config_manager()
            config = self.config_manager.load_config(Environment.DEVELOPMENT)

            # 開発ログ設定初期化
            self._setup_development_logging(config)

            # システムコンポーネント初期化（開発版）
            self._initialize_development_components()

            self.logger.info("開発環境初期化完了")
            return True

        except Exception as e:
            print(f"開発環境初期化エラー: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _setup_development_logging(self, config: Dict[str, Any]):
        """開発ログ設定"""
        log_config = config.get("system", {}).get("logging", {})

        # ログディレクトリ作成
        log_dir = Path(log_config.get("output_directory", "logs/development"))
        log_dir.mkdir(parents=True, exist_ok=True)

        # 開発ロガー設定（詳細ログ出力）
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "DEBUG")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "development.log", encoding="utf-8"),
                logging.StreamHandler(),  # 開発環境では常にコンソール出力
            ],
        )

        self.logger = logging.getLogger("DevelopmentSystem")

    def _initialize_development_components(self):
        """開発システムコンポーネント初期化"""
        try:
            # パフォーマンス監視システム（簡易版）
            try:
                self.performance_monitor = get_performance_monitor()
                self.logger.info("パフォーマンス監視システム初期化完了")
            except Exception as e:
                self.logger.warning(f"パフォーマンス監視システム初期化スキップ: {e}")

            # 自動復旧システム（開発版）
            try:
                self.recovery_system = get_recovery_system()
                # 開発環境では短い間隔で監視
                self.recovery_system.monitor_interval = 15  # 15秒間隔
                self.recovery_system.start_monitoring()
                self.logger.info("自動復旧システム監視開始（開発モード）")
            except Exception as e:
                self.logger.warning(f"自動復旧システム初期化スキップ: {e}")

            # 構造化ログシステム（開発版）
            try:
                self.structured_logger = get_structured_logger()
                self.logger.info("構造化ログシステム初期化完了")
            except Exception as e:
                self.logger.warning(f"構造化ログシステム初期化スキップ: {e}")
                self.structured_logger = None

        except Exception as e:
            self.logger.error(f"開発システムコンポーネント初期化エラー: {e}")
            # 開発環境ではエラーがあっても継続

    def start_development_session(self):
        """開発セッション開始"""
        try:
            self.logger.info("開発セッション開始")

            if self.structured_logger:
                with self.structured_logger.operation_context(
                    "development_session_start", "developer"
                ):
                    self._run_development_loop()
            else:
                self._run_development_loop()

        except KeyboardInterrupt:
            self.logger.info("キーボード割り込み - 開発セッション終了")
            self.shutdown_development()
        except Exception as e:
            self.logger.error(f"開発セッションエラー: {e}")
            import traceback

            traceback.print_exc()
            self.shutdown_development()

    def _run_development_loop(self):
        """開発処理ループ"""
        self.logger.info("開発処理ループ開始")

        # 開発環境情報表示
        self._display_development_info()

        loop_count = 0
        while True:
            try:
                # 開発環境での処理（デバッグ・テスト支援）

                if loop_count % 30 == 0:  # 30秒ごと
                    self._display_system_status()

                # 開発環境特有の処理
                if loop_count % 120 == 0:  # 2分ごと
                    self._run_development_diagnostics()

                time.sleep(1)  # 1秒待機
                loop_count += 1

            except Exception as e:
                self.logger.error(f"開発ループエラー: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(5)  # エラー時は5秒待機

    def _display_development_info(self):
        """開発環境情報表示"""
        try:
            env_info = self.config_manager.get_environment_info()

            self.logger.info("=== 開発環境情報 ===")
            self.logger.info(f"環境: {env_info['current_environment']}")
            self.logger.info(f"設定ディレクトリ: {env_info['config_directory']}")
            self.logger.info(f"利用可能設定: {env_info['available_configs']}")
            self.logger.info(
                f"環境変数: {list(env_info['environment_variables'].keys())}"
            )

            # システム設定表示
            system_config = self.config_manager.get_system_config()
            self.logger.info(
                f"パフォーマンス目標: {system_config.performance_target_seconds}秒"
            )
            self.logger.info(f"メモリ制限: {system_config.memory_limit_mb}MB")

        except Exception as e:
            self.logger.warning(f"開発環境情報表示エラー: {e}")

    def _display_system_status(self):
        """システム状態表示"""
        try:
            runtime = (datetime.now() - self.startup_time).total_seconds()

            status_info = [f"稼働時間: {runtime:.1f}秒", f"プロセスID: {os.getpid()}"]

            # 復旧システム状態
            if hasattr(self, "recovery_system") and self.recovery_system:
                try:
                    system_status = self.recovery_system.get_system_status()
                    status_info.append(f"監視中: {system_status['monitoring_active']}")
                    status_info.append(
                        f"稼働率推定: {system_status['uptime_estimate']['uptime_percent']:.1f}%"
                    )
                except Exception:
                    pass

            self.logger.info(f"システム状態: {' | '.join(status_info)}")

        except Exception as e:
            self.logger.warning(f"システム状態表示エラー: {e}")

    def _run_development_diagnostics(self):
        """開発診断実行"""
        try:
            self.logger.info("=== 開発診断実行 ===")

            # メモリ使用量チェック
            try:
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()

                self.logger.info(f"メモリ使用量: {memory_mb:.1f}MB")
                self.logger.info(f"CPU使用率: {cpu_percent:.1f}%")
            except ImportError:
                self.logger.info("psutil未インストール - メモリ・CPU監視スキップ")

            # 設定キャッシュ状態
            if self.config_manager:
                cache_info = self.config_manager.get_environment_info()["cache_status"]
                self.logger.info(f"設定キャッシュ: {cache_info['cache_size']}環境")

            # パフォーマンス監視状態
            if hasattr(self, "performance_monitor") and self.performance_monitor:
                try:
                    # パフォーマンス統計があれば表示
                    self.logger.info("パフォーマンス監視: 動作中")
                except Exception:
                    pass

        except Exception as e:
            self.logger.warning(f"開発診断エラー: {e}")

    def shutdown_development(self):
        """開発環境停止処理"""
        self.logger.info("開発環境停止処理開始")

        try:
            # 自動復旧システム停止
            if hasattr(self, "recovery_system") and self.recovery_system:
                self.recovery_system.stop_monitoring()
                self.logger.info("自動復旧システム停止完了")

            # 構造化ログでセッション終了を記録
            if hasattr(self, "structured_logger") and self.structured_logger:
                with self.structured_logger.operation_context(
                    "development_session_stop", "developer"
                ):
                    runtime = (datetime.now() - self.startup_time).total_seconds()
                    self.logger.info(f"開発セッション終了 - 稼働時間: {runtime:.1f}秒")

            self.logger.info("開発環境停止処理完了")

        except Exception as e:
            self.logger.error(f"開発環境停止処理エラー: {e}")


def main():
    """メイン実行関数"""
    print("Day Trade Development System - Starting")
    print("=" * 50)
    print("開発環境モード - デバッグ情報を詳細表示")
    print("Ctrl+C で終了")
    print()

    # システム管理インスタンス作成
    system_manager = DevelopmentSystemManager()

    try:
        # 開発環境初期化
        if not system_manager.initialize_development_environment():
            print("開発環境初期化失敗 - 終了")
            return 1

        print("開発環境初期化完了 - 開発セッション開始")
        print("ログは logs/development/development.log に保存されます")
        print()

        # 開発セッション開始
        system_manager.start_development_session()

        return 0

    except KeyboardInterrupt:
        print("\\nキーボード割り込み - 開発セッション終了")
        system_manager.shutdown_development()
        return 0

    except Exception as e:
        print(f"開発システム起動エラー: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
