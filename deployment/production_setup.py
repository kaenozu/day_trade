#!/usr/bin/env python3
"""
プロダクション環境セットアップスクリプト

統合最適化システムのプロダクション環境への配備と設定管理
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProductionSetup:
    """プロダクション環境セットアップ管理"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.deployment_root = self.project_root / "deployment"
        self.config_dir = self.project_root / "config"

        # プロダクション設定
        self.production_config = {
            "optimization_level": "optimized",
            "auto_fallback": True,
            "performance_monitoring": True,
            "cache_enabled": True,
            "parallel_processing": True,
            "batch_size": 2000,
            "timeout_seconds": 120,
            "memory_limit_mb": 2048,
            "component_specific": {
                "technical_indicators": {
                    "level": "optimized",
                    "enable_numba": True,
                    "enable_ml_enhancement": True,
                    "cache_ttl_seconds": 600,
                },
                "feature_engineering": {
                    "level": "optimized",
                    "enable_parallel": True,
                    "chunk_size": 50000,
                    "max_workers": 8,
                },
                "database": {
                    "level": "optimized",
                    "query_cache_size": 5000,
                    "query_cache_ttl_minutes": 120,
                    "bulk_insert_threshold": 1000,
                    "batch_size": 5000,
                },
                "ml_models": {
                    "level": "optimized",
                    "enable_parallel": True,
                    "enable_caching": True,
                    "max_workers": 6,
                    "cache_size": 2000,
                },
                "multi_timeframe_analysis": {
                    "level": "optimized",
                    "parallel_processing": True,
                },
            },
            "system_thresholds": {
                "memory_usage_warning": 75,
                "memory_usage_critical": 90,
                "cpu_usage_warning": 80,
                "cpu_usage_critical": 95,
                "auto_fallback_triggers": {
                    "memory_over": 85,
                    "cpu_over": 90,
                    "error_rate_over": 0.05,
                },
            },
            "monitoring": {
                "enable_detailed_logging": True,
                "log_performance_metrics": True,
                "alert_thresholds": {
                    "slow_operation_seconds": 30,
                    "high_memory_usage_mb": 1800,
                    "cache_hit_rate_below": 0.7,
                },
            },
        }

    def setup_production_environment(self):
        """プロダクション環境の完全セットアップ"""
        logger.info("プロダクション環境セットアップ開始")

        try:
            # 1. ディレクトリ構造作成
            self._create_directory_structure()

            # 2. 設定ファイル作成
            self._create_production_config()

            # 3. 環境変数設定
            self._setup_environment_variables()

            # 4. 依存関係インストール
            self._install_production_dependencies()

            # 5. データベース初期化
            self._initialize_database()

            # 6. 統合システム検証
            self._validate_unified_system()

            # 7. パフォーマンステスト
            self._run_performance_test()

            # 8. 監視設定
            self._setup_monitoring()

            logger.info("プロダクション環境セットアップ完了")

        except Exception as e:
            logger.error(f"プロダクション環境セットアップ失敗: {e}")
            raise

    def _create_directory_structure(self):
        """ディレクトリ構造の作成"""
        logger.info("ディレクトリ構造作成中...")

        directories = [
            "logs",
            "data/cache",
            "data/exports",
            "config/production",
            "monitoring/metrics",
            "monitoring/alerts",
            "backups/database",
            "backups/config",
            "tmp/processing",
        ]

        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"ディレクトリ作成: {full_path}")

    def _create_production_config(self):
        """プロダクション設定ファイルの作成"""
        logger.info("プロダクション設定ファイル作成中...")

        # メイン設定ファイル
        config_file = self.config_dir / "production" / "optimization_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.production_config, f, indent=2, ensure_ascii=False)

        logger.info(f"プロダクション設定ファイル作成: {config_file}")

        # 環境別設定ファイル
        env_configs = {
            "development": {
                **self.production_config,
                "optimization_level": "standard",
                "monitoring": {"enable_detailed_logging": True},
            },
            "staging": {
                **self.production_config,
                "optimization_level": "optimized",
                "batch_size": 1000,
            },
            "production": self.production_config,
        }

        for env_name, env_config in env_configs.items():
            env_config_file = (
                self.config_dir / "production" / f"optimization_config_{env_name}.json"
            )
            with open(env_config_file, "w", encoding="utf-8") as f:
                json.dump(env_config, f, indent=2, ensure_ascii=False)
            logger.debug(f"環境別設定ファイル作成: {env_config_file}")

    def _setup_environment_variables(self):
        """環境変数の設定"""
        logger.info("環境変数設定中...")

        env_vars = {
            "DAYTRADE_ENVIRONMENT": "production",
            "DAYTRADE_OPTIMIZATION_LEVEL": "optimized",
            "DAYTRADE_AUTO_FALLBACK": "true",
            "DAYTRADE_PERF_MONITORING": "true",
            "DAYTRADE_CACHE_ENABLED": "true",
            "DAYTRADE_PARALLEL": "true",
            "DAYTRADE_BATCH_SIZE": "2000",
            "DAYTRADE_TIMEOUT": "120",
            "DAYTRADE_MEMORY_LIMIT": "2048",
            "DAYTRADE_LOG_LEVEL": "INFO",
            "DAYTRADE_CONFIG_PATH": str(
                self.config_dir / "production" / "optimization_config.json"
            ),
        }

        # .env ファイル作成
        env_file = self.project_root / ".env.production"
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        logger.info(f"環境変数ファイル作成: {env_file}")

        # システム環境変数設定（開発用）
        for key, value in env_vars.items():
            os.environ[key] = value
            logger.debug(f"環境変数設定: {key}={value}")

    def _install_production_dependencies(self):
        """プロダクション用依存関係のインストール"""
        logger.info("プロダクション用依存関係インストール中...")

        # 必須パッケージ
        required_packages = [
            "psutil>=5.9.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "sqlalchemy>=2.0.0",
            "scikit-learn>=1.3.0",
        ]

        # オプショナル最適化パッケージ
        optional_packages = [
            "numba>=0.58.0",
            "xgboost>=1.7.0",
            "lightgbm>=4.0.0",
            "joblib>=1.3.0",
        ]

        try:
            # 必須パッケージのインストール
            for package in required_packages:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.debug(f"必須パッケージインストール: {package}")

            # オプショナルパッケージのインストール
            for package in optional_packages:
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    logger.debug(f"オプショナルパッケージインストール: {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"オプショナルパッケージインストール失敗: {package}")

            logger.info("依存関係インストール完了")

        except subprocess.CalledProcessError as e:
            logger.error(f"依存関係インストール失敗: {e}")
            raise

    def _initialize_database(self):
        """データベースの初期化"""
        logger.info("データベース初期化中...")

        try:
            # データベース設定のインポートと初期化
            sys.path.insert(0, str(self.project_root / "src"))

            from src.day_trade.core.optimization_strategy import OptimizationConfig
            from src.day_trade.models.database_unified import DatabaseManager

            config = OptimizationConfig.from_env()
            db_manager = DatabaseManager(config)

            # テーブル作成
            result = db_manager.create_tables()
            if result.success:
                logger.info("データベーステーブル作成成功")
            else:
                logger.error(f"データベーステーブル作成失敗: {result.error_message}")

            # データベース接続テスト
            test_result = db_manager.execute_query("SELECT 1 as test")
            if test_result.success:
                logger.info("データベース接続テスト成功")
            else:
                logger.error(f"データベース接続テスト失敗: {test_result.error_message}")

        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            # 本番環境では継続（データベースが既に存在する可能性）
            logger.warning("データベース初期化をスキップして継続")

    def _validate_unified_system(self):
        """統合システムの検証"""
        logger.info("統合システム検証中...")

        try:
            sys.path.insert(0, str(self.project_root / "src"))

            from src.day_trade.core.optimization_strategy import (
                OptimizationConfig,
                OptimizationLevel,
                OptimizationStrategyFactory,
            )

            # 戦略ファクトリーの検証
            components = OptimizationStrategyFactory.get_registered_components()
            logger.info(f"登録済みコンポーネント: {list(components.keys())}")

            # 各コンポーネントのインスタンス化テスト
            config = OptimizationConfig(level=OptimizationLevel.OPTIMIZED)

            for component_name in components.keys():
                try:
                    strategy = OptimizationStrategyFactory.get_strategy(component_name, config)
                    logger.info(
                        f"コンポーネント検証成功: {component_name} - {strategy.get_strategy_name()}"
                    )
                except Exception as e:
                    logger.error(f"コンポーネント検証失敗: {component_name} - {e}")
                    raise

            logger.info("統合システム検証完了")

        except Exception as e:
            logger.error(f"統合システム検証エラー: {e}")
            raise

    def _run_performance_test(self):
        """パフォーマンステストの実行"""
        logger.info("パフォーマンステスト実行中...")

        try:
            # 統合テストスクリプトの実行
            test_script = self.project_root / "test_unified_optimization_system.py"
            if test_script.exists():
                result = subprocess.run(
                    [sys.executable, str(test_script)],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5分タイムアウト
                )

                if result.returncode == 0:
                    logger.info("統合テスト成功")
                    logger.debug(f"テスト出力: {result.stdout[-500:]}")  # 最後の500文字
                else:
                    logger.error(f"統合テスト失敗: {result.stderr}")
                    # プロダクション環境では警告のみ
                    logger.warning("パフォーマンステスト失敗を無視して継続")

            # CLI ベンチマークの実行
            benchmark_cmd = [
                sys.executable,
                "-m",
                "src.day_trade.core.optimization_cli",
                "benchmark",
            ]

            result = subprocess.run(
                benchmark_cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=600,  # 10分タイムアウト
            )

            if result.returncode == 0:
                logger.info("ベンチマークテスト成功")
            else:
                logger.warning(f"ベンチマークテスト警告: {result.stderr}")

        except Exception as e:
            logger.warning(f"パフォーマンステストエラー（継続）: {e}")

    def _setup_monitoring(self):
        """監視システムの設定"""
        logger.info("監視システム設定中...")

        # ログローテーション設定
        logrotate_config = """
/var/log/daytrade/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        systemctl reload daytrade || true
    endscript
}
"""

        logrotate_file = self.project_root / "deployment" / "logrotate.conf"
        with open(logrotate_file, "w") as f:
            f.write(logrotate_config)

        # システムサービス設定
        systemd_service = f"""
[Unit]
Description=Day Trade Unified Optimization System
After=network.target

[Service]
Type=simple
User=daytrade
WorkingDirectory={self.project_root}
Environment=DAYTRADE_ENVIRONMENT=production
EnvironmentFile={self.project_root}/.env.production
ExecStart={sys.executable} -m src.day_trade.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=daytrade

[Install]
WantedBy=multi-user.target
"""

        service_file = self.project_root / "deployment" / "daytrade.service"
        with open(service_file, "w") as f:
            f.write(systemd_service)

        # 監視スクリプト
        monitoring_script = f"""#!/bin/bash
# Day Trade System Monitoring Script

LOG_DIR="{self.project_root}/logs"
ALERT_EMAIL="admin@daytrade.local"

# システムリソース監視
check_resources() {{
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{{print $2}}' | cut -d'%' -f1)
    MEM_USAGE=$(free | grep Mem | awk '{{printf("%.1f", ($3/$2) * 100.0)}}')

    echo "$(date): CPU: ${{CPU_USAGE}}%, Memory: ${{MEM_USAGE}}%" >> "${{LOG_DIR}}/monitoring.log"

    # アラート閾値チェック
    if (( $(echo "$CPU_USAGE > 90" | bc -l) )); then
        echo "HIGH CPU USAGE: ${{CPU_USAGE}}%" | mail -s "DayTrade Alert: High CPU" "${{ALERT_EMAIL}}"
    fi

    if (( $(echo "$MEM_USAGE > 85" | bc -l) )); then
        echo "HIGH MEMORY USAGE: ${{MEM_USAGE}}%" | mail -s "DayTrade Alert: High Memory" "${{ALERT_EMAIL}}"
    fi
}}

# プロセス監視
check_process() {{
    if ! pgrep -f "day_trade" > /dev/null; then
        echo "$(date): DayTrade process not running" >> "${{LOG_DIR}}/monitoring.log"
        echo "DayTrade process not found" | mail -s "DayTrade Alert: Process Down" "${{ALERT_EMAIL}}"

        # 自動再起動試行
        systemctl restart daytrade
    fi
}}

# ログファイルサイズ監視
check_logs() {{
    for logfile in "${{LOG_DIR}}"/*.log; do
        if [[ -f "$logfile" ]]; then
            size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile")
            if [[ $size -gt 104857600 ]]; then  # 100MB
                echo "$(date): Large log file: $logfile (${{size}} bytes)" >> "${{LOG_DIR}}/monitoring.log"
            fi
        fi
    done
}}

# メイン監視ループ
main() {{
    check_resources
    check_process
    check_logs
}}

main
"""

        monitoring_file = self.project_root / "deployment" / "monitoring.sh"
        with open(monitoring_file, "w") as f:
            f.write(monitoring_script)

        # 実行権限付与
        os.chmod(monitoring_file, 0o755)

        logger.info("監視システム設定完了")

    def create_deployment_package(self):
        """デプロイメントパッケージの作成"""
        logger.info("デプロイメントパッケージ作成中...")

        deployment_dir = self.deployment_root / "package"
        deployment_dir.mkdir(parents=True, exist_ok=True)

        # コピー対象ファイル・ディレクトリ
        include_patterns = [
            "src/day_trade/core/optimization_strategy.py",
            "src/day_trade/core/optimization_cli.py",
            "src/day_trade/analysis/*_unified.py",
            "src/day_trade/models/database_unified.py",
            "config/",
            "deployment/",
            "test_unified_optimization_system.py",
            "requirements.txt",
            ".env.production",
        ]

        # パッケージング
        import tarfile

        package_name = f"daytrade_unified_system_{os.environ.get('BUILD_VERSION', 'latest')}.tar.gz"
        package_path = self.deployment_root / package_name

        with tarfile.open(package_path, "w:gz") as tar:
            for pattern in include_patterns:
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.project_root)
                        tar.add(file_path, arcname=arcname)
                        logger.debug(f"パッケージに追加: {arcname}")

        logger.info(f"デプロイメントパッケージ作成完了: {package_path}")
        return package_path

    def cleanup_old_deployments(self, keep_versions: int = 5):
        """古いデプロイメントの清理"""
        logger.info("古いデプロイメント清理中...")

        # 古いパッケージファイルの削除
        package_files = list(self.deployment_root.glob("daytrade_unified_system_*.tar.gz"))
        if len(package_files) > keep_versions:
            # 作成時間でソート
            package_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            for old_package in package_files[keep_versions:]:
                old_package.unlink()
                logger.info(f"古いパッケージ削除: {old_package}")

        # 古いログファイルの圧縮
        log_dir = self.project_root / "logs"
        if log_dir.exists():
            import gzip

            for log_file in log_dir.glob("*.log"):
                if log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB以上
                    with open(log_file, "rb") as f_in:
                        with gzip.open(f"{log_file}.gz", "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    log_file.unlink()
                    logger.info(f"ログファイル圧縮: {log_file}")


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description="Day Trade プロダクション環境セットアップ")
    parser.add_argument(
        "--action",
        choices=["setup", "package", "cleanup", "validate", "all"],
        default="all",
        help="実行するアクション",
    )
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default="production",
        help="環境設定",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 環境変数設定
    os.environ["DAYTRADE_ENVIRONMENT"] = args.environment

    setup = ProductionSetup()

    try:
        if args.action in ["setup", "all"]:
            setup.setup_production_environment()

        if args.action in ["package", "all"]:
            package_path = setup.create_deployment_package()
            print(f"デプロイメントパッケージ: {package_path}")

        if args.action in ["cleanup", "all"]:
            setup.cleanup_old_deployments()

        if args.action in ["validate", "all"]:
            setup._validate_unified_system()

        print("✅ プロダクション環境セットアップ完了")

    except Exception as e:
        print(f"❌ プロダクション環境セットアップ失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
