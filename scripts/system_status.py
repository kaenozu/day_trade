#!/usr/bin/env python3
"""
システム状態確認スクリプト

Issue #320: システム本番稼働準備・設定最適化
システム稼働状況・パフォーマンス・ヘルス状態の確認
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from day_trade.config.environment_config import get_environment_config_manager
    from day_trade.utils.advanced_fault_tolerance import get_recovery_system
    from day_trade.utils.performance_monitor import get_performance_monitor
    from day_trade.utils.structured_logging import get_structured_logger
except ImportError as e:
    print(f"モジュールインポートエラー: {e}")
    print("一部機能が制限される可能性があります")


class SystemStatusChecker:
    """システム状態確認"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.pid_file = self.project_root / "daytrade.pid"

    def check_system_status(self) -> Dict[str, Any]:
        """システム状態総合確認"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "system_running": False,
            "process_info": {},
            "environment": {},
            "performance": {},
            "health": {},
            "resources": {},
            "logs": {},
            "errors": [],
        }

        try:
            # プロセス状態確認
            status["process_info"] = self._check_process_status()
            status["system_running"] = status["process_info"].get("running", False)

            # 環境設定確認
            status["environment"] = self._check_environment_status()

            # パフォーマンス確認
            status["performance"] = self._check_performance_status()

            # ヘルス状態確認
            status["health"] = self._check_health_status()

            # システムリソース確認
            status["resources"] = self._check_system_resources()

            # ログ状態確認
            status["logs"] = self._check_log_status()

        except Exception as e:
            status["errors"].append(f"システム状態確認エラー: {str(e)}")

        return status

    def _check_process_status(self) -> Dict[str, Any]:
        """プロセス状態確認"""
        process_info = {
            "running": False,
            "pid": None,
            "start_time": None,
            "uptime_seconds": None,
            "memory_usage_mb": None,
            "cpu_percent": None,
        }

        try:
            # PIDファイル確認
            if self.pid_file.exists():
                with open(self.pid_file) as f:
                    pid = int(f.read().strip())

                process_info["pid"] = pid

                # プロセス存在確認
                if self._process_exists(pid):
                    process_info["running"] = True

                    # プロセス詳細情報取得
                    try:
                        import psutil

                        process = psutil.Process(pid)

                        process_info["start_time"] = datetime.fromtimestamp(
                            process.create_time()
                        ).isoformat()

                        process_info["uptime_seconds"] = (
                            time.time() - process.create_time()
                        )
                        process_info["memory_usage_mb"] = (
                            process.memory_info().rss / 1024 / 1024
                        )
                        process_info["cpu_percent"] = process.cpu_percent()

                    except ImportError:
                        process_info["memory_usage_mb"] = "psutil未インストール"
                        process_info["cpu_percent"] = "psutil未インストール"
                    except Exception as e:
                        process_info["error"] = str(e)
                else:
                    process_info["running"] = False
                    process_info["error"] = "プロセス存在せず（PIDファイルは残存）"
            else:
                process_info["error"] = "PIDファイル存在せず"

        except Exception as e:
            process_info["error"] = str(e)

        return process_info

    def _process_exists(self, pid: int) -> bool:
        """プロセス存在確認"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _check_environment_status(self) -> Dict[str, Any]:
        """環境設定状態確認"""
        env_status = {
            "current_environment": "unknown",
            "config_loaded": False,
            "config_directory": str(self.project_root / "config"),
            "available_configs": [],
            "environment_variables": {},
            "validation_status": "unknown",
        }

        try:
            config_mgr = get_environment_config_manager()
            env_info = config_mgr.get_environment_info()

            env_status.update(
                {
                    "current_environment": env_info["current_environment"],
                    "config_loaded": True,
                    "config_directory": env_info["config_directory"],
                    "available_configs": env_info["available_configs"],
                    "environment_variables": env_info["environment_variables"],
                    "cache_status": env_info["cache_status"],
                }
            )

            # 設定検証
            try:
                config = config_mgr.load_config()
                env_status["validation_status"] = "valid"
                env_status["config_sections"] = list(config.keys())
            except Exception as e:
                env_status["validation_status"] = f"invalid: {e}"

        except Exception as e:
            env_status["error"] = str(e)

        return env_status

    def _check_performance_status(self) -> Dict[str, Any]:
        """パフォーマンス状態確認"""
        perf_status = {
            "monitoring_available": False,
            "recent_metrics": {},
            "target_performance": {},
            "alerts": [],
        }

        try:
            perf_monitor = get_performance_monitor()
            perf_status["monitoring_available"] = True

            # 最近のパフォーマンス統計
            if hasattr(perf_monitor, "get_performance_summary"):
                summary = perf_monitor.get_performance_summary(hours=1)
                perf_status["recent_metrics"] = summary

            # 目標パフォーマンス設定
            try:
                config_mgr = get_environment_config_manager()
                system_config = config_mgr.get_system_config()
                perf_status["target_performance"] = {
                    "ml_analysis_target_seconds": system_config.performance_target_seconds,
                    "memory_limit_mb": system_config.memory_limit_mb,
                    "cpu_limit_percent": system_config.cpu_limit_percent,
                }
            except Exception:
                pass

        except Exception as e:
            perf_status["error"] = str(e)

        return perf_status

    def _check_health_status(self) -> Dict[str, Any]:
        """ヘルス状態確認"""
        health_status = {
            "recovery_system_available": False,
            "monitoring_active": False,
            "degradation_level": 0,
            "recent_recovery_actions": 0,
            "uptime_estimate": {},
            "data_sources": {},
        }

        try:
            recovery_system = get_recovery_system()
            health_status["recovery_system_available"] = True

            # システム状態取得
            system_status = recovery_system.get_system_status()

            health_status.update(
                {
                    "monitoring_active": system_status["monitoring_active"],
                    "degradation_level": system_status["degradation"]["level"],
                    "recent_recovery_actions": system_status["recent_recovery_actions"],
                    "uptime_estimate": system_status["uptime_estimate"],
                    "data_sources": system_status["data_sources"],
                }
            )

        except Exception as e:
            health_status["error"] = str(e)

        return health_status

    def _check_system_resources(self) -> Dict[str, Any]:
        """システムリソース確認"""
        resources = {"disk_space": {}, "memory": {}, "cpu": {}, "network": {}}

        try:
            # ディスク容量
            import shutil

            total, used, free = shutil.disk_usage(self.project_root)
            resources["disk_space"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100,
            }

            # メモリ・CPU（psutilが利用可能な場合）
            try:
                import psutil

                memory = psutil.virtual_memory()
                resources["memory"] = {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "usage_percent": memory.percent,
                }

                resources["cpu"] = {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "core_count": psutil.cpu_count(),
                }

                # ネットワーク接続テスト
                resources["network"] = self._test_network_connectivity()

            except ImportError:
                resources["memory"]["error"] = "psutil未インストール"
                resources["cpu"]["error"] = "psutil未インストール"

        except Exception as e:
            resources["error"] = str(e)

        return resources

    def _test_network_connectivity(self) -> Dict[str, Any]:
        """ネットワーク接続テスト"""
        connectivity = {"internet": False, "api_endpoints": {}}

        # インターネット接続テスト
        try:
            import urllib.request

            urllib.request.urlopen("https://www.google.com", timeout=5)
            connectivity["internet"] = True
        except Exception:
            connectivity["internet"] = False

        # API エンドポイントテスト
        api_endpoints = [
            ("Yahoo Finance", "https://finance.yahoo.com"),
            ("Alpha Vantage", "https://www.alphavantage.co"),
        ]

        for name, url in api_endpoints:
            try:
                import urllib.request

                urllib.request.urlopen(url, timeout=5)
                connectivity["api_endpoints"][name] = True
            except Exception:
                connectivity["api_endpoints"][name] = False

        return connectivity

    def _check_log_status(self) -> Dict[str, Any]:
        """ログ状態確認"""
        log_status = {
            "log_directories": {},
            "recent_errors": 0,
            "log_file_sizes": {},
            "structured_logging_available": False,
        }

        try:
            # ログディレクトリ確認
            log_dirs = [
                self.project_root / "logs" / "production",
                self.project_root / "logs" / "development",
                self.project_root / "logs",
            ]

            for log_dir in log_dirs:
                if log_dir.exists():
                    log_files = list(log_dir.glob("*.log"))
                    log_status["log_directories"][str(log_dir)] = {
                        "exists": True,
                        "file_count": len(log_files),
                        "files": [f.name for f in log_files[:10]],  # 最大10ファイル表示
                    }
                else:
                    log_status["log_directories"][str(log_dir)] = {"exists": False}

            # 構造化ログシステム確認
            try:
                structured_logger = get_structured_logger()
                log_status["structured_logging_available"] = True

                # エラーレポート生成
                error_report = structured_logger.generate_error_report(hours=24)
                log_status["recent_errors"] = error_report["error_count"]

            except Exception:
                pass

        except Exception as e:
            log_status["error"] = str(e)

        return log_status

    def display_status_report(self, status: Dict[str, Any]):
        """状態レポート表示"""
        print("=" * 60)
        print("SYSTEM STATUS REPORT")
        print("=" * 60)
        print(f"Timestamp: {status['timestamp']}")
        print(f"System Running: {'YES' if status['system_running'] else 'NO'}")
        print()

        # プロセス情報
        self._display_process_info(status["process_info"])

        # 環境情報
        self._display_environment_info(status["environment"])

        # パフォーマンス情報
        self._display_performance_info(status["performance"])

        # ヘルス情報
        self._display_health_info(status["health"])

        # リソース情報
        self._display_resource_info(status["resources"])

        # ログ情報
        self._display_log_info(status["logs"])

        # エラー情報
        if status["errors"]:
            print("\n=== ERRORS ===")
            for error in status["errors"]:
                print(f"  - {error}")

    def _display_process_info(self, process_info: Dict[str, Any]):
        """プロセス情報表示"""
        print("=== PROCESS INFORMATION ===")

        if process_info.get("running"):
            print("  Status: RUNNING")
            print(f"  PID: {process_info.get('pid', 'N/A')}")

            if process_info.get("uptime_seconds"):
                uptime = timedelta(seconds=int(process_info["uptime_seconds"]))
                print(f"  Uptime: {uptime}")

            if process_info.get("memory_usage_mb"):
                print(f"  Memory Usage: {process_info['memory_usage_mb']:.1f} MB")

            if process_info.get("cpu_percent"):
                print(f"  CPU Usage: {process_info['cpu_percent']:.1f}%")
        else:
            print("  Status: NOT RUNNING")
            if process_info.get("error"):
                print(f"  Error: {process_info['error']}")

        print()

    def _display_environment_info(self, env_info: Dict[str, Any]):
        """環境情報表示"""
        print("=== ENVIRONMENT INFORMATION ===")
        print(
            f"  Current Environment: {env_info.get('current_environment', 'Unknown')}"
        )
        print(f"  Config Loaded: {env_info.get('config_loaded', False)}")
        print(f"  Config Directory: {env_info.get('config_directory', 'N/A')}")
        print(
            f"  Available Configs: {', '.join(env_info.get('available_configs', []))}"
        )
        print(f"  Validation Status: {env_info.get('validation_status', 'Unknown')}")

        env_vars = env_info.get("environment_variables", {})
        if env_vars:
            print(f"  Environment Variables: {', '.join(env_vars.keys())}")

        print()

    def _display_performance_info(self, perf_info: Dict[str, Any]):
        """パフォーマンス情報表示"""
        print("=== PERFORMANCE INFORMATION ===")
        print(f"  Monitoring Available: {perf_info.get('monitoring_available', False)}")

        target_perf = perf_info.get("target_performance", {})
        if target_perf:
            print(
                f"  ML Analysis Target: {target_perf.get('ml_analysis_target_seconds', 'N/A')}s"
            )
            print(f"  Memory Limit: {target_perf.get('memory_limit_mb', 'N/A')} MB")
            print(f"  CPU Limit: {target_perf.get('cpu_limit_percent', 'N/A')}%")

        recent_metrics = perf_info.get("recent_metrics", {})
        if recent_metrics and "error" not in recent_metrics:
            print("  Recent Performance:")
            for key, value in recent_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")

        print()

    def _display_health_info(self, health_info: Dict[str, Any]):
        """ヘルス情報表示"""
        print("=== HEALTH INFORMATION ===")
        print(
            f"  Recovery System: {health_info.get('recovery_system_available', False)}"
        )
        print(f"  Monitoring Active: {health_info.get('monitoring_active', False)}")
        print(f"  Degradation Level: {health_info.get('degradation_level', 0)}")
        print(
            f"  Recent Recovery Actions: {health_info.get('recent_recovery_actions', 0)}"
        )

        uptime_est = health_info.get("uptime_estimate", {})
        if uptime_est:
            print(f"  Uptime Estimate: {uptime_est.get('uptime_percent', 0):.1f}%")

        print()

    def _display_resource_info(self, resource_info: Dict[str, Any]):
        """リソース情報表示"""
        print("=== RESOURCE INFORMATION ===")

        disk = resource_info.get("disk_space", {})
        if disk and "error" not in disk:
            print(
                f"  Disk Usage: {disk.get('usage_percent', 0):.1f}% ({disk.get('free_gb', 0):.1f} GB free)"
            )

        memory = resource_info.get("memory", {})
        if memory and "error" not in memory:
            print(
                f"  Memory Usage: {memory.get('usage_percent', 0):.1f}% ({memory.get('available_gb', 0):.1f} GB available)"
            )

        cpu = resource_info.get("cpu", {})
        if cpu and "error" not in cpu:
            print(
                f"  CPU Usage: {cpu.get('usage_percent', 0):.1f}% ({cpu.get('core_count', 0)} cores)"
            )

        network = resource_info.get("network", {})
        if network:
            print(f"  Internet Connection: {network.get('internet', False)}")

        print()

    def _display_log_info(self, log_info: Dict[str, Any]):
        """ログ情報表示"""
        print("=== LOG INFORMATION ===")
        print(f"  Recent Errors (24h): {log_info.get('recent_errors', 0)}")
        print(
            f"  Structured Logging: {log_info.get('structured_logging_available', False)}"
        )

        log_dirs = log_info.get("log_directories", {})
        for log_dir, info in log_dirs.items():
            if info.get("exists"):
                print(f"  {log_dir}: {info.get('file_count', 0)} files")

        print()


def main():
    """メイン実行関数"""
    print("Day Trade System Status Checker")
    print("=" * 40)

    checker = SystemStatusChecker()

    try:
        # システム状態確認
        status = checker.check_system_status()

        # コマンドライン引数確認
        if len(sys.argv) > 1 and sys.argv[1].lower() == "json":
            # JSON形式で出力
            print(json.dumps(status, indent=2, ensure_ascii=False))
        else:
            # 人間が読みやすい形式で出力
            checker.display_status_report(status)

        return 0

    except Exception as e:
        print(f"システム状態確認エラー: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
