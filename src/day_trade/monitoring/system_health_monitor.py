#!/usr/bin/env python3
"""
システムヘルス監視システム
Issue #318: 監視・アラートシステム - Phase 1

各コンポーネント稼働状況監視・依存サービス監視・自動診断復旧
- リアルタイムヘルスチェック
- 依存サービス接続監視
- リソース使用量監視
- 自動診断・復旧機能
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import numpy as np
import psutil

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class HealthStatus(Enum):
    """ヘルス状態"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """コンポーネントタイプ"""

    DATABASE = "database"
    API_SERVICE = "api_service"
    ML_ENGINE = "ml_engine"
    CACHE_SYSTEM = "cache_system"
    FILE_SYSTEM = "file_system"
    NETWORK_SERVICE = "network_service"
    BACKGROUND_PROCESS = "background_process"


@dataclass
class HealthCheckConfig:
    """ヘルスチェック設定"""

    # チェック間隔設定
    check_interval_seconds: int = 30
    critical_check_interval_seconds: int = 10
    health_history_retention_hours: int = 24

    # しきい値設定
    cpu_warning_threshold: float = 70.0  # %
    cpu_critical_threshold: float = 90.0  # %
    memory_warning_threshold: float = 80.0  # %
    memory_critical_threshold: float = 95.0  # %
    disk_warning_threshold: float = 85.0  # %
    disk_critical_threshold: float = 95.0  # %

    # ネットワーク設定
    network_timeout_seconds: int = 5
    max_response_time_ms: float = 1000.0

    # 自動復旧設定
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_cooldown_minutes: int = 15


@dataclass
class HealthCheckResult:
    """ヘルスチェック結果"""

    component_id: str
    component_type: ComponentType
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_success: bool = False


@dataclass
class SystemResourceMetrics:
    """システムリソースメトリクス"""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_connections: int
    process_count: int
    load_average: List[float] = field(default_factory=list)
    temperature: Optional[float] = None


class SystemHealthMonitor:
    """システムヘルス監視システム"""

    def __init__(self, config: Optional[HealthCheckConfig] = None):
        self.config = config or HealthCheckConfig()
        self.components: Dict[str, Dict[str, Any]] = {}
        self.health_history: List[HealthCheckResult] = []
        self.resource_metrics: List[SystemResourceMetrics] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_attempt: Dict[str, datetime] = {}

        # 監視タスク管理
        self.monitor_task: Optional[asyncio.Task] = None
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

        # カスタム復旧ハンドラー
        self.recovery_handlers: Dict[str, Callable] = {}

    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        check_function: Callable,
        **kwargs,
    ) -> None:
        """コンポーネント登録"""
        self.components[component_id] = {
            "type": component_type,
            "check_function": check_function,
            "config": kwargs,
            "last_check": None,
            "consecutive_failures": 0,
        }

        logger.info(f"コンポーネント登録: {component_id} ({component_type.value})")

    def register_recovery_handler(
        self, component_id: str, recovery_function: Callable
    ) -> None:
        """復旧ハンドラー登録"""
        self.recovery_handlers[component_id] = recovery_function
        logger.info(f"復旧ハンドラー登録: {component_id}")

    async def start_monitoring(self) -> None:
        """監視開始"""
        if self._is_running:
            logger.warning("監視は既に実行中です")
            return

        self._is_running = True

        # ヘルスチェック監視タスク
        self.monitor_task = asyncio.create_task(self._health_monitor_loop())

        # リソース監視タスク
        self.resource_monitor_task = asyncio.create_task(self._resource_monitor_loop())

        logger.info("システムヘルス監視開始")

    async def stop_monitoring(self) -> None:
        """監視停止"""
        self._is_running = False

        # タスク停止
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
            try:
                await self.resource_monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("システムヘルス監視停止")

    async def _health_monitor_loop(self) -> None:
        """ヘルスチェック監視ループ"""
        while self._is_running:
            try:
                # 全コンポーネントヘルスチェック
                check_tasks = []
                for component_id in self.components.keys():
                    task = asyncio.create_task(
                        self._check_component_health(component_id)
                    )
                    check_tasks.append(task)

                # 並列実行
                if check_tasks:
                    results = await asyncio.gather(*check_tasks, return_exceptions=True)

                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            component_id = list(self.components.keys())[i]
                            logger.error(
                                f"ヘルスチェックエラー {component_id}: {result}"
                            )

                # 履歴クリーンアップ
                await self._cleanup_health_history()

                # 動的間隔調整
                interval = self._calculate_check_interval()
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ヘルス監視ループエラー: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)

    async def _resource_monitor_loop(self) -> None:
        """リソース監視ループ"""
        while self._is_running:
            try:
                # システムリソース取得
                metrics = await self._collect_resource_metrics()
                self.resource_metrics.append(metrics)

                # リソースアラートチェック
                await self._check_resource_alerts(metrics)

                # 古いメトリクス削除
                cutoff_time = datetime.now() - timedelta(
                    hours=self.config.health_history_retention_hours
                )
                self.resource_metrics = [
                    m for m in self.resource_metrics if m.timestamp > cutoff_time
                ]

                # 60秒間隔でリソース監視
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"リソース監視ループエラー: {e}")
                await asyncio.sleep(60)

    async def _check_component_health(self, component_id: str) -> HealthCheckResult:
        """個別コンポーネントヘルスチェック"""
        component_info = self.components[component_id]
        check_function = component_info["check_function"]

        start_time = time.time()

        try:
            # ヘルスチェック実行
            result = await self._execute_health_check(
                check_function, component_info["config"]
            )
            response_time = (time.time() - start_time) * 1000

            # 結果判定
            if result.get("healthy", False):
                status = HealthStatus.HEALTHY
                error_message = None
                component_info["consecutive_failures"] = 0
            else:
                status = HealthStatus.CRITICAL
                error_message = result.get("error", "不明なエラー")
                component_info["consecutive_failures"] += 1

            # ヘルスチェック結果作成
            health_result = HealthCheckResult(
                component_id=component_id,
                component_type=component_info["type"],
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                error_message=error_message,
                details=result.get("details", {}),
            )

            # 自動復旧判定
            if (
                status == HealthStatus.CRITICAL
                and self.config.enable_auto_recovery
                and component_info["consecutive_failures"] >= 2
            ):
                await self._attempt_auto_recovery(component_id, health_result)

            # 履歴記録
            self.health_history.append(health_result)
            component_info["last_check"] = datetime.now()

            return health_result

        except Exception as e:
            logger.error(f"ヘルスチェック実行エラー {component_id}: {e}")

            error_result = HealthCheckResult(
                component_id=component_id,
                component_type=component_info["type"],
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e),
            )

            self.health_history.append(error_result)
            return error_result

    async def _execute_health_check(
        self, check_function: Callable, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ヘルスチェック関数実行"""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function(**config)
        else:
            return check_function(**config)

    async def _attempt_auto_recovery(
        self, component_id: str, health_result: HealthCheckResult
    ) -> None:
        """自動復旧試行"""
        current_time = datetime.now()

        # 復旧試行制限チェック
        if (
            self.recovery_attempts.get(component_id, 0)
            >= self.config.max_recovery_attempts
        ):
            last_attempt = self.last_recovery_attempt.get(component_id)
            if last_attempt and current_time - last_attempt < timedelta(
                minutes=self.config.recovery_cooldown_minutes
            ):
                return
            else:
                # クールダウン期間経過後リセット
                self.recovery_attempts[component_id] = 0

        # 復旧ハンドラー実行
        if component_id in self.recovery_handlers:
            try:
                logger.info(f"自動復旧試行開始: {component_id}")

                recovery_function = self.recovery_handlers[component_id]
                if asyncio.iscoroutinefunction(recovery_function):
                    success = await recovery_function()
                else:
                    success = recovery_function()

                # 復旧結果記録
                health_result.recovery_attempted = True
                health_result.recovery_success = success

                self.recovery_attempts[component_id] = (
                    self.recovery_attempts.get(component_id, 0) + 1
                )
                self.last_recovery_attempt[component_id] = current_time

                if success:
                    logger.info(f"自動復旧成功: {component_id}")
                    self.components[component_id]["consecutive_failures"] = 0
                else:
                    logger.warning(f"自動復旧失敗: {component_id}")

            except Exception as e:
                logger.error(f"自動復旧エラー {component_id}: {e}")
                health_result.recovery_attempted = True
                health_result.recovery_success = False

    async def _collect_resource_metrics(self) -> SystemResourceMetrics:
        """システムリソースメトリクス収集"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)

        # メモリ使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # ディスク使用率（ルートパーティション）
        disk = psutil.disk_usage("/")
        disk_percent = (disk.used / disk.total) * 100

        # ネットワーク接続数
        network_connections = len(psutil.net_connections())

        # プロセス数
        process_count = len(psutil.pids())

        # 負荷平均（Linux/Unix系）
        load_average = []
        try:
            if hasattr(os, "getloadavg"):
                load_average = list(os.getloadavg())
        except:
            pass

        # システム温度（可能な場合）
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # CPU温度を優先取得
                for name, entries in temps.items():
                    if "cpu" in name.lower() or "core" in name.lower():
                        temperature = entries[0].current
                        break
        except:
            pass

        return SystemResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            network_connections=network_connections,
            process_count=process_count,
            load_average=load_average,
            temperature=temperature,
        )

    async def _check_resource_alerts(self, metrics: SystemResourceMetrics) -> None:
        """リソースアラートチェック"""
        alerts = []

        # CPU使用率アラート
        if metrics.cpu_percent >= self.config.cpu_critical_threshold:
            alerts.append(f"CPU使用率CRITICAL: {metrics.cpu_percent:.1f}%")
        elif metrics.cpu_percent >= self.config.cpu_warning_threshold:
            alerts.append(f"CPU使用率WARNING: {metrics.cpu_percent:.1f}%")

        # メモリ使用率アラート
        if metrics.memory_percent >= self.config.memory_critical_threshold:
            alerts.append(f"メモリ使用率CRITICAL: {metrics.memory_percent:.1f}%")
        elif metrics.memory_percent >= self.config.memory_warning_threshold:
            alerts.append(f"メモリ使用率WARNING: {metrics.memory_percent:.1f}%")

        # ディスク使用率アラート
        if metrics.disk_percent >= self.config.disk_critical_threshold:
            alerts.append(f"ディスク使用率CRITICAL: {metrics.disk_percent:.1f}%")
        elif metrics.disk_percent >= self.config.disk_warning_threshold:
            alerts.append(f"ディスク使用率WARNING: {metrics.disk_percent:.1f}%")

        # アラート記録
        for alert in alerts:
            logger.warning(alert)

    def _calculate_check_interval(self) -> int:
        """動的チェック間隔計算"""
        # 過去1時間の重要なアラート数をチェック
        recent_time = datetime.now() - timedelta(hours=1)
        recent_critical = [
            r
            for r in self.health_history
            if r.timestamp > recent_time and r.status == HealthStatus.CRITICAL
        ]

        if len(recent_critical) > 5:
            # 多数の重要アラートがある場合は頻繁にチェック
            return self.config.critical_check_interval_seconds
        else:
            return self.config.check_interval_seconds

    async def _cleanup_health_history(self) -> None:
        """ヘルス履歴クリーンアップ"""
        cutoff_time = datetime.now() - timedelta(
            hours=self.config.health_history_retention_hours
        )

        self.health_history = [
            result for result in self.health_history if result.timestamp > cutoff_time
        ]

    async def get_system_health_summary(self) -> Dict[str, Any]:
        """システムヘルス概要取得"""
        current_time = datetime.now()

        # 最新のコンポーネントステータス
        component_status = {}
        for component_id, component_info in self.components.items():
            latest_result = None
            for result in reversed(self.health_history):
                if result.component_id == component_id:
                    latest_result = result
                    break

            if latest_result:
                component_status[component_id] = {
                    "status": latest_result.status.value,
                    "last_check": latest_result.timestamp.isoformat(),
                    "response_time_ms": latest_result.response_time_ms,
                    "consecutive_failures": component_info["consecutive_failures"],
                }
            else:
                component_status[component_id] = {
                    "status": "unknown",
                    "last_check": None,
                    "response_time_ms": 0,
                    "consecutive_failures": 0,
                }

        # 最新のリソースメトリクス
        latest_metrics = self.resource_metrics[-1] if self.resource_metrics else None

        # 全体ステータス判定
        overall_status = self._determine_overall_status(component_status)

        return {
            "overall_status": overall_status,
            "timestamp": current_time.isoformat(),
            "components": component_status,
            "resource_metrics": (
                {
                    "cpu_percent": (
                        latest_metrics.cpu_percent if latest_metrics else None
                    ),
                    "memory_percent": (
                        latest_metrics.memory_percent if latest_metrics else None
                    ),
                    "disk_percent": (
                        latest_metrics.disk_percent if latest_metrics else None
                    ),
                    "network_connections": (
                        latest_metrics.network_connections if latest_metrics else None
                    ),
                    "process_count": (
                        latest_metrics.process_count if latest_metrics else None
                    ),
                }
                if latest_metrics
                else {}
            ),
            "monitoring_status": {
                "is_running": self._is_running,
                "registered_components": len(self.components),
                "total_health_checks": len(self.health_history),
                "recovery_attempts_total": sum(self.recovery_attempts.values()),
            },
        }

    def _determine_overall_status(
        self, component_status: Dict[str, Dict[str, Any]]
    ) -> str:
        """全体ステータス判定"""
        if not component_status:
            return "unknown"

        critical_count = sum(
            1 for status in component_status.values() if status["status"] == "critical"
        )
        warning_count = sum(
            1 for status in component_status.values() if status["status"] == "warning"
        )

        if critical_count > 0:
            return "critical"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"

    async def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """ヘルストレンド分析"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 期間内の履歴フィルタ
        recent_history = [
            result for result in self.health_history if result.timestamp > cutoff_time
        ]

        # コンポーネント別統計
        component_stats = {}
        for component_id in self.components.keys():
            component_history = [
                r for r in recent_history if r.component_id == component_id
            ]

            if component_history:
                total_checks = len(component_history)
                healthy_checks = sum(
                    1 for r in component_history if r.status == HealthStatus.HEALTHY
                )
                avg_response_time = (
                    sum(r.response_time_ms for r in component_history) / total_checks
                )

                component_stats[component_id] = {
                    "total_checks": total_checks,
                    "healthy_rate": (healthy_checks / total_checks) * 100,
                    "avg_response_time_ms": avg_response_time,
                    "uptime_percentage": (healthy_checks / total_checks) * 100,
                }

        # リソーストレンド
        recent_resources = [
            m for m in self.resource_metrics if m.timestamp > cutoff_time
        ]
        resource_trends = {}

        if recent_resources:
            resource_trends = {
                "avg_cpu_percent": np.mean([m.cpu_percent for m in recent_resources]),
                "max_cpu_percent": np.max([m.cpu_percent for m in recent_resources]),
                "avg_memory_percent": np.mean(
                    [m.memory_percent for m in recent_resources]
                ),
                "max_memory_percent": np.max(
                    [m.memory_percent for m in recent_resources]
                ),
                "avg_disk_percent": np.mean([m.disk_percent for m in recent_resources]),
                "samples": len(recent_resources),
            }

        return {
            "analysis_period_hours": hours,
            "component_statistics": component_stats,
            "resource_trends": resource_trends,
            "total_health_checks": len(recent_history),
        }

    async def force_health_check(
        self, component_id: Optional[str] = None
    ) -> List[HealthCheckResult]:
        """強制ヘルスチェック実行"""
        if component_id:
            if component_id not in self.components:
                raise ValueError(f"未登録のコンポーネント: {component_id}")
            result = await self._check_component_health(component_id)
            return [result]
        else:
            # 全コンポーネントチェック
            tasks = [
                self._check_component_health(comp_id)
                for comp_id in self.components.keys()
            ]
            return await asyncio.gather(*tasks)


# 標準ヘルスチェック関数


async def database_health_check(**kwargs) -> Dict[str, Any]:
    """データベースヘルスチェック"""
    # 実際の実装ではデータベース接続テストを行う
    connection_string = kwargs.get("connection_string")
    timeout = kwargs.get("timeout", 5)

    try:
        # 模擬データベース接続チェック
        await asyncio.sleep(0.1)  # 実際の接続処理の代替

        return {
            "healthy": True,
            "details": {
                "connection_test": "passed",
                "response_time_ms": 100,
                "active_connections": 5,
            },
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "details": {"connection_test": "failed"},
        }


async def api_service_health_check(**kwargs) -> Dict[str, Any]:
    """APIサービスヘルスチェック"""
    url = kwargs.get("url", "http://localhost:8000/health")
    timeout = kwargs.get("timeout", 5)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    return {
                        "healthy": True,
                        "details": {"status_code": response.status, "endpoint": url},
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status}",
                        "details": {"status_code": response.status},
                    }
    except Exception as e:
        return {"healthy": False, "error": str(e), "details": {"endpoint": url}}


def file_system_health_check(**kwargs) -> Dict[str, Any]:
    """ファイルシステムヘルスチェック"""
    path = kwargs.get("path", ".")

    try:
        test_path = Path(path)

        # 読み書きテスト
        if test_path.exists() and test_path.is_dir():
            # 一時ファイル作成テスト
            test_file = test_path / ".health_check_test"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()

            return {
                "healthy": True,
                "details": {"path": str(test_path), "writable": True, "exists": True},
            }
        else:
            return {
                "healthy": False,
                "error": "ディレクトリが存在しないかアクセスできません",
                "details": {"path": str(test_path)},
            }
    except Exception as e:
        return {"healthy": False, "error": str(e), "details": {"path": path}}


# 使用例・テスト関数
async def setup_standard_monitoring() -> SystemHealthMonitor:
    """標準監視セットアップ"""
    config = HealthCheckConfig(
        check_interval_seconds=30,
        cpu_warning_threshold=70.0,
        memory_warning_threshold=80.0,
        enable_auto_recovery=True,
    )

    monitor = SystemHealthMonitor(config)

    # 標準コンポーネント登録
    monitor.register_component(
        "database",
        ComponentType.DATABASE,
        database_health_check,
        connection_string="postgresql://localhost:5432/daytrading",
        timeout=5,
    )

    monitor.register_component(
        "api_service",
        ComponentType.API_SERVICE,
        api_service_health_check,
        url="http://localhost:8000/health",
        timeout=5,
    )

    monitor.register_component(
        "data_directory",
        ComponentType.FILE_SYSTEM,
        file_system_health_check,
        path="./data",
    )

    return monitor


if __name__ == "__main__":

    async def main():
        monitor = await setup_standard_monitoring()

        # 監視開始
        await monitor.start_monitoring()

        # 10秒間監視実行
        await asyncio.sleep(10)

        # 強制ヘルスチェック
        results = await monitor.force_health_check()
        print("強制ヘルスチェック結果:")
        for result in results:
            print(f"  {result.component_id}: {result.status.value}")

        # ヘルス概要取得
        summary = await monitor.get_system_health_summary()
        print(f"\nシステム全体ステータス: {summary['overall_status']}")

        # 監視停止
        await monitor.stop_monitoring()

    asyncio.run(main())
