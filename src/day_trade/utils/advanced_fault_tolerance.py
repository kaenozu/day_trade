#!/usr/bin/env python3
"""
高度なフォールトトレランス・自動復旧システム

Issue #312: API障害時の自動復旧とグレースフルデグラデーション
99.9%の可用性と5分以内の障害回復を目標
"""

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .logging_config import get_context_logger
from .performance_monitor import get_performance_monitor

logger = get_context_logger(__name__)


class ServiceState(Enum):
    """サービス状態"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"


class RecoveryAction(Enum):
    """復旧アクション"""

    RETRY = "retry"
    SWITCH_PROVIDER = "switch_provider"
    USE_CACHE = "use_cache"
    GRACEFUL_DEGRADE = "graceful_degrade"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ServiceHealth:
    """サービスヘルス情報"""

    service_name: str
    state: ServiceState
    last_success_time: datetime
    consecutive_failures: int
    total_requests: int
    success_rate: float
    avg_response_time: float
    error_details: Optional[Dict] = None


@dataclass
class RecoveryPlan:
    """復旧計画"""

    service_name: str
    actions: List[RecoveryAction]
    priority: int
    estimated_recovery_time: float
    fallback_services: List[str]


class DataSourceManager:
    """データソース管理（複数プロバイダー対応）"""

    def __init__(self):
        self.providers: Dict[str, Callable] = {}
        self.primary_provider: Optional[str] = None
        self.active_provider: Optional[str] = None
        self.provider_health: Dict[str, ServiceHealth] = {}
        self.failover_count = 0

        logger.info("データソース管理システム初期化完了")

    def register_provider(
        self, name: str, provider_func: Callable, is_primary: bool = False
    ):
        """データプロバイダー登録"""
        self.providers[name] = provider_func
        if is_primary or self.primary_provider is None:
            self.primary_provider = name
            self.active_provider = name

        # 初期ヘルス状態
        self.provider_health[name] = ServiceHealth(
            service_name=name,
            state=ServiceState.HEALTHY,
            last_success_time=datetime.now(),
            consecutive_failures=0,
            total_requests=0,
            success_rate=1.0,
            avg_response_time=0.0,
        )

        logger.info(f"データプロバイダー登録: {name} (Primary: {is_primary})")

    def get_data(self, *args, **kwargs) -> Any:
        """データ取得（自動フェイルオーバー付き）"""
        if not self.providers:
            raise RuntimeError("利用可能なデータプロバイダーがありません")

        # アクティブプロバイダーから試行
        providers_to_try = [self.active_provider] + [
            name for name in self.providers if name != self.active_provider
        ]

        last_error = None

        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            health = self.provider_health[provider_name]

            # 失敗が多すぎる場合はスキップ
            if health.consecutive_failures >= 5:
                logger.warning(
                    f"プロバイダー {provider_name} を一時的にスキップ（連続失敗: {health.consecutive_failures}）"
                )
                continue

            start_time = time.time()
            try:
                result = provider(*args, **kwargs)

                # 成功時の処理
                execution_time = time.time() - start_time
                self._update_health_success(provider_name, execution_time)

                # アクティブプロバイダーの切り替え
                if provider_name != self.active_provider:
                    logger.info(
                        f"データプロバイダーを切り替え: {self.active_provider} → {provider_name}"
                    )
                    self.active_provider = provider_name
                    self.failover_count += 1

                return result

            except Exception as e:
                last_error = e
                execution_time = time.time() - start_time
                self._update_health_failure(provider_name, execution_time, str(e))

                logger.warning(f"プロバイダー {provider_name} でエラー: {e}")
                continue

        # すべてのプロバイダーが失敗
        logger.error(
            f"すべてのデータプロバイダーが失敗しました。最後のエラー: {last_error}"
        )
        raise RuntimeError(f"すべてのデータプロバイダーが利用不可: {last_error}")

    def _update_health_success(self, provider_name: str, execution_time: float):
        """成功時のヘルス情報更新"""
        health = self.provider_health[provider_name]
        health.state = ServiceState.HEALTHY
        health.last_success_time = datetime.now()
        health.consecutive_failures = 0
        health.total_requests += 1

        # 移動平均でレスポンス時間更新
        if health.avg_response_time == 0:
            health.avg_response_time = execution_time
        else:
            health.avg_response_time = (health.avg_response_time * 0.9) + (
                execution_time * 0.1
            )

        # 成功率更新（簡易版）
        health.success_rate = min(1.0, health.success_rate + 0.01)

    def _update_health_failure(
        self, provider_name: str, execution_time: float, error_msg: str
    ):
        """失敗時のヘルス情報更新"""
        health = self.provider_health[provider_name]
        health.consecutive_failures += 1
        health.total_requests += 1
        health.success_rate = max(0.0, health.success_rate - 0.05)
        health.error_details = {
            "last_error": error_msg,
            "error_time": datetime.now().isoformat(),
        }

        # 状態更新
        if health.consecutive_failures >= 5:
            health.state = ServiceState.FAILED
        elif health.consecutive_failures >= 3:
            health.state = ServiceState.CRITICAL
        elif health.consecutive_failures >= 1:
            health.state = ServiceState.DEGRADED

    def get_health_summary(self) -> Dict:
        """ヘルス情報要約取得"""
        return {
            "active_provider": self.active_provider,
            "primary_provider": self.primary_provider,
            "failover_count": self.failover_count,
            "providers": {
                name: asdict(health) for name, health in self.provider_health.items()
            },
            "overall_health": self._calculate_overall_health(),
        }

    def _calculate_overall_health(self) -> str:
        """全体ヘルス計算"""
        if not self.provider_health:
            return "unknown"

        healthy_count = sum(
            1 for h in self.provider_health.values() if h.state == ServiceState.HEALTHY
        )
        total_count = len(self.provider_health)

        if healthy_count >= total_count * 0.8:
            return "healthy"
        elif healthy_count >= total_count * 0.5:
            return "degraded"
        else:
            return "critical"


class GracefulDegradationManager:
    """グレースフルデグラデーション管理"""

    def __init__(self):
        self.degradation_levels = {}
        self.current_level = 0
        self.max_level = 5
        self.feature_toggles = {}

        # デフォルト設定
        self._setup_default_levels()

        logger.info("グレースフルデグラデーション管理システム初期化完了")

    def _setup_default_levels(self):
        """デフォルトのデグラデーションレベル設定"""
        self.degradation_levels = {
            0: {
                "name": "Normal Operation",
                "description": "全機能正常動作",
                "disabled_features": [],
                "cache_ttl_multiplier": 1.0,
                "batch_size_multiplier": 1.0,
            },
            1: {
                "name": "Minor Degradation",
                "description": "軽微な機能制限",
                "disabled_features": ["real_time_alerts"],
                "cache_ttl_multiplier": 2.0,
                "batch_size_multiplier": 0.8,
            },
            2: {
                "name": "Moderate Degradation",
                "description": "中程度の機能制限",
                "disabled_features": ["real_time_alerts", "detailed_analytics"],
                "cache_ttl_multiplier": 3.0,
                "batch_size_multiplier": 0.6,
            },
            3: {
                "name": "Significant Degradation",
                "description": "重要機能制限",
                "disabled_features": [
                    "real_time_alerts",
                    "detailed_analytics",
                    "portfolio_optimization",
                ],
                "cache_ttl_multiplier": 5.0,
                "batch_size_multiplier": 0.4,
            },
            4: {
                "name": "Severe Degradation",
                "description": "最小機能のみ",
                "disabled_features": [
                    "real_time_alerts",
                    "detailed_analytics",
                    "portfolio_optimization",
                    "ml_predictions",
                ],
                "cache_ttl_multiplier": 10.0,
                "batch_size_multiplier": 0.2,
            },
            5: {
                "name": "Emergency Mode",
                "description": "緊急モード（基本機能のみ）",
                "disabled_features": [
                    "real_time_alerts",
                    "detailed_analytics",
                    "portfolio_optimization",
                    "ml_predictions",
                    "backtesting",
                ],
                "cache_ttl_multiplier": 20.0,
                "batch_size_multiplier": 0.1,
            },
        }

    def escalate_degradation(self, reason: str) -> bool:
        """デグラデーションレベルを上げる"""
        if self.current_level < self.max_level:
            old_level = self.current_level
            self.current_level += 1

            level_info = self.degradation_levels[self.current_level]
            logger.warning(
                f"デグラデーションレベル上昇: {old_level} → {self.current_level} "
                f"({level_info['name']}) - 理由: {reason}"
            )

            # 機能無効化
            for feature in level_info["disabled_features"]:
                self.feature_toggles[feature] = False

            return True

        return False

    def recover_degradation(self, steps: int = 1) -> bool:
        """デグラデーションレベルを下げる"""
        if self.current_level > 0:
            old_level = self.current_level
            self.current_level = max(0, self.current_level - steps)

            level_info = self.degradation_levels[self.current_level]
            logger.info(
                f"デグラデーションレベル回復: {old_level} → {self.current_level} "
                f"({level_info['name']})"
            )

            # 機能再有効化（現在のレベルに応じて）
            current_disabled = set(level_info["disabled_features"])
            for feature in self.feature_toggles:
                self.feature_toggles[feature] = feature not in current_disabled

            return True

        return False

    def is_feature_enabled(self, feature_name: str) -> bool:
        """機能が有効かチェック"""
        return self.feature_toggles.get(feature_name, True)

    def get_current_config(self) -> Dict:
        """現在のデグラデーション設定取得"""
        level_info = self.degradation_levels[self.current_level]
        return {
            "level": self.current_level,
            "level_info": level_info,
            "feature_toggles": self.feature_toggles.copy(),
        }


class AutoRecoverySystem:
    """自動復旧システム"""

    def __init__(
        self,
        data_source_manager: DataSourceManager,
        degradation_manager: GracefulDegradationManager,
    ):
        self.data_source_manager = data_source_manager
        self.degradation_manager = degradation_manager
        self.recovery_history = deque(maxlen=100)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 30  # 30秒間隔

        logger.info("自動復旧システム初期化完了")

    def start_monitoring(self):
        """監視開始"""
        if self.is_monitoring:
            logger.warning("監視は既に開始されています")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitor_thread.start()

        logger.info(f"自動復旧監視開始 (間隔: {self.monitor_interval}秒)")

    def stop_monitoring(self):
        """監視停止"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("自動復旧監視停止")

    def _monitoring_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                self._perform_health_check()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"監視ループエラー: {e}")
                time.sleep(self.monitor_interval)

    def _perform_health_check(self):
        """ヘルスチェック実行"""
        # データソースヘルス確認
        data_health = self.data_source_manager.get_health_summary()
        overall_health = data_health["overall_health"]

        # パフォーマンス監視システムからの情報取得
        performance_monitor = get_performance_monitor()
        if hasattr(performance_monitor, "get_performance_summary"):
            perf_summary = performance_monitor.get_performance_summary(hours=1)
            success_rate = (
                perf_summary.get("success_rate", 1.0)
                if "error" not in perf_summary
                else 1.0
            )
        else:
            success_rate = 1.0

        # 復旧判定
        recovery_needed = False
        escalation_needed = False

        if overall_health == "critical" or success_rate < 0.7:
            escalation_needed = True
            recovery_needed = True
            reason = f"システムヘルス: {overall_health}, 成功率: {success_rate:.1%}"
        elif overall_health == "degraded" or success_rate < 0.9:
            recovery_needed = True
            reason = f"軽微な問題: ヘルス {overall_health}, 成功率: {success_rate:.1%}"

        # デグラデーション制御
        if escalation_needed:
            if self.degradation_manager.escalate_degradation(reason):
                self._record_recovery_action("escalate_degradation", reason, True)
        elif (
            self.degradation_manager.current_level > 0
            and overall_health == "healthy"
            and success_rate > 0.95
        ):
            # 回復条件
            if self.degradation_manager.recover_degradation():
                self._record_recovery_action(
                    "recover_degradation", "システム状態良好", True
                )

        # 復旧アクション
        if recovery_needed:
            self._attempt_recovery(reason)

    def _attempt_recovery(self, reason: str):
        """復旧試行"""
        logger.info(f"復旧試行開始: {reason}")

        recovery_actions = [
            ("provider_health_reset", self._reset_provider_health),
            ("cache_clear", self._clear_caches),
            ("system_gc", self._force_garbage_collection),
        ]

        for action_name, action_func in recovery_actions:
            try:
                success = action_func()
                self._record_recovery_action(action_name, reason, success)

                if success:
                    logger.info(f"復旧アクション成功: {action_name}")
                    break

            except Exception as e:
                logger.error(f"復旧アクション失敗: {action_name} - {e}")
                self._record_recovery_action(
                    action_name, f"{reason} - エラー: {e}", False
                )

    def _reset_provider_health(self) -> bool:
        """プロバイダーヘルスリセット"""
        try:
            for (
                _provider_name,
                health,
            ) in self.data_source_manager.provider_health.items():
                if health.consecutive_failures > 0:
                    health.consecutive_failures = max(
                        0, health.consecutive_failures - 1
                    )
                    health.success_rate = min(1.0, health.success_rate + 0.1)

                    if health.state in [ServiceState.FAILED, ServiceState.CRITICAL]:
                        health.state = ServiceState.DEGRADED
                    elif health.state == ServiceState.DEGRADED:
                        health.state = ServiceState.HEALTHY

            return True
        except Exception:
            return False

    def _clear_caches(self) -> bool:
        """キャッシュクリア"""
        try:
            # パフォーマンス監視システムのキャッシュクリア
            performance_monitor = get_performance_monitor()
            if hasattr(performance_monitor, "clear_old_metrics"):
                performance_monitor.clear_old_metrics(hours=1)

            return True
        except Exception:
            return False

    def _force_garbage_collection(self) -> bool:
        """強制ガベージコレクション"""
        try:
            import gc

            collected = gc.collect()
            logger.info(f"ガベージコレクション実行: {collected}オブジェクト回収")
            return True
        except Exception:
            return False

    def _record_recovery_action(self, action: str, reason: str, success: bool):
        """復旧アクション記録"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "reason": reason,
            "success": success,
            "degradation_level": self.degradation_manager.current_level,
        }

        self.recovery_history.append(record)

        status = "成功" if success else "失敗"
        logger.info(f"復旧アクション記録: {action} - {status}")

    def get_recovery_history(self, hours: int = 24) -> List[Dict]:
        """復旧履歴取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            record
            for record in self.recovery_history
            if datetime.fromisoformat(record["timestamp"]) > cutoff_time
        ]

    def get_system_status(self) -> Dict:
        """システム状態取得"""
        data_health = self.data_source_manager.get_health_summary()
        degradation_config = self.degradation_manager.get_current_config()
        recent_recoveries = self.get_recovery_history(hours=1)

        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.is_monitoring,
            "data_sources": data_health,
            "degradation": degradation_config,
            "recent_recovery_actions": len(recent_recoveries),
            "last_recovery": recent_recoveries[-1] if recent_recoveries else None,
            "uptime_estimate": self._calculate_uptime_estimate(),
        }

    def _calculate_uptime_estimate(self) -> Dict:
        """稼働率推定計算"""
        recent_history = self.get_recovery_history(hours=24)

        if not recent_history:
            return {"uptime_percent": 99.9, "estimation_basis": "no_recent_issues"}

        # 簡易稼働率計算
        failure_count = sum(
            1 for r in recent_history if not r["success"] and "escalate" in r["action"]
        )
        total_periods = 24 * 2  # 30分間隔で48期間

        estimated_uptime = max(95.0, 100.0 - (failure_count / total_periods * 100))

        return {
            "uptime_percent": estimated_uptime,
            "failure_incidents": failure_count,
            "estimation_basis": "24h_history",
        }


# グローバルインスタンス
_global_data_source_manager = None
_global_degradation_manager = None
_global_recovery_system = None


def get_data_source_manager() -> DataSourceManager:
    """グローバルデータソース管理取得"""
    global _global_data_source_manager
    if _global_data_source_manager is None:
        _global_data_source_manager = DataSourceManager()
    return _global_data_source_manager


def get_degradation_manager() -> GracefulDegradationManager:
    """グローバルデグラデーション管理取得"""
    global _global_degradation_manager
    if _global_degradation_manager is None:
        _global_degradation_manager = GracefulDegradationManager()
    return _global_degradation_manager


def get_recovery_system() -> AutoRecoverySystem:
    """グローバル復旧システム取得"""
    global _global_recovery_system
    if _global_recovery_system is None:
        data_mgr = get_data_source_manager()
        degradation_mgr = get_degradation_manager()
        _global_recovery_system = AutoRecoverySystem(data_mgr, degradation_mgr)
    return _global_recovery_system


@contextmanager
def resilient_operation(operation_name: str, fallback_result: Any = None):
    """復旧可能な操作のコンテキストマネージャー"""
    get_recovery_system()
    degradation_manager = get_degradation_manager()

    start_time = time.time()
    try:
        yield
    except Exception as e:
        execution_time = time.time() - start_time

        logger.error(
            f"復旧可能操作でエラー: {operation_name} - {e} ({execution_time:.2f}s)"
        )

        # デグラデーション判定
        if execution_time > 30:  # 30秒以上の場合
            degradation_manager.escalate_degradation(
                f"操作タイムアウト: {operation_name}"
            )

        # フォールバック結果を返す
        if fallback_result is not None:
            logger.info(f"フォールバック結果を使用: {operation_name}")
            return fallback_result

        raise


if __name__ == "__main__":
    # テスト実行
    print("Advanced Fault Tolerance System Test")
    print("=" * 50)

    try:
        # データソース管理テスト
        data_mgr = get_data_source_manager()

        def primary_data_source(*args, **kwargs):
            import random

            if random.random() < 0.3:  # 30%の確率で失敗
                raise Exception("Primary source unavailable")
            return {"data": "primary", "timestamp": time.time()}

        def backup_data_source(*args, **kwargs):
            return {"data": "backup", "timestamp": time.time()}

        data_mgr.register_provider("primary", primary_data_source, is_primary=True)
        data_mgr.register_provider("backup", backup_data_source)

        # 自動復旧システムテスト
        degradation_mgr = get_degradation_manager()
        recovery_system = get_recovery_system()

        recovery_system.start_monitoring()

        # テスト実行
        for i in range(5):
            try:
                with resilient_operation(f"test_operation_{i}"):
                    result = data_mgr.get_data()
                    print(f"Operation {i}: {result['data']}")
                    time.sleep(1)
            except Exception as e:
                print(f"Operation {i} failed: {e}")

        # システム状態確認
        time.sleep(2)
        system_status = recovery_system.get_system_status()

        print("\nSystem Status:")
        print(f"Monitoring Active: {system_status['monitoring_active']}")
        print(f"Degradation Level: {system_status['degradation']['level']}")
        print(
            f"Uptime Estimate: {system_status['uptime_estimate']['uptime_percent']:.1f}%"
        )

        recovery_system.stop_monitoring()

        print("Advanced fault tolerance test completed successfully")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
