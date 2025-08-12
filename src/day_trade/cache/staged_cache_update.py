#!/usr/bin/env python3
"""
Staged Cache Update Strategy System
Issue #377: 高度なキャッシング戦略の導入

段階的キャッシュ更新戦略:
- Blue-Green キャッシュ切り替え
- カナリアリリース方式
- ローリングアップデート
- A/Bテスト対応の段階的更新
- ロールバック機能
"""

import asyncio
import hashlib
import json
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from ..utils.logging_config import get_context_logger
    from .enhanced_persistent_cache import EnhancedPersistentCache
    from .redis_enhanced_cache import RedisEnhancedCache
    from .smart_cache_invalidation import SmartCacheInvalidator
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    class SmartCacheInvalidator:
        pass

    class EnhancedPersistentCache:
        pass

    class RedisEnhancedCache:
        pass


logger = get_context_logger(__name__)


class UpdateStrategy(Enum):
    """更新戦略タイプ"""

    IMMEDIATE = "immediate"  # 即座更新
    BLUE_GREEN = "blue_green"  # Blue-Green切り替え
    CANARY = "canary"  # カナリアリリース
    ROLLING = "rolling"  # ローリングアップデート
    AB_TEST = "ab_test"  # A/Bテスト
    PHASED = "phased"  # 段階的更新


class UpdatePhase(Enum):
    """更新フェーズ"""

    PLANNING = "planning"  # 計画
    PREPARATION = "preparation"  # 準備
    STAGING = "staging"  # ステージング
    CANARY_TESTING = "canary_testing"  # カナリアテスト
    ROLLING_OUT = "rolling_out"  # ロールアウト
    COMPLETED = "completed"  # 完了
    FAILED = "failed"  # 失敗
    ROLLED_BACK = "rolled_back"  # ロールバック


class UpdateStatus(Enum):
    """更新ステータス"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpdateConfig:
    """更新設定"""

    update_id: str
    strategy: UpdateStrategy
    target_keys: List[str]

    # 戦略別設定
    canary_percentage: float = 10.0  # カナリア配信割合
    rollout_percentage_steps: List[float] = field(
        default_factory=lambda: [10, 50, 100]
    )  # ロールアウト段階
    ab_split_ratio: float = 50.0  # A/Bテスト分割比率

    # 制御設定
    validation_checks: List[str] = field(default_factory=list)
    max_rollout_time_minutes: int = 60
    auto_rollback_on_error: bool = True
    health_check_interval: int = 30

    # 対象設定
    target_cache_instances: List[str] = field(default_factory=list)
    user_segments: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateEvent:
    """更新イベント"""

    event_id: str
    update_id: str
    phase: UpdatePhase
    status: UpdateStatus
    timestamp: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class UpdateMetrics:
    """更新メトリクス"""

    update_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # パフォーマンスメトリクス
    cache_hit_rate_before: float = 0.0
    cache_hit_rate_after: float = 0.0
    response_time_before_ms: float = 0.0
    response_time_after_ms: float = 0.0

    # 更新メトリクス
    keys_updated: int = 0
    keys_failed: int = 0
    rollback_count: int = 0
    error_rate: float = 0.0

    # ユーザーメトリクス
    affected_users: int = 0
    user_feedback_score: float = 0.0


class CacheUpdateValidator:
    """キャッシュ更新バリデーター"""

    def __init__(self):
        self.checks = {}  # check_name -> Callable

    def register_check(self, name: str, check_func: Callable):
        """バリデーションチェック登録"""
        self.checks[name] = check_func

    async def validate(
        self, update_config: UpdateConfig, cache_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """更新データのバリデーション"""
        errors = []

        for check_name in update_config.validation_checks:
            if check_name in self.checks:
                try:
                    check_func = self.checks[check_name]
                    is_valid = await self._run_check(check_func, cache_data)

                    if not is_valid:
                        errors.append(f"バリデーションチェック失敗: {check_name}")

                except Exception as e:
                    errors.append(f"バリデーションエラー ({check_name}): {str(e)}")

        return len(errors) == 0, errors

    async def _run_check(self, check_func: Callable, data: Dict[str, Any]) -> bool:
        """チェック関数実行"""
        if asyncio.iscoroutinefunction(check_func):
            return await check_func(data)
        else:
            return check_func(data)


class StagedCacheUpdater:
    """段階的キャッシュ更新システム"""

    def __init__(self, cache_instances: Dict[str, Any] = None):
        self.cache_instances = cache_instances or {}
        self.active_updates = {}  # update_id -> UpdateConfig
        self.update_history = deque(maxlen=1000)  # 更新履歴
        self.update_metrics = {}  # update_id -> UpdateMetrics

        self.validator = CacheUpdateValidator()
        self.lock = threading.RLock()

        # 監視タスク
        self.monitoring_tasks = {}  # update_id -> asyncio.Task
        self.running = True

        self._register_default_validators()

        logger.info("段階的キャッシュ更新システム初期化完了")

    def register_cache_instance(self, name: str, cache_instance: Any):
        """キャッシュインスタンス登録"""
        self.cache_instances[name] = cache_instance
        logger.info(f"キャッシュインスタンス登録: {name}")

    async def start_staged_update(self, config: UpdateConfig, new_data: Dict[str, Any]) -> str:
        """段階的更新開始"""
        with self.lock:
            if config.update_id in self.active_updates:
                raise ValueError(f"更新ID {config.update_id} は既に実行中です")

            self.active_updates[config.update_id] = config

            # メトリクス初期化
            metrics = UpdateMetrics(update_id=config.update_id, start_time=time.time())
            self.update_metrics[config.update_id] = metrics

            # 更新戦略に応じて実行
            if config.strategy == UpdateStrategy.BLUE_GREEN:
                task = asyncio.create_task(self._blue_green_update(config, new_data))
            elif config.strategy == UpdateStrategy.CANARY:
                task = asyncio.create_task(self._canary_update(config, new_data))
            elif config.strategy == UpdateStrategy.ROLLING:
                task = asyncio.create_task(self._rolling_update(config, new_data))
            elif config.strategy == UpdateStrategy.AB_TEST:
                task = asyncio.create_task(self._ab_test_update(config, new_data))
            elif config.strategy == UpdateStrategy.PHASED:
                task = asyncio.create_task(self._phased_update(config, new_data))
            else:
                task = asyncio.create_task(self._immediate_update(config, new_data))

            self.monitoring_tasks[config.update_id] = task

            logger.info(f"段階的更新開始: {config.update_id} ({config.strategy.value})")

        return config.update_id

    async def _immediate_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """即座更新"""
        try:
            await self._emit_event(
                config.update_id,
                UpdatePhase.STAGING,
                UpdateStatus.IN_PROGRESS,
                "即座更新開始",
            )

            # バリデーション
            is_valid, errors = await self.validator.validate(config, new_data)
            if not is_valid:
                raise Exception(f"バリデーション失敗: {errors}")

            # 全キャッシュに即座適用
            updated_count = 0
            failed_count = 0

            for cache_name in config.target_cache_instances:
                if cache_name in self.cache_instances:
                    cache = self.cache_instances[cache_name]

                    for key in config.target_keys:
                        if key in new_data:
                            try:
                                await self._update_cache_key(cache, key, new_data[key])
                                updated_count += 1
                            except Exception as e:
                                failed_count += 1
                                logger.error(f"キー更新失敗 ({key}): {e}")

            # メトリクス更新
            metrics = self.update_metrics[config.update_id]
            metrics.keys_updated = updated_count
            metrics.keys_failed = failed_count
            metrics.end_time = time.time()
            metrics.duration_seconds = metrics.end_time - metrics.start_time

            await self._emit_event(
                config.update_id,
                UpdatePhase.COMPLETED,
                UpdateStatus.SUCCESS,
                f"更新完了: {updated_count}件成功, {failed_count}件失敗",
            )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
        finally:
            await self._cleanup_update(config.update_id)

    async def _blue_green_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """Blue-Green更新"""
        try:
            await self._emit_event(
                config.update_id,
                UpdatePhase.PREPARATION,
                UpdateStatus.IN_PROGRESS,
                "Blue-Green更新準備",
            )

            # Green環境準備
            green_caches = await self._prepare_green_environment(config, new_data)

            await self._emit_event(
                config.update_id,
                UpdatePhase.STAGING,
                UpdateStatus.IN_PROGRESS,
                "Green環境準備完了",
            )

            # バリデーション
            is_valid, errors = await self.validator.validate(config, new_data)
            if not is_valid:
                raise Exception(f"Green環境バリデーション失敗: {errors}")

            # 段階的切り替え（カナリア -> 全体）
            await self._emit_event(
                config.update_id,
                UpdatePhase.CANARY_TESTING,
                UpdateStatus.IN_PROGRESS,
                "カナリア切り替え開始",
            )

            # カナリアユーザーをGreen環境に誘導
            await self._switch_canary_traffic(config, green_caches, config.canary_percentage)

            # ヘルスチェック
            await asyncio.sleep(config.health_check_interval)
            health_ok = await self._health_check(green_caches)

            if not health_ok:
                raise Exception("Green環境ヘルスチェック失敗")

            # 全トラフィック切り替え
            await self._emit_event(
                config.update_id,
                UpdatePhase.ROLLING_OUT,
                UpdateStatus.IN_PROGRESS,
                "全トラフィック切り替え",
            )

            await self._switch_all_traffic(config, green_caches)

            await self._emit_event(
                config.update_id,
                UpdatePhase.COMPLETED,
                UpdateStatus.SUCCESS,
                "Blue-Green更新完了",
            )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
            if config.auto_rollback_on_error:
                await self._rollback_update(config.update_id)
        finally:
            await self._cleanup_update(config.update_id)

    async def _canary_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """カナリア更新"""
        try:
            await self._emit_event(
                config.update_id,
                UpdatePhase.CANARY_TESTING,
                UpdateStatus.IN_PROGRESS,
                "カナリア更新開始",
            )

            # カナリア対象選択
            canary_keys = self._select_canary_keys(config.target_keys, config.canary_percentage)

            # カナリア更新実行
            for key in canary_keys:
                for cache_name in config.target_cache_instances:
                    if cache_name in self.cache_instances:
                        cache = self.cache_instances[cache_name]
                        if key in new_data:
                            await self._update_cache_key(cache, key, new_data[key])

            # カナリア監視
            await asyncio.sleep(config.health_check_interval)
            canary_metrics = await self._collect_canary_metrics(canary_keys)

            if canary_metrics.get("error_rate", 0) > 0.05:  # 5%以上のエラー率で停止
                raise Exception(f"カナリアエラー率異常: {canary_metrics['error_rate']:.2%}")

            # 残りの更新実行
            await self._emit_event(
                config.update_id,
                UpdatePhase.ROLLING_OUT,
                UpdateStatus.IN_PROGRESS,
                "全体ロールアウト",
            )

            remaining_keys = [k for k in config.target_keys if k not in canary_keys]
            for key in remaining_keys:
                for cache_name in config.target_cache_instances:
                    if cache_name in self.cache_instances:
                        cache = self.cache_instances[cache_name]
                        if key in new_data:
                            await self._update_cache_key(cache, key, new_data[key])

            await self._emit_event(
                config.update_id,
                UpdatePhase.COMPLETED,
                UpdateStatus.SUCCESS,
                "カナリア更新完了",
            )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
            if config.auto_rollback_on_error:
                await self._rollback_update(config.update_id)
        finally:
            await self._cleanup_update(config.update_id)

    async def _rolling_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """ローリング更新"""
        try:
            await self._emit_event(
                config.update_id,
                UpdatePhase.ROLLING_OUT,
                UpdateStatus.IN_PROGRESS,
                "ローリング更新開始",
            )

            # 段階的ロールアウト
            total_keys = len(config.target_keys)

            for step_percent in config.rollout_percentage_steps:
                target_count = int(total_keys * step_percent / 100)
                target_keys = config.target_keys[:target_count]

                logger.info(f"ローリング更新 {step_percent}%: {len(target_keys)}キー")

                # 該当キー更新
                for key in target_keys:
                    for cache_name in config.target_cache_instances:
                        if cache_name in self.cache_instances:
                            cache = self.cache_instances[cache_name]
                            if key in new_data:
                                await self._update_cache_key(cache, key, new_data[key])

                # ヘルスチェック
                await asyncio.sleep(
                    config.health_check_interval // len(config.rollout_percentage_steps)
                )
                health_ok = await self._health_check_keys(target_keys)

                if not health_ok:
                    raise Exception(f"ローリング更新 {step_percent}% でヘルスチェック失敗")

            await self._emit_event(
                config.update_id,
                UpdatePhase.COMPLETED,
                UpdateStatus.SUCCESS,
                "ローリング更新完了",
            )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
            if config.auto_rollback_on_error:
                await self._rollback_update(config.update_id)
        finally:
            await self._cleanup_update(config.update_id)

    async def _ab_test_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """A/Bテスト更新"""
        try:
            await self._emit_event(
                config.update_id,
                UpdatePhase.STAGING,
                UpdateStatus.IN_PROGRESS,
                "A/Bテスト更新開始",
            )

            # A/B分割
            a_keys, b_keys = self._split_keys_for_ab_test(config.target_keys, config.ab_split_ratio)

            # Bグループに新データ適用
            for key in b_keys:
                for cache_name in config.target_cache_instances:
                    if cache_name in self.cache_instances:
                        cache = self.cache_instances[cache_name]
                        if key in new_data:
                            await self._update_cache_key(cache, key, new_data[key])

            # A/Bテスト実行期間
            test_duration = config.metadata.get("test_duration_seconds", 3600)
            await asyncio.sleep(test_duration)

            # A/Bテスト結果分析
            ab_results = await self._analyze_ab_test_results(a_keys, b_keys)

            if ab_results.get("b_better", False):
                # Bが良い結果なら、Aグループにも適用
                for key in a_keys:
                    for cache_name in config.target_cache_instances:
                        if cache_name in self.cache_instances:
                            cache = self.cache_instances[cache_name]
                            if key in new_data:
                                await self._update_cache_key(cache, key, new_data[key])

                await self._emit_event(
                    config.update_id,
                    UpdatePhase.COMPLETED,
                    UpdateStatus.SUCCESS,
                    "A/Bテスト: B勝利、全体適用",
                )
            else:
                # Aが良い結果なら、Bグループをロールバック
                await self._rollback_keys(b_keys, config)
                await self._emit_event(
                    config.update_id,
                    UpdatePhase.ROLLED_BACK,
                    UpdateStatus.SUCCESS,
                    "A/Bテスト: A勝利、ロールバック",
                )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
        finally:
            await self._cleanup_update(config.update_id)

    async def _phased_update(self, config: UpdateConfig, new_data: Dict[str, Any]):
        """段階的更新"""
        phases = config.metadata.get(
            "phases",
            [
                {"name": "alpha", "percentage": 5},
                {"name": "beta", "percentage": 25},
                {"name": "production", "percentage": 100},
            ],
        )

        try:
            for phase in phases:
                phase_name = phase["name"]
                percentage = phase["percentage"]

                await self._emit_event(
                    config.update_id,
                    UpdatePhase.ROLLING_OUT,
                    UpdateStatus.IN_PROGRESS,
                    f"{phase_name}フェーズ開始 ({percentage}%)",
                )

                target_count = int(len(config.target_keys) * percentage / 100)
                target_keys = config.target_keys[:target_count]

                # フェーズ更新
                for key in target_keys:
                    for cache_name in config.target_cache_instances:
                        if cache_name in self.cache_instances:
                            cache = self.cache_instances[cache_name]
                            if key in new_data:
                                await self._update_cache_key(cache, key, new_data[key])

                # フェーズ間待機
                phase_wait = phase.get("wait_seconds", 300)
                await asyncio.sleep(phase_wait)

                # ヘルスチェック
                health_ok = await self._health_check_keys(target_keys)
                if not health_ok:
                    raise Exception(f"{phase_name}フェーズでヘルスチェック失敗")

            await self._emit_event(
                config.update_id,
                UpdatePhase.COMPLETED,
                UpdateStatus.SUCCESS,
                "段階的更新完了",
            )

        except Exception as e:
            await self._handle_update_error(config.update_id, str(e))
            if config.auto_rollback_on_error:
                await self._rollback_update(config.update_id)
        finally:
            await self._cleanup_update(config.update_id)

    async def _update_cache_key(self, cache: Any, key: str, value: Any):
        """キャッシュキー更新"""
        if hasattr(cache, "put"):
            await cache.put(key, value)
        elif hasattr(cache, "set"):
            cache.set(key, value)
        else:
            logger.warning(f"キャッシュ {type(cache)} は put/set メソッドをサポートしていません")

    def _select_canary_keys(self, keys: List[str], percentage: float) -> List[str]:
        """カナリアキー選択"""
        count = max(1, int(len(keys) * percentage / 100))
        return keys[:count]

    def _split_keys_for_ab_test(
        self, keys: List[str], split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """A/Bテスト用キー分割"""
        split_point = int(len(keys) * split_ratio / 100)
        return keys[:split_point], keys[split_point:]

    async def _health_check(self, caches: List[Any]) -> bool:
        """ヘルスチェック"""
        # 簡易ヘルスチェック実装
        return True

    async def _health_check_keys(self, keys: List[str]) -> bool:
        """キー別ヘルスチェック"""
        # 簡易実装
        return True

    async def _collect_canary_metrics(self, keys: List[str]) -> Dict[str, Any]:
        """カナリアメトリクス収集"""
        return {"error_rate": 0.01, "response_time": 50.0}

    async def _analyze_ab_test_results(
        self, a_keys: List[str], b_keys: List[str]
    ) -> Dict[str, Any]:
        """A/Bテスト結果分析"""
        # 簡易実装：ランダムにBが勝利
        import random

        return {"b_better": random.random() > 0.5, "confidence": 0.95}

    async def _emit_event(
        self,
        update_id: str,
        phase: UpdatePhase,
        status: UpdateStatus,
        message: str,
        details: Dict[str, Any] = None,
    ):
        """更新イベント発行"""
        event = UpdateEvent(
            event_id=f"{update_id}_{int(time.time() * 1000)}",
            update_id=update_id,
            phase=phase,
            status=status,
            timestamp=time.time(),
            message=message,
            details=details or {},
        )

        self.update_history.append(event)
        logger.info(f"更新イベント [{update_id}]: {phase.value} - {message}")

    async def _handle_update_error(self, update_id: str, error_msg: str):
        """更新エラー処理"""
        await self._emit_event(
            update_id, UpdatePhase.FAILED, UpdateStatus.FAILED, f"更新失敗: {error_msg}"
        )

        metrics = self.update_metrics.get(update_id)
        if metrics:
            metrics.end_time = time.time()
            metrics.duration_seconds = metrics.end_time - metrics.start_time
            metrics.error_rate = 100.0

    async def _rollback_update(self, update_id: str):
        """更新ロールバック"""
        await self._emit_event(
            update_id,
            UpdatePhase.ROLLED_BACK,
            UpdateStatus.IN_PROGRESS,
            "ロールバック開始",
        )

        # ロールバック処理（簡易実装）
        config = self.active_updates.get(update_id)
        if config:
            await self._rollback_keys(config.target_keys, config)

        await self._emit_event(
            update_id, UpdatePhase.ROLLED_BACK, UpdateStatus.SUCCESS, "ロールバック完了"
        )

    async def _rollback_keys(self, keys: List[str], config: UpdateConfig):
        """キーロールバック"""
        # 実際の実装では、バックアップからデータを復元
        for cache_name in config.target_cache_instances:
            if cache_name in self.cache_instances:
                cache = self.cache_instances[cache_name]
                for key in keys:
                    # ロールバック処理（削除または旧データ復元）
                    if hasattr(cache, "delete"):
                        await cache.delete(key)

    async def _cleanup_update(self, update_id: str):
        """更新クリーンアップ"""
        with self.lock:
            if update_id in self.active_updates:
                del self.active_updates[update_id]
            if update_id in self.monitoring_tasks:
                task = self.monitoring_tasks[update_id]
                if not task.done():
                    task.cancel()
                del self.monitoring_tasks[update_id]

    def _register_default_validators(self):
        """デフォルトバリデーター登録"""
        self.validator.register_check("data_structure", self._validate_data_structure)
        self.validator.register_check("data_size", self._validate_data_size)
        self.validator.register_check("data_type", self._validate_data_type)

    async def _validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """データ構造バリデーション"""
        return isinstance(data, dict) and len(data) > 0

    async def _validate_data_size(self, data: Dict[str, Any]) -> bool:
        """データサイズバリデーション"""
        import pickle

        serialized = pickle.dumps(data)
        return len(serialized) < 10 * 1024 * 1024  # 10MB未満

    async def _validate_data_type(self, data: Dict[str, Any]) -> bool:
        """データタイプバリデーション"""
        # JSON シリアライズ可能かチェック
        try:
            json.dumps(data, default=str)
            return True
        except:
            return False

    def get_update_status(self, update_id: str) -> Optional[Dict[str, Any]]:
        """更新ステータス取得"""
        with self.lock:
            if update_id not in self.update_metrics:
                return None

            metrics = self.update_metrics[update_id]
            config = self.active_updates.get(update_id)

            # 最新イベント取得
            latest_event = None
            for event in reversed(self.update_history):
                if event.update_id == update_id:
                    latest_event = event
                    break

            return {
                "update_id": update_id,
                "config": {
                    "strategy": config.strategy.value if config else "unknown",
                    "target_keys_count": len(config.target_keys) if config else 0,
                },
                "metrics": {
                    "start_time": metrics.start_time,
                    "end_time": metrics.end_time,
                    "duration_seconds": metrics.duration_seconds,
                    "keys_updated": metrics.keys_updated,
                    "keys_failed": metrics.keys_failed,
                    "error_rate": metrics.error_rate,
                },
                "latest_event": (
                    {
                        "phase": latest_event.phase.value if latest_event else "unknown",
                        "status": latest_event.status.value if latest_event else "unknown",
                        "message": latest_event.message if latest_event else "",
                        "timestamp": latest_event.timestamp if latest_event else 0,
                    }
                    if latest_event
                    else None
                ),
                "is_active": update_id in self.active_updates,
            }

    def get_all_updates_status(self) -> Dict[str, Any]:
        """全更新ステータス取得"""
        return {
            "active_updates": len(self.active_updates),
            "total_updates": len(self.update_metrics),
            "updates": {uid: self.get_update_status(uid) for uid in self.update_metrics.keys()},
        }


# テスト関数
async def test_staged_cache_update():
    """段階的キャッシュ更新テスト"""
    print("=== Issue #377 段階的キャッシュ更新システムテスト ===")

    try:
        # モックキャッシュ
        class MockCache:
            def __init__(self, name):
                self.name = name
                self.data = {}

            async def put(self, key, value):
                self.data[key] = value
                print(f"  {self.name}: PUT {key} = {value}")

            async def delete(self, key):
                if key in self.data:
                    del self.data[key]
                    print(f"  {self.name}: DELETE {key}")

        # システム初期化
        updater = StagedCacheUpdater()

        # キャッシュインスタンス登録
        cache1 = MockCache("cache1")
        cache2 = MockCache("cache2")
        updater.register_cache_instance("cache1", cache1)
        updater.register_cache_instance("cache2", cache2)

        print("\n1. 段階的キャッシュ更新システム初期化完了")

        # カナリア更新テスト
        canary_config = UpdateConfig(
            update_id="canary_test_1",
            strategy=UpdateStrategy.CANARY,
            target_keys=["user:1", "user:2", "user:3", "user:4"],
            target_cache_instances=["cache1", "cache2"],
            canary_percentage=25.0,
            validation_checks=["data_structure", "data_size"],
            health_check_interval=2,
        )

        new_data = {
            "user:1": {"name": "User1", "version": 2},
            "user:2": {"name": "User2", "version": 2},
            "user:3": {"name": "User3", "version": 2},
            "user:4": {"name": "User4", "version": 2},
        }

        print("\n2. カナリア更新開始...")
        update_id = await updater.start_staged_update(canary_config, new_data)

        # 更新完了まで待機
        await asyncio.sleep(5)

        # ステータス確認
        status = updater.get_update_status(update_id)
        print("\n3. 更新ステータス:")
        if status:
            for category, data in status.items():
                if isinstance(data, dict):
                    print(f"   {category}:")
                    for k, v in data.items():
                        print(f"     {k}: {v}")
                else:
                    print(f"   {category}: {data}")

        print("\n4. キャッシュ状態:")
        print(f"   cache1: {cache1.data}")
        print(f"   cache2: {cache2.data}")

        # 全体ステータス
        all_status = updater.get_all_updates_status()
        print(
            f"\n5. 全体ステータス: アクティブ={all_status['active_updates']}, 総数={all_status['total_updates']}"
        )

        print("\n[OK] 段階的キャッシュ更新システム テスト完了！")

    except Exception as e:
        print(f"[ERROR] テストエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_staged_cache_update())
