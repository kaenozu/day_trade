#!/usr/bin/env python3
"""
特徴量重複計算排除システム
Issue #380: 機械学習モデルの最適化 - 特徴量生成の効率化

重複計算の検出・排除と計算効率の最適化
"""

import hashlib
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis.feature_engineering_unified import FeatureConfig
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ComputationTask:
    """計算タスク"""

    task_id: str
    symbol: str
    config_hash: str
    data_hash: str
    feature_config: FeatureConfig
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, computing, completed, failed
    result: Any = None
    computation_time: float = 0.0


@dataclass
class ComputationStats:
    """計算統計"""

    total_requests: int = 0
    duplicate_requests: int = 0
    completed_computations: int = 0
    failed_computations: int = 0
    total_computation_time: float = 0.0
    time_saved_by_dedup: float = 0.0

    @property
    def deduplication_rate(self) -> float:
        """重複排除率"""
        return (self.duplicate_requests / max(1, self.total_requests)) * 100

    @property
    def success_rate(self) -> float:
        """成功率"""
        total_attempts = self.completed_computations + self.failed_computations
        return (self.completed_computations / max(1, total_attempts)) * 100

    @property
    def avg_computation_time(self) -> float:
        """平均計算時間"""
        return self.total_computation_time / max(1, self.completed_computations)


class FeatureDeduplicationManager:
    """特徴量重複計算排除マネージャー"""

    def __init__(self, max_concurrent_tasks: int = 100, cache_ttl_seconds: int = 3600):
        """初期化"""
        self.max_concurrent_tasks = max_concurrent_tasks
        self.cache_ttl_seconds = cache_ttl_seconds

        # スレッドセーフな処理のためのロック
        self._lock = RLock()

        # アクティブなタスクの管理
        self.active_tasks: Dict[str, ComputationTask] = {}

        # 完了したタスクのキャッシュ（弱参照で自動クリーンアップ）
        self.completed_tasks: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )

        # 待機中の重複リクエスト管理
        self.waiting_requests: Dict[str, List[str]] = defaultdict(list)

        # 統計情報
        self.stats = ComputationStats()

        # タスクの実行履歴（FIFO）
        self.task_history: deque = deque(maxlen=1000)

        logger.info(
            "特徴量重複計算排除マネージャー初期化完了",
            extra={
                "max_concurrent_tasks": max_concurrent_tasks,
                "cache_ttl_seconds": cache_ttl_seconds,
            },
        )

    def _generate_task_key(
        self, symbol: str, data_hash: str, feature_config: FeatureConfig
    ) -> str:
        """タスクキーの生成（重複検出用）"""
        # 設定のハッシュ化
        config_dict = {
            "lookback_periods": sorted(feature_config.lookback_periods),
            "volatility_windows": sorted(feature_config.volatility_windows),
            "momentum_periods": sorted(feature_config.momentum_periods),
            "enable_cross_features": feature_config.enable_cross_features,
            "enable_statistical_features": feature_config.enable_statistical_features,
            "enable_regime_features": feature_config.enable_regime_features,
            "scaling_method": feature_config.scaling_method,
        }

        config_str = str(sorted(config_dict.items()))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # タスクキー生成
        key_string = f"{symbol}_{data_hash}_{config_hash}"
        task_key = hashlib.md5(key_string.encode()).hexdigest()

        return task_key

    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """データハッシュの生成"""
        # データの形状とサンプル値からハッシュを生成
        shape_str = f"{data.shape[0]}x{data.shape[1]}"

        # インデックスの情報
        if hasattr(data.index, "min") and hasattr(data.index, "max"):
            index_info = f"{data.index.min()}_{data.index.max()}"
        else:
            index_info = "no_datetime_index"

        # サンプルデータ（先頭・末尾数行）
        sample_data = pd.concat([data.head(3), data.tail(3)]) if len(data) > 6 else data

        try:
            # 数値データのハッシュ
            numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                sample_values = sample_data[numeric_cols].values.flatten()[
                    :50
                ]  # 最大50値
                values_str = ",".join(
                    [f"{v:.6f}" for v in sample_values if not np.isnan(v)]
                )
            else:
                values_str = "no_numeric_data"
        except:
            values_str = "hash_error"

        hash_input = f"{shape_str}_{index_info}_{values_str}"
        data_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        return data_hash

    def is_duplicate_request(
        self, symbol: str, data: pd.DataFrame, feature_config: FeatureConfig
    ) -> Tuple[bool, Optional[str]]:
        """重複リクエストの検出"""
        data_hash = self._generate_data_hash(data)
        task_key = self._generate_task_key(symbol, data_hash, feature_config)

        with self._lock:
            # アクティブなタスクでの重複チェック
            if task_key in self.active_tasks:
                active_task = self.active_tasks[task_key]

                # タスクが古すぎる場合は削除
                if time.time() - active_task.created_at > self.cache_ttl_seconds:
                    del self.active_tasks[task_key]
                    return False, task_key

                logger.info(
                    "重複リクエスト検出（アクティブタスク）",
                    extra={
                        "symbol": symbol,
                        "task_key": task_key[:8],
                        "status": active_task.status,
                        "age_seconds": round(time.time() - active_task.created_at, 2),
                    },
                )

                return True, task_key

            # 完了済みタスクでの重複チェック
            if task_key in self.completed_tasks:
                completed_task = self.completed_tasks[task_key]

                # タスクが古すぎる場合はキャッシュから削除
                if time.time() - completed_task.created_at > self.cache_ttl_seconds:
                    del self.completed_tasks[task_key]
                    return False, task_key

                logger.info(
                    "重複リクエスト検出（完了済み）",
                    extra={
                        "symbol": symbol,
                        "task_key": task_key[:8],
                        "cached_result_available": completed_task.result is not None,
                        "age_seconds": round(
                            time.time() - completed_task.created_at, 2
                        ),
                    },
                )

                return True, task_key

            return False, task_key

    def register_computation_task(
        self, symbol: str, data: pd.DataFrame, feature_config: FeatureConfig
    ) -> str:
        """計算タスクの登録"""
        data_hash = self._generate_data_hash(data)
        task_key = self._generate_task_key(symbol, data_hash, feature_config)

        with self._lock:
            self.stats.total_requests += 1

            # 重複チェック
            is_duplicate, _ = self.is_duplicate_request(symbol, data, feature_config)

            if is_duplicate:
                self.stats.duplicate_requests += 1
                # 待機リクエストに追加
                self.waiting_requests[task_key].append(symbol)
                logger.info(
                    "重複計算をスキップ",
                    extra={
                        "symbol": symbol,
                        "task_key": task_key[:8],
                        "waiting_requests": len(self.waiting_requests[task_key]),
                    },
                )
                return task_key

            # 新規タスクとして登録
            config_hash = hashlib.md5(str(feature_config).encode()).hexdigest()[:8]

            task = ComputationTask(
                task_id=f"task_{int(time.time()*1000)}_{len(self.active_tasks)}",
                symbol=symbol,
                config_hash=config_hash,
                data_hash=data_hash,
                feature_config=feature_config,
                status="pending",
            )

            self.active_tasks[task_key] = task

            # 最大同時タスク数の制限
            if len(self.active_tasks) > self.max_concurrent_tasks:
                self._cleanup_old_tasks()

            logger.info(
                "新規計算タスク登録",
                extra={
                    "symbol": symbol,
                    "task_key": task_key[:8],
                    "task_id": task.task_id,
                    "active_tasks_count": len(self.active_tasks),
                },
            )

            return task_key

    def start_computation(self, task_key: str) -> bool:
        """計算開始"""
        with self._lock:
            if task_key not in self.active_tasks:
                return False

            task = self.active_tasks[task_key]
            if task.status != "pending":
                return False

            task.status = "computing"
            task.created_at = time.time()  # 計算開始時刻で更新

            logger.info(
                "計算開始",
                extra={
                    "task_key": task_key[:8],
                    "symbol": task.symbol,
                    "task_id": task.task_id,
                },
            )

            return True

    def complete_computation(
        self, task_key: str, result: Any, computation_time: float, success: bool = True
    ) -> List[str]:
        """計算完了処理"""
        with self._lock:
            if task_key not in self.active_tasks:
                return []

            task = self.active_tasks[task_key]
            task.status = "completed" if success else "failed"
            task.result = result
            task.computation_time = computation_time

            # 統計更新
            if success:
                self.stats.completed_computations += 1
                self.stats.total_computation_time += computation_time

                # 重複排除による時間節約の推定
                waiting_count = len(self.waiting_requests.get(task_key, []))
                if waiting_count > 0:
                    self.stats.time_saved_by_dedup += computation_time * waiting_count
            else:
                self.stats.failed_computations += 1

            # 待機中のリクエストを取得
            waiting_symbols = self.waiting_requests.get(task_key, []).copy()

            # 完了済みタスクキャッシュに移動
            if success and result is not None:
                self.completed_tasks[task_key] = task

            # アクティブタスクから削除
            del self.active_tasks[task_key]

            # 待機リクエストをクリア
            if task_key in self.waiting_requests:
                del self.waiting_requests[task_key]

            # 履歴に追加
            self.task_history.append(
                {
                    "task_key": task_key[:8],
                    "symbol": task.symbol,
                    "status": task.status,
                    "computation_time": computation_time,
                    "waiting_requests": len(waiting_symbols),
                    "completed_at": time.time(),
                }
            )

            logger.info(
                "計算完了",
                extra={
                    "task_key": task_key[:8],
                    "symbol": task.symbol,
                    "success": success,
                    "computation_time_ms": round(computation_time * 1000, 2),
                    "waiting_requests_notified": len(waiting_symbols),
                    "active_tasks_remaining": len(self.active_tasks),
                },
            )

            return waiting_symbols

    def get_computation_result(self, task_key: str) -> Optional[Any]:
        """計算結果の取得"""
        with self._lock:
            # 完了済みキャッシュから取得
            if task_key in self.completed_tasks:
                task = self.completed_tasks[task_key]
                return task.result

            # アクティブタスクから取得（完了している場合）
            if task_key in self.active_tasks:
                task = self.active_tasks[task_key]
                if task.status == "completed" and task.result is not None:
                    return task.result

            return None

    def get_task_status(self, task_key: str) -> Optional[str]:
        """タスクステータスの取得"""
        with self._lock:
            if task_key in self.active_tasks:
                return self.active_tasks[task_key].status
            elif task_key in self.completed_tasks:
                return "cached"
            else:
                return None

    def _cleanup_old_tasks(self):
        """古いタスクのクリーンアップ"""
        current_time = time.time()
        tasks_to_remove = []

        for task_key, task in self.active_tasks.items():
            # 古いタスクや失敗したタスクを削除
            age = current_time - task.created_at
            if age > self.cache_ttl_seconds or task.status == "failed":
                tasks_to_remove.append(task_key)

        for task_key in tasks_to_remove:
            if task_key in self.active_tasks:
                del self.active_tasks[task_key]
            if task_key in self.waiting_requests:
                del self.waiting_requests[task_key]

        if tasks_to_remove:
            logger.info(f"古いタスクをクリーンアップ: {len(tasks_to_remove)}件")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報の取得"""
        with self._lock:
            return {
                "total_requests": self.stats.total_requests,
                "duplicate_requests": self.stats.duplicate_requests,
                "deduplication_rate_percent": round(self.stats.deduplication_rate, 2),
                "completed_computations": self.stats.completed_computations,
                "failed_computations": self.stats.failed_computations,
                "success_rate_percent": round(self.stats.success_rate, 2),
                "avg_computation_time_seconds": round(
                    self.stats.avg_computation_time, 3
                ),
                "time_saved_by_dedup_seconds": round(self.stats.time_saved_by_dedup, 2),
                "active_tasks_count": len(self.active_tasks),
                "cached_results_count": len(self.completed_tasks),
                "waiting_requests_count": sum(
                    len(requests) for requests in self.waiting_requests.values()
                ),
            }

    def reset_statistics(self):
        """統計情報のリセット"""
        with self._lock:
            self.stats = ComputationStats()
            logger.info("重複計算排除統計情報をリセット")

    def clear_cache(self):
        """キャッシュのクリア"""
        with self._lock:
            self.completed_tasks.clear()
            self.active_tasks.clear()
            self.waiting_requests.clear()
            self.task_history.clear()
            logger.info("重複計算排除キャッシュをクリア")


# グローバルインスタンス
_global_dedup_manager = None


def get_deduplication_manager() -> FeatureDeduplicationManager:
    """グローバル重複計算排除マネージャーの取得"""
    global _global_dedup_manager
    if _global_dedup_manager is None:
        _global_dedup_manager = FeatureDeduplicationManager()
    return _global_dedup_manager


def reset_deduplication_manager():
    """グローバル重複計算排除マネージャーのリセット"""
    global _global_dedup_manager
    _global_dedup_manager = None
