#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メモリ最適化ユーティリティ

システム全体のメモリ使用量を監視・最適化します。
"""

import gc
import os
import psutil
import logging
import threading
import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import weakref


@dataclass
class MemoryStats:
    """メモリ統計情報"""
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    process_mb: float
    timestamp: datetime


class MemoryOptimizer:
    """
    メモリ最適化管理クラス

    システムのメモリ使用量を監視し、必要に応じて最適化を実行
    """

    def __init__(self,
                 warning_threshold: float = 80.0,
                 critical_threshold: float = 90.0,
                 cleanup_interval: int = 300):
        """
        初期化

        Args:
            warning_threshold: 警告しきい値（%）
            critical_threshold: 緊急しきい値（%）
            cleanup_interval: クリーンアップ間隔（秒）
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.cleanup_interval = cleanup_interval

        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # コールバック関数リスト
        self.warning_callbacks: list[Callable[[MemoryStats], None]] = []
        self.critical_callbacks: list[Callable[[MemoryStats], None]] = []

        # オブジェクト参照管理
        self.cached_objects: weakref.WeakSet = weakref.WeakSet()

        self.logger.info(
            f"メモリ最適化システム初期化完了 "
            f"(警告: {warning_threshold}%, 緊急: {critical_threshold}%)"
        )

    def get_memory_stats(self) -> MemoryStats:
        """現在のメモリ統計を取得"""
        virtual_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()

        return MemoryStats(
            total_mb=virtual_memory.total / 1024 / 1024,
            available_mb=virtual_memory.available / 1024 / 1024,
            used_mb=virtual_memory.used / 1024 / 1024,
            usage_percent=virtual_memory.percent,
            process_mb=process_memory.rss / 1024 / 1024,
            timestamp=datetime.now()
        )

    def register_warning_callback(self, callback: Callable[[MemoryStats], None]):
        """警告時のコールバック関数を登録"""
        self.warning_callbacks.append(callback)

    def register_critical_callback(self, callback: Callable[[MemoryStats], None]):
        """緊急時のコールバック関数を登録"""
        self.critical_callbacks.append(callback)

    def register_cached_object(self, obj: Any):
        """キャッシュオブジェクトを登録（弱参照で管理）"""
        self.cached_objects.add(obj)

    def force_garbage_collection(self) -> Dict[str, int]:
        """強制ガベージコレクション実行"""
        self.logger.info("強制ガベージコレクション開始")

        # 3回実行して確実に回収
        collected = {
            'generation_0': 0,
            'generation_1': 0,
            'generation_2': 0
        }

        for i in range(3):
            collected['generation_0'] += gc.collect(0)
            collected['generation_1'] += gc.collect(1)
            collected['generation_2'] += gc.collect(2)

        total_collected = sum(collected.values())
        self.logger.info(f"ガベージコレクション完了: {total_collected}オブジェクト回収")

        return collected

    def clear_caches(self):
        """登録されたキャッシュをクリア"""
        cleared_count = 0

        # 弱参照で管理されているオブジェクトのクリア
        for obj in list(self.cached_objects):
            if hasattr(obj, 'clear'):
                try:
                    obj.clear()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"キャッシュクリアエラー: {e}")

        self.logger.info(f"キャッシュクリア完了: {cleared_count}個のキャッシュ")

    def emergency_memory_cleanup(self):
        """緊急メモリクリーンアップ"""
        self.logger.warning("緊急メモリクリーンアップ実行")

        stats_before = self.get_memory_stats()

        # 1. キャッシュクリア
        self.clear_caches()

        # 2. ガベージコレクション
        self.force_garbage_collection()

        # 3. 追加のPython最適化
        self._optimize_python_memory()

        stats_after = self.get_memory_stats()
        freed_mb = stats_before.process_mb - stats_after.process_mb

        self.logger.info(
            f"緊急クリーンアップ完了: {freed_mb:.1f}MB解放 "
            f"({stats_after.usage_percent:.1f}%使用)"
        )

    def _optimize_python_memory(self):
        """Python固有のメモリ最適化"""
        # インターンされた文字列をクリア
        if hasattr(sys, 'intern'):
            # Python内部の最適化
            pass

        # モジュールキャッシュの一部クリア（安全なもののみ）
        modules_to_clear = []
        for module_name in list(sys.modules.keys()):
            if (module_name.startswith('__pycache__') or
                module_name.endswith('.pyc')):
                modules_to_clear.append(module_name)

        for module_name in modules_to_clear:
            try:
                del sys.modules[module_name]
            except KeyError:
                pass

    def _monitor_loop(self):
        """メモリ監視ループ"""
        last_cleanup = time.time()

        while self.monitoring:
            try:
                stats = self.get_memory_stats()

                # 定期クリーンアップ
                current_time = time.time()
                if current_time - last_cleanup > self.cleanup_interval:
                    self.force_garbage_collection()
                    last_cleanup = current_time

                # しきい値チェック
                if stats.usage_percent >= self.critical_threshold:
                    self.logger.critical(
                        f"メモリ使用量が緊急レベル: {stats.usage_percent:.1f}%"
                    )
                    self.emergency_memory_cleanup()

                    # 緊急コールバック実行
                    for callback in self.critical_callbacks:
                        try:
                            callback(stats)
                        except Exception as e:
                            self.logger.error(f"緊急コールバックエラー: {e}")

                elif stats.usage_percent >= self.warning_threshold:
                    self.logger.warning(
                        f"メモリ使用量が警告レベル: {stats.usage_percent:.1f}%"
                    )

                    # 警告コールバック実行
                    for callback in self.warning_callbacks:
                        try:
                            callback(stats)
                        except Exception as e:
                            self.logger.error(f"警告コールバックエラー: {e}")

                # 監視間隔
                time.sleep(10)

            except Exception as e:
                self.logger.error(f"メモリ監視エラー: {e}")
                time.sleep(30)

    def start_monitoring(self):
        """メモリ監視開始"""
        if self.monitoring:
            self.logger.warning("メモリ監視は既に開始されています")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()

        self.logger.info("メモリ監視開始")

    def stop_monitoring(self):
        """メモリ監視停止"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("メモリ監視停止")

    def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポートを取得"""
        stats = self.get_memory_stats()
        gc_stats = gc.get_stats()

        return {
            "memory_stats": {
                "total_mb": round(stats.total_mb, 1),
                "available_mb": round(stats.available_mb, 1),
                "used_mb": round(stats.used_mb, 1),
                "usage_percent": round(stats.usage_percent, 1),
                "process_mb": round(stats.process_mb, 1)
            },
            "gc_stats": gc_stats,
            "thresholds": {
                "warning": self.warning_threshold,
                "critical": self.critical_threshold
            },
            "monitoring": self.monitoring,
            "cached_objects_count": len(self.cached_objects)
        }


# グローバルメモリ最適化インスタンス
_memory_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """グローバルメモリ最適化インスタンスを取得"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def optimize_memory():
    """メモリ最適化を実行"""
    optimizer = get_memory_optimizer()
    optimizer.force_garbage_collection()


def get_memory_stats() -> MemoryStats:
    """現在のメモリ統計を取得"""
    optimizer = get_memory_optimizer()
    return optimizer.get_memory_stats()


def register_cache(cache_obj: Any):
    """キャッシュオブジェクトを登録"""
    optimizer = get_memory_optimizer()
    optimizer.register_cached_object(cache_obj)


# デフォルトコールバック関数
def default_warning_callback(stats: MemoryStats):
    """デフォルト警告処理"""
    logging.getLogger(__name__).warning(
        f"メモリ使用量警告: {stats.usage_percent:.1f}% "
        f"(プロセス: {stats.process_mb:.1f}MB)"
    )


def default_critical_callback(stats: MemoryStats):
    """デフォルト緊急処理"""
    logging.getLogger(__name__).critical(
        f"メモリ使用量緊急: {stats.usage_percent:.1f}% "
        f"(プロセス: {stats.process_mb:.1f}MB)"
    )

    # 緊急時の追加処理
    optimizer = get_memory_optimizer()
    optimizer.emergency_memory_cleanup()


# 初期化時にデフォルトコールバックを登録
def _initialize_default_callbacks():
    """デフォルトコールバックを初期化"""
    optimizer = get_memory_optimizer()
    optimizer.register_warning_callback(default_warning_callback)
    optimizer.register_critical_callback(default_critical_callback)


# モジュール読み込み時に初期化
_initialize_default_callbacks()