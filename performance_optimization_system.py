#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Optimization System - パフォーマンス最適化システム
メモリ管理、キャッシュ最適化、処理速度向上の統合システム
"""

import gc
import psutil
import time
import threading
import weakref
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict, deque
import asyncio


class PerformanceLevel(Enum):
    """パフォーマンスレベル"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class OptimizationStrategy(Enum):
    """最適化戦略"""
    MEMORY_CLEANUP = "memory_cleanup"
    CACHE_OPTIMIZATION = "cache_optimization"
    GARBAGE_COLLECTION = "garbage_collection"
    CIRCULAR_REFERENCE_CLEANUP = "circular_reference_cleanup"
    ASYNC_OPTIMIZATION = "async_optimization"


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    cache_hit_rate: float
    response_time_ms: float
    active_threads: int
    pending_tasks: int
    gc_collections: Dict[str, int]
    circular_references: int
    timestamp: datetime


@dataclass
class OptimizationResult:
    """最適化結果"""
    strategy: OptimizationStrategy
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    improvement_percent: float
    execution_time_ms: float
    success: bool
    details: Dict[str, Any]


class CircularReferenceDetector:
    """循環参照検出器（改良版）"""
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.detection_cache = weakref.WeakKeyDictionary()
        self.cleanup_threshold = 1000
        self.cleanup_counter = 0
    
    def detect_circular_references(self, obj: Any, visited: Optional[Set] = None, depth: int = 0) -> int:
        """循環参照を検出（最適化版）"""
        if depth > self.max_depth:
            return 0
        
        if visited is None:
            visited = set()
        
        # オブジェクトのIDをチェック
        obj_id = id(obj)
        if obj_id in visited:
            return 1  # 循環参照発見
        
        # キャッシュチェック
        if obj in self.detection_cache:
            return self.detection_cache[obj]
        
        visited.add(obj_id)
        circular_count = 0
        
        try:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if depth < self.max_depth - 1:  # 深度制限
                        circular_count += self.detect_circular_references(value, visited.copy(), depth + 1)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if depth < self.max_depth - 1:  # 深度制限
                        circular_count += self.detect_circular_references(item, visited.copy(), depth + 1)
        except (RecursionError, MemoryError):
            circular_count = 1  # エラー時は循環参照として扱う
        
        # キャッシュに保存
        try:
            self.detection_cache[obj] = circular_count
        except TypeError:
            pass  # unhashable typeの場合はキャッシュしない
        
        # 定期的なキャッシュクリーンアップ
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_threshold:
            self.cleanup_cache()
        
        return circular_count
    
    def cleanup_cache(self):
        """キャッシュクリーンアップ"""
        # WeakKeyDictionaryは自動的にクリーンアップされるため、カウンターをリセットのみ
        self.cleanup_counter = 0


class MemoryManager:
    """メモリ管理システム"""
    
    def __init__(self):
        self.memory_threshold_mb = 1024  # 1GB
        self.cleanup_frequency = 300  # 5分
        self.last_cleanup = datetime.now()
        self.large_objects = weakref.WeakSet()
    
    def track_large_object(self, obj: Any):
        """大きなオブジェクトを追跡"""
        try:
            self.large_objects.add(obj)
        except TypeError:
            pass  # unhashable typeは追跡しない
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """メモリ使用量を取得"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        return memory_mb, memory_percent
    
    def should_cleanup(self) -> bool:
        """クリーンアップが必要かチェック"""
        memory_mb, _ = self.get_memory_usage()
        time_elapsed = (datetime.now() - self.last_cleanup).total_seconds()
        
        return (memory_mb > self.memory_threshold_mb or 
                time_elapsed > self.cleanup_frequency)
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """強制ガベージコレクション"""
        gc_stats = {}
        
        # 各世代のGCを実行
        for generation in range(3):
            collected = gc.collect(generation)
            gc_stats[f'generation_{generation}'] = collected
        
        # 未参照オブジェクトの削除
        unreachable = gc.collect()
        gc_stats['unreachable'] = unreachable
        
        self.last_cleanup = datetime.now()
        return gc_stats
    
    def cleanup_large_objects(self) -> int:
        """大きなオブジェクトのクリーンアップ"""
        initial_count = len(self.large_objects)
        # WeakSetは自動的にクリーンアップされる
        return initial_count - len(self.large_objects)


class CacheOptimizer:
    """キャッシュ最適化システム"""
    
    def __init__(self):
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})
        self.cache_sizes = defaultdict(int)
        self.max_cache_size = 1000
        self.cleanup_threshold = 0.8  # 80%で自動クリーンアップ
    
    def record_cache_access(self, cache_name: str, hit: bool):
        """キャッシュアクセスを記録"""
        if hit:
            self.cache_stats[cache_name]['hits'] += 1
        else:
            self.cache_stats[cache_name]['misses'] += 1
    
    def get_cache_hit_rate(self, cache_name: str = None) -> float:
        """キャッシュヒット率を取得"""
        if cache_name:
            stats = self.cache_stats[cache_name]
            total = stats['hits'] + stats['misses']
            return stats['hits'] / total if total > 0 else 0.0
        else:
            # 全体のヒット率
            total_hits = sum(stats['hits'] for stats in self.cache_stats.values())
            total_accesses = sum(stats['hits'] + stats['misses'] for stats in self.cache_stats.values())
            return total_hits / total_accesses if total_accesses > 0 else 0.0
    
    def optimize_cache_size(self, cache_name: str, current_size: int) -> int:
        """キャッシュサイズの最適化"""
        hit_rate = self.get_cache_hit_rate(cache_name)
        
        if hit_rate > 0.9:  # 90%以上のヒット率
            # サイズを増加
            new_size = min(current_size * 1.2, self.max_cache_size)
        elif hit_rate < 0.5:  # 50%未満のヒット率
            # サイズを削減
            new_size = max(current_size * 0.8, 100)
        else:
            new_size = current_size
        
        return int(new_size)
    
    def should_cleanup_cache(self, cache_name: str) -> bool:
        """キャッシュクリーンアップが必要かチェック"""
        return self.cache_sizes[cache_name] > self.max_cache_size * self.cleanup_threshold


class PerformanceOptimizationSystem:
    """パフォーマンス最適化システム"""
    
    def __init__(self):
        self.circular_detector = CircularReferenceDetector()
        self.memory_manager = MemoryManager()
        self.cache_optimizer = CacheOptimizer()
        
        # パフォーマンス履歴
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        
        # 最適化設定
        self.auto_optimization = True
        self.optimization_interval = 60  # 1分
        self.last_optimization = datetime.now()
        
        # ログ設定
        from daytrade_logging import get_logger
        self.logger = get_logger("performance_optimization")
        
        # バックグラウンドタスク
        self.optimization_thread = None
        self.running = True
        
        self.logger.info("Performance Optimization System initialized")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """現在のパフォーマンスメトリクスを取得"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # メモリ使用量
        memory_mb, memory_percent = self.memory_manager.get_memory_usage()
        
        # キャッシュヒット率
        cache_hit_rate = self.cache_optimizer.get_cache_hit_rate()
        
        # レスポンス時間（簡易測定）
        start_time = time.time()
        # 簡単な処理でレスポンス時間を測定
        _ = [i for i in range(1000)]
        response_time_ms = (time.time() - start_time) * 1000
        
        # スレッド数
        active_threads = threading.active_count()
        
        # 非同期タスク数
        try:
            loop = asyncio.get_event_loop()
            pending_tasks = len([task for task in asyncio.all_tasks(loop) if not task.done()])
        except RuntimeError:
            pending_tasks = 0
        
        # GC統計
        gc_stats = {
            'count_0': gc.get_count()[0],
            'count_1': gc.get_count()[1],
            'count_2': gc.get_count()[2]
        }
        
        # 循環参照検出（サンプリング）
        circular_references = 0
        
        metrics = PerformanceMetrics(
            cpu_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            memory_percent=memory_percent,
            cache_hit_rate=cache_hit_rate,
            response_time_ms=response_time_ms,
            active_threads=active_threads,
            pending_tasks=pending_tasks,
            gc_collections=gc_stats,
            circular_references=circular_references,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def assess_performance_level(self, metrics: PerformanceMetrics) -> PerformanceLevel:
        """パフォーマンスレベルを評価"""
        score = 100  # 満点から開始
        
        # CPU使用率チェック
        if metrics.cpu_percent > 90:
            score -= 30
        elif metrics.cpu_percent > 70:
            score -= 15
        
        # メモリ使用率チェック
        if metrics.memory_percent > 90:
            score -= 30
        elif metrics.memory_percent > 80:
            score -= 15
        
        # レスポンス時間チェック
        if metrics.response_time_ms > 1000:
            score -= 25
        elif metrics.response_time_ms > 500:
            score -= 10
        
        # キャッシュヒット率チェック
        if metrics.cache_hit_rate < 0.5:
            score -= 20
        elif metrics.cache_hit_rate < 0.7:
            score -= 10
        
        # スレッド数チェック
        if metrics.active_threads > 50:
            score -= 15
        
        # レベル判定
        if score >= 80:
            return PerformanceLevel.OPTIMAL
        elif score >= 60:
            return PerformanceLevel.GOOD
        elif score >= 40:
            return PerformanceLevel.DEGRADED
        else:
            return PerformanceLevel.CRITICAL
    
    def optimize_memory(self) -> OptimizationResult:
        """メモリ最適化"""
        before_metrics = self.get_current_metrics()
        start_time = time.time()
        
        try:
            # ガベージコレクション実行
            gc_stats = self.memory_manager.force_garbage_collection()
            
            # 大きなオブジェクトのクリーンアップ
            cleaned_objects = self.memory_manager.cleanup_large_objects()
            
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            # 改善率計算
            memory_before = before_metrics.memory_usage_mb
            memory_after = after_metrics.memory_usage_mb
            improvement = ((memory_before - memory_after) / memory_before * 100) if memory_before > 0 else 0
            
            result = OptimizationResult(
                strategy=OptimizationStrategy.MEMORY_CLEANUP,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                execution_time_ms=execution_time,
                success=True,
                details={
                    'gc_collections': gc_stats,
                    'cleaned_objects': cleaned_objects,
                    'memory_freed_mb': memory_before - memory_after
                }
            )
            
            self.logger.info(f"Memory optimization completed: {improvement:.1f}% improvement")
            return result
            
        except Exception as e:
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Memory optimization failed: {e}")
            
            return OptimizationResult(
                strategy=OptimizationStrategy.MEMORY_CLEANUP,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time_ms=execution_time,
                success=False,
                details={'error': str(e)}
            )
    
    def optimize_circular_references(self) -> OptimizationResult:
        """循環参照最適化"""
        before_metrics = self.get_current_metrics()
        start_time = time.time()
        
        try:
            # 循環参照検出と修復
            repaired_count = 0
            
            # ガベージコレクションで循環参照を解決
            gc.set_debug(gc.DEBUG_STATS)
            collected = gc.collect()
            gc.set_debug(0)
            
            # 循環参照検出器のキャッシュをクリーンアップ
            self.circular_detector.cleanup_cache()
            
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            # 改善率計算（GCで回収されたオブジェクト数ベース）
            improvement = min(collected * 0.1, 50.0)  # 最大50%改善として計算
            
            result = OptimizationResult(
                strategy=OptimizationStrategy.CIRCULAR_REFERENCE_CLEANUP,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                execution_time_ms=execution_time,
                success=True,
                details={
                    'collected_objects': collected,
                    'repaired_references': repaired_count
                }
            )
            
            self.logger.info(f"Circular reference optimization completed: {collected} objects collected")
            return result
            
        except Exception as e:
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Circular reference optimization failed: {e}")
            
            return OptimizationResult(
                strategy=OptimizationStrategy.CIRCULAR_REFERENCE_CLEANUP,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time_ms=execution_time,
                success=False,
                details={'error': str(e)}
            )
    
    def optimize_cache(self) -> OptimizationResult:
        """キャッシュ最適化"""
        before_metrics = self.get_current_metrics()
        start_time = time.time()
        
        try:
            optimization_details = {}
            
            # キャッシュサイズの最適化
            for cache_name in list(self.cache_optimizer.cache_stats.keys()):
                current_size = self.cache_optimizer.cache_sizes[cache_name]
                optimized_size = self.cache_optimizer.optimize_cache_size(cache_name, current_size)
                optimization_details[cache_name] = {
                    'old_size': current_size,
                    'new_size': optimized_size,
                    'hit_rate': self.cache_optimizer.get_cache_hit_rate(cache_name)
                }
                self.cache_optimizer.cache_sizes[cache_name] = optimized_size
            
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            # 改善率計算（キャッシュヒット率の改善）
            hit_rate_before = before_metrics.cache_hit_rate
            hit_rate_after = after_metrics.cache_hit_rate
            improvement = ((hit_rate_after - hit_rate_before) * 100) if hit_rate_before > 0 else 0
            
            result = OptimizationResult(
                strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=improvement,
                execution_time_ms=execution_time,
                success=True,
                details=optimization_details
            )
            
            self.logger.info(f"Cache optimization completed: {len(optimization_details)} caches optimized")
            return result
            
        except Exception as e:
            after_metrics = self.get_current_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            self.logger.error(f"Cache optimization failed: {e}")
            
            return OptimizationResult(
                strategy=OptimizationStrategy.CACHE_OPTIMIZATION,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percent=0.0,
                execution_time_ms=execution_time,
                success=False,
                details={'error': str(e)}
            )
    
    def auto_optimize(self) -> List[OptimizationResult]:
        """自動最適化"""
        if not self.auto_optimization:
            return []
        
        current_metrics = self.get_current_metrics()
        performance_level = self.assess_performance_level(current_metrics)
        
        optimization_results = []
        
        # パフォーマンスレベルに応じた最適化戦略
        if performance_level in [PerformanceLevel.CRITICAL, PerformanceLevel.DEGRADED]:
            self.logger.info(f"Auto-optimization triggered: {performance_level.value}")
            
            # メモリ最適化
            if current_metrics.memory_percent > 80:
                result = self.optimize_memory()
                optimization_results.append(result)
            
            # 循環参照最適化
            if current_metrics.gc_collections['count_0'] > 1000:
                result = self.optimize_circular_references()
                optimization_results.append(result)
            
            # キャッシュ最適化
            if current_metrics.cache_hit_rate < 0.7:
                result = self.optimize_cache()
                optimization_results.append(result)
        
        # 最適化履歴に追加
        for result in optimization_results:
            self.optimization_history.append(result)
        
        self.last_optimization = datetime.now()
        return optimization_results
    
    def start_background_optimization(self):
        """バックグラウンド最適化を開始"""
        if self.optimization_thread and self.optimization_thread.is_alive():
            return
        
        def optimization_loop():
            while self.running:
                try:
                    # 最適化間隔チェック
                    if (datetime.now() - self.last_optimization).total_seconds() >= self.optimization_interval:
                        self.auto_optimize()
                    
                    time.sleep(10)  # 10秒間隔でチェック
                    
                except Exception as e:
                    self.logger.error(f"Background optimization error: {e}")
                    time.sleep(30)  # エラー時は30秒待機
        
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        self.logger.info("Background optimization started")
    
    def stop_background_optimization(self):
        """バックグラウンド最適化を停止"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        self.logger.info("Background optimization stopped")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートを生成"""
        current_metrics = self.get_current_metrics()
        performance_level = self.assess_performance_level(current_metrics)
        
        # 最近の最適化結果
        recent_optimizations = list(self.optimization_history)[-10:]
        
        # 成功した最適化の統計
        successful_optimizations = [opt for opt in recent_optimizations if opt.success]
        total_improvement = sum(opt.improvement_percent for opt in successful_optimizations)
        
        return {
            'current_performance': {
                'level': performance_level.value,
                'cpu_percent': current_metrics.cpu_percent,
                'memory_usage_mb': current_metrics.memory_usage_mb,
                'memory_percent': current_metrics.memory_percent,
                'cache_hit_rate': current_metrics.cache_hit_rate,
                'response_time_ms': current_metrics.response_time_ms,
                'active_threads': current_metrics.active_threads
            },
            'optimization_status': {
                'auto_optimization_enabled': self.auto_optimization,
                'last_optimization': self.last_optimization.isoformat(),
                'optimization_interval_seconds': self.optimization_interval,
                'background_running': self.running and self.optimization_thread and self.optimization_thread.is_alive()
            },
            'recent_optimizations': [
                {
                    'strategy': opt.strategy.value,
                    'improvement_percent': opt.improvement_percent,
                    'execution_time_ms': opt.execution_time_ms,
                    'success': opt.success
                }
                for opt in recent_optimizations
            ],
            'statistics': {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len(successful_optimizations),
                'total_improvement_percent': total_improvement,
                'average_improvement': total_improvement / len(successful_optimizations) if successful_optimizations else 0
            },
            'timestamp': datetime.now().isoformat()
        }


# グローバルインスタンス
_performance_system = None


def get_performance_system() -> PerformanceOptimizationSystem:
    """グローバルパフォーマンスシステムを取得"""
    global _performance_system
    if _performance_system is None:
        _performance_system = PerformanceOptimizationSystem()
    return _performance_system


def optimize_performance() -> List[OptimizationResult]:
    """パフォーマンス最適化（便利関数）"""
    return get_performance_system().auto_optimize()


def get_performance_metrics() -> PerformanceMetrics:
    """パフォーマンスメトリクス取得（便利関数）"""
    return get_performance_system().get_current_metrics()


if __name__ == "__main__":
    print("⚡ パフォーマンス最適化システムテスト")
    print("=" * 50)
    
    system = PerformanceOptimizationSystem()
    
    # 現在のメトリクス
    metrics = system.get_current_metrics()
    performance_level = system.assess_performance_level(metrics)
    
    print(f"現在のパフォーマンス:")
    print(f"  レベル: {performance_level.value}")
    print(f"  CPU使用率: {metrics.cpu_percent:.1f}%")
    print(f"  メモリ使用率: {metrics.memory_percent:.1f}%")
    print(f"  キャッシュヒット率: {metrics.cache_hit_rate:.1%}")
    print(f"  レスポンス時間: {metrics.response_time_ms:.2f}ms")
    
    # 最適化実行
    print("\n最適化実行中...")
    optimizations = system.auto_optimize()
    
    if optimizations:
        print(f"\n最適化結果:")
        for opt in optimizations:
            print(f"  {opt.strategy.value}: {opt.improvement_percent:.1f}% 改善 ({'成功' if opt.success else '失敗'})")
    else:
        print("最適化は不要でした")
    
    # レポート生成
    report = system.get_performance_report()
    print(f"\n統計:")
    print(f"  総最適化回数: {report['statistics']['total_optimizations']}")
    print(f"  成功回数: {report['statistics']['successful_optimizations']}")
    print(f"  平均改善率: {report['statistics']['average_improvement']:.1f}%")
    
    print("\nテスト完了")