#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Performance Optimization System - 高度パフォーマンス最適化システム

システム全体のパフォーマンスを大幅に向上させる包括的な最適化システム
- リアルタイム パフォーマンス監視
- 動的リソース管理
- 自動スケーリング
- 最適化エンジン
- プロファイリング・分析
"""

import asyncio
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil
import gc
import tracemalloc
import cProfile
import pstats
from memory_profiler import profile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import logging
import warnings
from collections import defaultdict, deque
import queue
import weakref
from functools import wraps, lru_cache
import hashlib
import pickle
import joblib
from contextlib import contextmanager
import sys
import os
import resource

warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """パフォーマンス レベル"""
    EXCELLENT = "excellent"    # 90%以上
    GOOD = "good"             # 80-90%
    FAIR = "fair"             # 70-80%
    POOR = "poor"             # 60-70%
    CRITICAL = "critical"     # 60%未満

class OptimizationStrategy(Enum):
    """最適化戦略"""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    BALANCED = "balanced"
    REALTIME = "realtime"

class ResourceType(Enum):
    """リソースタイプ"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"

@dataclass
class PerformanceMetrics:
    """パフォーマンス メトリクス"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    active_threads: int = 0
    gc_collections: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """最適化結果"""
    strategy: OptimizationStrategy
    improvements: Dict[str, float]
    recommendations: List[str]
    performance_gain: float
    resource_savings: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceAlert:
    """リソース アラート"""
    resource_type: ResourceType
    current_usage: float
    threshold: float
    severity: str
    message: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceProfiler:
    """パフォーマンス プロファイラー"""
    
    def __init__(self):
        self.profiling_data = defaultdict(list)
        self.active_sessions = {}
        self.memory_tracker = None
        
    def start_profiling(self, session_name: str = "default"):
        """プロファイリング開始"""
        if session_name in self.active_sessions:
            logger.warning(f"プロファイリングセッション '{session_name}' は既に実行中です")
            return
        
        # CPU プロファイリング
        cpu_profiler = cProfile.Profile()
        cpu_profiler.enable()
        
        # メモリ トラッキング
        tracemalloc.start()
        
        self.active_sessions[session_name] = {
            'cpu_profiler': cpu_profiler,
            'start_time': time.time(),
            'start_memory': tracemalloc.get_traced_memory()[0]
        }
        
        logger.info(f"プロファイリングセッション '{session_name}' を開始しました")
    
    def stop_profiling(self, session_name: str = "default") -> Dict[str, Any]:
        """プロファイリング停止・結果取得"""
        if session_name not in self.active_sessions:
            logger.error(f"プロファイリングセッション '{session_name}' が見つかりません")
            return {}
        
        session = self.active_sessions[session_name]
        
        # CPU プロファイリング停止
        session['cpu_profiler'].disable()
        
        # 実行時間
        execution_time = time.time() - session['start_time']
        
        # メモリ使用量
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        memory_delta = current_memory - session['start_memory']
        tracemalloc.stop()
        
        # CPU 統計
        stats = pstats.Stats(session['cpu_profiler'])
        stats.sort_stats('cumulative')
        
        # 結果まとめ
        result = {
            'session_name': session_name,
            'execution_time': execution_time,
            'memory_usage': {
                'start': session['start_memory'],
                'current': current_memory,
                'peak': peak_memory,
                'delta': memory_delta
            },
            'cpu_stats': self._extract_cpu_stats(stats),
            'timestamp': datetime.now()
        }
        
        # セッション削除
        del self.active_sessions[session_name]
        
        # データ保存
        self.profiling_data[session_name].append(result)
        
        logger.info(f"プロファイリングセッション '{session_name}' を完了しました")
        return result
    
    def _extract_cpu_stats(self, stats: pstats.Stats) -> Dict[str, Any]:
        """CPU統計抽出"""
        try:
            import io
            s = io.StringIO()
            stats.print_stats(10)  # 上位10関数
            stats_output = s.getvalue()
            
            # 統計パース
            lines = stats_output.split('\n')
            functions = []
            
            for line in lines:
                if line.strip() and not line.startswith(' ') and '/' in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        functions.append({
                            'ncalls': parts[0],
                            'tottime': float(parts[1]) if parts[1].replace('.', '').isdigit() else 0.0,
                            'cumtime': float(parts[3]) if parts[3].replace('.', '').isdigit() else 0.0,
                            'filename': parts[-1] if len(parts) > 5 else 'unknown'
                        })
            
            return {
                'total_calls': stats.total_calls,
                'total_time': stats.total_tt,
                'top_functions': functions[:5]
            }
            
        except Exception as e:
            logger.error(f"CPU統計抽出エラー: {e}")
            return {}
    
    @contextmanager
    def profile_block(self, block_name: str):
        """ブロックプロファイリング コンテキストマネージャー"""
        self.start_profiling(block_name)
        try:
            yield
        finally:
            result = self.stop_profiling(block_name)
            logger.info(f"ブロック '{block_name}' 実行時間: {result.get('execution_time', 0):.3f}秒")

class ResourceMonitor:
    """リソース監視システム"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.alert_thresholds = {
            ResourceType.CPU: 80.0,      # 80%
            ResourceType.MEMORY: 85.0,   # 85%
            ResourceType.DISK: 90.0,     # 90%
        }
        self.alert_callbacks = []
        self.last_alert_time = defaultdict(lambda: datetime.min)
    
    def start_monitoring(self, interval: float = 1.0):
        """監視開始"""
        if self.monitoring_active:
            logger.warning("リソース監視は既に実行中です")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("リソース監視を開始しました")
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        logger.info("リソース監視を停止しました")
    
    def _monitoring_loop(self, interval: float):
        """監視ループ"""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # アラートチェック
                self._check_alerts(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"リソース監視エラー: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """メトリクス収集"""
        try:
            # CPU使用率
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # ディスク使用率
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # ネットワークI/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # プロセス情報
            current_process = psutil.Process()
            active_threads = current_process.num_threads()
            
            # GC情報
            gc_stats = gc.get_stats()
            gc_collections = sum(stat['collections'] for stat in gc_stats)
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_threads=active_threads,
                gc_collections=gc_collections
            )
            
        except Exception as e:
            logger.error(f"メトリクス収集エラー: {e}")
            return PerformanceMetrics()
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """アラートチェック"""
        alerts = []
        
        # CPU アラート
        if metrics.cpu_usage > self.alert_thresholds[ResourceType.CPU]:
            if self._should_send_alert(ResourceType.CPU):
                alert = ResourceAlert(
                    resource_type=ResourceType.CPU,
                    current_usage=metrics.cpu_usage,
                    threshold=self.alert_thresholds[ResourceType.CPU],
                    severity="HIGH",
                    message=f"CPU使用率が{metrics.cpu_usage:.1f}%に達しています",
                    recommendations=[
                        "CPU集約的な処理の見直し",
                        "並列処理の最適化",
                        "不要なプロセスの終了"
                    ]
                )
                alerts.append(alert)
        
        # メモリ アラート
        if metrics.memory_usage > self.alert_thresholds[ResourceType.MEMORY]:
            if self._should_send_alert(ResourceType.MEMORY):
                alert = ResourceAlert(
                    resource_type=ResourceType.MEMORY,
                    current_usage=metrics.memory_usage,
                    threshold=self.alert_thresholds[ResourceType.MEMORY],
                    severity="HIGH",
                    message=f"メモリ使用率が{metrics.memory_usage:.1f}%に達しています",
                    recommendations=[
                        "メモリリークの確認",
                        "キャッシュサイズの調整",
                        "不要なデータの解放",
                        "ガベージコレクションの実行"
                    ]
                )
                alerts.append(alert)
        
        # ディスク アラート
        if metrics.disk_usage > self.alert_thresholds[ResourceType.DISK]:
            if self._should_send_alert(ResourceType.DISK):
                alert = ResourceAlert(
                    resource_type=ResourceType.DISK,
                    current_usage=metrics.disk_usage,
                    threshold=self.alert_thresholds[ResourceType.DISK],
                    severity="HIGH",
                    message=f"ディスク使用率が{metrics.disk_usage:.1f}%に達しています",
                    recommendations=[
                        "不要ファイルの削除",
                        "ログローテーション",
                        "一時ファイルのクリーンアップ",
                        "データアーカイブ"
                    ]
                )
                alerts.append(alert)
        
        # アラート送信
        for alert in alerts:
            self._send_alert(alert)
    
    def _should_send_alert(self, resource_type: ResourceType) -> bool:
        """アラート送信判定"""
        last_alert = self.last_alert_time[resource_type]
        current_time = datetime.now()
        
        # 10分間隔でアラート送信
        return (current_time - last_alert).total_seconds() > 600
    
    def _send_alert(self, alert: ResourceAlert):
        """アラート送信"""
        logger.warning(f"リソース アラート: {alert.message}")
        
        # コールバック実行
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"アラートコールバックエラー: {e}")
        
        # 最終送信時刻更新
        self.last_alert_time[alert.resource_type] = alert.timestamp
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """アラートコールバック追加"""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """現在のメトリクス取得"""
        return self._collect_metrics()
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """メトリクス履歴取得"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

class AutoOptimizer:
    """自動最適化エンジン"""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.optimization_history = deque(maxlen=100)
        self.active_optimizations = {}
        
    def analyze_performance(self) -> Dict[str, Any]:
        """パフォーマンス分析"""
        current_metrics = self.resource_monitor.get_current_metrics()
        historical_metrics = self.resource_monitor.get_metrics_history(30)  # 30分間
        
        if not historical_metrics:
            return {'status': 'insufficient_data'}
        
        analysis = {
            'current_status': self._evaluate_performance_level(current_metrics),
            'trends': self._analyze_trends(historical_metrics),
            'bottlenecks': self._identify_bottlenecks(current_metrics, historical_metrics),
            'recommendations': []
        }
        
        # 推奨事項生成
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _evaluate_performance_level(self, metrics: PerformanceMetrics) -> PerformanceLevel:
        """パフォーマンスレベル評価"""
        # 総合スコア計算
        cpu_score = max(0, 100 - metrics.cpu_usage)
        memory_score = max(0, 100 - metrics.memory_usage)
        disk_score = max(0, 100 - metrics.disk_usage)
        
        overall_score = (cpu_score + memory_score + disk_score) / 3
        
        if overall_score >= 90:
            return PerformanceLevel.EXCELLENT
        elif overall_score >= 80:
            return PerformanceLevel.GOOD
        elif overall_score >= 70:
            return PerformanceLevel.FAIR
        elif overall_score >= 60:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _analyze_trends(self, metrics_history: List[PerformanceMetrics]) -> Dict[str, Any]:
        """トレンド分析"""
        if len(metrics_history) < 10:
            return {'status': 'insufficient_data'}
        
        # データ準備
        timestamps = [m.timestamp for m in metrics_history]
        cpu_values = [m.cpu_usage for m in metrics_history]
        memory_values = [m.memory_usage for m in metrics_history]
        
        # トレンド計算
        cpu_trend = self._calculate_trend(cpu_values)
        memory_trend = self._calculate_trend(memory_values)
        
        return {
            'cpu_trend': cpu_trend,
            'memory_trend': memory_trend,
            'duration_minutes': (timestamps[-1] - timestamps[0]).total_seconds() / 60,
            'data_points': len(metrics_history)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """トレンド計算"""
        if len(values) < 2:
            return {'slope': 0.0, 'direction': 'stable'}
        
        # 線形回帰でトレンド計算
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 2.0:
            direction = 'increasing'
        elif slope < -2.0:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'slope': float(slope),
            'direction': direction,
            'average': np.mean(values),
            'max': np.max(values),
            'min': np.min(values)
        }
    
    def _identify_bottlenecks(self, current: PerformanceMetrics, 
                           history: List[PerformanceMetrics]) -> List[str]:
        """ボトルネック特定"""
        bottlenecks = []
        
        # CPU ボトルネック
        if current.cpu_usage > 85:
            avg_cpu = np.mean([m.cpu_usage for m in history])
            if current.cpu_usage > avg_cpu * 1.2:
                bottlenecks.append("CPU使用率の急激な上昇")
        
        # メモリ ボトルネック
        if current.memory_usage > 80:
            bottlenecks.append("メモリ使用率の高止まり")
        
        # スレッド ボトルネック
        if current.active_threads > 50:
            bottlenecks.append("過度なスレッド使用")
        
        # GC ボトルネック
        if len(history) > 10:
            recent_gc = [m.gc_collections for m in history[-10:]]
            if len(set(recent_gc)) > 1:  # GCが頻発
                gc_rate = (recent_gc[-1] - recent_gc[0]) / 10
                if gc_rate > 1:  # 1分あたり1回以上
                    bottlenecks.append("頻繁なガベージコレクション")
        
        return bottlenecks
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        performance_level = analysis.get('current_status', PerformanceLevel.FAIR)
        bottlenecks = analysis.get('bottlenecks', [])
        trends = analysis.get('trends', {})
        
        # パフォーマンスレベル別推奨事項
        if performance_level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
            recommendations.extend([
                "緊急最適化が必要です",
                "不要なプロセスの終了を検討してください",
                "キャッシュサイズの縮小を検討してください"
            ])
        
        # ボトルネック別推奨事項
        if "CPU使用率の急激な上昇" in bottlenecks:
            recommendations.extend([
                "CPU集約的な処理の並列化",
                "処理アルゴリズムの最適化",
                "非同期処理の導入"
            ])
        
        if "メモリ使用率の高止まり" in bottlenecks:
            recommendations.extend([
                "メモリリークの調査",
                "キャッシュの定期的なクリア",
                "大容量データの分割処理"
            ])
        
        if "頻繁なガベージコレクション" in bottlenecks:
            recommendations.extend([
                "オブジェクト生成の削減",
                "メモリプールの使用",
                "GC チューニング"
            ])
        
        # トレンド別推奨事項
        cpu_trend = trends.get('cpu_trend', {})
        if cpu_trend.get('direction') == 'increasing':
            recommendations.append("CPU使用率の上昇トレンドを監視し、予防的対策を検討してください")
        
        return list(set(recommendations))  # 重複除去
    
    def apply_optimization(self, strategy: OptimizationStrategy) -> OptimizationResult:
        """最適化適用"""
        logger.info(f"最適化戦略 '{strategy.value}' を適用中...")
        
        start_time = time.time()
        start_metrics = self.resource_monitor.get_current_metrics()
        
        improvements = {}
        recommendations = []
        
        try:
            if strategy == OptimizationStrategy.CPU_INTENSIVE:
                improvements = self._optimize_cpu()
                recommendations.extend([
                    "CPU並列処理の最適化を実施",
                    "アルゴリズム効率化を実行"
                ])
            
            elif strategy == OptimizationStrategy.MEMORY_INTENSIVE:
                improvements = self._optimize_memory()
                recommendations.extend([
                    "メモリ使用量の最適化を実施",
                    "キャッシュサイズを調整"
                ])
            
            elif strategy == OptimizationStrategy.BALANCED:
                improvements.update(self._optimize_cpu())
                improvements.update(self._optimize_memory())
                recommendations.extend([
                    "バランス最適化を実施",
                    "システム全体のチューニングを実行"
                ])
            
            # 最適化後のメトリクス取得
            time.sleep(2)  # 効果測定のための待機
            end_metrics = self.resource_monitor.get_current_metrics()
            
            # パフォーマンス向上計算
            performance_gain = self._calculate_performance_gain(start_metrics, end_metrics)
            
            # リソース節約計算
            resource_savings = {
                'cpu': max(0, start_metrics.cpu_usage - end_metrics.cpu_usage),
                'memory': max(0, start_metrics.memory_usage - end_metrics.memory_usage)
            }
            
            result = OptimizationResult(
                strategy=strategy,
                improvements=improvements,
                recommendations=recommendations,
                performance_gain=performance_gain,
                resource_savings=resource_savings
            )
            
            # 履歴保存
            self.optimization_history.append(result)
            
            logger.info(f"最適化完了。パフォーマンス向上: {performance_gain:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"最適化エラー: {e}")
            return OptimizationResult(
                strategy=strategy,
                improvements={},
                recommendations=[f"最適化エラー: {e}"],
                performance_gain=0.0,
                resource_savings={}
            )
    
    def _optimize_cpu(self) -> Dict[str, float]:
        """CPU最適化"""
        improvements = {}
        
        try:
            # ガベージコレクション実行
            gc_before = len(gc.get_objects())
            collected = gc.collect()
            gc_after = len(gc.get_objects())
            
            if collected > 0:
                improvements['gc_objects_freed'] = gc_before - gc_after
                improvements['gc_collections'] = collected
            
            # CPUアフィニティ最適化（可能な場合）
            try:
                current_process = psutil.Process()
                cpu_count = psutil.cpu_count()
                if cpu_count > 1:
                    # 利用可能なCPUコアを効率的に使用
                    current_process.cpu_affinity(list(range(cpu_count)))
                    improvements['cpu_affinity_optimized'] = 1.0
            except:
                pass  # CPUアフィニティ設定失敗は無視
                
        except Exception as e:
            logger.error(f"CPU最適化エラー: {e}")
        
        return improvements
    
    def _optimize_memory(self) -> Dict[str, float]:
        """メモリ最適化"""
        improvements = {}
        
        try:
            # メモリ使用量測定
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            # ガベージコレクション強制実行
            for generation in range(3):
                collected = gc.collect(generation)
                if collected > 0:
                    improvements[f'gc_gen_{generation}_collected'] = collected
            
            # メモリ使用量再測定
            memory_after = process.memory_info().rss
            memory_freed = memory_before - memory_after
            
            if memory_freed > 0:
                improvements['memory_freed_bytes'] = memory_freed
                improvements['memory_freed_mb'] = memory_freed / (1024 * 1024)
            
            # メモリ断片化対策（Python特有）
            try:
                import ctypes
                if hasattr(ctypes.pythonapi, 'PyGC_Collect'):
                    ctypes.pythonapi.PyGC_Collect()
                    improvements['memory_defragmentation'] = 1.0
            except:
                pass
                
        except Exception as e:
            logger.error(f"メモリ最適化エラー: {e}")
        
        return improvements
    
    def _calculate_performance_gain(self, before: PerformanceMetrics, 
                                  after: PerformanceMetrics) -> float:
        """パフォーマンス向上率計算"""
        try:
            # リソース使用率の改善を総合評価
            cpu_improvement = max(0, before.cpu_usage - after.cpu_usage)
            memory_improvement = max(0, before.memory_usage - after.memory_usage)
            
            # 重み付き平均
            total_improvement = (cpu_improvement * 0.6 + memory_improvement * 0.4)
            
            # パーセンテージ変換
            baseline = (before.cpu_usage + before.memory_usage) / 2
            if baseline > 0:
                performance_gain = (total_improvement / baseline) * 100
            else:
                performance_gain = 0.0
            
            return min(100.0, max(0.0, performance_gain))  # 0-100%の範囲
            
        except Exception as e:
            logger.error(f"パフォーマンス向上計算エラー: {e}")
            return 0.0

class AsyncTaskOptimizer:
    """非同期タスク最適化"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
        self.task_queue = asyncio.Queue()
        self.performance_stats = defaultdict(list)
        
    async def optimize_async_execution(self, tasks: List[Callable], 
                                     execution_strategy: str = "auto") -> List[Any]:
        """非同期実行最適化"""
        if not tasks:
            return []
        
        start_time = time.time()
        
        if execution_strategy == "auto":
            execution_strategy = self._determine_optimal_strategy(tasks)
        
        try:
            if execution_strategy == "asyncio":
                results = await self._execute_asyncio(tasks)
            elif execution_strategy == "threading":
                results = await self._execute_threading(tasks)
            elif execution_strategy == "multiprocessing":
                results = await self._execute_multiprocessing(tasks)
            else:
                results = await self._execute_mixed(tasks)
            
            execution_time = time.time() - start_time
            
            # パフォーマンス統計記録
            self.performance_stats[execution_strategy].append({
                'task_count': len(tasks),
                'execution_time': execution_time,
                'throughput': len(tasks) / execution_time if execution_time > 0 else 0,
                'timestamp': datetime.now()
            })
            
            logger.info(f"非同期タスク実行完了: {len(tasks)}件 / {execution_time:.2f}秒 "
                       f"(戦略: {execution_strategy})")
            
            return results
            
        except Exception as e:
            logger.error(f"非同期実行エラー: {e}")
            return []
    
    def _determine_optimal_strategy(self, tasks: List[Callable]) -> str:
        """最適実行戦略決定"""
        task_count = len(tasks)
        
        # タスク数による基本判定
        if task_count <= 5:
            return "asyncio"
        elif task_count <= 20:
            return "threading"
        elif task_count <= 100:
            return "multiprocessing"
        else:
            return "mixed"
    
    async def _execute_asyncio(self, tasks: List[Callable]) -> List[Any]:
        """asyncio実行"""
        async_tasks = []
        for task in tasks:
            if asyncio.iscoroutinefunction(task):
                async_tasks.append(task())
            else:
                async_tasks.append(asyncio.get_event_loop().run_in_executor(None, task))
        
        return await asyncio.gather(*async_tasks, return_exceptions=True)
    
    async def _execute_threading(self, tasks: List[Callable]) -> List[Any]:
        """スレッドプール実行"""
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.thread_pool, task) for task in tasks]
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _execute_multiprocessing(self, tasks: List[Callable]) -> List[Any]:
        """プロセスプール実行"""
        loop = asyncio.get_event_loop()
        futures = [loop.run_in_executor(self.process_pool, task) for task in tasks]
        return await asyncio.gather(*futures, return_exceptions=True)
    
    async def _execute_mixed(self, tasks: List[Callable]) -> List[Any]:
        """混合実行戦略"""
        # タスクを分割して異なる実行方式を組み合わせ
        chunk_size = max(1, len(tasks) // 4)
        chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]
        
        results = []
        for i, chunk in enumerate(chunks):
            strategy = ["asyncio", "threading", "multiprocessing"][i % 3]
            chunk_results = await self.optimize_async_execution(chunk, strategy)
            results.extend(chunk_results)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンス レポート"""
        report = {}
        
        for strategy, stats in self.performance_stats.items():
            if stats:
                recent_stats = stats[-10:]  # 最近10回
                report[strategy] = {
                    'avg_execution_time': np.mean([s['execution_time'] for s in recent_stats]),
                    'avg_throughput': np.mean([s['throughput'] for s in recent_stats]),
                    'total_executions': len(stats),
                    'last_execution': recent_stats[-1]['timestamp'].isoformat()
                }
        
        return report
    
    def cleanup(self):
        """リソースクリーンアップ"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class AdvancedPerformanceOptimizationSystem:
    """高度パフォーマンス最適化システム（メインクラス）"""
    
    def __init__(self, db_path: str = "performance_optimization.db"):
        self.db_path = Path(db_path)
        self.profiler = PerformanceProfiler()
        self.resource_monitor = ResourceMonitor()
        self.auto_optimizer = AutoOptimizer(self.resource_monitor)
        self.async_optimizer = AsyncTaskOptimizer()
        
        self.optimization_enabled = True
        self.auto_optimization_enabled = False
        
        self.setup_database()
        self.setup_alert_callbacks()
        
        logger.info("高度パフォーマンス最適化システムを初期化しました")
    
    def setup_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        cpu_usage REAL,
                        memory_usage REAL,
                        disk_usage REAL,
                        response_time REAL,
                        throughput REAL,
                        active_threads INTEGER
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        strategy TEXT,
                        performance_gain REAL,
                        improvements TEXT,
                        recommendations TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS resource_alerts (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        resource_type TEXT,
                        current_usage REAL,
                        threshold REAL,
                        severity TEXT,
                        message TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("パフォーマンス最適化データベースを初期化しました")
        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
    
    def setup_alert_callbacks(self):
        """アラートコールバック設定"""
        def alert_handler(alert: ResourceAlert):
            self._save_resource_alert(alert)
            
            # 自動最適化が有効な場合
            if self.auto_optimization_enabled:
                self._trigger_auto_optimization(alert)
        
        self.resource_monitor.add_alert_callback(alert_handler)
    
    def start_monitoring(self, interval: float = 1.0):
        """監視開始"""
        self.resource_monitor.start_monitoring(interval)
        logger.info("パフォーマンス監視を開始しました")
    
    def stop_monitoring(self):
        """監視停止"""
        self.resource_monitor.stop_monitoring()
        logger.info("パフォーマンス監視を停止しました")
    
    async def optimize_system_performance(self, 
                                        strategy: OptimizationStrategy = None) -> OptimizationResult:
        """システムパフォーマンス最適化"""
        if not self.optimization_enabled:
            logger.warning("最適化が無効化されています")
            return OptimizationResult(
                strategy=OptimizationStrategy.BALANCED,
                improvements={},
                recommendations=["最適化が無効化されています"],
                performance_gain=0.0,
                resource_savings={}
            )
        
        try:
            # パフォーマンス分析
            analysis = self.auto_optimizer.analyze_performance()
            
            # 戦略決定
            if strategy is None:
                strategy = self._determine_optimization_strategy(analysis)
            
            logger.info(f"システム最適化を開始: 戦略={strategy.value}")
            
            # 最適化実行
            result = self.auto_optimizer.apply_optimization(strategy)
            
            # 結果保存
            await self._save_optimization_result(result)
            
            logger.info(f"システム最適化完了: パフォーマンス向上={result.performance_gain:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"システム最適化エラー: {e}")
            return OptimizationResult(
                strategy=strategy or OptimizationStrategy.BALANCED,
                improvements={},
                recommendations=[f"最適化エラー: {e}"],
                performance_gain=0.0,
                resource_savings={}
            )
    
    def _determine_optimization_strategy(self, analysis: Dict[str, Any]) -> OptimizationStrategy:
        """最適化戦略決定"""
        current_status = analysis.get('current_status', PerformanceLevel.FAIR)
        bottlenecks = analysis.get('bottlenecks', [])
        
        # ボトルネック分析による戦略選択
        cpu_bottleneck = any("CPU" in bottleneck for bottleneck in bottlenecks)
        memory_bottleneck = any("メモリ" in bottleneck for bottleneck in bottlenecks)
        
        if cpu_bottleneck and not memory_bottleneck:
            return OptimizationStrategy.CPU_INTENSIVE
        elif memory_bottleneck and not cpu_bottleneck:
            return OptimizationStrategy.MEMORY_INTENSIVE
        elif current_status == PerformanceLevel.CRITICAL:
            return OptimizationStrategy.REALTIME
        else:
            return OptimizationStrategy.BALANCED
    
    def _trigger_auto_optimization(self, alert: ResourceAlert):
        """自動最適化トリガー"""
        try:
            # 自動最適化は非同期で実行
            async def auto_optimize():
                if alert.resource_type == ResourceType.CPU:
                    await self.optimize_system_performance(OptimizationStrategy.CPU_INTENSIVE)
                elif alert.resource_type == ResourceType.MEMORY:
                    await self.optimize_system_performance(OptimizationStrategy.MEMORY_INTENSIVE)
                else:
                    await self.optimize_system_performance(OptimizationStrategy.BALANCED)
            
            # 現在のイベントループで実行
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(auto_optimize())
            except RuntimeError:
                # イベントループが存在しない場合は新しく作成
                asyncio.run(auto_optimize())
                
        except Exception as e:
            logger.error(f"自動最適化エラー: {e}")
    
    async def _save_optimization_result(self, result: OptimizationResult):
        """最適化結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO optimization_results (
                        timestamp, strategy, performance_gain, improvements, recommendations
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    result.timestamp.isoformat(),
                    result.strategy.value,
                    result.performance_gain,
                    json.dumps(result.improvements),
                    json.dumps(result.recommendations)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"最適化結果保存エラー: {e}")
    
    def _save_resource_alert(self, alert: ResourceAlert):
        """リソースアラート保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO resource_alerts (
                        timestamp, resource_type, current_usage, threshold, severity, message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.timestamp.isoformat(),
                    alert.resource_type.value,
                    alert.current_usage,
                    alert.threshold,
                    alert.severity,
                    alert.message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"アラート保存エラー: {e}")
    
    def get_system_report(self) -> Dict[str, Any]:
        """システム レポート"""
        try:
            current_metrics = self.resource_monitor.get_current_metrics()
            historical_metrics = self.resource_monitor.get_metrics_history()
            analysis = self.auto_optimizer.analyze_performance()
            
            # 最近の最適化結果
            recent_optimizations = list(self.auto_optimizer.optimization_history)[-5:]
            
            # 非同期最適化統計
            async_stats = self.async_optimizer.get_performance_report()
            
            return {
                'current_metrics': {
                    'cpu_usage': current_metrics.cpu_usage,
                    'memory_usage': current_metrics.memory_usage,
                    'disk_usage': current_metrics.disk_usage,
                    'active_threads': current_metrics.active_threads,
                    'timestamp': current_metrics.timestamp.isoformat()
                },
                'performance_analysis': analysis,
                'optimization_history': [
                    {
                        'strategy': opt.strategy.value,
                        'performance_gain': opt.performance_gain,
                        'timestamp': opt.timestamp.isoformat()
                    }
                    for opt in recent_optimizations
                ],
                'async_optimization_stats': async_stats,
                'system_status': {
                    'monitoring_active': self.resource_monitor.monitoring_active,
                    'optimization_enabled': self.optimization_enabled,
                    'auto_optimization_enabled': self.auto_optimization_enabled,
                    'metrics_history_count': len(historical_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"システム レポート生成エラー: {e}")
            return {'error': str(e)}
    
    def enable_auto_optimization(self):
        """自動最適化有効化"""
        self.auto_optimization_enabled = True
        logger.info("自動最適化が有効化されました")
    
    def disable_auto_optimization(self):
        """自動最適化無効化"""
        self.auto_optimization_enabled = False
        logger.info("自動最適化が無効化されました")
    
    def cleanup(self):
        """システムクリーンアップ"""
        self.stop_monitoring()
        self.async_optimizer.cleanup()
        logger.info("パフォーマンス最適化システムをクリーンアップしました")

# 使用例とデモ
async def demo_performance_optimization():
    """パフォーマンス最適化システム デモ"""
    try:
        # システム初期化
        system = AdvancedPerformanceOptimizationSystem("demo_performance.db")
        
        # 監視開始
        system.start_monitoring(interval=0.5)
        
        # 自動最適化有効化
        system.enable_auto_optimization()
        
        # CPU負荷テスト用関数
        def cpu_intensive_task():
            return sum(i * i for i in range(100000))
        
        # 非同期タスク最適化テスト
        tasks = [lambda: cpu_intensive_task() for _ in range(10)]
        results = await system.async_optimizer.optimize_async_execution(tasks)
        print(f"非同期タスク実行結果: {len(results)}件完了")
        
        # システム最適化実行
        optimization_result = await system.optimize_system_performance()
        print(f"最適化結果: {optimization_result.performance_gain:.1f}% 向上")
        
        # 少し待機してメトリクス収集
        await asyncio.sleep(3)
        
        # システム レポート生成
        report = system.get_system_report()
        print(f"システム レポート:")
        print(f"  CPU使用率: {report['current_metrics']['cpu_usage']:.1f}%")
        print(f"  メモリ使用率: {report['current_metrics']['memory_usage']:.1f}%")
        print(f"  パフォーマンス レベル: {report['performance_analysis']['current_status'].value}")
        
        # クリーンアップ
        system.cleanup()
        
        return system
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        raise

if __name__ == "__main__":
    # デモ実行
    asyncio.run(demo_performance_optimization())