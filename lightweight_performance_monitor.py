#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Performance Monitor - 軽量パフォーマンス監視
Issue #945対応: 高速・低負荷パフォーマンス監視システム
"""

import psutil
import gc
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class LightweightPerformanceMonitor:
    """軽量パフォーマンス監視システム"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_cache: Dict[str, Any] = {}
        self.last_update = 0
        self.update_interval = 5.0  # 5秒間隔
        
        # 最適化カウンター
        self.optimizations_performed = 0
        self.memory_cleanups = 0
        
        # しきい値
        self.memory_threshold_mb = 200  # 200MB
        self.cpu_threshold_percent = 80  # 80%
        
    def get_current_metrics(self, force_update: bool = False) -> Dict[str, Any]:
        """現在のメトリクス取得"""
        current_time = time.time()
        
        # キャッシュチェック
        if not force_update and (current_time - self.last_update) < self.update_interval:
            return self.metrics_cache
        
        try:
            process = psutil.Process()
            
            # 基本メトリクス
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # GC統計
            gc_count = sum(gc.get_count())
            
            # スレッド数
            thread_count = threading.active_count()
            
            # システム稼働時間
            uptime = current_time - self.start_time
            
            self.metrics_cache = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': round(cpu_percent, 1),
                'memory_mb': round(memory_mb, 1),
                'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 1),
                'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 1),
                'gc_objects': gc_count,
                'thread_count': thread_count,
                'uptime_seconds': round(uptime, 1),
                'optimizations_performed': self.optimizations_performed,
                'memory_cleanups': self.memory_cleanups
            }
            
            self.last_update = current_time
            
            # 自動最適化チェック
            self._check_auto_optimization()
            
            return self.metrics_cache
            
        except Exception as e:
            logging.error(f"Failed to collect metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_auto_optimization(self):
        """自動最適化チェック"""
        if not self.metrics_cache:
            return
        
        # メモリ使用量チェック
        if self.metrics_cache['memory_mb'] > self.memory_threshold_mb:
            self.perform_memory_cleanup()
        
        # CPU使用率チェック
        if self.metrics_cache['cpu_percent'] > self.cpu_threshold_percent:
            logging.warning(f"High CPU usage detected: {self.metrics_cache['cpu_percent']}%")
    
    def perform_memory_cleanup(self):
        """メモリクリーンアップ実行"""
        try:
            before_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # ガベージコレクション
            collected = gc.collect()
            
            after_memory = psutil.Process().memory_info().rss / 1024 / 1024
            freed_mb = before_memory - after_memory
            
            self.memory_cleanups += 1
            self.optimizations_performed += 1
            
            logging.info(f"Memory cleanup: collected {collected} objects, freed {freed_mb:.1f}MB")
            
        except Exception as e:
            logging.error(f"Memory cleanup failed: {e}")
    
    def get_health_score(self) -> int:
        """システム健全性スコア（0-100）"""
        metrics = self.get_current_metrics()
        
        if 'error' in metrics:
            return 0
        
        # CPU スコア（CPU使用率が低いほど高スコア）
        cpu_score = max(0, 100 - metrics['cpu_percent'])
        
        # メモリ スコア（使用量に基づく）
        memory_usage_ratio = metrics['memory_mb'] / self.memory_threshold_mb
        memory_score = max(0, 100 - (memory_usage_ratio * 50))
        
        # 総合スコア（重み付き平均）
        total_score = int((cpu_score * 0.6 + memory_score * 0.4))
        
        return max(0, min(100, total_score))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンスサマリー取得"""
        metrics = self.get_current_metrics()
        health_score = self.get_health_score()
        
        # ステータス判定
        if health_score >= 80:
            status = "EXCELLENT"
        elif health_score >= 60:
            status = "GOOD"
        elif health_score >= 40:
            status = "FAIR"
        else:
            status = "NEEDS_ATTENTION"
        
        # 推奨アクション
        recommendations = []
        if metrics.get('memory_mb', 0) > self.memory_threshold_mb * 0.8:
            recommendations.append("Consider memory optimization")
        if metrics.get('cpu_percent', 0) > 70:
            recommendations.append("High CPU usage detected")
        if metrics.get('thread_count', 0) > 20:
            recommendations.append("High thread count")
        
        return {
            'health_score': health_score,
            'status': status,
            'current_metrics': metrics,
            'recommendations': recommendations,
            'thresholds': {
                'memory_threshold_mb': self.memory_threshold_mb,
                'cpu_threshold_percent': self.cpu_threshold_percent
            }
        }
    
    def optimize_system(self) -> Dict[str, Any]:
        """システム最適化実行"""
        before_metrics = self.get_current_metrics()
        
        # メモリクリーンアップ
        self.perform_memory_cleanup()
        
        # メトリクス強制更新
        after_metrics = self.get_current_metrics(force_update=True)
        
        # 改善量計算
        memory_improvement = before_metrics.get('memory_mb', 0) - after_metrics.get('memory_mb', 0)
        
        return {
            'optimization_performed': True,
            'before_memory_mb': before_metrics.get('memory_mb', 0),
            'after_memory_mb': after_metrics.get('memory_mb', 0),
            'memory_freed_mb': round(memory_improvement, 1),
            'optimizations_total': self.optimizations_performed,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_counters(self):
        """カウンターリセット"""
        self.optimizations_performed = 0
        self.memory_cleanups = 0
        self.start_time = time.time()
        logging.info("Performance monitor counters reset")


# グローバルインスタンス
lightweight_monitor = LightweightPerformanceMonitor()


def get_quick_status() -> Dict[str, Any]:
    """クイックステータス取得"""
    return lightweight_monitor.get_performance_summary()


def optimize_now() -> Dict[str, Any]:
    """即座に最適化実行"""
    return lightweight_monitor.optimize_system()


def main():
    """メインテスト"""
    print("=== Lightweight Performance Monitor ===")
    
    # 現在のメトリクス
    metrics = lightweight_monitor.get_current_metrics()
    print(f"CPU: {metrics.get('cpu_percent', 'N/A')}%")
    print(f"Memory: {metrics.get('memory_mb', 'N/A')}MB") 
    print(f"Threads: {metrics.get('thread_count', 'N/A')}")
    print(f"GC Objects: {metrics.get('gc_objects', 'N/A')}")
    
    # 健全性スコア
    health_score = lightweight_monitor.get_health_score()
    print(f"Health Score: {health_score}/100")
    
    # パフォーマンスサマリー
    summary = lightweight_monitor.get_performance_summary()
    print(f"Status: {summary['status']}")
    
    if summary['recommendations']:
        print("Recommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    # 最適化テスト
    print("\nTesting optimization...")
    opt_result = lightweight_monitor.optimize_system()
    print(f"Memory freed: {opt_result['memory_freed_mb']}MB")
    
    print("\nMonitoring complete!")


if __name__ == "__main__":
    main()