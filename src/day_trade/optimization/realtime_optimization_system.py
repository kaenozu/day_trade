"""
リアルタイム最適化システム

システムパフォーマンスをリアルタイムで監視・最適化する
"""

import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import threading
import queue
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    active_threads: int
    queue_size: int
    processing_latency: float
    throughput: float


@dataclass
class OptimizationRule:
    """最適化ルール"""
    name: str
    condition: Callable[[PerformanceMetrics], bool]
    action: Callable[[], Any]
    priority: int = 1
    cooldown_seconds: int = 30
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0


@dataclass
class OptimizationResult:
    """最適化結果"""
    rule_name: str
    executed_at: datetime
    before_metrics: PerformanceMetrics
    after_metrics: Optional[PerformanceMetrics] = None
    success: bool = False
    improvement_percent: float = 0.0
    error_message: Optional[str] = None


class AdaptiveThresholds:
    """適応的閾値システム"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.thresholds: Dict[str, float] = {}
        
    def update_history(self, metrics: PerformanceMetrics):
        """履歴更新"""
        self.metrics_history.append(metrics)
        self._recalculate_thresholds()
    
    def _recalculate_thresholds(self):
        """閾値再計算"""
        if len(self.metrics_history) < 10:
            return
        
        # 統計値計算
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        latency_values = [m.processing_latency for m in self.metrics_history]
        
        # 動的閾値設定（平均 + 1.5 * 標準偏差）
        self.thresholds.update({
            'cpu_high': np.mean(cpu_values) + 1.5 * np.std(cpu_values),
            'memory_high': np.mean(memory_values) + 1.5 * np.std(memory_values),
            'latency_high': np.mean(latency_values) + 1.5 * np.std(latency_values),
            'cpu_low': np.mean(cpu_values) - 0.5 * np.std(cpu_values),
            'memory_low': np.mean(memory_values) - 0.5 * np.std(memory_values)
        })
    
    def get_threshold(self, metric_name: str) -> float:
        """閾値取得"""
        return self.thresholds.get(metric_name, 0.0)


class ResourcePool:
    """リソースプール管理"""
    
    def __init__(self, max_threads: int = None, max_memory_mb: int = None):
        self.max_threads = max_threads or psutil.cpu_count() * 2
        self.max_memory_mb = max_memory_mb or psutil.virtual_memory().total // 1024 // 1024 // 2
        
        self.active_threads = 0
        self.allocated_memory_mb = 0
        self.thread_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        
        # プール統計
        self.stats = {
            'thread_allocations': 0,
            'thread_releases': 0,
            'memory_allocations': 0,
            'memory_releases': 0,
            'allocation_failures': 0
        }
    
    def allocate_thread(self) -> bool:
        """スレッド割り当て"""
        with self.thread_lock:
            if self.active_threads < self.max_threads:
                self.active_threads += 1
                self.stats['thread_allocations'] += 1
                return True
            self.stats['allocation_failures'] += 1
            return False
    
    def release_thread(self):
        """スレッド解放"""
        with self.thread_lock:
            if self.active_threads > 0:
                self.active_threads -= 1
                self.stats['thread_releases'] += 1
    
    def allocate_memory(self, size_mb: float) -> bool:
        """メモリ割り当て"""
        with self.memory_lock:
            if self.allocated_memory_mb + size_mb <= self.max_memory_mb:
                self.allocated_memory_mb += size_mb
                self.stats['memory_allocations'] += 1
                return True
            self.stats['allocation_failures'] += 1
            return False
    
    def release_memory(self, size_mb: float):
        """メモリ解放"""
        with self.memory_lock:
            self.allocated_memory_mb = max(0, self.allocated_memory_mb - size_mb)
            self.stats['memory_releases'] += 1
    
    def get_utilization(self) -> Dict[str, float]:
        """利用率取得"""
        return {
            'thread_utilization': self.active_threads / self.max_threads,
            'memory_utilization': self.allocated_memory_mb / self.max_memory_mb,
            'stats': self.stats.copy()
        }


class RealtimeOptimizationSystem:
    """リアルタイム最適化システム"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # コンポーネント
        self.adaptive_thresholds = AdaptiveThresholds()
        self.resource_pool = ResourcePool()
        
        # 監視・最適化
        self.monitoring_active = False
        self.optimization_active = False
        self.metrics_queue = queue.Queue()
        self.optimization_queue = queue.Queue()
        
        # ルールとイベント
        self.optimization_rules: List[OptimizationRule] = []
        self.optimization_history: List[OptimizationResult] = []
        
        # スレッド
        self.monitor_thread: Optional[threading.Thread] = None
        self.optimizer_thread: Optional[threading.Thread] = None
        
        # 統計
        self.system_stats = {
            'start_time': datetime.now(),
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'monitoring_cycles': 0
        }
        
        # デフォルトルール登録
        self._register_default_rules()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _register_default_rules(self):
        """デフォルト最適化ルール登録"""
        
        # CPU高負荷時の最適化
        self.register_rule(OptimizationRule(
            name="cpu_high_optimization",
            condition=lambda m: m.cpu_percent > 80,
            action=self._optimize_cpu_usage,
            priority=1,
            cooldown_seconds=60
        ))
        
        # メモリ高使用時の最適化
        self.register_rule(OptimizationRule(
            name="memory_high_optimization", 
            condition=lambda m: m.memory_percent > 85,
            action=self._optimize_memory_usage,
            priority=1,
            cooldown_seconds=45
        ))
        
        # レイテンシ最適化
        self.register_rule(OptimizationRule(
            name="latency_optimization",
            condition=lambda m: m.processing_latency > 100,  # 100ms以上
            action=self._optimize_processing_latency,
            priority=2,
            cooldown_seconds=30
        ))
        
        # スループット最適化
        self.register_rule(OptimizationRule(
            name="throughput_optimization", 
            condition=lambda m: m.throughput < 1000,  # 1000 records/sec未満
            action=self._optimize_throughput,
            priority=3,
            cooldown_seconds=30
        ))
        
        # リソースプール調整
        self.register_rule(OptimizationRule(
            name="resource_pool_adjustment",
            condition=lambda m: self.resource_pool.get_utilization()['thread_utilization'] > 0.9,
            action=self._adjust_resource_pool,
            priority=2,
            cooldown_seconds=120
        ))
    
    def register_rule(self, rule: OptimizationRule):
        """最適化ルール登録"""
        self.optimization_rules.append(rule)
        self.optimization_rules.sort(key=lambda r: r.priority)
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """メトリクス収集"""
        start_time = time.time()
        
        # システムメトリクス
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # プロセスメトリクス
        current_process = psutil.Process()
        active_threads = current_process.num_threads()
        
        # アプリケーション固有メトリクス
        queue_size = self.metrics_queue.qsize()
        processing_latency = (time.time() - start_time) * 1000  # ms
        
        # スループット計算（簡易）
        throughput = 10000 / max(processing_latency / 1000, 0.001)  # records/sec
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / 1024 / 1024,
            disk_io_read=disk_io.read_bytes if disk_io else 0,
            disk_io_write=disk_io.write_bytes if disk_io else 0,
            network_sent=network_io.bytes_sent if network_io else 0,
            network_recv=network_io.bytes_recv if network_io else 0,
            active_threads=active_threads,
            queue_size=queue_size,
            processing_latency=processing_latency,
            throughput=throughput
        )
        
        return metrics
    
    def _monitoring_loop(self):
        """監視ループ"""
        self.logger.info("リアルタイム監視開始")
        
        while self.monitoring_active:
            try:
                # メトリクス収集
                metrics = asyncio.run(self.collect_metrics())
                
                # 適応的閾値更新
                self.adaptive_thresholds.update_history(metrics)
                
                # メトリクスキューに追加
                if not self.metrics_queue.full():
                    self.metrics_queue.put(metrics)
                
                # 最適化トリガー検査
                self._check_optimization_triggers(metrics)
                
                self.system_stats['monitoring_cycles'] += 1
                
                time.sleep(1)  # 1秒間隔
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(5)
    
    def _check_optimization_triggers(self, metrics: PerformanceMetrics):
        """最適化トリガー検査"""
        for rule in self.optimization_rules:
            try:
                # クールダウンチェック
                if rule.last_executed:
                    elapsed = (datetime.now() - rule.last_executed).seconds
                    if elapsed < rule.cooldown_seconds:
                        continue
                
                # 条件チェック
                if rule.condition(metrics):
                    # 最適化キューに追加
                    optimization_request = {
                        'rule': rule,
                        'metrics': metrics,
                        'timestamp': datetime.now()
                    }
                    
                    if not self.optimization_queue.full():
                        self.optimization_queue.put(optimization_request)
                        self.logger.info(f"最適化トリガー: {rule.name}")
                    
            except Exception as e:
                self.logger.error(f"トリガー検査エラー ({rule.name}): {e}")
    
    def _optimization_loop(self):
        """最適化ループ"""
        self.logger.info("リアルタイム最適化開始")
        
        while self.optimization_active:
            try:
                # 最適化要求を取得（タイムアウト付き）
                try:
                    request = self.optimization_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                rule = request['rule']
                before_metrics = request['metrics']
                
                self.logger.info(f"最適化実行: {rule.name}")
                
                # 最適化実行
                result = self._execute_optimization(rule, before_metrics)
                
                # 結果記録
                self.optimization_history.append(result)
                self.system_stats['total_optimizations'] += 1
                
                if result.success:
                    self.system_stats['successful_optimizations'] += 1
                
                # クールダウン設定
                rule.last_executed = datetime.now()
                rule.execution_count += 1
                
                if result.success:
                    rule.success_count += 1
                
                time.sleep(0.1)  # 短い休憩
                
            except Exception as e:
                self.logger.error(f"最適化ループエラー: {e}")
    
    def _execute_optimization(self, rule: OptimizationRule, before_metrics: PerformanceMetrics) -> OptimizationResult:
        """最適化実行"""
        result = OptimizationResult(
            rule_name=rule.name,
            executed_at=datetime.now(),
            before_metrics=before_metrics
        )
        
        try:
            # 最適化アクション実行
            rule.action()
            
            # 効果測定（少し待ってから）
            time.sleep(2)
            after_metrics = asyncio.run(self.collect_metrics())
            result.after_metrics = after_metrics
            
            # 改善度計算
            improvement = self._calculate_improvement(before_metrics, after_metrics)
            result.improvement_percent = improvement
            result.success = improvement > 0
            
            self.logger.info(f"最適化完了: {rule.name} (改善: {improvement:.1f}%)")
            
        except Exception as e:
            result.error_message = str(e)
            result.success = False
            self.logger.error(f"最適化失敗: {rule.name} - {e}")
        
        return result
    
    def _calculate_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics) -> float:
        """改善度計算"""
        improvements = []
        
        # CPU使用率改善
        if before.cpu_percent > after.cpu_percent:
            improvements.append((before.cpu_percent - after.cpu_percent) / before.cpu_percent * 100)
        
        # メモリ使用率改善
        if before.memory_percent > after.memory_percent:
            improvements.append((before.memory_percent - after.memory_percent) / before.memory_percent * 100)
        
        # レイテンシ改善
        if before.processing_latency > after.processing_latency:
            improvements.append((before.processing_latency - after.processing_latency) / before.processing_latency * 100)
        
        # スループット向上
        if after.throughput > before.throughput:
            improvements.append((after.throughput - before.throughput) / before.throughput * 100)
        
        return np.mean(improvements) if improvements else 0.0
    
    # 最適化アクション
    def _optimize_cpu_usage(self):
        """CPU使用率最適化"""
        # CPU集約タスクの並列度調整
        if hasattr(self, 'parallel_workers'):
            self.parallel_workers = max(1, self.parallel_workers - 1)
        
        # ガベージコレクション実行
        import gc
        gc.collect()
        
        self.logger.info("CPU使用率最適化実行")
    
    def _optimize_memory_usage(self):
        """メモリ使用量最適化"""
        # ガベージコレクション
        import gc
        gc.collect()
        
        # キャッシュクリア（システムによる）
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear_old_entries()
        
        self.logger.info("メモリ使用量最適化実行")
    
    def _optimize_processing_latency(self):
        """処理レイテンシ最適化"""
        # バッチサイズ調整
        if hasattr(self, 'batch_size'):
            self.batch_size = min(self.batch_size * 2, 1000)
        
        self.logger.info("処理レイテンシ最適化実行")
    
    def _optimize_throughput(self):
        """スループット最適化"""
        # 並列処理スレッド数増加
        if self.resource_pool.allocate_thread():
            self.logger.info("スループット向上のためスレッド追加")
    
    def _adjust_resource_pool(self):
        """リソースプール調整"""
        utilization = self.resource_pool.get_utilization()
        
        # スレッド数調整
        if utilization['thread_utilization'] > 0.9:
            self.resource_pool.max_threads = min(
                self.resource_pool.max_threads + 2,
                psutil.cpu_count() * 4
            )
            self.logger.info("リソースプール拡張")
    
    async def start_monitoring(self):
        """監視開始"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.optimization_active = True
        
        # 監視スレッド開始
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # 最適化スレッド開始
        self.optimizer_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimizer_thread.start()
        
        self.logger.info("リアルタイム最適化システム開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        self.optimization_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.optimizer_thread:
            self.optimizer_thread.join(timeout=5)
        
        self.logger.info("リアルタイム最適化システム停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        uptime = (datetime.now() - self.system_stats['start_time']).seconds
        
        # 最新メトリクス
        latest_metrics = None
        try:
            latest_metrics = asyncio.run(self.collect_metrics())
        except:
            pass
        
        # ルール統計
        rule_stats = {}
        for rule in self.optimization_rules:
            success_rate = rule.success_count / rule.execution_count if rule.execution_count > 0 else 0
            rule_stats[rule.name] = {
                'execution_count': rule.execution_count,
                'success_count': rule.success_count,
                'success_rate': success_rate,
                'last_executed': rule.last_executed.isoformat() if rule.last_executed else None
            }
        
        return {
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'optimization_active': self.optimization_active,
            'system_stats': self.system_stats,
            'latest_metrics': latest_metrics.__dict__ if latest_metrics else None,
            'rule_stats': rule_stats,
            'resource_pool_utilization': self.resource_pool.get_utilization(),
            'optimization_history_count': len(self.optimization_history)
        }
    
    async def run_optimization_demo(self, duration_seconds: int = 30):
        """最適化デモ実行"""
        print("=== リアルタイム最適化システム デモ ===")
        
        # 監視開始
        await self.start_monitoring()
        
        try:
            print(f"\n{duration_seconds}秒間の監視・最適化を実行...")
            
            # 負荷シミュレーション
            for i in range(duration_seconds):
                # CPU負荷生成
                if i % 10 == 0:
                    _ = sum([j ** 2 for j in range(10000)])
                
                # 進捗表示
                if i % 5 == 0:
                    status = self.get_system_status()
                    print(f"経過時間: {i}s, 最適化実行数: {status['system_stats']['total_optimizations']}")
                
                await asyncio.sleep(1)
            
            # 最終結果
            final_status = self.get_system_status()
            print(f"\n=== デモ結果 ===")
            print(f"総監視サイクル: {final_status['system_stats']['monitoring_cycles']}")
            print(f"最適化実行数: {final_status['system_stats']['total_optimizations']}")
            print(f"最適化成功数: {final_status['system_stats']['successful_optimizations']}")
            print(f"成功率: {final_status['system_stats']['successful_optimizations'] / max(1, final_status['system_stats']['total_optimizations']) * 100:.1f}%")
            
            if final_status['latest_metrics']:
                metrics = final_status['latest_metrics']
                print(f"\n=== 最新メトリクス ===")
                print(f"CPU使用率: {metrics['cpu_percent']:.1f}%")
                print(f"メモリ使用率: {metrics['memory_percent']:.1f}%")
                print(f"処理レイテンシ: {metrics['processing_latency']:.1f}ms")
                print(f"スループット: {metrics['throughput']:.1f} records/sec")
            
        finally:
            self.stop_monitoring()


async def demo_realtime_optimization():
    """リアルタイム最適化デモ"""
    optimization_system = RealtimeOptimizationSystem()
    await optimization_system.run_optimization_demo(20)


if __name__ == "__main__":
    asyncio.run(demo_realtime_optimization())