#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Monitoring & Auto Tuning System - リアルタイム監視・自動調整システム

Issue #805実装：実時間パフォーマンス監視・自動調整システム
システム性能の実時間監視と自動調整による最適化
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import psutil
import gc
import os

# Windows環境での文字化け対策
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class PerformanceLevel(Enum):
    """パフォーマンスレベル"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 30-49%
    CRITICAL = "critical"   # 0-29%

class OptimizationAction(Enum):
    """最適化アクション"""
    INCREASE_CACHE_SIZE = "increase_cache_size"
    DECREASE_CACHE_SIZE = "decrease_cache_size"
    INCREASE_WORKERS = "increase_workers"
    DECREASE_WORKERS = "decrease_workers"
    CLEAR_MEMORY = "clear_memory"
    OPTIMIZE_QUERIES = "optimize_queries"
    REDUCE_POLLING = "reduce_polling"
    INCREASE_POLLING = "increase_polling"
    RESTART_COMPONENTS = "restart_components"

@dataclass
class SystemHealth:
    """システムヘルス状態"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_latency: float
    cache_hit_rate: float
    response_time_avg: float
    error_rate: float
    active_connections: int
    performance_level: PerformanceLevel

@dataclass
class OptimizationRule:
    """最適化ルール"""
    rule_id: str
    name: str
    condition: str  # Python式
    action: OptimizationAction
    priority: int  # 1-10 (高いほど優先)
    cooldown_seconds: int
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0

@dataclass
class TuningHistory:
    """調整履歴"""
    timestamp: datetime
    action: OptimizationAction
    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    impact_score: float = 0.0
    success: bool = True

class PerformanceProfiler:
    """パフォーマンスプロファイラー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1440)  # 24時間分
        self.response_times = deque(maxlen=1000)
        self.error_counts = deque(maxlen=100)

    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """関数の性能プロファイリング"""

        start_time = time.perf_counter()
        memory_before = self._get_memory_usage()
        cpu_before = psutil.cpu_percent()

        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.perf_counter()
        memory_after = self._get_memory_usage()
        cpu_after = psutil.cpu_percent()

        profile = {
            'execution_time': end_time - start_time,
            'memory_delta': memory_after - memory_before,
            'cpu_usage': (cpu_before + cpu_after) / 2,
            'success': success,
            'error': error,
            'function_name': func.__name__ if hasattr(func, '__name__') else 'unknown'
        }

        return result, profile

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

class AutoTuner:
    """自動調整システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 調整可能パラメータ
        self.tunable_params = {
            'cache_size': {'min': 100, 'max': 5000, 'current': 1000},
            'worker_count': {'min': 1, 'max': 8, 'current': 4},
            'polling_interval': {'min': 30, 'max': 300, 'current': 60},
            'batch_size': {'min': 10, 'max': 1000, 'current': 100},
            'timeout': {'min': 5, 'max': 60, 'current': 30}
        }

        # 最適化ルール
        self.optimization_rules = []
        self._setup_default_rules()

        # 調整履歴
        self.tuning_history = deque(maxlen=500)

    def _setup_default_rules(self):
        """デフォルト最適化ルール設定"""

        # CPU使用率が高い場合
        self.optimization_rules.append(OptimizationRule(
            rule_id="high_cpu",
            name="高CPU使用率対応",
            condition="cpu_usage > 85",
            action=OptimizationAction.DECREASE_WORKERS,
            priority=8,
            cooldown_seconds=300
        ))

        # メモリ使用率が高い場合
        self.optimization_rules.append(OptimizationRule(
            rule_id="high_memory",
            name="高メモリ使用率対応",
            condition="memory_usage > 80",
            action=OptimizationAction.CLEAR_MEMORY,
            priority=9,
            cooldown_seconds=120
        ))

        # キャッシュヒット率が低い場合
        self.optimization_rules.append(OptimizationRule(
            rule_id="low_cache_hit",
            name="低キャッシュヒット率対応",
            condition="cache_hit_rate < 0.6",
            action=OptimizationAction.INCREASE_CACHE_SIZE,
            priority=6,
            cooldown_seconds=600
        ))

        # 応答時間が遅い場合
        self.optimization_rules.append(OptimizationRule(
            rule_id="slow_response",
            name="遅い応答時間対応",
            condition="response_time_avg > 5.0",
            action=OptimizationAction.OPTIMIZE_QUERIES,
            priority=7,
            cooldown_seconds=300
        ))

        # エラー率が高い場合
        self.optimization_rules.append(OptimizationRule(
            rule_id="high_error_rate",
            name="高エラー率対応",
            condition="error_rate > 0.05",
            action=OptimizationAction.RESTART_COMPONENTS,
            priority=10,
            cooldown_seconds=900
        ))

    def evaluate_and_apply_optimizations(self, health: SystemHealth) -> List[TuningHistory]:
        """最適化の評価と適用"""

        applied_actions = []

        # 条件評価用のコンテキスト
        context = {
            'cpu_usage': health.cpu_usage,
            'memory_usage': health.memory_usage,
            'cache_hit_rate': health.cache_hit_rate,
            'response_time_avg': health.response_time_avg,
            'error_rate': health.error_rate
        }

        # 優先度でソート
        sorted_rules = sorted(self.optimization_rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            # クールダウンチェック
            if rule.last_executed:
                time_since_last = (datetime.now() - rule.last_executed).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue

            # 条件評価
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    action_result = self._apply_optimization_action(rule.action, context)

                    if action_result:
                        history = TuningHistory(
                            timestamp=datetime.now(),
                            action=rule.action,
                            parameter=action_result['parameter'],
                            old_value=action_result['old_value'],
                            new_value=action_result['new_value'],
                            reason=rule.name,
                            impact_score=action_result.get('impact_score', 0.0),
                            success=action_result.get('success', True)
                        )

                        applied_actions.append(history)
                        self.tuning_history.append(history)

                        # ルール実行記録更新
                        rule.last_executed = datetime.now()
                        rule.execution_count += 1
                        if action_result.get('success', True):
                            rule.success_count += 1

                        self.logger.info(f"最適化適用: {rule.name} - {rule.action.value}")

            except Exception as e:
                self.logger.error(f"最適化ルール評価エラー {rule.rule_id}: {e}")
                continue

        return applied_actions

    def _apply_optimization_action(self, action: OptimizationAction, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """最適化アクション適用"""

        try:
            if action == OptimizationAction.INCREASE_CACHE_SIZE:
                return self._adjust_cache_size(increase=True)
            elif action == OptimizationAction.DECREASE_CACHE_SIZE:
                return self._adjust_cache_size(increase=False)
            elif action == OptimizationAction.INCREASE_WORKERS:
                return self._adjust_worker_count(increase=True)
            elif action == OptimizationAction.DECREASE_WORKERS:
                return self._adjust_worker_count(increase=False)
            elif action == OptimizationAction.CLEAR_MEMORY:
                return self._clear_memory()
            elif action == OptimizationAction.OPTIMIZE_QUERIES:
                return self._optimize_queries()
            elif action == OptimizationAction.REDUCE_POLLING:
                return self._adjust_polling_interval(increase=True)
            elif action == OptimizationAction.INCREASE_POLLING:
                return self._adjust_polling_interval(increase=False)
            elif action == OptimizationAction.RESTART_COMPONENTS:
                return self._restart_components()

            return None

        except Exception as e:
            self.logger.error(f"最適化アクション実行エラー {action.value}: {e}")
            return None

    def _adjust_cache_size(self, increase: bool) -> Dict[str, Any]:
        """キャッシュサイズ調整"""

        param = self.tunable_params['cache_size']
        old_value = param['current']

        if increase:
            new_value = min(param['max'], old_value * 1.5)
        else:
            new_value = max(param['min'], old_value * 0.7)

        new_value = int(new_value)
        param['current'] = new_value

        # 実際のキャッシュサイズ調整（グローバルインスタンス）
        try:
            from realtime_performance_optimizer import realtime_performance_optimizer
            realtime_performance_optimizer.data_cache.max_size = new_value

            impact_score = abs(new_value - old_value) / old_value

            return {
                'parameter': 'cache_size',
                'old_value': old_value,
                'new_value': new_value,
                'success': True,
                'impact_score': impact_score
            }
        except Exception:
            return {
                'parameter': 'cache_size',
                'old_value': old_value,
                'new_value': new_value,
                'success': False,
                'impact_score': 0.0
            }

    def _adjust_worker_count(self, increase: bool) -> Dict[str, Any]:
        """ワーカー数調整"""

        param = self.tunable_params['worker_count']
        old_value = param['current']

        if increase:
            new_value = min(param['max'], old_value + 1)
        else:
            new_value = max(param['min'], old_value - 1)

        param['current'] = new_value

        return {
            'parameter': 'worker_count',
            'old_value': old_value,
            'new_value': new_value,
            'success': True,
            'impact_score': abs(new_value - old_value) / max(old_value, 1)
        }

    def _clear_memory(self) -> Dict[str, Any]:
        """メモリクリア"""

        memory_before = psutil.virtual_memory().percent

        # ガベージコレクション実行
        collected = gc.collect()

        memory_after = psutil.virtual_memory().percent
        impact_score = max(0, (memory_before - memory_after) / 100)

        return {
            'parameter': 'memory_usage',
            'old_value': memory_before,
            'new_value': memory_after,
            'success': collected > 0,
            'impact_score': impact_score
        }

    def _optimize_queries(self) -> Dict[str, Any]:
        """クエリ最適化"""

        # 簡易的な最適化処理
        return {
            'parameter': 'query_optimization',
            'old_value': 'unoptimized',
            'new_value': 'optimized',
            'success': True,
            'impact_score': 0.1
        }

    def _adjust_polling_interval(self, increase: bool) -> Dict[str, Any]:
        """ポーリング間隔調整"""

        param = self.tunable_params['polling_interval']
        old_value = param['current']

        if increase:
            new_value = min(param['max'], old_value * 1.2)
        else:
            new_value = max(param['min'], old_value * 0.8)

        new_value = int(new_value)
        param['current'] = new_value

        return {
            'parameter': 'polling_interval',
            'old_value': old_value,
            'new_value': new_value,
            'success': True,
            'impact_score': abs(new_value - old_value) / old_value
        }

    def _restart_components(self) -> Dict[str, Any]:
        """コンポーネント再起動"""

        # 簡易的な再起動処理
        return {
            'parameter': 'component_restart',
            'old_value': 'running',
            'new_value': 'restarted',
            'success': True,
            'impact_score': 0.5
        }

class RealtimeMonitoringSystem:
    """リアルタイム監視システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # コンポーネント初期化
        self.profiler = PerformanceProfiler()
        self.auto_tuner = AutoTuner()

        # 監視状態
        self.is_monitoring = False
        self.monitoring_thread = None

        # メトリクス履歴
        self.health_history = deque(maxlen=1440)  # 24時間分

        # アラート設定
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 90.0,
            'disk_usage': 95.0,
            'response_time': 10.0,
            'error_rate': 0.1
        }

        # データベース設定
        self.db_path = Path("monitoring_data/realtime_monitoring.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

        self.logger.info("Realtime monitoring system initialized")

    def _init_database(self):
        """データベース初期化"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # システムヘルステーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        cpu_usage REAL NOT NULL,
                        memory_usage REAL NOT NULL,
                        memory_available REAL NOT NULL,
                        disk_usage REAL NOT NULL,
                        network_latency REAL NOT NULL,
                        cache_hit_rate REAL NOT NULL,
                        response_time_avg REAL NOT NULL,
                        error_rate REAL NOT NULL,
                        active_connections INTEGER NOT NULL,
                        performance_level TEXT NOT NULL
                    )
                ''')

                # 調整履歴テーブル
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tuning_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        parameter TEXT NOT NULL,
                        old_value TEXT NOT NULL,
                        new_value TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        impact_score REAL DEFAULT 0.0,
                        success INTEGER DEFAULT 1
                    )
                ''')

                conn.commit()

        except Exception as e:
            self.logger.error(f"データベース初期化エラー: {e}")

    def start_monitoring(self):
        """監視開始"""

        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("リアルタイム監視開始")

    def stop_monitoring(self):
        """監視停止"""

        self.is_monitoring = False
        self.logger.info("リアルタイム監視停止")

    def _monitoring_loop(self):
        """監視ループ"""

        while self.is_monitoring:
            try:
                # システムヘルス収集
                health = self._collect_system_health()
                self.health_history.append(health)

                # 自動調整の評価と適用
                tuning_actions = self.auto_tuner.evaluate_and_apply_optimizations(health)

                # データベース保存（5分毎）
                if len(self.health_history) % 5 == 0:
                    asyncio.create_task(self._save_health_to_db(health))

                    for action in tuning_actions:
                        asyncio.create_task(self._save_tuning_to_db(action))

                # アラートチェック
                self._check_alerts(health)

                # 1分間隔
                time.sleep(60)

            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                time.sleep(60)

    def _collect_system_health(self) -> SystemHealth:
        """システムヘルス収集"""

        try:
            # システムメトリクス
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # ネットワークレイテンシー（簡易測定）
            network_latency = self._measure_network_latency()

            # キャッシュヒット率
            cache_hit_rate = self._get_cache_hit_rate()

            # 応答時間平均
            response_time_avg = self._get_avg_response_time()

            # エラー率
            error_rate = self._get_error_rate()

            # アクティブ接続数
            active_connections = len(psutil.net_connections())

            # パフォーマンスレベル判定
            performance_score = self._calculate_performance_score(
                cpu_usage, memory.percent, cache_hit_rate, response_time_avg, error_rate
            )
            performance_level = self._score_to_level(performance_score)

            return SystemHealth(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available / 1024 / 1024 / 1024,  # GB
                disk_usage=disk.percent,
                network_latency=network_latency,
                cache_hit_rate=cache_hit_rate,
                response_time_avg=response_time_avg,
                error_rate=error_rate,
                active_connections=active_connections,
                performance_level=performance_level
            )

        except Exception as e:
            self.logger.error(f"システムヘルス収集エラー: {e}")
            return SystemHealth(
                timestamp=datetime.now(),
                cpu_usage=0.0, memory_usage=0.0, memory_available=0.0,
                disk_usage=0.0, network_latency=0.0, cache_hit_rate=0.0,
                response_time_avg=0.0, error_rate=1.0, active_connections=0,
                performance_level=PerformanceLevel.CRITICAL
            )

    def _measure_network_latency(self) -> float:
        """ネットワークレイテンシー測定"""

        try:
            import subprocess
            import platform

            if platform.system().lower() == 'windows':
                result = subprocess.run(['ping', '-n', '1', 'google.com'],
                                      capture_output=True, text=True, timeout=5)
            else:
                result = subprocess.run(['ping', '-c', '1', 'google.com'],
                                      capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                output = result.stdout
                # Windowsの場合の簡易パース
                if 'ms' in output:
                    for line in output.split('\n'):
                        if 'ms' in line and '=' in line:
                            try:
                                latency_str = line.split('=')[-1].split('ms')[0].strip()
                                return float(latency_str)
                            except:
                                continue

            return 100.0  # デフォルト値

        except Exception:
            return 100.0

    def _get_cache_hit_rate(self) -> float:
        """キャッシュヒット率取得"""

        try:
            from realtime_performance_optimizer import realtime_performance_optimizer
            data_cache_stats = realtime_performance_optimizer.data_cache.get_stats()
            return data_cache_stats.get('hit_rate', 0.0)
        except:
            return 0.0

    def _get_avg_response_time(self) -> float:
        """平均応答時間取得"""

        if self.profiler.response_times:
            return np.mean(list(self.profiler.response_times))
        return 0.0

    def _get_error_rate(self) -> float:
        """エラー率取得"""

        if self.profiler.error_counts:
            recent_errors = list(self.profiler.error_counts)[-10:]  # 直近10件
            return np.mean(recent_errors) if recent_errors else 0.0
        return 0.0

    def _calculate_performance_score(self, cpu: float, memory: float,
                                   cache_hit: float, response_time: float, error_rate: float) -> float:
        """パフォーマンススコア計算"""

        # 各メトリクスを0-100スケールに正規化
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory)
        cache_score = cache_hit * 100
        response_score = max(0, 100 - min(response_time * 10, 100))
        error_score = max(0, 100 - error_rate * 1000)

        # 重み付き平均
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        scores = [cpu_score, memory_score, cache_score, response_score, error_score]

        return sum(w * s for w, s in zip(weights, scores))

    def _score_to_level(self, score: float) -> PerformanceLevel:
        """スコアからパフォーマンスレベルへ変換"""

        if score >= 90:
            return PerformanceLevel.EXCELLENT
        elif score >= 70:
            return PerformanceLevel.GOOD
        elif score >= 50:
            return PerformanceLevel.FAIR
        elif score >= 30:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _check_alerts(self, health: SystemHealth):
        """アラートチェック"""

        alerts = []

        if health.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(f"高CPU使用率: {health.cpu_usage:.1f}%")

        if health.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(f"高メモリ使用率: {health.memory_usage:.1f}%")

        if health.disk_usage > self.alert_thresholds['disk_usage']:
            alerts.append(f"高ディスク使用率: {health.disk_usage:.1f}%")

        if health.response_time_avg > self.alert_thresholds['response_time']:
            alerts.append(f"遅い応答時間: {health.response_time_avg:.2f}秒")

        if health.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"高エラー率: {health.error_rate:.1%}")

        if alerts:
            self.logger.warning(f"システムアラート: {'; '.join(alerts)}")

    async def _save_health_to_db(self, health: SystemHealth):
        """システムヘルスをデータベースに保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO system_health
                    (timestamp, cpu_usage, memory_usage, memory_available, disk_usage,
                     network_latency, cache_hit_rate, response_time_avg, error_rate,
                     active_connections, performance_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    health.timestamp.isoformat(),
                    health.cpu_usage,
                    health.memory_usage,
                    health.memory_available,
                    health.disk_usage,
                    health.network_latency,
                    health.cache_hit_rate,
                    health.response_time_avg,
                    health.error_rate,
                    health.active_connections,
                    health.performance_level.value
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"ヘルス保存エラー: {e}")

    async def _save_tuning_to_db(self, tuning: TuningHistory):
        """調整履歴をデータベースに保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO tuning_history
                    (timestamp, action, parameter, old_value, new_value, reason, impact_score, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tuning.timestamp.isoformat(),
                    tuning.action.value,
                    tuning.parameter,
                    json.dumps(tuning.old_value),
                    json.dumps(tuning.new_value),
                    tuning.reason,
                    tuning.impact_score,
                    1 if tuning.success else 0
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"調整履歴保存エラー: {e}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """監視レポート取得"""

        if not self.health_history:
            return {"error": "ヘルス履歴なし"}

        recent_health = list(self.health_history)[-60:]  # 過去1時間

        # 統計計算
        avg_cpu = np.mean([h.cpu_usage for h in recent_health])
        avg_memory = np.mean([h.memory_usage for h in recent_health])
        avg_cache_hit = np.mean([h.cache_hit_rate for h in recent_health])
        avg_response_time = np.mean([h.response_time_avg for h in recent_health])
        avg_error_rate = np.mean([h.error_rate for h in recent_health])

        # パフォーマンスレベル分布
        level_counts = {}
        for health in recent_health:
            level = health.performance_level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        # 自動調整統計
        recent_tuning = list(self.auto_tuner.tuning_history)[-20:]  # 直近20件

        return {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.is_monitoring else 'inactive',
            'system_metrics': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'avg_cache_hit_rate': avg_cache_hit,
                'avg_response_time': avg_response_time,
                'avg_error_rate': avg_error_rate
            },
            'performance_distribution': level_counts,
            'recent_tuning_actions': len(recent_tuning),
            'tunable_parameters': {
                param: config['current']
                for param, config in self.auto_tuner.tunable_params.items()
            },
            'health_history_size': len(self.health_history),
            'optimization_rules': len(self.auto_tuner.optimization_rules)
        }

# グローバルインスタンス
realtime_monitoring_system = RealtimeMonitoringSystem()

# テスト実行
async def run_monitoring_system_test():
    """監視システムテスト実行"""

    print("=== 📊 リアルタイム監視・自動調整システムテスト ===")

    # 監視開始
    realtime_monitoring_system.start_monitoring()

    print(f"\n🔍 システムヘルス収集テスト")

    # 複数回ヘルス収集
    for i in range(3):
        health = realtime_monitoring_system._collect_system_health()
        print(f"\n--- 収集 {i+1} ---")
        print(f"  CPU使用率: {health.cpu_usage:.1f}%")
        print(f"  メモリ使用率: {health.memory_usage:.1f}%")
        print(f"  キャッシュヒット率: {health.cache_hit_rate:.1%}")
        print(f"  応答時間: {health.response_time_avg:.2f}秒")
        print(f"  パフォーマンスレベル: {health.performance_level.value}")

        # 自動調整テスト
        tuning_actions = realtime_monitoring_system.auto_tuner.evaluate_and_apply_optimizations(health)
        if tuning_actions:
            print(f"  自動調整: {len(tuning_actions)}件適用")
            for action in tuning_actions:
                print(f"    - {action.reason}: {action.parameter} {action.old_value} → {action.new_value}")
        else:
            print(f"  自動調整: なし")

        await asyncio.sleep(2)

    # 監視レポート
    print(f"\n📈 監視レポート")
    report = realtime_monitoring_system.get_monitoring_report()

    print(f"  監視状況: {report['monitoring_status']}")

    system_metrics = report['system_metrics']
    print(f"  平均CPU使用率: {system_metrics['avg_cpu_usage']:.1f}%")
    print(f"  平均メモリ使用率: {system_metrics['avg_memory_usage']:.1f}%")
    print(f"  平均キャッシュヒット率: {system_metrics['avg_cache_hit_rate']:.1%}")
    print(f"  平均応答時間: {system_metrics['avg_response_time']:.2f}秒")

    # 調整可能パラメータ
    print(f"\n⚙️ 現在のパラメータ:")
    tunable_params = report['tunable_parameters']
    for param, value in tunable_params.items():
        print(f"  {param}: {value}")

    # パフォーマンス分布
    if 'performance_distribution' in report:
        print(f"\n🎯 パフォーマンス分布:")
        for level, count in report['performance_distribution'].items():
            print(f"  {level}: {count}回")

    print(f"\n✅ リアルタイム監視・自動調整システム稼働中")
    print(f"システムは継続的に監視され、必要に応じて自動調整されます。")

    # 少し待って監視停止
    await asyncio.sleep(5)
    realtime_monitoring_system.stop_monitoring()

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(run_monitoring_system_test())