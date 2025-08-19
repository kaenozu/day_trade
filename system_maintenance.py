#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Maintenance - システムメンテナンス・長期稼働安定性
Issue #948対応: 自動メンテナンス + ヘルスチェック + 自己診断
"""

import os
import sys
import time
import shutil
import sqlite3
import psutil
import threading
import asyncio
import gc
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import json
import subprocess
import hashlib

# システム監視用
try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class MaintenanceLevel(Enum):
    """メンテナンスレベル"""
    LIGHT = "LIGHT"        # 軽微なクリーンアップ
    STANDARD = "STANDARD"  # 標準的なメンテナンス
    DEEP = "DEEP"         # ディープクリーンアップ
    CRITICAL = "CRITICAL"  # クリティカル修復


class HealthStatus(Enum):
    """ヘルス状態"""
    EXCELLENT = "EXCELLENT"  # 95-100%
    GOOD = "GOOD"           # 80-94%
    FAIR = "FAIR"           # 60-79%
    POOR = "POOR"           # 40-59%
    CRITICAL = "CRITICAL"   # 0-39%


@dataclass
class SystemHealth:
    """システムヘルス情報"""
    timestamp: datetime
    overall_score: int  # 0-100
    status: HealthStatus
    cpu_health: int
    memory_health: int
    disk_health: int
    database_health: int
    component_health: Dict[str, int]
    active_issues: List[str]
    recommendations: List[str]


@dataclass
class MaintenanceTask:
    """メンテナンスタスク"""
    task_id: str
    name: str
    description: str
    level: MaintenanceLevel
    frequency_hours: int
    last_run: Optional[datetime]
    next_run: datetime
    estimated_duration: int  # 秒
    auto_execute: bool
    maintenance_function: Callable
    dependencies: List[str]


class DatabaseMaintenance:
    """データベースメンテナンス"""

    def __init__(self):
        self.db_paths = self._discover_databases()
        self.maintenance_log = []

    def _discover_databases(self) -> List[str]:
        """データベースファイル検出"""
        db_files = []

        # 主要ディレクトリ内のSQLiteファイルを検索
        search_dirs = [
            'data/databases',
            'data',
            '.',
            'backups'
        ]

        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith('.db') or file.endswith('.sqlite'):
                            db_files.append(os.path.join(root, file))

        return db_files

    def vacuum_databases(self) -> Dict[str, Any]:
        """データベースVACUUM実行"""
        results = {}
        total_space_freed = 0

        for db_path in self.db_paths:
            try:
                if not os.path.exists(db_path):
                    continue

                # ファイルサイズ（前）
                size_before = os.path.getsize(db_path)

                # VACUUM実行
                conn = sqlite3.connect(db_path)
                conn.execute('VACUUM')
                conn.close()

                # ファイルサイズ（後）
                size_after = os.path.getsize(db_path)
                space_freed = size_before - size_after
                total_space_freed += space_freed

                results[db_path] = {
                    'status': 'success',
                    'size_before_mb': size_before / 1024 / 1024,
                    'size_after_mb': size_after / 1024 / 1024,
                    'space_freed_mb': space_freed / 1024 / 1024
                }

                logging.info(f"VACUUM completed for {db_path}: {space_freed/1024/1024:.2f}MB freed")

            except Exception as e:
                results[db_path] = {
                    'status': 'error',
                    'error': str(e)
                }
                logging.error(f"VACUUM failed for {db_path}: {e}")

        return {
            'databases_processed': len(results),
            'total_space_freed_mb': total_space_freed / 1024 / 1024,
            'results': results
        }

    def analyze_database_health(self) -> Dict[str, Any]:
        """データベースヘルス分析"""
        health_scores = {}
        overall_issues = []

        for db_path in self.db_paths:
            try:
                if not os.path.exists(db_path):
                    continue

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # 基本統計
                cursor.execute("PRAGMA database_list")
                db_info = cursor.fetchall()

                cursor.execute("PRAGMA table_list")
                tables = cursor.fetchall()

                # インデックス情報
                cursor.execute("PRAGMA index_list")
                indexes = cursor.fetchall()

                # ファイルサイズ
                file_size_mb = os.path.getsize(db_path) / 1024 / 1024

                # ヘルススコア計算
                score = 100
                issues = []

                # ファイルサイズチェック
                if file_size_mb > 100:  # 100MB超
                    score -= 10
                    issues.append("Large database file size")

                # テーブル数チェック
                if len(tables) > 50:
                    score -= 5
                    issues.append("High table count")

                # 破損チェック
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()
                if integrity_result[0] != 'ok':
                    score -= 30
                    issues.append("Database integrity issues")

                conn.close()

                health_scores[db_path] = {
                    'score': max(0, score),
                    'file_size_mb': file_size_mb,
                    'table_count': len(tables),
                    'index_count': len(indexes),
                    'issues': issues
                }

            except Exception as e:
                health_scores[db_path] = {
                    'score': 0,
                    'error': str(e),
                    'issues': ['Database access error']
                }
                overall_issues.append(f"Database error: {db_path}")

        # 全体スコア計算
        scores = [info['score'] for info in health_scores.values() if 'score' in info]
        overall_score = sum(scores) / len(scores) if scores else 0

        return {
            'overall_score': overall_score,
            'database_scores': health_scores,
            'total_databases': len(self.db_paths),
            'issues': overall_issues
        }


class FileSystemMaintenance:
    """ファイルシステムメンテナンス"""

    def __init__(self):
        self.temp_dirs = ['temp', 'cache', '__pycache__']
        self.log_retention_days = 30
        self.backup_retention_days = 7

    def cleanup_temporary_files(self) -> Dict[str, Any]:
        """一時ファイルクリーンアップ"""
        cleaned_files = 0
        freed_space = 0

        # 一時ディレクトリクリーンアップ
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                freed_space += self._cleanup_directory(temp_dir)
                cleaned_files += self._count_files_in_directory(temp_dir)

        # Pythonキャッシュファイル
        freed_space += self._cleanup_python_cache()

        # 古いログファイル
        log_cleanup = self._cleanup_old_logs()
        freed_space += log_cleanup['freed_space']
        cleaned_files += log_cleanup['files_removed']

        return {
            'files_cleaned': cleaned_files,
            'space_freed_mb': freed_space / 1024 / 1024,
            'directories_processed': len(self.temp_dirs)
        }

    def _cleanup_directory(self, directory: str) -> int:
        """ディレクトリクリーンアップ"""
        freed_space = 0

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_space += file_size
                    except Exception:
                        pass
        except Exception:
            pass

        return freed_space

    def _count_files_in_directory(self, directory: str) -> int:
        """ディレクトリ内ファイル数カウント"""
        try:
            return sum(len(files) for _, _, files in os.walk(directory))
        except Exception:
            return 0

    def _cleanup_python_cache(self) -> int:
        """Pythonキャッシュクリーンアップ"""
        freed_space = 0

        for root, dirs, files in os.walk('.'):
            for dir_name in dirs:
                if dir_name == '__pycache__':
                    cache_dir = os.path.join(root, dir_name)
                    try:
                        freed_space += self._cleanup_directory(cache_dir)
                        shutil.rmtree(cache_dir)
                    except Exception:
                        pass

        return freed_space

    def _cleanup_old_logs(self) -> Dict[str, Any]:
        """古いログファイルクリーンアップ"""
        cutoff_date = datetime.now() - timedelta(days=self.log_retention_days)
        freed_space = 0
        files_removed = 0

        log_dirs = ['logs', 'data/logs']

        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                continue

            for file in os.listdir(log_dir):
                file_path = os.path.join(log_dir, file)

                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_date and file.endswith('.log'):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_space += file_size
                        files_removed += 1
                except Exception:
                    pass

        return {
            'freed_space': freed_space,
            'files_removed': files_removed
        }


class SystemHealthChecker:
    """システムヘルスチェック"""

    def __init__(self):
        self.db_maintenance = DatabaseMaintenance()
        self.fs_maintenance = FileSystemMaintenance()
        self.component_checkers = {}

    def register_component_checker(self, component: str, checker_func: Callable):
        """コンポーネントチェッカー登録"""
        self.component_checkers[component] = checker_func

    def perform_comprehensive_health_check(self) -> SystemHealth:
        """包括的ヘルスチェック"""
        # CPU健全性
        cpu_health = self._check_cpu_health()

        # メモリ健全性
        memory_health = self._check_memory_health()

        # ディスク健全性
        disk_health = self._check_disk_health()

        # データベース健全性
        db_analysis = self.db_maintenance.analyze_database_health()
        database_health = int(db_analysis['overall_score'])

        # コンポーネント健全性
        component_health = {}
        for component, checker in self.component_checkers.items():
            try:
                component_health[component] = checker()
            except Exception as e:
                component_health[component] = 0
                logging.error(f"Component health check failed for {component}: {e}")

        # 全体スコア計算
        base_scores = [cpu_health, memory_health, disk_health, database_health]
        component_scores = list(component_health.values())
        all_scores = base_scores + component_scores

        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # ステータス決定
        if overall_score >= 95:
            status = HealthStatus.EXCELLENT
        elif overall_score >= 80:
            status = HealthStatus.GOOD
        elif overall_score >= 60:
            status = HealthStatus.FAIR
        elif overall_score >= 40:
            status = HealthStatus.POOR
        else:
            status = HealthStatus.CRITICAL

        # 問題と推奨事項
        issues = []
        recommendations = []

        if cpu_health < 70:
            issues.append("High CPU usage detected")
            recommendations.append("Review CPU-intensive processes")

        if memory_health < 70:
            issues.append("High memory usage detected")
            recommendations.append("Perform memory cleanup")

        if disk_health < 70:
            issues.append("Low disk space")
            recommendations.append("Clean up temporary files")

        if database_health < 80:
            issues.append("Database optimization needed")
            recommendations.append("Run database maintenance")

        return SystemHealth(
            timestamp=datetime.now(),
            overall_score=int(overall_score),
            status=status,
            cpu_health=cpu_health,
            memory_health=memory_health,
            disk_health=disk_health,
            database_health=database_health,
            component_health=component_health,
            active_issues=issues,
            recommendations=recommendations
        )

    def _check_cpu_health(self) -> int:
        """CPU健全性チェック"""
        try:
            # 1分間の平均CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent < 20:
                return 100
            elif cpu_percent < 40:
                return 90
            elif cpu_percent < 60:
                return 80
            elif cpu_percent < 80:
                return 60
            else:
                return 30
        except Exception:
            return 50

    def _check_memory_health(self) -> int:
        """メモリ健全性チェック"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            if usage_percent < 30:
                return 100
            elif usage_percent < 50:
                return 90
            elif usage_percent < 70:
                return 80
            elif usage_percent < 85:
                return 60
            else:
                return 30
        except Exception:
            return 50

    def _check_disk_health(self) -> int:
        """ディスク健全性チェック"""
        try:
            disk = psutil.disk_usage('.')
            usage_percent = (disk.used / disk.total) * 100

            if usage_percent < 50:
                return 100
            elif usage_percent < 70:
                return 90
            elif usage_percent < 80:
                return 80
            elif usage_percent < 90:
                return 60
            else:
                return 30
        except Exception:
            return 50


class AutoMaintenanceScheduler:
    """自動メンテナンススケジューラ"""

    def __init__(self):
        self.maintenance_tasks: Dict[str, MaintenanceTask] = {}
        self.scheduler_running = False
        self.scheduler_thread = None
        self.health_checker = SystemHealthChecker()

        # デフォルトタスク登録
        self._register_default_tasks()

    def _register_default_tasks(self):
        """デフォルトメンテナンスタスク登録"""
        # 軽量クリーンアップ（毎時）
        self.register_task(MaintenanceTask(
            task_id="light_cleanup",
            name="軽量クリーンアップ",
            description="メモリクリーンアップとガベージコレクション",
            level=MaintenanceLevel.LIGHT,
            frequency_hours=1,
            last_run=None,
            next_run=datetime.now(),
            estimated_duration=30,  # 30秒
            auto_execute=True,
            maintenance_function=self._light_cleanup,
            dependencies=[]
        ))

        # 標準メンテナンス（毎日）
        self.register_task(MaintenanceTask(
            task_id="daily_maintenance",
            name="日次メンテナンス",
            description="一時ファイル削除とログローテーション",
            level=MaintenanceLevel.STANDARD,
            frequency_hours=24,
            last_run=None,
            next_run=datetime.now() + timedelta(hours=1),
            estimated_duration=300,  # 5分
            auto_execute=True,
            maintenance_function=self._daily_maintenance,
            dependencies=[]
        ))

        # データベース最適化（週次）
        self.register_task(MaintenanceTask(
            task_id="database_optimization",
            name="データベース最適化",
            description="VACUUM実行とインデックス再構築",
            level=MaintenanceLevel.DEEP,
            frequency_hours=168,  # 7日
            last_run=None,
            next_run=datetime.now() + timedelta(hours=2),
            estimated_duration=600,  # 10分
            auto_execute=True,
            maintenance_function=self._database_optimization,
            dependencies=[]
        ))

    def register_task(self, task: MaintenanceTask):
        """メンテナンスタスク登録"""
        self.maintenance_tasks[task.task_id] = task
        logging.info(f"Registered maintenance task: {task.name}")

    def start_scheduler(self):
        """スケジューラ開始"""
        if self.scheduler_running:
            return

        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logging.info("Auto maintenance scheduler started")

    def stop_scheduler(self):
        """スケジューラ停止"""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logging.info("Auto maintenance scheduler stopped")

    def _scheduler_loop(self):
        """スケジューラメインループ"""
        while self.scheduler_running:
            try:
                current_time = datetime.now()

                for task in self.maintenance_tasks.values():
                    if task.auto_execute and current_time >= task.next_run:
                        self._execute_maintenance_task(task)

                # 30秒間隔でチェック
                time.sleep(30)

            except Exception as e:
                logging.error(f"Scheduler loop error: {e}")
                time.sleep(60)  # エラー時は1分待機

    def _execute_maintenance_task(self, task: MaintenanceTask):
        """メンテナンスタスク実行"""
        try:
            logging.info(f"Starting maintenance task: {task.name}")
            start_time = time.time()

            # 依存関係チェック
            for dep_id in task.dependencies:
                if dep_id in self.maintenance_tasks:
                    dep_task = self.maintenance_tasks[dep_id]
                    if dep_task.last_run is None or dep_task.last_run < datetime.now() - timedelta(hours=dep_task.frequency_hours):
                        logging.warning(f"Dependency not satisfied: {dep_id}")
                        return

            # タスク実行
            result = task.maintenance_function()

            execution_time = time.time() - start_time

            # 実行情報更新
            task.last_run = datetime.now()
            task.next_run = datetime.now() + timedelta(hours=task.frequency_hours)

            logging.info(f"Completed maintenance task: {task.name} ({execution_time:.1f}s)")

            # 結果ログ
            if isinstance(result, dict):
                logging.info(f"Task result: {result}")

        except Exception as e:
            logging.error(f"Maintenance task failed: {task.name} - {e}")
            # 失敗時は30分後に再試行
            task.next_run = datetime.now() + timedelta(minutes=30)

    def _light_cleanup(self) -> Dict[str, Any]:
        """軽量クリーンアップ"""
        # ガベージコレクション
        collected = gc.collect()

        # 基本統計
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024

        # 簡単なクリーンアップ
        try:
            # キャッシュクリア等（安全な操作のみ）
            pass
        except Exception as e:
            logging.error(f"Light cleanup error: {e}")

        memory_after = psutil.Process().memory_info().rss / 1024 / 1024

        return {
            'gc_collected': collected,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_before - memory_after
        }

    def _daily_maintenance(self) -> Dict[str, Any]:
        """日次メンテナンス"""
        # ファイルシステムクリーンアップ
        fs_result = self.health_checker.fs_maintenance.cleanup_temporary_files()

        # ヘルスチェック
        health = self.health_checker.perform_comprehensive_health_check()

        return {
            'filesystem_cleanup': fs_result,
            'health_score': health.overall_score,
            'issues_found': len(health.active_issues)
        }

    def _database_optimization(self) -> Dict[str, Any]:
        """データベース最適化"""
        # データベースVACUUM
        vacuum_result = self.health_checker.db_maintenance.vacuum_databases()

        # データベースヘルス分析
        db_health = self.health_checker.db_maintenance.analyze_database_health()

        return {
            'vacuum_result': vacuum_result,
            'database_health': db_health['overall_score'],
            'databases_processed': vacuum_result['databases_processed']
        }

    def get_maintenance_status(self) -> Dict[str, Any]:
        """メンテナンス状況取得"""
        current_time = datetime.now()

        task_status = {}
        for task_id, task in self.maintenance_tasks.items():
            next_run_in = (task.next_run - current_time).total_seconds() / 3600  # 時間

            task_status[task_id] = {
                'name': task.name,
                'level': task.level.value,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat(),
                'next_run_in_hours': round(next_run_in, 1),
                'auto_execute': task.auto_execute
            }

        return {
            'scheduler_running': self.scheduler_running,
            'registered_tasks': len(self.maintenance_tasks),
            'task_status': task_status,
            'current_time': current_time.isoformat()
        }


# グローバルインスタンス
maintenance_scheduler = AutoMaintenanceScheduler()


def start_maintenance_system():
    """メンテナンスシステム開始"""
    maintenance_scheduler.start_scheduler()


def stop_maintenance_system():
    """メンテナンスシステム停止"""
    maintenance_scheduler.stop_scheduler()


def get_system_health() -> SystemHealth:
    """システムヘルス取得"""
    return maintenance_scheduler.health_checker.perform_comprehensive_health_check()


async def test_system_maintenance():
    """システムメンテナンステスト"""
    print("=== System Maintenance Test ===")

    # ヘルスチェック
    print("1. Performing system health check...")
    health = get_system_health()
    print(f"Overall Health Score: {health.overall_score}/100 ({health.status.value})")
    print(f"CPU Health: {health.cpu_health}")
    print(f"Memory Health: {health.memory_health}")
    print(f"Disk Health: {health.disk_health}")
    print(f"Database Health: {health.database_health}")

    if health.active_issues:
        print("Active Issues:")
        for issue in health.active_issues:
            print(f"  - {issue}")

    if health.recommendations:
        print("Recommendations:")
        for rec in health.recommendations:
            print(f"  - {rec}")

    # 軽量メンテナンス実行
    print("\n2. Running light maintenance...")
    light_result = maintenance_scheduler._light_cleanup()
    print(f"GC collected: {light_result['gc_collected']} objects")
    print(f"Memory usage: {light_result['memory_after_mb']:.1f}MB")

    # メンテナンス状況
    print("\n3. Maintenance status:")
    status = maintenance_scheduler.get_maintenance_status()
    print(f"Scheduler running: {status['scheduler_running']}")
    print(f"Registered tasks: {status['registered_tasks']}")

    for task_id, task_info in status['task_status'].items():
        print(f"  {task_info['name']}: next run in {task_info['next_run_in_hours']:.1f}h")

    print("\nSystem maintenance test completed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asyncio.run(test_system_maintenance())