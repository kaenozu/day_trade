#!/usr/bin/env python3
"""
メモリ使用量最適化

効率的なメモリ管理とガベージコレクション最適化
"""

import gc
import sys
import threading
import time
import weakref
import os
from typing import Any, Dict, List, Optional
try:
    import psutil
except ImportError:
    psutil = None


class MemoryMonitor:
    """メモリ監視クラス"""

    def __init__(self, threshold_mb: float = 500.0):
        self.threshold_mb = threshold_mb
        self.monitoring = False
        self.monitor_thread = None
        self._callbacks = []

    def add_callback(self, callback):
        """メモリ不足時のコールバック追加"""
        self._callbacks.append(callback)

    def get_memory_usage(self) -> Dict[str, float]:
        """メモリ使用量取得"""
        if psutil:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        else:
            # psutilがない場合の代替実装
            return {
                'rss_mb': sys.getsizeof(gc.get_objects()) / 1024 / 1024,
                'vms_mb': 0.0,
                'percent': 0.0
            }

    def start_monitoring(self):
        """監視開始"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                memory_info = self.get_memory_usage()

                if memory_info['rss_mb'] > self.threshold_mb:
                    print(f"⚠️ メモリ使用量警告: {memory_info['rss_mb']:.1f}MB")

                    # コールバック実行
                    for callback in self._callbacks:
                        try:
                            callback(memory_info)
                        except Exception as e:
                            print(f"メモリコールバックエラー: {e}")

                time.sleep(5.0)  # 5秒間隔

            except Exception as e:
                print(f"メモリ監視エラー: {e}")
                time.sleep(10.0)


class MemoryOptimizer:
    """メモリ最適化クラス"""

    def __init__(self):
        self.weak_refs = weakref.WeakSet()
        self.monitor = MemoryMonitor()
        self.monitor.add_callback(self._on_memory_pressure)
        self._cleanup_callbacks = []

    def register_object(self, obj):
        """オブジェクト登録（弱参照）"""
        self.weak_refs.add(obj)

    def add_cleanup_callback(self, callback):
        """クリーンアップコールバック追加"""
        self._cleanup_callbacks.append(callback)

    def _on_memory_pressure(self, memory_info):
        """メモリ圧迫時の処理"""
        print("🧹 メモリクリーンアップ実行中...")

        # ガベージコレクション実行
        collected = gc.collect()
        print(f"   ガベージコレクション: {collected}個のオブジェクト解放")

        # 手動クリーンアップ
        self._manual_cleanup()

        # クリーンアップコールバック実行
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"クリーンアップコールバックエラー: {e}")

        # 再度メモリ使用量確認
        new_memory = self.monitor.get_memory_usage()
        saved_mb = memory_info['rss_mb'] - new_memory['rss_mb']
        print(f"   メモリ解放: {saved_mb:.1f}MB")

    def _manual_cleanup(self):
        """手動クリーンアップ"""
        # 弱参照オブジェクトのクリーンアップ
        alive_objects = []
        for obj in self.weak_refs:
            if hasattr(obj, 'cleanup'):
                try:
                    obj.cleanup()
                except:
                    pass
            alive_objects.append(obj)

        print(f"   弱参照オブジェクト: {len(alive_objects)}個クリーンアップ")

    def optimize_gc(self):
        """ガベージコレクション最適化"""
        # GC閾値調整（メモリ使用量に応じて）
        current_memory = self.monitor.get_memory_usage()

        if current_memory['rss_mb'] > 200:
            # メモリ使用量が多い場合は頻繁にGC
            gc.set_threshold(700, 10, 10)
        else:
            # 通常時はデフォルト
            gc.set_threshold(700, 10, 10)

        # 不要なデバッグ情報を無効化
        gc.set_debug(0)

        print("ガベージコレクション最適化完了")

    def force_cleanup(self):
        """強制クリーンアップ"""
        print("🔧 強制メモリクリーンアップ実行")
        
        # 複数回ガベージコレクション実行
        for i in range(3):
            collected = gc.collect()
            print(f"   GCラウンド{i+1}: {collected}個のオブジェクト解放")
        
        # 手動クリーンアップ
        self._manual_cleanup()


class DataFrameOptimizer:
    """DataFrame最適化"""

    @staticmethod
    def optimize_dtypes(df) -> Any:
        """データ型最適化"""
        try:
            import pandas as pd

            # 数値型最適化
            for col in df.select_dtypes(include=['int64']).columns:
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() >= -128 and df[col].max() <= 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() >= -32768 and df[col].max() <= 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                        df[col] = df[col].astype('int32')

            # float型最適化
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

            # カテゴリ型最適化
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # ユニーク値が50%未満
                    df[col] = df[col].astype('category')

            return df

        except ImportError:
            return df

    @staticmethod
    def memory_usage_summary(df) -> Dict[str, Any]:
        """メモリ使用量サマリー"""
        try:
            memory_usage = df.memory_usage(deep=True)
            return {
                'total_mb': memory_usage.sum() / 1024 / 1024,
                'by_column': {
                    col: usage / 1024 / 1024
                    for col, usage in memory_usage.items()
                }
            }
        except:
            return {}


class MemoryProfiler:
    """メモリプロファイラー"""
    
    def __init__(self):
        self._snapshots = []
    
    def take_snapshot(self, label: str = ""):
        """メモリスナップショット取得"""
        memory_info = get_memory_stats()
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'memory': memory_info
        }
        self._snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(self, start_idx: int = -2, end_idx: int = -1) -> Dict[str, Any]:
        """スナップショット比較"""
        if len(self._snapshots) < 2:
            return {}
        
        start = self._snapshots[start_idx]
        end = self._snapshots[end_idx]
        
        diff_mb = end['memory']['rss_mb'] - start['memory']['rss_mb']
        
        return {
            'start_label': start['label'],
            'end_label': end['label'],
            'memory_diff_mb': diff_mb,
            'time_diff_sec': end['timestamp'] - start['timestamp']
        }


# グローバル最適化インスタンス
memory_optimizer = MemoryOptimizer()
memory_profiler = MemoryProfiler()


def start_memory_monitoring():
    """メモリ監視開始"""
    memory_optimizer.monitor.start_monitoring()
    memory_optimizer.optimize_gc()


def stop_memory_monitoring():
    """メモリ監視停止"""
    memory_optimizer.monitor.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """メモリ統計取得"""
    return memory_optimizer.monitor.get_memory_usage()


def force_cleanup():
    """強制クリーンアップ"""
    memory_optimizer.force_cleanup()


# 自動クリーンアップデコレータ
def auto_cleanup(func):
    """自動クリーンアップデコレータ"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 関数終了時にガベージコレクション
            gc.collect()

    return wrapper


# プロファイリングデコレータ
def memory_profile(label: str = ""):
    """メモリプロファイリングデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 実行前のスナップショット
            memory_profiler.take_snapshot(f"{label}_start")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 実行後のスナップショット
                memory_profiler.take_snapshot(f"{label}_end")
                
                # 比較結果を出力
                comparison = memory_profiler.compare_snapshots()
                if comparison:
                    print(f"メモリ使用量変化: {comparison['memory_diff_mb']:.2f}MB")
        
        return wrapper
    return decorator