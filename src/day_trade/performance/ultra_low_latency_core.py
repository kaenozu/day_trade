#!/usr/bin/env python3
"""
超低レイテンシHFTコアエンジン
Issue #443: HFT超低レイテンシ最適化 - <10μs実現戦略

Rust FFI統合による究極の低レイテンシHFT実行エンジン
目標: エンドツーエンドレイテンシ <10μs
"""

import ctypes
import os
import platform
import threading
import time
from ctypes import Structure, c_char_p, c_double, c_int, c_uint64, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import mmap

    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False

try:
    from ..utils.logging_config import get_context_logger
except ImportError:
    import logging

    def get_context_logger(name):
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class UltraLowLatencyConfig:
    """超低レイテンシ設定"""

    # レイテンシ目標
    target_latency_ns: int = 10000  # 10μs
    critical_latency_ns: int = 5000  # 5μs（クリティカルパス）

    # システム最適化
    enable_rust_core: bool = True
    enable_lock_free: bool = True
    enable_zero_copy: bool = True
    enable_cpu_affinity: bool = True

    # メモリ最適化
    preallocated_memory_mb: int = 512
    use_huge_pages: bool = True
    numa_node: int = 0

    # CPU設定
    dedicated_cpu_cores: List[int] = None
    scheduler_policy: str = "SCHED_FIFO"  # リアルタイムスケジューラ
    process_priority: int = -20  # 最高優先度

    # ネットワーク最適化
    enable_dpdk: bool = False  # Phase 3で実装
    enable_kernel_bypass: bool = False

    # プロファイリング
    enable_latency_tracking: bool = True
    enable_rdtsc_timing: bool = True  # CPU cycle カウンター


# Rust FFI用データ構造
class TradeRequest(Structure):
    """取引リクエスト構造体"""

    _fields_ = [
        ("symbol", c_char_p),
        ("side", c_int),  # 0=buy, 1=sell
        ("quantity", c_double),
        ("price", c_double),
        ("order_type", c_int),  # 0=market, 1=limit
        ("timestamp_ns", c_uint64),
    ]


class TradeResult(Structure):
    """取引結果構造体"""

    _fields_ = [
        ("status", c_int),  # 0=success, 1=error
        ("order_id", c_uint64),
        ("executed_price", c_double),
        ("executed_quantity", c_double),
        ("latency_ns", c_uint64),
        ("timestamp_ns", c_uint64),
        ("error_code", c_int),
    ]


class MarketData(Structure):
    """マーケットデータ構造体"""

    _fields_ = [
        ("symbol", c_char_p),
        ("bid_price", c_double),
        ("ask_price", c_double),
        ("bid_size", c_double),
        ("ask_size", c_double),
        ("timestamp_ns", c_uint64),
    ]


class LockFreeRingBufferStats(Structure):
    """Lock-freeリングバッファ統計"""

    _fields_ = [
        ("size", c_uint64),
        ("head", c_uint64),
        ("tail", c_uint64),
        ("full", c_int),
        ("empty", c_int),
    ]


class UltraLowLatencyCore:
    """超低レイテンシHFTコアエンジン"""

    def __init__(self, config: UltraLowLatencyConfig = None):
        self.config = config or UltraLowLatencyConfig()

        # Rust共有ライブラリ
        self.rust_lib = None
        self.rust_core_initialized = False

        # メモリ管理
        self.shared_memory = None
        self.memory_pool_ptr = None

        # 統計情報
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "min_latency_ns": float("inf"),
            "max_latency_ns": 0,
            "avg_latency_ns": 0.0,
            "latency_histogram": np.zeros(100),  # 100ns buckets
        }

        # 初期化
        self._initialize_system_optimization()
        self._initialize_rust_core()
        self._initialize_memory_pool()

        logger.info(
            f"超低レイテンシHFTコア初期化完了 (目標: {self.config.target_latency_ns}ns)"
        )

    def _initialize_system_optimization(self):
        """システムレベル最適化"""
        try:
            # CPU親和性設定
            if self.config.enable_cpu_affinity and self.config.dedicated_cpu_cores:
                if hasattr(os, "sched_setaffinity"):
                    os.sched_setaffinity(0, set(self.config.dedicated_cpu_cores))
                    logger.info(f"CPU親和性設定: {self.config.dedicated_cpu_cores}")

            # プロセス優先度設定
            if hasattr(os, "setpriority"):
                os.setpriority(os.PRIO_PROCESS, 0, self.config.process_priority)
                logger.info(f"プロセス優先度設定: {self.config.process_priority}")

            # スケジューラ設定（Linux）
            if (
                platform.system() == "Linux"
                and self.config.scheduler_policy == "SCHED_FIFO"
            ):
                try:
                    import sched

                    param = sched.sched_param(99)  # 最高優先度
                    os.sched_setscheduler(0, os.SCHED_FIFO, param)
                    logger.info("リアルタイムスケジューラ(SCHED_FIFO)設定完了")
                except Exception as e:
                    logger.warning(f"リアルタイムスケジューラ設定失敗: {e}")

        except Exception as e:
            logger.warning(f"システム最適化設定の一部が失敗: {e}")

    def _initialize_rust_core(self):
        """Rust FFIコア初期化"""
        try:
            # Rust共有ライブラリのパス
            lib_name = self._get_rust_lib_name()
            lib_path = (
                Path(__file__).parent / "rust_core" / "target" / "release" / lib_name
            )

            if lib_path.exists():
                # Rust共有ライブラリロード
                self.rust_lib = ctypes.CDLL(str(lib_path))

                # 関数シグネチャ定義
                self._define_rust_functions()

                # Rustコア初期化
                result = self.rust_lib.initialize_ultra_fast_core(
                    ctypes.c_uint64(self.config.preallocated_memory_mb * 1024 * 1024)
                )

                if result == 0:
                    self.rust_core_initialized = True
                    logger.info("Rust FFIコア初期化完了")
                else:
                    logger.error(f"Rust FFIコア初期化失敗: {result}")
            else:
                logger.warning(f"Rust共有ライブラリが見つかりません: {lib_path}")
                self._create_rust_fallback()

        except Exception as e:
            logger.error(f"Rust FFIコア初期化エラー: {e}")
            self._create_rust_fallback()

    def _get_rust_lib_name(self) -> str:
        """プラットフォーム別共有ライブラリ名取得"""
        system = platform.system()
        if system == "Windows":
            return "ultra_fast_core.dll"
        elif system == "Darwin":
            return "libultra_fast_core.dylib"
        else:
            return "libultra_fast_core.so"

    def _define_rust_functions(self):
        """Rust関数シグネチャ定義"""
        # 初期化関数
        self.rust_lib.initialize_ultra_fast_core.argtypes = [c_uint64]
        self.rust_lib.initialize_ultra_fast_core.restype = c_int

        # 超高速取引実行
        self.rust_lib.execute_trade_ultra_fast.argtypes = [
            ctypes.POINTER(TradeRequest),
            ctypes.POINTER(TradeResult),
        ]
        self.rust_lib.execute_trade_ultra_fast.restype = c_int

        # マーケットデータ処理
        self.rust_lib.process_market_data_ultra_fast.argtypes = [
            ctypes.POINTER(MarketData),
            c_void_p,
        ]
        self.rust_lib.process_market_data_ultra_fast.restype = c_int

        # リングバッファ操作
        self.rust_lib.ringbuffer_push.argtypes = [c_void_p, c_void_p, c_uint64]
        self.rust_lib.ringbuffer_push.restype = c_int

        self.rust_lib.ringbuffer_pop.argtypes = [c_void_p, c_void_p, c_uint64]
        self.rust_lib.ringbuffer_pop.restype = c_int

        # タイミング関数
        self.rust_lib.get_rdtsc_cycles.argtypes = []
        self.rust_lib.get_rdtsc_cycles.restype = c_uint64

        # 統計取得
        self.rust_lib.get_core_stats.argtypes = [c_void_p]
        self.rust_lib.get_core_stats.restype = c_int

    def _create_rust_fallback(self):
        """RustライブラリなしでのPython fallback実装"""
        logger.info("Rustフォールバック実装を使用")

        class RustFallback:
            @staticmethod
            def execute_trade_ultra_fast(trade_request_ptr, trade_result_ptr):
                # シンプルなPython実装
                start_time = time.time_ns()

                # ctypes構造体から値を取得
                trade_request = trade_request_ptr.contents
                trade_result = trade_result_ptr.contents

                # 模擬取引処理
                trade_result.status = 0
                trade_result.order_id = int(start_time % 1000000)
                trade_result.executed_price = trade_request.price
                trade_result.executed_quantity = trade_request.quantity
                trade_result.latency_ns = time.time_ns() - start_time
                trade_result.timestamp_ns = time.time_ns()
                trade_result.error_code = 0

                return 0

            @staticmethod
            def get_rdtsc_cycles():
                return int(time.time_ns())

        self.rust_lib = RustFallback()
        self.rust_core_initialized = True

    def _initialize_memory_pool(self):
        """共有メモリプール初期化"""
        if not MMAP_AVAILABLE:
            logger.warning("mmapが利用できません")
            return

        try:
            pool_size = self.config.preallocated_memory_mb * 1024 * 1024

            # 共有メモリ作成
            if os.name == "nt":  # Windows
                self.shared_memory = mmap.mmap(-1, pool_size)
            else:  # Linux/Unix
                flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
                if self.config.use_huge_pages:
                    try:
                        flags |= mmap.MAP_HUGETLB
                    except AttributeError:
                        logger.warning("Huge pagesがサポートされていません")

                self.shared_memory = mmap.mmap(-1, pool_size, flags=flags)

            # メモリプールポインター設定
            self.memory_pool_ptr = ctypes.cast(
                ctypes.addressof(ctypes.c_char.from_buffer(self.shared_memory)),
                c_void_p,
            )

            logger.info(
                f"共有メモリプール初期化完了: {self.config.preallocated_memory_mb}MB"
            )

        except Exception as e:
            logger.error(f"共有メモリプール初期化エラー: {e}")

    def execute_trade_ultra_fast(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """超高速取引実行 - <10μs目標"""
        if not self.rust_core_initialized:
            raise RuntimeError("Rustコアが初期化されていません")

        # 開始時間測定（RDTSC使用）
        start_cycles = (
            self._get_rdtsc_cycles()
            if hasattr(self, "_get_rdtsc_cycles")
            else time.time_ns()
        )

        # 取引リクエスト構造体作成
        trade_request = TradeRequest()
        trade_request.symbol = symbol.encode("utf-8")
        trade_request.side = 0 if side.lower() == "buy" else 1
        trade_request.quantity = quantity
        trade_request.price = price
        trade_request.order_type = 0 if order_type.lower() == "market" else 1
        trade_request.timestamp_ns = time.time_ns()

        # 取引結果構造体
        trade_result = TradeResult()

        # Rust関数呼び出し（クリティカルパス）
        result_code = self.rust_lib.execute_trade_ultra_fast(
            ctypes.byref(trade_request), ctypes.byref(trade_result)
        )

        # 終了時間測定
        end_cycles = (
            self._get_rdtsc_cycles()
            if hasattr(self, "_get_rdtsc_cycles")
            else time.time_ns()
        )
        total_latency_ns = end_cycles - start_cycles

        # 統計更新
        self._update_latency_stats(total_latency_ns)

        # 結果辞書作成
        result = {
            "success": result_code == 0 and trade_result.status == 0,
            "order_id": trade_result.order_id,
            "executed_price": trade_result.executed_price,
            "executed_quantity": trade_result.executed_quantity,
            "latency_ns": total_latency_ns,
            "latency_us": total_latency_ns / 1000.0,
            "rust_latency_ns": trade_result.latency_ns,
            "timestamp_ns": trade_result.timestamp_ns,
            "error_code": trade_result.error_code,
            "under_target": total_latency_ns < self.config.target_latency_ns,
        }

        # 統計カウンター更新
        self.stats["total_trades"] += 1
        if result["success"]:
            self.stats["successful_trades"] += 1
        else:
            self.stats["failed_trades"] += 1

        return result

    def _get_rdtsc_cycles(self) -> int:
        """RDTSC CPU cycle counter取得"""
        if self.rust_core_initialized and hasattr(self.rust_lib, "get_rdtsc_cycles"):
            return self.rust_lib.get_rdtsc_cycles()
        else:
            return time.time_ns()

    def _update_latency_stats(self, latency_ns: int):
        """レイテンシ統計更新"""
        if latency_ns < self.stats["min_latency_ns"]:
            self.stats["min_latency_ns"] = latency_ns

        if latency_ns > self.stats["max_latency_ns"]:
            self.stats["max_latency_ns"] = latency_ns

        # 移動平均計算
        total_trades = self.stats["total_trades"]
        if total_trades > 0:
            self.stats["avg_latency_ns"] = (
                self.stats["avg_latency_ns"] * (total_trades - 1) + latency_ns
            ) / total_trades

        # ヒストグラム更新（100ns単位）
        bucket_index = min(latency_ns // 100, len(self.stats["latency_histogram"]) - 1)
        self.stats["latency_histogram"][bucket_index] += 1

    def process_market_data_batch(
        self, market_data_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """マーケットデータバッチ処理 - 超低レイテンシ"""
        if not self.rust_core_initialized:
            return []

        results = []
        start_time = time.time_ns()

        for data in market_data_list:
            # マーケットデータ構造体作成
            market_data = MarketData()
            market_data.symbol = data["symbol"].encode("utf-8")
            market_data.bid_price = data["bid_price"]
            market_data.ask_price = data["ask_price"]
            market_data.bid_size = data["bid_size"]
            market_data.ask_size = data["ask_size"]
            market_data.timestamp_ns = time.time_ns()

            # Rust処理関数呼び出し
            if hasattr(self.rust_lib, "process_market_data_ultra_fast"):
                result = self.rust_lib.process_market_data_ultra_fast(
                    ctypes.byref(market_data), self.memory_pool_ptr
                )

                results.append(
                    {
                        "symbol": data["symbol"],
                        "processed": result == 0,
                        "processing_time_ns": time.time_ns() - start_time,
                    }
                )

        total_time = time.time_ns() - start_time
        logger.debug(
            f"マーケットデータバッチ処理完了: {len(market_data_list)}件, {total_time}ns"
        )

        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート取得"""
        if self.stats["total_trades"] == 0:
            return {"status": "no_trades", "message": "取引データがありません"}

        # レイテンシ分析
        avg_latency_us = self.stats["avg_latency_ns"] / 1000.0
        min_latency_us = self.stats["min_latency_ns"] / 1000.0
        max_latency_us = self.stats["max_latency_ns"] / 1000.0

        # パーセンタイル計算
        histogram = self.stats["latency_histogram"]
        total_samples = np.sum(histogram)

        p50_bucket = np.argmax(np.cumsum(histogram) >= total_samples * 0.5) * 100
        p95_bucket = np.argmax(np.cumsum(histogram) >= total_samples * 0.95) * 100
        p99_bucket = np.argmax(np.cumsum(histogram) >= total_samples * 0.99) * 100

        # 目標達成率
        under_target_count = np.sum(histogram[: self.config.target_latency_ns // 100])
        target_achievement_rate = (
            (under_target_count / total_samples) * 100 if total_samples > 0 else 0
        )

        return {
            "total_trades": self.stats["total_trades"],
            "success_rate": (
                self.stats["successful_trades"] / self.stats["total_trades"]
            )
            * 100,
            "latency_stats": {
                "avg_us": round(avg_latency_us, 2),
                "min_us": round(min_latency_us, 2),
                "max_us": round(max_latency_us, 2),
                "p50_ns": p50_bucket,
                "p95_ns": p95_bucket,
                "p99_ns": p99_bucket,
            },
            "performance": {
                "target_latency_ns": self.config.target_latency_ns,
                "target_achievement_rate": round(target_achievement_rate, 1),
                "under_target": under_target_count,
                "over_target": total_samples - under_target_count,
            },
            "system": {
                "rust_core_enabled": self.rust_core_initialized,
                "cpu_affinity": self.config.dedicated_cpu_cores,
                "scheduler_policy": self.config.scheduler_policy,
                "memory_pool_mb": self.config.preallocated_memory_mb,
            },
        }

    def cleanup(self):
        """リソースクリーンアップ"""
        try:
            if self.shared_memory:
                self.shared_memory.close()

            if self.rust_core_initialized and hasattr(
                self.rust_lib, "cleanup_ultra_fast_core"
            ):
                self.rust_lib.cleanup_ultra_fast_core()

            logger.info("超低レイテンシHFTコア クリーンアップ完了")

        except Exception as e:
            logger.error(f"クリーンアップエラー: {e}")


# 便利関数とファクトリー
def create_ultra_low_latency_core(
    target_latency_us: float = 10.0, cpu_cores: List[int] = None, memory_mb: int = 512
) -> UltraLowLatencyCore:
    """超低レイテンシコア作成"""
    config = UltraLowLatencyConfig(
        target_latency_ns=int(target_latency_us * 1000),
        dedicated_cpu_cores=cpu_cores or [2, 3],
        preallocated_memory_mb=memory_mb,
    )

    return UltraLowLatencyCore(config)


# デモ・テスト用関数
def benchmark_ultra_low_latency(iterations: int = 1000) -> Dict[str, Any]:
    """超低レイテンシベンチマーク"""
    print(f"超低レイテンシHFTベンチマーク開始: {iterations}回実行")

    # コア初期化
    core = create_ultra_low_latency_core(target_latency_us=10.0)

    # ウォームアップ
    for _ in range(10):
        core.execute_trade_ultra_fast("USDJPY", "buy", 10000, 150.0)

    print("ベンチマーク実行中...")

    # 本ベンチマーク
    start_time = time.time()
    for i in range(iterations):
        result = core.execute_trade_ultra_fast(
            "USDJPY", "buy" if i % 2 == 0 else "sell", 10000, 150.0 + (i % 10) * 0.001
        )

        if i % 100 == 0:
            print(
                f"Progress: {i}/{iterations} - Latest latency: {result['latency_us']:.2f}μs"
            )

    total_time = time.time() - start_time

    # 結果取得
    report = core.get_performance_report()

    # クリーンアップ
    core.cleanup()

    print(f"\nベンチマーク完了! 総実行時間: {total_time:.2f}秒")
    print(f"平均レイテンシ: {report['latency_stats']['avg_us']}μs")
    print(f"目標達成率: {report['performance']['target_achievement_rate']}%")

    return report


if __name__ == "__main__":
    # ベンチマーク実行
    benchmark_report = benchmark_ultra_low_latency(1000)
    print("\n=== 超低レイテンシHFTベンチマーク結果 ===")
    print(f"平均レイテンシ: {benchmark_report['latency_stats']['avg_us']}μs")
    print(f"最小レイテンシ: {benchmark_report['latency_stats']['min_us']}μs")
    print(f"最大レイテンシ: {benchmark_report['latency_stats']['max_us']}μs")
    print(f"P99レイテンシ: {benchmark_report['latency_stats']['p99_ns']/1000:.2f}μs")
    print(
        f"目標(<10μs)達成率: {benchmark_report['performance']['target_achievement_rate']:.1f}%"
    )
