#!/usr/bin/env python3
"""
パフォーマンスモニター

処理のパフォーマンスを監視するためのコンテキストマネージャーを提供します。
"""

import time
from contextlib import contextmanager

import psutil


@contextmanager
def performance_monitor(operation_name: str):
    """パフォーマンス監視コンテキストマネージャー"""
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    print(f">> {operation_name} 開始")

    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory

        print(f"<< {operation_name} 完了")
        print(f"   実行時間: {execution_time:.3f}秒")
        print(f"   メモリ使用量変化: {memory_delta:+.2f}MB")
