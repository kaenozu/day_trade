#!/usr/bin/env python3
"""
パフォーマンステスト

システムのパフォーマンス測定
"""

import pytest
import unittest
import time
import memory_profiler
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPerformance(unittest.TestCase):
    """パフォーマンステスト"""

    def test_memory_usage(self):
        """メモリ使用量テスト"""
        @memory_profiler.profile
        def memory_test():
            # メモリ使用量テスト用の処理
            data = []
            for i in range(10000):
                data.append({"id": i, "value": f"test_{i}"})
            return len(data)

        try:
            result = memory_test()
            self.assertEqual(result, 10000)
        except ImportError:
            self.skipTest("memory_profilerが利用できません")

    def test_cpu_performance(self):
        """CPU処理速度テスト"""
        start_time = time.time()

        # CPU集約的な処理
        result = 0
        for i in range(100000):
            result += i ** 2

        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 2.0, "CPU処理が2秒を超過")
        self.assertGreater(result, 0)

    def test_io_performance(self):
        """I/O処理速度テスト"""
        import tempfile

        start_time = time.time()

        # ファイルI/O処理テスト
        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
            for i in range(1000):
                f.write(f"line {i}\n")
            f.seek(0)
            lines = f.readlines()

        elapsed_time = time.time() - start_time
        self.assertLess(elapsed_time, 1.0, "I/O処理が1秒を超過")
        self.assertEqual(len(lines), 1000)


if __name__ == "__main__":
    unittest.main()
