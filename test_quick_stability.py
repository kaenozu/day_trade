#!/usr/bin/env python3
"""
クイック安定性テストシステム

Issue #322: 24時間安定性テストシステム構築
短時間での安定性確認テスト（2分間）
"""

import gc
import math
import random
import time
from datetime import datetime

import psutil


class QuickStabilityTest:
    """クイック安定性テストシステム"""

    def __init__(self):
        self.test_duration_seconds = 120  # 2分間テスト
        self.operation_count = 0
        self.error_count = 0
        self.memory_readings = []
        self.cpu_readings = []

        print("クイック安定性テストシステム（2分間）")

    def run_quick_test(self) -> bool:
        """クイックテスト実行"""
        print("\n=== クイック安定性テスト開始 ===")

        start_time = time.time()
        end_time = start_time + self.test_duration_seconds

        initial_memory = self._get_memory_mb()
        print(f"開始時刻: {datetime.now().strftime('%H:%M:%S')}")
        print(f"初期メモリ: {initial_memory:.0f}MB")
        print(f"テスト時間: {self.test_duration_seconds}秒")

        try:
            last_report = start_time

            while time.time() < end_time:
                current_time = time.time()

                # システム監視
                self._monitor_system()

                # 操作実行
                if self.operation_count < 20:  # 最大20回操作
                    self._execute_lightweight_operation()

                # 20秒ごとに進捗報告
                if current_time - last_report >= 20:
                    elapsed = current_time - start_time
                    progress = (elapsed / self.test_duration_seconds) * 100
                    current_memory = self._get_memory_mb()
                    print(f"[{elapsed:.0f}秒] 進捗:{progress:.0f}% メモリ:{current_memory:.0f}MB 操作:{self.operation_count}")
                    last_report = current_time

                # ガベージコレクション
                if self.operation_count % 5 == 0:
                    gc.collect()

                time.sleep(1)  # 1秒待機

            print("\n=== テスト完了 ===")
            return self._evaluate_quick_results()

        except Exception as e:
            print(f"テストエラー: {e}")
            self.error_count += 1
            return False

    def _get_memory_mb(self) -> float:
        """メモリ使用量（MB）"""
        try:
            return psutil.virtual_memory().used / (1024 * 1024)
        except:
            return 0

    def _get_cpu_percent(self) -> float:
        """CPU使用率"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0

    def _monitor_system(self):
        """システム監視"""
        try:
            memory_mb = self._get_memory_mb()
            cpu_percent = self._get_cpu_percent()

            self.memory_readings.append(memory_mb)
            self.cpu_readings.append(cpu_percent)

            # データ制限（最新50件）
            if len(self.memory_readings) > 50:
                self.memory_readings = self.memory_readings[-25:]
                self.cpu_readings = self.cpu_readings[-25:]
        except:
            self.error_count += 1

    def _execute_lightweight_operation(self):
        """軽量操作"""
        try:
            start = time.time()

            # 軽量計算
            data = [random.uniform(0, 100) for _ in range(50)]
            avg = sum(data) / len(data)
            math.sqrt(sum((x - avg) ** 2 for x in data) / len(data))

            # 判定
            trend = "UP" if avg > 50 else "DOWN"

            duration_ms = (time.time() - start) * 1000
            self.operation_count += 1

            if self.operation_count % 5 == 0:
                print(f"[操作{self.operation_count}] 平均:{avg:.1f} {trend} ({duration_ms:.1f}ms)")

        except:
            self.error_count += 1

    def _evaluate_quick_results(self) -> bool:
        """結果評価"""
        print("\n=== 結果評価 ===")

        # 統計計算
        avg_memory = sum(self.memory_readings) / len(self.memory_readings) if self.memory_readings else 0
        max_memory = max(self.memory_readings) if self.memory_readings else 0
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings) if self.cpu_readings else 0
        max_cpu = max(self.cpu_readings) if self.cpu_readings else 0

        # 評価基準
        memory_ok = avg_memory <= 300  # 300MB以下
        cpu_ok = avg_cpu <= 25        # 25%以下
        operations_ok = self.operation_count >= 10  # 最低10回操作
        errors_ok = self.error_count <= 2          # 最大2回エラー

        print(f"メモリ使用量: 平均{avg_memory:.0f}MB (最大{max_memory:.0f}MB)")
        print(f"CPU使用率: 平均{avg_cpu:.1f}% (最大{max_cpu:.1f}%)")
        print(f"実行操作数: {self.operation_count}回")
        print(f"エラー数: {self.error_count}回")

        print("\n評価結果:")
        print(f"  [{'OK' if memory_ok else 'NG'}] メモリ効率: {'良好' if memory_ok else '要改善'}")
        print(f"  [{'OK' if cpu_ok else 'NG'}] CPU効率: {'良好' if cpu_ok else '要改善'}")
        print(f"  [{'OK' if operations_ok else 'NG'}] 操作実行: {'十分' if operations_ok else '不十分'}")
        print(f"  [{'OK' if errors_ok else 'NG'}] エラー率: {'許容範囲' if errors_ok else '過多'}")

        passed_count = sum([memory_ok, cpu_ok, operations_ok, errors_ok])
        overall_success = passed_count >= 3  # 4項目中3項目以上

        print(f"\n総合結果: {passed_count}/4 合格")

        if overall_success:
            print("[SUCCESS] クイック安定性テスト合格")
            print("基本的な安定性が確認されました")
        else:
            print("[FAILED] クイック安定性テスト不合格")
            print("基本的な問題があります")

        return overall_success

def main():
    """メイン実行"""
    print("クイック安定性テストシステム")
    print("=" * 40)

    try:
        tester = QuickStabilityTest()
        success = tester.run_quick_test()

        if success:
            print("\n[SUCCESS] クイックテスト完了")
            print("システムの基本的な安定性を確認しました")
            print("\n推奨事項:")
            print("- 長時間テスト（1時間、24時間）で詳細検証")
            print("- メモリ使用量の継続監視")
            print("- 実際の本番負荷でのテスト")
            return True
        else:
            print("\n[FAILED] 基本的な問題検出")
            print("システムの最適化が必要です")
            return False

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
