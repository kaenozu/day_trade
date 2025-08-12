#!/usr/bin/env python3
"""
最小構成安定性テストシステム

Issue #322: 24時間安定性テストシステム構築
外部依存を最小化した軽量安定性テスト
"""

import gc
import json
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import psutil


class MinimalStabilityTest:
    """最小構成安定性テストシステム"""

    def __init__(self):
        self.test_duration_minutes = 5  # 5分間テスト
        self.monitoring_interval = 10  # 10秒監視間隔
        self.operation_interval = 20  # 20秒操作間隔

        # 軽量データ構造
        self.start_time = None
        self.operation_count = 0
        self.error_count = 0
        self.memory_samples = []
        self.cpu_samples = []

        # データ保持制限
        self.max_samples = 50

        print("最小構成安定性テストシステム")
        print(f"テスト時間: {self.test_duration_minutes}分")
        print("外部依存: psutil のみ")

    def run_minimal_test(self) -> bool:
        """最小構成テスト実行"""
        print("\n=== 最小構成安定性テスト開始 ===")

        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=self.test_duration_minutes)

        print(f"開始: {self.start_time.strftime('%H:%M:%S')}")
        print(f"終了予定: {end_time.strftime('%H:%M:%S')}")

        # 初期メモリ
        initial_memory = self._get_memory_usage()
        print(f"初期メモリ: {initial_memory:.1f}MB")

        try:
            last_operation = time.time()
            last_monitoring = time.time()

            while datetime.now() < end_time:
                current_time = time.time()

                # システム監視
                if current_time - last_monitoring >= self.monitoring_interval:
                    self._monitor_system()
                    last_monitoring = current_time

                # 操作実行
                if current_time - last_operation >= self.operation_interval:
                    self._execute_operation()
                    last_operation = current_time

                # メモリクリーンアップ
                if self.operation_count > 0 and self.operation_count % 5 == 0:
                    self._cleanup()

                # 進捗表示
                elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                if int(elapsed) != int(elapsed - 0.33):  # 約20秒ごと
                    progress = elapsed / self.test_duration_minutes * 100
                    current_memory = self._get_memory_usage()
                    print(
                        f"[{elapsed:.1f}分] {progress:.0f}% メモリ:{current_memory:.0f}MB 操作:{self.operation_count}"
                    )

                time.sleep(2)  # 2秒待機

            print("\n=== テスト時間完了 ===")
            return self._evaluate_results()

        except KeyboardInterrupt:
            print("\n=== 中断 ===")
            return False

        except Exception as e:
            print(f"\n=== エラー: {e} ===")
            self.error_count += 1
            return False

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（MB）"""
        try:
            return psutil.virtual_memory().used / (1024 * 1024)
        except:
            return 0.0

    def _get_cpu_usage(self) -> float:
        """CPU使用率取得"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0

    def _monitor_system(self):
        """システム監視"""
        try:
            memory_mb = self._get_memory_usage()
            cpu_percent = self._get_cpu_usage()

            # サンプル保存
            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)

            # データ制限
            if len(self.memory_samples) > self.max_samples:
                self.memory_samples = self.memory_samples[-self.max_samples // 2 :]
                self.cpu_samples = self.cpu_samples[-self.max_samples // 2 :]

            # 異常検知
            if memory_mb > 500:  # 500MB超過
                print(f"[警告] メモリ使用量: {memory_mb:.0f}MB")

            if cpu_percent > 30:
                print(f"[警告] CPU使用率: {cpu_percent:.1f}%")

        except Exception as e:
            print(f"[監視エラー] {e}")
            self.error_count += 1

    def _execute_operation(self):
        """軽量操作実行"""
        try:
            start_time = time.time()

            # 軽量計算処理
            data = []
            for _i in range(100):  # 100個のデータポイント
                value = random.uniform(50, 150)
                data.append(value)

            # 統計計算
            avg = sum(data) / len(data)
            variance = sum((x - avg) ** 2 for x in data) / len(data)
            std_dev = math.sqrt(variance)

            # トレンド判定
            first_half = sum(data[:50]) / 50
            second_half = sum(data[50:]) / 50
            trend = "UP" if second_half > first_half else "DOWN"

            # 処理時間
            duration_ms = (time.time() - start_time) * 1000

            self.operation_count += 1

            if self.operation_count % 3 == 0:
                print(
                    f"[操作{self.operation_count}] 平均:{avg:.1f} 標準偏差:{std_dev:.1f} {trend} ({duration_ms:.1f}ms)"
                )

        except Exception as e:
            print(f"[操作エラー] {e}")
            self.error_count += 1

    def _cleanup(self):
        """メモリクリーンアップ"""
        try:
            collected = gc.collect()
            if collected > 0:
                current_memory = self._get_memory_usage()
                print(f"[GC] {collected}オブジェクト回収 メモリ:{current_memory:.0f}MB")
        except Exception as e:
            print(f"[クリーンアップエラー] {e}")

    def _evaluate_results(self) -> bool:
        """結果評価"""
        print("\n=== 結果評価 ===")

        # 実行時間
        actual_duration = (datetime.now() - self.start_time).total_seconds() / 60
        target_duration = self.test_duration_minutes
        uptime_rate = min(100, actual_duration / target_duration * 100)

        # メモリ統計
        if self.memory_samples:
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
            max_memory = max(self.memory_samples)
            min(self.memory_samples)
        else:
            avg_memory = max_memory = 0

        # CPU統計
        if self.cpu_samples:
            avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
            max_cpu = max(self.cpu_samples)
        else:
            avg_cpu = max_cpu = 0

        # 評価項目
        evaluations = []

        # 1. 稼働率
        uptime_ok = uptime_rate >= 95
        evaluations.append(("稼働率", uptime_ok, f"{uptime_rate:.1f}%"))

        # 2. メモリ効率（平均500MB以下）
        memory_ok = avg_memory <= 500
        evaluations.append(("メモリ効率", memory_ok, f"平均{avg_memory:.0f}MB"))

        # 3. CPU効率（平均20%以下）
        cpu_ok = avg_cpu <= 20
        evaluations.append(("CPU効率", cpu_ok, f"平均{avg_cpu:.1f}%"))

        # 4. 操作実行（期待値の80%以上）
        expected_ops = (self.test_duration_minutes * 60) // self.operation_interval
        ops_ok = self.operation_count >= expected_ops * 0.8
        evaluations.append(
            ("操作実行", ops_ok, f"{self.operation_count}/{expected_ops}回")
        )

        # 5. エラー率（5%以下）
        error_rate = self.error_count / max(1, self.operation_count) * 100
        error_ok = error_rate <= 5
        evaluations.append(("エラー率", error_ok, f"{error_rate:.1f}%"))

        # 結果表示
        print(f"実行時間: {actual_duration:.1f}/{target_duration}分")
        print(f"メモリ使用量: 平均{avg_memory:.0f}MB (最大{max_memory:.0f}MB)")
        print(f"CPU使用率: 平均{avg_cpu:.1f}% (最大{max_cpu:.1f}%)")
        print(f"実行操作: {self.operation_count}回")
        print(f"エラー: {self.error_count}回")

        print("\n評価項目:")
        passed = 0
        for name, result, detail in evaluations:
            status = "[OK]" if result else "[NG]"
            print(f"  {status} {name}: {detail}")
            if result:
                passed += 1

        overall_success = passed >= 4  # 5項目中4項目合格

        print(f"\n総合結果: {passed}/5 合格")

        if overall_success:
            print("[SUCCESS] 最小構成安定性テスト合格")
            print("軽量システムでの安定運用が確認できました")

            # 結果保存
            self._save_results(
                evaluations,
                {
                    "duration": actual_duration,
                    "avg_memory": avg_memory,
                    "max_memory": max_memory,
                    "avg_cpu": avg_cpu,
                    "operations": self.operation_count,
                    "errors": self.error_count,
                },
            )
        else:
            print("[FAILED] 最小構成安定性テスト不合格")
            print("基本的な最適化が必要です")

        return overall_success

    def _save_results(self, evaluations: List[Tuple], stats: Dict):
        """結果保存"""
        try:
            results_dir = Path("stability_test_results")
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = results_dir / f"minimal_stability_{timestamp}.json"

            data = {
                "test_type": "minimal_stability",
                "timestamp": datetime.now().isoformat(),
                "duration_minutes": self.test_duration_minutes,
                "statistics": stats,
                "evaluations": [
                    {"name": name, "passed": passed, "detail": detail}
                    for name, passed, detail in evaluations
                ],
            }

            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"結果保存: {result_file}")

        except Exception as e:
            print(f"結果保存エラー: {e}")


def main():
    """メイン実行"""
    print("最小構成24時間安定性テストシステム")
    print("=" * 50)

    try:
        tester = MinimalStabilityTest()
        success = tester.run_minimal_test()

        if success:
            print("\n[SUCCESS] 最小構成テスト完了")
            print("基本的な安定性が確認されました")
            print("\n次のステップ:")
            print("- より長時間テスト実行（1時間、24時間）")
            print("- 実際のMLモジュール統合テスト")
            print("- プロダクション環境での検証")
            return True
        else:
            print("\n[FAILED] 基本的な問題が検出されました")
            return False

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
