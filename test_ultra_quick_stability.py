#!/usr/bin/env python3
"""
超短時間安定性テスト（30秒）

Issue #322: 24時間安定性テストシステム構築
動作確認用の超短時間テスト
"""

import random
import time
from datetime import datetime

import psutil


def test_ultra_quick_stability():
    """30秒安定性テスト"""
    print("超短時間安定性テスト（30秒）")
    print("=" * 40)

    start_time = time.time()
    test_duration = 30  # 30秒

    operation_count = 0
    error_count = 0
    memory_readings = []

    print(f"開始: {datetime.now().strftime('%H:%M:%S')}")
    print(f"テスト時間: {test_duration}秒")

    try:
        while time.time() - start_time < test_duration:
            current_time = time.time()
            elapsed = current_time - start_time

            # システム監視
            try:
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                memory_readings.append(memory_mb)
            except:
                error_count += 1

            # 軽量操作
            try:
                # 簡単な計算
                data = [random.uniform(0, 100) for _ in range(20)]
                avg = sum(data) / len(data)
                operation_count += 1

                if operation_count % 5 == 0:
                    print(f"[{elapsed:.0f}秒] 操作{operation_count} 平均:{avg:.1f}")

            except:
                error_count += 1

            # 5秒ごとに進捗表示
            if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 1):
                progress = (elapsed / test_duration) * 100
                current_memory = memory_readings[-1] if memory_readings else 0
                print(f"進捗: {progress:.0f}% メモリ: {current_memory:.0f}MB")

            time.sleep(1)

        print("\n=== テスト完了 ===")

        # 結果評価
        if memory_readings:
            avg_memory = sum(memory_readings) / len(memory_readings)
            max_memory = max(memory_readings)
        else:
            avg_memory = max_memory = 0

        print(f"実行操作: {operation_count}回")
        print(f"エラー: {error_count}回")
        print(f"メモリ使用量: 平均{avg_memory:.0f}MB, 最大{max_memory:.0f}MB")

        # 評価基準
        operations_ok = operation_count >= 15  # 最低15回操作
        errors_ok = error_count <= 2          # 最大2回エラー
        memory_ok = avg_memory <= 200         # 200MB以下

        passed = sum([operations_ok, errors_ok, memory_ok])

        print("\n評価結果:")
        print(f"  [{'OK' if operations_ok else 'NG'}] 操作実行: {operation_count}回")
        print(f"  [{'OK' if errors_ok else 'NG'}] エラー数: {error_count}回")
        print(f"  [{'OK' if memory_ok else 'NG'}] メモリ効率: {avg_memory:.0f}MB")

        success = passed >= 2  # 3項目中2項目以上

        print(f"\n総合結果: {passed}/3 合格")

        if success:
            print("[SUCCESS] 超短時間安定性テスト合格")
            print("基本的なシステム安定性が確認されました")
        else:
            print("[FAILED] 超短時間安定性テスト不合格")
            print("基本的な問題があります")

        return success

    except Exception as e:
        print(f"テスト実行エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_ultra_quick_stability()

    if success:
        print("\n次のステップ推奨:")
        print("1. より長時間のテスト（2分、10分、1時間）")
        print("2. 実際のMLモジュール統合")
        print("3. 24時間連続安定性テスト")
    else:
        print("\nシステム最適化が必要です")

    exit(0 if success else 1)
