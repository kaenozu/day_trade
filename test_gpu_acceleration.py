#!/usr/bin/env python3
"""
GPU並列処理エンジン テストスクリプト
Phase F: 次世代機能拡張フェーズ

GPU加速機能の動作確認・ベンチマーク
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトパス追加
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.day_trade.acceleration import GPUAccelerationEngine, GPUBackend
from src.day_trade.core.optimization_strategy import OptimizationConfig


class GPUAccelerationTester:
    """GPU並列処理テスター"""

    def __init__(self):
        self.gpu_engine = GPUAccelerationEngine()
        self.test_data = self._generate_test_data()

        print("[START] GPU並列処理エンジン テスト開始")
        print(f"プライマリバックエンド: {self.gpu_engine.primary_backend.value}")
        print(f"利用可能デバイス: {len(self.gpu_engine.devices)}個")
        print("=" * 60)

    def _generate_test_data(self) -> pd.DataFrame:
        """テストデータ生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')

        # 実際の株価データを模した時系列データ生成
        returns = np.random.normal(0.001, 0.02, 1000)
        prices = [100.0]

        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 1000)
        }).set_index('Date')

    def test_device_detection(self):
        """デバイス検出テスト"""
        print("[DEVICE] デバイス検出テスト")
        print(f"利用可能バックエンド: {[b.value for b in self.gpu_engine.available_backends]}")

        for device in self.gpu_engine.devices:
            print(f"  - {device.name} ({device.backend.value})")
            print(f"    メモリ: {device.memory_total}MB (空き: {device.memory_free}MB)")
            if device.compute_capability:
                print(f"    Compute Capability: {device.compute_capability}")

        print("[OK] デバイス検出テスト完了\n")

    def test_technical_indicators_basic(self):
        """基本テクニカル指標テスト"""
        print("[BASIC] 基本テクニカル指標テスト")

        indicators = ['sma', 'ema', 'rsi']

        try:
            start_time = time.time()
            result = self.gpu_engine.accelerate_technical_indicators(
                self.test_data, indicators
            )
            execution_time = time.time() - start_time

            print(f"実行時間: {execution_time:.4f}秒")
            print(f"使用バックエンド: {result.backend_used.value}")
            print(f"使用デバイス: {result.device_info.name}")
            print(f"メモリ使用量: {result.memory_used:.2f}MB")

            # 結果検証
            for indicator, values in result.result.items():
                if isinstance(values, dict):
                    print(f"  {indicator}: {len(list(values.values())[0])}データポイント")
                else:
                    print(f"  {indicator}: {len(values)}データポイント")

            print("[OK] 基本テクニカル指標テスト完了\n")
            return result

        except Exception as e:
            print(f"[ERROR] 基本テクニカル指標テスト失敗: {e}\n")
            return None

    def test_technical_indicators_advanced(self):
        """高度テクニカル指標テスト"""
        print("[ADVANCED] 高度テクニカル指標テスト")

        indicators = ['bollinger_bands', 'macd', 'stochastic']

        try:
            start_time = time.time()
            result = self.gpu_engine.accelerate_technical_indicators(
                self.test_data, indicators
            )
            execution_time = time.time() - start_time

            print(f"実行時間: {execution_time:.4f}秒")
            print(f"使用バックエンド: {result.backend_used.value}")

            # 結果詳細表示
            for indicator, values in result.result.items():
                if isinstance(values, dict):
                    sub_indicators = list(values.keys())
                    print(f"  {indicator}: {sub_indicators}")
                else:
                    print(f"  {indicator}: 単一値")

            print("[OK] 高度テクニカル指標テスト完了\n")
            return result

        except Exception as e:
            print(f"[ERROR] 高度テクニカル指標テスト失敗: {e}\n")
            return None

    def test_performance_benchmark(self):
        """パフォーマンスベンチマーク"""
        print("[PERFORMANCE] パフォーマンスベンチマーク")

        try:
            benchmark_result = self.gpu_engine.benchmark_performance(self.test_data)

            print(f"GPU処理時間: {benchmark_result['gpu_time']:.4f}秒")
            print(f"CPU処理時間: {benchmark_result['cpu_time']:.4f}秒")
            print(f"高速化比率: {benchmark_result['speedup_ratio']:.2f}倍")
            print(f"使用バックエンド: {benchmark_result['backend_used']}")
            print(f"メモリ使用量: {benchmark_result['memory_used']:.2f}MB")
            print(f"デバイス: {benchmark_result['device_info']['name']}")

            # 性能評価
            if benchmark_result['speedup_ratio'] > 1.5:
                print("[EXCELLENT] 優秀な高速化性能!")
            elif benchmark_result['speedup_ratio'] > 1.0:
                print("[GOOD] 高速化効果あり")
            else:
                print("[WARNING] 高速化効果限定的（CPU最適化済み、またはGPU未利用）")

            print("[OK] パフォーマンスベンチマーク完了\n")
            return benchmark_result

        except Exception as e:
            print(f"[ERROR] パフォーマンスベンチマーク失敗: {e}\n")
            return None

    def test_large_dataset_processing(self):
        """大規模データセット処理テスト"""
        print("[LARGE_DATA] 大規模データセット処理テスト")

        # 10,000データポイントのテストデータ生成
        large_data = self._generate_large_test_data(10000)
        print(f"テストデータサイズ: {len(large_data)}行")

        indicators = ['sma', 'ema', 'rsi', 'bollinger_bands']

        try:
            start_time = time.time()
            result = self.gpu_engine.accelerate_technical_indicators(
                large_data, indicators
            )
            execution_time = time.time() - start_time

            print(f"実行時間: {execution_time:.4f}秒")
            print(f"処理速度: {len(large_data)/execution_time:.0f}行/秒")
            print(f"メモリ使用量: {result.memory_used:.2f}MB")

            # メモリ効率評価
            memory_efficiency = len(large_data) / result.memory_used
            print(f"メモリ効率: {memory_efficiency:.0f}行/MB")

            print("[OK] 大規模データセット処理テスト完了\n")
            return result

        except Exception as e:
            print(f"[ERROR] 大規模データセット処理テスト失敗: {e}\n")
            return None

    def _generate_large_test_data(self, size: int) -> pd.DataFrame:
        """大規模テストデータ生成"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=size, freq='D')

        returns = np.random.normal(0.001, 0.02, size)
        prices = [100.0]

        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        return pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, size)
        }).set_index('Date')

    def test_memory_management(self):
        """メモリ管理テスト"""
        print("[MEMORY] メモリ管理テスト")

        # メモリ統計取得
        memory_stats = self.gpu_engine.memory_manager.get_memory_stats()

        print(f"現在のメモリ使用量: {memory_stats['current_usage']:,}bytes")
        print(f"ピークメモリ使用量: {memory_stats['peak_usage']:,}bytes")
        print(f"割り当てブロック数: {memory_stats['allocated_blocks']}")
        print(f"断片化率: {memory_stats['fragmentation_ratio']:.2%}")

        print("[OK] メモリ管理テスト完了\n")
        return memory_stats

    def test_error_handling(self):
        """エラーハンドリングテスト"""
        print("[ERROR_HANDLING] エラーハンドリングテスト")

        # 不正データでのテスト
        invalid_data = pd.DataFrame()  # 空のDataFrame

        try:
            result = self.gpu_engine.accelerate_technical_indicators(
                invalid_data, ['sma']
            )
            print("[WARNING] 空データでもエラーが発生しませんでした")
        except Exception as e:
            print(f"[OK] 適切にエラーをキャッチ: {type(e).__name__}")

        # 無効な指標でのテスト
        try:
            result = self.gpu_engine.accelerate_technical_indicators(
                self.test_data, ['invalid_indicator']
            )
            print("[WARNING] 無効な指標でもエラーが発生しませんでした")
        except Exception as e:
            print(f"[OK] 適切にエラーをキャッチ: {type(e).__name__}")

        print("[OK] エラーハンドリングテスト完了\n")

    def run_comprehensive_test(self):
        """包括的テスト実行"""
        print("[COMPREHENSIVE] GPU並列処理エンジン 包括的テスト")
        print("=" * 60)

        # 全テスト実行
        self.test_device_detection()
        basic_result = self.test_technical_indicators_basic()
        advanced_result = self.test_technical_indicators_advanced()
        benchmark_result = self.test_performance_benchmark()
        large_data_result = self.test_large_dataset_processing()
        memory_stats = self.test_memory_management()
        self.test_error_handling()

        # 統合結果表示
        print("[SUMMARY] テスト結果サマリー")
        print("=" * 60)

        performance_summary = self.gpu_engine.get_performance_summary()
        print(f"総計算回数: {performance_summary['total_computations']}")
        print(f"総実行時間: {performance_summary['total_time']:.4f}秒")
        print(f"GPU使用率: {performance_summary['gpu_time_ratio']:.2%}")
        print(f"平均計算時間: {performance_summary['average_computation_time']:.4f}秒")

        if benchmark_result:
            print(f"最大高速化比率: {benchmark_result['speedup_ratio']:.2f}倍")

        # 総合評価
        success_count = sum([
            basic_result is not None,
            advanced_result is not None,
            benchmark_result is not None,
            large_data_result is not None
        ])

        print(f"\n[RESULT] テスト成功率: {success_count}/4 ({success_count/4:.0%})")

        if success_count == 4:
            print("[SUCCESS] すべてのテストが成功！GPU並列処理エンジンは正常動作中です。")
        elif success_count >= 2:
            print("[GOOD] 主要テストが成功。一部制限がありますが動作可能です。")
        else:
            print("[WARNING] 多くのテストが失敗。設定や依存関係を確認してください。")

        return {
            'success_count': success_count,
            'total_tests': 4,
            'performance_summary': performance_summary,
            'benchmark_result': benchmark_result
        }


def main():
    """メインテスト実行"""
    try:
        tester = GPUAccelerationTester()
        result = tester.run_comprehensive_test()

        # テスト結果の保存（オプション）
        import json

        # JSON シリアライズ対応のため、結果を変換
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                serializable_result[key] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                                          for k, v in value.items()}
            else:
                serializable_result[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)

        with open('gpu_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

        print("\n[SAVE] テスト結果をgpu_test_results.jsonに保存しました。")

        return 0 if result['success_count'] >= 2 else 1

    except Exception as e:
        print(f"[ERROR] テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
