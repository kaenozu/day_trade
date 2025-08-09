#!/usr/bin/env python3
"""
TOPIX500分析システム 軽量テスト版
Issue #314: 性能最適化テスト

高速実行に特化したテストスイート
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.day_trade.topix.topix500_analysis_system import (
        PerformanceMetrics,
        TOPIX500AnalysisSystem,
    )
    from src.day_trade.utils.logging_config import get_context_logger
except ImportError as e:
    print(f"インポートエラー: {e}")
    sys.exit(1)

logger = get_context_logger(__name__)


def generate_lightweight_data(symbol_count: int = 10) -> dict:
    """軽量テストデータ生成"""
    stock_data = {}

    for i in range(symbol_count):
        symbol = f"{3000 + i:04d}"

        # 最小限のデータ（30日分）
        dates = pd.date_range(start='2024-07-01', periods=30)
        base_price = 2000 + i * 100

        # 簡単な価格変動
        price_changes = np.random.normal(0, 0.01, 30)  # 1%標準偏差
        cumulative_changes = np.cumprod(1 + price_changes)
        close_prices = base_price * cumulative_changes

        stock_data[symbol] = pd.DataFrame({
            'Open': close_prices * 0.99,
            'High': close_prices * 1.02,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(100000, 1000000, 30),
            'Adj Close': close_prices
        }, index=dates)

    return stock_data


async def test_lightweight_system():
    """軽量システムテスト"""
    print("=== TOPIX500軽量テスト開始 ===")

    # 軽量設定でシステム初期化
    system = TOPIX500AnalysisSystem(
        enable_cache=False,  # キャッシュ無効でシンプル化
        enable_parallel=False,  # 並列処理無効で安定化
        max_concurrent_symbols=5,
        max_concurrent_sectors=2,
        memory_limit_gb=0.5,
        processing_timeout=10,
        batch_size=5
    )

    try:
        # 小規模データテスト（5銘柄）
        print("\n1. 小規模テスト（5銘柄）...")
        start_time = time.time()

        test_data = generate_lightweight_data(5)
        result = await system.analyze_batch_comprehensive(
            stock_data=test_data,
            enable_sector_analysis=True,
            enable_ml_prediction=False  # ML無効で高速化
        )

        elapsed = time.time() - start_time

        # 結果検証
        success_count = len([r for r in result['symbol_results'].values() if r.get('success', False)])
        sector_count = len(result['sector_analysis'])

        print(f"処理時間: {elapsed:.2f}秒")
        print(f"成功銘柄: {success_count}/{len(test_data)}")
        print(f"セクター数: {sector_count}")
        print(f"パフォーマンス要件: {'OK' if elapsed < 3.0 else 'NG'}")

        if elapsed < 3.0 and success_count > 0:
            print("OK 小規模テスト合格")
        else:
            print("NG 小規模テスト失敗")
            return False

        # 中規模データテスト（20銘柄）
        print("\n2. 中規模テスト（20銘柄）...")
        start_time = time.time()

        test_data = generate_lightweight_data(20)
        result = await system.analyze_batch_comprehensive(
            stock_data=test_data,
            enable_sector_analysis=True,
            enable_ml_prediction=False
        )

        elapsed = time.time() - start_time

        # 結果検証
        success_count = len([r for r in result['symbol_results'].values() if r.get('success', False)])
        sector_count = len(result['sector_analysis'])

        print(f"処理時間: {elapsed:.2f}秒")
        print(f"成功銘柄: {success_count}/{len(test_data)}")
        print(f"セクター数: {sector_count}")
        print(f"パフォーマンス要件: {'OK' if elapsed < 8.0 else 'NG'}")

        if elapsed < 8.0 and success_count >= 15:  # 75%以上成功
            print("OK 中規模テスト合格")
        else:
            print("NG 中規模テスト失敗")
            return False

        # パフォーマンスメトリクス確認
        metrics = result['performance_metrics']
        if isinstance(metrics, PerformanceMetrics):
            print("\n性能統計:")
            print(f"  - スループット: {metrics.throughput_symbols_per_second:.1f}銘柄/秒")
            print(f"  - 平均処理時間/銘柄: {metrics.avg_time_per_symbol_ms:.1f}ms")
            print(f"  - メモリ使用量: {metrics.peak_memory_mb:.1f}MB")
            print(f"  - 成功率: {metrics.successful_symbols / metrics.total_symbols * 100:.1f}%")

        return True

    except Exception as e:
        print(f"テストエラー: {e}")
        return False

    finally:
        if hasattr(system, 'shutdown'):
            system.shutdown()


def test_basic_functionality():
    """基本機能テスト（同期）"""
    print("=== 基本機能テスト ===")

    try:
        # システム初期化テスト
        system = TOPIX500AnalysisSystem(
            enable_cache=False,
            enable_parallel=False
        )

        print("OK システム初期化成功")

        # データ生成テスト
        test_data = generate_lightweight_data(3)
        print(f"OK テストデータ生成成功: {len(test_data)}銘柄")

        # データ構造確認
        for symbol, data in test_data.items():
            if not isinstance(data, pd.DataFrame) or data.empty:
                print(f"NG 無効データ: {symbol}")
                return False

        print("OK データ構造検証成功")

        if hasattr(system, 'shutdown'):
            system.shutdown()

        return True

    except Exception as e:
        print(f"NG 基本機能テストエラー: {e}")
        return False


async def main():
    """メインテスト実行"""
    print("TOPIX500分析システム - 軽量テストスイート")
    print("=" * 50)

    # 基本機能テスト
    if not test_basic_functionality():
        print("基本機能テストで失敗")
        return False

    # 軽量システムテスト
    if not await test_lightweight_system():
        print("軽量システムテストで失敗")
        return False

    print("\n" + "=" * 50)
    print("OK 全軽量テスト完了 - 基本動作確認済み")
    print("システムは次の段階：フルスケールテスト実行可能")

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\nテスト中断")
        sys.exit(1)
    except Exception as e:
        print(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
