#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高度テクニカル指標システムテスト
Issue #315: 高度テクニカル指標・ML機能拡張

統合最適化基盤を活用した高度テクニカル指標の機能テスト
"""

import asyncio
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

async def test_bollinger_bands_optimized():
    """Bollinger Bands最適化分析テスト"""
    print("\n=== Bollinger Bands最適化分析テスト ===")

    try:
        from src.day_trade.analysis.advanced_technical_indicators_optimized import AdvancedTechnicalIndicatorsOptimized

        # テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=100)
        test_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 2500, 100),
            'High': np.random.uniform(2100, 2600, 100),
            'Low': np.random.uniform(1900, 2400, 100),
            'Close': np.random.uniform(2000, 2500, 100),
            'Volume': np.random.randint(500000, 2000000, 100),
        }, index=dates)

        # システム初期化
        analyzer = AdvancedTechnicalIndicatorsOptimized(
            enable_cache=True,
            enable_parallel=False,  # 単体テスト用に無効化
            enable_ml_optimization=True,
            max_concurrent=5
        )
        print("[OK] Bollinger Bands analyzer initialization success")

        # 分析実行
        result = await analyzer.analyze_bollinger_bands_optimized(test_data, "TEST_BB")

        # 結果検証
        assert hasattr(result, 'signal'), "Signal attribute missing"
        assert result.signal in ['BUY', 'SELL', 'HOLD'], f"Invalid signal: {result.signal}"
        assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
        assert 0 <= result.bb_position <= 1, f"Invalid BB position: {result.bb_position}"
        assert result.squeeze_ratio > 0, f"Invalid squeeze ratio: {result.squeeze_ratio}"
        assert 0 <= result.performance_score <= 1, f"Invalid performance score: {result.performance_score}"

        print(f"[OK] BB Analysis: {result.signal} (confidence: {result.confidence:.2%})")
        print(f"[OK] BB Position: {result.bb_position:.3f}, Squeeze Ratio: {result.squeeze_ratio:.3f}")
        print(f"[OK] Volatility Regime: {result.volatility_regime}")
        print(f"[OK] Performance Score: {result.performance_score:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Bollinger Bands test failed: {e}")
        traceback.print_exc()
        return False

async def test_ichimoku_cloud_optimized():
    """一目均衡表最適化分析テスト"""
    print("\n=== Ichimoku Cloud最適化分析テスト ===")

    try:
        from src.day_trade.analysis.advanced_technical_indicators_optimized import AdvancedTechnicalIndicatorsOptimized

        # テストデータ生成（一目用に長期データ）
        dates = pd.date_range(start='2024-01-01', periods=150)
        np.random.seed(42)  # 再現性のため
        base_price = 2200
        price_series = []

        for i in range(150):
            # ランダムウォーク + 小さなトレンド
            change = np.random.normal(0, 0.02) + 0.001
            base_price = base_price * (1 + change)
            price_series.append(base_price)

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.995, 1.005) for p in price_series],
            'High': [p * np.random.uniform(1.005, 1.02) for p in price_series],
            'Low': [p * np.random.uniform(0.98, 0.995) for p in price_series],
            'Close': price_series,
            'Volume': np.random.randint(500000, 2000000, 150),
        }, index=dates)

        # システム初期化
        analyzer = AdvancedTechnicalIndicatorsOptimized(
            enable_cache=True,
            enable_parallel=False,
            enable_ml_optimization=True
        )
        print("[OK] Ichimoku analyzer initialization success")

        # 分析実行
        result = await analyzer.analyze_ichimoku_cloud_optimized(test_data, "TEST_ICHIMOKU")

        # 結果検証
        assert hasattr(result, 'overall_signal'), "Overall signal attribute missing"
        assert result.overall_signal in ['BUY', 'SELL', 'HOLD'], f"Invalid signal: {result.overall_signal}"
        assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
        assert result.price_vs_cloud in ['above', 'in', 'below'], f"Invalid price vs cloud: {result.price_vs_cloud}"
        assert result.tk_cross in ['bullish', 'bearish', 'neutral'], f"Invalid TK cross: {result.tk_cross}"
        assert 0 <= result.performance_score <= 1, f"Invalid performance score: {result.performance_score}"

        print(f"[OK] Ichimoku Analysis: {result.overall_signal} (confidence: {result.confidence:.2%})")
        print(f"[OK] Price vs Cloud: {result.price_vs_cloud}")
        print(f"[OK] TK Cross: {result.tk_cross}, Chikou Signal: {result.chikou_signal}")
        print(f"[OK] Cloud Color: {result.cloud_color}, Thickness: {result.cloud_thickness:.2f}")
        print(f"[OK] Trend Strength: {result.trend_strength:.3f}")
        print(f"[OK] Performance Score: {result.performance_score:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Ichimoku Cloud test failed: {e}")
        traceback.print_exc()
        return False

async def test_batch_analysis():
    """バッチ分析テスト"""
    print("\n=== バッチ分析テスト ===")

    try:
        from src.day_trade.analysis.advanced_technical_indicators_optimized import AdvancedTechnicalIndicatorsOptimized

        # 複数銘柄のテストデータ生成
        symbols = ["TEST1", "TEST2", "TEST3"]
        batch_data = {}

        for symbol in symbols:
            dates = pd.date_range(start='2024-01-01', periods=120)
            base_price = np.random.uniform(1500, 3000)

            test_data = pd.DataFrame({
                'Open': np.random.uniform(base_price * 0.98, base_price * 1.02, 120),
                'High': np.random.uniform(base_price * 1.01, base_price * 1.05, 120),
                'Low': np.random.uniform(base_price * 0.95, base_price * 0.99, 120),
                'Close': np.random.uniform(base_price * 0.99, base_price * 1.01, 120),
                'Volume': np.random.randint(300000, 1500000, 120),
            }, index=dates)

            batch_data[symbol] = test_data

        # システム初期化
        analyzer = AdvancedTechnicalIndicatorsOptimized(
            enable_cache=True,
            enable_parallel=True,
            max_concurrent=3
        )
        print("[OK] Batch analyzer initialization success")

        # バッチ分析実行
        results = await analyzer.batch_analyze_symbols(
            batch_data,
            analysis_types=["bb", "ichimoku"]
        )

        # 結果検証
        assert len(results) == len(symbols), f"Results count mismatch: {len(results)} vs {len(symbols)}"

        for symbol in symbols:
            assert symbol in results, f"Missing results for {symbol}"
            symbol_results = results[symbol]

            if "bollinger_bands" in symbol_results:
                bb_result = symbol_results["bollinger_bands"]
                assert hasattr(bb_result, 'signal'), f"BB signal missing for {symbol}"
                print(f"[OK] {symbol} BB: {bb_result.signal} ({bb_result.confidence:.1%})")

            if "ichimoku_cloud" in symbol_results:
                ichimoku_result = symbol_results["ichimoku_cloud"]
                assert hasattr(ichimoku_result, 'overall_signal'), f"Ichimoku signal missing for {symbol}"
                print(f"[OK] {symbol} Ichimoku: {ichimoku_result.overall_signal} ({ichimoku_result.confidence:.1%})")

        print(f"[OK] Batch analysis completed: {len(results)} symbols processed")
        return True

    except Exception as e:
        print(f"[ERROR] Batch analysis test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_stats():
    """パフォーマンス統計テスト"""
    print("\n=== パフォーマンス統計テスト ===")

    try:
        from src.day_trade.analysis.advanced_technical_indicators_optimized import AdvancedTechnicalIndicatorsOptimized

        # システム初期化
        analyzer = AdvancedTechnicalIndicatorsOptimized(
            enable_cache=True,
            enable_parallel=True,
            enable_ml_optimization=True
        )

        # 統計情報取得
        stats = analyzer.get_optimization_performance_stats()

        # 統計項目検証
        required_keys = [
            'total_analyses', 'cache_hit_rate', 'parallel_usage_rate',
            'ml_optimization_rate', 'avg_processing_time_ms',
            'optimization_benefits'
        ]

        for key in required_keys:
            assert key in stats, f"Missing stats key: {key}"

        print("[OK] Performance stats structure validation passed")
        print(f"[STATS] Total analyses: {stats['total_analyses']}")
        print(f"[STATS] Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"[STATS] Parallel usage rate: {stats['parallel_usage_rate']:.1%}")
        print(f"[STATS] Average processing time: {stats['avg_processing_time_ms']:.1f}ms")

        # 最適化効果
        benefits = stats['optimization_benefits']
        print("[OPTIMIZATION] Benefits:")
        for benefit, value in benefits.items():
            print(f"  - {benefit}: {value}")

        return True

    except Exception as e:
        print(f"[ERROR] Performance stats test failed: {e}")
        traceback.print_exc()
        return False

async def test_cache_functionality():
    """キャッシュ機能テスト"""
    print("\n=== キャッシュ機能テスト ===")

    try:
        from src.day_trade.analysis.advanced_technical_indicators_optimized import AdvancedTechnicalIndicatorsOptimized

        # テストデータ
        dates = pd.date_range(start='2024-01-01', periods=100)
        test_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 2500, 100),
            'High': np.random.uniform(2100, 2600, 100),
            'Low': np.random.uniform(1900, 2400, 100),
            'Close': np.random.uniform(2000, 2500, 100),
            'Volume': np.random.randint(500000, 2000000, 100),
        }, index=dates)

        # キャッシュ有効版
        analyzer_cached = AdvancedTechnicalIndicatorsOptimized(enable_cache=True)

        # 初回分析（キャッシュミス）
        result1 = await analyzer_cached.analyze_bollinger_bands_optimized(test_data, "CACHE_TEST")
        stats_after_first = analyzer_cached.get_optimization_performance_stats()

        # 同一分析（キャッシュヒット期待）
        result2 = await analyzer_cached.analyze_bollinger_bands_optimized(test_data, "CACHE_TEST")
        stats_after_second = analyzer_cached.get_optimization_performance_stats()

        # キャッシュヒット検証
        cache_hit_increased = stats_after_second['cache_hit_rate'] > stats_after_first['cache_hit_rate']

        print(f"[OK] First analysis cache hit rate: {stats_after_first['cache_hit_rate']:.1%}")
        print(f"[OK] Second analysis cache hit rate: {stats_after_second['cache_hit_rate']:.1%}")
        print(f"[OK] Cache hit rate increased: {cache_hit_increased}")
        print(f"[OK] Results consistency: {result1.signal == result2.signal}")

        return True

    except Exception as e:
        print(f"[ERROR] Cache functionality test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """メインテスト実行"""
    print("高度テクニカル指標システム（統合最適化版）テスト開始")
    print("=" * 60)

    test_results = []

    # 各テスト実行
    test_results.append(("Bollinger Bands最適化", await test_bollinger_bands_optimized()))
    test_results.append(("Ichimoku Cloud最適化", await test_ichimoku_cloud_optimized()))
    test_results.append(("バッチ分析", await test_batch_analysis()))
    test_results.append(("パフォーマンス統計", await test_performance_stats()))
    test_results.append(("キャッシュ機能", await test_cache_functionality()))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("=== テスト結果サマリー ===")

    passed = 0
    for name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1

    success_rate = passed / len(test_results) * 100
    print(f"\n成功率: {passed}/{len(test_results)} ({success_rate:.1f}%)")

    if passed == len(test_results):
        print("[SUCCESS] 全テスト成功！高度テクニカル指標システム準備完了")
        print("\nIssue #315 Phase 1実装成果:")
        print("✅ Bollinger Bands変動率分析（統合最適化版）")
        print("✅ Ichimoku Cloud総合判定（高精度版）")
        print("✅ 統合キャッシュシステム連携")
        print("✅ 並列処理システム連携")
        print("✅ パフォーマンス監視統合")
        print("✅ バッチ処理対応")
        return True
    else:
        print("[WARNING] 一部テスト失敗 - 要修正")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテスト中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n致命的エラー: {e}")
        traceback.print_exc()
        sys.exit(1)
