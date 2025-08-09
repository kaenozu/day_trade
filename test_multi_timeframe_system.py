#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マルチタイムフレーム分析システムテスト
Issue #315 Phase 2: マルチタイムフレーム分析実装

統合最適化基盤を活用したマルチタイムフレーム分析の機能テスト
"""

import asyncio
import sys
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

async def test_multi_timeframe_initialization():
    """マルチタイムフレーム分析システム初期化テスト"""
    print("\n=== マルチタイムフレーム分析システム初期化テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import MultiTimeframeAnalysisOptimized

        # システム初期化
        analyzer = MultiTimeframeAnalysisOptimized(
            enable_cache=True,
            enable_parallel=True,
            enable_ml_optimization=True,
            max_concurrent=8
        )
        print("[OK] MultiTimeframeAnalysisOptimized initialization success")

        # 設定確認
        assert len(analyzer.timeframe_configs) == 3, f"Expected 3 timeframes, got {len(analyzer.timeframe_configs)}"
        assert "daily" in analyzer.timeframe_configs, "Daily timeframe missing"
        assert "weekly" in analyzer.timeframe_configs, "Weekly timeframe missing"
        assert "monthly" in analyzer.timeframe_configs, "Monthly timeframe missing"

        # 重み確認
        total_weight = sum(config.weight for config in analyzer.timeframe_configs.values())
        assert abs(total_weight - 1.0) < 0.01, f"Total weight should be 1.0, got {total_weight}"

        print(f"[OK] Timeframes configured: {list(analyzer.timeframe_configs.keys())}")
        print(f"[OK] Weight distribution: {[(tf, config.weight) for tf, config in analyzer.timeframe_configs.items()]}")
        print(f"[OK] Cache enabled: {analyzer.cache_enabled}")
        print(f"[OK] Parallel enabled: {analyzer.parallel_enabled}")

        return True

    except Exception as e:
        print(f"[ERROR] MultiTimeframe initialization test failed: {e}")
        traceback.print_exc()
        return False

async def test_single_symbol_analysis():
    """単一銘柄マルチタイムフレーム分析テスト"""
    print("\n=== 単一銘柄マルチタイムフレーム分析テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import MultiTimeframeAnalysisOptimized

        # 長期間テストデータ生成
        dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
        np.random.seed(42)  # 再現性のため

        # リアルな価格トレンド生成
        base_price = 2500
        trend = np.linspace(0, 0.3, 300)  # 30%の上昇トレンド
        noise = np.random.normal(0, 0.02, 300)  # ノイズ

        price_series = []
        for i in range(300):
            price = base_price * (1 + trend[i] + noise[i])
            price_series.append(max(price, base_price * 0.5))  # 最低価格制限

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.995, 1.005) for p in price_series],
            'High': [p * np.random.uniform(1.005, 1.03) for p in price_series],
            'Low': [p * np.random.uniform(0.97, 0.995) for p in price_series],
            'Close': price_series,
            'Volume': np.random.randint(500000, 2000000, 300),
        }, index=dates)

        # システム初期化
        analyzer = MultiTimeframeAnalysisOptimized(
            enable_cache=True,
            enable_parallel=False,  # 単体テスト用に無効化
            enable_ml_optimization=True
        )
        print("[OK] System initialization success")

        # マルチタイムフレーム分析実行
        result = await analyzer.analyze_multi_timeframe(test_data, "TEST_SYMBOL")

        # 結果検証
        assert hasattr(result, 'weighted_signal'), "Weighted signal attribute missing"
        assert result.weighted_signal in ['BUY', 'SELL', 'HOLD'], f"Invalid weighted signal: {result.weighted_signal}"
        assert 0 <= result.weighted_confidence <= 1, f"Invalid weighted confidence: {result.weighted_confidence}"
        assert 0 <= result.risk_adjusted_score <= 1, f"Invalid risk adjusted score: {result.risk_adjusted_score}"
        assert 0 <= result.recommended_position_size <= 1, f"Invalid position size: {result.recommended_position_size}"

        print(f"[OK] Weighted Signal: {result.weighted_signal} (confidence: {result.weighted_confidence:.1%})")
        print(f"[OK] Trend Consistency: {result.trend_consistency.overall_consistency:.1%} ({result.trend_consistency.trend_alignment})")
        print(f"[OK] Dominant Timeframe: {result.trend_consistency.dominant_timeframe}")
        print(f"[OK] Risk Adjusted Score: {result.risk_adjusted_score:.3f}")
        print(f"[OK] Recommended Position Size: {result.recommended_position_size:.1%}")

        # 時間軸別シグナル確認
        timeframes_analyzed = len(result.timeframe_signals)
        assert timeframes_analyzed > 0, "No timeframe signals generated"
        print(f"[OK] Timeframes analyzed: {timeframes_analyzed}")

        for tf, signal in result.timeframe_signals.items():
            assert hasattr(signal, 'signal'), f"Signal attribute missing for {tf}"
            assert signal.signal in ['BUY', 'SELL', 'HOLD'], f"Invalid signal for {tf}: {signal.signal}"
            print(f"     {tf}: {signal.signal} (confidence: {signal.confidence:.1%}, trend_strength: {signal.trend_strength:.3f})")

        # トレンド整合性確認
        consistency = result.trend_consistency
        assert hasattr(consistency, 'overall_consistency'), "Overall consistency missing"
        assert 0 <= consistency.overall_consistency <= 1, f"Invalid consistency: {consistency.overall_consistency}"
        assert consistency.trend_alignment in ['aligned', 'mixed', 'conflicting'], f"Invalid alignment: {consistency.trend_alignment}"

        print(f"[OK] Trend Alignment: {consistency.trend_alignment}")
        print(f"[OK] Reliability Score: {consistency.reliability_score:.3f}")
        if consistency.conflicting_signals:
            print(f"[OK] Conflicting Signals: {consistency.conflicting_signals}")

        return True

    except Exception as e:
        print(f"[ERROR] Single symbol analysis test failed: {e}")
        traceback.print_exc()
        return False

async def test_batch_multi_timeframe_analysis():
    """バッチマルチタイムフレーム分析テスト"""
    print("\n=== バッチマルチタイムフレーム分析テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import MultiTimeframeAnalysisOptimized

        # 複数銘柄のテストデータ生成
        symbols = ["TEST_A", "TEST_B", "TEST_C"]
        batch_data = {}

        np.random.seed(100)

        for i, symbol in enumerate(symbols):
            dates = pd.date_range(start='2023-01-01', periods=250, freq='D')

            # 銘柄ごとに異なるトレンド
            base_price = 2000 + i * 500
            trend_factor = (i - 1) * 0.1  # -0.1, 0.0, 0.1

            prices = []
            current_price = base_price

            for day in range(250):
                daily_return = np.random.normal(0.001 + trend_factor/100, 0.02)
                current_price *= (1 + daily_return)
                prices.append(current_price)

            test_data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.995, 1.005) for p in prices],
                'High': [p * np.random.uniform(1.005, 1.02) for p in prices],
                'Low': [p * np.random.uniform(0.98, 0.995) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(300000, 1500000, 250),
            }, index=dates)

            batch_data[symbol] = test_data

        # システム初期化
        analyzer = MultiTimeframeAnalysisOptimized(
            enable_cache=True,
            enable_parallel=True,
            max_concurrent=len(symbols)
        )
        print("[OK] Batch analyzer initialization success")

        # バッチ分析実行
        results = await analyzer.batch_analyze_multi_timeframe(batch_data)

        # 結果検証
        assert len(results) == len(symbols), f"Results count mismatch: {len(results)} vs {len(symbols)}"

        for symbol in symbols:
            assert symbol in results, f"Missing results for {symbol}"
            result = results[symbol]

            assert hasattr(result, 'weighted_signal'), f"Weighted signal missing for {symbol}"
            assert hasattr(result, 'timeframe_signals'), f"Timeframe signals missing for {symbol}"
            assert hasattr(result, 'trend_consistency'), f"Trend consistency missing for {symbol}"

            print(f"[OK] {symbol}: {result.weighted_signal} (confidence: {result.weighted_confidence:.1%})")
            print(f"     Consistency: {result.trend_consistency.overall_consistency:.1%} ({result.trend_consistency.trend_alignment})")
            print(f"     Risk Score: {result.risk_adjusted_score:.3f}, Position: {result.recommended_position_size:.1%}")

            # 各時間軸のシグナル確認
            for tf, tf_signal in result.timeframe_signals.items():
                print(f"     {tf}: {tf_signal.signal} ({tf_signal.confidence:.1%})")

        print(f"[OK] Batch analysis completed: {len(results)} symbols processed")
        return True

    except Exception as e:
        print(f"[ERROR] Batch multi timeframe analysis test failed: {e}")
        traceback.print_exc()
        return False

async def test_trend_consistency_analysis():
    """トレンド整合性分析テスト"""
    print("\n=== トレンド整合性分析テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import (
            MultiTimeframeAnalysisOptimized, MultiTimeframeSignal
        )

        # システム初期化
        analyzer = MultiTimeframeAnalysisOptimized()

        # テスト用時間軸シグナル作成
        # Case 1: 整合性の高いシグナル（全てBUY）
        aligned_signals = {
            "daily": MultiTimeframeSignal("daily", "BUY", 0.8, 0.7, {}, 0.8),
            "weekly": MultiTimeframeSignal("weekly", "BUY", 0.7, 0.6, {}, 0.7),
            "monthly": MultiTimeframeSignal("monthly", "BUY", 0.6, 0.5, {}, 0.6)
        }

        consistency_aligned = await analyzer._analyze_trend_consistency(aligned_signals)

        assert consistency_aligned.overall_consistency >= 0.8, f"Expected high consistency, got {consistency_aligned.overall_consistency}"
        assert consistency_aligned.trend_alignment == "aligned", f"Expected aligned, got {consistency_aligned.trend_alignment}"
        assert len(consistency_aligned.conflicting_signals) == 0, f"Expected no conflicts, got {consistency_aligned.conflicting_signals}"

        print("[OK] Aligned signals consistency test passed")
        print(f"     Consistency: {consistency_aligned.overall_consistency:.1%}")
        print(f"     Alignment: {consistency_aligned.trend_alignment}")
        print(f"     Dominant: {consistency_aligned.dominant_timeframe}")

        # Case 2: 相反するシグナル（BUY vs SELL）
        conflicting_signals = {
            "daily": MultiTimeframeSignal("daily", "BUY", 0.8, 0.7, {}, 0.8),
            "weekly": MultiTimeframeSignal("weekly", "SELL", 0.7, 0.6, {}, 0.7),
            "monthly": MultiTimeframeSignal("monthly", "HOLD", 0.5, 0.3, {}, 0.5)
        }

        consistency_conflicting = await analyzer._analyze_trend_consistency(conflicting_signals)

        assert consistency_conflicting.overall_consistency <= 0.6, f"Expected low consistency, got {consistency_conflicting.overall_consistency}"
        assert consistency_conflicting.trend_alignment in ["mixed", "conflicting"], f"Expected mixed/conflicting, got {consistency_conflicting.trend_alignment}"
        assert len(consistency_conflicting.conflicting_signals) > 0, "Expected conflicting signals"

        print("[OK] Conflicting signals consistency test passed")
        print(f"     Consistency: {consistency_conflicting.overall_consistency:.1%}")
        print(f"     Alignment: {consistency_conflicting.trend_alignment}")
        print(f"     Conflicts: {consistency_conflicting.conflicting_signals}")

        return True

    except Exception as e:
        print(f"[ERROR] Trend consistency analysis test failed: {e}")
        traceback.print_exc()
        return False

async def test_weighted_signal_generation():
    """重み付きシグナル生成テスト"""
    print("\n=== 重み付きシグナル生成テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import (
            MultiTimeframeAnalysisOptimized, MultiTimeframeSignal, TrendConsistency
        )

        # システム初期化
        analyzer = MultiTimeframeAnalysisOptimized()

        # テストケース1: 強いBUYシグナル
        strong_buy_signals = {
            "daily": MultiTimeframeSignal("daily", "BUY", 0.9, 0.8, {}, 0.85),
            "weekly": MultiTimeframeSignal("weekly", "BUY", 0.8, 0.7, {}, 0.75),
            "monthly": MultiTimeframeSignal("monthly", "BUY", 0.7, 0.6, {}, 0.65)
        }

        strong_consistency = TrendConsistency(
            overall_consistency=1.0,
            conflicting_signals=[],
            dominant_timeframe="daily",
            trend_alignment="aligned",
            reliability_score=0.9
        )

        weighted_result = await analyzer._generate_weighted_signal(strong_buy_signals, strong_consistency)

        assert weighted_result['signal'] == "BUY", f"Expected BUY signal, got {weighted_result['signal']}"
        assert weighted_result['confidence'] >= 0.7, f"Expected high confidence, got {weighted_result['confidence']}"
        assert weighted_result['weighted_score'] > 0.5, f"Expected positive score, got {weighted_result['weighted_score']}"

        print("[OK] Strong BUY signal generation test passed")
        print(f"     Signal: {weighted_result['signal']}")
        print(f"     Confidence: {weighted_result['confidence']:.1%}")
        print(f"     Weighted Score: {weighted_result['weighted_score']:.3f}")

        # テストケース2: 混合シグナル
        mixed_signals = {
            "daily": MultiTimeframeSignal("daily", "BUY", 0.6, 0.5, {}, 0.55),
            "weekly": MultiTimeframeSignal("weekly", "HOLD", 0.5, 0.3, {}, 0.4),
            "monthly": MultiTimeframeSignal("monthly", "SELL", 0.7, 0.6, {}, 0.65)
        }

        mixed_consistency = TrendConsistency(
            overall_consistency=0.4,
            conflicting_signals=["daily", "monthly"],
            dominant_timeframe="monthly",
            trend_alignment="conflicting",
            reliability_score=0.5
        )

        mixed_result = await analyzer._generate_weighted_signal(mixed_signals, mixed_consistency)

        assert mixed_result['signal'] in ["BUY", "SELL", "HOLD"], f"Invalid signal: {mixed_result['signal']}"
        assert 0 <= mixed_result['confidence'] <= 1, f"Invalid confidence: {mixed_result['confidence']}"

        print("[OK] Mixed signal generation test passed")
        print(f"     Signal: {mixed_result['signal']}")
        print(f"     Confidence: {mixed_result['confidence']:.1%}")
        print(f"     Weighted Score: {mixed_result['weighted_score']:.3f}")

        return True

    except Exception as e:
        print(f"[ERROR] Weighted signal generation test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_optimization():
    """パフォーマンス最適化テスト"""
    print("\n=== パフォーマンス最適化テスト ===")

    try:
        from src.day_trade.analysis.multi_timeframe_analysis_optimized import MultiTimeframeAnalysisOptimized

        # テストデータ
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 2500, 200),
            'High': np.random.uniform(2100, 2600, 200),
            'Low': np.random.uniform(1900, 2400, 200),
            'Close': np.random.uniform(2000, 2500, 200),
            'Volume': np.random.randint(500000, 2000000, 200),
        }, index=dates)

        # キャッシュ有効版
        analyzer_cached = MultiTimeframeAnalysisOptimized(
            enable_cache=True,
            enable_parallel=True,
            enable_ml_optimization=True
        )

        # 初回分析（キャッシュミス）
        import time
        start_time = time.time()
        result1 = await analyzer_cached.analyze_multi_timeframe(test_data, "PERF_TEST")
        first_time = time.time() - start_time

        # 同一分析（キャッシュヒット期待）
        start_time = time.time()
        result2 = await analyzer_cached.analyze_multi_timeframe(test_data, "PERF_TEST")
        second_time = time.time() - start_time

        # パフォーマンス統計取得
        stats = analyzer_cached.get_optimization_performance_stats()

        print(f"[OK] First analysis time: {first_time:.3f}s")
        print(f"[OK] Second analysis time: {second_time:.3f}s")
        print(f"[OK] Speedup ratio: {first_time/max(second_time, 0.001):.1f}x")
        print(f"[OK] Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"[OK] Total analyses: {stats['total_analyses']}")

        # 結果一貫性確認
        assert result1.weighted_signal == result2.weighted_signal, "Inconsistent cached results"
        assert abs(result1.weighted_confidence - result2.weighted_confidence) < 0.01, "Inconsistent cached confidence"

        print("[OK] Cache consistency verified")

        # 最適化効果表示
        benefits = stats['optimization_benefits']
        print("[OK] Optimization benefits:")
        for benefit, value in benefits.items():
            print(f"     {benefit}: {value}")

        return True

    except Exception as e:
        print(f"[ERROR] Performance optimization test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """メインテスト実行"""
    print("マルチタイムフレーム分析システム（統合最適化版）テスト開始")
    print("=" * 70)

    test_results = []

    # 各テスト実行
    test_results.append(("システム初期化", await test_multi_timeframe_initialization()))
    test_results.append(("単一銘柄分析", await test_single_symbol_analysis()))
    test_results.append(("バッチ分析", await test_batch_multi_timeframe_analysis()))
    test_results.append(("トレンド整合性", await test_trend_consistency_analysis()))
    test_results.append(("重み付きシグナル", await test_weighted_signal_generation()))
    test_results.append(("パフォーマンス最適化", await test_performance_optimization()))

    # 結果サマリー
    print("\n" + "=" * 70)
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
        print("[SUCCESS] 全テスト成功！マルチタイムフレーム分析システム準備完了")
        print("\nIssue #315 Phase 2実装成果:")
        print("- 日足・週足・月足の組み合わせ分析")
        print("- 複数時間軸でのトレンド判定")
        print("- タイムフレーム間の整合性チェック")
        print("- 時間軸別重み付け最適化")
        print("- 統合最適化基盤フル活用")
        print("- リスク調整スコア・推奨ポジションサイズ算出")
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
