#!/usr/bin/env python3
"""
Advanced ML Models System テスト
Issue #315 Phase 3: Advanced ML models実装テスト

LSTM時系列予測、アンサンブル学習、高度特徴量エンジニアリングの総合テスト
"""

import asyncio
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

async def test_advanced_ml_initialization():
    """Advanced ML Models初期化テスト"""
    print("\n=== Advanced ML Models初期化テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # システム初期化
        ml_system = AdvancedMLModels(
            enable_cache=True,
            enable_parallel=True,
            enable_lstm=True,
            lstm_sequence_length=30,
            ensemble_models=3,
            max_concurrent=5
        )
        print("[OK] AdvancedMLModels initialization success")

        # 設定確認
        assert ml_system.enable_cache, "Cache should be enabled"
        assert ml_system.enable_parallel, "Parallel should be enabled"
        assert ml_system.lstm_sequence_length == 30, f"LSTM sequence length should be 30, got {ml_system.lstm_sequence_length}"
        assert ml_system.ensemble_models == 3, f"Ensemble models should be 3, got {ml_system.ensemble_models}"

        print(f"[OK] Cache enabled: {ml_system.enable_cache}")
        print(f"[OK] Parallel enabled: {ml_system.enable_parallel}")
        print(f"[OK] LSTM enabled: {ml_system.enable_lstm}")
        print(f"[OK] LSTM sequence length: {ml_system.lstm_sequence_length}")
        print(f"[OK] Ensemble models: {ml_system.ensemble_models}")

        return True

    except Exception as e:
        print(f"[ERROR] Advanced ML initialization test failed: {e}")
        traceback.print_exc()
        return False

async def test_advanced_feature_extraction():
    """高度特徴量エンジニアリングテスト"""
    print("\n=== 高度特徴量エンジニアリングテスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # テストデータ生成（十分な長さ）
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        base_price = 2500
        price_trend = np.cumsum(np.random.normal(0.001, 0.02, 100))
        prices = base_price * np.exp(price_trend)

        test_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.995, 1.005, 100),
            'High': prices * np.random.uniform(1.005, 1.03, 100),
            'Low': prices * np.random.uniform(0.97, 0.995, 100),
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, 100),
        }, index=dates)

        # システム初期化
        ml_system = AdvancedMLModels(
            enable_cache=False,  # テスト用にキャッシュ無効
            enable_parallel=False
        )
        print("[OK] ML system initialization success")

        # 特徴量抽出実行
        feature_set = await ml_system.extract_advanced_features(test_data, "TEST_FEATURES")

        # 結果検証
        assert hasattr(feature_set, 'symbol'), "Feature set missing symbol"
        assert feature_set.symbol == "TEST_FEATURES", f"Symbol mismatch: {feature_set.symbol}"
        assert hasattr(feature_set, 'feature_count'), "Feature set missing feature_count"
        assert feature_set.feature_count > 0, f"No features extracted: {feature_set.feature_count}"

        print(f"[OK] Symbol: {feature_set.symbol}")
        print(f"[OK] Total features: {feature_set.feature_count}")
        print(f"[OK] Technical features: {len(feature_set.technical_features)}")
        print(f"[OK] Price patterns: {len(feature_set.price_patterns)}")
        print(f"[OK] Volatility features: {len(feature_set.volatility_features)}")
        print(f"[OK] Momentum features: {len(feature_set.momentum_features)}")
        print(f"[OK] Volume features: {len(feature_set.volume_features)}")
        print(f"[OK] Multiframe features: {len(feature_set.multiframe_features)}")
        print(f"[OK] Processing time: {feature_set.processing_time:.3f}s")

        # 特徴量サンプル表示
        if feature_set.technical_features:
            print("[SAMPLE] Technical features:")
            for key, value in list(feature_set.technical_features.items())[:3]:
                print(f"  {key}: {value:.6f}")

        return True

    except Exception as e:
        print(f"[ERROR] Feature extraction test failed: {e}")
        traceback.print_exc()
        return False

async def test_ensemble_learning():
    """アンサンブル学習テスト"""
    print("\n=== アンサンブル学習テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # 長期テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
        np.random.seed(100)

        # リアルなトレンドデータ
        base_price = 2200
        trend = np.linspace(0, 0.2, 120)  # 20%上昇トレンド
        noise = np.random.normal(0, 0.02, 120)
        prices = base_price * np.exp(trend + np.cumsum(noise))

        test_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.998, 1.002, 120),
            'High': prices * np.random.uniform(1.005, 1.025, 120),
            'Low': prices * np.random.uniform(0.975, 0.995, 120),
            'Close': prices,
            'Volume': np.random.randint(400000, 1800000, 120),
        }, index=dates)

        # システム初期化
        ml_system = AdvancedMLModels(
            enable_cache=False,
            enable_parallel=False,
            ensemble_models=3
        )
        print("[OK] Ensemble ML system initialization success")

        # 特徴量抽出
        feature_set = await ml_system.extract_advanced_features(test_data, "TEST_ENSEMBLE")
        print(f"[OK] Features extracted: {feature_set.feature_count}")

        # アンサンブル学習実行
        ensemble_result = await ml_system.ensemble_prediction(test_data, "TEST_ENSEMBLE", feature_set)

        # 結果検証
        assert hasattr(ensemble_result, 'symbol'), "Ensemble result missing symbol"
        assert ensemble_result.symbol == "TEST_ENSEMBLE", f"Symbol mismatch: {ensemble_result.symbol}"
        assert hasattr(ensemble_result, 'ensemble_prediction'), "Ensemble result missing prediction"
        assert hasattr(ensemble_result, 'weighted_confidence'), "Ensemble result missing confidence"
        assert 0 <= ensemble_result.weighted_confidence <= 1, f"Invalid confidence: {ensemble_result.weighted_confidence}"
        assert hasattr(ensemble_result, 'consensus_strength'), "Ensemble result missing consensus"
        assert 0 <= ensemble_result.consensus_strength <= 1, f"Invalid consensus: {ensemble_result.consensus_strength}"

        print(f"[OK] Ensemble prediction: {ensemble_result.ensemble_prediction:.6f}")
        print(f"[OK] Weighted confidence: {ensemble_result.weighted_confidence:.1%}")
        print(f"[OK] Consensus strength: {ensemble_result.consensus_strength:.1%}")
        print(f"[OK] Processing time: {ensemble_result.processing_time:.3f}s")

        # 個別モデル予測確認
        print(f"[OK] Individual models: {len(ensemble_result.individual_predictions)}")
        for model_name, prediction in ensemble_result.individual_predictions.items():
            weight = ensemble_result.model_weights.get(model_name, 0.0)
            print(f"  {model_name}: {prediction:.6f} (weight: {weight:.3f})")

        # 外れ値検出確認
        if ensemble_result.outlier_detection:
            print(f"[OK] Outlier models detected: {ensemble_result.outlier_detection}")
        else:
            print("[OK] No outlier models detected")

        return True

    except Exception as e:
        print(f"[ERROR] Ensemble learning test failed: {e}")
        traceback.print_exc()
        return False

async def test_lstm_prediction():
    """LSTM時系列予測テスト"""
    print("\n=== LSTM時系列予測テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # LSTM用長期データ（100日以上必要）
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        np.random.seed(200)

        # 時系列パターンを持つデータ
        base_price = 3000
        trend = 0.0002 * np.arange(150)  # 線形トレンド
        seasonality = 0.05 * np.sin(2 * np.pi * np.arange(150) / 20)  # 20日周期
        noise = np.cumsum(np.random.normal(0, 0.01, 150))

        log_returns = trend + seasonality + noise
        prices = base_price * np.exp(log_returns)

        test_data = pd.DataFrame({
            'Open': prices * np.random.uniform(0.999, 1.001, 150),
            'High': prices * np.random.uniform(1.002, 1.015, 150),
            'Low': prices * np.random.uniform(0.985, 0.998, 150),
            'Close': prices,
            'Volume': np.random.randint(600000, 2200000, 150),
        }, index=dates)

        # システム初期化（LSTM有効）
        ml_system = AdvancedMLModels(
            enable_cache=False,
            enable_parallel=False,
            enable_lstm=True,
            lstm_sequence_length=30
        )
        print("[OK] LSTM ML system initialization success")

        # システムでLSTMが利用可能かチェック
        if not ml_system.enable_lstm:
            print("[WARNING] LSTM disabled - TensorFlow not available")
            # LSTM無効時のデフォルト結果テスト
            feature_set = await ml_system.extract_advanced_features(test_data, "TEST_LSTM")
            lstm_result = await ml_system.predict_with_lstm(test_data, "TEST_LSTM", feature_set)

            assert lstm_result.symbol == "TEST_LSTM", "Symbol should match"
            assert lstm_result.trend_direction == "sideways", "Should return default sideways"
            print("[OK] LSTM default result validation passed")
            return True

        # 特徴量抽出
        feature_set = await ml_system.extract_advanced_features(test_data, "TEST_LSTM")
        print(f"[OK] Features extracted: {feature_set.feature_count}")

        # LSTM予測実行
        lstm_result = await ml_system.predict_with_lstm(test_data, "TEST_LSTM", feature_set)

        # 結果検証
        assert hasattr(lstm_result, 'symbol'), "LSTM result missing symbol"
        assert lstm_result.symbol == "TEST_LSTM", f"Symbol mismatch: {lstm_result.symbol}"
        assert hasattr(lstm_result, 'predictions'), "LSTM result missing predictions"
        assert hasattr(lstm_result, 'confidence'), "LSTM result missing confidence"
        assert 0 <= lstm_result.confidence <= 1, f"Invalid confidence: {lstm_result.confidence}"
        assert hasattr(lstm_result, 'trend_direction'), "LSTM result missing trend_direction"
        assert lstm_result.trend_direction in ['up', 'down', 'sideways'], f"Invalid trend: {lstm_result.trend_direction}"
        assert hasattr(lstm_result, 'model_score'), "LSTM result missing model_score"
        assert 0 <= lstm_result.model_score <= 1, f"Invalid model score: {lstm_result.model_score}"

        print(f"[OK] LSTM predictions: {len(lstm_result.predictions)} points")
        print(f"[OK] Confidence: {lstm_result.confidence:.1%}")
        print(f"[OK] Trend direction: {lstm_result.trend_direction}")
        print(f"[OK] Volatility forecast: {lstm_result.volatility_forecast:.6f}")
        print(f"[OK] Model score: {lstm_result.model_score:.3f}")
        print(f"[OK] Processing time: {lstm_result.processing_time:.3f}s")

        # 特徴量重要度確認
        if lstm_result.feature_importance:
            print(f"[OK] Feature importance calculated: {len(lstm_result.feature_importance)} features")
            top_features = sorted(
                lstm_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for feature, importance in top_features:
                print(f"  {feature}: {importance:.4f}")

        return True

    except Exception as e:
        print(f"[ERROR] LSTM prediction test failed: {e}")
        traceback.print_exc()
        return False

async def test_batch_advanced_ml():
    """バッチAdvanced ML処理テスト"""
    print("\n=== バッチAdvanced ML処理テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # 複数銘柄データ
        symbols = ["TEST_A", "TEST_B", "TEST_C"]
        batch_data = {}

        for i, symbol in enumerate(symbols):
            dates = pd.date_range(start='2024-01-01', periods=80, freq='D')
            np.random.seed(300 + i)

            base_price = 2000 + i * 300
            trend_factor = (i - 1) * 0.05  # 異なるトレンド

            prices = []
            current_price = base_price
            for day in range(80):
                daily_return = np.random.normal(trend_factor/100, 0.015)
                current_price *= (1 + daily_return)
                prices.append(current_price)

            test_data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
                'High': [p * np.random.uniform(1.005, 1.020) for p in prices],
                'Low': [p * np.random.uniform(0.980, 0.995) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(400000, 1600000, 80),
            }, index=dates)

            batch_data[symbol] = test_data

        # システム初期化
        ml_system = AdvancedMLModels(
            enable_cache=True,
            enable_parallel=False,  # テスト用に単体実行
            ensemble_models=3
        )
        print("[OK] Batch ML system initialization success")

        # バッチ処理実行
        batch_results = {}
        for symbol, data in batch_data.items():
            # 特徴量抽出
            feature_set = await ml_system.extract_advanced_features(data, symbol)

            # アンサンブル予測
            ensemble_result = await ml_system.ensemble_prediction(data, symbol, feature_set)

            batch_results[symbol] = {
                'features': feature_set,
                'ensemble': ensemble_result
            }

            print(f"[OK] {symbol} processed:")
            print(f"     Features: {feature_set.feature_count}")
            print(f"     Ensemble prediction: {ensemble_result.ensemble_prediction:.6f}")
            print(f"     Confidence: {ensemble_result.weighted_confidence:.1%}")

        # バッチ結果検証
        assert len(batch_results) == len(symbols), f"Results count mismatch: {len(batch_results)} vs {len(symbols)}"

        for symbol in symbols:
            assert symbol in batch_results, f"Missing results for {symbol}"
            result = batch_results[symbol]

            assert 'features' in result, f"Missing features for {symbol}"
            assert 'ensemble' in result, f"Missing ensemble for {symbol}"
            assert result['features'].symbol == symbol, f"Symbol mismatch in features: {symbol}"
            assert result['ensemble'].symbol == symbol, f"Symbol mismatch in ensemble: {symbol}"

        print(f"[OK] Batch processing completed: {len(batch_results)} symbols")
        return True

    except Exception as e:
        print(f"[ERROR] Batch advanced ML test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_monitoring():
    """パフォーマンス監視テスト"""
    print("\n=== パフォーマンス監視テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # システム初期化
        ml_system = AdvancedMLModels(
            enable_cache=True,
            enable_parallel=True
        )

        # 初期統計取得
        initial_stats = ml_system.get_performance_stats()

        # 統計項目検証
        required_keys = [
            'total_predictions', 'cache_hit_rate', 'lstm_predictions',
            'ensemble_predictions', 'avg_processing_time', 'system_status'
        ]

        for key in required_keys:
            assert key in initial_stats, f"Missing stats key: {key}"

        print("[OK] Performance stats structure validation passed")
        print(f"[STATS] Total predictions: {initial_stats['total_predictions']}")
        print(f"[STATS] Cache hit rate: {initial_stats['cache_hit_rate']:.1%}")
        print(f"[STATS] LSTM predictions: {initial_stats['lstm_predictions']}")
        print(f"[STATS] Ensemble predictions: {initial_stats['ensemble_predictions']}")

        # システム状態確認
        system_status = initial_stats['system_status']
        print(f"[SYSTEM] Cache enabled: {system_status['cache_enabled']}")
        print(f"[SYSTEM] Parallel enabled: {system_status['parallel_enabled']}")
        print(f"[SYSTEM] LSTM enabled: {system_status['lstm_enabled']}")
        print(f"[SYSTEM] TensorFlow available: {system_status['tensorflow_available']}")
        print(f"[SYSTEM] Scikit-learn available: {system_status['sklearn_available']}")

        # 最適化効果確認
        benefits = initial_stats['optimization_benefits']
        print("[OPTIMIZATION] Benefits:")
        for benefit, description in benefits.items():
            print(f"  {benefit}: {description}")

        return True

    except Exception as e:
        print(f"[ERROR] Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False

async def test_cache_integration():
    """キャッシュ統合テスト"""
    print("\n=== キャッシュ統合テスト ===")

    try:
        from src.day_trade.ml.advanced_ml_models import AdvancedMLModels

        # テストデータ
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(2000, 2500, 90),
            'High': np.random.uniform(2100, 2600, 90),
            'Low': np.random.uniform(1900, 2400, 90),
            'Close': np.random.uniform(2000, 2500, 90),
            'Volume': np.random.randint(500000, 2000000, 90),
        }, index=dates)

        # キャッシュ有効版
        ml_system_cached = AdvancedMLModels(enable_cache=True)

        # 初回実行（キャッシュミス）
        import time
        start_time = time.time()
        feature_set_1 = await ml_system_cached.extract_advanced_features(test_data, "CACHE_TEST")
        first_time = time.time() - start_time

        # 2回目実行（キャッシュヒット期待）
        start_time = time.time()
        feature_set_2 = await ml_system_cached.extract_advanced_features(test_data, "CACHE_TEST")
        second_time = time.time() - start_time

        # パフォーマンス統計取得
        stats = ml_system_cached.get_performance_stats()

        print(f"[OK] First extraction time: {first_time:.3f}s")
        print(f"[OK] Second extraction time: {second_time:.3f}s")
        if second_time > 0:
            speedup = first_time / second_time
            print(f"[OK] Speedup ratio: {speedup:.1f}x")

        # 結果一貫性確認
        assert feature_set_1.symbol == feature_set_2.symbol, "Inconsistent cached symbol"
        assert feature_set_1.feature_count == feature_set_2.feature_count, "Inconsistent cached feature count"

        print("[OK] Cache consistency verified")

        # キャッシュ効果確認
        benefits = stats['optimization_benefits']
        print("[OK] Cache benefits:")
        for benefit, description in benefits.items():
            if 'cache' in benefit.lower():
                print(f"  {benefit}: {description}")

        return True

    except Exception as e:
        print(f"[ERROR] Cache integration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """メインテスト実行"""
    print("Advanced ML Models System（統合最適化版）テスト開始")
    print("=" * 70)

    test_results = []

    # 各テスト実行
    test_results.append(("システム初期化", await test_advanced_ml_initialization()))
    test_results.append(("高度特徴量エンジニアリング", await test_advanced_feature_extraction()))
    test_results.append(("アンサンブル学習", await test_ensemble_learning()))
    test_results.append(("LSTM時系列予測", await test_lstm_prediction()))
    test_results.append(("バッチML処理", await test_batch_advanced_ml()))
    test_results.append(("パフォーマンス監視", await test_performance_monitoring()))
    test_results.append(("キャッシュ統合", await test_cache_integration()))

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
        print("[SUCCESS] 全テスト成功！Advanced ML Models System準備完了")
        print("\nIssue #315 Phase 3実装成果:")
        print("- LSTM時系列予測モデル（TensorFlow/Keras）")
        print("- アンサンブル学習（Random Forest、Gradient Boosting、Linear Regression）")
        print("- 高度特徴量エンジニアリング（6カテゴリー自動抽出）")
        print("- 統合最適化基盤フル活用（Issues #322-325）")
        print("- バッチ処理・キャッシュ最適化")
        print("- 予測精度向上・リスク調整機能")
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
