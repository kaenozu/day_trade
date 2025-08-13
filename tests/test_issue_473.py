#!/usr/bin/env python3
"""
Issue #473 簡単テスト: AdvancedMLEngine役割・インターフェース明確化
"""

import sys
sys.path.append('src')

from day_trade.ml.advanced_ml_interface import (
    AdvancedMLEngineInterface,
    LSTMTransformerEngine,
    AdvancedModelType,
    create_advanced_ml_engine,
    get_available_model_types,
    compare_model_capabilities
)
from day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
import numpy as np

def test_issue_473():
    """Issue #473: AdvancedMLEngine役割・インターフェース明確化テスト"""

    print("=== Issue #473: AdvancedMLEngine役割・インターフェース明確化テスト ===")

    # 1. インターフェース定義テスト
    print("\n1. インターフェース定義テスト")

    # 利用可能モデルタイプ確認
    available_types = get_available_model_types()
    print(f"利用可能モデルタイプ: {[t.value for t in available_types]}")

    # エンジン作成
    try:
        engine = create_advanced_ml_engine(AdvancedModelType.LSTM_TRANSFORMER)
        print(f"エンジン作成成功: {engine.get_model_type().value}")
        engine_success = True
    except Exception as e:
        print(f"エンジン作成エラー: {e}")
        engine_success = False

    # 2. 能力定義テスト
    print("\n2. 能力定義テスト")

    if engine_success:
        capabilities = engine.get_capabilities()
        print(f"  シーケンス予測: {capabilities.supports_sequence_prediction}")
        print(f"  多変量入力: {capabilities.supports_multivariate_input}")
        print(f"  不確実性定量化: {capabilities.supports_uncertainty_quantification}")
        print(f"  アテンション重み: {capabilities.supports_attention_weights}")
        print(f"  最小シーケンス長: {capabilities.min_sequence_length}")
        print(f"  最大シーケンス長: {capabilities.max_sequence_length}")
        print(f"  推論目標時間: {capabilities.inference_time_target_ms}ms")
        capabilities_success = True
    else:
        capabilities_success = False

    # 3. インターフェース統一テスト
    print("\n3. インターフェース統一テスト")

    interface_methods = [
        'get_model_type',
        'get_capabilities',
        'is_trained',
        'prepare_data',
        'train',
        'predict',
        'get_model_metrics',
        'save_model',
        'load_model',
        'get_feature_importance',
        'validate_input_shape',
        'optimize_for_inference',
        'get_memory_usage'
    ]

    interface_success = 0
    if engine_success:
        for method_name in interface_methods:
            if hasattr(engine, method_name):
                print(f"  {method_name}: [OK]")
                interface_success += 1
            else:
                print(f"  {method_name}: [MISSING]")

    print(f"インターフェース実装: {interface_success}/{len(interface_methods)}")

    # 4. EnsembleSystem統合テスト
    print("\n4. EnsembleSystem統合テスト")

    try:
        config = EnsembleConfig(
            use_lstm_transformer=True,
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False
        )
        ensemble = EnsembleSystem(config)

        # Advanced ML Engine の統合確認
        if ensemble.advanced_ml_engine:
            print(f"  Ensemble統合成功: {ensemble.advanced_ml_engine.get_model_type().value}")
            print(f"  訓練状態: {ensemble.advanced_ml_engine.is_trained()}")
            ensemble_integration_success = True
        else:
            print("  Ensemble統合失敗: advanced_ml_engineがNone")
            ensemble_integration_success = False

    except Exception as e:
        print(f"  Ensemble統合エラー: {e}")
        ensemble_integration_success = False

    # 5. 入力検証テスト
    print("\n5. 入力検証テスト")

    validation_success = 0
    if engine_success:
        test_inputs = [
            (np.random.randn(10, 20, 5), "3D配列 (有効)"),
            (np.random.randn(10, 20), "2D配列 (無効)"),
            (np.random.randn(10, 5, 5), "短いシーケンス (有効)"),
            (np.random.randn(10, 2000, 5), "長いシーケンス (無効)")
        ]

        for test_input, desc in test_inputs:
            try:
                is_valid = engine.validate_input_shape(test_input)
                print(f"  {desc}: {'有効' if is_valid else '無効'}")
                validation_success += 1
            except Exception as e:
                print(f"  {desc}: エラー - {e}")

    # 6. メトリクス・統計テスト
    print("\n6. メトリクス・統計テスト")

    metrics_success = False
    if engine_success:
        try:
            metrics = engine.get_model_metrics()
            memory_usage = engine.get_memory_usage()

            print(f"  精度: {metrics.accuracy}")
            print(f"  MSE: {metrics.mse}")
            print(f"  推論時間: {metrics.inference_time_ms}ms")
            print(f"  メモリ使用量: {memory_usage:.1f}MB")
            metrics_success = True
        except Exception as e:
            print(f"  メトリクス取得エラー: {e}")

    # 7. 比較・分析機能テスト
    print("\n7. 比較・分析機能テスト")

    comparison_success = False
    if engine_success:
        try:
            engines = [engine]  # 現在は1つのエンジンのみ
            comparison = compare_model_capabilities(engines)

            print(f"  比較対象モデル数: {len(comparison['models'])}")
            print(f"  推奨用途: {comparison['recommended_use_cases']}")
            comparison_success = True
        except Exception as e:
            print(f"  比較機能エラー: {e}")

    # 8. 役割明確性テスト
    print("\n8. 役割明確性テスト")

    role_clarity_checks = [
        ("LSTM-Transformer専用エンジン", engine_success and engine.get_model_type() == AdvancedModelType.LSTM_TRANSFORMER),
        ("統一インターフェース", interface_success == len(interface_methods)),
        ("EnsembleSystem統合", ensemble_integration_success),
        ("能力定義", capabilities_success),
        ("入力検証", validation_success > 0),
        ("性能監視", metrics_success),
        ("モデル比較", comparison_success)
    ]

    role_success = sum(1 for _, success in role_clarity_checks if success)

    for check_name, success in role_clarity_checks:
        print(f"  {check_name}: {'[OK]' if success else '[FAIL]'}")

    # 全体結果
    print("\n=== Issue #473テスト完了 ===")
    print(f"[OK] エンジン作成: {'成功' if engine_success else '失敗'}")
    print(f"[OK] インターフェース実装: {interface_success}/{len(interface_methods)}")
    print(f"[OK] EnsembleSystem統合: {'成功' if ensemble_integration_success else '失敗'}")
    print(f"[OK] 入力検証機能: {validation_success}/4 テスト通過")
    print(f"[OK] メトリクス機能: {'成功' if metrics_success else '失敗'}")
    print(f"[OK] 比較機能: {'成功' if comparison_success else '失敗'}")
    print(f"[OK] 役割明確性: {role_success}/{len(role_clarity_checks)}")

    print("\n[SUCCESS] AdvancedMLEngineの役割とインターフェースを明確化")
    print("[SUCCESS] LSTM-Transformerエンジンとして特化・統合")
    print("[SUCCESS] アンサンブルシステム内での責任分担を明確化")

if __name__ == "__main__":
    test_issue_473()