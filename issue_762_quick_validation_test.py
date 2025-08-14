#!/usr/bin/env python3
"""
Issue #762 高度なアンサンブルシステム動作確認テスト
Quick Validation Test for Advanced Ensemble System
"""

import asyncio
import numpy as np
import sys
import os
import warnings

# パス追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 警告抑制
warnings.filterwarnings('ignore')

async def test_advanced_ensemble_system():
    """高度なアンサンブルシステム動作確認"""

    print("="*60)
    print("Issue #762 高度なアンサンブルシステム動作確認テスト")
    print("="*60)

    try:
        # インポートテスト
        print("\n1. モジュールインポートテスト...")
        from day_trade.ensemble.advanced_ensemble import AdvancedEnsembleSystem, create_and_train_ensemble
        from day_trade.ensemble import (
            AdaptiveWeightingEngine,
            MetaLearnerEngine,
            EnsembleOptimizer,
            EnsembleAnalyzer
        )
        print("✓ 全モジュールのインポート成功")

        # テストデータ生成
        print("\n2. テストデータ生成...")
        np.random.seed(42)
        n_samples = 200
        n_features = 8

        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features, 1)
        y = (X @ true_weights + np.random.randn(n_samples, 1) * 0.1).flatten()

        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        print(f"✓ データ生成完了: 訓練{X_train.shape}, テスト{X_test.shape}")

        # システム初期化テスト
        print("\n3. システム初期化テスト...")
        system = AdvancedEnsembleSystem(
            enable_optimization=False,  # 高速化のため無効
            enable_analysis=False       # 高速化のため無効
        )
        print("✓ システム初期化成功")

        # 学習テスト
        print("\n4. システム学習テスト...")
        await system.fit(X_train, y_train)
        print("✓ システム学習完了")

        # 予測テスト
        print("\n5. 予測機能テスト...")
        result = await system.predict(X_test)

        print(f"✓ 予測完了:")
        print(f"  - 予測数: {result.predictions.shape[0]}")
        print(f"  - 平均信頼度: {np.mean(result.confidence_scores):.3f}")
        print(f"  - 処理時間: {result.processing_time:.3f}秒")
        print(f"  - 個別モデル数: {len(result.individual_predictions)}")

        # パフォーマンス計算
        print("\n6. パフォーマンステスト...")
        mse = np.mean((result.predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(result.predictions.flatten() - y_test))

        print(f"✓ パフォーマンス指標:")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - MAE: {mae:.4f}")

        # システム状態確認
        print("\n7. システム状態確認...")
        status = system.get_system_status()
        print(f"✓ システム状態:")
        print(f"  - 学習済み: {status['is_fitted']}")
        print(f"  - モデル数: {status['n_models']}")
        print(f"  - 有効コンポーネント: {sum(status['components'].values())}")

        # 保存・読み込みテスト
        print("\n8. 保存・読み込みテスト...")
        save_path = "test_ensemble_system.pkl"
        system.save_system(save_path)

        loaded_system = AdvancedEnsembleSystem.load_system(save_path)

        # 読み込み後予測テスト
        loaded_result = await loaded_system.predict(X_test[:5])

        # ファイル削除
        if os.path.exists(save_path):
            os.remove(save_path)

        print("✓ 保存・読み込み機能正常")

        print("\n" + "="*60)
        print("🎉 Issue #762 高度なアンサンブルシステム動作確認完了!")
        print("✅ 全ての基本機能が正常に動作しています")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_component_integration():
    """コンポーネント統合テスト"""

    print("\n" + "-"*40)
    print("コンポーネント統合テスト")
    print("-"*40)

    try:
        # 個別コンポーネントテスト
        print("\n1. 動的重み付けエンジンテスト...")
        weighting_engine = AdaptiveWeightingEngine(n_models=3)
        print("✓ AdaptiveWeightingEngine初期化成功")

        print("\n2. メタ学習エンジンテスト...")
        meta_learner = MetaLearnerEngine(input_dim=5)
        print("✓ MetaLearnerEngine初期化成功")

        print("\n3. アンサンブル最適化エンジンテスト...")
        optimizer = EnsembleOptimizer(optimization_budget=10)
        print("✓ EnsembleOptimizer初期化成功")

        print("\n4. パフォーマンス分析エンジンテスト...")
        analyzer = EnsembleAnalyzer()
        print("✓ EnsembleAnalyzer初期化成功")

        print("\n✅ 全コンポーネントの初期化完了")

        return True

    except Exception as e:
        print(f"\n❌ コンポーネントテストエラー: {e}")
        return False

async def main():
    """メインテスト実行"""

    print("Issue #762 高度なアンサンブル予測システム")
    print("動作確認テスト開始...")

    # 基本動作テスト
    basic_test_result = await test_advanced_ensemble_system()

    # コンポーネント統合テスト
    component_test_result = await test_component_integration()

    # 最終結果
    if basic_test_result and component_test_result:
        print("\n🎯 総合結果: ✅ 全テスト成功")
        print("\nIssue #762 の実装は正常に動作しています!")
        print("高度なアンサンブル予測システムの準備が完了しました。")
    else:
        print("\n❌ 総合結果: 一部テスト失敗")

if __name__ == "__main__":
    asyncio.run(main())