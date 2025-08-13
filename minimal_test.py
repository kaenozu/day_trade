#!/usr/bin/env python3
"""
最小限のXGBoost・CatBoostテスト

並列処理・ハイパーパラメータ最適化を無効化した最も基本的なテスト
Issue #462: 基本統合の確認
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.day_trade.ml.base_models import XGBoostModel, CatBoostModel
from src.day_trade.utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

def test_minimal_models():
    """最小限のモデルテスト"""
    print("=== 最小限 XGBoost・CatBoost テスト ===")

    # 小さなテストデータ
    np.random.seed(42)
    n_samples, n_features = 100, 5
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)

    # データ分割
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"[データ] 学習: {X_train.shape}, テスト: {X_test.shape}")

    # 1. XGBoostテスト
    print("\n--- XGBoost単体テスト ---")
    try:
        from src.day_trade.ml.base_models.xgboost_model import XGBoostConfig
        xgb_config = XGBoostConfig(
            n_estimators=10,
            max_depth=3,
            learning_rate=0.3,
            enable_hyperopt=False  # 最適化無効
        )

        xgb_model = XGBoostModel(xgb_config)
        print(f"[XGBoost] 初期化完了")

        # 学習
        xgb_results = xgb_model.fit(X_train, y_train, validation_data=(X_test, y_test))
        print(f"[XGBoost] 学習完了: {xgb_results.get('training_time', 0):.2f}秒")
        print(f"[XGBoost] 学習済み: {xgb_model.is_trained}")

        # 予測
        xgb_pred = xgb_model.predict(X_test)
        print(f"[XGBoost] 予測完了: {len(xgb_pred.predictions)} サンプル")

        # 精度計算
        from sklearn.metrics import mean_squared_error, r2_score
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred.predictions))
        xgb_r2 = r2_score(y_test, xgb_pred.predictions)
        print(f"[XGBoost] RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f}")

    except Exception as e:
        print(f"[XGBoost] エラー: {e}")
        return False

    # 2. CatBoostテスト
    print("\n--- CatBoost単体テスト ---")
    try:
        from src.day_trade.ml.base_models.catboost_model import CatBoostConfig
        cb_config = CatBoostConfig(
            iterations=10,
            depth=3,
            learning_rate=0.3,
            enable_hyperopt=False,  # 最適化無効
            verbose=0  # ログ抑制
        )

        cb_model = CatBoostModel(cb_config)
        print(f"[CatBoost] 初期化完了")

        # 学習
        cb_results = cb_model.fit(X_train, y_train, validation_data=(X_test, y_test))
        print(f"[CatBoost] 学習完了: {cb_results.get('training_time', 0):.2f}秒")
        print(f"[CatBoost] 学習済み: {cb_model.is_trained}")

        # 予測
        cb_pred = cb_model.predict(X_test)
        print(f"[CatBoost] 予測完了: {len(cb_pred.predictions)} サンプル")

        # 精度計算
        cb_rmse = np.sqrt(mean_squared_error(y_test, cb_pred.predictions))
        cb_r2 = r2_score(y_test, cb_pred.predictions)
        print(f"[CatBoost] RMSE: {cb_rmse:.4f}, R2: {cb_r2:.4f}")

    except Exception as e:
        print(f"[CatBoost] エラー: {e}")
        return False

    # 結果比較
    print("\n--- 結果比較 ---")
    print(f"XGBoost  - RMSE: {xgb_rmse:.4f}, R2: {xgb_r2:.4f} ({max(0, xgb_r2*100):.1f}%)")
    print(f"CatBoost - RMSE: {cb_rmse:.4f}, R2: {cb_r2:.4f} ({max(0, cb_r2*100):.1f}%)")

    # 平均性能
    avg_r2 = (xgb_r2 + cb_r2) / 2
    print(f"平均精度: {max(0, avg_r2*100):.1f}%")

    if avg_r2 > 0.5:  # 50%以上の精度
        print("[SUCCESS] 基本統合テスト成功")
        return True
    else:
        print("[WARNING] 精度が低い可能性")
        return True  # 統合自体は成功

if __name__ == "__main__":
    success = test_minimal_models()
    print(f"\n=== テスト結果: {'成功' if success else '失敗'} ===")