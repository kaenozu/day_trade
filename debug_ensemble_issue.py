#!/usr/bin/env python3
"""
Debug Ensemble Issue - Issue #462対応

アンサンブルレベルでの統合エラーを特定・修正
"""

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# プロジェクトルートをパスに追加
from pathlib import Path
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 個別インポートでエラー箇所を特定
try:
    from src.day_trade.ml.base_models.xgboost_model import XGBoostModel, XGBoostConfig
    print("[OK] XGBoostModel import success")
except Exception as e:
    print(f"[ERROR] XGBoostModel import failed: {e}")

try:
    from src.day_trade.ml.base_models.catboost_model import CatBoostModel, CatBoostConfig
    print("[OK] CatBoostModel import success")
except Exception as e:
    print(f"[ERROR] CatBoostModel import failed: {e}")

try:
    from src.day_trade.ml.ensemble_system import EnsembleSystem, EnsembleConfig
    print("[OK] EnsembleSystem import success")
except Exception as e:
    print(f"[ERROR] EnsembleSystem import failed: {e}")


def generate_debug_data(n_samples=200):
    """デバッグ用最小データ"""
    np.random.seed(42)
    n_features = 6

    X = np.random.randn(n_samples, n_features)
    y = (
        1.0 * X[:, 0] * X[:, 1] +
        0.8 * np.sin(X[:, 2]) +
        0.6 * X[:, 3] ** 2 +
        np.sum(X[:, 4:] * 0.3, axis=1) +
        0.05 * np.random.randn(n_samples)
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def debug_step_by_step():
    """段階的デバッグ"""
    print("=" * 60)
    print("Issue #462: アンサンブル統合エラー段階的デバッグ")
    print("=" * 60)

    X, y, feature_names = generate_debug_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    print(f"データ準備完了: 訓練={X_train_sub.shape}, 検証={X_val.shape}, テスト={X_test.shape}")

    # ステップ1: 個別モデル直接テスト
    print("\n--- ステップ1: 個別モデル直接テスト ---")

    # XGBoost直接テスト
    print("XGBoost直接テスト...")
    try:
        xgb_config = XGBoostConfig(n_estimators=20, max_depth=3, learning_rate=0.1, enable_hyperopt=False)
        xgb_model = XGBoostModel(xgb_config)

        xgb_result = xgb_model.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred.predictions)

        print(f"  XGBoost成功: R2={xgb_r2:.4f}")

    except Exception as e:
        print(f"  XGBoost失敗: {e}")
        return False

    # CatBoost直接テスト
    print("CatBoost直接テスト...")
    try:
        cb_config = CatBoostConfig(iterations=20, depth=3, learning_rate=0.1, enable_hyperopt=False, verbose=0)
        cb_model = CatBoostModel(cb_config)

        cb_result = cb_model.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)
        cb_pred = cb_model.predict(X_test)
        cb_r2 = r2_score(y_test, cb_pred.predictions)

        print(f"  CatBoost成功: R2={cb_r2:.4f}")

    except Exception as e:
        print(f"  CatBoost失敗: {e}")
        return False

    # ステップ2: 最小アンサンブル（XGBoostのみ）
    print("\n--- ステップ2: 最小アンサンブル（XGBoostのみ） ---")
    try:
        config = EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False,
            use_xgboost=True,
            use_catboost=False,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            xgboost_params={'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.1, 'enable_hyperopt': False}
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)

        prediction = ensemble.predict(X_test)
        r2 = r2_score(y_test, prediction.final_predictions)

        print(f"  XGBoost単体アンサンブル成功: R2={r2:.4f}")

    except Exception as e:
        print(f"  XGBoost単体アンサンブル失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ステップ3: 最小アンサンブル（CatBoostのみ）
    print("\n--- ステップ3: 最小アンサンブル（CatBoostのみ） ---")
    try:
        config = EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False,
            use_xgboost=False,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            catboost_params={'iterations': 20, 'depth': 3, 'learning_rate': 0.1, 'enable_hyperopt': False, 'verbose': 0}
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)

        prediction = ensemble.predict(X_test)
        r2 = r2_score(y_test, prediction.final_predictions)

        print(f"  CatBoost単体アンサンブル成功: R2={r2:.4f}")

    except Exception as e:
        print(f"  CatBoost単体アンサンブル失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ステップ4: XGBoost + CatBoost アンサンブル
    print("\n--- ステップ4: XGBoost + CatBoost アンサンブル ---")
    try:
        config = EnsembleConfig(
            use_lstm_transformer=False,
            use_random_forest=False,
            use_gradient_boosting=False,
            use_svr=False,
            use_xgboost=True,
            use_catboost=True,
            enable_stacking=False,
            enable_dynamic_weighting=False,
            xgboost_params={'n_estimators': 20, 'max_depth': 3, 'learning_rate': 0.1, 'enable_hyperopt': False},
            catboost_params={'iterations': 20, 'depth': 3, 'learning_rate': 0.1, 'enable_hyperopt': False, 'verbose': 0}
        )

        ensemble = EnsembleSystem(config)
        ensemble.fit(X_train_sub, y_train_sub, validation_data=(X_val, y_val), feature_names=feature_names)

        prediction = ensemble.predict(X_test)
        r2 = r2_score(y_test, prediction.final_predictions)

        print(f"  XGBoost+CatBoostアンサンブル成功: R2={r2:.4f}")
        print("  [SUCCESS] 全ステップクリア！統合問題解決！")

        return True

    except Exception as e:
        print(f"  XGBoost+CatBoostアンサンブル失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Issue #462: アンサンブル統合エラー解析開始")
    print("開始:", time.strftime("%H:%M:%S"))

    success = debug_step_by_step()

    if success:
        print("\n[RESULT] 統合エラー解決完了！")
        print("XGBoost・CatBoostアンサンブル統合成功")
    else:
        print("\n[RESULT] 統合エラーが残存")
        print("詳細なデバッグが必要")

    print("完了:", time.strftime("%H:%M:%S"))