#!/usr/bin/env python3
"""
Minimal XGBoost Test - Issue #462対応

最小構成でXGBoost・CatBoostの動作確認
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

from src.day_trade.ml.base_models.xgboost_model import XGBoostModel, XGBoostConfig
from src.day_trade.ml.base_models.catboost_model import CatBoostModel, CatBoostConfig


def generate_minimal_data(n_samples=300):
    """最小テストデータ生成"""
    np.random.seed(42)
    n_features = 8

    X = np.random.randn(n_samples, n_features)
    # シンプルな非線形関係
    y = (
        1.5 * X[:, 0] * X[:, 1] +
        1.0 * np.sin(X[:, 2]) +
        0.8 * X[:, 3] ** 2 +
        0.5 * np.sum(X[:, 4:], axis=1) +
        0.1 * np.random.randn(n_samples)
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]
    return X, y, feature_names


def run_minimal_test():
    """最小構成テスト"""
    print("=" * 50)
    print("Issue #462: 最小XGBoost・CatBoost動作確認")
    print("=" * 50)

    X, y, feature_names = generate_minimal_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    print(f"データサイズ: 訓練={X_train_sub.shape}, 検証={X_val.shape}, テスト={X_test.shape}")

    results = []

    # 1. XGBoostテスト
    print("\n1. XGBoost単体テスト...")
    try:
        start_time = time.time()

        # 最小設定
        xgb_config = XGBoostConfig(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            enable_hyperopt=False  # 最適化オフ
        )

        xgb_model = XGBoostModel(xgb_config)
        xgb_result = xgb_model.fit(
            X_train_sub, y_train_sub,
            validation_data=(X_val, y_val),
            feature_names=feature_names
        )

        # 予測
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred.predictions)
        elapsed = time.time() - start_time

        print(f"  成功: R2={xgb_r2:.4f}, 時間={elapsed:.1f}s")
        results.append(("XGBoost", xgb_r2, elapsed, True))

    except Exception as e:
        print(f"  エラー: {e}")
        results.append(("XGBoost", 0, 0, False))

    # 2. CatBoostテスト
    print("\n2. CatBoost単体テスト...")
    try:
        start_time = time.time()

        # 最小設定
        cb_config = CatBoostConfig(
            iterations=50,
            depth=4,
            learning_rate=0.1,
            enable_hyperopt=False,  # 最適化オフ
            verbose=0
        )

        cb_model = CatBoostModel(cb_config)
        cb_result = cb_model.fit(
            X_train_sub, y_train_sub,
            validation_data=(X_val, y_val),
            feature_names=feature_names
        )

        # 予測
        cb_pred = cb_model.predict(X_test)
        cb_r2 = r2_score(y_test, cb_pred.predictions)
        elapsed = time.time() - start_time

        print(f"  成功: R2={cb_r2:.4f}, 時間={elapsed:.1f}s")
        results.append(("CatBoost", cb_r2, elapsed, True))

    except Exception as e:
        print(f"  エラー: {e}")
        results.append(("CatBoost", 0, 0, False))

    # 結果サマリー
    print("\n" + "=" * 50)
    print("最小テスト結果")
    print("=" * 50)
    print(f"{'モデル':<12} {'R2スコア':<10} {'時間':<8} {'状態'}")
    print("-" * 50)

    for name, r2, time_taken, success in results:
        if success:
            print(f"{name:<12} {r2:8.4f}   {time_taken:6.1f}s  成功")
        else:
            print(f"{name:<12} {'ERROR':<10} {'N/A':<8} 失敗")

    # 成功判定
    success_count = sum(1 for _, _, _, success in results if success)
    total_count = len(results)

    print(f"\n結果: {success_count}/{total_count} モデル成功")

    if success_count == total_count:
        print("[SUCCESS] 全モデル動作確認完了！")
        print("Issue #462 基盤実装完了")
        return True
    else:
        print("[PARTIAL] 一部モデルで問題発生")
        return False


if __name__ == "__main__":
    print("Issue #462: XGBoost・CatBoost 基盤確認")
    print("開始:", time.strftime("%H:%M:%S"))

    success = run_minimal_test()

    if success:
        print("\n[RESULT] XGBoost・CatBoost実装完了！")
        print("次のステップ: アンサンブル統合とベンチマーク")
    else:
        print("\n[RESULT] 追加調整が必要")
        print("個別モデルの問題解決が必要")

    print("完了:", time.strftime("%H:%M:%S"))