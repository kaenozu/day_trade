#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小ハイパーパラメータ最適化テスト
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

def minimal_test():
    print("=== 最小ハイパーパラメータテスト ===")

    # 小さなデータセット
    X, y = make_classification(n_samples=50, n_features=5, n_informative=2, random_state=42)

    # 最小パラメータ空間
    params = {
        'n_estimators': [10, 20],
        'max_depth': [3, 5]
    }

    print(f"データ: {X.shape}")
    print(f"パラメータ: {params}")

    # 最適化実行
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, params, cv=2, verbose=1)

    print("\n最適化実行...")
    grid_search.fit(X, y)

    print(f"\n結果:")
    print(f"最適スコア: {grid_search.best_score_:.4f}")
    print(f"最適パラメータ: {grid_search.best_params_}")

    # ベースライン比較
    baseline_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    baseline_model.fit(X, y)
    baseline_score = baseline_model.score(X, y)

    improvement = (grid_search.best_score_ - baseline_score) / baseline_score * 100
    print(f"改善率: {improvement:.2f}%")

    print(f"\n✅ ハイパーパラメータ最適化システム動作確認完了")

if __name__ == "__main__":
    minimal_test()