#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡単なハイパーパラメータ最適化テスト
"""

import asyncio
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from hyperparameter_optimizer import HyperparameterOptimizer, ModelType, PredictionTask

async def simple_optimization_test():
    print("=== 簡単ハイパーパラメータ最適化テスト ===")

    # 簡単なテストデータ
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    print(f"データ: {X.shape}, ターゲット: {len(y)}")

    # 簡単なパラメータ空間
    simple_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }

    # Grid Search実行
    model = RandomForestClassifier(random_state=42)
    optimizer = GridSearchCV(
        model,
        simple_params,
        cv=3,
        scoring='accuracy',
        verbose=1
    )

    print("\n最適化実行中...")
    optimizer.fit(X, y)

    print(f"\n結果:")
    print(f"最適スコア: {optimizer.best_score_:.4f}")
    print(f"最適パラメータ: {optimizer.best_params_}")

    # システム統合テスト
    print(f"\n[ システム統合テスト ]")

    try:
        hyperopt = HyperparameterOptimizer()

        # 制限されたパラメータ空間でテスト
        limited_space = hyperopt._limit_param_space(simple_params, 2)
        print(f"制限パラメータ空間: {limited_space}")

        result = await hyperopt.optimize_model(
            "TEST", ModelType.RANDOM_FOREST, PredictionTask.PRICE_DIRECTION,
            X, y, baseline_score=0.5, method='grid'
        )

        print(f"\nシステム最適化結果:")
        print(f"  スコア: {result.best_score:.4f}")
        print(f"  改善率: {result.improvement:.2f}%")
        print(f"  時間: {result.optimization_time:.1f}秒")
        print(f"  最適パラメータ: {result.best_params}")

    except Exception as e:
        print(f"❌ システムテストエラー: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n=== テスト完了 ===")

if __name__ == "__main__":
    asyncio.run(simple_optimization_test())