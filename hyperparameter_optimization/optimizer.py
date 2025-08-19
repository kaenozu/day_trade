#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Optimizer using Optuna
Issue #939対応: 動的なハイパーパラメータ最適化
"""

import optuna
import lightgbm as lgb
import polars as pl
import numpy as np
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.day_trade.data.providers.real_data_provider import ImprovedMultiSourceDataProvider

# 特徴量生成ロジック (models.pyから流用・簡略化)
def create_features(data: pl.DataFrame) -> pl.DataFrame:
    features = data.with_columns([
        pl.col("Close").rolling_mean(window_size=5).alias("sma5"),
        pl.col("Close").rolling_mean(window_size=25).alias("sma25"),
        pl.col("Close").diff().clip_lower(0).rolling_mean(14).alias("gain"),
        (-pl.col("Close").diff().clip_upper(0)).rolling_mean(14).alias("loss"),
    ])
    features = features.with_columns([
        (100 - (100 / (1 + pl.col("gain") / pl.col("loss")))).alias("rsi")
    ])
    return features.drop_nulls()

# 目的変数生成 (例: 3日後の株価が上がるか下がるか)
def create_target(data: pl.DataFrame) -> pl.DataFrame:
    future_price = data['Close'].shift(-3)
    target = (future_price > data['Close']).cast(pl.Int8)
    return data.with_columns(target.alias('target')).drop_nulls()

def objective(trial: optuna.Trial, data: pl.DataFrame) -> float:
    """Optunaの目的関数"""
    # ハイパーパラメータの探索空間を定義
    params = {
        'objective': 'multiclass',
        'num_class': 3, # BUY, SELL, HOLD
        'metric': 'multi_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }

    # 1. 特徴量と目的変数の作成
    features = create_features(data)
    full_data = create_target(features)

    X = full_data.drop(['Date', 'target', 'gain', 'loss'])
    y = full_data['target']

    # 2. 訓練データと検証データに分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. モデルの学習
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train.to_pandas(), y_train.to_pandas()) # LGBMはPandasを要求

    # 4. モデルの評価
    preds = model.predict(X_val.to_pandas())
    accuracy = accuracy_score(y_val.to_pandas(), preds)

    return accuracy

def main():
    """最適化を実行するメイン関数"""
    print("Starting hyperparameter optimization...")

    # データの取得 (例: トヨタ自動車)
    print("Fetching data...")
    provider = ImprovedMultiSourceDataProvider()
    result = provider.get_stock_data_sync("7203.T", period="1y")
    if not result.success or result.data is None:
        print("Failed to fetch data. Aborting.")
        return

    data = pl.from_pandas(result.data.reset_index())

    # OptunaのStudyを作成して最適化を開始
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data), n_trials=50) # 試行回数50回

    print("Optimization finished.")
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 最適なパラメータをJSONファイルに保存
    best_params_path = Path(__file__).parent / 'best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(trial.params, f, indent=4)
    
    print(f"Best parameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
