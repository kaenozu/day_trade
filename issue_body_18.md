### 問題の概要
`src/day_trade/ml/advanced_ml_models.py` の `_train_ensemble_models` メソッド内で、各モデルの訓練が直列で行われています。アンサンブル内の各モデルの訓練は独立しているため、並列化の大きな機会があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/advanced_ml_models.py`
    - `_train_ensemble_models`

### 具体的な改善提案
`_train_ensemble_models` メソッド内で、各モデルの訓練を `joblib.Parallel` や `multiprocessing` を使用して並列化することを強く推奨します。これにより、アンサンブル訓練の時間を大幅に短縮できます。

**変更例（`_train_ensemble_models` メソッド内）:**
```python
from joblib import Parallel, delayed

# ...

async def _train_ensemble_models(
    self, X_train: np.ndarray, y_train: np.ndarray, symbol: str
) -> Dict[str, Any]:
    models = {}

    # 各モデルの訓練を並列実行
    results = Parallel(n_jobs=self.max_concurrent)(
        delayed(self._train_single_ensemble_model)(model_type, X_train, y_train)
        for model_type in ["random_forest", "gradient_boosting", "linear_regression", "extra_trees"]
    )

    for model_name, model_instance in results:
        if model_instance:
            models[model_name] = model_instance

    return models

def _train_single_ensemble_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[str, Any]:
    try:
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=3)
        elif model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "extra_trees":
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=7, bootstrap=False, max_features="sqrt")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        return model_type, model
    except Exception as e:
        logger.warning(f"アンサンブルモデル訓練エラー ({model_type}): {e}")
        return model_type, None
```

### 期待される効果
- アンサンブルモデル訓練時間の短縮
- CPUリソースの有効活用
