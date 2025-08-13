### 問題の概要
`src/day_trade/ml/base_models/random_forest_model.py` の `_hyperparameter_optimization` メソッド内の `GridSearchCV` の `n_jobs` が `1` に設定されています。`RandomForestRegressor` 自体は `n_jobs=-1` で並列化されますが、`GridSearchCV` が `n_jobs=1` だと、ハイパーパラメータの各組み合わせの評価が直列で行われるため、全体の探索時間が長くなります。

### 関連ファイルとメソッド
- `src/day_trade/ml/base_models/random_forest_model.py`
    - `_hyperparameter_optimization`

### 具体的な改善提案
`_hyperparameter_optimization` メソッド内の `GridSearchCV` の `n_jobs` パラメータを `-1` に変更してください。これにより、利用可能なすべてのCPUコアを使用してハイパーパラメータ探索を並列化し、訓練時間を大幅に短縮できます。

**変更例（`_hyperparameter_optimization` メソッド内）:**
```python
# ...
            grid_search = GridSearchCV(
                rf_base,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,  # ここを-1に変更
                verbose=self.config['verbose']
            )
# ...
```

### 期待される効果
- ハイパーパラメータ探索時間の短縮
- CPUリソースの有効活用
