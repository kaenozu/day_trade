### 問題の概要
`src/day_trade/ml/base_models/gradient_boosting_model.py` の `fit` および `predict` メソッド内で、`X.copy()` が使用されています。`StandardScaler` の `fit_transform` や `transform` は新しい配列を返すため、このコピーは不要である可能性があります。大規模なデータセットの場合、データのコピーはメモリとCPUのオーバーヘッドになります。

### 関連ファイルとメソッド
- `src/day_trade/ml/base_models/gradient_boosting_model.py`
    - `fit`
    - `predict`

### 具体的な改善提案
`fit` および `predict` メソッド内の `X.copy()` を削除し、`StandardScaler` の `fit_transform` や `transform` の結果を直接使用することで、このコピーのオーバーヘッドを削減できます。

**変更例（`fit` メソッド内）:**
```python
# ...
            # 特徴量正規化
            if self.scaler is not None:
                X_processed = self.scaler.fit_transform(X) # X_scaled = X.copy() を削除
            else:
                X_processed = X # X_scaled = X.copy() を削除

            # ハイパーパラメータ最適化
            if self.config['enable_hyperopt']:
                logger.info("ハイパーパラメータ最適化開始")
                self.model = self._hyperparameter_optimization(X_processed, y) # X_scaled を X_processed に変更
            else:
                # デフォルトパラメータで学習
                self.model = GradientBoostingRegressor(**self._get_gbm_params())
                self.model.fit(X_processed, y) # X_scaled を X_processed に変更
# ...
```
`predict` メソッドも同様に修正します。

### 期待される効果
- メモリ使用量の削減
- 処理時間の短縮
