### 問題の概要
`src/day_trade/ml/base_models/gradient_boosting_model.py` の `_get_staged_predictions` メソッドでは、`scikit-learn` の `GradientBoostingRegressor` の `staged_predict` を使用して段階別予測を取得しています。この方法は、イテレータを最後まで回す必要があり、不要な計算が発生する可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/base_models/gradient_boosting_model.py`
    - `_get_staged_predictions`

### 具体的な改善提案
`scikit-learn` の `GradientBoostingRegressor` の `staged_predict` の代わりに、`model.predict(X, n_estimators=stage)` のように、特定のイテレーション数までの予測を直接取得できる方法がないか確認してください。これにより、不要な計算を避けることができます。

### 期待される効果
- 段階別予測計算の高速化
- 計算リソースの削減
