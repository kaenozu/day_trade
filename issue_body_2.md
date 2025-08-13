### 問題の概要
`src/day_trade/ml/stacking_ensemble.py` の `_optimize_meta_learner_hyperparams` メソッドでは、メタ学習器のハイパーパラメータ最適化に `GridSearchCV` を使用しています。`GridSearchCV` は探索空間が大きい場合に計算コストが非常に高くなる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/stacking_ensemble.py`
    - `_optimize_meta_learner_hyperparams`

### 具体的な改善提案
`_optimize_meta_learner_hyperparams` メソッドにおいて、`GridSearchCV` の代わりに `Optuna` や `Hyperopt` のようなベイズ最適化ライブラリの導入を検討してください。これにより、探索回数を減らしつつ、より良いハイパーパラメータを見つけられる可能性が高まります。

### 期待される効果
- ハイパーパラメータ探索時間の短縮
- より効率的なハイパーパラメータの発見
