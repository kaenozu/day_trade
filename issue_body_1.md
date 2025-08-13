### 問題の概要
`src/day_trade/ml/stacking_ensemble.py` の `_generate_meta_features` メソッド内で、ベースモデルのコピーに `_copy_model` メソッドが使用されています。現在の実装では、`model_class(model.config)` のように新しいインスタンスを作成していますが、特に複雑なモデルの場合、インスタンス化のオーバーヘッドが大きくなる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/stacking_ensemble.py`
    - `_generate_meta_features`
    - `_copy_model`

### 具体的な改善提案
`_copy_model` メソッドにおいて、`sklearn.base.clone` を使用してモデルのコピーをより効率的に行うことを検討してください。これにより、特に複雑なベースモデルを使用する場合のオーバーヘッドを削減できます。

**変更例（`_copy_model` メソッド内）:**
```python
from sklearn.base import clone

def _copy_model(self, model: BaseModelInterface) -> BaseModelInterface:
    """モデルのコピー作成"""
    # sklearn.base.clone を使用して、より効率的にモデルをコピー
    return clone(model)
```
ただし、`BaseModelInterface` が `sklearn.base.BaseEstimator` を継承しているか、または `clone` 関数が期待するインターフェース（`get_params` と `set_params` メソッド）を実装している必要があります。

### 期待される効果
- 複雑なベースモデルを使用する際の学習時間の短縮
- メモリ使用量の最適化
