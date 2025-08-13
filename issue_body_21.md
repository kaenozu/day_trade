### 問題の概要
`src/day_trade/ml/concept_drift_detector.py` の `ConceptDriftDetector` クラスでは、`performance_history` をPythonのリストで管理し、`add_performance_data` メソッド内で `self.performance_history = self.performance_history[-self.window_size :]` のようにスライスと再代入を行っています。`window_size` が非常に大きい場合、この操作はオーバーヘッドになる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/concept_drift_detector.py`
    - `__init__`
    - `add_performance_data`
    - `get_performance_summary`

### 具体的な改善提案
`__init__` メソッドで `self.performance_history` を `collections.deque` に変更し、`add_performance_data` メソッドで `append` を使用するようにします。これにより、履歴の管理がより効率的になり、`add_performance_data` 内のリストスライスのオーバーヘッドがなくなります。

**変更例（`__init__` メソッド内）:**
```python
from collections import deque
# ...
        self.performance_history: deque = deque(maxlen=window_size) # 変更
```

**変更例（`add_performance_data` メソッド内）:**
```python
# ...
        self.performance_history.append(
            {
                "timestamp": timestamp.isoformat(),
                "mae": mae,
                "rmse": rmse,
                "num_samples": len(predictions),
            }
        )

        # 履歴を最新のwindow_sizeに保つ (dequeを使用するため不要になる)
        # self.performance_history = self.performance_history[-self.window_size :] # 削除
```

### 期待される効果
- 性能履歴管理の効率化
- 大規模な履歴データでのオーバーヘッド削減
