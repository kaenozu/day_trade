### 問題の概要
`src/day_trade/ml/batch_inference_optimizer.py` の `AdaptiveBatchSizer` クラス内の `_optimize_for_latency`, `_optimize_for_throughput`, `_optimize_balanced`, `_adaptive_optimization` メソッドで `statistics.mean` を使用しています。NumPyはCで実装されているため、Pythonの `statistics` モジュールよりも高速です。

### 関連ファイルとメソッド
- `src/day_trade/ml/batch_inference_optimizer.py`
    - `AdaptiveBatchSizer`
    - `_optimize_for_latency`
    - `_optimize_for_throughput`
    - `_optimize_balanced`
    - `_adaptive_optimization`

### 具体的な改善提案
`AdaptiveBatchSizer` クラス内の `_optimize_for_latency`, `_optimize_for_throughput`, `_optimize_balanced`, `_adaptive_optimization` メソッドで `statistics.mean` を使用している箇所を `numpy.mean` に置き換えることを検討してください。

**変更例（`_optimize_for_latency` メソッド内）:**
```python
# ...
            # avg_latency = statistics.mean(latencies) # 変更前
            avg_latency = np.mean(latencies) # 変更後
# ...
```
同様の修正を他の関連するメソッドにも適用します。

### 期待される効果
- バッチサイズ最適化計算の高速化
- CPUリソースの有効活用
