### 問題の概要
`src/day_trade/ml/data_drift_detector.py` の `_calculate_statistics` メソッド内の "values": series.tolist() の行で、Pandas Seriesの全値をPythonのリストに変換しています。データセットが非常に大きい場合、この変換はメモリとCPUのオーバーヘッドになります。特に、`detect_drift` メソッドで `baseline_stat["values"]` をNumPy配列に変換し直しているため、二重の変換コストが発生します。

### 関連ファイルとメソッド
- `src/day_trade/ml/data_drift_detector.py`
    - `_calculate_statistics`

### 具体的な改善提案
`_calculate_statistics` メソッド内の "values": series.tolist() の行を "values": series.values に変更してください。これにより、不要なリストへの変換と、その後のNumPy配列への再変換のオーバーヘッドを削減できます。

**変更例（`_calculate_statistics` メソッド内）:**
```python
# ...
                    stats[col] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "min": series.min(),
                        "max": series.max(),
                        "median": series.median(),
                        "values": series.values,  # ここを series.values に変更
                    }
    # ...
```

### 期待される効果
- データ統計計算の高速化
- メモリ使用量の削減
