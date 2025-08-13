### 問題の概要
`src/day_trade/ml/feature_pipeline.py` の `_cpu_batch_features` メソッド内のループはPythonレベルで実行されるため、大規模なデータセットの場合にボトルネックになる可能性があります。このメソッドはCPUフォールバックとして機能します。

### 関連ファイルとメソッド
- `src/day_trade/ml/feature_pipeline.py`
    - `_cpu_batch_features`

### 具体的な改善提案
`_cpu_batch_features` メソッド内のループはPythonレベルで実行されるため、大規模なデータセットの場合にボトルネックになる可能性があります。もしこのフォールバックが頻繁に利用され、パフォーマンスがボトルネックになる場合、NumPyのベクトル化された操作やNumba/Cythonを使用して、このループを最適化することを検討します。

### 期待される効果
- CPUフォールバック時の特徴量生成時間の短縮
- CPUリソースの有効活用
