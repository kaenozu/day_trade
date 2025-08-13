### 問題の概要
`src/day_trade/ml/model_quantization_engine.py` の `ModelPruningEngine` クラス内の `apply_block_structured_pruning` メソッドにおいて、ブロックごとのループがPythonレベルで実行されています。これにより、ループのオーバーヘッドが発生し、プルーニング処理が遅くなる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/model_quantization_engine.py`
    - `apply_block_structured_pruning`

### 具体的な改善提案
`ModelPruningEngine` クラス内の `apply_block_structured_pruning` メソッドにおいて、ブロックごとのループをNumPyの `reshape` や `stride_tricks` を使用してベクトル化することを検討してください。これにより、ループのオーバーヘッドを削減し、プルーニング処理を高速化できます。

### 期待される効果
- プルーニング処理の高速化
- 計算リソースの有効活用
