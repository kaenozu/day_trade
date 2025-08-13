### 問題の概要
`src/day_trade/ml/feature_store.py` の `batch_generate_features` メソッド内のループは順次実行されるため、多数のシンボルを処理する場合、ボトルネックになります。

### 関連ファイルとメソッド
- `src/day_trade/ml/feature_store.py`
    - `batch_generate_features`

### 具体的な改善提案
`batch_generate_features` メソッド内のループを並列化することを強く推奨します。`joblib.Parallel` や `multiprocessing` を使用して、各シンボルの特徴量生成を並行して実行することで、全体の実行時間を大幅に短縮できます。

### 期待される効果
- バッチ特徴量生成時間の短縮
- CPUリソースの有効活用
