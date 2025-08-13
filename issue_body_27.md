### 問題の概要
`src/day_trade/ml/feature_store.py` の `_load_metadata_index` および `_save_metadata_index` メソッドにおいて、メタデータインデックスをJSONファイルとして保存・読み込みしています。メタデータインデックスが非常に大きくなる場合（多数の特徴量が保存される場合）、これらの操作がボトルネックになる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/feature_store.py`
    - `_load_metadata_index`
    - `_save_metadata_index`

### 具体的な改善提案
`_load_metadata_index` および `_save_metadata_index` メソッドにおいて、メタデータインデックスが非常に大きくなる場合、JSONではなく、より効率的なバイナリ形式（例: Parquet, HDF5）で保存することを検討します。あるいは、メタデータをデータベース（SQLiteなど）に保存し、必要な情報だけをクエリで取得するように変更することで、メモリ使用量とI/Oを最適化できます。

### 期待される効果
- 大規模なメタデータインデックスの読み込み・保存時間の短縮
- メモリ使用量の最適化
