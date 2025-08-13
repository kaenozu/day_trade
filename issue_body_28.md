### 問題の概要
`src/day_trade/ml/feature_store.py` の `load_feature` および `save_feature` メソッドにおいて、特徴量データを `pickle` 形式で保存・読み込みしています。特徴量データが主にNumPy配列である場合、`pickle` よりも効率的な形式（例: Parquet, HDF5）が存在します。

### 関連ファイルとメソッド
- `src/day_trade/ml/feature_store.py`
    - `load_feature`
    - `save_feature`

### 具体的な改善提案
`load_feature` および `save_feature` メソッドにおいて、特徴量データが主にNumPy配列である場合、`pickle` の代わりに `numpy.save` / `numpy.load` や、より高速なシリアライザ（例: `joblib`）を使用することを検討します。もし特徴量データが構造化されている場合、ParquetやHDF5のようなカラムナー形式で保存することで、I/O効率が向上します。

### 期待される効果
- 特徴量データの保存・読み込み時間の短縮
- ディスクI/Oの削減
- ファイルサイズの削減
