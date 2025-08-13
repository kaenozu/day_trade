### 問題の概要
`src/day_trade/ml/data_drift_detector.py` の `save_baseline` および `load_baseline` メソッドにおいて、ベースライン統計情報をJSON形式で保存・読み込みしています。もしベースラインデータが非常に大きい場合、JSONのシリアライズ/デシリアライズはI/OとCPUのオーバーヘッドになります。

### 関連ファイルとメソッド
- `src/day_trade/ml/data_drift_detector.py`
    - `save_baseline`
    - `load_baseline`

### 具体的な改善提案
もしベースラインデータが非常に大きい場合、`save_baseline` および `load_baseline` メソッドにおいて、JSONではなく、より効率的なバイナリ形式（例: Parquet, HDF5, Joblib）で保存することを検討してください。これにより、I/Oパフォーマンスが向上し、ファイルサイズも削減できる可能性があります。

### 期待される効果
- 大規模ベースラインデータの保存・読み込み時間の短縮
- ファイルサイズの削減
