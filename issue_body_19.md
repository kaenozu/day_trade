### 問題の概要
`src/day_trade/ml/advanced_ml_models.py` の `_prepare_lstm_data` メソッド内のシーケンスデータ作成において、Pythonのループと `append` を繰り返す方法が使用されています。大規模なデータセットの場合、この方法はメモリとCPUのオーバーヘッドが大きくなる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/advanced_ml_models.py`
    - `_prepare_lstm_data`

### 具体的な改善提案
`_prepare_lstm_data` メソッド内のシーケンスデータ作成において、Pythonのループと `append` を繰り返す代わりに、NumPyの `stride_tricks` を使用してメモリコピーなしでシーケンスビューを作成することを検討してください。

### 期待される効果
- 大規模データセットでのLSTMデータ準備時間の短縮
- メモリ使用量の削減
