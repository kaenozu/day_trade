### 問題の概要
`src/day_trade/ml/hybrid_lstm_transformer.py` の `TimeSeriesDataset` の `__init__` メソッドで、NumPy配列を `torch.FloatTensor` に変換しています。また、`_train_pytorch_hybrid` メソッド内で、GPU上のテンソルを一度CPUのNumPy配列に戻してから `TimeSeriesDataset` に渡しており、非効率的です。

### 関連ファイルとメソッド
- `src/day_trade/ml/hybrid_lstm_transformer.py`
    - `TimeSeriesDataset`
    - `_train_pytorch_hybrid`

### 具体的な改善提案
`TimeSeriesDataset` を `torch.Tensor` を直接受け取るように変更し、`_train_pytorch_hybrid` メソッド内で `DataLoader` を使用する際に、`pin_memory=True` と `num_workers` を設定することを検討してください。これにより、CPUとGPU間のデータ転送のボトルネックを軽減し、データローディングを並列化できます。

### 期待される効果
- データローディング時間の短縮
- GPU利用率の向上
