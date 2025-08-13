### 問題の概要
`src/day_trade/ml/optimized_inference_engine.py` の `ONNXModelOptimizer` クラス内の `convert_pytorch_to_onnx` メソッドは、「PyTorch ONNX変換は手動実装が必要」とコメントされています。PyTorchモデルをONNXに変換する機能を完全に実装することで、PyTorchベースのモデルもこの最適化推論エンジンの恩恵を最大限に受けられるようになります。

### 関連ファイルとメソッド
- `src/day_trade/ml/optimized_inference_engine.py`
    - `ONNXModelOptimizer.convert_pytorch_to_onnx`

### 具体的な改善提案
`ONNXModelOptimizer` クラス内の `convert_pytorch_to_onnx` メソッドは、「PyTorch ONNX変換は手動実装が必要」とコメントされています。PyTorchモデルをONNXに変換する機能を完全に実装することで、PyTorchベースのモデルもこの最適化推論エンジンの恩恵を最大限に受けられるようになります。

### 期待される効果
- PyTorchモデルのONNX最適化推論への対応
- 推論パフォーマンスの向上