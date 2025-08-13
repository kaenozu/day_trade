### 問題の概要
`src/day_trade/ml/model_quantization_engine.py` の `ONNXQuantizationEngine` クラス内の `apply_mixed_precision_quantization` メソッドは、現在ONNX Runtimeのグラフ最適化レベルを設定するだけです。真のFP16量子化を行うには、モデルの重みをFP16に変換する必要があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/model_quantization_engine.py`
    - `apply_mixed_precision_quantization`

### 具体的な改善提案
`ONNXQuantizationEngine` クラス内の `apply_mixed_precision_quantization` メソッドは、現在ONNX Runtimeのグラフ最適化レベルを設定するだけです。真のFP16量子化を行うには、モデルの重みをFP16に変換する必要があります。PyTorchやTensorFlowの量子化ツールを使用するか、ONNX Runtimeの `quantize_static` で `QuantType.Float16` を指定することを検討します。

### 期待される効果
- FP16量子化によるモデルサイズ削減
- 推論速度の向上
