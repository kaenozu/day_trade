### 問題の概要
`src/day_trade/ml/deep_learning_models.py` および `src/day_trade/ml/hybrid_lstm_transformer.py` では、PyTorchが利用できない場合にNumPyベースの簡易実装にフォールバックするロジックが実装されています。これらのNumPy実装は「簡易実装」であり、実際の深層学習モデルのような性能や収束速度は期待できません。

### 関連ファイルとメソッド
- `src/day_trade/ml/deep_learning_models.py`
- `src/day_trade/ml/hybrid_lstm_transformer.py`

### 具体的な改善提案
NumPyフォールバック実装は、PyTorchが利用できない場合の簡易的な代替手段としてのみ使用し、本番環境やパフォーマンスが重要な場面ではPyTorchのインストールを必須とすることを強く推奨します。NumPy実装のパフォーマンスがボトルネックになる場合、NumbaやCythonの利用を検討するか、PyTorchのインストールを強制するべきです。

### 期待される効果
- 本番環境での予測性能の安定化
- 開発・デバッグ環境での柔軟性の維持
