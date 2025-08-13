### 問題の概要
`src/day_trade/ml/base_models/svr_model.py` の `SVRModel` の `__init__` メソッド内の `default_config` で、`SVR` の `cache_size` がデフォルト値の200MBに設定されています。大規模なデータセットの場合、このキャッシュサイズでは不足し、カーネル行列の計算がディスクI/Oに依存してパフォーマンスが低下する可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/base_models/svr_model.py`
    - `__init__`

### 具体的な改善提案
`SVRModel` の `__init__` メソッド内の `default_config` で、`cache_size` の値をデータセットのサイズや利用可能なメモリに応じて調整することを検討してください。大規模なデータセットの場合、デフォルトの200MBでは不足する可能性があります。

### 期待される効果
- 大規模データセットでのSVR学習時間の短縮
- ディスクI/Oの削減
