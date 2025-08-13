### 問題の概要
`src/day_trade/ml/advanced_ml_models.py` の `extract_advanced_features` メソッド内の各特徴量抽出メソッド（`_extract_technical_features` など）はCPUバウンドな処理です。これらは `async` 関数として定義されていますが、真の並列実行にはならず、イベントループをブロックする可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/advanced_ml_models.py`
    - `extract_advanced_features`
    - `_extract_technical_features`
    - `_extract_price_patterns`
    - `_extract_volatility_features`
    - `_extract_momentum_features`
    - `_extract_volume_features`
    - `_extract_multiframe_features`

### 具体的な改善提案
`extract_advanced_features` メソッド内の各特徴量抽出メソッド（`_extract_technical_features` など）はCPUバウンドな処理です。これらを `asyncio.to_thread` を使用してスレッドプールで実行することで、イベントループをブロックせずに他の非同期タスクと並行して実行できるようになります。

**変更例（`extract_advanced_features` メソッド内）:**
```python
import asyncio

# ...

            # 1. テクニカル指標特徴量
            technical_features = await asyncio.to_thread(self._extract_technical_features, data)

            # 2. 価格パターン特徴量
            price_patterns = await asyncio.to_thread(self._extract_price_patterns, data)
# ...
```
同様に、他の特徴量抽出メソッドにも適用します。

### 期待される効果
- 特徴量抽出の並列実行による処理時間の短縮
- イベントループのブロック回避によるアプリケーションの応答性向上
