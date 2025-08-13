### 問題の概要
`src/day_trade/ml/feature_pipeline.py` の `batch_generate_features` メソッド内の各バッチ処理と、`precompute_features_for_symbols` メソッド内の各シンボルの特徴量生成は、順次実行されています。これらの処理は独立しているため、並列化の大きな機会があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/feature_pipeline.py`
    - `batch_generate_features`
    - `precompute_features_for_symbols`

### 具体的な改善提案
`batch_generate_features` メソッド内の各バッチ処理と、`precompute_features_for_symbols` メソッド内の各シンボルの特徴量生成を並列化することを強く推奨します。`joblib.Parallel` や `multiprocessing` を使用して、これらの処理を並行して実行することで、全体の実行時間を大幅に短縮できます。

**変更例（`batch_generate_features` メソッド内のバッチ処理ループ）:**
```python
from joblib import Parallel, delayed

# ...

        for batch_idx, symbol_batch in enumerate(symbol_batches):
            # ...
            # バッチ内処理を並列化
            batch_results_list = Parallel(n_jobs=self.config.max_parallel_symbols)(
                delayed(self._process_single_symbol_in_batch)(symbol, symbols_data[symbol], feature_config, force_regenerate)
                for symbol in symbol_batch
            )
            for symbol, result in batch_results_list:
                if result:
                    batch_results[symbol] = result
            # ...

def _process_single_symbol_in_batch(self, symbol: str, data: pd.DataFrame, feature_config: FeatureConfig, force_regenerate: bool) -> Tuple[str, Optional[FeatureResult]]:
    try:
        if not force_regenerate:
            cached_result = self.feature_store.load_feature(symbol, data.index.min().strftime("%Y-%m-%d"), data.index.max().strftime("%Y-%m-%d"), feature_config)
            if cached_result:
                return symbol, cached_result

        result = self.feature_store.get_or_generate_feature(
            symbol=symbol,
            data=data,
            start_date=data.index.min().strftime("%Y-%m-%d"),
            end_date=data.index.max().strftime("%Y-%m-%d"),
            feature_config=feature_config,
            optimization_config=self.config.optimization_config,
        )
        return symbol, result
    except Exception as e:
        logger.error(f"バッチ内シンボル処理エラー: {symbol} - {e}")
        return symbol, None
```
`precompute_features_for_symbols` も同様に並列化できます。

### 期待される効果
- 特徴量生成時間の短縮
- CPUリソースの有効活用
