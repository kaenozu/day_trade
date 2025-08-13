### 問題の概要
`src/day_trade/ml/ensemble_system.py` の `fit` メソッド内のベースモデルの学習ループと、`predict` メソッド内の個別モデルからの予測収集ループは、順次実行されています。各ベースモデルの学習と予測は独立しているため、並列化の大きな機会があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/ensemble_system.py`
    - `fit`
    - `predict`

### 具体的な改善提案
`fit` メソッド内のベースモデルの学習ループと、`predict` メソッド内の個別モデルからの予測収集ループにおいて、`joblib.Parallel` や `multiprocessing` モジュールを使用して、各ベースモデルの処理を並列で実行することを強く推奨します。これにより、全体の学習時間と予測時間を大幅に短縮できます。

**変更例（`fit` メソッド内の学習ループ）:**
```python
from joblib import Parallel, delayed

# ...

            # 2. 従来MLモデル学習
            # 並列処理で各ベースモデルを学習
            results_list = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._train_single_model)(model_name, model, X, y, validation_data)
                for model_name, model in self.base_models.items()
            )
            for model_name, result in results_list:
                model_results[model_name] = result

# 新しいヘルパーメソッドを追加
def _train_single_model(self, model_name: str, model: BaseModelInterface, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[str, Dict[str, Any]]:
    try:
        logger.info(f"{model_name}学習開始")
        result = model.fit(X, y, validation_data=validation_data)
        logger.info(f"{model_name}学習完了")
        return model_name, result
    except Exception as e:
        logger.error(f"{model_name}学習エラー: {e}")
        return model_name, {"status": "失敗", "error": str(e)}
```
`predict` メソッド内の予測収集ループも同様に並列化できます。

### 期待される効果
- アンサンブルモデルの学習時間の短縮
- アンサンブル予測時間の短縮
- CPUリソースの有効活用
