### 問題の概要
`src/day_trade/ml/base_models/base_model_interface.py` の `save_model` および `load_model` メソッドでは、モデルの保存と読み込みに `pickle` を使用しています。大規模なモデルの場合、`pickle` によるシリアライズ・デシリアライズはI/O操作とCPU処理の両方で時間がかかる可能性があります。

### 関連ファイルとメソッド
- `src/day_trade/ml/base_models/base_model_interface.py`
    - `save_model`
    - `load_model`

### 具体的な改善提案
`save_model` および `load_model` メソッドにおいて、`self.model` の具体的なタイプに応じて、より効率的な保存・読み込み方法を検討してください。
- `sklearn` モデルや `XGBoost` モデルの場合は `joblib` を使用する。
- ディープラーニングモデルの場合は、そのフレームワーク（TensorFlow/Keras, PyTorch）が提供する専用の保存・読み込み機能を使用する。
これにより、大規模なモデルのI/Oパフォーマンスが向上する可能性があります。

**変更例（`save_model` メソッド内）:**
```python
import joblib # joblibをインポート

# ...

def save_model(self, filepath: str) -> bool:
    """
    モデル保存
    """
    try:
        # self.model が具体的なモデルインスタンスの場合、joblibを使用
        if self.model and hasattr(self.model, 'save_model'): # XGBoostなど
            self.model.save_model(filepath)
            logger.info(f"{self.model_name}モデル保存完了 (専用メソッド): {filepath}")
            return True
        elif self.model and hasattr(self.model, 'get_params'): # scikit-learnなど
            joblib.dump(self.model, filepath)
            logger.info(f"{self.model_name}モデル保存完了 (joblib): {filepath}")
            return True
        else:
            # それ以外はpickleを使用
            import pickle
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"{self.model_name}モデル保存完了 (pickle): {filepath}")
            return True

    except Exception as e:
        logger.error(f"{self.model_name}モデル保存エラー: {e}")
        return False
```
`load_model` も同様に、`joblib.load()` や各フレームワークの `load_model()` に対応させる必要があります。

### 期待される効果
- 大規模なモデルの保存・読み込み時間の短縮
- I/Oパフォーマンスの向上
