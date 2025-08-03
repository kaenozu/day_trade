"""
パフォーマンス最適化済み機械学習モデル

既存のml_models.pyにパフォーマンス最適化を適用した版。
並列処理、キャッシュ、メモリ最適化を実装。
"""

import gc
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..utils.logging_config import get_context_logger
from ..utils.performance_analyzer import profile_performance

warnings.filterwarnings('ignore')
logger = get_context_logger(__name__)


@dataclass
class OptimizedModelPrediction:
    """最適化済みモデル予測結果"""

    prediction: float
    confidence: float = 0.0
    model_name: str = ""
    prediction_time: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    cache_hit: bool = False


class PerformanceCache:
    """パフォーマンス向けキャッシュシステム"""

    def __init__(self, max_size: int = 1000, ttl_hours: int = 1):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.access_times = {}

    def _generate_key(self, data: pd.DataFrame) -> str:
        """データのハッシュキー生成"""
        # データの形状とサンプル値からハッシュ生成
        if len(data) == 0:
            return "empty_data"

        # 利用可能な列を使用
        available_cols = data.columns[:min(2, len(data.columns))]
        if len(available_cols) > 0:
            recent_data = data[available_cols].tail(min(5, len(data)))
            key_str = f"{recent_data.values.tobytes().hex()}"
            return key_str[:32]  # 32文字に制限
        else:
            return f"shape_{data.shape[0]}_{data.shape[1]}"

    def get(self, data: pd.DataFrame) -> Optional[List[OptimizedModelPrediction]]:
        """キャッシュから予測結果を取得"""
        key = self._generate_key(data)

        if key in self.cache:
            cached_item = self.cache[key]

            # TTLチェック
            if datetime.now() - cached_item['timestamp'] < self.ttl:
                self.access_times[key] = datetime.now()

                # キャッシュヒットをマーク
                predictions = cached_item['predictions']
                for pred in predictions:
                    pred.cache_hit = True

                logger.debug(
                    "キャッシュヒット",
                    section="ml_cache",
                    key=key[:8],
                    predictions_count=len(predictions)
                )

                return predictions
            else:
                # 期限切れのため削除
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        return None

    def put(self, data: pd.DataFrame, predictions: List[OptimizedModelPrediction]):
        """予測結果をキャッシュに保存"""
        key = self._generate_key(data)

        # キャッシュサイズ制限
        if len(self.cache) >= self.max_size:
            # 最も古いアクセスのキーを削除
            oldest_key = min(self.access_times.keys(),
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[key] = {
            'predictions': predictions,
            'timestamp': datetime.now()
        }
        self.access_times[key] = datetime.now()

        logger.debug(
            "キャッシュ保存",
            section="ml_cache",
            key=key[:8],
            cache_size=len(self.cache)
        )


class OptimizedBasePredictionModel:
    """最適化済みベース予測モデル"""

    def __init__(self, model_name: str, enable_parallel: bool = True):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.enable_parallel = enable_parallel
        self.feature_names = []
        self.training_history = []

        # パフォーマンス統計
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.avg_prediction_time = 0.0

    @profile_performance
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """モデル訓練（最適化済み）"""
        logger.info(
            f"モデル訓練開始: {self.model_name}",
            section="ml_training",
            data_shape=X.shape
        )

        try:
            # メモリ効率的な前処理
            X_processed = self._preprocess_features(X)

            # 並列処理対応のモデル設定
            model_params = self._get_optimized_model_params()
            self.model = self._create_model(**model_params)

            # 特徴量正規化
            X_scaled = self.scaler.fit_transform(X_processed)

            # 訓練実行
            self.model.fit(X_scaled, y)
            self.is_trained = True
            self.feature_names = list(X_processed.columns)

            # 訓練履歴記録
            self.training_history.append({
                'timestamp': datetime.now(),
                'data_size': len(X),
                'features_count': X_processed.shape[1]
            })

            logger.info(
                f"モデル訓練完了: {self.model_name}",
                section="ml_training",
                features_count=len(self.feature_names)
            )

        except Exception as e:
            logger.error(
                f"モデル訓練エラー: {self.model_name}",
                section="ml_training",
                error=str(e)
            )
            raise

    @profile_performance
    def predict(self, X: pd.DataFrame) -> OptimizedModelPrediction:
        """予測実行（最適化済み）"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained")

        start_time = datetime.now()

        try:
            # 特徴量前処理
            X_processed = self._preprocess_features(X)

            # 欠損している特徴量を補完
            X_aligned = self._align_features(X_processed)

            # 正規化
            X_scaled = self.scaler.transform(X_aligned)

            # 予測実行
            prediction = self.model.predict(X_scaled)[0]

            # 信頼度計算（簡易版）
            confidence = self._calculate_confidence(X_scaled, prediction)

            # パフォーマンス統計更新
            computation_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(computation_time)

            return OptimizedModelPrediction(
                prediction=prediction,
                confidence=confidence,
                model_name=self.model_name,
                computation_time=computation_time
            )

        except Exception as e:
            logger.error(
                f"予測エラー: {self.model_name}",
                section="ml_prediction",
                error=str(e)
            )
            raise

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """特徴量前処理（メモリ効率化）"""
        # 欠損値を効率的に処理
        X_filled = X.fillna(method='ffill').fillna(0)

        # 無限値の処理
        X_clean = X_filled.replace([np.inf, -np.inf], 0)

        return X_clean

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """特徴量の整合性確保"""
        if not self.feature_names:
            return X

        # 訓練時の特徴量に合わせる
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)

        # 不足している特徴量を0で補完
        for feature in missing_features:
            X[feature] = 0.0

        # 余分な特徴量を削除
        X_aligned = X[self.feature_names]

        if missing_features or extra_features:
            logger.debug(
                f"特徴量調整: {self.model_name}",
                section="feature_alignment",
                missing=len(missing_features),
                extra=len(extra_features)
            )

        return X_aligned

    def _calculate_confidence(self, X_scaled: np.ndarray, prediction: float) -> float:
        """信頼度計算（簡易版）"""
        # 簡易信頼度計算（実装を軽量化）
        base_confidence = 70.0

        # 特徴量の分散による調整
        feature_variance = np.var(X_scaled)
        confidence_adjustment = max(-20, min(20, (0.5 - feature_variance) * 40))

        return max(0, min(100, base_confidence + confidence_adjustment))

    def _update_performance_stats(self, computation_time: float):
        """パフォーマンス統計更新"""
        self.prediction_count += 1
        self.total_prediction_time += computation_time
        self.avg_prediction_time = self.total_prediction_time / self.prediction_count

    def _get_optimized_model_params(self) -> Dict[str, Any]:
        """最適化済みモデルパラメータ"""
        base_params = {}

        if self.enable_parallel:
            base_params['n_jobs'] = -1  # 全CPUコア使用

        return base_params

    def _create_model(self, **params):
        """モデル作成（サブクラスで実装）"""
        raise NotImplementedError


class OptimizedLinearRegressionModel(OptimizedBasePredictionModel):
    """最適化済み線形回帰モデル"""

    def __init__(self):
        super().__init__("optimized_linear_regression")

    def _create_model(self, **params):
        return LinearRegression(**params)


class OptimizedRandomForestModel(OptimizedBasePredictionModel):
    """最適化済みランダムフォレストモデル"""

    def __init__(self):
        super().__init__("optimized_random_forest")

    def _get_optimized_model_params(self) -> Dict[str, Any]:
        params = super()._get_optimized_model_params()
        params.update({
            'n_estimators': 50,      # 軽量化のため削減
            'max_depth': 10,         # 深さ制限
            'min_samples_split': 10, # 分割条件を緩和
            'min_samples_leaf': 5,   # リーフ条件を緩和
            'random_state': 42
        })
        return params

    def _create_model(self, **params):
        return RandomForestRegressor(**params)


class OptimizedGradientBoostingModel(OptimizedBasePredictionModel):
    """最適化済み勾配ブースティングモデル"""

    def __init__(self):
        super().__init__("optimized_gradient_boosting")

    def _get_optimized_model_params(self) -> Dict[str, Any]:
        params = super()._get_optimized_model_params()
        params.update({
            'n_estimators': 30,      # 軽量化
            'max_depth': 6,          # 深さ制限
            'learning_rate': 0.1,
            'subsample': 0.8,        # サブサンプリング
            'random_state': 42
        })
        return params

    def _create_model(self, **params):
        return GradientBoostingRegressor(**params)


class OptimizedEnsemblePredictor:
    """最適化済みアンサンブル予測器"""

    def __init__(self, enable_parallel: bool = True, enable_cache: bool = True):
        self.models: List[OptimizedBasePredictionModel] = []
        self.model_weights: Dict[str, float] = {}
        self.enable_parallel = enable_parallel
        self.enable_cache = enable_cache

        # パフォーマンス最適化
        self.cache = PerformanceCache() if enable_cache else None
        self.executor = ThreadPoolExecutor(max_workers=4) if enable_parallel else None

        logger.info(
            "最適化済みアンサンブル予測器初期化",
            section="ensemble_init",
            parallel_enabled=enable_parallel,
            cache_enabled=enable_cache
        )

    def add_model(self, model: OptimizedBasePredictionModel, weight: float = 1.0):
        """モデル追加"""
        self.models.append(model)
        self.model_weights[model.model_name] = weight

        logger.debug(
            f"モデル追加: {model.model_name}",
            section="ensemble_setup",
            weight=weight,
            total_models=len(self.models)
        )

    @profile_performance
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """アンサンブル訓練"""
        logger.info(
            "アンサンブル訓練開始",
            section="ensemble_training",
            models_count=len(self.models),
            data_shape=X.shape
        )

        if self.enable_parallel:
            # 並列訓練
            self._fit_parallel(X, y)
        else:
            # 逐次訓練
            self._fit_sequential(X, y)

        # 重み正規化
        self._normalize_weights()

        logger.info(
            "アンサンブル訓練完了",
            section="ensemble_training",
            trained_models=len([m for m in self.models if m.is_trained])
        )

    def _fit_parallel(self, X: pd.DataFrame, y: pd.Series):
        """並列訓練実行"""
        def train_model(model):
            try:
                model.fit(X, y)
                return model.model_name, True
            except Exception as e:
                logger.error(f"並列訓練エラー: {model.model_name}", error=str(e))
                return model.model_name, False

        # 並列実行
        futures = [self.executor.submit(train_model, model) for model in self.models]

        for future in futures:
            model_name, success = future.result()
            if not success:
                logger.warning(f"モデル訓練失敗: {model_name}")

    def _fit_sequential(self, X: pd.DataFrame, y: pd.Series):
        """逐次訓練実行"""
        for model in self.models:
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"逐次訓練エラー: {model.model_name}", error=str(e))

    @profile_performance
    def predict(self, X: pd.DataFrame) -> List[OptimizedModelPrediction]:
        """アンサンブル予測（最適化済み）"""

        # キャッシュチェック
        if self.cache:
            cached_predictions = self.cache.get(X)
            if cached_predictions:
                return cached_predictions

        # 予測実行
        predictions = []

        if self.enable_parallel and len(self.models) > 1:
            predictions = self._predict_parallel(X)
        else:
            predictions = self._predict_sequential(X)

        # キャッシュ保存
        if self.cache and predictions:
            self.cache.put(X, predictions)

        return predictions

    def _predict_parallel(self, X: pd.DataFrame) -> List[OptimizedModelPrediction]:
        """並列予測実行"""
        def predict_model(model):
            try:
                if model.is_trained:
                    return model.predict(X)
            except Exception as e:
                logger.error(f"並列予測エラー: {model.model_name}", error=str(e))
            return None

        # 並列実行
        futures = [self.executor.submit(predict_model, model) for model in self.models]

        predictions = []
        for future in futures:
            prediction = future.result()
            if prediction:
                predictions.append(prediction)

        return predictions

    def _predict_sequential(self, X: pd.DataFrame) -> List[OptimizedModelPrediction]:
        """逐次予測実行"""
        predictions = []

        for model in self.models:
            try:
                if model.is_trained:
                    prediction = model.predict(X)
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"逐次予測エラー: {model.model_name}", error=str(e))

        return predictions

    def _normalize_weights(self):
        """重み正規化"""
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                name: weight / total_weight
                for name, weight in self.model_weights.items()
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        stats = {
            'total_models': len(self.models),
            'trained_models': len([m for m in self.models if m.is_trained]),
            'cache_enabled': self.enable_cache,
            'parallel_enabled': self.enable_parallel,
            'model_stats': {}
        }

        for model in self.models:
            stats['model_stats'][model.model_name] = {
                'prediction_count': model.prediction_count,
                'avg_prediction_time': model.avg_prediction_time,
                'is_trained': model.is_trained
            }

        if self.cache:
            stats['cache_stats'] = {
                'cache_size': len(self.cache.cache),
                'max_size': self.cache.max_size,
                'ttl_hours': self.cache.ttl.total_seconds() / 3600
            }

        return stats

    def __del__(self):
        """リソースクリーンアップ"""
        if self.executor:
            self.executor.shutdown(wait=False)


def create_optimized_model_ensemble() -> OptimizedEnsemblePredictor:
    """最適化済みモデルアンサンブル作成"""
    ensemble = OptimizedEnsemblePredictor(enable_parallel=True, enable_cache=True)

    # 軽量化されたモデル群
    ensemble.add_model(OptimizedLinearRegressionModel(), weight=1.0)
    ensemble.add_model(OptimizedRandomForestModel(), weight=1.5)
    ensemble.add_model(OptimizedGradientBoostingModel(), weight=1.2)

    logger.info(
        "最適化済みアンサンブル作成完了",
        section="ensemble_creation",
        models_count=len(ensemble.models)
    )

    return ensemble


@profile_performance
def optimize_prediction_accuracy(
    predictions: List[OptimizedModelPrediction],
    actual: np.ndarray
) -> Dict[str, float]:
    """予測精度評価（最適化済み）"""
    if not predictions or len(actual) == 0:
        return {}

    pred_values = np.array([p.prediction for p in predictions])

    # 複数予測がある場合は重み付き平均
    if len(predictions) > 1:
        weights = np.array([p.confidence / 100.0 for p in predictions])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        final_prediction = np.average(pred_values, weights=weights)
    else:
        final_prediction = pred_values[0]

    # 精度指標計算（ベクトル化）
    try:
        mae = float(np.abs(final_prediction - actual[0]))
        mse = float((final_prediction - actual[0]) ** 2)
        rmse = float(np.sqrt(mse))

        # 相対誤差
        if actual[0] != 0:
            mape = float(np.abs((final_prediction - actual[0]) / actual[0]) * 100)
        else:
            mape = 0.0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'prediction_count': len(predictions),
            'avg_confidence': float(np.mean([p.confidence for p in predictions])),
            'cache_hit_rate': sum(1 for p in predictions if p.cache_hit) / len(predictions) * 100
        }

    except Exception as e:
        logger.error("精度評価エラー", section="accuracy_evaluation", error=str(e))
        return {}


# メモリ最適化ユーティリティ
def optimize_memory_usage():
    """メモリ使用量最適化"""
    collected = gc.collect()
    logger.debug(
        "ガベージコレクション実行",
        section="memory_optimization",
        collected_objects=collected
    )
    return collected


# 使用例とデモ
if __name__ == "__main__":
    # メモリ最適化デモ
    logger.info("最適化済みMLモデルデモ開始", section="demo")

    try:
        # テストデータ生成（軽量）
        np.random.seed(42)
        n_samples = 1000  # サンプル数を削減

        features = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'feature_4': np.random.randn(n_samples)
        })

        target = (features['feature_1'] * 0.5 +
                 features['feature_2'] * 0.3 +
                 np.random.randn(n_samples) * 0.2)

        # アンサンブル作成・訓練
        ensemble = create_optimized_model_ensemble()

        # 訓練データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # 訓練実行
        ensemble.fit(X_train, y_train)

        # 予測実行（キャッシュテスト含む）
        predictions1 = ensemble.predict(X_test.tail(1))
        predictions2 = ensemble.predict(X_test.tail(1))  # キャッシュヒットテスト

        # 精度評価
        if predictions1:
            accuracy_metrics = optimize_prediction_accuracy(predictions1, y_test.tail(1).values)

            # パフォーマンス統計
            perf_stats = ensemble.get_performance_stats()

            logger.info(
                "最適化済みMLモデルデモ完了",
                section="demo",
                accuracy_metrics=accuracy_metrics,
                performance_stats=perf_stats
            )

            # メモリクリーンアップ
            optimize_memory_usage()

    except Exception as e:
        logger.error(f"デモ実行エラー: {e}", section="demo")

    finally:
        # リソース解放
        if 'ensemble' in locals():
            del ensemble
        optimize_memory_usage()
