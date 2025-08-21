"""
ML Prediction Models - 主要予測システムクラス

ml_prediction_models_improved.py からのリファクタリング抽出
MLPredictionModels メインクラスと EnhancedEnsemblePredictor
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import pickle
import joblib
import sqlite3
import sys
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import Counter

# 設定とモジュールのインポート
from .ml_config import ModelType, PredictionTask, DataQuality, TrainingConfig
from .ml_exceptions import ModelTrainingError, DataPreparationError
from .ml_model_base import BaseModelTrainer, RandomForestTrainer, XGBoostTrainer, LightGBMTrainer
from .ml_utilities import (
    ModelMetadataManager, DataPreparationPipeline, ModelMetadata,
    ModelPerformance, PredictionResult, EnsemblePrediction
)

# 機械学習ライブラリ
try:
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class MLPredictionModels:
    """機械学習予測モデルシステム（Issue #850対応強化版）"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required")

        # ディレクトリ初期化
        self.data_dir = Path("ml_models_data_improved")
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        # 設定読み込み
        self.config = self._load_config(config_path)

        # データベース初期化
        self.db_path = self.data_dir / "ml_predictions_improved.db"

        # メタデータ管理システム
        self.metadata_manager = ModelMetadataManager(self.db_path)

        # データ準備パイプライン
        self.data_pipeline = DataPreparationPipeline(TrainingConfig())

        # モデル訓練器
        self.trainers = self._init_trainers()

        # 訓練済みモデル
        self.trained_models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}

        # アンサンブル重み
        self.ensemble_weights: Dict[str, Dict[PredictionTask, Dict[ModelType, float]]] = {}

        # 強化されたアンサンブル予測器
        self.ensemble_predictor = EnhancedEnsemblePredictor(self)

        # 訓練済みモデルのロード
        self._load_existing_models()

        self.logger.info("ML prediction models (improved) initialized successfully")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """設定ファイル読み込み"""
        default_config = {
            'model_configs': {
                ModelType.RANDOM_FOREST: {
                    'classifier_params': {
                        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                        'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
                        'n_jobs': -1, 'class_weight': 'balanced'
                    },
                    'regressor_params': {
                        'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10,
                        'min_samples_leaf': 5, 'max_features': 'sqrt', 'random_state': 42,
                        'n_jobs': -1
                    }
                },
                ModelType.XGBOOST: {
                    'classifier_params': {
                        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                        'n_jobs': -1, 'eval_metric': 'mlogloss'
                    },
                    'regressor_params': {
                        'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42,
                        'n_jobs': -1, 'eval_metric': 'rmse'
                    }
                },
                ModelType.LIGHTGBM: {
                    'classifier_params': {
                        'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
                        'random_state': 42, 'n_jobs': -1, 'verbose': -1
                    },
                    'regressor_params': {
                        'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1,
                        'random_state': 42, 'n_jobs': -1, 'verbose': -1
                    }
                }
            },
            'training_config': {
                'performance_threshold': 0.6,
                'min_data_quality': 'fair',
                'enable_cross_validation': True,
                'save_metadata': True
            }
        }

        if config_path and Path(config_path).exists():
            try:
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.yaml'):
                        import yaml
                        loaded_config = yaml.safe_load(f)
                    else:
                        loaded_config = json.load(f)

                # 設定をマージ
                for key, value in loaded_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value

                self.logger.info(f"設定ファイル読み込み完了: {config_path}")
            except Exception as e:
                self.logger.warning(f"設定ファイル読み込み失敗: {e}, デフォルト設定を使用")

        return default_config

    def _init_trainers(self) -> Dict[ModelType, BaseModelTrainer]:
        """モデル訓練器初期化"""
        trainers = {}
        model_configs = self.config.get('model_configs', {})

        # Random Forest
        trainers[ModelType.RANDOM_FOREST] = RandomForestTrainer(
            ModelType.RANDOM_FOREST,
            model_configs.get(ModelType.RANDOM_FOREST, {}),
            self.logger
        )

        # XGBoost
        if XGBOOST_AVAILABLE:
            trainers[ModelType.XGBOOST] = XGBoostTrainer(
                ModelType.XGBOOST,
                model_configs.get(ModelType.XGBOOST, {}),
                self.logger
            )

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            trainers[ModelType.LIGHTGBM] = LightGBMTrainer(
                ModelType.LIGHTGBM,
                model_configs.get(ModelType.LIGHTGBM, {}),
                self.logger
            )

        self.logger.info(f"訓練器初期化完了: {list(trainers.keys())}")
        return trainers

    def _load_existing_models(self):
        """既存モデルの読み込み"""
        try:
            loaded_count = 0
            for model_file in self.models_dir.glob("*.joblib"):
                try:
                    model_data = joblib.load(model_file)
                    model_key = model_file.stem

                    self.trained_models[model_key] = model_data['model']

                    if 'label_encoder' in model_data:
                        self.label_encoders[model_key] = model_data['label_encoder']

                    if 'metadata' in model_data:
                        self.model_metadata[model_key] = model_data['metadata']

                    loaded_count += 1

                except Exception as e:
                    self.logger.error(f"モデル読み込み失敗 {model_file.name}: {e}")

            self.logger.info(f"既存モデル読み込み完了: {loaded_count}件")

        except Exception as e:
            self.logger.error(f"モデル読み込み処理エラー: {e}")

    async def train_models(self, symbol: str, period: str = "1y",
                          config: Optional[TrainingConfig] = None,
                          optimized_params: Optional[Dict[str, Any]] = None) -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練（統合改善版）"""

        config = config or TrainingConfig()
        self.logger.info(f"モデル訓練開始: {symbol}")

        try:
            # データ準備
            features, targets, data_quality = await self.data_pipeline.prepare_training_data(symbol, period)

            # データ品質チェック
            if data_quality < config.min_data_quality:
                raise DataPreparationError(f"データ品質不足: {data_quality.value}")

            # 有効なインデックス取得
            valid_idx = features.index[:-1]  # 最後の行は未来の値が不明
            X = features.loc[valid_idx]

            performances = {}

            # 各モデルタイプで訓練
            for model_type, trainer in self.trainers.items():
                self.logger.info(f"{model_type.value} 訓練開始")
                performances[model_type] = {}

                # 各予測タスクで訓練
                for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                    if task not in targets:
                        continue

                    try:
                        perf = await self._train_single_model(
                            model_type, task, trainer, X, targets, symbol,
                            valid_idx, config, data_quality, optimized_params
                        )
                        performances[model_type][task] = perf

                    except Exception as e:
                        self.logger.error(f"{model_type.value}-{task.value} 訓練失敗: {e}")

            # アンサンブル重み計算
            self._calculate_ensemble_weights(performances, symbol)

            # 性能結果保存
            await self._save_training_results(performances, symbol)

            self.logger.info(f"モデル訓練完了: {symbol}")
            return performances

        except Exception as e:
            self.logger.error(f"モデル訓練エラー: {e}")
            raise ModelTrainingError(f"モデル訓練失敗: {e}") from e

    async def _train_single_model(self, model_type: ModelType, task: PredictionTask,
                                trainer: BaseModelTrainer, X: pd.DataFrame,
                                targets: Dict[PredictionTask, pd.Series], symbol: str,
                                valid_idx: pd.Index, config: TrainingConfig,
                                data_quality: DataQuality,
                                optimized_params: Optional[Dict[str, Any]] = None) -> ModelPerformance:
        """単一モデル訓練（共通化された処理）"""

        start_time = datetime.now()

        # ターゲット準備
        y = targets[task].loc[valid_idx].dropna()
        X_clean = X.loc[y.index]

        # データ品質再チェック
        is_valid, quality, message = trainer.validate_data_quality(X_clean, y, task)
        if not is_valid:
            raise ModelTrainingError(f"データ品質チェック失敗: {message}")

        # データ分割
        X_train, X_test, y_train, y_test = trainer.prepare_data(X_clean, y, config)

        # モデル作成
        is_classifier = task == PredictionTask.PRICE_DIRECTION
        hyperparams = optimized_params.get(model_type.value, {}) if optimized_params else {}
        model = trainer.create_model(is_classifier, hyperparams)

        # ラベルエンコーダー（分類の場合）
        label_encoder = None
        if is_classifier:
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test

        # モデル訓練
        model.fit(X_train, y_train_encoded)

        # 予測と評価
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if is_classifier and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        metrics = trainer.calculate_metrics(y_test_encoded, y_pred, y_pred_proba, is_classifier)

        # クロスバリデーション
        cv_scores = []
        if config.enable_cross_validation:
            cv_scores = trainer.cross_validate(model, X_clean, y_train_encoded if is_classifier else y, config)
            metrics.update({
                'cross_val_mean': np.mean(cv_scores),
                'cross_val_std': np.std(cv_scores)
            })

        # 特徴量重要度
        feature_importance = trainer.get_feature_importance(model, X.columns.tolist())

        # モデル保存
        model_key = f"{symbol}_{model_type.value}_{task.value}"
        if config.save_model:
            await self._save_model(model_key, model, label_encoder, feature_importance, metrics, config)

        # メタデータ作成
        metadata = self._create_model_metadata(
            model_key, model_type, task, symbol, X.columns.tolist(),
            len(X_train), data_quality, hyperparams, metrics, cv_scores,
            feature_importance, is_classifier, start_time
        )

        if config.save_metadata:
            self.metadata_manager.save_metadata(metadata)
            self.model_metadata[model_key] = metadata

        # 性能オブジェクト作成
        training_time = (datetime.now() - start_time).total_seconds()

        performance = ModelPerformance(
            model_id=model_key,
            symbol=symbol,
            task=task,
            model_type=model_type,
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', 0.0),
            cross_val_mean=metrics.get('cross_val_mean', 0.0),
            cross_val_std=metrics.get('cross_val_std', 0.0),
            cross_val_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
            r2_score=metrics.get('r2_score', None),
            mse=metrics.get('mse', None),
            rmse=metrics.get('rmse', None),
            mae=metrics.get('mae', None),
            feature_importance=feature_importance,
            training_time=training_time
        )

        self.logger.info(f"{model_key} 訓練完了: {metrics}")
        return performance

    async def _save_model(self, model_key: str, model: Any, label_encoder: Optional[LabelEncoder],
                         feature_importance: Dict[str, float], metrics: Dict[str, float],
                         config: TrainingConfig):
        """モデル保存"""
        try:
            model_data = {
                'model': model,
                'feature_importance': feature_importance,
                'metrics': metrics,
                'saved_at': datetime.now().isoformat()
            }

            if label_encoder:
                model_data['label_encoder'] = label_encoder

            model_file = self.models_dir / f"{model_key}.joblib"
            joblib.dump(model_data, model_file)

            # メモリに追加
            self.trained_models[model_key] = model
            if label_encoder:
                self.label_encoders[model_key] = label_encoder

            self.logger.info(f"モデル保存完了: {model_file}")

        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")

    def _create_model_metadata(self, model_key: str, model_type: ModelType, task: PredictionTask,
                              symbol: str, feature_columns: List[str], training_samples: int,
                              data_quality: DataQuality, hyperparameters: Dict[str, Any],
                              metrics: Dict[str, float], cv_scores: np.ndarray,
                              feature_importance: Dict[str, float], is_classifier: bool,
                              start_time: datetime) -> ModelMetadata:
        """モデルメタデータ作成"""
        try:
            model_size = sys.getsizeof(pickle.dumps(self.trained_models.get(model_key, {}))) / (1024 * 1024)
        except:
            model_size = 0.0

        return ModelMetadata(
            model_id=model_key,
            model_type=model_type,
            task=task,
            symbol=symbol,
            version="1.0",
            created_at=start_time,
            updated_at=datetime.now(),
            feature_columns=feature_columns,
            target_info={"task": task.value, "is_classifier": is_classifier},
            training_samples=training_samples,
            training_period="unknown",
            data_quality=data_quality,
            hyperparameters=hyperparameters,
            preprocessing_config={},
            feature_selection_config={},
            performance_metrics=metrics,
            cross_validation_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
            feature_importance=feature_importance,
            is_classifier=is_classifier,
            model_size_mb=model_size,
            training_time_seconds=(datetime.now() - start_time).total_seconds(),
            python_version=sys.version,
            sklearn_version="unknown",
            framework_versions={}
        )

    def _calculate_ensemble_weights(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """アンサンブル重み計算"""
        try:
            if symbol not in self.ensemble_weights:
                self.ensemble_weights[symbol] = {}

            for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                task_performances = {}

                for model_type, task_perfs in performances.items():
                    if task in task_perfs:
                        perf = task_perfs[task]
                        if task == PredictionTask.PRICE_DIRECTION:
                            score = perf.f1_score * 0.6 + perf.accuracy * 0.4
                        else:
                            score = max(0, perf.r2_score or 0) if perf.r2_score else 0
                        task_performances[model_type] = score

                # 重み正規化
                total_score = sum(task_performances.values())
                if total_score > 0:
                    weights = {model_type: score / total_score
                             for model_type, score in task_performances.items()}
                else:
                    weights = {model_type: 1.0 / len(task_performances)
                             for model_type in task_performances.keys()}

                self.ensemble_weights[symbol][task] = weights

            self.logger.info(f"アンサンブル重み計算完了: {symbol}")

        except Exception as e:
            self.logger.error(f"アンサンブル重み計算エラー: {e}")

    async def _save_training_results(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """訓練結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for model_type, task_perfs in performances.items():
                    for task, performance in task_perfs.items():
                        conn.execute("""
                            INSERT INTO model_performance_history
                            (model_id, evaluation_date, dataset_type, accuracy, precision_score,
                             recall_score, f1_score, r2_score, mse, rmse, mae, cross_val_mean, cross_val_std)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            performance.model_id, datetime.now().isoformat(), "training",
                            performance.accuracy, performance.precision, performance.recall,
                            performance.f1_score, performance.r2_score, performance.mse,
                            performance.rmse, performance.mae, performance.cross_val_mean,
                            performance.cross_val_std
                        ))

            self.logger.info(f"訓練結果保存完了: {symbol}")

        except Exception as e:
            self.logger.error(f"訓練結果保存エラー: {e}")

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得"""
        try:
            summary = {
                'total_models': len(self.trained_models),
                'available_trainers': list(self.trainers.keys()),
                'data_directory': str(self.data_dir),
                'database_path': str(self.db_path),
                'model_types_available': [],
                'recent_training_activity': [],
                'symbols_covered': [],
                'tasks_covered': []
            }

            # 利用可能なモデルタイプ
            for trainer_type in self.trainers.keys():
                summary['model_types_available'].append(trainer_type.value)

            # 対象シンボルとタスク
            symbols = set()
            tasks = set()
            for model_key in self.trained_models.keys():
                parts = model_key.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    task = '_'.join(parts[1:-1])
                    symbols.add(symbol)
                    tasks.add(task)

            summary['symbols_covered'] = list(symbols)
            summary['tasks_covered'] = list(tasks)

            # データベースから最新情報取得
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT model_id, accuracy, f1_score, r2_score, created_at
                        FROM model_performance_history
                        ORDER BY created_at DESC LIMIT 10
                    """)

                    for row in cursor.fetchall():
                        summary['recent_training_activity'].append({
                            'model_id': row[0],
                            'accuracy': row[1],
                            'f1_score': row[2],
                            'r2_score': row[3],
                            'date': row[4]
                        })
            except Exception as db_error:
                self.logger.warning(f"データベース取得エラー: {db_error}")
                summary['recent_training_activity'] = []

            return summary

        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}

    # 統一された予測インターフェース
    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（Issue #850-4対応）"""
        return await self.ensemble_predictor.predict(symbol, features)

    async def predict_list(self, symbol: str, features: pd.DataFrame) -> List[EnsemblePrediction]:
        """リスト形式予測（後方互換性）"""
        predictions_dict = await self.predict(symbol, features)
        return list(predictions_dict.values())


class EnhancedEnsemblePredictor:
    """強化されたアンサンブル予測システム"""

    def __init__(self, ml_models):
        self.ml_models = ml_models
        self.logger = logging.getLogger(__name__)

    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（強化版）"""

        predictions = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            try:
                ensemble_pred = await self._make_enhanced_ensemble_prediction(symbol, features, task)
                if ensemble_pred:
                    predictions[task] = ensemble_pred
            except Exception as e:
                self.logger.error(f"予測失敗 {task.value}: {e}")

        return predictions

    async def _make_enhanced_ensemble_prediction(self, symbol: str, features: pd.DataFrame,
                                               task: PredictionTask) -> Optional[EnsemblePrediction]:
        """強化されたアンサンブル予測"""

        model_predictions = {}
        model_confidences = {}
        model_quality_scores = {}
        excluded_models = []

        # 各モデルで予測
        for model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]:
            model_key = f"{symbol}_{model_type.value}_{task.value}"

            if model_key not in self.ml_models.trained_models:
                excluded_models.append(f"{model_type.value} (not trained)")
                continue

            try:
                model = self.ml_models.trained_models[model_key]
                metadata = self.ml_models.model_metadata.get(model_key)

                # モデル品質スコア
                quality_score = await self._get_model_quality_score(model_key, metadata)
                model_quality_scores[model_type.value] = quality_score

                # 品質が低すぎる場合は除外
                if quality_score < 0.3:
                    excluded_models.append(f"{model_type.value} (low quality: {quality_score:.2f})")
                    continue

                if task == PredictionTask.PRICE_DIRECTION:
                    # 分類予測
                    pred_proba = model.predict_proba(features)
                    pred_class = model.predict(features)[0]

                    # ラベルエンコーダーで逆変換
                    if model_key in self.ml_models.label_encoders:
                        le = self.ml_models.label_encoders[model_key]
                        pred_class = le.inverse_transform([pred_class])[0]

                    confidence = self._calculate_enhanced_classification_confidence(
                        pred_proba[0], quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_class
                    model_confidences[model_type.value] = confidence

                else:  # 回帰予測
                    pred_value = model.predict(features)[0]
                    confidence = self._calculate_enhanced_regression_confidence(
                        model, features, pred_value, quality_score, metadata
                    )

                    model_predictions[model_type.value] = pred_value
                    model_confidences[model_type.value] = confidence

            except Exception as e:
                self.logger.error(f"予測失敗 {model_key}: {e}")
                excluded_models.append(f"{model_type.value} (error: {str(e)[:50]})")

        if not model_predictions:
            return None

        # 動的重み計算
        dynamic_weights = await self._calculate_enhanced_dynamic_weights(
            symbol, task, model_quality_scores, model_confidences
        )

        # アンサンブル統合
        if task == PredictionTask.PRICE_DIRECTION:
            ensemble_result = self._enhanced_ensemble_classification(
                model_predictions, model_confidences, dynamic_weights
            )
        else:
            ensemble_result = self._enhanced_ensemble_regression(
                model_predictions, model_confidences, dynamic_weights
            )

        # 品質メトリクス計算
        diversity_score = self._calculate_diversity_score(model_predictions)

        return EnsemblePrediction(
            symbol=symbol,
            timestamp=datetime.now(),
            final_prediction=ensemble_result['prediction'],
            confidence=ensemble_result['confidence'],
            prediction_interval=ensemble_result.get('prediction_interval'),
            model_predictions=model_predictions,
            model_confidences=model_confidences,
            model_weights={k: v for k, v in dynamic_weights.items()},
            model_quality_scores=model_quality_scores,
            consensus_strength=ensemble_result['consensus_strength'],
            disagreement_score=ensemble_result['disagreement_score'],
            prediction_stability=ensemble_result.get('prediction_stability', 0.0),
            diversity_score=diversity_score,
            total_models_used=len(model_predictions),
            excluded_models=excluded_models,
            ensemble_method="enhanced_weighted_ensemble"
        )

    async def _get_model_quality_score(self, model_key: str, metadata: Optional[ModelMetadata]) -> float:
        """モデル品質スコア取得（強化版）"""
        try:
            if metadata:
                # メタデータから品質スコア計算
                performance_metrics = metadata.performance_metrics

                # 複数メトリクスの重み付き平均
                if metadata.is_classifier:
                    score = (0.4 * performance_metrics.get('accuracy', 0.5) +
                            0.3 * performance_metrics.get('f1_score', 0.5) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5))
                else:
                    score = (0.5 * max(0, performance_metrics.get('r2_score', 0.0)) +
                            0.3 * performance_metrics.get('cross_val_mean', 0.5) +
                            0.2 * (1.0 - min(1.0, performance_metrics.get('rmse', 1.0))))

                # データ品質による調整
                quality_multiplier = {
                    DataQuality.EXCELLENT: 1.0,
                    DataQuality.GOOD: 0.9,
                    DataQuality.FAIR: 0.8,
                    DataQuality.POOR: 0.6
                }.get(metadata.data_quality, 0.5)

                return np.clip(score * quality_multiplier, 0.0, 1.0)
            else:
                return 0.6  # デフォルト値

        except Exception as e:
            self.logger.error(f"品質スコア計算エラー: {e}")
            return 0.5

    def _calculate_enhanced_classification_confidence(self, pred_proba: np.ndarray,
                                                    quality_score: float,
                                                    metadata: Optional[ModelMetadata]) -> float:
        """強化された分類信頼度計算"""
        # 確率の最大値
        max_prob = np.max(pred_proba)

        # エントロピーベースの不確実性
        entropy = -np.sum(pred_proba * np.log(pred_proba + 1e-15))
        normalized_entropy = entropy / np.log(len(pred_proba))
        certainty = 1.0 - normalized_entropy

        # メタデータベースの調整
        metadata_adjustment = 1.0
        if metadata:
            cv_std = metadata.performance_metrics.get('cross_val_std', 0.1)
            metadata_adjustment = max(0.5, 1.0 - cv_std)  # CV標準偏差が低いほど信頼度高

        # 最終信頼度計算
        confidence = (0.4 * max_prob + 0.3 * certainty + 0.2 * quality_score + 0.1 * metadata_adjustment)

        return np.clip(confidence, 0.1, 0.95)

    def _calculate_enhanced_regression_confidence(self, model, features: pd.DataFrame,
                                                prediction: float, quality_score: float,
                                                metadata: Optional[ModelMetadata]) -> float:
        """強化された回帰信頼度計算"""
        try:
            # アンサンブルモデルの場合、個別予測のばらつきから信頼度推定
            if hasattr(model, 'estimators_'):
                individual_predictions = []
                for estimator in model.estimators_[:min(10, len(model.estimators_))]:
                    try:
                        pred = estimator.predict(features)[0]
                        individual_predictions.append(pred)
                    except:
                        continue

                if individual_predictions:
                    pred_std = np.std(individual_predictions)
                    pred_mean = np.mean(individual_predictions)
                    if pred_mean != 0:
                        cv = abs(pred_std / pred_mean)
                        prediction_consistency = max(0.1, 1.0 - cv)
                    else:
                        prediction_consistency = 0.5
                else:
                    prediction_consistency = 0.5
            else:
                prediction_consistency = 0.7  # デフォルト値

            # メタデータベースの調整
            metadata_adjustment = 1.0
            if metadata:
                r2_score = metadata.performance_metrics.get('r2_score', 0.5)
                metadata_adjustment = max(0.3, r2_score)

            # 最終信頼度計算
            confidence = (0.4 * quality_score + 0.3 * prediction_consistency + 0.3 * metadata_adjustment)

            return np.clip(confidence, 0.1, 0.95)

        except Exception as e:
            self.logger.error(f"回帰信頼度計算エラー: {e}")
            return quality_score * 0.8

    async def _calculate_enhanced_dynamic_weights(self, symbol: str, task: PredictionTask,
                                                quality_scores: Dict[str, float],
                                                confidences: Dict[str, float]) -> Dict[str, float]:
        """強化された動的重み計算"""
        try:
            # 基本重み（過去の性能ベース）
            base_weights = self.ml_models.ensemble_weights.get(symbol, {}).get(task, {})

            dynamic_weights = {}
            total_score = 0

            for model_name in quality_scores.keys():
                try:
                    model_type = ModelType(model_name)
                except ValueError:
                    continue

                # 各要素の重み
                quality = quality_scores[model_name]
                confidence = confidences[model_name]
                base_weight = base_weights.get(model_type, 1.0)

                # 時間減衰ファクター（新しいモデルほど重視）
                time_decay = 1.0  # 実装可能: モデルの新しさに基づく重み

                # 動的重み計算
                dynamic_weight = (0.3 * quality + 0.3 * confidence + 0.2 * base_weight + 0.2 * time_decay)
                dynamic_weights[model_name] = dynamic_weight
                total_score += dynamic_weight

            # 正規化
            if total_score > 0:
                for model_name in dynamic_weights:
                    dynamic_weights[model_name] /= total_score

            return dynamic_weights

        except Exception as e:
            self.logger.error(f"動的重み計算エラー: {e}")
            # フォールバック: 均等重み
            num_models = len(quality_scores)
            return {name: 1.0/num_models for name in quality_scores.keys()}

    def _enhanced_ensemble_classification(self, predictions: Dict[str, Any],
                                        confidences: Dict[str, float],
                                        weights: Dict[str, float]) -> Dict[str, Any]:
        """強化された分類アンサンブル"""
        weighted_votes = {}
        total_weight = 0
        prediction_values = list(predictions.values())

        # 重み付き投票
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            vote_strength = weight * confidence
            if prediction not in weighted_votes:
                weighted_votes[prediction] = 0
            weighted_votes[prediction] += vote_strength
            total_weight += vote_strength

        # 最終予測
        final_prediction = max(weighted_votes.items(), key=lambda x: x[1])[0]

        # アンサンブル信頼度
        if total_weight > 0:
            ensemble_confidence = weighted_votes[final_prediction] / total_weight
        else:
            ensemble_confidence = 0.5

        # コンセンサス強度
        unique_predictions = len(set(prediction_values))
        consensus_strength = 1.0 - (unique_predictions - 1) / max(1, len(predictions) - 1)

        # 予測安定性
        prediction_stability = self._calculate_prediction_stability_classification(predictions)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'consensus_strength': consensus_strength,
            'disagreement_score': 1.0 - consensus_strength,
            'prediction_stability': prediction_stability
        }

    def _enhanced_ensemble_regression(self, predictions: Dict[str, float],
                                    confidences: Dict[str, float],
                                    weights: Dict[str, float]) -> Dict[str, Any]:
        """強化された回帰アンサンブル"""
        weighted_sum = 0
        total_weight = 0
        pred_values = list(predictions.values())

        # 重み付き平均
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            confidence = confidences[model_name]

            adjusted_weight = weight * confidence
            weighted_sum += prediction * adjusted_weight
            total_weight += adjusted_weight

        final_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(pred_values)

        # 予測区間推定
        prediction_std = np.std(pred_values)
        prediction_interval = (final_prediction - 1.96 * prediction_std,
                             final_prediction + 1.96 * prediction_std)

        # アンサンブル信頼度
        avg_confidence = np.mean(list(confidences.values()))
        prediction_variance = np.var(pred_values)
        prediction_mean = np.mean(pred_values)

        if prediction_mean != 0:
            cv = prediction_variance / abs(prediction_mean)
            consistency_factor = max(0.1, 1.0 - np.tanh(cv))
        else:
            consistency_factor = 0.5

        ensemble_confidence = avg_confidence * consistency_factor

        # コンセンサス強度
        if prediction_mean != 0:
            consensus_strength = max(0.0, 1.0 - (prediction_std / abs(prediction_mean)))
        else:
            consensus_strength = max(0.0, 1.0 - prediction_std)

        return {
            'prediction': final_prediction,
            'confidence': np.clip(ensemble_confidence, 0.1, 0.95),
            'prediction_interval': prediction_interval,
            'consensus_strength': np.clip(consensus_strength, 0.0, 1.0),
            'disagreement_score': 1.0 - np.clip(consensus_strength, 0.0, 1.0),
            'prediction_stability': self._calculate_prediction_stability_regression(predictions)
        }

    def _calculate_diversity_score(self, predictions: Dict[str, Any]) -> float:
        """予測多様性スコア計算"""
        try:
            pred_values = list(predictions.values())
            if len(set(str(p) for p in pred_values)) == 1:
                return 0.0  # 完全一致

            if all(isinstance(p, (int, float)) for p in pred_values):
                # 数値予測の多様性
                normalized_std = np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8)
                return min(1.0, normalized_std)
            else:
                # カテゴリ予測の多様性
                unique_count = len(set(str(p) for p in pred_values))
                return (unique_count - 1) / max(1, len(pred_values) - 1)

        except Exception:
            return 0.5

    def _calculate_prediction_stability_classification(self, predictions: Dict[str, Any]) -> float:
        """分類予測安定性計算"""
        pred_values = list(predictions.values())
        if len(set(str(p) for p in pred_values)) == 1:
            return 1.0  # 完全一致

        # 最も多い予測の割合
        counts = Counter(str(p) for p in pred_values)
        most_common_count = counts.most_common(1)[0][1]
        stability = most_common_count / len(pred_values)

        return stability

    def _calculate_prediction_stability_regression(self, predictions: Dict[str, float]) -> float:
        """回帰予測安定性計算"""
        pred_values = list(predictions.values())
        if len(pred_values) <= 1:
            return 1.0

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)

        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            stability = max(0.0, 1.0 - cv)
        else:
            stability = max(0.0, 1.0 - std_pred)

        return np.clip(stability, 0.0, 1.0)


# ファクトリー関数とユーティリティ
def create_improved_ml_prediction_models(config_path: Optional[str] = None) -> MLPredictionModels:
    """改善版MLPredictionModelsの作成"""
    return MLPredictionModels(config_path)


# グローバルインスタンス（後方互換性）
try:
    ml_prediction_models_improved = MLPredictionModels()
except Exception as e:
    logging.getLogger(__name__).error(f"改善版MLモデル初期化失敗: {e}")
    ml_prediction_models_improved = None