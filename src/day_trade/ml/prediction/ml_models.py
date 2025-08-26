#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メインMLPredictionModelsクラス

機械学習予測システムの中核となるクラスです。
モデル訓練、管理、予測統合を行います。
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .base_types import (
    TrainingConfig,
    ModelTrainingResult,
    DEFAULT_MODEL_CONFIGS,
    EnsemblePrediction
)
from .metadata_manager import ModelMetadataManager
from .data_preparation import DataPreparationPipeline
from .ensemble_predictor import EnhancedEnsemblePredictor
from src.day_trade.ml.core_types import (
    ModelTrainingError,
    ModelMetadataError,
    DataPreparationError,
    ModelType,
    PredictionTask,
    ModelMetadata,
    ModelPerformance,
    BaseModelTrainer,
    DataQuality
)

# 外部依存の確認
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from src.day_trade.ml.model_trainers import (
        RandomForestTrainer, XGBoostTrainer, LightGBMTrainer, 
        XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE
    )
    MODEL_TRAINERS_AVAILABLE = True
except ImportError:
    MODEL_TRAINERS_AVAILABLE = False
    XGBOOST_AVAILABLE = False
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
            'model_configs': DEFAULT_MODEL_CONFIGS,
            'training_config': {
                'performance_threshold': 0.6,
                'min_data_quality': 'fair',
                'enable_cross_validation': True,
                'save_metadata': True
            }
        }

        if config_path and Path(config_path).exists():
            try:
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
        
        if not MODEL_TRAINERS_AVAILABLE:
            self.logger.warning("モデル訓練器が利用できません。基本実装を使用します")
            return self._create_basic_trainers()
        
        model_configs = self.config.get('model_configs', {})

        # Random Forest
        if MODEL_TRAINERS_AVAILABLE:
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

    def _create_basic_trainers(self) -> Dict[ModelType, BaseModelTrainer]:
        """基本訓練器作成（フォールバック）"""
        class BasicRandomForestTrainer(BaseModelTrainer):
            def create_model(self, is_classifier: bool, hyperparams: Dict[str, Any]):
                if is_classifier:
                    return RandomForestClassifier(**hyperparams)
                else:
                    return RandomForestRegressor(**hyperparams)

        trainers = {}
        trainers[ModelType.RANDOM_FOREST] = BasicRandomForestTrainer(
            ModelType.RANDOM_FOREST, {}, self.logger
        )
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

        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=config.test_size, 
            random_state=config.random_state,
            stratify=y if task == PredictionTask.PRICE_DIRECTION else None
        )

        # モデル作成
        is_classifier = task == PredictionTask.PRICE_DIRECTION
        hyperparams = optimized_params.get(model_type.value, {}) if optimized_params else {}
        
        # デフォルトパラメータを使用
        default_params = self.config['model_configs'].get(model_type, {})
        if is_classifier:
            hyperparams = {**default_params.get('classifier_params', {}), **hyperparams}
        else:
            hyperparams = {**default_params.get('regressor_params', {}), **hyperparams}

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
        metrics = self._calculate_metrics(y_test_encoded, y_pred, is_classifier)

        # 特徴量重要度
        feature_importance = self._get_feature_importance(model, list(X_clean.columns))

        # モデル保存（メタデータ付き）
        if config.save_model:
            await self._save_model_with_enhanced_metadata(
                model, model_type, task, symbol, X_clean.columns, targets[task],
                hyperparams, metrics, [], feature_importance, is_classifier,
                data_quality, start_time, label_encoder
            )

        training_time = (datetime.now() - start_time).total_seconds()

        # ModelPerformance作成
        return ModelPerformance(
            model_id=f"{symbol}_{model_type.value}_{task.value}",
            symbol=symbol,
            task=task,
            model_type=model_type,
            accuracy=metrics.get('accuracy', 0.0),
            precision=metrics.get('precision', 0.0),
            recall=metrics.get('recall', 0.0),
            f1_score=metrics.get('f1_score', 0.0),
            cross_val_mean=0.0,
            cross_val_std=0.0,
            cross_val_scores=[],
            r2_score=metrics.get('r2_score'),
            mse=metrics.get('mse'),
            rmse=metrics.get('rmse'),
            mae=metrics.get('mae'),
            feature_importance=feature_importance,
            training_time=training_time
        )

    def _calculate_metrics(self, y_true, y_pred, is_classifier: bool) -> Dict[str, float]:
        """メトリクス計算"""
        metrics = {}
        
        try:
            if is_classifier:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            else:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                metrics['r2_score'] = r2_score(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
        except Exception as e:
            self.logger.error(f"メトリクス計算エラー: {e}")
            
        return metrics

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度取得"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(zip(feature_names, importances.tolist()))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if len(importances.shape) > 1:
                    importances = np.mean(importances, axis=0)
                return dict(zip(feature_names, importances.tolist()))
            else:
                return {}
        except Exception as e:
            self.logger.error(f"特徴量重要度取得エラー: {e}")
            return {}

    async def _save_model_with_enhanced_metadata(self, model, model_type: ModelType, task: PredictionTask,
                                               symbol: str, feature_columns: pd.Index, target_series: pd.Series,
                                               hyperparams: Dict[str, Any], metrics: Dict[str, float],
                                               cv_scores: List[float], feature_importance: Dict[str, float],
                                               is_classifier: bool, data_quality: DataQuality,
                                               start_time: datetime, label_encoder=None):
        """強化されたメタデータ付きモデル保存"""

        try:
            # モデルID生成
            model_id = f"{symbol}_{model_type.value}_{task.value}"
            model_key = model_id

            # バージョン生成
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # フレームワークバージョン情報
            framework_versions = {'sklearn': '1.0.0'}  # 実際のバージョンを取得すべき

            # メタデータ作成
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                task=task,
                symbol=symbol,
                version=version,
                created_at=start_time,
                updated_at=datetime.now(),
                feature_columns=list(feature_columns),
                target_info={'name': task.value, 'type': 'classification' if is_classifier else 'regression'},
                training_samples=len(target_series.dropna()),
                training_period="1y",
                data_quality=data_quality,
                hyperparameters=hyperparams,
                preprocessing_config={},
                feature_selection_config={},
                performance_metrics=metrics,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                is_classifier=is_classifier,
                model_size_mb=0.0,  # 後で更新
                training_time_seconds=(datetime.now() - start_time).total_seconds(),
                python_version=sys.version,
                sklearn_version='1.0.0',
                framework_versions=framework_versions
            )

            # モデル保存
            file_path = self.models_dir / f"{model_key}.joblib"
            model_data = {
                'model': model,
                'metadata': metadata
            }
            if label_encoder:
                model_data['label_encoder'] = label_encoder

            joblib.dump(model_data, file_path)

            # ファイルサイズ更新
            metadata.model_size_mb = file_path.stat().st_size / (1024 * 1024)

            # メタデータ保存
            self.metadata_manager.save_metadata(metadata)

            # メモリに保存
            self.trained_models[model_key] = model
            self.model_metadata[model_key] = metadata
            if label_encoder:
                self.label_encoders[model_key] = label_encoder

            self.logger.info(f"モデル保存完了: {model_key} (サイズ: {metadata.model_size_mb:.2f}MB)")

        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            raise ModelMetadataError(f"モデル保存失敗: {e}") from e

    def _calculate_ensemble_weights(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """アンサンブル重み計算（改良版）"""

        if symbol not in self.ensemble_weights:
            self.ensemble_weights[symbol] = {}

        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            task_performances = []
            model_types = []

            for model_type, task_perfs in performances.items():
                if task in task_perfs:
                    perf = task_perfs[task]
                    # 適切なメトリクスを選択
                    if task == PredictionTask.PRICE_DIRECTION:
                        score = perf.f1_score  # 分類ではF1スコア
                    else:
                        score = perf.r2_score or 0.0  # 回帰ではR2スコア

                    task_performances.append(score)
                    model_types.append(model_type)

            if task_performances:
                # 性能に基づくソフトマックス重み計算
                performances_array = np.array(task_performances)
                # 負の値を避けるために最小値を0にシフト
                performances_array = performances_array - np.min(performances_array) + 0.1
                exp_performances = np.exp(performances_array * 5)  # スケーリング
                weights = exp_performances / exp_performances.sum()

                self.ensemble_weights[symbol][task] = dict(zip(model_types, weights))

                self.logger.info(f"アンサンブル重み計算完了 {symbol}-{task.value}: {dict(zip([mt.value for mt in model_types], weights))}")

    async def _save_training_results(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], symbol: str):
        """訓練結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for model_type, task_perfs in performances.items():
                    for task, perf in task_perfs.items():
                        # model_performance_historyテーブルに保存
                        conn.execute("""
                            INSERT INTO model_performance_history
                            (model_id, evaluation_date, dataset_type, accuracy, precision_score,
                             recall_score, f1_score, r2_score, mse, rmse, mae, cross_val_mean, cross_val_std)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            perf.model_id,
                            datetime.now().isoformat(),
                            'training',
                            perf.accuracy,
                            perf.precision,
                            perf.recall,
                            perf.f1_score,
                            perf.r2_score,
                            perf.mse,
                            perf.rmse,
                            perf.mae,
                            perf.cross_val_mean,
                            perf.cross_val_std
                        ))

            self.logger.info(f"訓練結果保存完了: {symbol}")

        except Exception as e:
            self.logger.error(f"訓練結果保存エラー: {e}")

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得（強化版）"""
        try:
            summary = {
                'total_models': len(self.trained_models),
                'model_types': [mt.value for mt in self.trainers.keys()],
                'symbols_covered': [],
                'tasks_covered': [],
                'recent_training_activity': [],
                'performance_summary': {}
            }

            # 銘柄とタスクの分析
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