#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル訓練モジュール

機械学習モデルの訓練、評価、保存を担当
"""

import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.day_trade.ml.core_types import (
    ModelType,
    PredictionTask,
    DataQuality,
    BaseModelTrainer,
    ModelTrainingError,
    ModelMetadataError
)
from src.day_trade.ml.model_trainers import (
    RandomForestTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE
)
from .data_structures import TrainingConfig, ModelPerformance, ModelMetadata
from .metadata_manager import ModelMetadataManager


class ModelTrainingManager:
    """モデル訓練管理システム"""

    def __init__(self, models_dir: Path, metadata_manager: ModelMetadataManager, 
                 config: Dict[str, Any]):
        self.models_dir = models_dir
        self.metadata_manager = metadata_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # モデル訓練器初期化
        self.trainers = self._init_trainers()
        
        # 訓練済みモデルとメタデータ
        self.trained_models: Dict[str, Any] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}

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

    async def train_single_model(self, model_type: ModelType, task: PredictionTask,
                               X: pd.DataFrame, targets: Dict[PredictionTask, pd.Series],
                               symbol: str, valid_idx: pd.Index, config: TrainingConfig,
                               data_quality: DataQuality,
                               optimized_params: Optional[Dict[str, Any]] = None) -> ModelPerformance:
        """単一モデル訓練（共通化された処理）"""

        start_time = datetime.now()

        if model_type not in self.trainers:
            raise ModelTrainingError(f"訓練器が見つかりません: {model_type}")

        trainer = self.trainers[model_type]

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
        feature_importance = trainer.get_feature_importance(model, list(X_clean.columns))

        # モデル保存（メタデータ付き）
        if config.save_model:
            await self._save_model_with_enhanced_metadata(
                model, model_type, task, symbol, X_clean.columns, targets[task],
                hyperparams, metrics, cv_scores, feature_importance, is_classifier,
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
            cross_val_mean=metrics.get('cross_val_mean', 0.0),
            cross_val_std=metrics.get('cross_val_std', 0.0),
            cross_val_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
            r2_score=metrics.get('r2_score'),
            mse=metrics.get('mse'),
            rmse=metrics.get('rmse'),
            mae=metrics.get('mae'),
            feature_importance=feature_importance,
            training_time=training_time
        )

    async def _save_model_with_enhanced_metadata(self, model, model_type: ModelType, task: PredictionTask,
                                               symbol: str, feature_columns: pd.Index, target_series: pd.Series,
                                               hyperparams: Dict[str, Any], metrics: Dict[str, float],
                                               cv_scores: np.ndarray, feature_importance: Dict[str, float],
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
            framework_versions = {'sklearn': '1.0.0'}  # 実際のバージョンを取得
            if model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
                try:
                    import xgboost as xgb
                    framework_versions['xgboost'] = xgb.__version__
                except ImportError:
                    pass
            elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                try:
                    import lightgbm as lgb
                    framework_versions['lightgbm'] = lgb.__version__
                except ImportError:
                    pass

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
                training_period="1y",  # 設定可能にする
                data_quality=data_quality,
                hyperparameters=hyperparams,
                preprocessing_config={},
                feature_selection_config={},
                performance_metrics=metrics,
                cross_validation_scores=cv_scores.tolist() if len(cv_scores) > 0 else [],
                feature_importance=feature_importance,
                is_classifier=is_classifier,
                model_size_mb=0.0,  # 後で更新
                training_time_seconds=(datetime.now() - start_time).total_seconds(),
                python_version=sys.version,
                sklearn_version='1.0.0',  # 実際のバージョンを取得
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

    def load_existing_models(self):
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

    def calculate_ensemble_weights(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], 
                                 symbol: str) -> Dict[PredictionTask, Dict[ModelType, float]]:
        """アンサンブル重み計算（改良版）"""
        
        ensemble_weights = {}

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

                ensemble_weights[task] = dict(zip(model_types, weights))

                self.logger.info(f"アンサンブル重み計算完了 {symbol}-{task.value}: {dict(zip(model_types, weights))}")

        return ensemble_weights

    async def save_training_results(self, performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], 
                                  symbol: str, db_path: Path):
        """訓練結果保存"""
        try:
            with sqlite3.connect(db_path) as conn:
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
                'model_types': list(self.trainers.keys()),
                'symbols_covered': [],
                'tasks_covered': [],
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

            return summary

        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}