#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル訓練システム - ML Prediction Models Model Trainer

ML予測モデルの訓練とメタデータ保存機能を提供します。
"""

import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.day_trade.ml.core_types import (
    BaseModelTrainer,
    DataQuality,
    ModelMetadata,
    ModelMetadataError,
    ModelPerformance,
    ModelTrainingError,
    ModelType,
    PredictionTask,
)
from src.day_trade.ml.model_trainers import LIGHTGBM_AVAILABLE, XGBOOST_AVAILABLE
from .data_types import TrainingConfig


class ModelTrainerSystem:
    """モデル訓練システム"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = model_manager.logger

    async def train_models(
        self, 
        symbol: str, 
        period: str = "1y",
        config: Optional[TrainingConfig] = None,
        optimized_params: Optional[Dict[str, Any]] = None
    ) -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練（統合改善版）"""

        config = config or TrainingConfig()
        self.logger.info(f"モデル訓練開始: {symbol}")

        try:
            # データ準備
            features, targets, data_quality = await self.model_manager.data_pipeline.prepare_training_data(symbol, period)

            # データ品質チェック
            if data_quality < config.min_data_quality:
                raise ModelTrainingError(f"データ品質不足: {data_quality.value}")

            # 有効なインデックス取得
            valid_idx = features.index[:-1]  # 最後の行は未来の値が不明
            X = features.loc[valid_idx]

            performances = {}

            # 各モデルタイプで訓練
            for model_type, trainer in self.model_manager.trainers.items():
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

    async def _train_single_model(
        self, 
        model_type: ModelType, 
        task: PredictionTask,
        trainer: BaseModelTrainer, 
        X: pd.DataFrame,
        targets: Dict[PredictionTask, pd.Series], 
        symbol: str,
        valid_idx: pd.Index, 
        config: TrainingConfig,
        data_quality: DataQuality,
        optimized_params: Optional[Dict[str, Any]] = None
    ) -> ModelPerformance:
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

    async def _save_model_with_enhanced_metadata(
        self, 
        model, 
        model_type: ModelType, 
        task: PredictionTask,
        symbol: str, 
        feature_columns: pd.Index, 
        target_series: pd.Series,
        hyperparams: Dict[str, Any], 
        metrics: Dict[str, float],
        cv_scores: np.ndarray, 
        feature_importance: Dict[str, float],
        is_classifier: bool, 
        data_quality: DataQuality,
        start_time: datetime, 
        label_encoder=None
    ) -> None:
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
                import xgboost as xgb
                framework_versions['xgboost'] = xgb.__version__
            elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
                import lightgbm as lgb
                framework_versions['lightgbm'] = lgb.__version__

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
            self.model_manager.save_model(model_key, model, metadata, label_encoder)

            # ファイルサイズ更新
            file_path = self.model_manager.models_dir / f"{model_key}.joblib"
            if file_path.exists():
                metadata.model_size_mb = file_path.stat().st_size / (1024 * 1024)

            # メタデータ保存
            self.model_manager.metadata_manager.save_metadata(metadata)

            self.logger.info(f"モデル保存完了: {model_key} (サイズ: {metadata.model_size_mb:.2f}MB)")

        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            raise ModelMetadataError(f"モデル保存失敗: {e}") from e

    def _calculate_ensemble_weights(
        self, 
        performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], 
        symbol: str
    ) -> None:
        """アンサンブル重み計算（改良版）"""

        if symbol not in self.model_manager.ensemble_weights:
            self.model_manager.ensemble_weights[symbol] = {}

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

                self.model_manager.ensemble_weights[symbol][task] = dict(zip(model_types, weights))

                self.logger.info(f"アンサンブル重み計算完了 {symbol}-{task.value}: {dict(zip(model_types, weights))}")

    async def _save_training_results(
        self, 
        performances: Dict[ModelType, Dict[PredictionTask, ModelPerformance]], 
        symbol: str
    ) -> None:
        """訓練結果保存"""
        try:
            with sqlite3.connect(self.model_manager.db_path) as conn:
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

    def retrain_model(
        self, 
        symbol: str, 
        model_type: ModelType, 
        task: PredictionTask,
        config: Optional[TrainingConfig] = None
    ) -> Optional[ModelPerformance]:
        """単一モデル再訓練"""
        try:
            self.logger.info(f"モデル再訓練開始: {symbol}_{model_type.value}_{task.value}")

            # 訓練実行（非同期を同期で実行）
            import asyncio
            loop = asyncio.get_event_loop()
            performances = loop.run_until_complete(
                self.train_models(symbol, config=config)
            )

            # 該当モデルの結果を返す
            if model_type in performances and task in performances[model_type]:
                return performances[model_type][task]
            else:
                return None

        except Exception as e:
            self.logger.error(f"モデル再訓練エラー: {e}")
            return None

    def validate_training_data(
        self, 
        features: pd.DataFrame, 
        targets: Dict[PredictionTask, pd.Series]
    ) -> Dict[str, Any]:
        """訓練データ検証"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_summary': {}
        }

        try:
            # 基本チェック
            if features.empty:
                validation_result['errors'].append("特徴量データが空です")

            if not targets:
                validation_result['errors'].append("ターゲットデータがありません")

            # データ形状チェック
            if len(features) < 30:
                validation_result['warnings'].append(f"データ数が少ないです: {len(features)}行")

            # 欠損値チェック
            missing_features = features.isnull().sum().sum()
            if missing_features > 0:
                validation_result['warnings'].append(f"特徴量に欠損値があります: {missing_features}個")

            # ターゲット検証
            for task, target in targets.items():
                if target.isnull().sum() > 0:
                    validation_result['warnings'].append(f"ターゲット{task.value}に欠損値があります")

            # データサマリー
            validation_result['data_summary'] = {
                'feature_count': len(features.columns),
                'sample_count': len(features),
                'target_tasks': list(targets.keys()),
                'missing_values': missing_features
            }

            # 全体の有効性判定
            validation_result['is_valid'] = len(validation_result['errors']) == 0

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"訓練データ検証エラー: {e}")

        return validation_result