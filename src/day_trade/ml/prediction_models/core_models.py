#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コアモデルクラス - MLPredictionModels

Issue #850-2 & #850-3 & #850-4: メインのMLPredictionModelsクラス（統合改善版）
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from src.day_trade.ml.core_types import (
    SKLEARN_AVAILABLE,
    ModelType,
    PredictionTask,
    DataPreparationError,
    ModelTrainingError
)

from .data_structures import TrainingConfig, ModelPerformance, EnsemblePrediction
from .data_preparation import DataPreparationPipeline
from .metadata_manager import ModelMetadataManager
from .model_training import ModelTrainingManager
from .ensemble_predictor import EnhancedEnsemblePredictor


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

        # サブシステム初期化
        self.metadata_manager = ModelMetadataManager(self.db_path)
        self.data_pipeline = DataPreparationPipeline(TrainingConfig())
        self.training_manager = ModelTrainingManager(
            self.models_dir, self.metadata_manager, self.config
        )
        self.ensemble_predictor = EnhancedEnsemblePredictor(self)

        # アンサンブル重み
        self.ensemble_weights: Dict[str, Dict[PredictionTask, Dict[ModelType, float]]] = {}

        # 既存モデルの読み込み
        self.training_manager.load_existing_models()

        # プロパティ委譲（後方互換性のため）
        self.trained_models = self.training_manager.trained_models
        self.label_encoders = self.training_manager.label_encoders
        self.model_metadata = self.training_manager.model_metadata

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
            for model_type in self.training_manager.trainers.keys():
                self.logger.info(f"{model_type.value} 訓練開始")
                performances[model_type] = {}

                # 各予測タスクで訓練
                for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
                    if task not in targets:
                        continue

                    try:
                        perf = await self.training_manager.train_single_model(
                            model_type, task, X, targets, symbol,
                            valid_idx, config, data_quality, optimized_params
                        )
                        performances[model_type][task] = perf

                    except Exception as e:
                        self.logger.error(f"{model_type.value}-{task.value} 訓練失敗: {e}")

            # アンサンブル重み計算
            self.ensemble_weights[symbol] = self.training_manager.calculate_ensemble_weights(performances, symbol)

            # 性能結果保存
            await self.training_manager.save_training_results(performances, symbol, self.db_path)

            self.logger.info(f"モデル訓練完了: {symbol}")
            return performances

        except Exception as e:
            self.logger.error(f"モデル訓練エラー: {e}")
            raise ModelTrainingError(f"モデル訓練失敗: {e}") from e

    async def predict(self, symbol: str, features: pd.DataFrame) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（Issue #850-4対応）"""
        return await self.ensemble_predictor.predict(symbol, features)

    async def predict_list(self, symbol: str, features: pd.DataFrame) -> List[EnsemblePrediction]:
        """リスト形式予測（後方互換性）"""
        predictions_dict = await self.predict(symbol, features)
        return list(predictions_dict.values())

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得（強化版）"""
        try:
            # 基本サマリー
            summary = self.training_manager.get_model_summary()
            
            # メタデータマネージャーからの詳細情報
            metadata_summary = self.metadata_manager.get_model_summary()
            summary.update(metadata_summary)
            
            # アンサンブル重み情報
            summary['ensemble_weights'] = {
                symbol: {
                    task.value: {model_type.value: weight 
                               for model_type, weight in weights.items()}
                    for task, weights in tasks.items()
                }
                for symbol, tasks in self.ensemble_weights.items()
            }
            
            return summary

        except Exception as e:
            self.logger.error(f"サマリー取得エラー: {e}")
            return {'error': str(e)}

    def get_prediction_explanation(self, prediction: EnsemblePrediction) -> str:
        """予測結果の説明取得"""
        return self.ensemble_predictor.get_prediction_explanation(prediction)

    # 後方互換性のためのプロパティアクセス
    @property
    def trainers(self):
        """訓練器への後方互換性アクセス"""
        return self.training_manager.trainers

    # データベース関連の後方互換性メソッド
    async def save_ensemble_prediction(self, symbol: str, task: PredictionTask, 
                                      ensemble_data: Dict) -> bool:
        """アンサンブル予測保存（後方互換性）"""
        from datetime import datetime
        return self.metadata_manager.save_ensemble_prediction(
            symbol, datetime.now(), task, ensemble_data
        )

    def get_model_versions(self, symbol: str, model_type: ModelType, 
                          task: PredictionTask) -> List[str]:
        """モデルバージョン取得（後方互換性）"""
        return self.metadata_manager.get_model_versions(symbol, model_type, task)

    def load_metadata(self, model_id: str):
        """メタデータ読み込み（後方互換性）"""
        return self.metadata_manager.load_metadata(model_id)

    # 設定アクセス用プロパティ
    @property
    def model_configs(self):
        """モデル設定への簡単アクセス"""
        return self.config.get('model_configs', {})

    @property
    def training_config(self):
        """訓練設定への簡単アクセス"""
        return self.config.get('training_config', {})