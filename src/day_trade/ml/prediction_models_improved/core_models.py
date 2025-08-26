#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
コアMLPredictionModelsクラス - ML Prediction Models Core

ML予測モデルのコアシステム統合クラスです。
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.day_trade.ml.core_types import (
    ModelMetadata,
    ModelPerformance,
    ModelType,
    PredictionTask,
)
from .data_types import EnsemblePrediction, TrainingConfig
from .model_manager import ModelManager
from .model_trainer import ModelTrainerSystem


class MLPredictionModels:
    """機械学習予測モデルシステム（Issue #850対応強化版）"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # モデル管理システム初期化
        self.model_manager = ModelManager(config_path)
        
        # モデル訓練システム初期化
        self.trainer_system = ModelTrainerSystem(self.model_manager)

        # 後方互換性のため、model_managerの属性を直接公開
        self.data_dir = self.model_manager.data_dir
        self.models_dir = self.model_manager.models_dir
        self.db_path = self.model_manager.db_path
        self.config = self.model_manager.config
        self.metadata_manager = self.model_manager.metadata_manager
        self.data_pipeline = self.model_manager.data_pipeline
        self.trainers = self.model_manager.trainers
        self.trained_models = self.model_manager.trained_models
        self.label_encoders = self.model_manager.label_encoders
        self.model_metadata = self.model_manager.model_metadata
        self.ensemble_weights = self.model_manager.ensemble_weights
        self.ensemble_predictor = self.model_manager.ensemble_predictor

        self.logger.info("ML prediction models (improved) initialized successfully")

    # 訓練関連メソッド
    async def train_models(
        self, 
        symbol: str, 
        period: str = "1y",
        config: Optional[TrainingConfig] = None,
        optimized_params: Optional[Dict[str, Any]] = None
    ) -> Dict[ModelType, Dict[PredictionTask, ModelPerformance]]:
        """モデル訓練（統合改善版）"""
        return await self.trainer_system.train_models(symbol, period, config, optimized_params)

    def retrain_model(
        self, 
        symbol: str, 
        model_type: ModelType, 
        task: PredictionTask,
        config: Optional[TrainingConfig] = None
    ) -> Optional[ModelPerformance]:
        """単一モデル再訓練"""
        return self.trainer_system.retrain_model(symbol, model_type, task, config)

    # モデル管理関連メソッド
    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得"""
        return self.model_manager.get_model_summary()

    def get_model_by_key(self, model_key: str) -> Optional[Any]:
        """モデルキーによるモデル取得"""
        return self.model_manager.get_model_by_key(model_key)

    def get_label_encoder_by_key(self, model_key: str) -> Optional[LabelEncoder]:
        """モデルキーによるラベルエンコーダー取得"""
        return self.model_manager.get_label_encoder_by_key(model_key)

    def get_metadata_by_key(self, model_key: str) -> Optional[ModelMetadata]:
        """モデルキーによるメタデータ取得"""
        return self.model_manager.get_metadata_by_key(model_key)

    def has_trained_model(self, symbol: str, model_type: ModelType, task: PredictionTask) -> bool:
        """訓練済みモデル存在チェック"""
        return self.model_manager.has_trained_model(symbol, model_type, task)

    def get_available_models(self, symbol: str) -> Dict[PredictionTask, List[ModelType]]:
        """利用可能モデル一覧取得"""
        return self.model_manager.get_available_models(symbol)

    def set_ensemble_predictor(self, ensemble_predictor) -> None:
        """アンサンブル予測器設定"""
        self.model_manager.set_ensemble_predictor(ensemble_predictor)
        self.ensemble_predictor = ensemble_predictor

    def delete_model(self, model_key: str) -> bool:
        """モデル削除"""
        return self.model_manager.delete_model(model_key)

    def cleanup_old_models(self, days_to_keep: int = 30) -> Dict[str, int]:
        """古いモデルのクリーンアップ"""
        return self.model_manager.cleanup_old_models(days_to_keep)

    # 予測関連メソッド（アンサンブル予測器に委譲）
    async def predict(
        self, 
        symbol: str, 
        features: pd.DataFrame
    ) -> Dict[PredictionTask, EnsemblePrediction]:
        """統一された予測インターフェース（Issue #850-4対応）"""
        if self.ensemble_predictor:
            return await self.ensemble_predictor.predict(symbol, features)
        else:
            raise RuntimeError("アンサンブル予測器が設定されていません")

    async def predict_list(
        self, 
        symbol: str, 
        features: pd.DataFrame
    ) -> List[EnsemblePrediction]:
        """リスト形式予測（後方互換性）"""
        predictions_dict = await self.predict(symbol, features)
        return list(predictions_dict.values())

    # 設定・検証関連メソッド
    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリー取得"""
        return self.model_manager.get_config_summary()

    def validate_configuration(self) -> Dict[str, Any]:
        """設定検証"""
        return self.model_manager.validate_configuration()

    def validate_training_data(
        self, 
        features: pd.DataFrame, 
        targets: Dict[PredictionTask, pd.Series]
    ) -> Dict[str, Any]:
        """訓練データ検証"""
        return self.trainer_system.validate_training_data(features, targets)

    # 統計・分析関連メソッド
    def get_performance_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """性能統計取得"""
        try:
            stats = {
                'total_models': len(self.trained_models),
                'model_type_distribution': {},
                'task_distribution': {},
                'average_performance': {}
            }

            # モデル種別分布
            for model_key in self.trained_models.keys():
                parts = model_key.split('_')
                if len(parts) >= 2:
                    model_type = parts[-2]
                    stats['model_type_distribution'][model_type] = \
                        stats['model_type_distribution'].get(model_type, 0) + 1

            # タスク分布
            for model_key in self.trained_models.keys():
                parts = model_key.split('_')
                if len(parts) >= 1:
                    task = parts[-1]
                    stats['task_distribution'][task] = \
                        stats['task_distribution'].get(task, 0) + 1

            # 特定シンボルの統計
            if symbol:
                symbol_models = [k for k in self.trained_models.keys() if k.startswith(symbol)]
                stats['symbol_specific'] = {
                    'model_count': len(symbol_models),
                    'available_tasks': list(self.get_available_models(symbol).keys())
                }

            return stats

        except Exception as e:
            self.logger.error(f"性能統計取得エラー: {e}")
            return {'error': str(e)}

    def export_model_info(self, format: str = 'json') -> Dict[str, Any]:
        """モデル情報エクスポート"""
        try:
            export_data = {
                'version': '2.0.0',
                'export_timestamp': pd.Timestamp.now().isoformat(),
                'summary': self.get_model_summary(),
                'config': self.get_config_summary(),
                'models': {}
            }

            # 各モデルの詳細情報
            for model_key, metadata in self.model_metadata.items():
                if hasattr(metadata, 'to_dict'):
                    export_data['models'][model_key] = metadata.to_dict()
                else:
                    export_data['models'][model_key] = {
                        'model_id': getattr(metadata, 'model_id', model_key),
                        'model_type': getattr(metadata, 'model_type', 'unknown'),
                        'created_at': str(getattr(metadata, 'created_at', 'unknown'))
                    }

            if format == 'json':
                return export_data
            else:
                # 他の形式も拡張可能
                return export_data

        except Exception as e:
            self.logger.error(f"モデル情報エクスポートエラー: {e}")
            return {'error': str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        """システムヘルス状態取得"""
        try:
            health = {
                'status': 'healthy',
                'issues': [],
                'warnings': []
            }

            # 設定検証
            config_validation = self.validate_configuration()
            if not config_validation['is_valid']:
                health['status'] = 'unhealthy'
                health['issues'].extend(config_validation['errors'])
            health['warnings'].extend(config_validation.get('warnings', []))

            # モデル状態チェック
            model_count = len(self.trained_models)
            if model_count == 0:
                health['warnings'].append("訓練済みモデルがありません")

            # データベース接続チェック
            try:
                summary = self.get_model_summary()
                if 'error' in summary:
                    health['issues'].append(f"データベース接続エラー: {summary['error']}")
                    health['status'] = 'unhealthy'
            except Exception as e:
                health['issues'].append(f"データベース接続エラー: {e}")
                health['status'] = 'unhealthy'

            # アンサンブル予測器チェック
            if not self.ensemble_predictor:
                health['warnings'].append("アンサンブル予測器が設定されていません")

            health['summary'] = {
                'total_models': model_count,
                'trainers_available': len(self.trainers),
                'ensemble_predictor_available': self.ensemble_predictor is not None
            }

            return health

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'summary': {}
            }