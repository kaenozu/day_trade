#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル管理システム - ML Prediction Models Model Manager

ML予測モデルの設定、初期化、管理機能を提供します。
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from sklearn.preprocessing import LabelEncoder

from src.day_trade.ml.core_types import (
    BaseModelTrainer,
    ModelMetadata,
    ModelType,
    PredictionTask,
)
from src.day_trade.ml.model_trainers import (
    LIGHTGBM_AVAILABLE,
    SKLEARN_AVAILABLE,
    XGBOOST_AVAILABLE,
    LightGBMTrainer,
    RandomForestTrainer,
    XGBoostTrainer,
)
from .data_preparation import DataPreparationPipeline
from .data_types import TrainingConfig
from .metadata_manager import ModelMetadataManager


class ModelManager:
    """モデル管理システム"""

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

        # 強化されたアンサンブル予測器（後で初期化）
        self.ensemble_predictor = None

        # 訓練済みモデルのロード
        self._load_existing_models()

        self.logger.info("Model manager initialized successfully")

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

    def _load_existing_models(self) -> None:
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

    def get_model_summary(self) -> Dict[str, Any]:
        """モデルサマリー取得（強化版）"""
        try:
            summary = {
                'total_models': len(self.trained_models),
                'model_types': list(self.trainers.keys()),
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

    def get_model_by_key(self, model_key: str) -> Optional[Any]:
        """モデルキーによるモデル取得"""
        return self.trained_models.get(model_key)

    def get_label_encoder_by_key(self, model_key: str) -> Optional[LabelEncoder]:
        """モデルキーによるラベルエンコーダー取得"""
        return self.label_encoders.get(model_key)

    def get_metadata_by_key(self, model_key: str) -> Optional[ModelMetadata]:
        """モデルキーによるメタデータ取得"""
        return self.model_metadata.get(model_key)

    def has_trained_model(self, symbol: str, model_type: ModelType, task: PredictionTask) -> bool:
        """訓練済みモデル存在チェック"""
        model_key = f"{symbol}_{model_type.value}_{task.value}"
        return model_key in self.trained_models

    def get_available_models(self, symbol: str) -> Dict[PredictionTask, List[ModelType]]:
        """利用可能モデル一覧取得"""
        available = {}
        
        for task in [PredictionTask.PRICE_DIRECTION, PredictionTask.PRICE_REGRESSION]:
            available[task] = []
            for model_type in self.trainers.keys():
                if self.has_trained_model(symbol, model_type, task):
                    available[task].append(model_type)
        
        return available

    def set_ensemble_predictor(self, ensemble_predictor) -> None:
        """アンサンブル予測器設定"""
        self.ensemble_predictor = ensemble_predictor

    def save_model(self, model_key: str, model: Any, metadata: Optional[ModelMetadata] = None, 
                   label_encoder: Optional[LabelEncoder] = None) -> None:
        """モデル保存"""
        try:
            # メモリに保存
            self.trained_models[model_key] = model
            if metadata:
                self.model_metadata[model_key] = metadata
            if label_encoder:
                self.label_encoders[model_key] = label_encoder

            # ファイルに保存
            file_path = self.models_dir / f"{model_key}.joblib"
            model_data = {'model': model}
            if metadata:
                model_data['metadata'] = metadata
            if label_encoder:
                model_data['label_encoder'] = label_encoder

            joblib.dump(model_data, file_path)
            self.logger.debug(f"モデル保存完了: {model_key}")

        except Exception as e:
            self.logger.error(f"モデル保存エラー: {e}")
            raise

    def delete_model(self, model_key: str) -> bool:
        """モデル削除"""
        try:
            # メモリから削除
            if model_key in self.trained_models:
                del self.trained_models[model_key]
            if model_key in self.model_metadata:
                del self.model_metadata[model_key]
            if model_key in self.label_encoders:
                del self.label_encoders[model_key]

            # ファイル削除
            file_path = self.models_dir / f"{model_key}.joblib"
            if file_path.exists():
                file_path.unlink()

            # データベースから削除
            self.metadata_manager.delete_model_metadata(model_key)

            self.logger.info(f"モデル削除完了: {model_key}")
            return True

        except Exception as e:
            self.logger.error(f"モデル削除エラー: {e}")
            return False

    def cleanup_old_models(self, days_to_keep: int = 30) -> Dict[str, int]:
        """古いモデルのクリーンアップ"""
        try:
            cutoff_date = datetime.now() - datetime.timedelta(days=days_to_keep)
            deleted_models = 0
            deleted_files = 0

            # ファイルベースのクリーンアップ
            for model_file in self.models_dir.glob("*.joblib"):
                if datetime.fromtimestamp(model_file.stat().st_mtime) < cutoff_date:
                    model_key = model_file.stem
                    if self.delete_model(model_key):
                        deleted_models += 1
                        deleted_files += 1

            # データベースクリーンアップ
            self.metadata_manager.cleanup_old_records(days_to_keep)

            result = {
                'deleted_models': deleted_models,
                'deleted_files': deleted_files
            }

            self.logger.info(f"古いモデルクリーンアップ完了: {result}")
            return result

        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return {'error': str(e)}

    def get_config_summary(self) -> Dict[str, Any]:
        """設定サマリー取得"""
        return {
            'model_configs': list(self.config.get('model_configs', {}).keys()),
            'training_config': self.config.get('training_config', {}),
            'data_dir': str(self.data_dir),
            'models_dir': str(self.models_dir),
            'db_path': str(self.db_path),
            'available_trainers': list(self.trainers.keys())
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """設定検証"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            # ディレクトリ存在確認
            if not self.data_dir.exists():
                validation_result['errors'].append(f"データディレクトリが存在しません: {self.data_dir}")

            # 訓練器確認
            if not self.trainers:
                validation_result['errors'].append("利用可能な訓練器がありません")

            # データベース接続確認
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("SELECT 1")
            except Exception as e:
                validation_result['errors'].append(f"データベース接続エラー: {e}")

            # 依存関係確認
            if not SKLEARN_AVAILABLE:
                validation_result['errors'].append("scikit-learnが利用できません")

            if not XGBOOST_AVAILABLE:
                validation_result['warnings'].append("XGBoostが利用できません")

            if not LIGHTGBM_AVAILABLE:
                validation_result['warnings'].append("LightGBMが利用できません")

            # 全体の有効性判定
            validation_result['is_valid'] = len(validation_result['errors']) == 0

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"設定検証エラー: {e}")

        return validation_result