#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Prediction Models Improved Package

ML予測モデルの改善版パッケージ - Issue #850対応

このパッケージは、元の ml_prediction_models_improved.py を論理的な単位で分割し、
300行以下のモジュールに整理したものです。後方互換性を保つため、
元のクラスと関数をここで再エクスポートします。
"""

# メインコンポーネントのインポート
from .core_models import MLPredictionModels
from .data_preparation import DataPreparationPipeline
from .data_types import (
    DEFAULT_MODEL_CONFIGS,
    DEFAULT_TRAINING_CONFIG,
    QUALITY_THRESHOLDS,
    EnsemblePrediction,
    EnsembleWeights,
    ModelConfiguration,
    ModelStatus,
    PredictionMetrics,
    PredictionRequest,
    PredictionResult,
    TrainingConfig,
    ValidationResult,
)
from .ensemble_predictor import EnhancedEnsemblePredictor
from .feature_engineering import FeatureEngineer
from .metadata_manager import ModelMetadataManager
from .model_manager import ModelManager
from .model_trainer import ModelTrainerSystem
from .prediction_utils import PredictionUtils
from .utils import (
    create_ensemble_predictor,
    create_improved_ml_prediction_models,
    get_model_info,
    ml_prediction_models_improved,
    validate_model_dependencies,
)

# 後方互換性のための別名
MLPredictionModelsImproved = MLPredictionModels

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade Sub Team"
__description__ = "ML Prediction Models (Issue #850対応改善版)"

# 公開API
__all__ = [
    # メインクラス
    "MLPredictionModels",
    "MLPredictionModelsImproved",  # 後方互換性
    "DataPreparationPipeline", 
    "EnhancedEnsemblePredictor",
    "ModelMetadataManager",
    "ModelManager",
    "ModelTrainerSystem",
    "FeatureEngineer",
    "PredictionUtils",
    
    # データ型
    "TrainingConfig",
    "PredictionResult",
    "EnsemblePrediction",
    "ModelConfiguration",
    "ValidationResult",
    "PredictionRequest",
    "ModelStatus",
    "EnsembleWeights",
    "PredictionMetrics",
    
    # 定数
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_MODEL_CONFIGS",
    "QUALITY_THRESHOLDS",
    
    # ユーティリティ関数
    "create_improved_ml_prediction_models",
    "create_ensemble_predictor",
    "validate_model_dependencies",
    "get_model_info",
    
    # グローバルインスタンス
    "ml_prediction_models_improved",
    
    # メタ情報
    "__version__",
    "__author__",
    "__description__",
]

# パッケージ初期化時の設定
def _configure_package():
    """パッケージ設定"""
    import logging
    
    # ログ設定
    logger = logging.getLogger(__name__)
    logger.info(f"ML Prediction Models Improved パッケージ初期化完了 v{__version__}")
    
    # 依存関係チェック（警告レベル）
    try:
        deps = validate_model_dependencies()
        missing_deps = [k for k, v in deps.items() if not v]
        if missing_deps:
            logger.warning(f"一部の依存関係が利用できません: {missing_deps}")
    except Exception as e:
        logger.warning(f"依存関係チェック中にエラー: {e}")

# パッケージ初期化実行
_configure_package()


# 統一された予測インターフェース（後方互換性）
async def predict(symbol: str, features, **kwargs):
    """統一された予測インターフェース（後方互換性）"""
    if ml_prediction_models_improved and ml_prediction_models_improved.ensemble_predictor:
        return await ml_prediction_models_improved.ensemble_predictor.predict(symbol, features)
    else:
        raise RuntimeError("MLPredictionModelsが初期化されていません")


async def predict_list(symbol: str, features, **kwargs):
    """リスト形式予測（後方互換性）"""
    predictions_dict = await predict(symbol, features, **kwargs)
    return list(predictions_dict.values())


# パッケージレベルの設定関数
def configure_logging(level=None, format_string=None):
    """ログ設定"""
    import logging
    
    if level:
        logging.getLogger(__name__).setLevel(level)
    
    if format_string:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logging.getLogger(__name__).addHandler(handler)


def get_package_status():
    """パッケージ状態取得"""
    status = {
        'version': __version__,
        'description': __description__,
        'dependencies': validate_model_dependencies(),
        'global_instance_available': ml_prediction_models_improved is not None,
        'modules_loaded': {
            'core_models': 'MLPredictionModels' in globals(),
            'data_preparation': 'DataPreparationPipeline' in globals(),
            'ensemble_predictor': 'EnhancedEnsemblePredictor' in globals(),
            'metadata_manager': 'ModelMetadataManager' in globals(),
            'utils': 'create_improved_ml_prediction_models' in globals(),
        }
    }
    
    if ml_prediction_models_improved:
        try:
            summary = ml_prediction_models_improved.get_model_summary()
            status['model_summary'] = summary
        except Exception as e:
            status['model_summary_error'] = str(e)
    
    return status


# デバッグ用
def debug_info():
    """デバッグ情報出力"""
    import sys
    print(f"Python: {sys.version}")
    print(f"Package: {__name__} v{__version__}")
    print(f"Description: {__description__}")
    print("\nPackage Status:")
    
    status = get_package_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\nAvailable Classes and Functions:")
    for item in __all__:
        if item in globals():
            obj = globals()[item]
            obj_type = type(obj).__name__
            print(f"  {item}: {obj_type}")
        else:
            print(f"  {item}: NOT AVAILABLE")


# 開発・テスト用のメイン実行
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== ML Prediction Models Improved Package ===")
    debug_info()
    
    # 基本動作テスト
    try:
        print("\n=== 基本動作テスト ===")
        models = create_improved_ml_prediction_models()
        print(f"✓ MLPredictionModels作成成功: {type(models)}")
        
        summary = models.get_model_summary()
        print(f"✓ サマリー取得成功: {summary.get('total_models', 0)}モデル")
        
        predictor = create_ensemble_predictor(models)
        print(f"✓ アンサンブル予測器作成成功: {type(predictor)}")
        
    except Exception as e:
        print(f"✗ 基本動作テストエラー: {e}")