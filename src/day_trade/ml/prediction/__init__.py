#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機械学習予測システム - 統合モジュール

分割された機能別モジュールを統合し、バックワード互換性を提供します。
元のml_prediction_models_improved.pyファイルと同じインターフェースを維持しながら、
改良された内部実装を使用します。
"""

import logging
from typing import Dict, List, Optional, Any, Union

# 基本型・データクラス
from .base_types import (
    TrainingConfig,
    PredictionResult, 
    EnsemblePrediction,
    FeatureEngineringConfig,
    DataQualityReport,
    ModelTrainingResult,
    DEFAULT_MODEL_CONFIGS,
    DATA_QUALITY_THRESHOLDS,
    PERFORMANCE_THRESHOLDS
)

# コアシステム
from .metadata_manager import ModelMetadataManager
from .data_preparation import DataPreparationPipeline
from .ensemble_predictor import EnhancedEnsemblePredictor
from .ml_models import MLPredictionModels

# ユーティリティ
from .utils import (
    create_improved_ml_prediction_models,
    create_default_training_config,
    create_enhanced_training_config,
    create_feature_engineering_config,
    validate_prediction_input,
    format_prediction_result,
    analyze_training_results,
    create_performance_report,
    cleanup_old_model_files,
    ml_prediction_models_improved
)

# 外部依存からのインポート
from src.day_trade.ml.core_types import (
    MLPredictionError,
    DataPreparationError,
    ModelTrainingError,
    ModelMetadataError,
    PredictionError,
    ModelType,
    PredictionTask,
    DataQuality,
    ModelMetadata,
    ModelPerformance,
    DataProvider,
    BaseModelTrainer
)

# ロギング設定
logger = logging.getLogger(__name__)

# バックワード互換性のためのエイリアス
# 元のファイルで使用されていたクラス名やインターフェースを維持

# 元のファイルのメインクラス名を維持
MLPredictionModelsImproved = MLPredictionModels

# 元のファイルで定義されていた関数のエイリアス
def create_ml_prediction_models_improved(config_path: Optional[str] = None) -> MLPredictionModels:
    """元のファクトリー関数名を維持（バックワード互換性）"""
    return create_improved_ml_prediction_models(config_path)

# 全ての公開APIをエクスポート
__all__ = [
    # メインクラス
    'MLPredictionModels',
    'MLPredictionModelsImproved',  # バックワード互換性
    
    # コンポーネントクラス
    'ModelMetadataManager',
    'DataPreparationPipeline', 
    'EnhancedEnsemblePredictor',
    
    # データクラス・設定
    'TrainingConfig',
    'PredictionResult',
    'EnsemblePrediction',
    'FeatureEngineringConfig',
    'DataQualityReport',
    'ModelTrainingResult',
    
    # ファクトリー関数
    'create_improved_ml_prediction_models',
    'create_ml_prediction_models_improved',  # バックワード互換性
    'create_default_training_config',
    'create_enhanced_training_config',
    'create_feature_engineering_config',
    
    # ユーティリティ関数
    'validate_prediction_input',
    'format_prediction_result',
    'analyze_training_results',
    'create_performance_report',
    'cleanup_old_model_files',
    
    # 定数
    'DEFAULT_MODEL_CONFIGS',
    'DATA_QUALITY_THRESHOLDS', 
    'PERFORMANCE_THRESHOLDS',
    
    # 外部型定義
    'MLPredictionError',
    'DataPreparationError',
    'ModelTrainingError',
    'ModelMetadataError',
    'PredictionError',
    'ModelType',
    'PredictionTask',
    'DataQuality',
    'ModelMetadata',
    'ModelPerformance',
    'DataProvider',
    'BaseModelTrainer',
    
    # グローバルインスタンス（バックワード互換性）
    'ml_prediction_models_improved'
]

# モジュール情報
__version__ = "2.0.0"
__author__ = "ML Prediction System"
__description__ = "機械学習予測システム - Issue #850対応改善版"

# 初期化ログ
logger.info("ML Prediction System initialized successfully")
logger.info("主要改善点:")
logger.info("- データ準備と特徴量エンジニアリングの頑健化")
logger.info("- モデル訓練ロジックの重複排除と抽象化")
logger.info("- モデルの永続化とメタデータ管理の強化")
logger.info("- アンサンブル予測ロジックの洗練")
logger.info("- データベーススキーマとデータ管理の改善")
logger.info("- モジュール化とコード分割")

# バックワード互換性のための警告
def _warn_deprecated(old_name: str, new_name: str):
    """非推奨機能の警告"""
    import warnings
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# 元のファイルの動作を模擬するための関数群（必要に応じて追加）
def get_improved_models_instance() -> Optional[MLPredictionModels]:
    """改善版モデルインスタンスの取得"""
    return ml_prediction_models_improved

def initialize_improved_models(config_path: Optional[str] = None) -> MLPredictionModels:
    """改善版モデルシステムの初期化"""
    try:
        return create_improved_ml_prediction_models(config_path)
    except Exception as e:
        logger.error(f"モデルシステム初期化失敗: {e}")
        raise

# 使用例とドキュメント
def get_usage_examples() -> Dict[str, str]:
    """使用例の取得"""
    examples = {
        'basic_usage': """
# 基本的な使用方法
from src.day_trade.ml.prediction import create_improved_ml_prediction_models

# モデルシステムの作成
ml_models = create_improved_ml_prediction_models()

# モデルの訓練
await ml_models.train_models('AAPL', period='1y')

# 予測の実行
import pandas as pd
features = pd.DataFrame(...)  # 特徴量データ
predictions = await ml_models.predict('AAPL', features)
""",
        
        'advanced_usage': """
# 高度な設定を使用した例
from src.day_trade.ml.prediction import (
    create_improved_ml_prediction_models,
    create_enhanced_training_config,
    create_feature_engineering_config
)

# カスタム設定の作成
training_config = create_enhanced_training_config(
    performance_threshold=0.7,
    min_data_quality="good",
    enable_cross_validation=True,
    outlier_detection=True
)

feature_config = create_feature_engineering_config(
    enable_technical_indicators=True,
    sma_periods=[10, 20, 50, 200],
    enable_rsi=True,
    enable_macd=True
)

# モデルシステムの作成
ml_models = create_improved_ml_prediction_models('config.json')

# カスタム設定での訓練
await ml_models.train_models(
    'AAPL', 
    period='1y', 
    config=training_config
)
""",

        'prediction_analysis': """
# 予測結果の分析
from src.day_trade.ml.prediction import format_prediction_result

# 予測実行
predictions = await ml_models.predict('AAPL', features)

# 結果のフォーマット
for task, prediction in predictions.items():
    formatted = format_prediction_result(prediction, include_details=True)
    print(f"{task.value}: {formatted}")
    
    # 予測品質の検証
    validation = ml_models.ensemble_predictor.validate_prediction_quality(prediction)
    if not validation['is_valid']:
        print(f"警告: {validation['issues']}")
"""
    }
    
    return examples

# ヘルプ機能
def show_help():
    """ヘルプ情報の表示"""
    help_text = f"""
機械学習予測システム v{__version__}
{__description__}

主要コンポーネント:
- MLPredictionModels: メインの予測システム
- ModelMetadataManager: モデルメタデータ管理
- DataPreparationPipeline: データ前処理と特徴量エンジニアリング  
- EnhancedEnsemblePredictor: アンサンブル予測システム

基本的な使用方法:
    from src.day_trade.ml.prediction import create_improved_ml_prediction_models
    
    # システム初期化
    ml_models = create_improved_ml_prediction_models()
    
    # モデル訓練
    await ml_models.train_models('SYMBOL')
    
    # 予測実行
    predictions = await ml_models.predict('SYMBOL', features_df)

詳細な使用例については get_usage_examples() を参照してください。
"""
    print(help_text)

# モジュール読み込み時の基本チェック
try:
    # 必須依存の確認
    import pandas as pd
    import numpy as np
    import sklearn
    logger.debug("必須依存関係の確認完了")
    
    # 設定可能な依存関係の確認
    optional_deps = []
    
    try:
        import xgboost
        optional_deps.append("XGBoost")
    except ImportError:
        pass
        
    try:
        import lightgbm
        optional_deps.append("LightGBM")
    except ImportError:
        pass
    
    if optional_deps:
        logger.info(f"オプショナル依存関係が利用可能: {', '.join(optional_deps)}")
    
except ImportError as e:
    logger.error(f"必須依存関係が不足しています: {e}")
    logger.error("pip install pandas numpy scikit-learn を実行してください")

# バージョン互換性チェック
import sys
if sys.version_info < (3, 8):
    logger.warning("Python 3.8以上の使用を推奨します")

# 初期化完了
logger.debug(f"ML Prediction System v{__version__} 読み込み完了")