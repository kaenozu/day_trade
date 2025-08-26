#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Prediction Models Package - 機械学習予測モデルパッケージ

Issue #850対応: 大規模ファイルの分割と保守性改善
"""

# バージョン情報
__version__ = "2.0.0"
__author__ = "Day Trade System"
__description__ = "Enhanced ML Prediction Models for Stock Trading"

# メインクラスとコンポーネントのインポート
from .core_models import MLPredictionModels
from .data_structures import (
    TrainingConfig,
    PredictionResult,
    EnsemblePrediction,
    ModelPerformance,
    ModelMetadata
)
from .metadata_manager import ModelMetadataManager
from .data_preparation import DataPreparationPipeline
from .model_training import ModelTrainingManager
from .ensemble_predictor import EnhancedEnsemblePredictor
from .confidence_calculator import ConfidenceCalculator
from .factory import (
    create_ml_prediction_models,
    create_improved_ml_prediction_models,  # 後方互換性エイリアス
    create_training_config,
    create_data_preparation_pipeline,
    create_metadata_manager,
    create_development_environment,
    create_production_environment,
    create_test_environment,
    get_global_ml_models,
    cleanup_global_instance
)
from .utils import (
    setup_logging,
    validate_symbol,
    format_performance_metrics,
    create_model_summary_report,
    ModelConstants,
    DEFAULT_MODEL_CONFIGS,
    ERROR_MESSAGES
)

# 後方互換性のための重要なエイリアス
# 元のファイルから直接インポートしていたコードが動作するように

# グローバルインスタンス（後方互換性）
ml_prediction_models_improved = None

def _initialize_global_instance():
    """グローバルインスタンスの初期化（遅延初期化）"""
    global ml_prediction_models_improved
    if ml_prediction_models_improved is None:
        try:
            ml_prediction_models_improved = create_ml_prediction_models()
        except Exception:
            # 初期化に失敗した場合はNoneのまま
            pass

# パッケージインポート時に実行される初期化処理
try:
    _initialize_global_instance()
except Exception:
    # 初期化エラーを無視して継続
    pass

# すべてのエクスポート対象を明示的に定義
__all__ = [
    # メインクラス
    'MLPredictionModels',
    
    # データ構造
    'TrainingConfig',
    'PredictionResult', 
    'EnsemblePrediction',
    'ModelPerformance',
    'ModelMetadata',
    
    # コンポーネントクラス
    'ModelMetadataManager',
    'DataPreparationPipeline',
    'ModelTrainingManager',
    'EnhancedEnsemblePredictor',
    'ConfidenceCalculator',
    
    # ファクトリー関数
    'create_ml_prediction_models',
    'create_improved_ml_prediction_models',
    'create_training_config',
    'create_data_preparation_pipeline',
    'create_metadata_manager',
    'create_development_environment',
    'create_production_environment',
    'create_test_environment',
    'get_global_ml_models',
    'cleanup_global_instance',
    
    # ユーティリティ
    'setup_logging',
    'validate_symbol',
    'format_performance_metrics',
    'create_model_summary_report',
    'ModelConstants',
    'DEFAULT_MODEL_CONFIGS',
    'ERROR_MESSAGES',
    
    # 後方互換性
    'ml_prediction_models_improved',
    
    # メタ情報
    '__version__',
    '__author__',
    '__description__'
]

# パッケージレベルの便利関数

def get_version_info():
    """パッケージバージョン情報を取得"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': len([name for name in __all__ if not name.startswith('__')]),
        'main_classes': [
            'MLPredictionModels',
            'ModelMetadataManager', 
            'DataPreparationPipeline',
            'ModelTrainingManager',
            'EnhancedEnsemblePredictor'
        ]
    }

def list_available_components():
    """利用可能なコンポーネント一覧を取得"""
    return {
        'main_classes': [
            'MLPredictionModels - メイン機械学習予測システム',
            'ModelMetadataManager - モデルメタデータ管理',
            'DataPreparationPipeline - データ準備パイプライン',
            'ModelTrainingManager - モデル訓練管理',
            'EnhancedEnsemblePredictor - アンサンブル予測システム',
            'ConfidenceCalculator - 信頼度計算システム'
        ],
        'data_structures': [
            'TrainingConfig - 訓練設定',
            'PredictionResult - 予測結果',
            'EnsemblePrediction - アンサンブル予測結果',
            'ModelPerformance - モデル性能情報',
            'ModelMetadata - モデルメタデータ'
        ],
        'factory_functions': [
            'create_ml_prediction_models - MLシステム作成',
            'create_training_config - 訓練設定作成',
            'create_development_environment - 開発環境セットアップ',
            'create_production_environment - 本番環境セットアップ',
            'create_test_environment - テスト環境セットアップ'
        ]
    }

def create_quick_start_example():
    """クイックスタート例の生成"""
    return '''
# ML Prediction Models パッケージのクイックスタート例

from src.day_trade.ml.prediction_models import (
    create_ml_prediction_models,
    create_training_config
)

# 1. 基本的な使用方法
models = create_ml_prediction_models()

# 2. カスタム設定での作成
config = create_training_config(
    test_size=0.2,
    enable_cross_validation=True,
    cv_folds=5
)
models_custom = create_ml_prediction_models()

# 3. モデル訓練
await models.train_models("7203", period="1y", config=config)

# 4. 予測実行
import pandas as pd
features = pd.DataFrame(...)  # 特徴量データ
predictions = await models.predict("7203", features)

# 5. サマリー取得
summary = models.get_model_summary()
print(summary)
'''

# ログ設定（オプション）
def configure_package_logging(level="INFO", log_file=None):
    """パッケージレベルのログ設定"""
    return setup_logging(level, log_file)

# パッケージ情報の表示（デバッグ用）
def show_package_info():
    """パッケージ情報の表示"""
    print(f"ML Prediction Models Package v{__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print(f"Available components: {len(__all__)} items")
    
    if ml_prediction_models_improved is not None:
        print("✓ Global instance initialized successfully")
    else:
        print("⚠ Global instance not initialized")

# 開発者向けのデバッグ情報
def _debug_package_status():
    """パッケージの状態をデバッグ出力（開発者用）"""
    import sys
    print(f"Python version: {sys.version}")
    print(f"Package version: {__version__}")
    print(f"Available modules: {__all__}")
    
    # 依存関係チェック
    try:
        import pandas
        print(f"✓ pandas {pandas.__version__}")
    except ImportError:
        print("✗ pandas not available")
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy not available")
        
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not available")