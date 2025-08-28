#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファクトリーモジュール

ML予測システムの各種コンポーネントを生成するファクトリー関数
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from .core_models import MLPredictionModels
from .data_structures import TrainingConfig
from .data_preparation import DataPreparationPipeline
from .metadata_manager import ModelMetadataManager
from .utils import setup_logging


def create_ml_prediction_models(config_path: Optional[str] = None,
                               data_dir: Optional[str] = None,
                               log_level: str = "INFO") -> MLPredictionModels:
    """MLPredictionModelsインスタンスを作成するファクトリー関数
    
    Args:
        config_path: 設定ファイルのパス
        data_dir: データディレクトリのパス 
        log_level: ログレベル
        
    Returns:
        MLPredictionModelsインスタンス
    """
    # ロギング設定
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    try:
        models = MLPredictionModels(config_path)
        
        # カスタムデータディレクトリが指定された場合
        if data_dir:
            custom_data_dir = Path(data_dir)
            custom_data_dir.mkdir(exist_ok=True)
            models.data_dir = custom_data_dir
            models.models_dir = custom_data_dir / "models"
            models.models_dir.mkdir(exist_ok=True)
            
            # データベースパスも更新
            models.db_path = custom_data_dir / "ml_predictions_improved.db"
            
            # メタデータマネージャーを再初期化
            models.metadata_manager = ModelMetadataManager(models.db_path)
            
            logger.info(f"カスタムデータディレクトリを使用: {data_dir}")
        
        logger.info("MLPredictionModelsの作成が完了しました")
        return models
        
    except Exception as e:
        logger.error(f"MLPredictionModels作成エラー: {e}")
        raise


def create_training_config(test_size: float = 0.2,
                          enable_cross_validation: bool = True,
                          cv_folds: int = 5,
                          save_model: bool = True,
                          enable_scaling: bool = True,
                          n_jobs: int = -1,
                          **kwargs) -> TrainingConfig:
    """TrainingConfigインスタンスを作成するファクトリー関数
    
    Args:
        test_size: テストデータの割合
        enable_cross_validation: クロスバリデーション有効化
        cv_folds: CVのフォールド数
        save_model: モデル保存フラグ
        enable_scaling: スケーリング有効化
        n_jobs: 並列処理数
        **kwargs: その他のパラメータ
        
    Returns:
        TrainingConfigインスタンス
    """
    config = TrainingConfig(
        test_size=test_size,
        enable_cross_validation=enable_cross_validation,
        cv_folds=cv_folds,
        save_model=save_model,
        enable_scaling=enable_scaling,
        n_jobs=n_jobs
    )
    
    # 追加パラメータの設定
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_data_preparation_pipeline(config: Optional[TrainingConfig] = None,
                                   enable_scaling: bool = True,
                                   handle_missing_values: bool = True,
                                   outlier_detection: bool = False) -> DataPreparationPipeline:
    """DataPreparationPipelineインスタンスを作成するファクトリー関数
    
    Args:
        config: 訓練設定
        enable_scaling: スケーリング有効化
        handle_missing_values: 欠損値処理有効化
        outlier_detection: 外れ値検出有効化
        
    Returns:
        DataPreparationPipelineインスタンス
    """
    if config is None:
        config = TrainingConfig(
            enable_scaling=enable_scaling,
            handle_missing_values=handle_missing_values,
            outlier_detection=outlier_detection
        )
    
    return DataPreparationPipeline(config)


def create_metadata_manager(db_path: str) -> ModelMetadataManager:
    """ModelMetadataManagerインスタンスを作成するファクトリー関数
    
    Args:
        db_path: データベースファイルのパス
        
    Returns:
        ModelMetadataManagerインスタンス
    """
    return ModelMetadataManager(Path(db_path))


def create_development_environment(project_dir: Optional[str] = None,
                                 log_level: str = "DEBUG") -> Dict[str, Any]:
    """開発環境用のセットアップを行うファクトリー関数
    
    Args:
        project_dir: プロジェクトディレクトリのパス
        log_level: ログレベル
        
    Returns:
        開発環境のコンポーネント辞書
    """
    # プロジェクトディレクトリ設定
    if project_dir:
        data_dir = Path(project_dir) / "ml_dev_data"
    else:
        data_dir = Path("ml_dev_data")
    
    data_dir.mkdir(exist_ok=True)
    
    # ログファイル設定
    log_file = data_dir / "ml_development.log"
    setup_logging(log_level, log_file)
    
    # 開発用設定
    dev_config = create_training_config(
        test_size=0.2,
        enable_cross_validation=True,
        cv_folds=3,  # 開発用に少なく
        save_model=True,
        enable_scaling=True,
        n_jobs=1,  # 開発用にシングルスレッド
        verbose=True
    )
    
    # コンポーネント作成
    models = create_ml_prediction_models(
        data_dir=str(data_dir),
        log_level=log_level
    )
    
    data_pipeline = create_data_preparation_pipeline(dev_config)
    
    return {
        'models': models,
        'config': dev_config,
        'data_pipeline': data_pipeline,
        'data_dir': data_dir,
        'log_file': log_file
    }


def create_production_environment(config_path: str,
                                data_dir: str,
                                log_level: str = "INFO") -> Dict[str, Any]:
    """本番環境用のセットアップを行うファクトリー関数
    
    Args:
        config_path: 設定ファイルのパス
        data_dir: データディレクトリのパス
        log_level: ログレベル
        
    Returns:
        本番環境のコンポーネント辞書
    """
    # データディレクトリ確認
    prod_data_dir = Path(data_dir)
    if not prod_data_dir.exists():
        raise FileNotFoundError(f"本番データディレクトリが存在しません: {data_dir}")
    
    # ログファイル設定
    log_file = prod_data_dir / "ml_production.log"
    setup_logging(log_level, log_file)
    
    # 本番用設定
    prod_config = create_training_config(
        test_size=0.15,
        enable_cross_validation=True,
        cv_folds=5,
        save_model=True,
        enable_scaling=True,
        n_jobs=-1,
        verbose=False,
        performance_threshold=0.7,  # 本番では高い閾値
        min_data_quality='good'     # 本番では高品質データが必要
    )
    
    # コンポーネント作成
    models = create_ml_prediction_models(
        config_path=config_path,
        data_dir=data_dir,
        log_level=log_level
    )
    
    data_pipeline = create_data_preparation_pipeline(prod_config)
    
    return {
        'models': models,
        'config': prod_config,
        'data_pipeline': data_pipeline,
        'data_dir': prod_data_dir,
        'log_file': log_file
    }


def create_test_environment(test_data_dir: Optional[str] = None) -> Dict[str, Any]:
    """テスト環境用のセットアップを行うファクトリー関数
    
    Args:
        test_data_dir: テスト用データディレクトリのパス
        
    Returns:
        テスト環境のコンポーネント辞書
    """
    import tempfile
    
    # 一時ディレクトリの使用
    if test_data_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="ml_test_")
        test_dir = Path(temp_dir)
    else:
        test_dir = Path(test_data_dir)
        test_dir.mkdir(exist_ok=True)
    
    # テスト用設定
    test_config = create_training_config(
        test_size=0.3,  # テスト用に大きく
        enable_cross_validation=False,  # テスト高速化
        save_model=False,  # テストでは保存しない
        enable_scaling=True,
        n_jobs=1,  # テスト用にシングルスレッド
        verbose=False
    )
    
    # コンポーネント作成
    models = create_ml_prediction_models(
        data_dir=str(test_dir),
        log_level="WARNING"  # テスト時はエラーのみ
    )
    
    data_pipeline = create_data_preparation_pipeline(test_config)
    
    return {
        'models': models,
        'config': test_config,
        'data_pipeline': data_pipeline,
        'data_dir': test_dir,
        'temp_dir': test_data_dir is None  # 一時ディレクトリかどうかのフラグ
    }


# 後方互換性のためのエイリアス
create_improved_ml_prediction_models = create_ml_prediction_models


# グローバルインスタンス作成のヘルパー
_global_instance = None

def get_global_ml_models(config_path: Optional[str] = None,
                        force_recreate: bool = False) -> MLPredictionModels:
    """グローバルMLPredictionModelsインスタンスの取得
    
    Args:
        config_path: 設定ファイルのパス
        force_recreate: 強制再作成フラグ
        
    Returns:
        グローバルMLPredictionModelsインスタンス
    """
    global _global_instance
    
    if _global_instance is None or force_recreate:
        _global_instance = create_ml_prediction_models(config_path)
    
    return _global_instance


def cleanup_global_instance():
    """グローバルインスタンスのクリーンアップ"""
    global _global_instance
    _global_instance = None