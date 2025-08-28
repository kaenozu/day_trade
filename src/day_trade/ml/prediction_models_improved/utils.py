#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ユーティリティ関数とファクトリー - ML Prediction Models Utils

ML予測モデルのユーティリティ関数とファクトリー関数を提供します。
"""

import logging
from typing import Optional

from .core_models import MLPredictionModels
from .ensemble_predictor import EnhancedEnsemblePredictor


def create_improved_ml_prediction_models(config_path: Optional[str] = None) -> MLPredictionModels:
    """改善版MLPredictionModelsの作成"""
    try:
        # MLPredictionModelsインスタンス作成
        ml_models = MLPredictionModels(config_path)
        
        # アンサンブル予測器を作成してセット
        ensemble_predictor = EnhancedEnsemblePredictor(ml_models)
        ml_models.set_ensemble_predictor(ensemble_predictor)
        
        return ml_models
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"MLPredictionModels作成エラー: {e}")
        raise


def create_ensemble_predictor(ml_models: MLPredictionModels) -> EnhancedEnsemblePredictor:
    """アンサンブル予測器の作成"""
    return EnhancedEnsemblePredictor(ml_models)


def validate_model_dependencies() -> dict:
    """モデル依存関係の検証"""
    dependencies = {
        'sklearn': False,
        'xgboost': False,
        'lightgbm': False,
        'feature_engineering': False,
        'data_provider': False
    }
    
    try:
        import sklearn
        dependencies['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import xgboost
        dependencies['xgboost'] = True
    except ImportError:
        pass
    
    try:
        import lightgbm
        dependencies['lightgbm'] = True
    except ImportError:
        pass
    
    try:
        from src.day_trade.analysis.enhanced_feature_engineering import enhanced_feature_engineer
        dependencies['feature_engineering'] = True
    except ImportError:
        pass
    
    try:
        from src.day_trade.data.stock_fetcher import StockFetcher
        dependencies['data_provider'] = True
    except ImportError:
        pass
    
    return dependencies


def get_model_info() -> dict:
    """モデル情報取得"""
    return {
        'version': '2.0.0',
        'description': 'ML Prediction Models (Issue #850対応改善版)',
        'features': [
            'データ準備と特徴量エンジニアリングの頑健化',
            'モデル訓練ロジックの重複排除と抽象化',
            'モデルの永続化とメタデータ管理の強化',
            'アンサンブル予測ロジックの洗練',
            'データベーススキーマとデータ管理の改善',
            'テストコードの分離とフレームワーク統合'
        ],
        'supported_models': ['RandomForest', 'XGBoost', 'LightGBM'],
        'supported_tasks': ['PRICE_DIRECTION', 'PRICE_REGRESSION', 'VOLATILITY', 'TREND_STRENGTH']
    }


# グローバルインスタンス（後方互換性）
try:
    ml_prediction_models_improved = create_improved_ml_prediction_models()
except Exception as e:
    logging.getLogger(__name__).error(f"改善版MLモデル初期化失敗: {e}")
    ml_prediction_models_improved = None


# テスト用のメイン実行部分
if __name__ == "__main__":
    # 基本的な動作確認のみ
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info("ML Prediction Models (Issue #850対応改善版)")
    logger.info("主要改善点:")
    logger.info("- データ準備と特徴量エンジニアリングの頑健化")
    logger.info("- モデル訓練ロジックの重複排除と抽象化")
    logger.info("- モデルの永続化とメタデータ管理の強化")
    logger.info("- アンサンブル予測ロジックの洗練")
    logger.info("- データベーススキーマとデータ管理の改善")
    logger.info("- テストコードの分離とフレームワーク統合")
    logger.info("")
    logger.info("詳細なテストは tests/test_ml_prediction_models_improved.py を実行してください")

    try:
        # 依存関係チェック
        deps = validate_model_dependencies()
        logger.info(f"依存関係チェック: {deps}")
        
        # モデル情報表示
        info = get_model_info()
        logger.info(f"モデル情報: {info['description']} v{info['version']}")
        
        # インスタンス作成テスト
        models = create_improved_ml_prediction_models()
        logger.info(f"✓ 初期化成功: {models.data_dir}")
        logger.info(f"✓ データベース: {models.db_path}")
        logger.info(f"✓ 利用可能訓練器: {list(models.trainers.keys())}")

        # 基本設定確認
        summary = models.get_model_summary()
        logger.info(f"✓ モデルサマリー: {summary.get('total_models', 0)}個のモデル")

    except Exception as e:
        logger.error(f"✗ 初期化エラー: {e}")