#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ユーティリティ・ファクトリー関数

機械学習予測システムの便利な関数、ヘルパー関数、
ファクトリー関数を提供します。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np

from .base_types import (
    TrainingConfig,
    FeatureEngineringConfig,
    EnsemblePrediction,
    ModelTrainingResult,
    DataQualityReport
)
from .ml_models import MLPredictionModels
from src.day_trade.ml.core_types import ModelType, PredictionTask, DataQuality


def create_improved_ml_prediction_models(config_path: Optional[str] = None) -> MLPredictionModels:
    """改善版MLPredictionModelsの作成"""
    return MLPredictionModels(config_path)


def create_default_training_config() -> TrainingConfig:
    """デフォルト訓練設定の作成"""
    return TrainingConfig()


def create_enhanced_training_config(
    performance_threshold: float = 0.65,
    min_data_quality: str = "good",
    enable_cross_validation: bool = True,
    cv_folds: int = 5,
    enable_scaling: bool = True,
    outlier_detection: bool = True
) -> TrainingConfig:
    """強化された訓練設定の作成"""
    quality_mapping = {
        "excellent": DataQuality.EXCELLENT,
        "good": DataQuality.GOOD,
        "fair": DataQuality.FAIR,
        "poor": DataQuality.POOR,
        "insufficient": DataQuality.INSUFFICIENT
    }
    
    return TrainingConfig(
        performance_threshold=performance_threshold,
        min_data_quality=quality_mapping.get(min_data_quality, DataQuality.FAIR),
        enable_cross_validation=enable_cross_validation,
        cv_folds=cv_folds,
        enable_scaling=enable_scaling,
        outlier_detection=outlier_detection
    )


def create_feature_engineering_config(
    enable_technical_indicators: bool = True,
    sma_periods: Optional[List[int]] = None,
    enable_rsi: bool = True,
    enable_macd: bool = True,
    enable_bollinger_bands: bool = True
) -> FeatureEngineringConfig:
    """特徴量エンジニアリング設定の作成"""
    return FeatureEngineringConfig(
        enable_technical_indicators=enable_technical_indicators,
        sma_periods=sma_periods or [5, 10, 20, 50],
        enable_rsi=enable_rsi,
        enable_macd=enable_macd,
        enable_bollinger_bands=enable_bollinger_bands
    )


def validate_prediction_input(features: pd.DataFrame) -> Dict[str, Any]:
    """予測入力の検証"""
    validation_result = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'recommendations': []
    }

    try:
        # 基本チェック
        if features.empty:
            validation_result['is_valid'] = False
            validation_result['issues'].append("特徴量データが空です")
            return validation_result

        if len(features) == 0:
            validation_result['is_valid'] = False
            validation_result['issues'].append("特徴量データに行がありません")
            return validation_result

        # 欠損値チェック
        missing_count = features.isnull().sum().sum()
        if missing_count > 0:
            missing_rate = missing_count / (len(features) * len(features.columns))
            if missing_rate > 0.1:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"欠損値率が高すぎます: {missing_rate:.1%}")
            else:
                validation_result['warnings'].append(f"欠損値が存在します: {missing_count}個")

        # 無限値チェック
        inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"無限値が存在します: {inf_count}個")

        # 特徴量数チェック
        if len(features.columns) < 5:
            validation_result['warnings'].append("特徴量数が少ないです")
            validation_result['recommendations'].append("より多くの特徴量を追加することを検討してください")

        # データ型チェック
        non_numeric_cols = features.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            validation_result['warnings'].append(f"非数値カラムが存在します: {list(non_numeric_cols)}")
            validation_result['recommendations'].append("非数値カラムの処理を確認してください")

    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['issues'].append(f"検証処理エラー: {e}")

    return validation_result


def format_prediction_result(prediction: EnsemblePrediction, 
                           include_details: bool = False) -> Dict[str, Any]:
    """予測結果のフォーマット"""
    formatted = {
        'symbol': prediction.symbol,
        'timestamp': prediction.timestamp.isoformat(),
        'prediction': prediction.final_prediction,
        'confidence': round(prediction.confidence, 3),
        'consensus_strength': round(prediction.consensus_strength, 3),
        'models_used': prediction.total_models_used,
        'ensemble_method': prediction.ensemble_method
    }

    if include_details:
        formatted.update({
            'model_predictions': prediction.model_predictions,
            'model_confidences': {k: round(v, 3) for k, v in prediction.model_confidences.items()},
            'model_weights': {str(k): round(v, 3) for k, v in prediction.model_weights.items()},
            'model_quality_scores': {k: round(v, 3) for k, v in prediction.model_quality_scores.items()},
            'disagreement_score': round(prediction.disagreement_score, 3),
            'prediction_stability': round(prediction.prediction_stability, 3),
            'diversity_score': round(prediction.diversity_score, 3),
            'excluded_models': prediction.excluded_models
        })

        if prediction.prediction_interval:
            formatted['prediction_interval'] = [
                round(prediction.prediction_interval[0], 3),
                round(prediction.prediction_interval[1], 3)
            ]

    return formatted


def analyze_training_results(results: Dict[ModelType, Dict[PredictionTask, Any]]) -> Dict[str, Any]:
    """訓練結果の分析"""
    analysis = {
        'total_models_trained': 0,
        'successful_trainings': 0,
        'failed_trainings': 0,
        'best_performing_models': {},
        'performance_summary': {},
        'recommendations': []
    }

    try:
        for model_type, task_results in results.items():
            analysis['total_models_trained'] += len(task_results)
            
            for task, result in task_results.items():
                if hasattr(result, 'accuracy') or hasattr(result, 'r2_score'):
                    analysis['successful_trainings'] += 1
                    
                    # 性能メトリクスの記録
                    task_key = f"{model_type.value}_{task.value}"
                    if task == PredictionTask.PRICE_DIRECTION:
                        performance_score = getattr(result, 'f1_score', 0.0)
                        analysis['performance_summary'][task_key] = {
                            'metric': 'f1_score',
                            'value': round(performance_score, 3)
                        }
                    else:
                        performance_score = getattr(result, 'r2_score', 0.0)
                        analysis['performance_summary'][task_key] = {
                            'metric': 'r2_score',
                            'value': round(performance_score or 0.0, 3)
                        }

                    # 最高性能モデルの特定
                    task_str = task.value
                    if task_str not in analysis['best_performing_models']:
                        analysis['best_performing_models'][task_str] = {
                            'model': model_type.value,
                            'score': performance_score
                        }
                    elif performance_score > analysis['best_performing_models'][task_str]['score']:
                        analysis['best_performing_models'][task_str] = {
                            'model': model_type.value,
                            'score': performance_score
                        }

                else:
                    analysis['failed_trainings'] += 1

        # 成功率
        if analysis['total_models_trained'] > 0:
            success_rate = analysis['successful_trainings'] / analysis['total_models_trained']
            analysis['success_rate'] = round(success_rate, 3)

            # 推奨事項
            if success_rate < 0.5:
                analysis['recommendations'].append("訓練成功率が低いです。データ品質やパラメータを見直してください")
            elif success_rate < 0.8:
                analysis['recommendations'].append("一部のモデル訓練が失敗しています。ログを確認してください")

            # 性能に基づく推奨事項
            avg_performance = np.mean([
                info['score'] for info in analysis['best_performing_models'].values()
            ])
            
            if avg_performance < 0.6:
                analysis['recommendations'].append("モデル性能が低いです。特徴量エンジニアリングの改善を検討してください")
            elif avg_performance > 0.8:
                analysis['recommendations'].append("優秀な性能です。本番環境への展開を検討できます")

    except Exception as e:
        analysis['error'] = f"分析処理エラー: {e}"

    return analysis


def create_performance_report(model_performances: List[Any], 
                            symbol: str) -> Dict[str, Any]:
    """性能レポートの作成"""
    report = {
        'symbol': symbol,
        'report_date': datetime.now().isoformat(),
        'total_models': len(model_performances),
        'performance_metrics': {},
        'model_comparison': [],
        'summary': {}
    }

    try:
        if not model_performances:
            report['summary'] = {'status': 'no_data', 'message': '性能データがありません'}
            return report

        # 各モデルの性能指標を集計
        classification_scores = []
        regression_scores = []

        for perf in model_performances:
            model_info = {
                'model_id': getattr(perf, 'model_id', 'unknown'),
                'model_type': getattr(perf, 'model_type', 'unknown'),
                'task': getattr(perf, 'task', 'unknown')
            }

            if hasattr(perf, 'f1_score') and perf.f1_score is not None:
                model_info.update({
                    'accuracy': round(getattr(perf, 'accuracy', 0.0), 3),
                    'precision': round(getattr(perf, 'precision', 0.0), 3),
                    'recall': round(getattr(perf, 'recall', 0.0), 3),
                    'f1_score': round(perf.f1_score, 3)
                })
                classification_scores.append(perf.f1_score)

            elif hasattr(perf, 'r2_score') and perf.r2_score is not None:
                model_info.update({
                    'r2_score': round(perf.r2_score, 3),
                    'mse': round(getattr(perf, 'mse', 0.0), 3),
                    'rmse': round(getattr(perf, 'rmse', 0.0), 3),
                    'mae': round(getattr(perf, 'mae', 0.0), 3)
                })
                regression_scores.append(perf.r2_score)

            report['model_comparison'].append(model_info)

        # サマリー統計
        if classification_scores:
            report['performance_metrics']['classification'] = {
                'avg_f1_score': round(np.mean(classification_scores), 3),
                'max_f1_score': round(np.max(classification_scores), 3),
                'min_f1_score': round(np.min(classification_scores), 3),
                'std_f1_score': round(np.std(classification_scores), 3)
            }

        if regression_scores:
            report['performance_metrics']['regression'] = {
                'avg_r2_score': round(np.mean(regression_scores), 3),
                'max_r2_score': round(np.max(regression_scores), 3),
                'min_r2_score': round(np.min(regression_scores), 3),
                'std_r2_score': round(np.std(regression_scores), 3)
            }

        # 総合評価
        all_scores = classification_scores + regression_scores
        if all_scores:
            avg_score = np.mean(all_scores)
            if avg_score >= 0.8:
                report['summary'] = {'status': 'excellent', 'avg_score': round(avg_score, 3)}
            elif avg_score >= 0.7:
                report['summary'] = {'status': 'good', 'avg_score': round(avg_score, 3)}
            elif avg_score >= 0.6:
                report['summary'] = {'status': 'fair', 'avg_score': round(avg_score, 3)}
            else:
                report['summary'] = {'status': 'poor', 'avg_score': round(avg_score, 3)}

    except Exception as e:
        report['error'] = f"レポート作成エラー: {e}"

    return report


def cleanup_old_model_files(models_dir: Path, days_to_keep: int = 30) -> int:
    """古いモデルファイルのクリーンアップ"""
    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_count = 0
        for model_file in models_dir.glob("*.joblib"):
            try:
                file_modified = datetime.fromtimestamp(model_file.stat().st_mtime)
                if file_modified < cutoff_date:
                    model_file.unlink()
                    deleted_count += 1
            except Exception as e:
                logging.getLogger(__name__).error(f"ファイル削除エラー {model_file}: {e}")

        logging.getLogger(__name__).info(f"古いモデルファイルを削除しました: {deleted_count}件")
        return deleted_count

    except Exception as e:
        logging.getLogger(__name__).error(f"クリーンアップエラー: {e}")
        return 0


# 後方互換性のためのグローバル変数
try:
    ml_prediction_models_improved = create_improved_ml_prediction_models()
except Exception as e:
    logging.getLogger(__name__).error(f"改善版MLモデル初期化失敗: {e}")
    ml_prediction_models_improved = None