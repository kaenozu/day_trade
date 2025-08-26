#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Evaluator - メトリクス計算・性能評価

MLモデルの性能メトリクス計算と評価を行うモジュール
実際のvalidatorとの連携、模擬データ生成、性能ステータス判定を提供
"""

import asyncio
import logging
import numpy as np
import time
from datetime import datetime
from typing import Optional
from .config import EnhancedPerformanceConfigManager
from .types import PerformanceMetrics, PerformanceStatus

logger = logging.getLogger(__name__)

# 外部システム連携チェック
try:
    from prediction_accuracy_validator import PredictionAccuracyValidator
    ACCURACY_VALIDATOR_AVAILABLE = True
except ImportError:
    ACCURACY_VALIDATOR_AVAILABLE = False


class MetricsEvaluator:
    """メトリクス評価クラス
    
    MLモデルの性能メトリクスを評価し、PerformanceMetricsオブジェクトを生成します。
    実際のvalidatorが利用可能な場合はそれを使用し、なければ模擬データを生成します。
    
    Attributes:
        config_manager: 設定管理インスタンス
        accuracy_validator: 精度検証システムインスタンス
    """

    def __init__(self, config_manager: EnhancedPerformanceConfigManager):
        """初期化
        
        Args:
            config_manager: 設定管理インスタンス
        """
        self.config_manager = config_manager
        self.accuracy_validator = None

        # 精度検証システムの初期化
        if ACCURACY_VALIDATOR_AVAILABLE:
            try:
                self.accuracy_validator = PredictionAccuracyValidator()
                logger.info("PredictionAccuracyValidator統合完了")
            except Exception as e:
                logger.warning(f"PredictionAccuracyValidator統合失敗: {e}")
                self.accuracy_validator = None
        else:
            logger.info("PredictionAccuracyValidatorは利用不可、模擬データを使用")

    async def evaluate_model_performance(
            self, symbol: str, model_type: str
    ) -> Optional[PerformanceMetrics]:
        """個別モデル性能評価
        
        指定された銘柄とモデルタイプの性能を評価し、PerformanceMetricsを生成します。
        
        Args:
            symbol: 銘柄コード
            model_type: モデルタイプ
            
        Returns:
            性能メトリクス（評価失敗時はNone）
        """
        try:
            if not self.accuracy_validator:
                # 模擬データで評価
                return self._generate_mock_performance(symbol, model_type)

            # 実際の精度検証
            monitoring_config = self.config_manager.get_monitoring_config()
            validation_hours = monitoring_config.get('validation_hours', 168)

            logger.debug(f"性能評価開始: {symbol}_{model_type}")

            if hasattr(self.accuracy_validator, 'validate_current_system_accuracy'):
                result = await self.accuracy_validator.validate_current_system_accuracy(
                    [symbol], validation_hours
                )
                accuracy = (result.overall_accuracy 
                           if hasattr(result, 'overall_accuracy') else 0.0)
                precision = getattr(result, 'precision', 0.0)
                recall = getattr(result, 'recall', 0.0)
                f1_score = getattr(result, 'f1_score', 0.0)
                confidence_avg = getattr(result, 'confidence_avg', 0.0)
                sample_size = getattr(result, 'sample_count', 0)
                
            else:
                # レガシーインターフェース
                test_symbols = [symbol]
                results = await self.accuracy_validator.validate_prediction_accuracy(
                    test_symbols
                )
                if symbol not in results:
                    logger.warning(f"検証結果に{symbol}が含まれていません")
                    return None
                    
                result = results[symbol]
                accuracy = result.get('accuracy', 0.0)
                precision = result.get('precision', 0.0)
                recall = result.get('recall', 0.0)
                f1_score = result.get('f1_score', 0.0)
                confidence_avg = result.get('confidence_avg', 0.0)
                sample_size = result.get('sample_size', 0)

            # 予測時間測定（模擬）
            start_time = time.time()
            await asyncio.sleep(0.001)  # 模擬処理時間
            prediction_time = (time.time() - start_time) * 1000

            # PerformanceMetricsオブジェクト作成
            metrics = PerformanceMetrics(
                symbol=symbol,
                model_type=model_type,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                prediction_time_ms=prediction_time,
                confidence_avg=confidence_avg,
                sample_size=sample_size,
                prediction_accuracy=getattr(result, 'prediction_accuracy', 0.0),
                return_prediction=getattr(result, 'return_prediction', 0.0),
                volatility_prediction=getattr(result, 'volatility_prediction', 0.0),
                source='enhanced_validator'
            )

            # ステータス判定
            metrics.status = self._determine_performance_status(metrics)

            logger.debug(f"性能評価完了: {symbol}_{model_type} - 精度: {accuracy:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"モデル性能評価エラー {symbol}_{model_type}: {e}")
            return None

    def _generate_mock_performance(
            self, symbol: str, model_type: str
    ) -> PerformanceMetrics:
        """模擬性能データ生成
        
        実際のvalidatorが利用できない場合の模擬データを生成します。
        銘柄やモデルタイプに基づいて現実的なばらつきを持つデータを生成します。
        
        Args:
            symbol: 銘柄コード
            model_type: モデルタイプ
            
        Returns:
            模擬性能メトリクス
        """
        # 銘柄別の基準性能設定
        symbol_base_performance = {
            '7203': 0.82,   # トヨタ（高性能）
            '8306': 0.78,   # 三菱UFJ
            '9984': 0.80,   # ソフトバンク
            '4751': 0.76,   # サイバーエージェント
            '2914': 0.74    # JT
        }

        # モデルタイプ別の調整
        model_adjustments = {
            'RandomForestClassifier': 0.02,
            'XGBoostClassifier': 0.04,
            'LogisticRegression': -0.02,
            'SVM': 0.01,
            'Default': 0.0
        }

        # 基準精度の計算
        base_accuracy = symbol_base_performance.get(symbol, 0.78)
        model_adjustment = model_adjustments.get(model_type, 0.0)
        target_accuracy = base_accuracy + model_adjustment

        # ランダムなノイズ追加
        accuracy = target_accuracy + np.random.normal(0, 0.05)
        accuracy = max(0.6, min(0.95, accuracy))  # 60-95%の範囲に制限

        # 他のメトリクスを生成（精度に基づく相関）
        precision = accuracy + np.random.normal(0, 0.02)
        recall = accuracy + np.random.normal(0, 0.02)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        confidence_avg = accuracy + np.random.normal(0, 0.03)

        # 予測精度系のメトリクス
        prediction_accuracy = accuracy * 100 + np.random.normal(0, 5)
        return_prediction = (accuracy - 0.05) * 100 + np.random.normal(0, 3)
        volatility_prediction = (accuracy + 0.02) * 100 + np.random.normal(0, 4)

        metrics = PerformanceMetrics(
            symbol=symbol,
            model_type=model_type,
            timestamp=datetime.now(),
            accuracy=accuracy,
            precision=max(0.0, min(1.0, precision)),
            recall=max(0.0, min(1.0, recall)),
            f1_score=max(0.0, min(1.0, f1_score)),
            prediction_time_ms=np.random.uniform(50, 200),
            confidence_avg=max(0.0, min(1.0, confidence_avg)),
            sample_size=np.random.randint(100, 1000),
            prediction_accuracy=max(0.0, min(100.0, prediction_accuracy)),
            return_prediction=max(0.0, min(100.0, return_prediction)),
            volatility_prediction=max(0.0, min(100.0, volatility_prediction)),
            source='mock_generator'
        )

        metrics.status = self._determine_performance_status(metrics)
        logger.debug(f"模擬データ生成: {symbol}_{model_type} - 精度: {accuracy:.3f}")
        
        return metrics

    def _determine_performance_status(
            self, metrics: PerformanceMetrics
    ) -> PerformanceStatus:
        """性能ステータス判定
        
        メトリクスの値に基づいて性能ステータスを判定します。
        
        Args:
            metrics: 性能メトリクス
            
        Returns:
            判定された性能ステータス
        """
        thresholds = (self.config_manager.config.get('performance_thresholds', {})
                     .get('accuracy', {}))
        accuracy = metrics.accuracy or 0.0

        if accuracy >= thresholds.get('target_threshold', 0.90):
            return PerformanceStatus.EXCELLENT
        elif accuracy >= thresholds.get('warning_threshold', 0.80):
            return PerformanceStatus.GOOD
        elif accuracy >= thresholds.get('minimum_threshold', 0.75):
            return PerformanceStatus.WARNING
        elif accuracy >= thresholds.get('critical_threshold', 0.70):
            return PerformanceStatus.CRITICAL
        else:
            return PerformanceStatus.CRITICAL

    def validate_metrics(self, metrics: PerformanceMetrics) -> bool:
        """メトリクスの妥当性検証
        
        生成されたメトリクスが妥当な範囲にあるかチェックします。
        
        Args:
            metrics: 検証対象のメトリクス
            
        Returns:
            妥当な場合はTrue
        """
        try:
            # 必須フィールドのチェック
            if not metrics.symbol or not metrics.model_type:
                return False

            # 数値範囲のチェック
            if metrics.accuracy is not None:
                if not (0.0 <= metrics.accuracy <= 1.0):
                    return False

            if metrics.precision is not None:
                if not (0.0 <= metrics.precision <= 1.0):
                    return False

            if metrics.recall is not None:
                if not (0.0 <= metrics.recall <= 1.0):
                    return False

            if metrics.prediction_time_ms is not None:
                if metrics.prediction_time_ms < 0:
                    return False

            # サンプルサイズのチェック
            if metrics.sample_size < 0:
                return False

            return True

        except Exception as e:
            logger.error(f"メトリクス検証エラー: {e}")
            return False

    def calculate_performance_trend(
            self, historical_metrics: list
    ) -> dict:
        """性能トレンドの計算
        
        履歴メトリクスから性能のトレンドを計算します。
        
        Args:
            historical_metrics: 履歴メトリクスのリスト
            
        Returns:
            トレンド情報辞書
        """
        if len(historical_metrics) < 2:
            return {'trend': 'insufficient_data', 'change_rate': 0.0}

        # 精度の変化を計算
        accuracies = [m.accuracy for m in historical_metrics if m.accuracy is not None]
        
        if len(accuracies) < 2:
            return {'trend': 'insufficient_data', 'change_rate': 0.0}

        # 線形回帰による傾向分析
        x = np.arange(len(accuracies))
        coefficients = np.polyfit(x, accuracies, 1)
        slope = coefficients[0]

        # トレンド判定
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'degrading'
        else:
            trend = 'stable'

        # 変化率計算
        recent_avg = np.mean(accuracies[-3:]) if len(accuracies) >= 3 else accuracies[-1]
        older_avg = np.mean(accuracies[:3]) if len(accuracies) >= 6 else accuracies[0]
        change_rate = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0

        return {
            'trend': trend,
            'slope': slope,
            'change_rate': change_rate,
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'data_points': len(accuracies)
        }