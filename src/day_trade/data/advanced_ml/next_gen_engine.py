#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine Module

ハイブリッドLSTM-Transformer統合エンジン
"""

import time
from typing import Any, Dict, Optional

import pandas as pd

from ...ml.deep_learning_models import DeepLearningConfig, DeepLearningModelManager
from ...ml.hybrid_lstm_transformer import HybridModelConfig, create_hybrid_model
from ...utils.logging_config import get_context_logger
from .performance_evaluator import PerformanceEvaluator
from .technical_indicators import TechnicalIndicatorCalculator

logger = get_context_logger(__name__)


class NextGenAITradingEngine:
    """
    Next-Gen AI Trading Engine - ハイブリッドLSTM-Transformer統合エンジン

    最新のハイブリッドアーキテクチャによる高精度市場予測システム
    目標性能: 95%+ 予測精度, <100ms推論時間, MAE<0.6, RMSE<0.8
    """

    def __init__(self, config: Optional[HybridModelConfig] = None):
        self.config = config or HybridModelConfig()
        self.hybrid_model = None
        self.dl_manager = None

        # 性能評価器
        self.performance_evaluator = PerformanceEvaluator()

        # テクニカル指標計算器
        self.technical_calculator = TechnicalIndicatorCalculator()

        # 既存システムとの互換性維持
        self.legacy_engine = None

        # メトリクス統合（監視システム対応）
        self.metrics_integration = True

        logger.info("Next-Gen AI Trading Engine 初期化完了")

    def initialize_hybrid_model(self) -> bool:
        """ハイブリッドモデル初期化"""
        try:
            self.hybrid_model = create_hybrid_model(self.config)

            # DeepLearningModelManager統合
            dl_config = DeepLearningConfig(
                sequence_length=self.config.sequence_length,
                prediction_horizon=self.config.prediction_horizon,
                hidden_dim=self.config.lstm_hidden_size,
                num_layers=self.config.lstm_num_layers,
                dropout_rate=self.config.dropout_rate,
                learning_rate=self.config.learning_rate,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
            )

            self.dl_manager = DeepLearningModelManager(dl_config)

            # ハイブリッドモデル登録
            self.dl_manager.register_model("hybrid_lstm_transformer", self.hybrid_model)

            logger.info("ハイブリッドLSTM-Transformerモデル初期化完了")
            return True

        except Exception as e:
            logger.error(f"ハイブリッドモデル初期化エラー: {e}")
            return False

    def train_next_gen_model(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        enable_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """次世代モデル訓練"""
        start_time = time.time()

        try:
            # ハイブリッドモデル初期化
            if self.hybrid_model is None:
                if not self.initialize_hybrid_model():
                    raise RuntimeError("ハイブリッドモデル初期化失敗")

            # 単体ハイブリッドモデル訓練
            logger.info("ハイブリッドLSTM-Transformerモデル訓練開始")
            hybrid_result = self.hybrid_model.train(data)

            results = {
                "hybrid_lstm_transformer": {
                    "final_loss": hybrid_result.final_loss,
                    "best_loss": hybrid_result.best_loss,
                    "epochs_run": hybrid_result.epochs_run,
                    "training_time": hybrid_result.training_time,
                    "validation_metrics": hybrid_result.validation_metrics,
                    "convergence_achieved": hybrid_result.convergence_achieved,
                }
            }

            # アンサンブル訓練（オプション）
            if enable_ensemble and self.dl_manager:
                logger.info("従来モデルとのアンサンブル訓練開始")

                # 従来モデル追加
                from ...ml.deep_learning_models import LSTMModel, TransformerModel

                transformer_model = TransformerModel(self.config)
                lstm_model = LSTMModel(self.config)

                self.dl_manager.register_model("transformer", transformer_model)
                self.dl_manager.register_model("lstm", lstm_model)

                # アンサンブル訓練
                ensemble_results = self.dl_manager.train_ensemble(data)
                results.update(ensemble_results)

            total_training_time = time.time() - start_time

            # 性能評価
            performance_summary = self.performance_evaluator.evaluate_model_performance(
                self.hybrid_model, data, target_column
            )

            final_result = {
                "training_results": results,
                "performance_summary": performance_summary,
                "total_training_time": total_training_time,
                "models_trained": list(results.keys()),
                "target_achievement": {
                    "accuracy_target_95%": performance_summary.get("accuracy", 0)
                    >= 0.95,
                    "mae_target_0.6": performance_summary.get("mae", 1.0) <= 0.6,
                    "rmse_target_0.8": performance_summary.get("rmse", 1.0) <= 0.8,
                    "inference_time_100ms": performance_summary.get(
                        "avg_inference_time", 1000
                    )
                    <= 100,
                },
            }

            logger.info(f"次世代モデル訓練完了: {total_training_time:.2f}秒")
            logger.info(
                f"目標達成状況: {sum(final_result['target_achievement'].values())}/4"
            )

            return final_result

        except Exception as e:
            logger.error(f"次世代モデル訓練エラー: {e}")
            raise

    def predict_next_gen(
        self,
        data: pd.DataFrame,
        use_uncertainty: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """次世代予測システム"""
        start_time = time.time()

        try:
            if self.hybrid_model is None or not self.hybrid_model.is_trained:
                raise ValueError("モデルが訓練されていません")

            # ハイブリッドモデル予測
            if use_uncertainty:
                hybrid_result = self.hybrid_model.predict_with_uncertainty(data)
            else:
                hybrid_result = self.hybrid_model.predict(data)

            predictions = {
                "hybrid_lstm_transformer": {
                    "predictions": hybrid_result.predictions,
                    "confidence": hybrid_result.confidence,
                    "model_used": hybrid_result.model_used,
                    "uncertainty": (
                        hybrid_result.uncertainty.__dict__
                        if hybrid_result.uncertainty
                        else None
                    ),
                }
            }

            # アンサンブル予測（オプション）
            if use_ensemble and self.dl_manager and len(self.dl_manager.models) > 1:
                ensemble_result = self.dl_manager.predict_ensemble(data)
                predictions["ensemble"] = {
                    "predictions": ensemble_result.predictions,
                    "confidence": ensemble_result.confidence,
                    "model_weights": ensemble_result.model_weights,
                    "model_used": ensemble_result.model_used,
                }

            # アテンション分析（ハイブリッドモデル専用）
            attention_analysis = self.hybrid_model.get_attention_analysis(data)

            inference_time = time.time() - start_time

            # 性能メトリクス更新
            self.performance_evaluator.record_inference_time(inference_time * 1000)

            final_result = {
                "predictions": predictions,
                "attention_analysis": attention_analysis,
                "inference_time_ms": inference_time * 1000,
                "performance_targets": {
                    "inference_time_target_achieved": inference_time * 1000 <= 100,
                    "avg_inference_time": self.performance_evaluator.get_performance_summary().get(
                        "avg_inference_time_ms", 0
                    ),
                },
                "metadata": {
                    "timestamp": time.time(),
                    "data_points": len(data),
                    "prediction_horizon": self.config.prediction_horizon,
                    "uncertainty_estimation": use_uncertainty,
                    "ensemble_used": use_ensemble,
                },
            }

            logger.info(f"次世代予測完了: {inference_time * 1000:.2f}ms")

            return final_result

        except Exception as e:
            logger.error(f"次世代予測エラー: {e}")
            raise

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """包括的システム概要"""
        performance_summary = self.performance_evaluator.get_performance_summary()
        
        return {
            "engine_info": {
                "name": "Next-Gen AI Trading Engine",
                "version": "2.0",
                "architecture": "Hybrid LSTM-Transformer",
                "performance_targets": {
                    "accuracy": "95%+",
                    "mae": "<0.6",
                    "rmse": "<0.8",
                    "inference_time": "<100ms",
                },
            },
            "model_config": {
                "sequence_length": self.config.sequence_length,
                "prediction_horizon": self.config.prediction_horizon,
                "lstm_hidden_size": self.config.lstm_hidden_size,
                "transformer_d_model": self.config.transformer_d_model,
                "cross_attention_heads": self.config.cross_attention_heads,
            },
            "performance_history": performance_summary,
            "system_status": {
                "hybrid_model_initialized": self.hybrid_model is not None,
                "model_trained": (
                    self.hybrid_model.is_trained if self.hybrid_model else False
                ),
                "dl_manager_available": self.dl_manager is not None,
                "metrics_integration": self.metrics_integration,
            },
        }

    def calculate_advanced_technical_indicators(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, float]:
        """
        高度テクニカル指標計算（ML拡張版）

        Args:
            data: 価格データ
            symbol: 銘柄コード

        Returns:
            テクニカル指標スコア辞書
        """
        return self.technical_calculator.calculate_advanced_technical_indicators(
            data, symbol
        )