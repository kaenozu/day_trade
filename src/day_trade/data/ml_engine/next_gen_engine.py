#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine Module
次世代AI取引エンジンモジュール

このモジュールは最新のハイブリッドアーキテクチャによる
高精度市場予測システムを提供します。

Classes:
    NextGenAITradingEngine: 次世代AI取引エンジン
"""

import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
# ML関連のインポートをオプション化
try:
    from ...ml.hybrid_lstm_transformer import (
        HybridModelConfig,
        create_hybrid_model,
    )
    from ...ml.deep_learning_models import (
        DeepLearningConfig,
        DeepLearningModelManager,
    )
except ImportError:
    # フォールバック用のダミークラス
    class HybridModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        sequence_length = 1000
        prediction_horizon = 1
        lstm_hidden_size = 256
        lstm_num_layers = 2
        dropout_rate = 0.2
        learning_rate = 0.001
        epochs = 100
        batch_size = 32
        transformer_d_model = 512
        cross_attention_heads = 8
    
    def create_hybrid_model(config):
        raise NotImplementedError("ハイブリッドモデルは利用できません")
    
    class DeepLearningConfig:
        def __init__(self, **kwargs):
            pass
    
    class DeepLearningModelManager:
        def __init__(self, config):
            pass
from .technical_indicators import calculate_advanced_technical_indicators

logger = get_context_logger(__name__)


class NextGenAITradingEngine:
    """
    Next-Gen AI Trading Engine - ハイブリッドLSTM-Transformer統合エンジン

    最新のハイブリッドアーキテクチャによる高精度市場予測システム
    目標性能: 95%+ 予測精度, <100ms推論時間, MAE<0.6, RMSE<0.8
    
    Attributes:
        config: ハイブリッドモデル設定
        hybrid_model: ハイブリッドモデル
        dl_manager: 深層学習モデル管理器
        performance_metrics: パフォーマンスメトリクス
        legacy_engine: 既存システムとの互換性維持
        metrics_integration: メトリクス統合フラグ
    """

    def __init__(self, config: Optional[HybridModelConfig] = None):
        """
        初期化
        
        Args:
            config: ハイブリッドモデル設定
        """
        self.config = config or HybridModelConfig()
        self.hybrid_model = None
        self.dl_manager = None
        self.performance_metrics = {
            "accuracy_history": [],
            "inference_times": [],
            "mae_history": [],
            "rmse_history": [],
        }

        # 既存システムとの互換性維持
        self.legacy_engine = None

        # メトリクス統合（監視システム対応）
        self.metrics_integration = True

        logger.info("Next-Gen AI Trading Engine 初期化完了")

    def initialize_hybrid_model(self) -> bool:
        """
        ハイブリッドモデル初期化
        
        Returns:
            bool: 初期化成功フラグ
        """
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
        """
        次世代モデル訓練
        
        Args:
            data: 訓練データ
            target_column: ターゲット列名
            enable_ensemble: アンサンブル学習有効フラグ
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
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
                results.update(self._train_ensemble_models(data))

            total_training_time = time.time() - start_time

            # 性能評価
            performance_summary = self._evaluate_model_performance(data, target_column)

            final_result = {
                "training_results": results,
                "performance_summary": performance_summary,
                "total_training_time": total_training_time,
                "models_trained": list(results.keys()),
                "target_achievement": self._check_target_achievement(performance_summary),
            }

            logger.info(f"次世代モデル訓練完了: {total_training_time:.2f}秒")
            logger.info(
                f"目標達成状況: {sum(final_result['target_achievement'].values())}/4"
            )

            return final_result

        except Exception as e:
            logger.error(f"次世代モデル訓練エラー: {e}")
            raise

    def _train_ensemble_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """アンサンブルモデル訓練"""
        logger.info("従来モデルとのアンサンブル訓練開始")

        try:
            # 従来モデル追加
            from ...ml.deep_learning_models import LSTMModel, TransformerModel

            transformer_model = TransformerModel(self.config)
            lstm_model = LSTMModel(self.config)

            self.dl_manager.register_model("transformer", transformer_model)
            self.dl_manager.register_model("lstm", lstm_model)

            # アンサンブル訓練
            ensemble_results = self.dl_manager.train_ensemble(data)
            return ensemble_results

        except Exception as e:
            logger.warning(f"アンサンブル訓練エラー: {e}")
            return {}

    def _check_target_achievement(self, performance_summary: Dict) -> Dict[str, bool]:
        """目標達成状況チェック"""
        return {
            "accuracy_target_95%": performance_summary.get("accuracy", 0) >= 0.95,
            "mae_target_0.6": performance_summary.get("mae", 1.0) <= 0.6,
            "rmse_target_0.8": performance_summary.get("rmse", 1.0) <= 0.8,
            "inference_time_100ms": performance_summary.get(
                "avg_inference_time", 1000
            ) <= 100,
        }

    def predict_next_gen(
        self,
        data: pd.DataFrame,
        use_uncertainty: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """
        次世代予測システム
        
        Args:
            data: 予測用データ
            use_uncertainty: 不確実性推定使用フラグ
            use_ensemble: アンサンブル予測使用フラグ
            
        Returns:
            Dict[str, Any]: 予測結果
        """
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
            self.performance_metrics["inference_times"].append(
                inference_time * 1000
            )  # ms変換

            final_result = {
                "predictions": predictions,
                "attention_analysis": attention_analysis,
                "inference_time_ms": inference_time * 1000,
                "performance_targets": {
                    "inference_time_target_achieved": inference_time * 1000 <= 100,
                    "avg_inference_time": (
                        np.mean(self.performance_metrics["inference_times"])
                        if self.performance_metrics["inference_times"]
                        else 0
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

    def _evaluate_model_performance(
        self, data: pd.DataFrame, target_column: str
    ) -> Dict[str, float]:
        """
        モデル性能評価
        
        Args:
            data: 評価用データ
            target_column: ターゲット列名
            
        Returns:
            Dict[str, float]: 性能指標
        """
        try:
            # テストデータで予測実行
            test_data = data.tail(min(100, len(data) // 4))  # 最後の25%をテスト用

            prediction_result = self.hybrid_model.predict(test_data)
            predictions = prediction_result.predictions

            # 実際の値取得
            X, y_true = self.hybrid_model.prepare_data(test_data, target_column)

            if len(y_true) > 0 and len(predictions) > 0:
                # 長さ調整
                min_len = min(len(y_true), len(predictions))
                y_true = y_true[:min_len]
                predictions = predictions[:min_len]

                # メトリクス計算
                performance_metrics = self._calculate_performance_metrics(
                    y_true, predictions
                )

                # 推論時間測定
                avg_inference_time = self._measure_inference_time(test_data)

                # メトリクス履歴更新
                self._update_performance_history(performance_metrics)

                performance_metrics["avg_inference_time"] = avg_inference_time
                performance_metrics["test_samples"] = min_len

                return performance_metrics
            else:
                logger.warning("性能評価用データが不足")
                return self._get_default_performance_metrics()

        except Exception as e:
            logger.error(f"性能評価エラー: {e}")
            return {
                **self._get_default_performance_metrics(),
                "error": str(e),
            }

    def _calculate_performance_metrics(self, y_true, predictions):
        """パフォーマンスメトリクス計算"""
        mae = np.mean(np.abs(y_true - predictions))
        rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
        mape = (
            np.mean(
                np.abs(
                    (y_true - predictions) / np.where(y_true != 0, y_true, 1e-8)
                )
            ) * 100
        )
        accuracy = max(0, 100 - mape) / 100

        return {
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }

    def _measure_inference_time(self, test_data, n_iterations=10):
        """推論時間測定"""
        inference_times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = self.hybrid_model.predict(test_data.tail(10))
            inference_times.append((time.time() - start) * 1000)  # ms変換
        return np.mean(inference_times)

    def _update_performance_history(self, metrics):
        """パフォーマンス履歴更新"""
        self.performance_metrics["accuracy_history"].append(metrics["accuracy"])
        self.performance_metrics["mae_history"].append(metrics["mae"])
        self.performance_metrics["rmse_history"].append(metrics["rmse"])

    def _get_default_performance_metrics(self):
        """デフォルト性能指標"""
        return {
            "accuracy": 0.0,
            "mae": 1.0,
            "rmse": 1.0,
            "mape": 100.0,
            "avg_inference_time": 1000.0,
            "test_samples": 0,
        }

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """
        包括的システム概要
        
        Returns:
            Dict[str, Any]: システム概要
        """
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
            "performance_history": self._get_performance_history_summary(),
            "system_status": self._get_system_status(),
        }

    def _get_performance_history_summary(self):
        """パフォーマンス履歴サマリー"""
        return {
            "total_predictions": len(self.performance_metrics["inference_times"]),
            "avg_accuracy": (
                np.mean(self.performance_metrics["accuracy_history"])
                if self.performance_metrics["accuracy_history"]
                else 0
            ),
            "avg_mae": (
                np.mean(self.performance_metrics["mae_history"])
                if self.performance_metrics["mae_history"]
                else 0
            ),
            "avg_rmse": (
                np.mean(self.performance_metrics["rmse_history"])
                if self.performance_metrics["rmse_history"]
                else 0
            ),
            "avg_inference_time_ms": (
                np.mean(self.performance_metrics["inference_times"])
                if self.performance_metrics["inference_times"]
                else 0
            ),
        }

    def _get_system_status(self):
        """システム状態"""
        return {
            "hybrid_model_initialized": self.hybrid_model is not None,
            "model_trained": (
                self.hybrid_model.is_trained if self.hybrid_model else False
            ),
            "dl_manager_available": self.dl_manager is not None,
            "metrics_integration": self.metrics_integration,
        }

    def calculate_advanced_technical_indicators(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> Dict[str, float]:
        """
        高度テクニカル指標計算のラッパー
        
        Args:
            data: 価格データ
            symbol: 銘柄コード
            
        Returns:
            Dict[str, float]: テクニカル指標スコア辞書
        """
        return calculate_advanced_technical_indicators(data, symbol)


def create_next_gen_engine(config: Optional[Dict] = None) -> NextGenAITradingEngine:
    """
    次世代AIエンジンファクトリ関数
    
    Args:
        config: 設定辞書
        
    Returns:
        NextGenAITradingEngine: エンジンインスタンス
    """
    if config:
        hybrid_config = HybridModelConfig(**config)
    else:
        hybrid_config = HybridModelConfig()

    return NextGenAITradingEngine(hybrid_config)