#!/usr/bin/env python3
"""
深層学習統合システム - モデル管理
Phase F: 次世代機能拡張フェーズ

深層学習モデル統合管理とアンサンブル機能
"""

import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_model import BaseDeepLearningModel
from .transformer_model import TransformerModel
from .lstm_model import LSTMModel
from .model_types import (
    ModelType, ModelConfig, DeepLearningConfig, 
    ModelTrainingResult, PredictionResult, TrainingResult
)
try:
    from ...core.optimization_strategy import OptimizationConfig
except ImportError:
    # フォールバック: 基本的なコンフィグクラスを定義
    class OptimizationConfig:
        def __init__(self, **kwargs):
            pass
try:
    from ...utils.logging_config import get_context_logger
    logger = get_context_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class DeepLearningModelManager:
    """深層学習モデル統合管理"""

    def __init__(
        self,
        dl_config: DeepLearningConfig,
        opt_config: Optional[OptimizationConfig] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,  # Issue #545対応: アンサンブル重み
    ):
        self.dl_config = dl_config
        self.opt_config = opt_config or OptimizationConfig()
        self.models = {}
        self.ensemble_weights = ensemble_weights if ensemble_weights is not None else {}  # Issue #545対応: アンサンブル重み
        self.performance_stats = {"total_predictions": 0, "total_time": 0.0}

        logger.info("深層学習モデル管理システム初期化完了")

    def create_model(
        self, model_type: ModelType, model_config: ModelConfig
    ) -> BaseDeepLearningModel:
        """モデル作成"""
        if model_type == ModelType.TRANSFORMER:
            return TransformerModel(model_config)
        elif model_type == ModelType.LSTM:
            return LSTMModel(model_config)
        elif model_type == ModelType.HYBRID_LSTM_TRANSFORMER:
            # ハイブリッドLSTM-Transformerモデル作成
            try:
                from ..hybrid_lstm_transformer import HybridLSTMTransformerEngine
                return HybridLSTMTransformerEngine(model_config)
            except ImportError:
                logger.warning("HybridLSTMTransformerEngineが見つかりません。LSTMにフォールバック")
                return LSTMModel(model_config)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

    def train_model(
        self, model_type: ModelType, data: pd.DataFrame, target_column: str = "Close"
    ) -> TrainingResult:
        """モデル訓練"""
        model_config = ModelConfig(model_type=model_type)
        model = self.create_model(model_type, model_config)

        # データ準備
        X, y = model.prepare_data(data, target_column)

        # モデル構築
        input_shape = X.shape[1:]
        model.model = model.build_model(input_shape)

        # 訓練実行
        training_result = model.train(data)

        # モデル保存
        self.models[model_type.value] = model

        logger.info(
            f"{model_type.value}モデル訓練完了: 精度{training_result.validation_metrics.get('accuracy', 0.0):.3f}"
        )

        # TrainingResultに変換
        return TrainingResult(
            model_type=model_type,
            training_loss=training_result.final_loss,
            validation_loss=training_result.best_loss,
            training_accuracy=training_result.validation_metrics.get('accuracy', 0.0),
            validation_accuracy=training_result.validation_metrics.get('val_accuracy', 0.0),
            training_time=training_result.training_time,
            epochs_trained=training_result.epochs_run,
            model_parameters=0,  # 実際の実装では適切に計算
            convergence_achieved=training_result.convergence_achieved,
        )

    def predict_ensemble(self, data: pd.DataFrame) -> PredictionResult:
        """アンサンブル予測"""
        if not self.models:
            raise ValueError("訓練済みモデルがありません")

        predictions = []
        confidences = []
        model_names = []

        # 各モデルで予測
        for model_name, model in self.models.items():
            try:
                X, _ = model.prepare_data(data)
                result = model.predict(data)

                predictions.append(result.predictions)
                confidences.append(result.confidence)
                model_names.append(model_name)
            except Exception as e:
                logger.warning(f"モデル {model_name} の予測に失敗: {e}")
                continue

        # アンサンブル統合
        if len(predictions) == 0:
            raise ValueError("予測可能なモデルがありません")
        elif len(predictions) == 1:
            ensemble_pred = predictions[0]
            ensemble_conf = confidences[0]
            ensemble_model_weights = {model_names[0]: 1.0}
        else:
            # Issue #545対応: アンサンブル重みの動的管理
            calculated_weights = np.ones(len(predictions))  # デフォルトは均等重み

            if self.ensemble_weights:  # 外部から重みが指定されている場合
                for i, model_name in enumerate(model_names):
                    if model_name in self.ensemble_weights:
                        calculated_weights[i] = self.ensemble_weights[model_name]
                    else:
                        logger.warning(f"指定されたモデル '{model_name}' のアンサンブル重みが見つかりません。均等重みを使用します。")

            # 重みを正規化
            if np.sum(calculated_weights) > 0:
                calculated_weights = calculated_weights / np.sum(calculated_weights)
            else:
                logger.warning("アンサンブル重みの合計がゼロです。均等重みにフォールバックします。")
                calculated_weights = np.ones(len(predictions)) / len(predictions)

            ensemble_pred = np.average(predictions, axis=0, weights=calculated_weights)
            ensemble_conf = np.average(confidences, axis=0, weights=calculated_weights)
            ensemble_model_weights = dict(zip(model_names, calculated_weights))

        return PredictionResult(
            predictions=ensemble_pred,
            confidence=ensemble_conf,
            model_used=f"Ensemble({'+'.join(model_names)})",
            model_weights=ensemble_model_weights,  # Issue #545対応: 使用された重みを記録
        )

    def register_model(self, name: str, model: BaseDeepLearningModel):
        """モデル登録"""
        self.models[name] = model
        logger.info(f"モデル登録完了: {name}")

    def train_ensemble(self, data: pd.DataFrame, parallel: bool = True) -> Dict[str, ModelTrainingResult]:
        """
        アンサンブル訓練 - Issue #708対応並列化版

        Args:
            data: 訓練データ
            parallel: 並列訓練を有効にするか

        Returns:
            Dict[str, ModelTrainingResult]: 各モデルの訓練結果
        """
        if parallel and len(self.models) > 1:
            return self._parallel_train_ensemble(data)
        else:
            return self._sequential_train_ensemble(data)

    def _parallel_train_ensemble(self, data: pd.DataFrame) -> Dict[str, ModelTrainingResult]:
        """並列アンサンブル訓練実装"""

        def train_single_deep_model(args):
            """単一深層学習モデルの訓練関数"""
            model_name, model, data_copy, dl_config = args
            try:
                logger.info(f"深層学習モデル並列訓練開始: {model_name}")
                start_time = time.time()

                # ModelConfigからDeepLearningConfigに変換
                model_config = ModelConfig(
                    sequence_length=dl_config.sequence_length,
                    prediction_horizon=dl_config.prediction_horizon,
                    hidden_size=dl_config.hidden_dim,
                    num_layers=dl_config.num_layers,
                    dropout_rate=dl_config.dropout_rate,
                    learning_rate=dl_config.learning_rate,
                    epochs=dl_config.epochs,
                    batch_size=dl_config.batch_size,
                    early_stopping_patience=dl_config.early_stopping_patience,
                )
                model.config = model_config

                # データ準備と訓練
                result = model.train(data_copy)
                training_time = time.time() - start_time

                # ModelTrainingResult形式に変換
                training_result = ModelTrainingResult(
                    final_loss=result.final_loss,
                    best_loss=result.best_loss,
                    epochs_run=result.epochs_run,
                    training_time=training_time,
                    validation_metrics={"mse": result.final_loss},
                    convergence_achieved=True,
                )

                logger.info(f"深層学習モデル並列訓練完了: {model_name} ({training_time:.2f}秒)")
                return model_name, training_result

            except Exception as e:
                logger.error(f"深層学習モデル並列訓練エラー {model_name}: {e}")
                return model_name, ModelTrainingResult(
                    final_loss=float('inf'),
                    best_loss=float('inf'),
                    epochs_run=0,
                    training_time=0.0,
                    validation_metrics={"error": str(e)},
                    convergence_achieved=False,
                )

        results = {}

        # 並列処理設定 (深層学習はCPU集約的でないため、スレッドプールを使用)
        n_jobs = min(len(self.models), mp.cpu_count())
        logger.info(f"アンサンブル並列訓練開始: {len(self.models)}モデルを{n_jobs}スレッドで実行")

        # 訓練引数準備
        training_args = [
            (model_name, model, data.copy(), self.dl_config)
            for model_name, model in self.models.items()
        ]

        try:
            # ThreadPoolExecutorを使用 (深層学習モデルはI/O待機が多い)
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_model = {
                    executor.submit(train_single_deep_model, args): args[0]
                    for args in training_args
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result_name, result = future.result(timeout=7200)  # 2時間のタイムアウト
                        results[result_name] = result
                    except Exception as e:
                        logger.error(f"深層学習モデル並列訓練例外 {model_name}: {e}")
                        results[model_name] = ModelTrainingResult(
                            final_loss=float('inf'),
                            best_loss=float('inf'),
                            epochs_run=0,
                            training_time=0.0,
                            validation_metrics={"timeout_error": str(e)},
                            convergence_achieved=False,
                        )

        except Exception as e:
            logger.warning(f"並列アンサンブル訓練フォールバック: {e}")
            # フォールバック: 順次実行
            return self._sequential_train_ensemble(data)

        return results

    def _sequential_train_ensemble(self, data: pd.DataFrame) -> Dict[str, ModelTrainingResult]:
        """順次アンサンブル訓練実装（フォールバック）"""
        results = {}

        for name, model in self.models.items():
            logger.info(f"モデル訓練開始: {name}")
            start_time = time.time()

            # ModelConfigからDeepLearningConfigに変換
            model_config = ModelConfig(
                sequence_length=self.dl_config.sequence_length,
                prediction_horizon=self.dl_config.prediction_horizon,
                hidden_size=self.dl_config.hidden_dim,
                num_layers=self.dl_config.num_layers,
                dropout_rate=self.dl_config.dropout_rate,
                learning_rate=self.dl_config.learning_rate,
                epochs=self.dl_config.epochs,
                batch_size=self.dl_config.batch_size,
                early_stopping_patience=self.dl_config.early_stopping_patience,
            )
            model.config = model_config

            # データ準備と訓練
            result = model.train(data)
            training_time = time.time() - start_time

            # ModelTrainingResult形式に変換
            training_result = ModelTrainingResult(
                final_loss=result.final_loss,
                best_loss=result.best_loss,
                epochs_run=result.epochs_run,
                training_time=training_time,
                validation_metrics={"mse": result.final_loss},
                convergence_achieved=True,
            )

            results[name] = training_result

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要"""
        avg_time = self.performance_stats["total_time"] / max(
            self.performance_stats["total_predictions"], 1
        )

        return {
            "trained_models": len(self.models),
            "available_models": list(self.models.keys()),
            "ensemble_ready": len(self.models) > 1,
            "total_predictions": self.performance_stats["total_predictions"],
            "average_prediction_time": avg_time,
        }