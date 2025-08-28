#!/usr/bin/env python3
"""
Advanced ML Engine Core Module

メインのML予測エンジンクラスのコア機能
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# PyTorch遅延インポート
PYTORCH_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        PYTORCH_AVAILABLE = True
except ImportError:
    pass

# MLflow遅延インポート
MLFLOW_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("mlflow") is not None:
        import mlflow

        MLFLOW_AVAILABLE = True
except ImportError:
    pass

from ...core.optimization_strategy import OptimizationConfig
from ...utils.logging_config import get_context_logger
from ..macro_economic_features import MacroEconomicFeatures
from .config import ModelConfig, PredictionResult
from .feature_engineering import FeatureEngineer
from .models import LSTMTransformerHybrid

logger = get_context_logger(__name__)


class AdvancedMLEngineCore:
    """Advanced ML Engine - 次世代AI予測システムのコア機能"""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        self.ci_mode = os.getenv("CI", "false").lower() == "true"

        self.config = config or ModelConfig()
        self.optimization_config = optimization_config or OptimizationConfig()

        # CI環境では軽量設定
        if self.ci_mode:
            self.config.sequence_length = min(self.config.sequence_length, 100)
            self.config.num_features = min(self.config.num_features, 10)
            self.config.batch_size = min(self.config.batch_size, 8)
            self.config.use_gpu = False
            logger.info("CI軽量モード: パラメータを削減")

        # デバイス設定
        if (
            PYTORCH_AVAILABLE
            and self.config.use_gpu
            and torch.cuda.is_available()
            and not self.ci_mode
        ):
            self.device = torch.device("cuda")
            logger.info(f"GPU使用: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            logger.info("CPU使用")

        # モデル初期化
        self.model = None
        self.model_metadata = {
            "version": "1.0.0",
            "created_at": time.time(),
            "training_data_hash": None,
            "performance": {},
        }

        # 特徴量エンジニアリング
        self.feature_engineer = FeatureEngineer()

        # マクロ経済特徴量エンジン
        self.macro_features = MacroEconomicFeatures()

        # パフォーマンス監視
        self.performance_history = []

        logger.info("Advanced ML Engine Core 初期化完了")

    def prepare_data(
        self,
        market_data: pd.DataFrame,
        target_column: str = "終値",
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """高度データ前処理パイプライン"""

        logger.info(f"データ前処理開始: {len(market_data)} レコード")

        if feature_columns is None:
            # 数値カラムを自動選択
            numeric_columns = market_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            feature_columns = [col for col in numeric_columns if col != target_column]

        # 基本特徴量エンジニアリング
        processed_data = self.feature_engineer.engineer_features(
            market_data, feature_columns
        )

        # マクロ経済特徴量の追加
        try:
            symbol = getattr(market_data, "symbol", "UNKNOWN")
            processed_data = self.macro_features.add_macro_features(
                processed_data, symbol
            )
            logger.info(f"マクロ経済特徴量統合完了: {symbol}")
        except Exception as e:
            logger.warning(f"マクロ経済特徴量統合スキップ: {e}")

        # 特徴量選択（相関による）
        if processed_data.shape[1] > self.config.num_features:
            correlation_with_target = (
                processed_data.corr()[target_column].abs().sort_values(ascending=False)
            )
            selected_features = correlation_with_target.head(
                self.config.num_features
            ).index.tolist()
            processed_data = processed_data[selected_features]

        # 系列データ作成
        sequences, targets = self.feature_engineer.create_sequences(
            processed_data,
            target_column,
            self.config.sequence_length,
            self.config.prediction_horizon,
        )

        logger.info(f"前処理完了: {sequences.shape} -> {targets.shape}")
        return sequences, targets

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """モデル訓練"""

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch が必要です")

        logger.info(f"モデル訓練開始: {X_train.shape} -> {y_train.shape}")

        # 設定更新
        self.config.num_features = X_train.shape[-1]
        self.config.prediction_horizon = (
            y_train.shape[-1] if len(y_train.shape) > 1 else 1
        )

        # モデル初期化
        self.model = LSTMTransformerHybrid(self.config).to(self.device)

        # オプティマイザーとロス関数
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        criterion = nn.MSELoss()

        # データローダー
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val).to(self.device),
                torch.FloatTensor(y_val).to(self.device),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # 学習履歴
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        # MLflow追跡
        if MLFLOW_AVAILABLE:
            mlflow.start_run()
            mlflow.log_params(
                {
                    "lstm_hidden_size": self.config.lstm_hidden_size,
                    "transformer_d_model": self.config.transformer_d_model,
                    "sequence_length": self.config.sequence_length,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                }
            )

        # 訓練ループ
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_train_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()

                # 順伝播
                predictions, confidence = self.model(data)

                # 損失計算（信頼度重み付け）
                loss = criterion(predictions, target.squeeze())

                # 逆伝播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()

            # 平均訓練損失
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 検証
            avg_val_loss = 0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for data, target in val_loader:
                        predictions, _ = self.model(data)
                        val_loss += criterion(predictions, target.squeeze()).item()
                    avg_val_loss = val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

                # 早期終了
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # ベストモデル保存
                    if save_path:
                        self._save_model(save_path + "_best.pth")
                else:
                    patience_counter += 1

                # 学習率調整
                scheduler.step(avg_val_loss)
                self.model.train()

            # ログ出力
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, "
                    f"Val Loss={avg_val_loss:.6f}"
                )

            # MLflow記録
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics(
                    {"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch
                )

            # 早期終了
            if patience_counter >= self.config.patience:
                logger.info(f"早期終了: Epoch {epoch + 1}")
                break

        # 最終モデル保存
        if save_path:
            self._save_model(save_path + "_final.pth")

        # MLflow終了
        if MLFLOW_AVAILABLE:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

        # 訓練結果
        training_result = {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "converged": patience_counter < self.config.patience,
        }

        self.model_metadata["performance"] = training_result
        self.model_metadata["training_data_hash"] = hash(str(X_train.tobytes()))

        logger.info(f"モデル訓練完了: {training_result}")
        return training_result

    def predict(
        self, X: np.ndarray, return_confidence: bool = True
    ) -> PredictionResult:
        """予測実行"""

        if self.model is None:
            raise ValueError("モデルが訓練されていません")

        start_time = time.time()

        # 推論モード
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions, confidence = self.model(X_tensor)

            predictions = predictions.cpu().numpy()
            confidence_scores = confidence.cpu().numpy() if return_confidence else None

        inference_time = time.time() - start_time

        # パフォーマンス記録
        performance_metrics = {
            "inference_time": inference_time,
            "throughput": len(X) / inference_time,
            "model_version": self.model_metadata["version"],
        }

        result = PredictionResult(
            predictions=predictions,
            confidence=confidence_scores,
            probabilities=None,
            model_version=self.model_metadata["version"],
            timestamp=time.time(),
            features_used=list(range(X.shape[-1])),
            performance_metrics=performance_metrics,
        )

        self.performance_history.append(performance_metrics)
        logger.info(
            f"予測完了: {X.shape} -> {predictions.shape}, "
            f"処理時間: {inference_time:.3f}秒"
        )

        return result

    def _save_model(self, path: str):
        """モデル保存"""
        if self.model is not None:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "metadata": self.model_metadata,
                    "scaler": self.feature_engineer.scaler,
                },
                path,
            )
            logger.info(f"モデル保存完了: {path}")

    def load_model(self, path: str):
        """モデル読み込み"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch が必要です")

        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint["config"]
        self.model_metadata = checkpoint["metadata"]
        self.feature_engineer.scaler = checkpoint.get("scaler")

        self.model = LSTMTransformerHybrid(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"モデル読み込み完了: {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """モデル概要取得"""
        if self.model is None:
            return {"status": "モデル未初期化"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "model_version": self.model_metadata["version"],
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "config": {
                "lstm_hidden_size": self.config.lstm_hidden_size,
                "transformer_d_model": self.config.transformer_d_model,
                "sequence_length": self.config.sequence_length,
                "num_features": self.config.num_features,
            },
            "performance": self.model_metadata.get("performance", {}),
            "average_inference_time": (
                np.mean([p["inference_time"] for p in self.performance_history])
                if self.performance_history
                else None
            ),
        }