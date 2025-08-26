#!/usr/bin/env python3
"""
ハイブリッドLSTM-Transformerエンジン統合クラス
"""

import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

from ...utils.logging_config import get_context_logger
from ..deep_learning_models import BaseDeepLearningModel, DeepLearningConfig, PredictionResult, UncertaintyEstimate
from .config import HybridModelConfig
from .dataset import TimeSeriesDataset
from .pytorch_model import HybridLSTMTransformerModel
from .utils import build_numpy_hybrid_model, calculate_accuracy, numpy_hybrid_forward, train_numpy_hybrid

logger = get_context_logger(__name__)

# Issue #696対応: PyTorch可用性チェック
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    PYTORCH_AVAILABLE = True
    logger.info("PyTorch利用可能 - 完全なHybridLSTMTransformerEngineを使用")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.critical("PyTorch未インストール - NumPyフォールバック実装を使用（大幅な性能制限あり）")


class HybridLSTMTransformerEngine(BaseDeepLearningModel):
    """ハイブリッドLSTM-Transformerエンジン統合クラス"""

    def __init__(self, config: Union[HybridModelConfig, DeepLearningConfig]):
        # 設定変換
        if isinstance(config, DeepLearningConfig):
            hybrid_config = HybridModelConfig(
                sequence_length=config.sequence_length,
                prediction_horizon=config.prediction_horizon,
                lstm_hidden_size=getattr(config, 'hidden_dim', 128),
                lstm_num_layers=getattr(config, 'num_layers', 2),
                dropout_rate=config.dropout_rate,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                batch_size=config.batch_size,
                early_stopping_patience=config.early_stopping_patience,
            )
        else:
            hybrid_config = config

        super().__init__(hybrid_config)
        self.hybrid_config = hybrid_config

        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Issue #697対応: データローディング最適化設定
            self._dataloader_config = self._initialize_dataloader_config()
        else:
            self.device = "cpu"

        logger.info(f"ハイブリッドLSTM-Transformerエンジン初期化完了 (デバイス: {self.device})")

    def _initialize_dataloader_config(self) -> Dict[str, Any]:
        """
        Issue #697対応: DataLoader最適化設定の初期化

        Returns:
            DataLoader最適化設定辞書
        """
        if not PYTORCH_AVAILABLE:
            return {}

        # CPU/GPU環境に応じた最適設定
        if self.device.type == 'cuda':
            # GPU使用時の設定
            num_workers = min(4, max(1, torch.get_num_threads() // 2))
            pin_memory = True
            persistent_workers = True
        else:
            # CPU使用時の設定
            num_workers = min(2, max(1, torch.get_num_threads() // 4))
            pin_memory = False
            persistent_workers = False

        return {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers,
            'drop_last': True,
            'prefetch_factor': 2 if num_workers > 0 else 2,  # データの先読み
        }

    def build_model(self, input_shape: Tuple[int, ...]) -> Any:
        """ハイブリッドモデル構築"""
        seq_len, n_features = input_shape

        if PYTORCH_AVAILABLE:
            model = HybridLSTMTransformerModel(self.hybrid_config, n_features)
            model = model.to(self.device)

            logger.info("PyTorch ハイブリッドモデル構築完了")
            logger.info(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

            return model
        else:
            # NumPy フォールバック実装
            logger.warning("PyTorch未利用 - NumPy簡易実装を使用")
            return build_numpy_hybrid_model(input_shape, self.hybrid_config)

    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """内部訓練メソッド"""
        start_time = time.time()

        if PYTORCH_AVAILABLE and hasattr(self.model, "parameters"):
            result = self._train_pytorch_hybrid(X, y, start_time)
        else:
            result = self._train_numpy_hybrid(X, y, start_time)

        self.is_trained = True
        return result

    def _train_pytorch_hybrid(self, X: np.ndarray, y: np.ndarray, start_time: float) -> Dict[str, Any]:
        """PyTorch ハイブリッドモデル訓練 - Issue #697対応: データ読み込み最適化"""
        # Issue #697対応: 効率的なデータ前処理
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        # 訓練・検証分割
        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X_tensor[:-val_size], X_tensor[-val_size:]
        y_train, y_val = y_tensor[:-val_size], y_tensor[-val_size:]

        # Issue #697対応: 効率的データセット作成
        train_dataset = TimeSeriesDataset.create_efficient_dataset(
            X_train, y_train, device=torch.device('cpu'), use_pinned_memory=self.device.type == 'cuda'
        )
        val_dataset = TimeSeriesDataset.create_efficient_dataset(
            X_val, y_val, device=torch.device('cpu'), use_pinned_memory=self.device.type == 'cuda'
        )

        # DataLoader設定
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, **self._dataloader_config
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, **self._dataloader_config
        )

        # オプティマイザー・スケジューラー
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.hybrid_config.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hybrid_config.lr_scheduler_factor,
            patience=self.hybrid_config.lr_scheduler_patience,
        )

        criterion = nn.MSELoss()

        # 訓練ループ
        best_val_loss = float("inf")
        patience_counter = 0
        training_losses = []
        validation_losses = []

        for epoch in range(self.config.epochs):
            # 訓練フェーズ
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hybrid_config.gradient_clip_value)

                optimizer.step()
                train_loss += loss.item()

            # 検証フェーズ
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    predictions = self.model(batch_X)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # 早期停止チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"早期停止: エポック {epoch + 1}")
                    break

            # ログ出力（10エポック毎）
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"エポック {epoch + 1}/{self.config.epochs}: "
                    f"訓練損失={train_loss:.6f}, 検証損失={val_loss:.6f}"
                )

        training_time = time.time() - start_time

        # 精度計算
        with torch.no_grad():
            self.model.eval()
            train_pred = []
            val_pred = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                pred = self.model(batch_X)
                train_pred.append(pred.cpu())

            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                pred = self.model(batch_X)
                val_pred.append(pred.cpu())

            train_pred = torch.cat(train_pred, dim=0).numpy()
            val_pred = torch.cat(val_pred, dim=0).numpy()

        train_accuracy = calculate_accuracy(y_train.cpu().numpy(), train_pred)
        val_accuracy = calculate_accuracy(y_val.cpu().numpy(), val_pred)

        return {
            "final_loss": validation_losses[-1],
            "best_loss": best_val_loss,
            "epochs_run": epoch + 1,
            "validation_metrics": {
                "mse": best_val_loss,
                "mae": np.mean(np.abs(y_val.cpu().numpy() - val_pred)),
                "rmse": np.sqrt(best_val_loss),
            },
            "convergence_achieved": patience_counter < self.config.early_stopping_patience,
            "training_time": training_time,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        }

    def _train_numpy_hybrid(self, X: np.ndarray, y: np.ndarray, start_time: float) -> Dict[str, Any]:
        """NumPy ハイブリッドモデル訓練（簡易実装）"""
        result = train_numpy_hybrid(X, y, self.hybrid_config, self.model)
        result["training_time"] = time.time() - start_time

        val_size = int(len(X) * self.config.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        train_pred = numpy_hybrid_forward(X_train, self.model)
        val_pred = numpy_hybrid_forward(X_val, self.model)

        result["training_accuracy"] = calculate_accuracy(y_train, train_pred)
        result["validation_accuracy"] = calculate_accuracy(y_val, val_pred)

        return result

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """内部予測メソッド"""
        if PYTORCH_AVAILABLE and hasattr(self.model, "parameters"):
            return self._predict_pytorch_hybrid(X)
        else:
            return numpy_hybrid_forward(X, self.model)

    def _predict_pytorch_hybrid(self, X: np.ndarray) -> np.ndarray:
        """PyTorch ハイブリッドモデル予測"""
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def predict_with_uncertainty(self, data: pd.DataFrame, num_samples: int = None) -> PredictionResult:
        """不確実性付き予測"""
        num_samples = num_samples or self.hybrid_config.monte_carlo_samples

        X, _ = self.prepare_data(data)

        if PYTORCH_AVAILABLE and hasattr(self.model, "parameters"):
            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                mean_pred, std_pred = self.model.forward_with_uncertainty(X_tensor, num_samples)
                mean_pred = mean_pred.cpu().numpy()
                std_pred = std_pred.cpu().numpy()

            # 不確実性推定
            uncertainty = UncertaintyEstimate(
                mean=float(np.mean(std_pred)),
                std=float(np.std(std_pred)),
                lower_bound=mean_pred - 1.96 * std_pred,
                upper_bound=mean_pred + 1.96 * std_pred,
                epistemic=float(np.mean(std_pred)),
                aleatoric=0.1,  # 仮の値
            )

            # 信頼度計算
            confidence = 1.0 - (std_pred / (np.abs(mean_pred) + 1e-8))
            confidence = np.clip(confidence, 0.0, 1.0)

            return PredictionResult(
                predictions=mean_pred,
                confidence=confidence,
                uncertainty=uncertainty,
                model_used="HybridLSTMTransformer",
                metrics={"uncertainty_mean": uncertainty.mean},
            )
        else:
            # NumPy版は不確実性推定なし
            predictions = numpy_hybrid_forward(X, self.model)
            confidence = np.ones(len(predictions)) * 0.8

            return PredictionResult(
                predictions=predictions, confidence=confidence, model_used="HybridLSTMTransformer (NumPy)"
            )

    def get_attention_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """アテンション重み分析"""
        if not PYTORCH_AVAILABLE or not hasattr(self.model, "parameters"):
            return {"error": "PyTorchモデルが利用できません"}

        X, _ = self.prepare_data(data)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions, attention_info = self.model(X_tensor, return_attention=True)

        return {
            "lstm_contribution": attention_info["lstm_contribution"],
            "transformer_contribution": attention_info["transformer_contribution"],
            "attention_balance": (
                "LSTM優位" if attention_info["lstm_contribution"] > 0.5 else "Transformer優位"
            ),
            "predictions": predictions.cpu().numpy(),
            "analysis": "アテンション重みによる特徴量重要度分析完了",
        }