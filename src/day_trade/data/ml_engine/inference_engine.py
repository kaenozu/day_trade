#!/usr/bin/env python3
"""
Inference Engine Module
推論エンジンモジュール

このモジュールは機械学習モデルの推論機能を
提供するAdvancedMLEngineクラスを定義します。

Classes:
    AdvancedMLEngine: 高度機械学習エンジン
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np

from ...utils.logging_config import get_context_logger
# MacroEconomicFeaturesのインポートをオプション化
try:
    from ..macro_economic_features import MacroEconomicFeatures
except ImportError:
    # フォールバック用のダミークラス
    class MacroEconomicFeatures:
        def add_macro_features(self, data, symbol):
            return data
# OptimizationConfigのインポートをオプション化
try:
    from ...core.optimization_strategy import OptimizationConfig
except ImportError:
    # フォールバック用のダミークラス
    class OptimizationConfig:
        def __init__(self, **kwargs):
            pass
from .config import ModelConfig, PredictionResult
from .data_processor import DataProcessor
from .models import LSTMTransformerHybrid, PYTORCH_AVAILABLE

# PyTorch遅延インポート
if PYTORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

# MLflow遅延インポート
MLFLOW_AVAILABLE = False
try:
    import importlib.util
    if importlib.util.find_spec("mlflow") is not None:
        import mlflow
        MLFLOW_AVAILABLE = True
except ImportError:
    pass

logger = get_context_logger(__name__)


class AdvancedMLEngine:
    """
    Advanced ML Engine - 次世代AI予測システム
    
    LSTM-Transformerハイブリッドモデルを使用した
    高精度な市場予測システム。
    
    Attributes:
        config: モデル設定
        optimization_config: 最適化設定
        device: 実行デバイス
        model: 訓練済みモデル
        model_metadata: モデルメタデータ
        data_processor: データ処理器
        macro_features: マクロ経済特徴量エンジン
        performance_history: パフォーマンス履歴
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """
        初期化
        
        Args:
            config: モデル設定
            optimization_config: 最適化設定
        """
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
        self._setup_device()

        # モデル初期化
        self.model = None
        self.model_metadata = {
            "version": "1.0.0",
            "created_at": time.time(),
            "training_data_hash": None,
            "performance": {},
        }

        # データ前処理パイプライン
        self.data_processor = DataProcessor(self.config)

        # マクロ経済特徴量エンジン
        self.macro_features = MacroEconomicFeatures()

        # パフォーマンス監視
        self.performance_history = []

        logger.info("Advanced ML Engine 初期化完了")

    def _setup_device(self):
        """デバイス設定"""
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

    def prepare_data(self, market_data, target_column="終値", feature_columns=None):
        """
        データ前処理のラッパー
        
        Args:
            market_data: 市場データ
            target_column: ターゲット列名
            feature_columns: 特徴量列名リスト
            
        Returns:
            Tuple: (特徴量配列, ターゲット配列)
        """
        return self.data_processor.prepare_data(
            market_data, target_column, feature_columns
        )

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        モデル訓練
        
        Args:
            X_train: 訓練データ特徴量
            y_train: 訓練データターゲット
            X_val: 検証データ特徴量
            y_val: 検証データターゲット
            save_path: モデル保存パス
            
        Returns:
            Dict[str, Any]: 訓練結果
        """
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10
        )
        criterion = nn.MSELoss()

        # データローダー準備
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        # 学習履歴
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        # MLflow追跡
        if MLFLOW_AVAILABLE:
            self._start_mlflow_run()

        # 訓練ループ
        training_result = self._training_loop(
            train_loader, val_loader, optimizer, scheduler, criterion,
            train_losses, val_losses, best_val_loss, patience_counter, save_path
        )

        # MLflow終了
        if MLFLOW_AVAILABLE:
            self._end_mlflow_run()

        # 結果処理
        self.model_metadata["performance"] = training_result
        self.model_metadata["training_data_hash"] = hash(str(X_train.tobytes()))

        logger.info(f"モデル訓練完了: {training_result}")
        return training_result

    def _create_dataloader(self, X, y, shuffle=False):
        """データローダー作成"""
        dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(y).to(self.device),
        )
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle
        )

    def _start_mlflow_run(self):
        """MLflow実行開始"""
        mlflow.start_run()
        mlflow.log_params({
            "lstm_hidden_size": self.config.lstm_hidden_size,
            "transformer_d_model": self.config.transformer_d_model,
            "sequence_length": self.config.sequence_length,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
        })

    def _training_loop(
        self, train_loader, val_loader, optimizer, scheduler, criterion,
        train_losses, val_losses, best_val_loss, patience_counter, save_path
    ):
        """訓練ループ"""
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            # 訓練フェーズ
            epoch_train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                predictions, confidence = self.model(data)
                loss = criterion(predictions, target.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 検証フェーズ
            avg_val_loss = self._validation_phase(val_loader, criterion)
            if avg_val_loss is not None:
                val_losses.append(avg_val_loss)
                
                # 早期終了チェック
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if save_path:
                        self._save_model(save_path + "_best.pth")
                else:
                    patience_counter += 1

                scheduler.step(avg_val_loss)

            # ログ出力
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, "
                    f"Val Loss={avg_val_loss:.6f}"
                )

            # MLflow記録
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss or 0
                }, step=epoch)

            # 早期終了
            if patience_counter >= self.config.patience:
                logger.info(f"早期終了: Epoch {epoch + 1}")
                break

        # 最終モデル保存
        if save_path:
            self._save_model(save_path + "_final.pth")

        return {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(train_losses),
            "converged": patience_counter < self.config.patience,
        }

    def _validation_phase(self, val_loader, criterion):
        """検証フェーズ"""
        if not val_loader:
            return None
            
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                predictions, _ = self.model(data)
                val_loss += criterion(predictions, target.squeeze()).item()
        
        self.model.train()
        return val_loss / len(val_loader)

    def _end_mlflow_run(self):
        """MLflow実行終了"""
        mlflow.pytorch.log_model(self.model, "model")
        mlflow.end_run()

    def predict(
        self, X: np.ndarray, return_confidence: bool = True
    ) -> PredictionResult:
        """
        予測実行
        
        Args:
            X: 入力データ
            return_confidence: 信頼度を返すかどうか
            
        Returns:
            PredictionResult: 予測結果
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")

        start_time = time.time()

        # 推論モード
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions, confidence = self.model(X_tensor)

            predictions = predictions.cpu().numpy()
            confidence_scores = (
                confidence.cpu().numpy() if return_confidence else None
            )

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
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "metadata": self.model_metadata,
                "scaler": self.data_processor.scaler,
            }, path)
            logger.info(f"モデル保存完了: {path}")

    def load_model(self, path: str):
        """
        モデル読み込み
        
        Args:
            path: モデルファイルパス
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch が必要です")

        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint["config"]
        self.model_metadata = checkpoint["metadata"]
        if "scaler" in checkpoint:
            self.data_processor.scaler = checkpoint["scaler"]

        self.model = LSTMTransformerHybrid(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"モデル読み込み完了: {path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        モデル概要取得
        
        Returns:
            Dict[str, Any]: モデル概要
        """
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
                if self.performance_history else None
            ),
        }


def create_advanced_ml_engine(config_dict: Optional[Dict] = None) -> AdvancedMLEngine:
    """
    Advanced ML Engine インスタンス作成
    
    Args:
        config_dict: 設定辞書
        
    Returns:
        AdvancedMLEngine: エンジンインスタンス
    """
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()

    return AdvancedMLEngine(config)


# PyTorchが利用できない場合のフォールバック
if not PYTORCH_AVAILABLE:
    logger.warning("PyTorch 未インストール - Advanced ML Engine は制限モードで動作")

    class AdvancedMLEngine:
        """フォールバック版 - 基本機能のみ"""

        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch未インストールのため、基本機能のみ利用可能")
            self.model = None
            self.performance_history = []

        def prepare_data(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def train_model(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def predict(self, *args, **kwargs):
            raise NotImplementedError("PyTorchが必要です")

        def get_model_summary(self):
            return {"status": "PyTorch未インストール", "features": "制限モード"}