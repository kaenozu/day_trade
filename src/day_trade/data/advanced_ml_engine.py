#!/usr/bin/env python3
"""
Next-Gen AI Trading Engine - Advanced ML Engine
高度なLSTM-Transformer ハイブリッドモデル実装

リアルタイム市場データから高精度予測を行う次世代AI駆動取引エンジン

Version: 2.0 - ハイブリッドLSTM-Transformer統合
Performance Target: 95%+ 予測精度, <100ms推論時間
"""

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 深層学習フレームワーク（遅延インポート - メモリ効率化）
PYTORCH_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset

        PYTORCH_AVAILABLE = True
except ImportError:
    pass

# Transformers（遅延インポート）
TRANSFORMERS_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("transformers") is not None:
        TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# オプショナルライブラリ（遅延インポート）
MLFLOW_AVAILABLE = False
try:
    import importlib.util

    if importlib.util.find_spec("mlflow") is not None:
        MLFLOW_AVAILABLE = True
except ImportError:
    pass

# プロジェクト内モジュール
from ..core.optimization_strategy import OptimizationConfig
from ..ml.deep_learning_models import (
    DeepLearningConfig,
    DeepLearningModelManager,
)
from ..ml.hybrid_lstm_transformer import (
    HybridModelConfig,
    create_hybrid_model,
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelConfig:
    """ML モデル設定"""

    # LSTM設定
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2

    # Transformer設定
    transformer_d_model: int = 512
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dim_feedforward: int = 2048
    transformer_dropout: float = 0.1

    # 入力・出力設定
    sequence_length: int = 1000
    num_features: int = 50
    prediction_horizon: int = 1

    # 学習設定
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 20

    # GPU設定
    use_gpu: bool = True
    mixed_precision: bool = True


@dataclass
class PredictionResult:
    """予測結果データクラス"""

    predictions: np.ndarray
    confidence: np.ndarray
    probabilities: Optional[np.ndarray]
    model_version: str
    timestamp: float
    features_used: List[str]
    performance_metrics: Dict[str, float]


if PYTORCH_AVAILABLE:

    class LSTMTransformerHybrid(nn.Module):
        """ハイブリッド LSTM-Transformer 予測モデル"""

        def __init__(self, config: ModelConfig):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch が必要です")

            super().__init__()
            self.config = config

            # LSTM分岐
            self.lstm = nn.LSTM(
                input_size=config.num_features,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
                batch_first=True,
                bidirectional=True,
            )

            # Transformer分岐
            self.positional_encoding = PositionalEncoding(config.transformer_d_model)
            self.input_projection = nn.Linear(
                config.num_features, config.transformer_d_model
            )

            transformer_layer = nn.TransformerEncoderLayer(
                d_model=config.transformer_d_model,
                nhead=config.transformer_nhead,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                transformer_layer, config.transformer_num_layers
            )

            # Feature Fusion Layer
            lstm_output_size = config.lstm_hidden_size * 2  # 双方向
            fusion_input_size = lstm_output_size + config.transformer_d_model

            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # 予測ヘッド
            self.prediction_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, config.prediction_horizon),
            )

            # 信頼度推定ヘッド
            self.confidence_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, config.prediction_horizon),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """順伝播"""
        batch_size, seq_len, features = x.shape

        # LSTM分岐
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]  # 最後の出力を使用

        # Transformer分岐
        transformer_input = self.input_projection(x)
        transformer_input = self.positional_encoding(transformer_input)
        transformer_out = self.transformer(transformer_input)
        transformer_features = transformer_out.mean(dim=1)  # グローバル平均プーリング

        # Feature Fusion
        fused_features = torch.cat([lstm_features, transformer_features], dim=1)
        fused_output = self.fusion_layer(fused_features)

        # 予測と信頼度
        predictions = self.prediction_head(fused_output)
        confidence = self.confidence_head(fused_output)

        return predictions, confidence


if PYTORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """位置エンコーディング"""

        def __init__(self, d_model: int, max_len: int = 2000):
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch が必要です")

            super().__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)

            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """位置エンコーディング追加"""
        return x + self.pe[: x.size(0), :].transpose(0, 1)


class AdvancedMLEngine:
    """Advanced ML Engine - 次世代AI予測システム"""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        import os

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

        # データ前処理パイプライン
        self.scaler = None
        self.feature_selector = None

        # パフォーマンス監視
        self.performance_history = []

        logger.info("Advanced ML Engine 初期化完了")

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

        # 特徴量エンジニアリング
        processed_data = self._engineer_features(market_data, feature_columns)

        # 系列データ作成
        sequences, targets = self._create_sequences(
            processed_data,
            target_column,
            self.config.sequence_length,
            self.config.prediction_horizon,
        )

        logger.info(f"前処理完了: {sequences.shape} -> {targets.shape}")
        return sequences, targets

    def _engineer_features(
        self, data: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """高度特徴量エンジニアリング"""

        result = data[feature_columns + ["終値"]].copy()

        # テクニカル指標強化
        for col in ["終値", "高値", "安値"]:
            if col in result.columns:
                # 多期間移動平均
                for period in [5, 10, 20, 50, 100, 200]:
                    result[f"{col}_MA_{period}"] = result[col].rolling(period).mean()
                    result[f"{col}_EMA_{period}"] = result[col].ewm(span=period).mean()

                # ボラティリティ指標
                result[f"{col}_volatility_10"] = (
                    result[col].pct_change().rolling(10).std()
                )
                result[f"{col}_volatility_20"] = (
                    result[col].pct_change().rolling(20).std()
                )

                # モメンタム指標
                for period in [5, 10, 20]:
                    result[f"{col}_momentum_{period}"] = result[col].pct_change(period)
                    result[f"{col}_roc_{period}"] = (
                        result[col] / result[col].shift(period) - 1
                    ) * 100

        # RSI (複数期間)
        if "終値" in result.columns:
            for period in [14, 21, 30]:
                result[f"RSI_{period}"] = self._calculate_rsi(result["終値"], period)

        # MACD
        if "終値" in result.columns:
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                result["終値"]
            )
            result["MACD"] = macd_line
            result["MACD_Signal"] = macd_signal
            result["MACD_Histogram"] = macd_histogram

        # ボリンジャーバンド
        if "終値" in result.columns:
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                    result["終値"], period
                )
                result[f"BB_Upper_{period}"] = bb_upper
                result[f"BB_Middle_{period}"] = bb_middle
                result[f"BB_Lower_{period}"] = bb_lower
                result[f"BB_Width_{period}"] = (bb_upper - bb_lower) / bb_middle
                result[f"BB_Position_{period}"] = (result["終値"] - bb_lower) / (
                    bb_upper - bb_lower
                )

        # 時系列分解特徴量
        if "終値" in result.columns:
            # 短期・中期・長期トレンド
            result["trend_short"] = result["終値"].rolling(20).mean()
            result["trend_medium"] = result["終値"].rolling(50).mean()
            result["trend_long"] = result["終値"].rolling(200).mean()

            # フーリエ変換特徴量（周期性検出）
            if len(result) >= 100:
                fft_features = self._extract_fft_features(result["終値"], n_features=10)
                for i, feature in enumerate(fft_features):
                    result[f"FFT_feature_{i}"] = feature

        # 欠損値処理
        result = result.fillna(method="ffill").fillna(method="bfill")

        # 正規化
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = scaler.fit_transform(result[numeric_columns])
        self.scaler = scaler

        # 特徴量選択（相関による）
        if len(numeric_columns) > self.config.num_features:
            correlation_with_target = (
                result.corr()["終値"].abs().sort_values(ascending=False)
            )
            selected_features = correlation_with_target.head(
                self.config.num_features
            ).index.tolist()
            result = result[selected_features]

        logger.info(f"特徴量エンジニアリング完了: {result.shape[1]} 特徴量")
        return result

    def _create_sequences(
        self, data: pd.DataFrame, target_col: str, seq_len: int, pred_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """時系列シーケンスデータ作成"""

        data_values = data.values
        target_values = data[target_col].values

        sequences = []
        targets = []

        for i in range(len(data_values) - seq_len - pred_horizon + 1):
            # 入力シーケンス
            seq = data_values[i : (i + seq_len)]
            # 予測ターゲット
            target = target_values[i + seq_len : (i + seq_len + pred_horizon)]

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

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
                    f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}"
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
            f"予測完了: {X.shape} -> {predictions.shape}, 処理時間: {inference_time:.3f}秒"
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
                    "scaler": self.scaler,
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
        self.scaler = checkpoint.get("scaler")

        self.model = LSTMTransformerHybrid(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"モデル読み込み完了: {path}")

    # ヘルパー関数
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(
        self, prices: pd.Series, fast=12, slow=26, signal=9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ボリンジャーバンド計算"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _extract_fft_features(
        self, prices: pd.Series, n_features: int = 10
    ) -> List[pd.Series]:
        """FFT特徴量抽出"""
        fft = np.fft.fft(prices.dropna().values)
        fft_features = []

        for i in range(1, n_features + 1):
            # 各周波数成分の振幅
            amplitude = np.abs(fft[i])
            # 全データに対して同じ値を繰り返し
            feature_series = pd.Series([amplitude] * len(prices), index=prices.index)
            fft_features.append(feature_series)

        return fft_features

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


# 便利関数
def create_advanced_ml_engine(config_dict: Optional[Dict] = None) -> AdvancedMLEngine:
    """Advanced ML Engine インスタンス作成"""
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
                from ..ml.deep_learning_models import LSTMModel, TransformerModel

                transformer_model = TransformerModel(self.config)
                lstm_model = LSTMModel(self.config)

                self.dl_manager.register_model("transformer", transformer_model)
                self.dl_manager.register_model("lstm", lstm_model)

                # アンサンブル訓練
                ensemble_results = self.dl_manager.train_ensemble(data)
                results.update(ensemble_results)

            total_training_time = time.time() - start_time

            # 性能評価
            performance_summary = self._evaluate_model_performance(data, target_column)

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
        """モデル性能評価"""
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
                mae = np.mean(np.abs(y_true - predictions))
                rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
                mape = (
                    np.mean(
                        np.abs(
                            (y_true - predictions) / np.where(y_true != 0, y_true, 1e-8)
                        )
                    )
                    * 100
                )
                accuracy = max(0, 100 - mape) / 100

                # 推論時間測定
                inference_times = []
                for _ in range(10):  # 10回測定
                    start = time.time()
                    _ = self.hybrid_model.predict(test_data.tail(10))
                    inference_times.append((time.time() - start) * 1000)  # ms変換

                avg_inference_time = np.mean(inference_times)

                # メトリクス履歴更新
                self.performance_metrics["accuracy_history"].append(accuracy)
                self.performance_metrics["mae_history"].append(mae)
                self.performance_metrics["rmse_history"].append(rmse)

                return {
                    "accuracy": accuracy,
                    "mae": mae,
                    "rmse": rmse,
                    "mape": mape,
                    "avg_inference_time": avg_inference_time,
                    "test_samples": min_len,
                }
            else:
                logger.warning("性能評価用データが不足")
                return {
                    "accuracy": 0.0,
                    "mae": 1.0,
                    "rmse": 1.0,
                    "mape": 100.0,
                    "avg_inference_time": 1000.0,
                    "test_samples": 0,
                }

        except Exception as e:
            logger.error(f"性能評価エラー: {e}")
            return {
                "accuracy": 0.0,
                "mae": 1.0,
                "rmse": 1.0,
                "mape": 100.0,
                "avg_inference_time": 1000.0,
                "test_samples": 0,
                "error": str(e),
            }

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """包括的システム概要"""
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
            "performance_history": {
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
            },
            "system_status": {
                "hybrid_model_initialized": self.hybrid_model is not None,
                "model_trained": (
                    self.hybrid_model.is_trained if self.hybrid_model else False
                ),
                "dl_manager_available": self.dl_manager is not None,
                "pytorch_available": PYTORCH_AVAILABLE,
                "metrics_integration": self.metrics_integration,
            },
        }


def create_next_gen_engine(config: Optional[Dict] = None) -> NextGenAITradingEngine:
    """次世代AIエンジンファクトリ関数"""
    if config:
        hybrid_config = HybridModelConfig(**config)
    else:
        hybrid_config = HybridModelConfig()

    return NextGenAITradingEngine(hybrid_config)
