#!/usr/bin/env python3
"""
ハイブリッドLSTM-Transformerモデル - PyTorch実装
"""

from typing import Dict, Tuple, Union

from ...utils.logging_config import get_context_logger
from .attention_layers import CrossAttentionLayer
from .config import HybridModelConfig
from .transformer_layers import ModifiedTransformerEncoder

logger = get_context_logger(__name__)

# Issue #696対応: PyTorch可用性チェック
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    PYTORCH_AVAILABLE = True
    logger.info("PyTorch利用可能 - 完全なHybridLSTMTransformerModelを使用")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch未インストール - HybridLSTMTransformerModelは利用できません")

    # フォールバック用のダミークラス
    class nn:
        class Module:
            pass

        class Linear:
            pass

        class LSTM:
            pass

        class Sequential:
            pass

        class LayerNorm:
            pass

        class GELU:
            pass

        class Dropout:
            pass

        class init:
            @staticmethod
            def xavier_uniform_(param):
                pass

            @staticmethod
            def zeros_(param):
                pass

            @staticmethod
            def kaiming_normal_(weight, mode='fan_out', nonlinearity='relu'):
                pass

    class torch:
        class Tensor:
            pass

        @staticmethod
        def cat(*args, **kwargs):
            return None

        @staticmethod
        def stack(*args, **kwargs):
            return None


if PYTORCH_AVAILABLE:

    class HybridLSTMTransformerModel(nn.Module):
        """ハイブリッドLSTM-Transformerモデル - PyTorch実装"""

        def __init__(self, config: HybridModelConfig, input_dim: int):
            super().__init__()
            self.config = config
            self.input_dim = input_dim

            # Input projection
            self.input_projection = nn.Linear(input_dim, config.lstm_hidden_size)

            # LSTM Branch
            self.lstm = nn.LSTM(
                input_size=config.lstm_hidden_size,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0.0,
                bidirectional=config.lstm_bidirectional,
            )

            lstm_output_dim = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)

            # Transformer Branch - Input projection for transformer
            self.transformer_projection = nn.Linear(config.lstm_hidden_size, config.transformer_d_model)

            # Modified Transformer
            self.modified_transformer = ModifiedTransformerEncoder(
                d_model=config.transformer_d_model,
                num_heads=config.transformer_num_heads,
                dim_feedforward=config.transformer_dim_feedforward,
                num_layers=config.transformer_num_layers,
                dropout=config.dropout_rate,
            )

            # Cross-Attention Fusion
            self.cross_attention = CrossAttentionLayer(
                lstm_dim=lstm_output_dim,
                transformer_dim=config.transformer_d_model,
                attention_heads=config.cross_attention_heads,
                attention_dim=config.cross_attention_dim,
            )

            # Prediction Head (MLP)
            fusion_layers = []
            prev_dim = config.cross_attention_dim

            for hidden_dim in config.fusion_hidden_dims:
                fusion_layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(config.fusion_dropout),
                    ]
                )
                prev_dim = hidden_dim

            # Final prediction layer
            fusion_layers.append(nn.Linear(prev_dim, config.prediction_horizon))

            self.prediction_head = nn.Sequential(*fusion_layers)

            # 不確実性推定用（Monte Carlo Dropout）
            self.uncertainty_dropout = nn.Dropout(0.5)

            # Initialize weights
            self._initialize_weights()

            # Issue #698対応: メモリ効率化フラグ
            self._use_efficient_tensor_ops = True

        def _initialize_weights(self):
            """Xavier/Kaiming初期化 - Issue #698対応: 効率的な重み初期化"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LSTM):
                    for name, param in module.named_parameters():
                        if "weight" in name:
                            nn.init.xavier_uniform_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)
                elif isinstance(module, nn.Conv1d):
                    # Issue #698対応: Conv1d の効率的な初期化
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

        def forward(
            self, x: torch.Tensor, return_attention: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
            """
            ハイブリッドモデル推論 - Issue #698対応: テンソル操作最適化

            Args:
                x: Input tensor [batch, seq_len, features]
                return_attention: アテンション重み返却フラグ

            Returns:
                predictions: 予測結果 [batch, prediction_horizon]
                attention_info: アテンション情報 (return_attention=True時)
            """
            batch_size, seq_len, _ = x.shape

            # Issue #698対応: 効率的な入力射影
            x_proj = self.input_projection(x)  # [batch, seq_len, lstm_hidden_size]

            # LSTM Branch
            lstm_out, (hidden, cell) = self.lstm(x_proj)

            # Issue #698対応: 効率的な隠れ状態抽出
            if self.config.lstm_bidirectional:
                # 最適化: viewを使用してメモリ効率向上
                if self._use_efficient_tensor_ops:
                    # hidden: [num_layers*2, batch, hidden_size] -> [batch, hidden_size*2]
                    lstm_features = (
                        hidden.view(self.config.lstm_num_layers, 2, batch_size, -1)[-1]
                        .transpose(0, 1)
                        .contiguous()
                        .view(batch_size, -1)
                    )
                else:
                    lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                lstm_features = hidden[-1]  # [batch, hidden_size]

            # Transformer Branch
            transformer_input = self.transformer_projection(x_proj)  # [batch, seq_len, d_model]
            transformer_out = self.modified_transformer(transformer_input)  # [batch, seq_len, d_model]

            # Issue #698対応: 効率的な最終特徴量抽出（インデックス使用）
            transformer_features = transformer_out[:, -1]  # [batch, d_model] (sliceは:を削除)

            # Cross-Attention Fusion
            fused_features, attention_weights = self.cross_attention(lstm_features, transformer_features)

            # Prediction Head
            predictions = self.prediction_head(fused_features)

            if return_attention:
                # Issue #698対応: メモリ効率的な注意重み抽出
                if self._use_efficient_tensor_ops:
                    # GPU->CPUのコピーを最小化
                    attention_mean = attention_weights.mean(dim=(0, 1))  # 一度に計算
                    attention_info = {
                        "cross_attention_weights": attention_weights.detach().cpu().numpy(),
                        "lstm_contribution": attention_weights[:, 0, 0].mean().item(),
                        "transformer_contribution": attention_weights[:, 1, 1].mean().item(),
                    }
                else:
                    attention_info = {
                        "cross_attention_weights": attention_weights.detach().cpu().numpy(),
                        "lstm_contribution": attention_weights[:, 0, 0].mean().item(),
                        "transformer_contribution": attention_weights[:, 1, 1].mean().item(),
                    }
                return predictions, attention_info

            return predictions

        def forward_with_uncertainty(
            self, x: torch.Tensor, num_samples: int = 50
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Monte Carlo Dropout による不確実性推定 - Issue #698対応: テンソル操作最適化

            Args:
                x: Input tensor
                num_samples: MC samples数

            Returns:
                mean_pred: 平均予測
                std_pred: 標準偏差（不確実性）
            """
            self.train()  # Dropoutを有効化

            # Issue #698対応: 効率的なバッチ予測（メモリ使用量削減）
            if self._use_efficient_tensor_ops and num_samples > 10:
                # 大きなサンプル数の場合はバッチ処理
                batch_size = min(10, num_samples // 2)
                predictions = []

                with torch.no_grad():
                    for i in range(0, num_samples, batch_size):
                        current_batch_size = min(batch_size, num_samples - i)
                        batch_preds = []

                        for _ in range(current_batch_size):
                            pred = self(x)
                            batch_preds.append(pred)

                        batch_stack = torch.stack(batch_preds)
                        predictions.append(batch_stack)

                # Issue #698対応: メモリ効率的な結合
                predictions = torch.cat(predictions, dim=0)  # [num_samples, batch, prediction_horizon]
            else:
                # 従来の実装（少数サンプル用）
                predictions = []
                with torch.no_grad():
                    for _ in range(num_samples):
                        pred = self(x)
                        predictions.append(pred)

                predictions = torch.stack(predictions)  # [num_samples, batch, prediction_horizon]

            # Issue #698対応: 効率的な統計計算
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)

            return mean_pred, std_pred

else:

    class HybridLSTMTransformerModel:
        """PyTorch未利用時のダミークラス"""

        def __init__(self, config: HybridModelConfig, input_dim: int):
            logger.warning("PyTorchが利用できません - HybridLSTMTransformerModelは無効化されています")

        def forward(self, x, return_attention=False):
            logger.warning("HybridLSTMTransformerModel.forward() - PyTorchが必要です")
            return None

        def forward_with_uncertainty(self, x, num_samples=50):
            logger.warning("HybridLSTMTransformerModel.forward_with_uncertainty() - PyTorchが必要です")
            return None, None

        def parameters(self):
            return []

        def train(self):
            pass

        def eval(self):
            pass