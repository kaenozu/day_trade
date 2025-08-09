#!/usr/bin/env python3
"""
Global Trading Engine - Specialized AI Models
通貨ペア・暗号通貨専用AIモデル

Forex・Crypto市場の特性に最適化されたニューラルネットワーク
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)

@dataclass
class GlobalModelConfig:
    """グローバル市場AIモデル設定"""
    # 入力データ設定
    sequence_length: int = 60  # 1時間分のデータ（1分足）
    forex_features: int = 24   # Forex専用特徴量数
    crypto_features: int = 32  # Crypto専用特徴量数

    # モデルアーキテクチャ設定
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    attention_heads: int = 8

    # 出力設定
    prediction_horizons: List[int] = None  # [1, 5, 15, 60] 分後の予測

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 15, 60]

class ForexSpecializedLSTM(nn.Module):
    """外国為替専用LSTM"""

    def __init__(self, config: GlobalModelConfig):
        super().__init__()
        self.config = config

        # 通貨ペア別embedding
        self.pair_embedding = nn.Embedding(50, 32)  # 最大50通貨ペア

        # マルチタイムフレーム特徴抽出
        self.timeframe_convs = nn.ModuleList([
            nn.Conv1d(config.forex_features, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # 短期・中期・長期パターン
        ])

        # 双方向LSTM
        self.lstm = nn.LSTM(
            input_size=64 * 3 + 32,  # Conv出力 + embedding
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )

        # アテンション機構
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size * 2,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 通貨ペア特性考慮レイヤー
        self.pair_specific_layer = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + 32, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # マルチホライズン予測ヘッド
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}m': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, 4)  # price, direction, volatility, confidence
            ) for h in config.prediction_horizons
        })

        logger.info("Forex Specialized LSTM initialized")

    def forward(self, x: torch.Tensor, pair_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length, forex_features]
            pair_ids: [batch_size] 通貨ペアID
        """
        batch_size, seq_len, features = x.shape

        # 通貨ペアembedding
        pair_emb = self.pair_embedding(pair_ids)  # [batch_size, embedding_dim]

        # マルチタイムフレーム特徴抽出
        x_conv = x.transpose(1, 2)  # [batch_size, features, sequence_length]
        conv_outputs = []

        for conv in self.timeframe_convs:
            conv_out = F.relu(conv(x_conv))
            conv_outputs.append(conv_out)

        # 特徴量結合
        conv_features = torch.cat(conv_outputs, dim=1)  # [batch_size, 64*3, sequence_length]
        conv_features = conv_features.transpose(1, 2)  # [batch_size, sequence_length, 64*3]

        # ペアembeddingを各時点に追加
        pair_emb_expanded = pair_emb.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([conv_features, pair_emb_expanded], dim=2)

        # LSTM処理
        lstm_out, _ = self.lstm(lstm_input)

        # アテンション
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # 最終隠れ状態取得
        final_hidden = attended_out[:, -1, :]  # [batch_size, hidden_size*2]

        # 通貨ペア特性統合
        pair_specific_input = torch.cat([final_hidden, pair_emb], dim=1)
        pair_features = self.pair_specific_layer(pair_specific_input)

        # マルチホライズン予測
        predictions = {}
        for horizon_name, head in self.prediction_heads.items():
            pred = head(pair_features)
            predictions[horizon_name] = {
                'price_change': pred[:, 0],      # 価格変動率
                'direction': pred[:, 1],         # 方向性 (-1 to 1)
                'volatility': pred[:, 2],        # ボラティリティ
                'confidence': torch.sigmoid(pred[:, 3])  # 信頼度 (0 to 1)
            }

        return predictions

class CryptoSpecializedTransformer(nn.Module):
    """暗号通貨専用Transformer"""

    def __init__(self, config: GlobalModelConfig):
        super().__init__()
        self.config = config

        # 暗号通貨別embedding
        self.crypto_embedding = nn.Embedding(200, 64)  # 最大200暗号通貨

        # マルチスケール特徴抽出
        self.feature_projection = nn.Linear(config.crypto_features, config.hidden_size)

        # Positional Encoding
        self.pos_encoding = self._create_positional_encoding(
            config.sequence_length, config.hidden_size
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # 暗号通貨特有パターン認識
        self.pattern_layers = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size,
                     kernel_size=k, padding=k//2)
            for k in [3, 7, 15]  # 短期・中期・長期パターン
        ])

        # DeFi・NFT関連特徴抽出
        self.defi_layer = nn.Sequential(
            nn.Linear(config.hidden_size + 64, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # ボラティリティ予測専用ブランチ
        self.volatility_branch = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

        # マルチホライズン予測ヘッド
        self.prediction_heads = nn.ModuleDict({
            f'horizon_{h}m': nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, 5)  # price, direction, volume, volatility, confidence
            ) for h in config.prediction_horizons
        })

        logger.info("Crypto Specialized Transformer initialized")

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """位置エンコーディング生成"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def forward(self, x: torch.Tensor, crypto_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, sequence_length, crypto_features]
            crypto_ids: [batch_size] 暗号通貨ID
        """
        batch_size, seq_len, features = x.shape

        # 暗号通貨embedding
        crypto_emb = self.crypto_embedding(crypto_ids)  # [batch_size, embedding_dim]

        # 特徴量投影
        x_proj = self.feature_projection(x)  # [batch_size, seq_len, hidden_size]

        # 位置エンコーディング追加
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x_pos = x_proj + pos_enc

        # Transformer処理
        transformer_out = self.transformer_encoder(x_pos)

        # パターン認識
        transformer_conv = transformer_out.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        pattern_features = []

        for pattern_layer in self.pattern_layers:
            pattern_out = F.gelu(pattern_layer(transformer_conv))
            # Global Average Pooling
            pattern_pooled = F.adaptive_avg_pool1d(pattern_out, 1).squeeze(-1)
            pattern_features.append(pattern_pooled)

        # パターン特徴統合
        pattern_combined = torch.mean(torch.stack(pattern_features), dim=0)

        # 最終隠れ状態取得
        final_hidden = transformer_out[:, -1, :]  # [batch_size, hidden_size]

        # DeFi特徴統合
        crypto_emb_expanded = crypto_emb
        defi_input = torch.cat([final_hidden, crypto_emb_expanded], dim=1)
        defi_features = self.defi_layer(defi_input)

        # パターン特徴との統合
        combined_features = defi_features + pattern_combined

        # ボラティリティ予測
        volatility_pred = self.volatility_branch(combined_features)

        # マルチホライズン予測
        predictions = {}
        for horizon_name, head in self.prediction_heads.items():
            pred = head(combined_features)
            predictions[horizon_name] = {
                'price_change': pred[:, 0],           # 価格変動率
                'direction': torch.tanh(pred[:, 1]),  # 方向性 (-1 to 1)
                'volume_change': pred[:, 2],          # ボリューム変動
                'volatility': F.softplus(pred[:, 3]), # ボラティリティ (正値)
                'confidence': torch.sigmoid(pred[:, 4])  # 信頼度 (0 to 1)
            }

        # 全体ボラティリティ追加
        for horizon_pred in predictions.values():
            horizon_pred['overall_volatility'] = torch.sigmoid(volatility_pred.squeeze())

        return predictions

class CrossMarketFusionModel(nn.Module):
    """クロスマーケット統合モデル"""

    def __init__(self, config: GlobalModelConfig):
        super().__init__()
        self.config = config

        # 市場別特徴抽出器
        self.forex_extractor = nn.Sequential(
            nn.Linear(config.forex_features, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.crypto_extractor = nn.Sequential(
            nn.Linear(config.crypto_features, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # クロス市場アテンション
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size // 2,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )

        # 統合特徴処理
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )

        # 相関予測ヘッド
        self.correlation_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Tanh()  # -1 to 1 の相関係数
        )

        # マーケット間影響度予測
        self.influence_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 2),  # forex->crypto, crypto->forex
            nn.Softmax(dim=1)
        )

        logger.info("Cross Market Fusion Model initialized")

    def forward(self, forex_features: torch.Tensor,
                crypto_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            forex_features: [batch_size, sequence_length, forex_features]
            crypto_features: [batch_size, sequence_length, crypto_features]
        """
        # 市場別特徴抽出
        forex_extracted = self.forex_extractor(forex_features)
        crypto_extracted = self.crypto_extractor(crypto_features)

        # クロスアテンション（Forex -> Crypto）
        forex_to_crypto, _ = self.cross_attention(
            crypto_extracted, forex_extracted, forex_extracted
        )

        # クロスアテンション（Crypto -> Forex）
        crypto_to_forex, _ = self.cross_attention(
            forex_extracted, crypto_extracted, crypto_extracted
        )

        # 最終状態統合
        forex_final = crypto_to_forex[:, -1, :]  # [batch_size, hidden_size//2]
        crypto_final = forex_to_crypto[:, -1, :]  # [batch_size, hidden_size//2]

        combined_features = torch.cat([forex_final, crypto_final], dim=1)

        # 統合特徴処理
        fused_features = self.fusion_layer(combined_features)

        # 予測計算
        correlation = self.correlation_head(fused_features)
        influence = self.influence_head(fused_features)

        return {
            'correlation': correlation.squeeze(),  # 市場間相関係数
            'forex_to_crypto_influence': influence[:, 0],  # Forex->Crypto影響度
            'crypto_to_forex_influence': influence[:, 1],  # Crypto->Forex影響度
            'combined_features': fused_features  # 統合特徴（他のモデルで利用可能）
        }

class GlobalMarketPredictor(nn.Module):
    """グローバル市場統合予測器"""

    def __init__(self, config: GlobalModelConfig):
        super().__init__()
        self.config = config

        # 専用モデル
        self.forex_model = ForexSpecializedLSTM(config)
        self.crypto_model = CryptoSpecializedTransformer(config)
        self.fusion_model = CrossMarketFusionModel(config)

        # モデル統合重み
        self.model_weights = nn.Parameter(torch.ones(3) / 3)  # 初期重み均等

        logger.info("Global Market Predictor initialized")

    def forward(self, forex_data: torch.Tensor, crypto_data: torch.Tensor,
                forex_pair_ids: torch.Tensor, crypto_ids: torch.Tensor) -> Dict[str, Any]:
        """統合予測実行"""

        # 個別モデル予測
        forex_predictions = self.forex_model(forex_data, forex_pair_ids)
        crypto_predictions = self.crypto_model(crypto_data, crypto_ids)

        # クロスマーケット分析
        fusion_results = self.fusion_model(forex_data, crypto_data)

        # 重み正規化
        weights = F.softmax(self.model_weights, dim=0)

        return {
            'forex_predictions': forex_predictions,
            'crypto_predictions': crypto_predictions,
            'cross_market_analysis': fusion_results,
            'model_weights': weights,
            'timestamp': datetime.now()
        }

def create_global_ai_models(config: GlobalModelConfig = None) -> GlobalMarketPredictor:
    """グローバルAIモデル作成"""
    if config is None:
        config = GlobalModelConfig()

    return GlobalMarketPredictor(config)

# 使用例・テスト関数
def test_global_models():
    """モデルテスト"""
    config = GlobalModelConfig(
        sequence_length=60,
        forex_features=24,
        crypto_features=32,
        hidden_size=128,
        num_layers=2
    )

    model = create_global_ai_models(config)

    # ダミーデータ
    batch_size = 4
    forex_data = torch.randn(batch_size, config.sequence_length, config.forex_features)
    crypto_data = torch.randn(batch_size, config.sequence_length, config.crypto_features)
    forex_pair_ids = torch.randint(0, 10, (batch_size,))
    crypto_ids = torch.randint(0, 20, (batch_size,))

    # 予測実行
    with torch.no_grad():
        results = model(forex_data, crypto_data, forex_pair_ids, crypto_ids)

    print("Global AI Models Test Results:")
    print(f"Forex predictions: {len(results['forex_predictions'])} horizons")
    print(f"Crypto predictions: {len(results['crypto_predictions'])} horizons")
    print(f"Cross-market correlation: {results['cross_market_analysis']['correlation'].mean():.4f}")
    print(f"Model weights: {results['model_weights']}")

if __name__ == "__main__":
    test_global_models()
