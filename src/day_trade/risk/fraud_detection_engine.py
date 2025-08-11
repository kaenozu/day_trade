#!/usr/bin/env python3
"""
Deep Learning Fraud Detection Engine
深層学習不正検知エンジン

LSTM + Transformer + Ensemble による高精度不正取引検知システム
年間10億円規模の損失防止を目標とする次世代検知システム
"""

import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 深層学習フレームワーク
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# プロジェクト内インポート
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class FraudDetectionRequest:
    """不正検知リクエスト"""

    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    transaction_type: str
    account_balance: float
    location: str
    device_info: Dict[str, Any]
    transaction_history: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]


@dataclass
class FraudDetectionResult:
    """不正検知結果"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float  # 0-1
    confidence: float  # 0-1
    risk_factors: Dict[str, float]
    anomaly_score: float
    explanation: str
    recommended_action: str
    models_used: List[str]
    processing_time: float
    timestamp: datetime


class LSTMFraudModel(nn.Module):
    """LSTM不正検知モデル"""

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM層
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Attention メカニズム
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=8, dropout=dropout
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),  # Normal(0) / Fraud(1)
        )

    def forward(self, x):
        # LSTM処理
        lstm_out, _ = self.lstm(x)

        # Attention適用
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # 最終時刻の出力を使用
        final_output = attn_out[:, -1, :]

        # 分類
        logits = self.classifier(final_output)
        probabilities = F.softmax(logits, dim=1)

        return probabilities, attention_weights


class TransformerAnomalyModel(nn.Module):
    """Transformer異常検知モデル"""

    def __init__(
        self,
        input_dim: int = 50,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim

        # 入力埋め込み
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_encoding = self._create_positional_encoding(1000, model_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 異常スコア出力
        self.anomaly_head = nn.Sequential(
            nn.Linear(model_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def _create_positional_encoding(self, max_len: int, model_dim: int):
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        # 入力投影 + 位置エンコーディング
        x = self.input_projection(x)
        x += self.positional_encoding[:, :seq_len, :].to(x.device)

        # Transformer処理 (seq_len, batch, model_dim)
        x = x.transpose(0, 1)
        transformer_out = self.transformer(x)

        # 異常スコア計算（最終時刻）
        final_output = transformer_out[-1]  # (batch, model_dim)
        anomaly_score = self.anomaly_head(final_output)

        return anomaly_score.squeeze(-1)


class EnsembleFraudDetector:
    """アンサンブル不正検知器"""

    def __init__(self):
        self.models = {"lstm": None, "transformer": None, "isolation_forest": None}
        self.weights = {"lstm": 0.4, "transformer": 0.4, "isolation_forest": 0.2}
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """アンサンブルモデル学習"""

        logger.info("アンサンブル不正検知モデル学習開始")

        # データ前処理
        X_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_scaled = X_scaled.reshape(X_train.shape)

        # LSTM学習
        if TORCH_AVAILABLE:
            self._train_lstm(X_scaled, y_train)
            self._train_transformer(X_scaled, y_train)

        # Isolation Forest学習
        self._train_isolation_forest(X_scaled.reshape(len(X_scaled), -1))

        self.is_fitted = True
        logger.info("アンサンブルモデル学習完了")

    def _train_lstm(self, X: np.ndarray, y: np.ndarray):
        """LSTM学習"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # データ準備
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # モデル初期化
        input_size = X.shape[-1]
        self.models["lstm"] = LSTMFraudModel(input_size).to(device)

        optimizer = torch.optim.Adam(self.models["lstm"].parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # 学習ループ
        self.models["lstm"].train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                probabilities, _ = self.models["lstm"](batch_x)
                loss = criterion(probabilities, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(
                    f"LSTM Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}"
                )

        self.models["lstm"].eval()

    def _train_transformer(self, X: np.ndarray, y: np.ndarray):
        """Transformer学習"""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_dim = X.shape[-1]
        self.models["transformer"] = TransformerAnomalyModel(input_dim).to(device)

        optimizer = torch.optim.AdamW(
            self.models["transformer"].parameters(), lr=0.0001, weight_decay=0.01
        )
        criterion = nn.BCELoss()

        # 学習
        self.models["transformer"].train()
        for epoch in range(30):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                anomaly_scores = self.models["transformer"](batch_x)
                loss = criterion(anomaly_scores, batch_y)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0:
                logger.info(
                    f"Transformer Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}"
                )

        self.models["transformer"].eval()

    def _train_isolation_forest(self, X: np.ndarray):
        """Isolation Forest学習"""

        self.models["isolation_forest"] = IsolationForest(
            contamination=0.1,  # 10%を異常と仮定
            random_state=42,
            n_estimators=200,
        )
        self.models["isolation_forest"].fit(X)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """アンサンブル予測"""

        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")

        # データ前処理
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)

        predictions = {}

        # LSTM予測
        if self.models["lstm"]:
            device = next(self.models["lstm"].parameters()).device
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                probabilities, _ = self.models["lstm"](X_tensor)
                predictions["lstm"] = probabilities[:, 1].cpu().numpy()  # 不正確率

        # Transformer予測
        if self.models["transformer"]:
            device = next(self.models["transformer"].parameters()).device
            X_tensor = torch.FloatTensor(X_scaled).to(device)

            with torch.no_grad():
                anomaly_scores = self.models["transformer"](X_tensor)
                predictions["transformer"] = anomaly_scores.cpu().numpy()

        # Isolation Forest予測
        if self.models["isolation_forest"]:
            X_flat = X_scaled.reshape(len(X_scaled), -1)
            anomaly_scores = self.models["isolation_forest"].decision_function(X_flat)
            # 正規化 (異常ほど高いスコア)
            predictions["isolation_forest"] = 1 / (1 + np.exp(anomaly_scores))

        # 重み付きアンサンブル
        ensemble_scores = np.zeros(len(X))
        for model_name, scores in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_scores += weight * scores

        return ensemble_scores, predictions


class FraudDetectionEngine:
    """不正検知エンジン（統合システム）"""

    def __init__(self):
        self.ensemble_detector = EnsembleFraudDetector()
        self.feature_extractor = FeatureExtractor()
        self.models_loaded = False

        # パフォーマンス統計
        self.stats = {
            "total_detections": 0,
            "fraud_detected": 0,
            "false_positives": 0,
            "avg_processing_time": 0.0,
            "model_accuracies": {},
        }

        logger.info("不正検知エンジン初期化完了")

    def load_models(self, model_path: str = "models/fraud_detection/"):
        """学習済みモデル読み込み"""

        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.warning(f"モデルディレクトリが存在しません: {model_path}")
            return False

        try:
            # アンサンブルモデル読み込み
            ensemble_path = model_dir / "ensemble_fraud_detector.pkl"
            if ensemble_path.exists():
                with open(ensemble_path, "rb") as f:
                    self.ensemble_detector = pickle.load(f)

            self.models_loaded = True
            logger.info("不正検知モデル読み込み完了")
            return True

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

    def save_models(self, model_path: str = "models/fraud_detection/"):
        """モデル保存"""

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # アンサンブルモデル保存
            with open(model_dir / "ensemble_fraud_detector.pkl", "wb") as f:
                pickle.dump(self.ensemble_detector, f)

            logger.info(f"モデル保存完了: {model_path}")

        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")

    async def detect_fraud(
        self, request: FraudDetectionRequest
    ) -> FraudDetectionResult:
        """不正取引検知（メイン処理）"""

        start_time = time.time()

        try:
            # 特徴量抽出
            features = self.feature_extractor.extract_features(request)

            if not self.models_loaded:
                # モデルが未学習の場合、基本的なルールベース検知
                return self._rule_based_detection(request, start_time)

            # 深層学習予測
            X = np.array([features]).reshape(1, -1, len(features))
            ensemble_scores, individual_predictions = self.ensemble_detector.predict(X)

            fraud_probability = float(ensemble_scores[0])
            is_fraud = fraud_probability > 0.5

            # リスク要因分析
            risk_factors = self._analyze_risk_factors(
                request, features, individual_predictions
            )

            # 説明文生成
            explanation = self._generate_explanation(fraud_probability, risk_factors)

            # 推奨アクション
            recommended_action = self._get_recommended_action(
                fraud_probability, request
            )

            result = FraudDetectionResult(
                transaction_id=request.transaction_id,
                is_fraud=is_fraud,
                fraud_probability=fraud_probability,
                confidence=self._calculate_confidence(individual_predictions),
                risk_factors=risk_factors,
                anomaly_score=fraud_probability,
                explanation=explanation,
                recommended_action=recommended_action,
                models_used=["lstm", "transformer", "isolation_forest"],
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
            )

            # 統計更新
            self._update_stats(result)

            return result

        except Exception as e:
            logger.error(f"不正検知エラー: {e}")
            return self._create_error_result(request, str(e), start_time)

    def _rule_based_detection(
        self, request: FraudDetectionRequest, start_time: float
    ) -> FraudDetectionResult:
        """ルールベース検知（フォールバック）"""

        risk_score = 0.0
        risk_factors = {}

        # 高額取引チェック
        if request.amount > 1000000:  # 100万円以上
            risk_score += 0.3
            risk_factors["high_amount"] = 0.3

        # 時間帯チェック
        hour = request.timestamp.hour
        if hour < 6 or hour > 23:
            risk_score += 0.2
            risk_factors["unusual_time"] = 0.2

        # 口座残高比チェック
        if request.amount > request.account_balance * 0.8:
            risk_score += 0.4
            risk_factors["balance_ratio"] = 0.4

        # 取引履歴チェック
        if len(request.transaction_history) > 0:
            recent_transactions = [
                t
                for t in request.transaction_history
                if (request.timestamp - datetime.fromisoformat(t["timestamp"])).days
                <= 1
            ]
            if len(recent_transactions) > 10:
                risk_score += 0.3
                risk_factors["frequent_transactions"] = 0.3

        fraud_probability = min(1.0, risk_score)
        is_fraud = fraud_probability > 0.6

        return FraudDetectionResult(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            confidence=0.7,
            risk_factors=risk_factors,
            anomaly_score=fraud_probability,
            explanation=f"ルールベース検知: リスクスコア {fraud_probability:.2f}",
            recommended_action="追加認証" if is_fraud else "通常処理",
            models_used=["rule_based"],
            processing_time=time.time() - start_time,
            timestamp=datetime.now(),
        )

    def _analyze_risk_factors(
        self,
        request: FraudDetectionRequest,
        features: List[float],
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """リスク要因分析"""

        risk_factors = {}

        # 金額リスク
        amount_percentile = min(1.0, request.amount / 10000000)  # 1000万円基準
        risk_factors["amount_risk"] = amount_percentile

        # 時間リスク
        hour = request.timestamp.hour
        if hour < 9 or hour > 17:
            risk_factors["time_risk"] = 0.8
        else:
            risk_factors["time_risk"] = 0.2

        # 残高リスク
        balance_ratio = request.amount / max(request.account_balance, 1)
        risk_factors["balance_risk"] = min(1.0, balance_ratio)

        # 取引頻度リスク
        recent_count = len(
            [
                t
                for t in request.transaction_history
                if (request.timestamp - datetime.fromisoformat(t["timestamp"])).hours
                <= 24
            ]
        )
        risk_factors["frequency_risk"] = min(1.0, recent_count / 20)

        return risk_factors

    def _generate_explanation(
        self, fraud_probability: float, risk_factors: Dict[str, float]
    ) -> str:
        """説明文生成"""

        if fraud_probability > 0.8:
            severity = "非常に高い"
        elif fraud_probability > 0.6:
            severity = "高い"
        elif fraud_probability > 0.4:
            severity = "中程度"
        else:
            severity = "低い"

        top_factors = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)[:3]
        factor_names = {
            "amount_risk": "取引金額",
            "time_risk": "取引時間",
            "balance_risk": "残高比率",
            "frequency_risk": "取引頻度",
        }

        factors_text = ", ".join(
            [factor_names.get(name, name) for name, _ in top_factors]
        )

        return (
            f"不正取引の可能性: {severity} ({fraud_probability:.2f})\n"
            f"主要リスク要因: {factors_text}\n"
            f"深層学習モデルによる高精度分析結果"
        )

    def _get_recommended_action(
        self, fraud_probability: float, request: FraudDetectionRequest
    ) -> str:
        """推奨アクション決定"""

        if fraud_probability > 0.9:
            return "取引即座停止・緊急調査"
        elif fraud_probability > 0.7:
            return "取引保留・詳細確認"
        elif fraud_probability > 0.5:
            return "追加認証・監視強化"
        elif fraud_probability > 0.3:
            return "通常監視継続"
        else:
            return "通常処理継続"

    def _calculate_confidence(self, predictions: Dict[str, np.ndarray]) -> float:
        """信頼度計算"""

        if not predictions:
            return 0.5

        # モデル間の一致度を信頼度とする
        scores = list(predictions.values())
        if len(scores) < 2:
            return 0.8

        variance = np.var([s[0] if isinstance(s, np.ndarray) else s for s in scores])
        confidence = max(0.3, 1.0 - variance * 2)
        return confidence

    def _create_error_result(
        self, request: FraudDetectionRequest, error: str, start_time: float
    ) -> FraudDetectionResult:
        """エラー結果作成"""

        return FraudDetectionResult(
            transaction_id=request.transaction_id,
            is_fraud=False,
            fraud_probability=0.5,
            confidence=0.3,
            risk_factors={"error": 1.0},
            anomaly_score=0.5,
            explanation=f"検知エラー: {error}",
            recommended_action="手動確認",
            models_used=["error_fallback"],
            processing_time=time.time() - start_time,
            timestamp=datetime.now(),
        )

    def _update_stats(self, result: FraudDetectionResult):
        """統計更新"""

        self.stats["total_detections"] += 1
        if result.is_fraud:
            self.stats["fraud_detected"] += 1

        # 処理時間更新
        total = self.stats["total_detections"]
        old_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (
            old_avg * (total - 1) + result.processing_time
        ) / total

    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        return {
            **self.stats,
            "fraud_rate": self.stats["fraud_detected"]
            / max(1, self.stats["total_detections"]),
            "models_loaded": self.models_loaded,
        }


class FeatureExtractor:
    """特徴量抽出器"""

    def extract_features(self, request: FraudDetectionRequest) -> List[float]:
        """取引データから特徴量抽出"""

        features = []

        # 基本特徴量
        features.extend(
            [
                request.amount,
                request.account_balance,
                request.amount / max(request.account_balance, 1),  # 残高比
                request.timestamp.hour,
                request.timestamp.weekday(),
                len(request.transaction_history),
            ]
        )

        # 取引履歴統計
        if request.transaction_history:
            amounts = [t.get("amount", 0) for t in request.transaction_history]
            features.extend(
                [np.mean(amounts), np.std(amounts), np.max(amounts), np.min(amounts)]
            )
        else:
            features.extend([0, 0, 0, 0])

        # デバイス情報
        device_features = self._extract_device_features(request.device_info)
        features.extend(device_features)

        # 市場状況
        market_features = self._extract_market_features(request.market_conditions)
        features.extend(market_features)

        return features

    def _extract_device_features(self, device_info: Dict[str, Any]) -> List[float]:
        """デバイス特徴量抽出"""

        features = []

        # デバイスタイプ（エンコード）
        device_type = device_info.get("type", "unknown")
        device_encoding = {"mobile": 1, "desktop": 2, "tablet": 3}
        features.append(device_encoding.get(device_type, 0))

        # OS情報
        os_info = device_info.get("os", "unknown")
        os_encoding = {"ios": 1, "android": 2, "windows": 3, "macos": 4}
        features.append(os_encoding.get(os_info, 0))

        # 新規デバイスフラグ
        features.append(1 if device_info.get("is_new_device", False) else 0)

        return features

    def _extract_market_features(
        self, market_conditions: Dict[str, Any]
    ) -> List[float]:
        """市場特徴量抽出"""

        features = []

        # 市場ボラティリティ
        features.append(market_conditions.get("volatility", 0.2))

        # 取引量
        features.append(market_conditions.get("volume", 1000000))

        # 市場トレンド
        trend = market_conditions.get("trend", "neutral")
        trend_encoding = {"bullish": 1, "bearish": -1, "neutral": 0}
        features.append(trend_encoding.get(trend, 0))

        return features


# テスト用関数
async def test_fraud_detection_engine():
    """不正検知エンジンテスト"""

    engine = FraudDetectionEngine()

    # テスト用リクエスト（不正の可能性が高い）
    suspicious_request = FraudDetectionRequest(
        transaction_id="FRAUD_TEST_001",
        user_id="user_123",
        amount=5000000,  # 500万円（高額）
        timestamp=datetime(2025, 1, 1, 3, 0, 0),  # 深夜
        transaction_type="transfer",
        account_balance=1000000,  # 残高100万円（比率高い）
        location="foreign",
        device_info={"type": "mobile", "os": "android", "is_new_device": True},
        transaction_history=[
            {"amount": 100000, "timestamp": "2025-01-01T02:00:00"},
            {"amount": 200000, "timestamp": "2025-01-01T02:30:00"},
        ],
        market_conditions={"volatility": 0.4, "volume": 500000, "trend": "bearish"},
    )

    print("🔍 不正検知テスト開始...")

    result = await engine.detect_fraud(suspicious_request)

    print("✅ 検知完了!")
    print(f"⚠️ 不正判定: {'はい' if result.is_fraud else 'いいえ'}")
    print(f"📊 不正確率: {result.fraud_probability:.2f}")
    print(f"🎯 信頼度: {result.confidence:.2f}")
    print(f"🕐 処理時間: {result.processing_time:.3f}秒")
    print(f"📝 説明: {result.explanation}")
    print(f"🎯 推奨アクション: {result.recommended_action}")

    # 統計情報
    stats = engine.get_stats()
    print(f"📈 統計: {stats}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_fraud_detection_engine())
