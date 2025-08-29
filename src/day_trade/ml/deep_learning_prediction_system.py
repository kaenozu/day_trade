"""
深層学習予測システム

高精度な予測を実現する深層学習モデル統合システム
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class DeepLearningConfig:
    """深層学習設定"""
    input_size: int = 20
    hidden_sizes: List[int] = None
    output_size: int = 1
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 15
    use_attention: bool = True
    use_residual: bool = True
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]


@dataclass
class PredictionResult:
    """予測結果"""
    prediction: float
    confidence: float
    probability: float
    features_importance: Dict[str, float]
    model_scores: Dict[str, float]


class AttentionModule(nn.Module):
    """アテンション機構"""
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        attention_weights = self.attention(x)
        attended_output = torch.sum(x * attention_weights, dim=1, keepdim=True)
        return attended_output, attention_weights


class ResidualBlock(nn.Module):
    """残差ブロック"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(input_size)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class AdvancedDeepLearningModel(nn.Module):
    """高度な深層学習モデル"""
    
    def __init__(self, config: DeepLearningConfig):
        super().__init__()
        self.config = config
        
        # 入力層
        self.input_layer = nn.Linear(config.input_size, config.hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(config.hidden_sizes[0])
        
        # 残差ブロック
        self.residual_blocks = nn.ModuleList()
        if config.use_residual:
            for i in range(len(config.hidden_sizes) - 1):
                self.residual_blocks.append(
                    ResidualBlock(config.hidden_sizes[i], config.hidden_sizes[i+1], config.dropout_rate)
                )
        
        # 隠れ層
        self.hidden_layers = nn.ModuleList()
        for i in range(len(config.hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i+1]))
            self.hidden_layers.append(nn.BatchNorm1d(config.hidden_sizes[i+1]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(config.dropout_rate))
        
        # アテンション機構
        if config.use_attention:
            self.attention = AttentionModule(config.hidden_sizes[-1])
            final_size = 1
        else:
            final_size = config.hidden_sizes[-1]
        
        # 出力層
        self.output_layer = nn.Sequential(
            nn.Linear(final_size, config.output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 入力処理
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.relu(x)
        
        # 残差ブロック適用
        if self.config.use_residual and self.residual_blocks:
            for residual_block in self.residual_blocks:
                x = residual_block(x)
        
        # 隠れ層処理
        for layer in self.hidden_layers:
            x = layer(x)
        
        # アテンション適用
        if self.config.use_attention:
            x = x.unsqueeze(1)  # バッチ次元追加
            x, attention_weights = self.attention(x)
            x = x.squeeze(1)
        
        # 出力
        output = self.output_layer(x)
        return output


class EnsembleDeepLearningSystem:
    """アンサンブル深層学習システム"""
    
    def __init__(self, num_models: int = 5):
        self.num_models = num_models
        self.models: List[AdvancedDeepLearningModel] = []
        self.scalers: List[StandardScaler] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量作成"""
        features_df = data.copy()
        
        # 価格関連特徴量
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['volume_price'] = features_df['volume'] * features_df['close']
        
        # 移動平均
        for window in [5, 10, 20]:
            features_df[f'ma_{window}'] = features_df['close'].rolling(window).mean()
            features_df[f'price_ma_ratio_{window}'] = features_df['close'] / features_df[f'ma_{window}']
        
        # ボリンジャーバンド
        rolling_mean = features_df['close'].rolling(20).mean()
        rolling_std = features_df['close'].rolling(20).std()
        features_df['bb_upper'] = rolling_mean + (rolling_std * 2)
        features_df['bb_lower'] = rolling_mean - (rolling_std * 2)
        features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (
            features_df['bb_upper'] - features_df['bb_lower']
        )
        
        # RSI
        delta = features_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features_df['close'].ewm(span=12).mean()
        exp2 = features_df['close'].ewm(span=26).mean()
        features_df['macd'] = exp1 - exp2
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        
        # ボラティリティ
        features_df['volatility'] = features_df['close'].rolling(10).std()
        features_df['volume_volatility'] = features_df['volume'].rolling(10).std()
        
        # 特徴量選択（数値列のみ）
        feature_columns = []
        for col in features_df.columns:
            if col not in ['timestamp', 'symbol'] and features_df[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)
        
        return features_df[feature_columns].fillna(0)
    
    def prepare_training_data(self, data: pd.DataFrame, target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """訓練データ準備"""
        # 高度な特徴量作成
        features_df = self._create_advanced_features(data)
        
        # ターゲット準備
        if target_column not in data.columns:
            # 次の価格が上昇するかを予測
            target = (data['close'].shift(-1) > data['close']).astype(int)
        else:
            target = data[target_column]
        
        # NaN除去
        valid_indices = ~(features_df.isna().any(axis=1) | target.isna())
        X = features_df[valid_indices].values
        y = target[valid_indices].values
        
        return X, y
    
    async def train_ensemble(self, data: pd.DataFrame, target_column: str = 'target') -> Dict[str, float]:
        """アンサンブルモデル訓練"""
        self.logger.info("深層学習アンサンブル訓練開始")
        
        # データ準備
        X, y = self.prepare_training_data(data, target_column)
        
        if len(X) < 100:
            raise ValueError("訓練データが不足しています")
        
        # データ分割
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        training_results = {}
        
        # 複数モデルの訓練
        for i in range(self.num_models):
            self.logger.info(f"モデル {i+1}/{self.num_models} 訓練中")
            
            # 設定の多様化
            config = DeepLearningConfig(
                input_size=X.shape[1],
                hidden_sizes=[128 + i*16, 64 + i*8, 32 + i*4],
                dropout_rate=0.2 + i*0.05,
                learning_rate=0.001 * (0.5 ** i),
                use_attention=i % 2 == 0,
                use_residual=i < 3
            )
            
            # モデル作成
            model = AdvancedDeepLearningModel(config).to(self.device)
            self.models.append(model)
            
            # スケーラー
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            self.scalers.append(scaler)
            
            # データローダー
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled).to(self.device),
                torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
            )
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            
            # 最適化
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            criterion = nn.BCELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
            
            # 訓練
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(config.epochs):
                # 訓練モード
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # 検証
                model.eval()
                with torch.no_grad():
                    val_outputs = model(torch.FloatTensor(X_val_scaled).to(self.device))
                    val_loss = criterion(val_outputs, torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device))
                    scheduler.step(val_loss)
                
                # 早期停止
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    self.logger.info(f"モデル {i+1} 早期停止 (エポック {epoch})")
                    break
            
            # 検証精度計算
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.FloatTensor(X_val_scaled).to(self.device))
                val_pred_binary = (val_pred.cpu().numpy() > 0.5).astype(int).flatten()
                accuracy = accuracy_score(y_val, val_pred_binary)
                training_results[f'model_{i+1}_accuracy'] = accuracy
        
        # 全体精度
        ensemble_accuracy = await self._evaluate_ensemble(X_val, y_val)
        training_results['ensemble_accuracy'] = ensemble_accuracy
        
        self.logger.info(f"アンサンブル訓練完了 - 精度: {ensemble_accuracy:.3f}")
        return training_results
    
    async def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> float:
        """アンサンブル評価"""
        if not self.models:
            return 0.0
        
        predictions = []
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            model.eval()
            with torch.no_grad():
                X_scaled = scaler.transform(X)
                pred = model(torch.FloatTensor(X_scaled).to(self.device))
                predictions.append(pred.cpu().numpy())
        
        # アンサンブル予測（平均）
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_binary = (ensemble_pred > 0.5).astype(int).flatten()
        
        return accuracy_score(y, ensemble_binary)
    
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """予測実行"""
        if not self.models:
            raise ValueError("モデルが訓練されていません")
        
        # 特徴量作成
        features_df = self._create_advanced_features(data)
        X = features_df.tail(1).values  # 最新データのみ
        
        predictions = []
        probabilities = []
        
        # 各モデルで予測
        for model, scaler in zip(self.models, self.scalers):
            model.eval()
            with torch.no_grad():
                X_scaled = scaler.transform(X)
                pred = model(torch.FloatTensor(X_scaled).to(self.device))
                prob = pred.cpu().numpy()[0][0]
                predictions.append(1 if prob > 0.5 else 0)
                probabilities.append(prob)
        
        # アンサンブル結果
        avg_probability = np.mean(probabilities)
        final_prediction = 1 if avg_probability > 0.5 else 0
        confidence = abs(avg_probability - 0.5) * 2  # 0.5からの距離を信頼度に
        
        # 特徴量重要度（簡易版）
        features_importance = {
            f'feature_{i}': 1.0 / len(features_df.columns) 
            for i in range(len(features_df.columns))
        }
        
        # モデルスコア
        model_scores = {
            f'model_{i+1}': prob 
            for i, prob in enumerate(probabilities)
        }
        
        return PredictionResult(
            prediction=final_prediction,
            confidence=confidence,
            probability=avg_probability,
            features_importance=features_importance,
            model_scores=model_scores
        )
    
    def save_models(self, directory: str = "models"):
        """モデル保存"""
        Path(directory).mkdir(exist_ok=True)
        
        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # PyTorchモデル保存
            torch.save(model.state_dict(), f"{directory}/deep_model_{i}.pt")
            # スケーラー保存
            joblib.dump(scaler, f"{directory}/scaler_{i}.pkl")
        
        self.logger.info(f"モデル保存完了: {directory}")
    
    def load_models(self, directory: str = "models"):
        """モデル読み込み"""
        self.models = []
        self.scalers = []
        
        model_files = list(Path(directory).glob("deep_model_*.pt"))
        
        for i in range(len(model_files)):
            # 設定（デフォルト）
            config = DeepLearningConfig()
            
            # モデル読み込み
            model = AdvancedDeepLearningModel(config).to(self.device)
            model.load_state_dict(torch.load(f"{directory}/deep_model_{i}.pt"))
            self.models.append(model)
            
            # スケーラー読み込み
            scaler = joblib.load(f"{directory}/scaler_{i}.pkl")
            self.scalers.append(scaler)
        
        self.logger.info(f"モデル読み込み完了: {len(self.models)}個")


async def demo_deep_learning_prediction():
    """深層学習予測デモ"""
    print("=== 深層学習予測システム デモ ===")
    
    # システム初期化
    dl_system = EnsembleDeepLearningSystem(num_models=3)
    
    # テストデータ作成
    np.random.seed(42)
    size = 2000
    dates = pd.date_range(start=datetime.now() - timedelta(days=size), periods=size, freq='1min')
    
    prices = [1000]
    for i in range(size-1):
        change = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'symbol': ['TEST'] * size,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 100000, size),
        'target': (pd.Series(prices).shift(-1) > pd.Series(prices)).astype(int)
    })
    
    try:
        print("\n1. 深層学習モデル訓練中...")
        training_results = await dl_system.train_ensemble(data)
        
        print("=== 訓練結果 ===")
        for key, value in training_results.items():
            print(f"{key}: {value:.3f}")
        
        print("\n2. 予測テスト...")
        test_data = data.tail(100)
        prediction = await dl_system.predict(test_data)
        
        print(f"=== 予測結果 ===")
        print(f"予測: {prediction.prediction}")
        print(f"確率: {prediction.probability:.3f}")
        print(f"信頼度: {prediction.confidence:.3f}")
        
        print(f"\n3. モデル保存...")
        dl_system.save_models("deep_learning_models")
        
        print(f"✅ 深層学習予測システム完了")
        
        return training_results
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    asyncio.run(demo_deep_learning_prediction())