#!/usr/bin/env python3
"""
予測精度向上アルゴリズム
Issue #881: 自動更新の更新時間を考える - 予測精度向上

機械学習と統計手法を組み合わせた予測精度向上システム
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class PredictionModel(Enum):
    """予測モデル種類"""
    ENSEMBLE = "ensemble"
    LSTM = "lstm"
    ARIMA = "arima"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    HYBRID = "hybrid"


class AccuracyMetric(Enum):
    """精度メトリクス"""
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    DIRECTIONAL = "directional"
    PROFIT_ACCURACY = "profit_accuracy"


@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    timestamp: datetime
    predicted_price: float
    actual_price: Optional[float] = None
    confidence: float = 0.0
    model_used: PredictionModel = PredictionModel.ENSEMBLE
    accuracy_score: Optional[float] = None
    features_used: List[str] = field(default_factory=list)


@dataclass
class ModelPerformance:
    """モデル性能"""
    model_type: PredictionModel
    symbol: str
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0
    profit_accuracy: float = 0.0
    prediction_count: int = 0
    last_updated: Optional[datetime] = None


class PredictionAccuracyEnhancer:
    """予測精度向上システム"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.models: Dict[str, Dict[PredictionModel, Any]] = {}
        self.performance_tracker: Dict[str, Dict[PredictionModel, ModelPerformance]] = {}
        self.predictions_history: List[PredictionResult] = []
        self.enhancement_active = True
        
        # 特徴量エンジニアリング設定
        self.feature_windows = [5, 10, 20, 50]
        self.technical_indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
        
        # モデル設定
        self.ensemble_weights = {
            PredictionModel.RANDOM_FOREST: 0.3,
            PredictionModel.GRADIENT_BOOSTING: 0.3,
            PredictionModel.LSTM: 0.25,
            PredictionModel.ARIMA: 0.15
        }
        
        self.setup_logging()
        self._initialize_models()
    
    def setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/prediction_accuracy.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> dict:
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"設定ファイルが見つかりません: {self.config_path}")
            return {}
    
    def _initialize_models(self) -> None:
        """モデル初期化"""
        self.logger.info("予測モデル初期化開始")
        
        symbols = self.config.get('watchlist', {}).get('symbols', [])
        
        for symbol_data in symbols:
            symbol = symbol_data['code']
            self.models[symbol] = {}
            self.performance_tracker[symbol] = {}
            
            # 各モデルタイプを初期化
            for model_type in PredictionModel:
                self.models[symbol][model_type] = self._create_model(model_type)
                self.performance_tracker[symbol][model_type] = ModelPerformance(
                    model_type=model_type,
                    symbol=symbol
                )
        
        self.logger.info(f"予測モデル初期化完了: {len(symbols)}銘柄")
    
    def _create_model(self, model_type: PredictionModel) -> Any:
        """モデル作成"""
        if model_type == PredictionModel.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == PredictionModel.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == PredictionModel.LSTM:
            # LSTMモデル（簡素化版）
            return self._create_lstm_model()
        elif model_type == PredictionModel.ARIMA:
            # ARIMAモデル（statsmodels利用想定）
            return None  # 実装時にstatsmodelsを使用
        elif model_type == PredictionModel.ENSEMBLE:
            return None  # アンサンブル用
        else:
            return None
    
    def _create_lstm_model(self):
        """LSTMモデル作成（簡素化版）"""
        # 実際の実装ではTensorFlow/Kerasを使用
        class SimpleLSTM:
            def __init__(self):
                self.trained = False
                self.scaler = StandardScaler()
            
            def fit(self, X, y):
                self.scaler.fit(X)
                self.trained = True
                return self
            
            def predict(self, X):
                if not self.trained:
                    return np.zeros(len(X))
                X_scaled = self.scaler.transform(X)
                # 簡単な線形予測（実際はLSTM）
                return np.mean(X_scaled, axis=1) * 1.01
        
        return SimpleLSTM()
    
    def generate_features(self, symbol: str, price_data: pd.DataFrame) -> pd.DataFrame:
        """特徴量生成"""
        features = pd.DataFrame(index=price_data.index)
        
        # 基本価格特徴量
        features['price'] = price_data['close']
        features['volume'] = price_data.get('volume', 0)
        features['high'] = price_data.get('high', price_data['close'])
        features['low'] = price_data.get('low', price_data['close'])
        
        # テクニカル指標
        for window in self.feature_windows:
            # 移動平均
            features[f'sma_{window}'] = price_data['close'].rolling(window).mean()
            features[f'ema_{window}'] = price_data['close'].ewm(span=window).mean()
            
            # ボラティリティ
            features[f'volatility_{window}'] = price_data['close'].rolling(window).std()
            
            # リターン
            features[f'return_{window}'] = price_data['close'].pct_change(window)
        
        # RSI
        delta = price_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = price_data['close'].ewm(span=12).mean()
        ema26 = price_data['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # ボリンジャーバンド
        sma20 = price_data['close'].rolling(20).mean()
        std20 = price_data['close'].rolling(20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_position'] = (price_data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # 時間特徴量
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        
        # NaN値処理
        features = features.fillna(method='forward').fillna(0)
        
        return features
    
    def train_models(self, symbol: str, price_data: pd.DataFrame) -> None:
        """モデル訓練"""
        self.logger.info(f"モデル訓練開始: {symbol}")
        
        # 特徴量生成
        features = self.generate_features(symbol, price_data)
        
        # ターゲット生成（次期価格）
        target = price_data['close'].shift(-1).dropna()
        features = features[:-1]  # 最後の行を削除（ターゲットと合わせる）
        
        # 訓練・テストデータ分割
        split_point = int(len(features) * 0.8)
        X_train, X_test = features[:split_point], features[split_point:]
        y_train, y_test = target[:split_point], target[split_point:]
        
        # 各モデルを訓練
        for model_type in [PredictionModel.RANDOM_FOREST, PredictionModel.GRADIENT_BOOSTING]:
            if model_type in self.models[symbol]:
                model = self.models[symbol][model_type]
                
                try:
                    model.fit(X_train, y_train)
                    
                    # 予測と評価
                    y_pred = model.predict(X_test)
                    self._update_performance(symbol, model_type, y_test, y_pred)
                    
                except Exception as e:
                    self.logger.error(f"モデル訓練エラー ({symbol}, {model_type.value}): {e}")
        
        # LSTMモデル訓練
        if PredictionModel.LSTM in self.models[symbol]:
            try:
                lstm_model = self.models[symbol][PredictionModel.LSTM]
                lstm_model.fit(X_train, y_train)
                
                y_pred = lstm_model.predict(X_test)
                self._update_performance(symbol, PredictionModel.LSTM, y_test, y_pred)
                
            except Exception as e:
                self.logger.error(f"LSTM訓練エラー ({symbol}): {e}")
        
        self.logger.info(f"モデル訓練完了: {symbol}")
    
    def _update_performance(self, symbol: str, model_type: PredictionModel, 
                           y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """性能指標更新"""
        if symbol not in self.performance_tracker:
            return
        
        performance = self.performance_tracker[symbol][model_type]
        
        # 各種メトリクス計算
        performance.mse = mean_squared_error(y_true, y_pred)
        performance.mae = mean_absolute_error(y_true, y_pred)
        performance.r2 = r2_score(y_true, y_pred)
        
        # 方向性精度
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        performance.directional_accuracy = np.mean(true_direction == pred_direction)
        
        # 利益精度（簡素化）
        returns_true = np.diff(y_true) / y_true[:-1]
        returns_pred = np.diff(y_pred) / y_true[:-1]  # 実際の価格ベース
        
        correct_profits = np.sum((returns_true > 0) & (returns_pred > 0)) + \
                         np.sum((returns_true < 0) & (returns_pred < 0))
        performance.profit_accuracy = correct_profits / len(returns_true)
        
        performance.prediction_count += len(y_pred)
        performance.last_updated = datetime.now()
    
    def make_ensemble_prediction(self, symbol: str, features: pd.DataFrame) -> PredictionResult:
        """アンサンブル予測"""
        predictions = {}
        confidences = {}
        
        # 各モデルで予測
        for model_type, weight in self.ensemble_weights.items():
            if symbol in self.models and model_type in self.models[symbol]:
                model = self.models[symbol][model_type]
                
                try:
                    if model_type in [PredictionModel.RANDOM_FOREST, PredictionModel.GRADIENT_BOOSTING]:
                        pred = model.predict(features.iloc[-1:].values)[0]
                    elif model_type == PredictionModel.LSTM:
                        pred = model.predict(features.iloc[-1:].values)[0]
                    else:
                        continue
                    
                    predictions[model_type] = pred
                    # 性能に基づく信頼度
                    performance = self.performance_tracker[symbol][model_type]
                    confidences[model_type] = max(0.1, performance.r2)
                    
                except Exception as e:
                    self.logger.warning(f"予測エラー ({symbol}, {model_type.value}): {e}")
        
        if not predictions:
            # フォールバック予測
            current_price = features['price'].iloc[-1]
            return PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                predicted_price=current_price,
                confidence=0.1,
                model_used=PredictionModel.ENSEMBLE
            )
        
        # 重み付き平均予測
        total_weight = 0
        weighted_prediction = 0
        
        for model_type, pred in predictions.items():
            weight = self.ensemble_weights.get(model_type, 0.1)
            confidence = confidences.get(model_type, 0.1)
            effective_weight = weight * confidence
            
            weighted_prediction += pred * effective_weight
            total_weight += effective_weight
        
        final_prediction = weighted_prediction / total_weight if total_weight > 0 else features['price'].iloc[-1]
        final_confidence = min(1.0, total_weight / sum(self.ensemble_weights.values()))
        
        result = PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            predicted_price=final_prediction,
            confidence=final_confidence,
            model_used=PredictionModel.ENSEMBLE,
            features_used=list(features.columns)
        )
        
        self.predictions_history.append(result)
        return result
    
    def evaluate_prediction_accuracy(self, symbol: str, actual_price: float, 
                                   prediction: PredictionResult) -> float:
        """予測精度評価"""
        if prediction.predicted_price == 0:
            return 0.0
        
        # 価格精度
        price_error = abs(actual_price - prediction.predicted_price) / actual_price
        price_accuracy = max(0, 1 - price_error)
        
        # 方向性精度（前回予測と比較）
        direction_accuracy = 0.5  # デフォルト
        
        if len(self.predictions_history) > 1:
            prev_prediction = None
            for pred in reversed(self.predictions_history[:-1]):
                if pred.symbol == symbol:
                    prev_prediction = pred
                    break
            
            if prev_prediction and prev_prediction.actual_price:
                true_direction = actual_price > prev_prediction.actual_price
                pred_direction = prediction.predicted_price > prev_prediction.predicted_price
                direction_accuracy = 1.0 if true_direction == pred_direction else 0.0
        
        # 総合精度
        accuracy = (price_accuracy * 0.6 + direction_accuracy * 0.4)
        
        # 予測結果更新
        prediction.actual_price = actual_price
        prediction.accuracy_score = accuracy
        
        return accuracy
    
    def optimize_ensemble_weights(self) -> None:
        """アンサンブル重み最適化"""
        self.logger.info("アンサンブル重み最適化開始")
        
        # 各モデルの平均性能計算
        model_performances = {model_type: [] for model_type in PredictionModel}
        
        for symbol_performances in self.performance_tracker.values():
            for model_type, performance in symbol_performances.items():
                if performance.prediction_count > 10:  # 十分なデータがある場合のみ
                    # 総合スコア計算
                    score = (
                        performance.r2 * 0.4 +
                        performance.directional_accuracy * 0.3 +
                        performance.profit_accuracy * 0.3
                    )
                    model_performances[model_type].append(score)
        
        # 新しい重みの計算
        new_weights = {}
        total_score = 0
        
        for model_type, scores in model_performances.items():
            if scores:
                avg_score = np.mean(scores)
                new_weights[model_type] = max(0.05, avg_score)  # 最小重み5%
                total_score += new_weights[model_type]
        
        # 正規化
        if total_score > 0:
            for model_type in new_weights:
                new_weights[model_type] /= total_score
            
            self.ensemble_weights.update(new_weights)
            self.logger.info(f"アンサンブル重み更新: {new_weights}")
    
    def generate_accuracy_report(self) -> dict:
        """精度レポート生成"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.predictions_history),
            "symbol_performance": {},
            "model_performance": {},
            "ensemble_weights": {k.value: v for k, v in self.ensemble_weights.items()}
        }
        
        # 銘柄別性能
        for symbol, model_performances in self.performance_tracker.items():
            symbol_stats = {}
            for model_type, performance in model_performances.items():
                if performance.prediction_count > 0:
                    symbol_stats[model_type.value] = {
                        "mse": round(performance.mse, 4),
                        "mae": round(performance.mae, 4),
                        "r2": round(performance.r2, 4),
                        "directional_accuracy": round(performance.directional_accuracy, 4),
                        "profit_accuracy": round(performance.profit_accuracy, 4),
                        "prediction_count": performance.prediction_count
                    }
            report["symbol_performance"][symbol] = symbol_stats
        
        # モデル別全体性能
        for model_type in PredictionModel:
            all_performances = []
            for symbol_performances in self.performance_tracker.values():
                if model_type in symbol_performances:
                    perf = symbol_performances[model_type]
                    if perf.prediction_count > 0:
                        all_performances.append(perf)
            
            if all_performances:
                report["model_performance"][model_type.value] = {
                    "avg_mse": round(np.mean([p.mse for p in all_performances]), 4),
                    "avg_mae": round(np.mean([p.mae for p in all_performances]), 4),
                    "avg_r2": round(np.mean([p.r2 for p in all_performances]), 4),
                    "avg_directional": round(np.mean([p.directional_accuracy for p in all_performances]), 4),
                    "avg_profit": round(np.mean([p.profit_accuracy for p in all_performances]), 4),
                    "total_predictions": sum(p.prediction_count for p in all_performances)
                }
        
        # 最近の予測精度
        recent_predictions = [p for p in self.predictions_history[-100:] if p.accuracy_score is not None]
        if recent_predictions:
            report["recent_accuracy"] = {
                "avg_accuracy": round(np.mean([p.accuracy_score for p in recent_predictions]), 4),
                "avg_confidence": round(np.mean([p.confidence for p in recent_predictions]), 4),
                "predictions_count": len(recent_predictions)
            }
        
        return report
    
    def save_accuracy_report(self, filepath: str = "reports/prediction_accuracy.json") -> None:
        """精度レポート保存"""
        report = self.generate_accuracy_report()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"精度レポート保存: {filepath}")
    
    async def run_accuracy_enhancement_cycle(self) -> None:
        """精度向上サイクル実行"""
        self.logger.info("予測精度向上システム開始")
        
        cycle_count = 0
        
        while self.enhancement_active:
            try:
                # アンサンブル重み最適化（1時間毎）
                if cycle_count % 3600 == 0:
                    self.optimize_ensemble_weights()
                
                # レポート生成（30分毎）
                if cycle_count % 1800 == 0:
                    self.save_accuracy_report()
                
                cycle_count += 1
                await asyncio.sleep(1)  # 1秒待機
                
            except KeyboardInterrupt:
                self.logger.info("精度向上システム停止要求受信")
                self.enhancement_active = False
            except Exception as e:
                self.logger.error(f"精度向上サイクルエラー: {e}")
                await asyncio.sleep(60)  # エラー時は1分待機
        
        self.logger.info("予測精度向上システム終了")


def main():
    """メイン実行"""
    enhancer = PredictionAccuracyEnhancer()
    
    try:
        # テスト用の仮想データで実行
        print("=== 予測精度向上システム ===")
        
        # 仮想価格データ生成
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        price_data = pd.DataFrame({
            'close': np.random.walk(len(dates)) * 100 + 1000,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        # テスト銘柄でモデル訓練
        test_symbol = "7203"
        enhancer.train_models(test_symbol, price_data)
        
        # 予測実行
        features = enhancer.generate_features(test_symbol, price_data)
        prediction = enhancer.make_ensemble_prediction(test_symbol, features)
        
        print(f"予測結果: {prediction.symbol}")
        print(f"予測価格: {prediction.predicted_price:.2f}")
        print(f"信頼度: {prediction.confidence:.3f}")
        print(f"使用モデル: {prediction.model_used.value}")
        
        # レポート生成
        report = enhancer.generate_accuracy_report()
        print(f"\n=== 精度レポート ===")
        print(f"総予測数: {report['total_predictions']}")
        
    except KeyboardInterrupt:
        print("\n予測精度向上システム停止")
    except Exception as e:
        print(f"システムエラー: {e}")


if __name__ == "__main__":
    main()