#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Prediction System - 最適化予測システム

現実的なデータ量で動作する改良版予測システム
Issue #800-2-1実装：予測精度改善計画
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path

# 機械学習ライブラリ
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class ModelType(Enum):
    """モデルタイプ"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE_VOTING = "ensemble_voting"

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cross_val_score: float
    feature_importance: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """予測結果"""
    symbol: str
    prediction: int  # 0: 下降, 1: 上昇
    confidence: float
    model_consensus: Dict[str, int]
    timestamp: datetime

class OptimizedFeatureEngineering:
    """最適化特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_optimized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """最適化特徴量作成（最小30データポイントで動作）"""

        if len(data) < 30:
            raise ValueError("データが不足しています（最低30データポイント必要）")

        features = pd.DataFrame(index=data.index)

        # 基本価格データ
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        open_price = data['Open']

        try:
            # 1. 基本価格指標（10種類）
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_range'] = (high - low) / close
            features['gap'] = (open_price - close.shift(1)) / close.shift(1)
            features['body_size'] = abs(close - open_price) / close
            features['upper_shadow'] = (high - np.maximum(close, open_price)) / close
            features['lower_shadow'] = (np.minimum(close, open_price) - low) / close
            features['hl2'] = (high + low) / 2
            features['hlc3'] = (high + low + close) / 3
            features['ohlc4'] = (open_price + high + low + close) / 4

            # 2. 移動平均系（12種類）
            for period in [3, 5, 10, 15]:
                if len(data) > period:
                    sma = close.rolling(period).mean()
                    features[f'sma_{period}'] = sma
                    features[f'sma_ratio_{period}'] = close / sma
                    features[f'sma_distance_{period}'] = (close - sma) / sma

            # 指数移動平均
            for period in [5, 12]:
                if len(data) > period:
                    ema = close.ewm(span=period).mean()
                    features[f'ema_{period}'] = ema
                    features[f'ema_ratio_{period}'] = close / ema

            # MACD（簡単版）
            if len(data) > 12:
                ema_5 = close.ewm(span=5).mean()
                ema_12 = close.ewm(span=12).mean()
                features['macd'] = ema_5 - ema_12
                features['macd_signal'] = features['macd'].ewm(span=3).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']

            # 3. ボラティリティ指標（8種類）
            for period in [3, 5, 10, 15]:
                if len(data) > period:
                    vol = features['returns'].rolling(period).std()
                    features[f'volatility_{period}'] = vol

                    # ATR計算
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    features[f'atr_{period}'] = true_range.rolling(period).mean()

            # 4. モメンタム指標（10種類）
            # RSI
            for period in [5, 10, 14]:
                if len(data) > period:
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                    rs = gain / loss.replace(0, 1)
                    features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # ROC (Rate of Change)
            for period in [3, 5, 10]:
                if len(data) > period:
                    features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
                    features[f'momentum_{period}'] = close / close.shift(period)

            # Stochastic（簡単版）
            if len(data) > 10:
                period = 10
                lowest_low = low.rolling(period).min()
                highest_high = high.rolling(period).max()
                k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
                features['stoch_k'] = k_percent
                features['stoch_d'] = k_percent.rolling(3).mean()

            # Williams %R
            if len(data) > 10:
                period = 10
                highest_high = high.rolling(period).max()
                lowest_low = low.rolling(period).min()
                features['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)

            # 5. ボリューム指標（6種類）
            for period in [5, 10]:
                if len(data) > period:
                    vol_sma = volume.rolling(period).mean()
                    features[f'volume_sma_{period}'] = vol_sma
                    features[f'volume_ratio_{period}'] = volume / vol_sma

            features['price_volume'] = close * volume
            if len(data) > 10:
                features['vwap'] = (features['price_volume'].rolling(10).sum() / volume.rolling(10).sum())

            # Volume Price Trend
            if len(data) > 5:
                features['vpt'] = ((close - close.shift(1)) / close.shift(1) * volume).rolling(5).sum()

            # On Balance Volume（簡単版）
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            features['obv'] = obv

            # 6. ボリンジャーバンド（4種類）
            if len(data) > 15:
                period = 15
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()

                upper = sma + (std * 2)
                lower = sma - (std * 2)

                features['bb_upper'] = upper
                features['bb_lower'] = lower
                features['bb_width'] = (upper - lower) / sma
                features['bb_position'] = (close - lower) / (upper - lower).replace(0, 1)

            # 7. トレンド指標（5種類）
            # ADX計算（簡単版）
            if len(data) > 10:
                period = 10
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0

                if f'atr_{period}' in features.columns:
                    atr = features[f'atr_{period}']
                    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
                    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

                    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
                    features['adx'] = dx.rolling(period).mean()
                    features['plus_di'] = plus_di
                    features['minus_di'] = minus_di

            # Aroon（簡単版）
            if len(data) > 10:
                period = 10
                def find_highest_index(x):
                    return len(x) - 1 - np.argmax(x) if len(x) > 0 else 0
                def find_lowest_index(x):
                    return len(x) - 1 - np.argmin(x) if len(x) > 0 else 0

                aroon_up = high.rolling(period + 1).apply(find_highest_index, raw=True) / period * 100
                aroon_down = low.rolling(period + 1).apply(find_lowest_index, raw=True) / period * 100

                features['aroon_up'] = aroon_up
                features['aroon_down'] = aroon_down

            # 8. 統計的指標（6種類）
            for period in [5, 10]:
                if len(data) > period:
                    ret_window = features['returns'].rolling(period)
                    features[f'skewness_{period}'] = ret_window.skew()
                    features[f'kurtosis_{period}'] = ret_window.kurt()
                    features[f'sharpe_ratio_{period}'] = (ret_window.mean() / ret_window.std() * np.sqrt(252))

            # 9. カスタム複合指標（8種類）
            # トレンド強度
            if len(data) > 5:
                features['trend_strength'] = (close > close.shift(1)).rolling(5).sum() / 5

            # 価格効率性
            if len(data) > 10 and 'atr_10' in features.columns:
                features['price_efficiency'] = abs(close - close.shift(10)) / features['atr_10'] / 10

            # ボラティリティブレイクアウト
            if 'volatility_5' in features.columns:
                features['vol_breakout'] = (features['volatility_5'] > features['volatility_5'].rolling(10).mean()).astype(int)

            # モメンタム発散
            if 'rsi_10' in features.columns and 'roc_5' in features.columns:
                features['momentum_divergence'] = features['rsi_10'] / 50 - features['roc_5'] / features['roc_5'].rolling(10).std()

            # ボリューム・価格発散
            if 'volume_ratio_5' in features.columns:
                features['volume_price_divergence'] = (features['volume_ratio_5'] - 1) * (features['returns'] > 0).astype(int)

            # 複合強度指標
            if all(col in features.columns for col in ['rsi_10', 'stoch_k', 'williams_r']):
                features['composite_strength'] = (
                    features['rsi_10'] / 100 +
                    features['stoch_k'] / 100 +
                    (features['williams_r'] + 100) / 100
                ) / 3

            # トレンド・ボリューム合流
            if 'adx' in features.columns and 'volume_ratio_5' in features.columns:
                features['trend_volume_confluence'] = features['adx'] * features['volume_ratio_5'] / 100

            # ボラティリティ・モメンタム
            if 'volatility_10' in features.columns and 'roc_5' in features.columns:
                features['volatility_momentum'] = features['volatility_10'] * abs(features['roc_5'])

        except Exception as e:
            self.logger.error(f"特徴量計算エラー: {e}")
            raise

        # NaN値処理（非推奨警告対応）
        features = features.ffill().bfill()

        # 無限大値処理
        features = features.replace([np.inf, -np.inf], 0)

        # 異常値除去（3シグマルール）
        for col in features.select_dtypes(include=[np.number]).columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = features[col].clip(mean - 3*std, mean + 3*std)

        # 最初の20行を削除（指標計算のため）
        if len(features) > 20:
            features = features.iloc[20:].copy()

        self.logger.info(f"最適化特徴量作成完了: {features.shape[1]}特徴量, {features.shape[0]}サンプル")
        return features

class OptimizedPredictionSystem:
    """最適化予測システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = OptimizedFeatureEngineering()

        # データベース設定
        self.db_path = Path("ml_models_data/optimized_predictions.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # 訓練されたモデル
        self.trained_models: Dict[str, Any] = {}

        self.logger.info("Optimized prediction system initialized")

    async def train_optimized_models(self, symbol: str, period: str = "6mo") -> Dict[ModelType, ModelPerformance]:
        """最適化モデル訓練"""

        self.logger.info(f"最適化モデル訓練開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 30:
                raise ValueError("訓練に十分なデータがありません")

            # 特徴量エンジニアリング
            features = self.feature_engineering.create_optimized_features(data)

            # ターゲット作成（翌日の上昇/下降）
            targets = self._create_targets(data.iloc[20:]['Close'])  # 特徴量と同じ期間

            # データ同期
            min_length = min(len(features), len(targets))
            features = features.iloc[:min_length]
            targets = targets[:min_length]

            if len(features) < 15:
                raise ValueError("訓練データが不足しています")

            # 特徴選択（重要な特徴量のみ）
            max_features = min(30, features.shape[1])
            features_selected = self._select_best_features(features, targets, k=max_features)

            # 各モデル訓練
            performances = {}

            # Random Forest
            rf_performance = await self._train_random_forest(features_selected, targets, symbol)
            performances[ModelType.RANDOM_FOREST] = rf_performance

            # Gradient Boosting
            gb_performance = await self._train_gradient_boosting(features_selected, targets, symbol)
            performances[ModelType.GRADIENT_BOOSTING] = gb_performance

            # Logistic Regression
            lr_performance = await self._train_logistic_regression(features_selected, targets, symbol)
            performances[ModelType.LOGISTIC_REGRESSION] = lr_performance

            # アンサンブルモデル
            if len(performances) >= 2:
                ensemble_performance = await self._create_ensemble_model(features_selected, targets, symbol)
                performances[ModelType.ENSEMBLE_VOTING] = ensemble_performance

            # 結果保存
            await self._save_model_performances(symbol, performances)

            self.logger.info(f"最適化モデル訓練完了: {len(performances)}モデル")
            return performances

        except Exception as e:
            self.logger.error(f"最適化モデル訓練エラー: {e}")
            raise

    def _create_targets(self, prices: pd.Series) -> np.ndarray:
        """ターゲット作成"""
        # 翌日の価格変動率
        returns = prices.pct_change().shift(-1)  # 翌日のリターン

        # 閾値以上の上昇を1、それ以外を0
        threshold = 0.002  # 0.2%以上の上昇
        targets = (returns > threshold).astype(int)

        return targets.values[:-1]  # 最後の要素（NaN）を除去

    def _select_best_features(self, features: pd.DataFrame, targets: np.ndarray, k: int = 30) -> pd.DataFrame:
        """最適特徴量選択"""

        k = min(k, features.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)

        features_selected = selector.fit_transform(features, targets)
        selected_columns = features.columns[selector.get_support()]

        return pd.DataFrame(features_selected, columns=selected_columns, index=features.index)

    async def _train_random_forest(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """Random Forest訓練"""

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42
        )

        return await self._train_and_evaluate_model(model, features, targets, ModelType.RANDOM_FOREST, symbol)

    async def _train_gradient_boosting(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """Gradient Boosting訓練"""

        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=3,
            random_state=42
        )

        return await self._train_and_evaluate_model(model, features, targets, ModelType.GRADIENT_BOOSTING, symbol)

    async def _train_logistic_regression(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """Logistic Regression訓練"""

        # スケーリング
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_scaled = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)

        model = LogisticRegression(
            C=1.0,
            penalty='l2',
            max_iter=500,
            random_state=42
        )

        performance = await self._train_and_evaluate_model(model, features_scaled, targets, ModelType.LOGISTIC_REGRESSION, symbol)

        # スケーラーも保存
        model_key = f"{symbol}_{ModelType.LOGISTIC_REGRESSION.value}"
        if model_key in self.trained_models:
            self.trained_models[model_key]['scaler'] = scaler

        return performance

    async def _train_and_evaluate_model(self, model, features: pd.DataFrame, targets: np.ndarray,
                                      model_type: ModelType, symbol: str) -> ModelPerformance:
        """モデル訓練・評価"""

        # 訓練
        model.fit(features, targets)

        # 予測
        predictions = model.predict(features)
        prediction_proba = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else predictions

        # クロスバリデーション（データが少ない場合は2-fold）
        cv_folds = min(3, len(features) // 5) if len(features) >= 10 else 2
        cv_scores = cross_val_score(model, features, targets, cv=cv_folds)

        # 特徴重要度
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance[features.columns[i]] = float(importance)

        performance = ModelPerformance(
            model_type=model_type,
            accuracy=accuracy_score(targets, predictions),
            precision=precision_score(targets, predictions, average='weighted', zero_division=0),
            recall=recall_score(targets, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(targets, predictions, average='weighted', zero_division=0),
            roc_auc=roc_auc_score(targets, prediction_proba) if len(np.unique(targets)) > 1 else 0.5,
            cross_val_score=float(cv_scores.mean()),
            feature_importance=feature_importance
        )

        # モデル保存
        model_key = f"{symbol}_{model_type.value}"
        self.trained_models[model_key] = {
            'model': model,
            'feature_columns': features.columns.tolist(),
            'performance': performance
        }

        return performance

    async def _create_ensemble_model(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """アンサンブルモデル作成"""

        # 最高性能のモデルを選択
        best_models = []
        for model_key, model_data in self.trained_models.items():
            if symbol in model_key and model_data['performance'].accuracy > 0.45:
                model_name = model_data['performance'].model_type.value
                best_models.append((model_name, model_data['model']))

        if len(best_models) < 2:
            # フォールバック：全モデル使用
            best_models = [
                (model_data['performance'].model_type.value, model_data['model'])
                for model_key, model_data in self.trained_models.items()
                if symbol in model_key
            ]

        # Voting Classifier作成
        voting_classifier = VotingClassifier(
            estimators=best_models,
            voting='soft' if all(hasattr(model, 'predict_proba') for _, model in best_models) else 'hard'
        )

        return await self._train_and_evaluate_model(voting_classifier, features, targets, ModelType.ENSEMBLE_VOTING, symbol)

    async def _save_model_performances(self, symbol: str, performances: Dict[ModelType, ModelPerformance]):
        """モデル性能保存"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # テーブル作成
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimized_performances (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        roc_auc REAL,
                        cross_val_score REAL,
                        feature_importance TEXT,
                        created_at TEXT,
                        UNIQUE(symbol, model_type)
                    )
                ''')

                # データ挿入
                for model_type, performance in performances.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO optimized_performances
                        (symbol, model_type, accuracy, precision_score, recall_score,
                         f1_score, roc_auc, cross_val_score, feature_importance, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        model_type.value,
                        performance.accuracy,
                        performance.precision,
                        performance.recall,
                        performance.f1_score,
                        performance.roc_auc,
                        performance.cross_val_score,
                        json.dumps(performance.feature_importance),
                        datetime.now().isoformat()
                    ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"性能保存エラー: {e}")

    async def predict_with_optimized_models(self, symbol: str) -> PredictionResult:
        """最適化モデルによる予測"""

        try:
            # 最新データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, "2mo")

            # 特徴量作成
            features = self.feature_engineering.create_optimized_features(data)
            latest_features = features.iloc[-1:].copy()

            # 各モデルで予測
            model_predictions = {}
            confidences = []

            for model_key, model_data in self.trained_models.items():
                if symbol in model_key:
                    try:
                        model = model_data['model']
                        feature_columns = model_data['feature_columns']

                        # 特徴量を合わせる
                        X = latest_features[feature_columns].copy()

                        # Logistic Regressionの場合はスケーリング
                        if 'scaler' in model_data:
                            X = model_data['scaler'].transform(X)
                            X = pd.DataFrame(X, columns=feature_columns, index=latest_features.index)

                        # 予測
                        prediction = model.predict(X)[0]
                        confidence = model.predict_proba(X)[0].max() if hasattr(model, 'predict_proba') else 0.6

                        model_type = model_data['performance'].model_type.value
                        model_predictions[model_type] = int(prediction)
                        confidences.append(confidence)

                    except Exception as e:
                        self.logger.warning(f"予測失敗 {model_key}: {e}")
                        continue

            # 最終予測（多数決）
            if model_predictions:
                final_prediction = int(np.mean(list(model_predictions.values())) >= 0.5)
                final_confidence = float(np.mean(confidences))
            else:
                final_prediction = 1  # デフォルト予測
                final_confidence = 0.5

            return PredictionResult(
                symbol=symbol,
                prediction=final_prediction,
                confidence=final_confidence,
                model_consensus=model_predictions,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"予測エラー: {e}")
            # デフォルト予測を返す
            return PredictionResult(
                symbol=symbol,
                prediction=1,
                confidence=0.5,
                model_consensus={},
                timestamp=datetime.now()
            )

# グローバルインスタンス
optimized_prediction_system = OptimizedPredictionSystem()

# テスト実行
async def test_optimized_system():
    """最適化システムテスト"""

    print("=== 最適化予測システムテスト ===")

    test_symbols = ["7203", "8306", "4751"]

    for symbol in test_symbols:
        print(f"\n🤖 {symbol} 最適化モデル訓練開始...")

        try:
            performances = await optimized_prediction_system.train_optimized_models(symbol, "6mo")

            print(f"📊 {symbol} 訓練結果:")
            for model_type, performance in performances.items():
                print(f"  {model_type.value}: 精度{performance.accuracy:.3f} F1{performance.f1_score:.3f} CV{performance.cross_val_score:.3f}")

            # 予測テスト
            prediction = await optimized_prediction_system.predict_with_optimized_models(symbol)
            print(f"🔮 {symbol} 予測: {'上昇' if prediction.prediction else '下降'} (信頼度: {prediction.confidence:.3f})")

        except Exception as e:
            print(f"❌ {symbol} エラー: {e}")

    print(f"\n✅ 最適化予測システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_optimized_system())