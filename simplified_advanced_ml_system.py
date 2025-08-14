#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Advanced ML System - 簡略化高度機械学習システム

TALibなしで実装可能な改良版予測システム
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
        sys.stderr = codecs.getwriter('utf-8')(sys.stdout.buffer)

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

class SimplifiedAdvancedFeatureEngineering:
    """簡略化された高度特徴量エンジニアリング"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度特徴量作成（TALib非依存版）"""

        if len(data) < 60:
            raise ValueError("データが不足しています（最低60データポイント必要）")

        features = pd.DataFrame(index=data.index)

        # 基本価格データ
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        open_price = data['Open']

        try:
            # 1. 基本価格指標（12種類）
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
            features['typical_price'] = (high + low + close) / 3
            features['weighted_close'] = (high + low + 2*close) / 4

            # 2. 移動平均系（20種類）
            for period in [5, 10, 15, 20, 30, 50]:
                sma = close.rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'sma_ratio_{period}'] = close / sma
                features[f'sma_distance_{period}'] = (close - sma) / sma

                if period <= 20:
                    features[f'sma_slope_{period}'] = (sma - sma.shift(5)) / sma.shift(5)

            # 指数移動平均
            for period in [12, 26, 50]:
                ema = close.ewm(span=period).mean()
                features[f'ema_{period}'] = ema
                features[f'ema_ratio_{period}'] = close / ema

            # MACD系
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            features['macd_ratio'] = features['macd'] / features['macd_signal']

            # 3. ボラティリティ指標（15種類）
            for period in [5, 10, 15, 20, 30]:
                vol = features['returns'].rolling(period).std()
                features[f'volatility_{period}'] = vol
                features[f'vol_ratio_{period}'] = vol / vol.rolling(20).mean()

                # ATR計算
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                features[f'atr_{period}'] = true_range.rolling(period).mean()

            # 4. モメンタム指標（18種類）
            # RSI
            for period in [7, 14, 21]:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
                rs = gain / loss.replace(0, 1)
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # ROC (Rate of Change)
            for period in [5, 10, 15, 20]:
                features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100
                features[f'momentum_{period}'] = close / close.shift(period)

            # Stochastic
            for period in [14, 21]:
                lowest_low = low.rolling(period).min()
                highest_high = high.rolling(period).max()
                k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
                features[f'stoch_k_{period}'] = k_percent
                features[f'stoch_d_{period}'] = k_percent.rolling(3).mean()

            # Williams %R
            for period in [14, 21]:
                highest_high = high.rolling(period).max()
                lowest_low = low.rolling(period).min()
                features[f'williams_r_{period}'] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1)

            # 5. ボリューム指標（12種類）
            for period in [10, 20]:
                vol_sma = volume.rolling(period).mean()
                features[f'volume_sma_{period}'] = vol_sma
                features[f'volume_ratio_{period}'] = volume / vol_sma

            features['price_volume'] = close * volume
            features['vwap_10'] = (features['price_volume'].rolling(10).sum() / volume.rolling(10).sum())
            features['vwap_20'] = (features['price_volume'].rolling(20).sum() / volume.rolling(20).sum())

            # Volume Price Trend
            features['vpt'] = ((close - close.shift(1)) / close.shift(1) * volume).rolling(10).sum()

            # On Balance Volume
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
            features['obv_sma'] = obv.rolling(10).mean()

            # Accumulation/Distribution Line
            clv = ((close - low) - (high - close)) / (high - low).replace(0, 1)
            features['ad_line'] = (clv * volume).cumsum()
            features['ad_line_sma'] = features['ad_line'].rolling(10).mean()

            # 6. ボリンジャーバンド（8種類）
            for period in [20, 30]:
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()

                upper = sma + (std * 2)
                lower = sma - (std * 2)

                features[f'bb_upper_{period}'] = upper
                features[f'bb_lower_{period}'] = lower
                features[f'bb_width_{period}'] = (upper - lower) / sma
                features[f'bb_position_{period}'] = (close - lower) / (upper - lower).replace(0, 1)

            # 7. トレンド指標（10種類）
            # ADX計算
            for period in [14, 21]:
                plus_dm = high.diff()
                minus_dm = -low.diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0

                atr = features[f'atr_{period}']
                plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
                features[f'adx_{period}'] = dx.rolling(period).mean()
                features[f'plus_di_{period}'] = plus_di
                features[f'minus_di_{period}'] = minus_di

            # Aroon
            for period in [14, 25]:
                def find_highest_index(x):
                    return len(x) - 1 - np.argmax(x) if len(x) > 0 else 0
                def find_lowest_index(x):
                    return len(x) - 1 - np.argmin(x) if len(x) > 0 else 0

                aroon_up = high.rolling(period + 1).apply(find_highest_index, raw=True) / period * 100
                aroon_down = low.rolling(period + 1).apply(find_lowest_index, raw=True) / period * 100

                features[f'aroon_up_{period}'] = aroon_up
                features[f'aroon_down_{period}'] = aroon_down
                features[f'aroon_osc_{period}'] = aroon_up - aroon_down

            # 8. 統計的指標（12種類）
            for period in [10, 20]:
                ret_window = features['returns'].rolling(period)
                features[f'skewness_{period}'] = ret_window.skew()
                features[f'kurtosis_{period}'] = ret_window.kurt()
                features[f'var_95_{period}'] = ret_window.quantile(0.05)
                features[f'var_05_{period}'] = ret_window.quantile(0.95)
                features[f'sharpe_ratio_{period}'] = (ret_window.mean() / ret_window.std() * np.sqrt(252))
                features[f'sortino_ratio_{period}'] = (ret_window.mean() / ret_window[ret_window < 0].std() * np.sqrt(252))

            # 9. カスタム複合指標（15種類）
            # トレンド強度
            features['trend_strength_5'] = (close > close.shift(1)).rolling(5).sum() / 5
            features['trend_strength_10'] = (close > close.shift(1)).rolling(10).sum() / 10
            features['trend_strength_20'] = (close > close.shift(1)).rolling(20).sum() / 20

            # 価格効率性
            for period in [10, 20]:
                features[f'price_efficiency_{period}'] = abs(close - close.shift(period)) / features[f'atr_{period}'] / period

            # ボラティリティブレイクアウト
            features['vol_breakout_5'] = (features['volatility_5'] > features['volatility_5'].rolling(20).mean()).astype(int)
            features['vol_breakout_10'] = (features['volatility_10'] > features['volatility_10'].rolling(20).mean()).astype(int)

            # モメンタム発散
            features['momentum_divergence'] = features['rsi_14'] / 50 - features['roc_14'] / features['roc_14'].rolling(20).std()

            # ボリューム・価格発散
            features['volume_price_divergence'] = (features['volume_ratio_10'] - 1) * (features['returns'] > 0).astype(int)

            # 市場効率性
            features['market_efficiency'] = abs(close - close.shift(20)) / (features['atr_20'] * 20)

            # 複合強度指標
            features['composite_strength'] = (
                features['rsi_14'] / 100 +
                features['stoch_k_14'] / 100 +
                (features['williams_r_14'] + 100) / 100
            ) / 3

            # トレンド・ボリューム合流
            features['trend_volume_confluence'] = features['adx_14'] * features['volume_ratio_10'] / 100

            # ボラティリティ・モメンタム
            features['volatility_momentum'] = features['volatility_20'] * abs(features['roc_10'])

            # 価格・モメンタム・ボリューム
            features['price_momentum_volume'] = features['roc_10'] * features['volume_ratio_10']

            # レジーム変化シグナル
            price_change = features['returns'].abs()
            vol_change = features['volatility_20'].pct_change().abs()
            price_threshold = price_change.rolling(20).quantile(0.95)
            vol_threshold = vol_change.rolling(20).quantile(0.95)
            features['regime_change'] = ((price_change > price_threshold) | (vol_change > vol_threshold)).astype(int)

        except Exception as e:
            self.logger.error(f"特徴量計算エラー: {e}")
            raise

        # NaN値処理
        features = features.fillna(method='ffill').fillna(method='bfill')

        # 無限大値処理
        features = features.replace([np.inf, -np.inf], 0)

        # 異常値除去（3シグマルール）
        for col in features.select_dtypes(include=[np.number]).columns:
            mean = features[col].mean()
            std = features[col].std()
            if std > 0:
                features[col] = features[col].clip(mean - 3*std, mean + 3*std)

        # 最初の60行を削除（指標計算のため）
        if len(features) > 60:
            features = features.iloc[60:].copy()

        self.logger.info(f"簡略化高度特徴量作成完了: {features.shape[1]}特徴量, {features.shape[0]}サンプル")
        return features

class SimplifiedAdvancedMLSystem:
    """簡略化高度機械学習システム"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = SimplifiedAdvancedFeatureEngineering()

        # データベース設定
        self.db_path = Path("ml_models_data/simplified_advanced_ml.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # 訓練されたモデル
        self.trained_models: Dict[str, Any] = {}

        self.logger.info("Simplified advanced ML system initialized")

    async def train_simplified_advanced_models(self, symbol: str, period: str = "6mo") -> Dict[ModelType, ModelPerformance]:
        """簡略化高度モデル訓練"""

        self.logger.info(f"簡略化高度モデル訓練開始: {symbol}")

        try:
            # データ取得
            from real_data_provider_v2 import real_data_provider
            data = await real_data_provider.get_stock_data(symbol, period)

            if data is None or len(data) < 100:
                raise ValueError("訓練に十分なデータがありません")

            # 特徴量エンジニアリング
            features = self.feature_engineering.create_advanced_features(data)

            # ターゲット作成（翌日の上昇/下降）
            targets = self._create_targets(data.iloc[60:]['Close'])  # 特徴量と同じ期間

            # データ同期
            min_length = min(len(features), len(targets))
            features = features.iloc[:min_length]
            targets = targets[:min_length]

            if len(features) < 30:
                raise ValueError("訓練データが不足しています")

            # 特徴選択（重要な特徴量のみ）
            features_selected = self._select_best_features(features, targets, k=60)

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

            self.logger.info(f"簡略化高度モデル訓練完了: {len(performances)}モデル")
            return performances

        except Exception as e:
            self.logger.error(f"簡略化高度モデル訓練エラー: {e}")
            raise

    def _create_targets(self, prices: pd.Series) -> np.ndarray:
        """ターゲット作成"""
        # 翌日の価格変動率
        returns = prices.pct_change().shift(-1)  # 翌日のリターン

        # 閾値以上の上昇を1、それ以外を0
        threshold = 0.003  # 0.3%以上の上昇
        targets = (returns > threshold).astype(int)

        return targets.values[:-1]  # 最後の要素（NaN）を除去

    def _select_best_features(self, features: pd.DataFrame, targets: np.ndarray, k: int = 60) -> pd.DataFrame:
        """最適特徴量選択"""

        k = min(k, features.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)

        features_selected = selector.fit_transform(features, targets)
        selected_columns = features.columns[selector.get_support()]

        return pd.DataFrame(features_selected, columns=selected_columns, index=features.index)

    async def _train_random_forest(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """Random Forest訓練"""

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        return await self._train_and_evaluate_model(model, features, targets, ModelType.RANDOM_FOREST, symbol)

    async def _train_gradient_boosting(self, features: pd.DataFrame, targets: np.ndarray, symbol: str) -> ModelPerformance:
        """Gradient Boosting訓練"""

        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
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
            max_iter=1000,
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

        # クロスバリデーション
        cv_scores = cross_val_score(model, features, targets, cv=3)

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
            if symbol in model_key and model_data['performance'].accuracy > 0.50:
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
                    CREATE TABLE IF NOT EXISTS simplified_advanced_performances (
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
                        INSERT OR REPLACE INTO simplified_advanced_performances
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

# グローバルインスタンス
simplified_advanced_ml_system = SimplifiedAdvancedMLSystem()

# テスト実行
async def test_simplified_advanced_system():
    """簡略化高度システムテスト"""

    print("=== 簡略化高度機械学習システムテスト ===")

    test_symbol = "7203"

    # モデル訓練
    print(f"\n🤖 {test_symbol} 簡略化高度モデル訓練開始...")
    performances = await simplified_advanced_ml_system.train_simplified_advanced_models(test_symbol, "4mo")

    print(f"\n📊 訓練結果:")
    for model_type, performance in performances.items():
        print(f"  {model_type.value}: 精度{performance.accuracy:.3f} F1{performance.f1_score:.3f} CV{performance.cross_val_score:.3f}")

    print(f"\n✅ 簡略化高度機械学習システムテスト完了")

if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # テスト実行
    asyncio.run(test_simplified_advanced_system())