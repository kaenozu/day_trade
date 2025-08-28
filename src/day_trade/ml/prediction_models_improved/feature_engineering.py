#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特徴量エンジニアリング - ML Prediction Models Feature Engineering

ML予測モデルの特徴量エンジニアリング機能を提供します。
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.day_trade.ml.core_types import PredictionTask

# Feature Engineering availability check
try:
    from src.day_trade.analysis.enhanced_feature_engineering import enhanced_feature_engineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def engineer_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング（強化版）"""
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                # 既存の特徴量エンジニアリングシステムを使用
                feature_set = await enhanced_feature_engineer.extract_comprehensive_features(symbol, data)

                if hasattr(feature_set, 'to_dataframe'):
                    features = feature_set.to_dataframe()
                else:
                    features = self._convert_featureset_to_dataframe(feature_set, data)

                # 品質チェック
                if features.empty or len(features.columns) < 5:
                    self.logger.warning("高度特徴量エンジニアリング結果不十分、基本特徴量を使用")
                    features = self._extract_basic_features(data)
                else:
                    self.logger.info(f"高度特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            else:
                # 基本特徴量のみ
                features = self._extract_basic_features(data)
                self.logger.info(f"基本特徴量エンジニアリング完了: {len(features.columns)}特徴量")

            return features

        except Exception as e:
            self.logger.error(f"特徴量エンジニアリングエラー: {e}")
            # フォールバック
            return self._extract_basic_features(data)

    def _extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量抽出（改良版）"""
        features = pd.DataFrame(index=data.index)

        try:
            # 価格系特徴量
            features['returns'] = data['Close'].pct_change()
            features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            features['price_range'] = (data['High'] - data['Low']) / data['Close']
            features['body_size'] = abs(data['Close'] - data['Open']) / data['Close']
            features['upper_shadow'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / data['Close']
            features['lower_shadow'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / data['Close']

            # 移動平均とその比率
            for window in [5, 10, 20, 50]:
                sma = data['Close'].rolling(window).mean()
                features[f'sma_{window}'] = sma
                features[f'sma_ratio_{window}'] = data['Close'] / sma
                features[f'sma_slope_{window}'] = sma.diff() / sma.shift(1)

            # ボラティリティ
            for window in [5, 10, 20]:
                features[f'volatility_{window}'] = features['returns'].rolling(window).std()
                features[f'volatility_ratio_{window}'] = (features[f'volatility_{window}'] /
                                                         features[f'volatility_{window}'].rolling(window*2).mean())

            # 出来高特徴量
            if 'Volume' in data.columns:
                features['volume_ma_20'] = data['Volume'].rolling(20).mean()
                features['volume_ratio'] = data['Volume'] / features['volume_ma_20']
                features['volume_price_trend'] = (data['Volume'] * features['returns']).rolling(5).mean()

            # テクニカル指標
            features = self._add_technical_indicators(features, data)

            # 欠損値処理
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features

        except Exception as e:
            self.logger.error(f"基本特徴量抽出エラー: {e}")
            # 最小限の特徴量
            min_features = pd.DataFrame(index=data.index)
            min_features['returns'] = data['Close'].pct_change().fillna(0)
            min_features['sma_20'] = data['Close'].rolling(20).mean().fillna(method='ffill').fillna(data['Close'])
            return min_features

    def _add_technical_indicators(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標追加"""
        try:
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            features['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # ボリンジャーバンド
            sma_20 = data['Close'].rolling(20).mean()
            std_20 = data['Close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

            # ストキャスティクス
            low_min = data['Low'].rolling(14).min()
            high_max = data['High'].rolling(14).max()
            features['stoch_k'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
            features['stoch_d'] = features['stoch_k'].rolling(3).mean()

            # ATR（Average True Range）
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr'] = true_range.rolling(14).mean()

            # CCI（Commodity Channel Index）
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(20).mean()
            mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            features['cci'] = (typical_price - sma_tp) / (0.015 * mad)

            return features

        except Exception as e:
            self.logger.warning(f"テクニカル指標追加エラー: {e}")
            return features

    def _convert_featureset_to_dataframe(
        self, 
        feature_set, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """FeatureSetをDataFrameに変換（改良版）"""
        try:
            if isinstance(feature_set, list):
                # 時系列FeatureSetリストの場合
                feature_rows = []
                timestamps = []

                for fs in feature_set:
                    row_features = {}
                    # 各カテゴリの特徴量を統合
                    for category in ['price_features', 'technical_features', 'volume_features',
                                   'momentum_features', 'volatility_features', 'pattern_features',
                                   'market_features', 'statistical_features']:
                        if hasattr(fs, category):
                            features_dict = getattr(fs, category)
                            if isinstance(features_dict, dict):
                                # プレフィックスを追加して名前衝突を回避
                                prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                                row_features.update(prefixed_features)

                    if row_features:
                        feature_rows.append(row_features)
                        timestamps.append(getattr(fs, 'timestamp', data.index[len(timestamps)]))

                if feature_rows:
                    features_df = pd.DataFrame(feature_rows, index=timestamps[:len(feature_rows)])
                else:
                    features_df = self._extract_basic_features(data)

            else:
                # 単一FeatureSetの場合
                all_features = {}
                for category in ['price_features', 'technical_features', 'volume_features',
                               'momentum_features', 'volatility_features', 'pattern_features',
                               'market_features', 'statistical_features']:
                    if hasattr(feature_set, category):
                        features_dict = getattr(feature_set, category)
                        if isinstance(features_dict, dict):
                            prefixed_features = {f"{category}_{k}": v for k, v in features_dict.items()}
                            all_features.update(prefixed_features)

                if all_features:
                    timestamp = getattr(feature_set, 'timestamp', data.index[-1])
                    features_df = pd.DataFrame([all_features], index=[timestamp])
                else:
                    features_df = self._extract_basic_features(data)

            # 欠損値処理
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            return features_df

        except Exception as e:
            self.logger.error(f"FeatureSet変換エラー: {e}")
            return self._extract_basic_features(data)

    def create_target_variables(self, data: pd.DataFrame) -> dict[PredictionTask, pd.Series]:
        """ターゲット変数作成（改良版）"""
        targets = {}

        try:
            # 価格方向予測（翌日の価格変動方向）
            returns = data['Close'].pct_change().shift(-1)  # 翌日のリターン

            # 閾値を動的に調整（ボラティリティベース）
            volatility = returns.rolling(20).std().fillna(returns.std())
            threshold = volatility * 0.5  # ボラティリティの半分を閾値とする

            direction = pd.Series(index=data.index, dtype='int')
            direction[returns > threshold] = 1   # 上昇
            direction[returns < -threshold] = -1  # 下落
            direction[(returns >= -threshold) & (returns <= threshold)] = 0  # 横ばい

            targets[PredictionTask.PRICE_DIRECTION] = direction

            # 価格回帰予測（翌日の終値）
            targets[PredictionTask.PRICE_REGRESSION] = data['Close'].shift(-1)

            # ボラティリティ予測（翌日の変動率）
            high_low_range = (data['High'] - data['Low']) / data['Close']
            targets[PredictionTask.VOLATILITY] = high_low_range.shift(-1)

            # トレンド強度予測
            trend_strength = abs(returns) / volatility
            targets[PredictionTask.TREND_STRENGTH] = trend_strength.shift(-1)

            return targets

        except Exception as e:
            self.logger.error(f"ターゲット変数作成エラー: {e}")
            # 最小限のターゲット
            simple_direction = pd.Series(0, index=data.index)
            simple_direction[data['Close'].pct_change().shift(-1) > 0] = 1
            simple_direction[data['Close'].pct_change().shift(-1) < 0] = -1
            return {PredictionTask.PRICE_DIRECTION: simple_direction}

    def prepare_prediction_features(
        self, 
        symbol: str, 
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """予測用特徴量準備（同期版）"""
        try:
            self.logger.debug(f"予測用特徴量準備開始: {symbol}")

            # 基本特徴量抽出
            features = self._extract_basic_features(data)

            self.logger.debug(f"予測用特徴量準備完了: {features.shape}")
            return features

        except Exception as e:
            self.logger.error(f"予測用特徴量準備エラー: {e}")
            raise

    def validate_features(self, features: pd.DataFrame) -> tuple[bool, str]:
        """特徴量検証"""
        try:
            if features.empty:
                return False, "特徴量データが空です"

            if features.isnull().all().any():
                return False, "全て欠損値の特徴量があります"

            if len(features.columns) < 3:
                return False, f"特徴量数が不十分です: {len(features.columns)}個"

            # 無限値チェック
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                return False, f"無限値が含まれています: {inf_count}個"

            return True, "特徴量検証成功"

        except Exception as e:
            return False, f"特徴量検証エラー: {e}"

    def get_feature_importance_ranking(self, feature_importance: dict[str, float]) -> list[tuple[str, float]]:
        """特徴量重要度ランキング取得"""
        try:
            # 重要度順にソート
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return sorted_features

        except Exception as e:
            self.logger.error(f"特徴量重要度ランキング取得エラー: {e}")
            return []

    def analyze_feature_correlation(self, features: pd.DataFrame) -> pd.DataFrame:
        """特徴量相関分析"""
        try:
            # 数値列のみ選択
            numeric_features = features.select_dtypes(include=[np.number])
            
            # 相関マトリックス計算
            correlation_matrix = numeric_features.corr()
            
            return correlation_matrix

        except Exception as e:
            self.logger.error(f"特徴量相関分析エラー: {e}")
            return pd.DataFrame()

    def remove_highly_correlated_features(
        self, 
        features: pd.DataFrame, 
        threshold: float = 0.95
    ) -> pd.DataFrame:
        """高相関特徴量除去"""
        try:
            # 相関マトリックス取得
            corr_matrix = self.analyze_feature_correlation(features)
            
            if corr_matrix.empty:
                return features

            # 高相関特徴量を特定
            high_corr_features = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > threshold:
                        # より重要でない方の特徴量を除去対象にする
                        # ここでは後者を除去（改善の余地あり）
                        high_corr_features.add(corr_matrix.columns[j])

            # 高相関特徴量を除去
            features_filtered = features.drop(columns=list(high_corr_features))
            
            self.logger.info(f"高相関特徴量除去: {len(high_corr_features)}個の特徴量を除去")
            
            return features_filtered

        except Exception as e:
            self.logger.error(f"高相関特徴量除去エラー: {e}")
            return features