#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Prediction
機械学習予測機能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """機械学習予測クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ml_models = {}
        self.scalers = {}

    async def generate_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """機械学習予測生成"""
        try:
            if len(df) < 50:
                return None

            # 特徴量作成
            features = self._create_ml_features(df)

            if features is None or len(features) < 20:
                return None

            # 簡易予測（実装例）
            returns = df['Close'].pct_change().dropna()
            recent_trend = returns.rolling(10).mean().iloc[-1]
            trend_strength = abs(recent_trend)

            # 方向性予測
            direction = "上昇" if recent_trend > 0.001 else "下落" if recent_trend < -0.001 else "横ばい"
            confidence = min(90, trend_strength * 1000 + 50)

            return {
                'direction': direction,
                'confidence': confidence,
                'expected_return': recent_trend * 100,
                'risk_level': "高" if trend_strength > 0.02 else "中" if trend_strength > 0.01 else "低",
                'model_type': 'trend_based',
                'features_used': len(features)
            }

        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return None

    def _create_ml_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """機械学習用特徴量作成"""
        try:
            features = pd.DataFrame(index=df.index)

            # 価格特徴量
            features['return_1d'] = df['Close'].pct_change()
            features['return_5d'] = df['Close'].pct_change(5)
            features['return_20d'] = df['Close'].pct_change(20)

            # 移動平均特徴量
            features['ma_5_ratio'] = df['Close'] / df['Close'].rolling(5).mean()
            features['ma_20_ratio'] = df['Close'] / df['Close'].rolling(20).mean()

            # ボラティリティ特徴量
            features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
            features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()

            # 出来高特徴量
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            return features.dropna()

        except Exception as e:
            self.logger.error(f"Feature creation error: {e}")
            return None

    def perform_clustering_analysis(self, features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """クラスタリング分析"""
        try:
            if len(features) < 10:
                return None

            # データの正規化
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features.fillna(0))

            # K-means クラスタリング
            n_clusters = min(5, len(features) // 10)
            if n_clusters < 2:
                return None

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)

            # PCA による次元削減
            pca = PCA(n_components=min(3, features.shape[1]))
            pca_features = pca.fit_transform(scaled_features)

            # 異常検出
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = isolation_forest.fit_predict(scaled_features)

            return {
                'n_clusters': n_clusters,
                'current_cluster': int(clusters[-1]),
                'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
                'is_anomaly': bool(anomalies[-1] == -1),
                'cluster_distribution': {
                    int(i): int(np.sum(clusters == i)) for i in range(n_clusters)
                }
            }

        except Exception as e:
            self.logger.error(f"Clustering analysis error: {e}")
            return None

    def calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """特徴量重要度計算（簡易版）"""
        try:
            importance_scores = {}

            for col in features.columns:
                # 相関係数による簡易重要度
                correlation = abs(features[col].corr(target))
                importance_scores[col] = float(correlation if not np.isnan(correlation) else 0.0)

            # 正規化
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    k: v / total_importance for k, v in importance_scores.items()
                }

            return importance_scores

        except Exception as e:
            self.logger.error(f"Feature importance calculation error: {e}")
            return {}

    def generate_ensemble_prediction(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """アンサンブル予測生成"""
        try:
            if not predictions:
                return {}

            # 信頼度加重平均
            total_confidence = sum(pred.get('confidence', 0) for pred in predictions)
            if total_confidence == 0:
                return {}

            # 方向性の投票
            directions = [pred.get('direction', 'neutral') for pred in predictions]
            direction_counts = {
                '上昇': directions.count('上昇'),
                '下落': directions.count('下落'),
                '横ばい': directions.count('横ばい')
            }

            final_direction = max(direction_counts, key=direction_counts.get)

            # 加重平均信頼度
            weighted_confidence = sum(
                pred.get('confidence', 0) * (1 if pred.get('direction', '') == final_direction else 0.5)
                for pred in predictions
            ) / len(predictions)

            # 期待リターンの加重平均
            weighted_return = sum(
                pred.get('expected_return', 0) * pred.get('confidence', 0)
                for pred in predictions
            ) / total_confidence

            return {
                'final_direction': final_direction,
                'ensemble_confidence': min(100, weighted_confidence),
                'expected_return': weighted_return,
                'model_consensus': direction_counts[final_direction] / len(predictions) * 100,
                'num_models': len(predictions)
            }

        except Exception as e:
            self.logger.error(f"Ensemble prediction error: {e}")
            return {}