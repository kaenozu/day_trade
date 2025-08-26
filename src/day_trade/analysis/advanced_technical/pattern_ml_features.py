#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Technical Pattern Recognition and ML Features - パターン認識・機械学習関連モジュール
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from scipy.signal import find_peaks


class PatternMLProcessor:
    """パターン認識・機械学習処理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def generate_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        機械学習予測生成
        
        Args:
            symbol: 銘柄コード
            df: 価格データ DataFrame
            
        Returns:
            ML予測結果辞書
        """
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
            direction = (
                "上昇" if recent_trend > 0.001 else 
                "下落" if recent_trend < -0.001 else 
                "横ばい"
            )
            confidence = min(90, trend_strength * 1000 + 50)

            return {
                'direction': direction,
                'confidence': confidence,
                'expected_return': recent_trend * 100,
                'risk_level': (
                    "高" if trend_strength > 0.02 else 
                    "中" if trend_strength > 0.01 else 
                    "低"
                ),
                'model_type': 'trend_based',
                'features_used': len(features)
            }

        except Exception as e:
            self.logger.error(f"ML prediction error: {e}")
            return None
    
    def _create_ml_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        機械学習用特徴量作成
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            特徴量 DataFrame
        """
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
    
    def perform_pattern_recognition(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        パターン認識実行
        
        Args:
            df: 価格データ DataFrame
            
        Returns:
            パターン認識結果辞書
        """
        try:
            closes = df['Close']

            # サポート・レジスタンスレベル検出
            support_levels, resistance_levels = self._find_support_resistance(closes)

            # チャートパターン検出（簡易版）
            pattern = self._detect_chart_patterns(closes)

            return {
                'support_levels': support_levels[-3:] if support_levels else [],  # 直近3レベル
                'resistance_levels': resistance_levels[-3:] if resistance_levels else [],
                'detected_pattern': pattern,
                'current_position': (
                    'above_support' if support_levels and closes.iloc[-1] > max(support_levels) 
                    else 'neutral'
                )
            }

        except Exception as e:
            self.logger.error(f"Pattern recognition error: {e}")
            return None
    
    def _find_support_resistance(self, prices: pd.Series) -> Tuple[List[float], List[float]]:
        """
        サポート・レジスタンスレベル検出
        
        Args:
            prices: 価格系列
            
        Returns:
            (サポートレベルリスト, レジスタンスレベルリスト)
        """
        try:
            # ピーク・ボトム検出
            peaks, _ = find_peaks(prices, distance=5)
            troughs, _ = find_peaks(-prices, distance=5)

            # レジスタンス（ピーク）
            resistance_levels = [float(prices.iloc[i]) for i in peaks[-5:]]

            # サポート（ボトム）
            support_levels = [float(prices.iloc[i]) for i in troughs[-5:]]

            return support_levels, resistance_levels

        except Exception as e:
            self.logger.error(f"Support/resistance detection error: {e}")
            return [], []
    
    def _detect_chart_patterns(self, prices: pd.Series) -> str:
        """
        チャートパターン検出（簡易版）
        
        Args:
            prices: 価格系列
            
        Returns:
            検出パターン名
        """
        try:
            if len(prices) < 20:
                return "insufficient_data"

            recent_prices = prices.iloc[-20:]

            # 上昇トレンド
            if recent_prices.iloc[-1] > recent_prices.iloc[0] * 1.05:
                return "uptrend"

            # 下降トレンド
            elif recent_prices.iloc[-1] < recent_prices.iloc[0] * 0.95:
                return "downtrend"

            # レンジ相場
            elif recent_prices.std() / recent_prices.mean() < 0.02:
                return "sideways"

            else:
                return "unclear"

        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
            return "error"