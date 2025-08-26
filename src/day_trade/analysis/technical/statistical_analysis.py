#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis and Pattern Recognition
統計分析とパターン認識
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.signal import find_peaks


class StatisticalAnalysis:
    """統計分析・パターン認識クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.statistical_lookback = 252  # 1年間
        self.pattern_configs = {
            "support_resistance_strength": 3,
            "pattern_min_length": 10,
            "breakout_threshold": 0.02
        }

    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, float]:
        """統計分析実行"""
        try:
            closes = df['Close']
            returns = closes.pct_change().dropna()

            return {
                'mean_return': float(returns.mean() * 252),  # 年率化
                'volatility': float(returns.std() * np.sqrt(252)),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)),
                'max_drawdown': self._calculate_max_drawdown(closes),
                'var_95': float(np.percentile(returns, 5) * 100),
                'autocorrelation': float(returns.autocorr(lag=1))
            }

        except Exception as e:
            self.logger.error(f"Statistical analysis error: {e}")
            return {}

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            cumulative = (1 + prices.pct_change()).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min() * 100)
        except:
            return 0.0

    def calculate_anomaly_score(self, df: pd.DataFrame) -> float:
        """異常度スコア計算"""
        try:
            # 価格変動率の異常度
            returns = df['Close'].pct_change().dropna()
            if len(returns) < 10:
                return 0.0

            # Z-score計算
            z_score = abs((returns.iloc[-1] - returns.mean()) / returns.std())

            # 0-100スケールに変換
            anomaly_score = min(100, z_score * 20)

            return float(anomaly_score)

        except Exception as e:
            self.logger.error(f"Anomaly score calculation error: {e}")
            return 0.0

    def perform_pattern_recognition(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """パターン認識実行"""
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
                'current_position': 'above_support' if (
                    support_levels and closes.iloc[-1] > max(support_levels)
                ) else 'neutral'
            }

        except Exception as e:
            self.logger.error(f"Pattern recognition error: {e}")
            return None

    def _find_support_resistance(self, prices: pd.Series) -> Tuple[List[float], List[float]]:
        """サポート・レジスタンスレベル検出"""
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
        """チャートパターン検出（簡易版）"""
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