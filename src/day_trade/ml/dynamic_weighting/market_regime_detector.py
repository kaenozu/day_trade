#!/usr/bin/env python3
"""
Dynamic Weighting System - Market Regime Detector

市場状態検出と履歴管理
"""

import time
from typing import List, Dict, Any
import numpy as np
from collections import deque

from .core import DynamicWeightingConfig, MarketRegime
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketRegimeDetector:
    """市場状態検出器クラス"""

    def __init__(self, config: DynamicWeightingConfig):
        """
        初期化

        Args:
            config: 動的重み調整設定
        """
        self.config = config
        
        # 市場状態関連
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.market_indicators = deque(maxlen=50)

    def detect_market_regime(self, recent_actuals: deque) -> MarketRegime:
        """
        市場状態検出

        Args:
            recent_actuals: 最近の実際値のdeque

        Returns:
            検出された市場状態
        """
        if not self.config.enable_regime_detection or len(recent_actuals) < 20:
            return self.current_regime

        try:
            # 直近のリターンを計算
            recent_values = np.array(list(recent_actuals)[-20:])
            returns = np.diff(recent_values) / recent_values[:-1]

            # 統計量計算
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            # 市場指標を記録
            self.market_indicators.append({
                'mean_return': mean_return,
                'volatility': volatility,
                'timestamp': int(time.time())
            })

            # 市場状態判定
            new_regime = self._classify_market_regime(mean_return, volatility)

            # 市場状態変更
            if new_regime != self.current_regime:
                self._update_regime(new_regime)

            return self.current_regime

        except Exception as e:
            logger.warning(f"市場状態検出エラー: {e}")
            return self.current_regime

    def _classify_market_regime(self, mean_return: float, volatility: float) -> MarketRegime:
        """
        市場状態分類

        Args:
            mean_return: 平均リターン
            volatility: ボラティリティ

        Returns:
            分類された市場状態
        """
        # ボラティリティベースの判定を優先
        if volatility > self.config.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.volatility_threshold * 0.5:
            return MarketRegime.LOW_VOLATILITY
        
        # トレンドベースの判定
        elif mean_return > 0.001:
            return MarketRegime.BULL_MARKET
        elif mean_return < -0.001:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS

    def _update_regime(self, new_regime: MarketRegime):
        """
        市場状態更新

        Args:
            new_regime: 新しい市場状態
        """
        # 履歴記録
        self.regime_history.append({
            'old_regime': self.current_regime,
            'new_regime': new_regime,
            'timestamp': int(time.time())
        })
        
        # 現在の状態を更新
        old_regime = self.current_regime
        self.current_regime = new_regime

        if self.config.verbose:
            logger.info(f"市場状態変更: {old_regime.value} → {new_regime.value}")

    def get_current_regime(self) -> MarketRegime:
        """現在の市場状態取得"""
        return self.current_regime

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """市場状態履歴取得"""
        return self.regime_history.copy()

    def get_market_indicators(self) -> List[Dict[str, Any]]:
        """市場指標履歴取得"""
        return list(self.market_indicators)

    def force_regime_change(self, new_regime: MarketRegime):
        """
        強制的な市場状態変更（テスト用）

        Args:
            new_regime: 設定する市場状態
        """
        if new_regime != self.current_regime:
            self._update_regime(new_regime)
            logger.info(f"市場状態を強制変更: {new_regime.value}")

    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        市場状態統計情報取得

        Returns:
            統計情報辞書
        """
        stats = {
            'current_regime': self.current_regime.value,
            'total_regime_changes': len(self.regime_history),
            'regime_distribution': {},
            'avg_regime_duration': 0.0,
            'latest_indicators': {}
        }

        # 市場状態の分布計算
        if self.regime_history:
            regime_counts = {}
            for change in self.regime_history:
                old_regime = change['old_regime'].value
                regime_counts[old_regime] = regime_counts.get(old_regime, 0) + 1
            
            total_changes = len(self.regime_history)
            stats['regime_distribution'] = {
                regime: count / total_changes for regime, count in regime_counts.items()
            }

            # 平均持続時間計算（概算）
            if len(self.regime_history) > 1:
                durations = []
                for i in range(1, len(self.regime_history)):
                    duration = self.regime_history[i]['timestamp'] - self.regime_history[i-1]['timestamp']
                    durations.append(duration)
                stats['avg_regime_duration'] = np.mean(durations) if durations else 0.0

        # 最新の市場指標
        if self.market_indicators:
            latest_indicator = list(self.market_indicators)[-1]
            stats['latest_indicators'] = {
                'mean_return': latest_indicator['mean_return'],
                'volatility': latest_indicator['volatility'],
                'volatility_threshold': self.config.volatility_threshold
            }

        return stats

    def reset_regime_history(self):
        """市場状態履歴をリセット"""
        self.regime_history.clear()
        self.market_indicators.clear()
        logger.info("市場状態履歴をリセットしました")