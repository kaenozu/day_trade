#!/usr/bin/env python3
"""
Market regime detection module for Dynamic Weighting System

このモジュールは市場状態（強気相場、弱気相場、横ばい等）の検出と
管理を行います。リアルタイムでの市場状態変化を監視し、
重み調整アルゴリズムに市場状態情報を提供します。
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from collections import deque

from .core import MarketRegime, DynamicWeightingConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class MarketRegimeDetector:
    """
    市場状態検出器

    価格データの統計的特性を分析して市場状態を分類します。
    ボラティリティ、トレンド、価格変動パターンを基に判定を行います。
    """

    def __init__(self, config: DynamicWeightingConfig):
        """
        初期化

        Args:
            config: システム設定
        """
        self.config = config
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.market_indicators = deque(maxlen=50)

    def detect_market_regime(self, recent_actuals: deque) -> MarketRegime:
        """
        市場状態の検出

        直近の価格データから市場状態を分析・判定します。

        Args:
            recent_actuals: 直近の実際価格データ

        Returns:
            検出された市場状態
        """
        if len(recent_actuals) < 20:
            return self.current_regime

        try:
            # 直近のリターンを計算
            recent_values = np.array(list(recent_actuals)[-20:])
            returns = np.diff(recent_values) / recent_values[:-1]

            # 統計量計算
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            
            self.market_indicators.append({
                'mean_return': mean_return,
                'volatility': volatility,
                'timestamp': int(time.time())
            })

            # 市場状態判定ロジック
            new_regime = self._classify_regime(mean_return, volatility)

            # 市場状態変更の記録
            if new_regime != self.current_regime:
                self._record_regime_change(new_regime)
                self.current_regime = new_regime

                if self.config.verbose:
                    logger.info(f"市場状態変更: {self.current_regime.value}")

            return self.current_regime

        except Exception as e:
            logger.warning(f"市場状態検出エラー: {e}")
            return self.current_regime

    def _classify_regime(self, mean_return: float, volatility: float) -> MarketRegime:
        """
        統計量から市場状態を分類

        Args:
            mean_return: 平均リターン
            volatility: ボラティリティ（標準偏差）

        Returns:
            分類された市場状態
        """
        # ボラティリティ判定を優先
        if volatility > self.config.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.volatility_threshold * 0.5:
            return MarketRegime.LOW_VOLATILITY
        
        # トレンド判定
        elif mean_return > 0.001:
            return MarketRegime.BULL_MARKET
        elif mean_return < -0.001:
            return MarketRegime.BEAR_MARKET
        else:
            return MarketRegime.SIDEWAYS

    def _record_regime_change(self, new_regime: MarketRegime):
        """
        市場状態変更の記録

        Args:
            new_regime: 新しい市場状態
        """
        change_record = {
            'old_regime': self.current_regime,
            'new_regime': new_regime,
            'timestamp': int(time.time())
        }
        self.regime_history.append(change_record)

        # 履歴サイズ管理（最大100件）
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

    def get_current_regime(self) -> MarketRegime:
        """
        現在の市場状態を取得

        Returns:
            現在の市場状態
        """
        return self.current_regime

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """
        市場状態変更履歴を取得

        Returns:
            市場状態変更履歴のリスト
        """
        return self.regime_history.copy()

    def get_market_indicators(self) -> List[Dict[str, Any]]:
        """
        市場指標履歴を取得

        Returns:
            市場指標（リターン、ボラティリティ等）の履歴
        """
        return list(self.market_indicators)

    def get_regime_stability(self, window_size: int = 10) -> float:
        """
        市場状態の安定性を評価

        直近の状態変更頻度から安定性を計算します。

        Args:
            window_size: 評価ウィンドウサイズ

        Returns:
            安定性スコア（0.0-1.0、高いほど安定）
        """
        if len(self.regime_history) < 2:
            return 1.0  # 変更がない場合は完全安定

        # 直近のwindow_size期間の変更回数を計算
        recent_changes = self.regime_history[-window_size:]
        stability = max(0.0, 1.0 - len(recent_changes) / window_size)
        
        return stability

    def get_volatility_trend(self, window_size: int = 10) -> str:
        """
        ボラティリティトレンドを取得

        Args:
            window_size: 評価ウィンドウサイズ

        Returns:
            トレンド（"increasing", "decreasing", "stable"）
        """
        if len(self.market_indicators) < window_size:
            return "stable"

        recent_indicators = list(self.market_indicators)[-window_size:]
        volatilities = [ind['volatility'] for ind in recent_indicators]
        
        # 線形回帰でトレンドを判定
        x = np.arange(len(volatilities))
        slope = np.polyfit(x, volatilities, 1)[0]
        
        if slope > 0.001:
            return "increasing"
        elif slope < -0.001:
            return "decreasing"
        else:
            return "stable"

    def reset_detection_state(self):
        """
        検出状態をリセット

        システムの初期化や再起動時に使用します。
        """
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history.clear()
        self.market_indicators.clear()
        
        if self.config.verbose:
            logger.info("市場状態検出器をリセットしました")

    def get_detection_summary(self) -> Dict[str, Any]:
        """
        検出状況の要約を取得

        Returns:
            検出状況要約辞書
        """
        return {
            'current_regime': self.current_regime.value,
            'regime_changes': len(self.regime_history),
            'stability': self.get_regime_stability(),
            'volatility_trend': self.get_volatility_trend(),
            'indicators_count': len(self.market_indicators),
            'last_change': (self.regime_history[-1]['timestamp'] 
                          if self.regime_history else None)
        }