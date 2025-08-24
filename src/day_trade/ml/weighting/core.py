#!/usr/bin/env python3
"""
Core data structures and configurations for Dynamic Weighting System

This module contains fundamental data classes, enums, and configuration
classes used throughout the dynamic weighting system.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class MarketRegime(Enum):
    """市場状態を表す列挙型"""
    BULL_MARKET = "bull"         # 強気相場
    BEAR_MARKET = "bear"         # 弱気相場
    SIDEWAYS = "sideways"        # 横ばい
    HIGH_VOLATILITY = "high_vol" # 高ボラティリティ
    LOW_VOLATILITY = "low_vol"   # 低ボラティリティ


@dataclass
class PerformanceWindow:
    """
    パフォーマンス評価ウィンドウ

    特定期間のモデル予測値と実際値を格納し、
    各種メトリクスを計算する機能を提供します。

    Attributes:
        predictions: 予測値配列
        actuals: 実際値配列
        timestamps: タイムスタンプリスト
        market_regime: 市場状態（オプション）
    """
    predictions: np.ndarray
    actuals: np.ndarray
    timestamps: List[int]
    market_regime: Optional[MarketRegime] = None

    def calculate_metrics(self) -> Dict[str, float]:
        """
        パフォーマンスメトリクスを計算

        Returns:
            メトリクス辞書（RMSE、MAE、方向的中率、サンプル数）
        """
        if len(self.predictions) == 0:
            return {}

        # 基本メトリクス計算
        mse = np.mean((self.actuals - self.predictions) ** 2)
        mae = np.mean(np.abs(self.actuals - self.predictions))

        # 方向的中率計算
        if len(self.predictions) > 1:
            actual_diff = np.diff(self.actuals)
            pred_diff = np.diff(self.predictions)
            hit_rate = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
        else:
            hit_rate = 0.5  # 単一データ点の場合はニュートラル

        return {
            'rmse': np.sqrt(mse),
            'mae': mae,
            'hit_rate': hit_rate,
            'sample_count': len(self.predictions)
        }


@dataclass
class DynamicWeightingConfig:
    """
    動的重み調整システムの設定クラス

    システムの動作を制御する各種パラメータを管理します。
    設定項目は大きく以下のカテゴリに分類されます：
    - パフォーマンス評価設定
    - 重み調整アルゴリズム設定  
    - 市場状態適応設定
    - コンセプトドリフト検出設定
    - リスク管理設定
    """
    
    # パフォーマンス評価設定
    window_size: int = 100              # 評価ウィンドウサイズ
    min_samples_for_update: int = 50    # 重み更新最小サンプル数
    update_frequency: int = 20          # 更新頻度（サンプル数）

    # 重み調整アルゴリズム設定
    weighting_method: str = "performance_based"  # 重み調整手法
    decay_factor: float = 0.95          # 過去データの減衰率
    momentum_factor: float = 0.1        # モーメンタム要素

    # 市場状態適応設定
    enable_regime_detection: bool = True
    regime_sensitivity: float = 0.3      # 市場状態変化への感度
    volatility_threshold: float = 0.02   # ボラティリティ閾値

    # Issue #478対応: レジーム調整設定の外部化
    regime_adjustments: Optional[Dict[MarketRegime, Dict[str, float]]] = None

    # Issue #477対応: スコアリング明確化・カスタマイズ
    sharpe_clip_min: float = 0.1        # シャープレシオ下限クリップ値
    accuracy_weight: float = 1.0        # 精度スコア重み係数
    direction_weight: float = 1.0       # 方向スコア重み係数
    enable_score_logging: bool = False  # スコア詳細ログ出力

    # コンセプトドリフト検出設定
    enable_concept_drift_detection: bool = False
    drift_detection_metric: str = "rmse"
    drift_detection_threshold: float = 0.1
    drift_detection_window_size: int = 50

    # リスク管理設定
    max_weight_change: float = 0.1      # 1回の最大重み変更
    min_weight: float = 0.05            # 最小重み
    max_weight: float = 0.6             # 最大重み

    # システム設定
    verbose: bool = True                # 詳細ログ出力

    def validate_config(self) -> bool:
        """
        設定の妥当性を検証

        Returns:
            妥当性検証結果（True: 有効、False: 無効）
        """
        try:
            # 基本範囲チェック
            if self.window_size <= 0 or self.min_samples_for_update <= 0:
                return False
            if self.update_frequency <= 0:
                return False
            if self.decay_factor < 0 or self.decay_factor > 1:
                return False
            if self.momentum_factor < 0 or self.momentum_factor > 1:
                return False
            if self.regime_sensitivity < 0 or self.regime_sensitivity > 1:
                return False
            if self.volatility_threshold <= 0:
                return False
            
            # スコアリング設定チェック
            if self.sharpe_clip_min < 0:
                return False
            if self.accuracy_weight < 0 or self.direction_weight < 0:
                return False
            
            # コンセプトドリフト設定チェック
            if self.drift_detection_threshold <= 0:
                return False
            if self.drift_detection_window_size <= 0:
                return False
            
            # リスク管理設定チェック
            if self.max_weight_change <= 0 or self.max_weight_change > 1:
                return False
            if self.min_weight < 0 or self.min_weight > 1:
                return False
            if self.max_weight < 0 or self.max_weight > 1:
                return False
            if self.min_weight >= self.max_weight:
                return False
            
            # 重み調整手法チェック
            valid_methods = ["performance_based", "sharpe_based", "regime_aware"]
            if self.weighting_method not in valid_methods:
                return False
            
            return True
            
        except Exception:
            return False

    def get_config_summary(self) -> Dict[str, Any]:
        """
        設定の要約を取得

        Returns:
            設定要約辞書
        """
        return {
            'weighting_method': self.weighting_method,
            'window_size': self.window_size,
            'update_frequency': self.update_frequency,
            'min_samples': self.min_samples_for_update,
            'momentum_factor': self.momentum_factor,
            'regime_detection': self.enable_regime_detection,
            'drift_detection': self.enable_concept_drift_detection,
            'weight_constraints': {
                'min': self.min_weight,
                'max': self.max_weight,
                'max_change': self.max_weight_change
            },
            'scoring_config': {
                'accuracy_weight': self.accuracy_weight,
                'direction_weight': self.direction_weight,
                'sharpe_clip_min': self.sharpe_clip_min
            }
        }


@dataclass
class WeightingState:
    """
    動的重み調整システムの内部状態

    システムの現在状態を管理するデータクラス。
    重み履歴、パフォーマンスデータ、市場状態等を保持します。
    """
    
    # 重み関連
    current_weights: Dict[str, float] = field(default_factory=dict)
    weight_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # パフォーマンス履歴
    performance_windows: Dict[str, deque] = field(default_factory=dict)
    recent_predictions: Dict[str, deque] = field(default_factory=dict)
    recent_actuals: deque = field(default_factory=lambda: deque())
    recent_timestamps: deque = field(default_factory=lambda: deque())
    
    # 市場状態
    current_regime: MarketRegime = MarketRegime.SIDEWAYS
    regime_history: List[Dict[str, Any]] = field(default_factory=list)
    market_indicators: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # カウンタ
    update_counter: int = 0
    total_updates: int = 0
    
    # フラグ
    re_evaluation_needed: bool = False
    drift_detection_updates_count: int = 0

    def reset_counters(self):
        """カウンタをリセット"""
        self.update_counter = 0
        self.total_updates = 0
        self.drift_detection_updates_count = 0

    def get_state_summary(self) -> Dict[str, Any]:
        """
        現在状態の要約を取得

        Returns:
            状態要約辞書
        """
        return {
            'current_weights': self.current_weights,
            'current_regime': self.current_regime.value,
            'total_updates': self.total_updates,
            'data_points': len(self.recent_actuals),
            'models': list(self.current_weights.keys()),
            'regime_changes': len(self.regime_history),
            're_evaluation_needed': self.re_evaluation_needed
        }