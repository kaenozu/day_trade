#!/usr/bin/env python3
"""
Dynamic Weighting System - Core Components

基本的なデータクラスとコンフィグレーション定義を含むコアモジュール
"""

import time
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class MarketRegime(Enum):
    """市場状態"""
    BULL_MARKET = "bull"      # 強気相場
    BEAR_MARKET = "bear"      # 弱気相場
    SIDEWAYS = "sideways"     # 横ばい
    HIGH_VOLATILITY = "high_vol"  # 高ボラティリティ
    LOW_VOLATILITY = "low_vol"    # 低ボラティリティ


@dataclass
class PerformanceWindow:
    """パフォーマンス評価ウィンドウ"""
    predictions: np.ndarray
    actuals: np.ndarray
    timestamps: List[int]
    market_regime: Optional[MarketRegime] = None

    def calculate_metrics(self) -> Dict[str, float]:
        """メトリクス計算"""
        if len(self.predictions) == 0:
            return {}

        # 基本メトリクス
        mse = np.mean((self.actuals - self.predictions) ** 2)
        mae = np.mean(np.abs(self.actuals - self.predictions))

        # 方向的中率
        if len(self.predictions) > 1:
            actual_diff = np.diff(self.actuals)
            pred_diff = np.diff(self.predictions)
            hit_rate = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
        else:
            hit_rate = 0.5

        return {
            'rmse': np.sqrt(mse),
            'mae': mae,
            'hit_rate': hit_rate,
            'sample_count': len(self.predictions)
        }


@dataclass
class DynamicWeightingConfig:
    """動的重み調整設定"""
    # パフォーマンス評価
    window_size: int = 100           # 評価ウィンドウサイズ
    min_samples_for_update: int = 50  # 重み更新最小サンプル数
    update_frequency: int = 20        # 更新頻度（サンプル数）

    # 重み調整アルゴリズム
    weighting_method: str = "performance_based"  # performance_based, sharpe_based, regime_aware
    decay_factor: float = 0.95       # 過去データの減衰率
    momentum_factor: float = 0.1     # モーメンタム要素

    # 市場状態適応
    enable_regime_detection: bool = True
    regime_sensitivity: float = 0.3   # 市場状態変化への感度
    volatility_threshold: float = 0.02  # ボラティリティ閾値

    # Issue #478対応: レジーム認識調整外部化
    regime_adjustments: Optional[Dict[MarketRegime, Dict[str, float]]] = None

    # Issue #477対応: スコアリング明確化・カスタマイズ
    sharpe_clip_min: float = 0.1      # シャープレシオ下限クリップ値
    accuracy_weight: float = 1.0      # 精度スコア重み係数
    direction_weight: float = 1.0     # 方向スコア重み係数
    enable_score_logging: bool = False # スコア詳細ログ出力

    # コンセプトドリフト検出
    enable_concept_drift_detection: bool = False
    drift_detection_metric: str = "rmse"
    drift_detection_threshold: float = 0.1
    drift_detection_window_size: int = 50

    # リスク管理
    max_weight_change: float = 0.1    # 1回の最大重み変更
    min_weight: float = 0.05          # 最小重み
    max_weight: float = 0.6           # 最大重み

    # パフォーマンス設定
    verbose: bool = True


def get_default_regime_adjustments(model_names: List[str]) -> Dict[MarketRegime, Dict[str, float]]:
    """
    Issue #478対応: デフォルトレジーム調整設定取得

    Args:
        model_names: モデル名のリスト

    Returns:
        市場状態別の調整係数辞書
    """
    # 一般的なモデル名のデフォルト設定
    default_adjustments = {
        MarketRegime.BULL_MARKET: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9},
        MarketRegime.BEAR_MARKET: {'svr': 1.2, 'gradient_boosting': 1.1, 'random_forest': 0.9},
        MarketRegime.SIDEWAYS: {'gradient_boosting': 1.1, 'random_forest': 1.0, 'svr': 1.0},
        MarketRegime.HIGH_VOLATILITY: {'svr': 1.3, 'gradient_boosting': 0.9, 'random_forest': 0.8},
        MarketRegime.LOW_VOLATILITY: {'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9}
    }

    # 実際のモデル名に基づいて調整
    model_adjusted = {}
    for regime, adjustments in default_adjustments.items():
        model_adjusted[regime] = {}
        for model_name in model_names:
            # モデル名のマッピング（部分一致で判定）
            adjustment = 1.0  # デフォルト係数
            for pattern, value in adjustments.items():
                if pattern in model_name.lower():
                    adjustment = value
                    break
            model_adjusted[regime][model_name] = adjustment

    return model_adjusted


def create_scoring_explanation(config: DynamicWeightingConfig) -> Dict[str, Any]:
    """
    Issue #477対応: スコアリング手法の説明取得

    Args:
        config: 動的重み調整設定

    Returns:
        各スコアリング手法の詳細説明
    """
    explanations = {
        'performance_based': {
            'description': 'RMSE逆数と方向的中率の重み付き合計',
            'formula': f'{config.accuracy_weight} × (1/(1+RMSE)) + {config.direction_weight} × 方向的中率',
            'range': f'0 - {config.accuracy_weight + config.direction_weight}',
            'components': {
                'accuracy_score': '1/(1+RMSE) - 予測誤差の逆数（範囲: 0-1）',
                'direction_score': '方向的中率 - 価格変動方向の予測精度（範囲: 0-1）'
            }
        },
        'sharpe_based': {
            'description': 'リスク調整後の予測精度評価（シャープレシオ）',
            'formula': 'max(mean(accuracy_returns) / std(accuracy_returns), clip_min)',
            'range': f'{config.sharpe_clip_min} - ∞',
            'components': {
                'accuracy_returns': 'pred_returns × actual_returns - 方向一致度',
                'sharpe_ratio': 'accuracy_returnsの平均/標準偏差',
                'clipping': f'下限値{config.sharpe_clip_min}でクリップ'
            }
        },
        'regime_aware': {
            'description': 'performance_basedに市場状態別調整係数を適用',
            'formula': 'performance_score × regime_adjustment_factor',
            'range': '動的（レジーム調整係数に依存）',
            'components': {
                'base_score': 'performance_basedスコア',
                'regime_factor': '現在の市場状態に応じた調整係数'
            }
        }
    }

    return explanations