#!/usr/bin/env python3
"""
Scoring algorithms for weight calculation

このモジュールは重み計算に使用される各種スコアリング
アルゴリズムを実装します。パフォーマンス評価、
シャープレシオ計算等の数値的手法を提供します。
"""

import numpy as np
from typing import Dict, Any

from .core import DynamicWeightingConfig
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ScoringAlgorithms:
    """
    スコアリングアルゴリズム集

    重み計算で使用される数値的評価手法を提供します。
    """

    def __init__(self, config: DynamicWeightingConfig):
        """
        初期化

        Args:
            config: システム設定
        """
        self.config = config

    def calculate_performance_score(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        パフォーマンススコア計算

        精度スコア（RMSE逆数）と方向スコア（方向的中率）の
        重み付き合計を算出します。

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            パフォーマンススコア
        """
        try:
            # 1. 精度スコア計算（RMSE based）
            rmse = np.sqrt(np.mean((actual_array - pred_array) ** 2))
            accuracy_score = 1.0 / (1.0 + rmse)

            # 2. 方向スコア計算
            direction_score = self.calculate_direction_score(pred_array, actual_array)

            # 3. 総合スコア算出
            composite_score = (
                self.config.accuracy_weight * accuracy_score +
                self.config.direction_weight * direction_score
            )

            # 詳細ログ（設定に応じて）
            if self.config.enable_score_logging or self.config.verbose:
                logger.debug(
                    f"スコア詳細: RMSE={rmse:.4f}, "
                    f"精度={accuracy_score:.3f}(×{self.config.accuracy_weight}), "
                    f"方向={direction_score:.3f}(×{self.config.direction_weight}), "
                    f"総合={composite_score:.3f}"
                )

            return float(composite_score)

        except Exception as e:
            logger.warning(f"パフォーマンススコア計算エラー: {e}")
            return 1.0  # 中立スコア

    def calculate_direction_score(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        方向的中率を計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            方向的中率（0.0-1.0）
        """
        try:
            if len(pred_array) > 1:
                # 実際の価格変化方向
                actual_diff = np.diff(actual_array)
                # 予測の価格変化方向
                pred_diff = np.diff(pred_array)
                # 方向一致率: sign関数で方向を判定し、一致率を計算
                direction_score = np.mean(np.sign(actual_diff) == np.sign(pred_diff))
            else:
                # データ不足時はニュートラル（0.5）
                direction_score = 0.5

            return float(direction_score)

        except Exception as e:
            logger.warning(f"方向スコア計算エラー: {e}")
            return 0.5

    def calculate_sharpe_ratio(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        シャープレシオを計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            シャープレシオ
        """
        try:
            # 1. リターン計算
            pred_returns = np.diff(pred_array) / pred_array[:-1]
            actual_returns = np.diff(actual_array) / actual_array[:-1]

            # 2. 予測精度リターン計算
            accuracy_returns = pred_returns * actual_returns

            # 3. シャープレシオ計算
            accuracy_std = np.std(accuracy_returns)
            if accuracy_std > 0:
                sharpe_ratio = np.mean(accuracy_returns) / accuracy_std
            else:
                sharpe_ratio = 0.0

            # 詳細ログ（設定に応じて）
            if self.config.enable_score_logging or self.config.verbose:
                logger.debug(
                    f"シャープ詳細: 生シャープ={sharpe_ratio:.4f}, "
                    f"精度リターン平均={np.mean(accuracy_returns):.4f}"
                )

            return float(sharpe_ratio)

        except Exception as e:
            logger.warning(f"シャープレシオ計算エラー: {e}")
            return 0.0

    def calculate_information_ratio(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray,
        benchmark_array: np.ndarray = None
    ) -> float:
        """
        インフォメーションレシオを計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列
            benchmark_array: ベンチマーク配列（未指定時は単純平均）

        Returns:
            インフォメーションレシオ
        """
        try:
            if benchmark_array is None:
                benchmark_array = np.full_like(actual_array, np.mean(actual_array))

            # 超過リターン計算
            pred_returns = np.diff(pred_array) / pred_array[:-1]
            actual_returns = np.diff(actual_array) / actual_array[:-1]
            benchmark_returns = np.diff(benchmark_array) / benchmark_array[:-1]

            # 超過リターン
            excess_returns = actual_returns - benchmark_returns
            
            # 追跡エラー
            tracking_error = np.std(excess_returns)

            if tracking_error > 0:
                info_ratio = np.mean(excess_returns) / tracking_error
            else:
                info_ratio = 0.0

            return float(info_ratio)

        except Exception as e:
            logger.warning(f"インフォメーションレシオ計算エラー: {e}")
            return 0.0

    def calculate_sortino_ratio(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray,
        target_return: float = 0.0
    ) -> float:
        """
        ソルティノレシオを計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列
            target_return: 目標リターン

        Returns:
            ソルティノレシオ
        """
        try:
            # リターン計算
            pred_returns = np.diff(pred_array) / pred_array[:-1]
            actual_returns = np.diff(actual_array) / actual_array[:-1]
            
            # 精度リターン
            accuracy_returns = pred_returns * actual_returns

            # 下方偏差計算
            downside_returns = accuracy_returns - target_return
            downside_deviation = np.sqrt(
                np.mean(np.minimum(downside_returns, 0) ** 2)
            )

            if downside_deviation > 0:
                sortino_ratio = (np.mean(accuracy_returns) - target_return) / downside_deviation
            else:
                sortino_ratio = 0.0

            return float(sortino_ratio)

        except Exception as e:
            logger.warning(f"ソルティノレシオ計算エラー: {e}")
            return 0.0

    def calculate_calmar_ratio(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray
    ) -> float:
        """
        カルマーレシオを計算

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列

        Returns:
            カルマーレシオ
        """
        try:
            # リターン計算
            pred_returns = np.diff(pred_array) / pred_array[:-1]
            actual_returns = np.diff(actual_array) / actual_array[:-1]
            
            # 精度リターン
            accuracy_returns = pred_returns * actual_returns

            # 累積リターン
            cumulative_returns = np.cumprod(1 + accuracy_returns)
            
            # 最大ドローダウン計算
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)

            if abs(max_drawdown) > 0:
                calmar_ratio = np.mean(accuracy_returns) / abs(max_drawdown)
            else:
                calmar_ratio = 0.0

            return float(calmar_ratio)

        except Exception as e:
            logger.warning(f"カルマーレシオ計算エラー: {e}")
            return 0.0

    def get_comprehensive_score(
        self,
        pred_array: np.ndarray,
        actual_array: np.ndarray,
        weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        包括的スコア評価

        複数のリスク調整指標を計算し、統合スコアを算出

        Args:
            pred_array: 予測値配列
            actual_array: 実際値配列
            weights: 各指標の重み（未指定時は均等）

        Returns:
            各指標とその統合スコア
        """
        if weights is None:
            weights = {
                'sharpe': 0.3,
                'information': 0.2,
                'sortino': 0.25,
                'calmar': 0.25
            }

        try:
            scores = {
                'sharpe': self.calculate_sharpe_ratio(pred_array, actual_array),
                'information': self.calculate_information_ratio(pred_array, actual_array),
                'sortino': self.calculate_sortino_ratio(pred_array, actual_array),
                'calmar': self.calculate_calmar_ratio(pred_array, actual_array)
            }

            # 正規化（-1〜1の範囲にクリップ）
            normalized_scores = {}
            for name, score in scores.items():
                # 極値をクリップ
                clipped_score = max(-1.0, min(1.0, score))
                # 0-1の範囲に変換
                normalized_scores[name] = (clipped_score + 1.0) / 2.0

            # 統合スコア計算
            composite_score = sum(
                normalized_scores[name] * weights.get(name, 0.25)
                for name in normalized_scores
            )

            result = normalized_scores.copy()
            result['composite'] = composite_score

            return result

        except Exception as e:
            logger.error(f"包括的スコア計算エラー: {e}")
            return {'composite': 0.5}  # 中立値

    def get_scoring_explanation(self) -> Dict[str, Dict[str, str]]:
        """
        スコアリング手法の詳細説明

        Returns:
            各手法の説明辞書
        """
        return {
            'performance_based': {
                'description': 'RMSE逆数と方向的中率の重み付き合計',
                'formula': (f'{self.config.accuracy_weight} × (1/(1+RMSE)) + '
                           f'{self.config.direction_weight} × 方向的中率'),
                'range': f'0 - {self.config.accuracy_weight + self.config.direction_weight}',
                'components': {
                    'accuracy_score': '1/(1+RMSE) - 予測誤差の逆数（範囲: 0-1）',
                    'direction_score': '方向的中率 - 価格変動方向の予測精度（範囲: 0-1）'
                }
            },
            'sharpe_based': {
                'description': 'リスク調整後の予測精度評価（シャープレシオ）',
                'formula': 'max(mean(accuracy_returns) / std(accuracy_returns), clip_min)',
                'range': f'{self.config.sharpe_clip_min} - ∞',
                'components': {
                    'accuracy_returns': 'pred_returns × actual_returns - 方向一致度',
                    'sharpe_ratio': 'accuracy_returnsの平均/標準偏差',
                    'clipping': f'下限値{self.config.sharpe_clip_min}でクリップ'
                }
            },
            'information_ratio': {
                'description': '超過リターンをリスクで調整した指標',
                'formula': '(portfolio_return - benchmark_return) / tracking_error',
                'range': '-∞ - ∞',
                'components': {
                    'excess_return': 'ベンチマークに対する超過リターン',
                    'tracking_error': '超過リターンの標準偏差'
                }
            },
            'sortino_ratio': {
                'description': '下方リスクに焦点を当てたリスク調整指標',
                'formula': '(return - target) / downside_deviation',
                'range': '-∞ - ∞',
                'components': {
                    'downside_deviation': '目標リターン下回る期間の標準偏差'
                }
            }
        }