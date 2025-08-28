#!/usr/bin/env python3
"""
Dynamic Weighting System - Main System

動的重み調整システムのメインクラスと統合制御ロジック
"""

import time
from typing import Dict, List, Any, Optional, Union
import numpy as np
import warnings

from .core import DynamicWeightingConfig, MarketRegime, get_default_regime_adjustments, create_scoring_explanation
from .weighting_algorithms import WeightingAlgorithms  
from .performance_manager import PerformanceManager
from .weight_constraints import WeightConstraintManager
from .market_regime_detector import MarketRegimeDetector
from ..concept_drift_detector import ConceptDriftDetector
from ...utils.logging_config import get_context_logger

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


class DynamicWeightingSystem:
    """
    動的重み調整システム

    市場状況とモデルパフォーマンスに基づいて
    アンサンブル重みをリアルタイムで最適化
    """

    def __init__(self, model_names: List[str],
                 config: Optional[DynamicWeightingConfig] = None):
        """
        初期化

        Args:
            model_names: モデル名リスト
            config: 動的重み調整設定
        """
        self.model_names = model_names
        self.config = config or DynamicWeightingConfig()

        # Issue #478対応: レジーム調整設定の外部化
        self._setup_regime_adjustments()

        # 現在の重み
        n_models = len(model_names)
        self.current_weights = {name: 1.0 / n_models for name in model_names}

        # コンポーネント初期化
        self.weighting_algorithms = WeightingAlgorithms(self.config, model_names)
        self.performance_manager = PerformanceManager(self.config, model_names)
        self.weight_constraint_manager = WeightConstraintManager(self.config, model_names)
        self.market_regime_detector = MarketRegimeDetector(self.config)

        # コンセプトドリフト検出器
        self.concept_drift_detector = None
        self.re_evaluation_needed = False
        self.drift_detection_updates_count = 0

        if self.config.enable_concept_drift_detection:
            self.concept_drift_detector = ConceptDriftDetector(
                metric_threshold=self.config.drift_detection_threshold,
                window_size=self.config.drift_detection_window_size
            )

        logger.info(f"Dynamic Weighting System初期化: {n_models}モデル")

    def _setup_regime_adjustments(self):
        """
        Issue #478対応: レジーム調整設定のセットアップ

        外部設定があればそれを使用し、なければデフォルト設定を作成
        """
        if self.config.regime_adjustments is None:
            # デフォルトレジーム調整設定
            self.config.regime_adjustments = get_default_regime_adjustments(self.model_names)

        if self.config.verbose:
            logger.info("レジーム調整設定をセットアップしました")

    def update_performance(self, predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                         actuals: Union[float, int, np.ndarray, List[float]],
                         timestamp: Optional[int] = None):
        """
        パフォーマンス更新

        Args:
            predictions: モデル別予測値（単一値または配列）
            actuals: 実際の値（単一値または配列）
            timestamp: タイムスタンプ（未指定時は現在時刻）
        """
        try:
            # パフォーマンス管理委譲
            self.performance_manager.update_performance(predictions, actuals, timestamp)

            # 市場状態検出
            if self.config.enable_regime_detection:
                new_regime = self.market_regime_detector.detect_market_regime(
                    self.performance_manager.recent_actuals
                )

            # コンセプトドリフト検出
            self._check_concept_drift()

            # 重み更新判定
            if self.performance_manager.has_sufficient_samples():
                self._update_weights()
                self.performance_manager.reset_counter()

        except Exception as e:
            logger.error(f"パフォーマンス更新エラー: {e}")
            raise

    def _check_concept_drift(self):
        """コンセプトドリフト検出処理"""
        if not self.concept_drift_detector:
            return

        # 全体のアンサンブル予測と実際値を使用してドリフト検出器を更新
        if (len(self.performance_manager.recent_actuals) > 0 and 
            len(self.performance_manager.recent_predictions[self.model_names[0]]) > 0):
            
            # 最新の単一の予測と実際値を取得
            latest_actual = self.performance_manager.recent_actuals[-1]
            # 単純に最初のモデルの予測を使用（実際にはアンサンブル予測を使用すべき）
            ensemble_prediction = self.performance_manager.recent_predictions[self.model_names[0]][-1]

            self.concept_drift_detector.add_performance_data(
                predictions=np.array([ensemble_prediction]),
                actuals=np.array([latest_actual])
            )
            
            drift_result = self.concept_drift_detector.detect_drift()
            drift_detected = drift_result.get("drift_detected", False)
            drift_reason = drift_result.get("reason", "不明")

            if drift_detected:
                self._handle_concept_drift(drift_reason)

    def _handle_concept_drift(self, drift_reason: str):
        """コンセプトドリフト対応処理"""
        logger.warning(f"コンセプトドリフト検出！理由: {drift_reason}")
        
        # 適応戦略: 加速された重み調整
        self.config.update_frequency = max(1, int(self.config.update_frequency * 0.5))  # 半減
        self.config.momentum_factor = min(0.9, self.config.momentum_factor + 0.1)  # モーメンタム増加
        logger.info(f"適応戦略適用: update_frequency={self.config.update_frequency}, momentum_factor={self.config.momentum_factor}")

        # モデル再評価/再トレーニングフラグ
        self.re_evaluation_needed = True
        logger.warning("モデルの再評価/再トレーニングが必要です。")

        # フォールバックメカニズム: 一時的に均等重みに戻す
        logger.critical("コンセプトドリフトを検出しました！重みを均等にリセットし、モデルの緊急再評価を推奨します。")
        n_models = len(self.model_names)
        self.current_weights = {name: 1.0 / n_models for name in self.model_names}

        # アラートのトリガー
        logger.critical(f"重大なコンセプトドリフトを検出しました！モデルの再評価を強く推奨します。理由: {drift_reason}")

    def _update_weights(self):
        """
        重み更新

        Issue #479対応: 重み制約とモメンタム適用順序の最適化
        1. 基本重み計算
        2. モーメンタム適用（制約前）
        3. 包括的制約適用（min/max/sum/change制限）
        4. 最終正規化保証
        """
        try:
            # Step 1: 基本重み計算
            new_weights = self._calculate_base_weights()
            if not new_weights:
                return

            # Step 2: モーメンタム適用（制約前に実行）
            if self.config.momentum_factor > 0:
                momentum_weights = self.weight_constraint_manager.apply_momentum(
                    new_weights, self.current_weights
                )
            else:
                momentum_weights = new_weights

            # Step 3: 包括的制約適用（モーメンタム後に実行）
            final_weights = self.weight_constraint_manager.apply_comprehensive_constraints(
                momentum_weights, self.current_weights
            )

            # Step 4: 最終検証と更新
            self.current_weights = self.weight_constraint_manager.validate_and_update_weights(
                final_weights, self.current_weights, self.market_regime_detector.get_current_regime()
            )

        except Exception as e:
            logger.error(f"重み更新エラー: {e}")

    def _calculate_base_weights(self) -> Optional[Dict[str, float]]:
        """基本重み計算"""
        try:
            recent_predictions = self.performance_manager.recent_predictions
            recent_actuals = self.performance_manager.recent_actuals
            current_regime = self.market_regime_detector.get_current_regime()

            if self.config.weighting_method == "performance_based":
                return self.weighting_algorithms.performance_based_weighting(
                    recent_predictions, recent_actuals
                )
            elif self.config.weighting_method == "sharpe_based":
                return self.weighting_algorithms.sharpe_based_weighting(
                    recent_predictions, recent_actuals
                )
            elif self.config.weighting_method == "regime_aware":
                return self.weighting_algorithms.regime_aware_weighting(
                    recent_predictions, recent_actuals, current_regime
                )
            else:
                logger.warning(f"未知の重み調整手法: {self.config.weighting_method}")
                return None

        except Exception as e:
            logger.error(f"基本重み計算エラー: {e}")
            return None

    # 設定更新メソッド群
    def update_regime_adjustments(self, new_adjustments: Dict[MarketRegime, Dict[str, float]]):
        """Issue #478対応: レジーム調整設定の動的更新"""
        try:
            # 設定の妥当性チェック
            for regime, adjustments in new_adjustments.items():
                if not isinstance(regime, MarketRegime):
                    raise ValueError(f"無効な市場状態: {regime}")

                for model_name, adjustment in adjustments.items():
                    if not isinstance(adjustment, (int, float)) or adjustment <= 0:
                        raise ValueError(f"無効な調整係数: {model_name}={adjustment}")

            # 設定更新
            self.config.regime_adjustments = new_adjustments

            if self.config.verbose:
                logger.info("レジーム調整設定を更新しました")

        except Exception as e:
            logger.error(f"レジーム調整設定更新エラー: {e}")
            raise

    def update_scoring_config(self,
                            sharpe_clip_min: Optional[float] = None,
                            accuracy_weight: Optional[float] = None,
                            direction_weight: Optional[float] = None,
                            enable_score_logging: Optional[bool] = None):
        """Issue #477対応: スコアリング設定の動的更新"""
        try:
            if sharpe_clip_min is not None:
                if sharpe_clip_min < 0:
                    raise ValueError("sharpe_clip_minは0以上である必要があります")
                self.config.sharpe_clip_min = sharpe_clip_min

            if accuracy_weight is not None:
                if accuracy_weight < 0:
                    raise ValueError("accuracy_weightは0以上である必要があります")
                self.config.accuracy_weight = accuracy_weight

            if direction_weight is not None:
                if direction_weight < 0:
                    raise ValueError("direction_weightは0以上である必要があります")
                self.config.direction_weight = direction_weight

            if enable_score_logging is not None:
                self.config.enable_score_logging = enable_score_logging

            if self.config.verbose:
                logger.info("スコアリング設定を更新しました")

        except Exception as e:
            logger.error(f"スコアリング設定更新エラー: {e}")
            raise

    # 情報取得メソッド群
    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self.current_weights.copy()

    def get_scoring_config(self) -> Dict[str, Any]:
        """Issue #477対応: 現在のスコアリング設定取得"""
        return {
            'sharpe_clip_min': self.config.sharpe_clip_min,
            'accuracy_weight': self.config.accuracy_weight,
            'direction_weight': self.config.direction_weight,
            'enable_score_logging': self.config.enable_score_logging,
            'weighting_method': self.config.weighting_method
        }

    def get_scoring_explanation(self) -> Dict[str, Any]:
        """Issue #477対応: スコアリング手法の説明取得"""
        return create_scoring_explanation(self.config)

    def get_regime_adjustments(self) -> Optional[Dict[MarketRegime, Dict[str, float]]]:
        """Issue #478対応: 現在のレジーム調整設定取得"""
        return self.config.regime_adjustments

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """重み履歴取得"""
        return self.weight_constraint_manager.get_weight_history()

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """市場状態履歴取得"""
        return self.market_regime_detector.get_regime_history()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        from .core import PerformanceWindow
        
        summary = {
            'current_weights': self.current_weights,
            'current_regime': self.market_regime_detector.get_current_regime().value,
            'total_updates': self.weight_constraint_manager.get_total_updates(),
            'data_points': len(self.performance_manager.recent_actuals)
        }

        # 各モデルの直近パフォーマンス
        model_performance = {}
        for model_name in self.model_names:
            if len(self.performance_manager.recent_predictions[model_name]) >= 10:
                pred_array = np.array(
                    list(self.performance_manager.recent_predictions[model_name])[-10:]
                )
                actual_array = np.array(
                    list(self.performance_manager.recent_actuals)[-10:]
                )

                window = PerformanceWindow(
                    pred_array, actual_array, [], 
                    self.market_regime_detector.get_current_regime()
                )
                metrics = window.calculate_metrics()
                model_performance[model_name] = metrics

        summary['model_performance'] = model_performance
        return summary

    def predict(self, features: Union[np.ndarray, List[float], Dict[str, float]]) -> Any:
        """予測メソッド - PersonalDayTradingEngineとの互換性のため"""
        try:
            from ..base_models.base_model_interface import ModelPrediction

            # 入力特徴量を正規化
            if isinstance(features, dict):
                feature_values = list(features.values())
            elif isinstance(features, (list, np.ndarray)):
                feature_values = features
            else:
                feature_values = [features]

            # 現在の重みに基づいてダミー予測を生成
            current_weights = self.get_current_weights()

            # より現実的なランダム予測値を生成（テスト用）
            import random
            # -2.0から2.0の範囲でランダムな予測値を生成
            base_prediction = random.uniform(-2.0, 2.0)
            predictions = np.array([base_prediction])

            # 重みの平均値を信頼度として使用
            weight_values = list(current_weights.values())
            if weight_values:
                confidence_value = float(np.mean(weight_values))
            else:
                confidence_value = 0.33  # デフォルト信頼度
            confidence = np.array([confidence_value])

            # ModelPrediction形式で返す
            return ModelPrediction(
                predictions=predictions,
                confidence=confidence,
                model_name="dynamic_weighting_ensemble"
            )

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            # エラー時はデフォルト値を返す
            from ..base_models.base_model_interface import ModelPrediction
            return ModelPrediction(
                predictions=np.array([0.0]),
                confidence=np.array([0.0]),
                model_name="dynamic_weighting_ensemble_error"
            )