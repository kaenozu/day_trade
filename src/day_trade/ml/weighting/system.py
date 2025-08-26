#!/usr/bin/env python3
"""
Main Dynamic Weighting System implementation

このモジュールは動的重み調整システムのメインクラス
DynamicWeightingSystemを実装します。各種モジュールを統合し、
アンサンブル学習の重みをリアルタイムで最適化します。
"""

import time
from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from collections import deque

from .core import DynamicWeightingConfig, MarketRegime, WeightingState
from .market_regime import MarketRegimeDetector
from .weight_calculator import WeightCalculator
from .weight_optimizer import WeightOptimizer
from .performance import PerformanceManager
from .visualization import WeightVisualization
from ..concept_drift_detector import ConceptDriftDetector, ConceptDriftConfig
from ...utils.logging_config import get_context_logger

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

        # 現在の重み初期化
        n_models = len(model_names)
        initial_weights = {name: 1.0 / n_models for name in model_names}

        # サブシステム初期化
        self.regime_detector = MarketRegimeDetector(self.config)
        self.weight_calculator = WeightCalculator(model_names, self.config)
        self.weight_optimizer = WeightOptimizer(model_names, self.config)
        self.performance_manager = PerformanceManager(model_names, self.config)

        # 状態初期化
        self.state = self.performance_manager.state
        self.state.current_weights = initial_weights
        
        # 重み履歴の初期化
        self.state.weight_history = []
        
        # コンセプトドリフト検出器
        self.concept_drift_detector = None
        if self.config.enable_concept_drift_detection:
            self.concept_drift_detector = ConceptDriftDetector(
                metric_threshold=self.config.drift_detection_threshold,
                window_size=self.config.drift_detection_window_size
            )

        logger.info(f"Dynamic Weighting System初期化: {n_models}モデル")
        # 可視化機能
        self.visualization = WeightVisualization(model_names)
        

    def _setup_regime_adjustments(self):
        """
        Issue #478対応: レジーム調整設定のセットアップ

        外部設定があればそれを使用し、なければデフォルト設定を作成
        """
        if self.config.regime_adjustments is None:
            # デフォルトレジーム調整設定
            self.config.regime_adjustments = self._get_default_regime_adjustments()

        if self.config.verbose:
            logger.info("レジーム調整設定をセットアップしました")

    def _get_default_regime_adjustments(self) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Issue #478対応: デフォルトレジーム調整設定取得

        Returns:
            市場状態別の調整係数辞書
        """
        # 一般的なモデル名のデフォルト設定
        default_adjustments = {
            MarketRegime.BULL_MARKET: {
                'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9
            },
            MarketRegime.BEAR_MARKET: {
                'svr': 1.2, 'gradient_boosting': 1.1, 'random_forest': 0.9
            },
            MarketRegime.SIDEWAYS: {
                'gradient_boosting': 1.1, 'random_forest': 1.0, 'svr': 1.0
            },
            MarketRegime.HIGH_VOLATILITY: {
                'svr': 1.3, 'gradient_boosting': 0.9, 'random_forest': 0.8
            },
            MarketRegime.LOW_VOLATILITY: {
                'random_forest': 1.2, 'gradient_boosting': 1.1, 'svr': 0.9
            }
        }

        # 実際のモデル名に基づいて調整
        model_adjusted = {}
        for regime, adjustments in default_adjustments.items():
            model_adjusted[regime] = {}
            for model_name in self.model_names:
                # モデル名のマッピング（部分一致で判定）
                adjustment = 1.0  # デフォルト係数
                for pattern, value in adjustments.items():
                    if pattern in model_name.lower():
                        adjustment = value
                        break
                model_adjusted[regime][model_name] = adjustment

        return model_adjusted

    def update_performance(
        self, 
        predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
        actuals: Union[float, int, np.ndarray, List[float]],
        timestamp: Optional[int] = None
    ):
        """
        Issue #475対応: パフォーマンス更新（改善版）

        Args:
            predictions: モデル別予測値（単一値または配列）
            actuals: 実際の値（単一値または配列）
            timestamp: タイムスタンプ（未指定時は現在時刻）

        Raises:
            ValueError: 予測値と実際値の次元が一致しない場合
            TypeError: サポートされていない型の場合
        """
        try:
            # パフォーマンス更新
            success = self.performance_manager.update_performance(
                predictions, actuals, timestamp
            )
            
            if not success:
                logger.warning("パフォーマンス更新に失敗しました")
                return

            # 市場状態検出
            if self.config.enable_regime_detection:
                new_regime = self.regime_detector.detect_market_regime(
                    self.state.recent_actuals
                )
                self.state.current_regime = new_regime

            # コンセプトドリフト検出
            if self.concept_drift_detector:
                self._handle_concept_drift_detection()

            # 重み更新判定
            if self.performance_manager.should_update_weights():
                self._update_weights()
                self.performance_manager.reset_update_counter()

        except Exception as e:
            logger.error(f"パフォーマンス更新エラー: {e}")
            raise

    def _handle_concept_drift_detection(self):
        """
        コンセプトドリフト検出処理
        """
        try:
            if (len(self.state.recent_actuals) > 0 and 
                len(self.state.recent_predictions[self.model_names[0]]) > 0):
                
                # 最新の単一の予測と実際値を取得
                latest_actual = self.state.recent_actuals[-1]
                # アンサンブル予測として最初のモデルの予測を代表に使用
                ensemble_prediction = self.state.recent_predictions[self.model_names[0]][-1]

                self.concept_drift_detector.add_performance_data(
                    predictions=np.array([ensemble_prediction]),
                    actuals=np.array([latest_actual])
                )
                
                drift_result = self.concept_drift_detector.detect_drift()
                drift_detected = drift_result.get("drift_detected", False)
                drift_reason = drift_result.get("reason", "不明")

                if drift_detected:
                    logger.warning(f"コンセプトドリフト検出！理由: {drift_reason}")
                    
                    # 適応戦略: 加速された重み調整
                    self.config.update_frequency = max(
                        1, int(self.config.update_frequency * 0.5)
                    )
                    self.config.momentum_factor = min(
                        0.9, self.config.momentum_factor + 0.1
                    )
                    
                    logger.info(
                        f"適応戦略適用: update_frequency={self.config.update_frequency}, "
                        f"momentum_factor={self.config.momentum_factor}"
                    )

                    # モデル再評価フラグ
                    self.state.re_evaluation_needed = True
                    logger.warning("モデルの再評価/再トレーニングが必要です。")

                    # フォールバックメカニズム: 均等重みにリセット
                    logger.critical(
                        "コンセプトドリフトを検出しました！重みを均等にリセット"
                    )
                    n_models = len(self.model_names)
                    self.state.current_weights = {
                        name: 1.0 / n_models for name in self.model_names
                    }

        except Exception as e:
            logger.error(f"コンセプトドリフト検出エラー: {e}")

    def _update_weights(self):
        """
        重み更新

        Issue #479対応: 重み制約とモメンタム適用順序の最適化
        """
        try:
            # Step 1: 基本重み計算
            new_weights = self.weight_calculator.calculate_weights(
                method=self.config.weighting_method,
                recent_predictions=self.state.recent_predictions,
                recent_actuals=self.state.recent_actuals,
                current_regime=self.state.current_regime
            )

            # Step 2: 重み最適化（モーメンタム＋制約適用）
            optimized_weights = self.weight_optimizer.optimize_weights(
                new_weights=new_weights,
                current_weights=self.state.current_weights,
                total_updates=self.state.total_updates
            )

            # Step 3: 最終検証と更新
            self._validate_and_update_weights(optimized_weights)

        except Exception as e:
            logger.error(f"重み更新エラー: {e}")

    def _validate_and_update_weights(self, weights: Dict[str, float]):
        """
        最終検証と重み更新

        Args:
            weights: 更新対象の重み
        """
        try:
            # 重み検証
            validation_result = self.weight_optimizer.validate_weights(weights)
            
            if not validation_result['valid']:
                logger.warning(f"無効な重みです: {validation_result['errors']}")
                return

            # 重み更新実行
            old_weights = self.state.current_weights.copy()
            self.state.current_weights = weights.copy()

            # 履歴記録
            self.state.weight_history.append({
                'weights': weights.copy(),
                'timestamp': int(time.time()),
                'regime': self.state.current_regime,
                'total_updates': self.state.total_updates
            })

            self.state.total_updates += 1

            # ログ出力
            if self.config.verbose:
                optimization_summary = self.weight_optimizer.get_optimization_summary(
                    old_weights, weights
                )
                significant_changes = optimization_summary.get('significant_changes', [])
                
                if significant_changes:
                    logger.info(f"重み更新: {', '.join(significant_changes)}")

        except Exception as e:
            logger.error(f"重み検証・更新エラー: {e}")

    # 設定管理メソッド
    def update_regime_adjustments(self, new_adjustments: Dict[MarketRegime, Dict[str, float]]):
        """
        Issue #478対応: レジーム調整設定の動的更新

        Args:
            new_adjustments: 新しい調整係数辞書
        """
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

    def load_regime_adjustments_from_dict(self, adjustments_dict: Dict[str, Dict[str, float]]):
        """
        Issue #478対応: 辞書からレジーム調整設定を読み込み

        Args:
            adjustments_dict: 市場状態名をキーとする調整係数辞書
        """
        try:
            regime_adjustments = {}

            for regime_name, adjustments in adjustments_dict.items():
                # 市場状態名を MarketRegime に変換
                regime = None
                for market_regime in MarketRegime:
                    if market_regime.value == regime_name:
                        regime = market_regime
                        break

                if regime is None:
                    logger.warning(f"未知の市場状態: {regime_name}")
                    continue

                regime_adjustments[regime] = adjustments

            self.update_regime_adjustments(regime_adjustments)

        except Exception as e:
            logger.error(f"辞書からのレジーム調整設定読み込みエラー: {e}")
            raise

    def update_scoring_config(self,
                            sharpe_clip_min: Optional[float] = None,
                            accuracy_weight: Optional[float] = None,
                            direction_weight: Optional[float] = None,
                            enable_score_logging: Optional[bool] = None):
        """
        Issue #477対応: スコアリング設定の動的更新

        Args:
            sharpe_clip_min: シャープレシオ下限クリップ値
            accuracy_weight: 精度スコア重み係数
            direction_weight: 方向スコア重み係数
            enable_score_logging: スコア詳細ログ出力フラグ
        """
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

    # 情報取得メソッド
    def get_current_weights(self) -> Dict[str, float]:
        """現在の重み取得"""
        return self.state.current_weights.copy()

    def get_regime_adjustments(self) -> Optional[Dict[MarketRegime, Dict[str, float]]]:
        """
        Issue #478対応: 現在のレジーム調整設定取得

        Returns:
            現在の調整係数辞書
        """
        return self.config.regime_adjustments

    def get_scoring_config(self) -> Dict[str, Any]:
        """
        Issue #477対応: 現在のスコアリング設定取得

        Returns:
            現在のスコアリング設定辞書
        """
        return {
            'sharpe_clip_min': self.config.sharpe_clip_min,
            'accuracy_weight': self.config.accuracy_weight,
            'direction_weight': self.config.direction_weight,
            'enable_score_logging': self.config.enable_score_logging,
            'weighting_method': self.config.weighting_method
        }

    def get_scoring_explanation(self) -> Dict[str, str]:
        """
        Issue #477対応: スコアリング手法の説明取得

        Returns:
            各スコアリング手法の詳細説明
        """
        return self.weight_calculator.get_scoring_explanation()

    def get_weight_history(self) -> List[Dict[str, Any]]:
        """重み履歴取得"""
        return self.state.weight_history.copy()

    def get_regime_history(self) -> List[Dict[str, Any]]:
        """市場状態履歴取得"""
        return self.regime_detector.get_regime_history()

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        summary = self.performance_manager.get_performance_summary()
        summary.update({
            'current_weights': self.state.current_weights,
            'current_regime': self.state.current_regime.value,
        })
        return summary

    # バッチ処理メソッド
    def update_performance_batch(self,
                               batch_predictions: List[Dict[str, Union[float, int, np.ndarray, List[float]]]],
                               batch_actuals: List[Union[float, int, np.ndarray, List[float]]],
                               batch_timestamps: Optional[List[int]] = None):
        """
        Issue #475対応: バッチ形式でのパフォーマンス更新

        Args:
            batch_predictions: モデル別予測値のリスト
            batch_actuals: 実際の値のリスト
            batch_timestamps: タイムスタンプのリスト
        """
        self.performance_manager.update_performance_batch(
            batch_predictions, batch_actuals, batch_timestamps
        )

    def validate_input_data(self, predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                           actuals: Union[float, int, np.ndarray, List[float]]) -> Dict[str, Any]:
        """
        Issue #475対応: 入力データの検証

        Args:
            predictions: 予測値
            actuals: 実際値

        Returns:
            検証結果レポート
        """
        return self.performance_manager.validate_input_data(predictions, actuals)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Issue #475対応: データ統計の取得

        Returns:
            データ統計情報
        """
        return self.performance_manager.get_data_statistics()

    # 外部システム連携メソッド
    def update_external_weights(self, external_weights: Dict[str, float]) -> bool:
        """
        Issue #472対応: 外部重みシステムへの直接更新

        Args:
            external_weights: 更新対象の外部重み辞書への参照

        Returns:
            更新成功可否
        """
        try:
            # 現在の重みを取得
            current_weights = self.get_current_weights()

            # 外部重み辞書を更新
            updated_count = 0
            for model_name, weight in current_weights.items():
                if model_name in external_weights:
                    external_weights[model_name] = weight
                    updated_count += 1

            if self.config.verbose:
                logger.info(f"外部重み更新完了: {updated_count}モデルの重みを更新")

            return updated_count > 0

        except Exception as e:
            logger.error(f"外部重み更新エラー: {e}")
            return False

    def sync_and_update_performance(self,
                                  predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                                  actuals: Union[float, int, np.ndarray, List[float]],
                                  external_weights: Optional[Dict[str, float]] = None,
                                  timestamp: Optional[int] = None) -> Dict[str, float]:
        """
        Issue #472対応: 性能更新と外部重み同期を一括処理

        Args:
            predictions: モデル別予測値
            actuals: 実際の値
            external_weights: 同期対象の外部重み辞書（オプション）
            timestamp: タイムスタンプ（オプション）

        Returns:
            更新後の重み辞書
        """
        try:
            # パフォーマンス更新
            self.update_performance(predictions, actuals, timestamp)

            # 重み更新判定と外部同期
            if self.performance_manager.should_update_weights():
                self._update_weights()
                self.performance_manager.reset_update_counter()

                # 外部重み同期
                if external_weights is not None:
                    self.update_external_weights(external_weights)

                if self.config.verbose:
                    logger.info("統合重み更新完了: パフォーマンス更新 → 重み計算 → 外部同期")

            return self.get_current_weights()

        except Exception as e:
            logger.error(f"統合重み更新エラー: {e}")
            return self.get_current_weights()

    def create_weight_updater(self) -> Callable:
        """
        Issue #472対応: 重み更新関数の生成

        Returns:
            重み更新用の関数オブジェクト
        """
        def weight_updater(predictions: Dict[str, Union[float, int, np.ndarray, List[float]]],
                          actuals: Union[float, int, np.ndarray, List[float]],
                          external_weights: Dict[str, float],
                          timestamp: Optional[int] = None) -> bool:
            """生成された重み更新関数"""
            try:
                updated_weights = self.sync_and_update_performance(
                    predictions, actuals, external_weights, timestamp
                )
                return len(updated_weights) > 0
            except Exception as e:
                logger.warning(f"重み更新関数エラー: {e}")
                return False

        return weight_updater

    # 可視化メソッド（WeightVisualizationクラスに委譲）
    def plot_weight_evolution(self, save_path: Optional[str] = None):
        """重み変化の可視化（WeightVisualizationに委譲）"""
        return self.visualization.plot_weight_evolution(
            self.state.weight_history, save_path
        )

    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """モデル性能比較グラフ（WeightVisualizationに委譲）"""
        performance_data = self.get_performance_summary().get('model_performance', {})
        return self.visualization.plot_performance_comparison(
            performance_data, save_path
        )

    def create_dashboard(self, save_path: Optional[str] = None):
        """総合ダッシュボード作成（WeightVisualizationに委譲）"""
        return self.visualization.create_dashboard(
            self.state.weight_history,
            self.get_performance_summary().get('model_performance', {}),
            self.get_regime_history(),
            save_path
        )

    # 予測メソッド（PersonalDayTradingEngine互換性）
    def predict(self, features: Union[np.ndarray, List[float], Dict[str, float]]) -> Any:
        """
        予測メソッド - PersonalDayTradingEngineとの互換性のため

        Args:
            features: 入力特徴量

        Returns:
            予測結果オブジェクト（predictionsとconfidence属性を持つ）
        """
        try:
            from ..base_models.base_model_interface import ModelPrediction

            # 現在の重みに基づいてダミー予測を生成
            current_weights = self.get_current_weights()

            # より現実的なランダム予測値を生成（テスト用）
            import random
            base_prediction = random.uniform(-2.0, 2.0)
            predictions = np.array([base_prediction])

            # 重みの平均値を信頼度として使用
            weight_values = list(current_weights.values())
            if weight_values:
                confidence_value = float(np.mean(weight_values))
            else:
                confidence_value = 0.33

            confidence = np.array([confidence_value])

            return ModelPrediction(
                predictions=predictions,
                confidence=confidence,
                model_name="dynamic_weighting_ensemble"
            )

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            from ..base_models.base_model_interface import ModelPrediction
            return ModelPrediction(
                predictions=np.array([0.0]),
                confidence=np.array([0.0]),
                model_name="dynamic_weighting_ensemble_error"
            )