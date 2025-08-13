#!/usr/bin/env python3
"""
適応的最適化システム

Issue #487対応: 完全自動化システム実装 - Phase 2
ハイパーパラメータ自動最適化・市場レジーム検出・動的モデル調整
"""

import asyncio
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import pickle
import json
import os

from ..utils.logging_config import get_context_logger
from ..ml.ensemble_system import EnsembleSystem, EnsembleConfig
from .smart_symbol_selector import SmartSymbolSelector
from .notification_system import get_notification_system

logger = get_context_logger(__name__)


class MarketRegime(Enum):
    """市場レジーム"""
    BULL = "bull"           # 強気相場
    BEAR = "bear"           # 弱気相場
    SIDEWAYS = "sideways"   # 横ばい相場
    VOLATILE = "volatile"   # 高ボラティリティ相場
    UNKNOWN = "unknown"     # 判定不可


class OptimizationScope(Enum):
    """最適化スコープ"""
    HYPERPARAMETERS = "hyperparameters"     # ハイパーパラメータ最適化
    ENSEMBLE_WEIGHTS = "ensemble_weights"   # アンサンブル重み最適化
    FEATURE_SELECTION = "feature_selection" # 特徴量選択最適化
    FULL_OPTIMIZATION = "full_optimization" # 全体最適化


@dataclass
class OptimizationConfig:
    """最適化設定"""
    n_trials: int = 100                    # 試行回数
    timeout: int = 3600                    # タイムアウト(秒)
    n_jobs: int = 1                        # 並列ジョブ数
    sampler: str = "TPE"                   # サンプラー種別
    pruner: str = "MedianPruner"          # プルーナー種別
    cv_folds: int = 5                      # クロスバリデーション分割数
    optimization_metric: str = "r2_score"  # 最適化指標
    min_trials_for_pruning: int = 10       # プルーニング開始試行数


@dataclass
class MarketRegimeMetrics:
    """市場レジーム指標"""
    regime: MarketRegime
    confidence: float                      # 信頼度 (0-1)
    volatility: float                      # ボラティリティ
    trend_strength: float                  # トレンド強度 (-1から1)
    momentum: float                        # モメンタム
    regime_duration_days: int              # レジーム継続日数
    transition_probability: float          # レジーム変化確率


@dataclass
class OptimizationResult:
    """最適化結果"""
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time: float
    market_regime: MarketRegime
    timestamp: datetime
    model_performance: Dict[str, float]
    convergence_achieved: bool


class AdaptiveOptimizationSystem:
    """
    Issue #487対応: 適応的最適化システム

    Phase 2の核心機能:
    - Optuna統合ハイパーパラメータ最適化
    - 市場レジーム自動検出
    - 適応的モデル調整
    - 長期的パフォーマンス追跡
    """

    def __init__(self, config: OptimizationConfig = None):
        """初期化"""
        self.config = config or OptimizationConfig()
        self.optimization_history: List[OptimizationResult] = []
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self.regime_history: List[MarketRegimeMetrics] = []

        # 最適化結果保存パス
        self.results_path = "optimization_results"
        os.makedirs(self.results_path, exist_ok=True)

        # Optuna設定
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = None

        # 通知システム
        self.notification = get_notification_system()

    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegimeMetrics:
        """
        市場レジーム自動検出

        Args:
            price_data: 価格データ (OHLCV)

        Returns:
            市場レジーム指標
        """
        try:
            if price_data.empty or len(price_data) < 30:
                return MarketRegimeMetrics(
                    regime=MarketRegime.UNKNOWN,
                    confidence=0.0,
                    volatility=0.0,
                    trend_strength=0.0,
                    momentum=0.0,
                    regime_duration_days=0,
                    transition_probability=0.5
                )

            # 価格データの前処理
            closes = price_data['Close'] if 'Close' in price_data.columns else price_data.iloc[:, -1]
            returns = closes.pct_change().dropna()

            # 1. ボラティリティ計算 (年率化)
            volatility = returns.std() * np.sqrt(252)

            # 2. トレンド強度計算 (線形回帰の傾き)
            x = np.arange(len(closes))
            trend_slope = np.polyfit(x, closes.values, 1)[0]
            price_range = closes.max() - closes.min()
            trend_strength = trend_slope / price_range if price_range > 0 else 0
            trend_strength = np.clip(trend_strength * 100, -1, 1)  # -1から1に正規化

            # 3. モメンタム計算 (短期・長期移動平均の関係)
            if len(closes) >= 50:
                ma_short = closes.rolling(10).mean().iloc[-1]
                ma_long = closes.rolling(50).mean().iloc[-1]
                momentum = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
            else:
                momentum = 0

            # 4. レジーム判定ロジック
            regime, confidence = self._classify_regime(
                volatility, trend_strength, momentum, returns
            )

            # 5. レジーム継続期間推定
            regime_duration = self._estimate_regime_duration()

            # 6. レジーム変化確率計算
            transition_prob = self._calculate_transition_probability(
                volatility, abs(trend_strength), abs(momentum)
            )

            metrics = MarketRegimeMetrics(
                regime=regime,
                confidence=confidence,
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                regime_duration_days=regime_duration,
                transition_probability=transition_prob
            )

            # 履歴に追加
            self.regime_history.append(metrics)
            if len(self.regime_history) > 365:  # 1年分保持
                self.regime_history = self.regime_history[-365:]

            self.current_regime = regime

            logger.info(f"市場レジーム検出: {regime.value} (信頼度: {confidence:.2f})")
            logger.info(f"  ボラティリティ: {volatility:.3f}")
            logger.info(f"  トレンド強度: {trend_strength:.3f}")
            logger.info(f"  モメンタム: {momentum:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"市場レジーム検出エラー: {e}")
            return MarketRegimeMetrics(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                regime_duration_days=0,
                transition_probability=0.5
            )

    def _classify_regime(self, volatility: float, trend_strength: float,
                        momentum: float, returns: pd.Series) -> Tuple[MarketRegime, float]:
        """レジーム分類"""

        # 閾値設定
        high_vol_threshold = 0.25  # 25%年率ボラティリティ
        trend_threshold = 0.3      # トレンド強度閾値
        momentum_threshold = 0.05   # モメンタム閾値

        confidence_scores = {}

        # 高ボラティリティ判定
        if volatility > high_vol_threshold:
            confidence_scores[MarketRegime.VOLATILE] = min(volatility / high_vol_threshold, 2.0) - 1.0

        # 強気・弱気判定
        if trend_strength > trend_threshold and momentum > momentum_threshold:
            strength = min((trend_strength + momentum) / 2, 1.0)
            confidence_scores[MarketRegime.BULL] = strength
        elif trend_strength < -trend_threshold and momentum < -momentum_threshold:
            strength = min(abs(trend_strength + momentum) / 2, 1.0)
            confidence_scores[MarketRegime.BEAR] = strength

        # 横ばい判定
        if abs(trend_strength) < trend_threshold / 2 and abs(momentum) < momentum_threshold / 2:
            sideways_strength = 1.0 - max(abs(trend_strength), abs(momentum)) * 2
            confidence_scores[MarketRegime.SIDEWAYS] = max(sideways_strength, 0.0)

        # 最も確信度の高いレジームを選択
        if confidence_scores:
            best_regime = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            confidence = min(confidence_scores[best_regime], 1.0)
            return best_regime, confidence
        else:
            return MarketRegime.UNKNOWN, 0.0

    def _estimate_regime_duration(self) -> int:
        """レジーム継続期間推定"""
        if len(self.regime_history) < 2:
            return 1

        current_regime = self.current_regime
        duration = 1

        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i].regime == current_regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_transition_probability(self, volatility: float,
                                        trend_strength: float, momentum: float) -> float:
        """レジーム変化確率計算"""
        # 変化要因のスコア化
        volatility_factor = min(volatility / 0.3, 1.0)  # 高ボラティリティほど変化確率高
        trend_factor = min(trend_strength, 1.0)         # 強トレンドほど変化確率高
        momentum_factor = min(momentum, 1.0)            # 強モメンタムほど変化確率高

        # 継続期間による変化確率調整
        duration = self._estimate_regime_duration()
        duration_factor = min(duration / 30, 1.0)       # 30日以上継続で変化確率上昇

        # 総合変化確率 (0-1)
        transition_prob = (volatility_factor * 0.4 +
                          (trend_factor + momentum_factor) * 0.3 +
                          duration_factor * 0.3)

        return min(transition_prob, 0.95)  # 最大95%

    async def optimize_hyperparameters(self, ensemble_system: EnsembleSystem,
                                     X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray,
                                     market_regime: MarketRegime = None) -> OptimizationResult:
        """
        ハイパーパラメータ自動最適化

        Args:
            ensemble_system: アンサンブルシステム
            X_train: 訓練データ
            y_train: 訓練目標
            X_val: 検証データ
            y_val: 検証目標
            market_regime: 市場レジーム

        Returns:
            最適化結果
        """
        logger.info("ハイパーパラメータ自動最適化開始")
        start_time = time.time()

        try:
            # Optunaスタディ作成
            regime_suffix = f"_{market_regime.value}" if market_regime else ""
            study_name = f"ensemble_optimization{regime_suffix}_{int(start_time)}"

            self.study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=self.config.min_trials_for_pruning),
                study_name=study_name
            )

            # 最適化実行
            objective = self._create_objective_function(
                ensemble_system, X_train, y_train, X_val, y_val
            )

            self.study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs
            )

            optimization_time = time.time() - start_time

            # 最適化結果
            best_trial = self.study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value

            # モデル性能評価
            performance_metrics = await self._evaluate_optimized_model(
                ensemble_system, best_params, X_train, y_train, X_val, y_val
            )

            # 収束判定
            convergence_achieved = self._check_convergence()

            result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                n_trials=len(self.study.trials),
                optimization_time=optimization_time,
                market_regime=market_regime or self.current_regime,
                timestamp=datetime.now(),
                model_performance=performance_metrics,
                convergence_achieved=convergence_achieved
            )

            # 結果保存
            self._save_optimization_result(result)

            # 履歴に追加
            self.optimization_history.append(result)

            logger.info(f"最適化完了: スコア={best_score:.4f}, 時間={optimization_time:.1f}秒")
            logger.info(f"最適パラメータ: {best_params}")

            # 通知送信
            await self._send_optimization_notification(result)

            return result

        except Exception as e:
            logger.error(f"ハイパーパラメータ最適化エラー: {e}")
            raise

    def _create_objective_function(self, ensemble_system: EnsembleSystem,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray) -> Callable:
        """Optuna目的関数作成"""

        def objective(trial: optuna.Trial) -> float:
            try:
                # パラメータサンプリング
                params = self._sample_hyperparameters(trial, self.current_regime)

                # アンサンブル設定更新
                config = self._create_ensemble_config(params)
                optimized_ensemble = EnsembleSystem(config)

                # 訓練実行
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
                optimized_ensemble.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    feature_names=feature_names
                )

                # 予測・評価
                predictions = optimized_ensemble.predict(X_val)

                # スコア計算
                from sklearn.metrics import r2_score, mean_squared_error
                if self.config.optimization_metric == "r2_score":
                    score = r2_score(y_val, predictions.final_predictions)
                elif self.config.optimization_metric == "neg_mse":
                    score = -mean_squared_error(y_val, predictions.final_predictions)
                else:
                    score = r2_score(y_val, predictions.final_predictions)

                return score

            except Exception as e:
                logger.warning(f"試行失敗: {e}")
                return -1000.0  # 非常に悪いスコア

        return objective

    def _sample_hyperparameters(self, trial: optuna.Trial, regime: MarketRegime) -> Dict[str, Any]:
        """レジーム適応的パラメータサンプリング"""

        params = {}

        # XGBoost パラメータ
        if regime == MarketRegime.VOLATILE:
            # 高ボラティリティ環境: 過学習抑制重視
            params['xgboost_n_estimators'] = trial.suggest_int('xgboost_n_estimators', 50, 200)
            params['xgboost_max_depth'] = trial.suggest_int('xgboost_max_depth', 3, 6)
            params['xgboost_learning_rate'] = trial.suggest_float('xgboost_learning_rate', 0.01, 0.1)
            params['xgboost_reg_alpha'] = trial.suggest_float('xgboost_reg_alpha', 0.1, 1.0)
            params['xgboost_reg_lambda'] = trial.suggest_float('xgboost_reg_lambda', 1.0, 5.0)
        elif regime == MarketRegime.BULL:
            # 強気相場: トレンド追従重視
            params['xgboost_n_estimators'] = trial.suggest_int('xgboost_n_estimators', 200, 500)
            params['xgboost_max_depth'] = trial.suggest_int('xgboost_max_depth', 6, 10)
            params['xgboost_learning_rate'] = trial.suggest_float('xgboost_learning_rate', 0.05, 0.15)
            params['xgboost_reg_alpha'] = trial.suggest_float('xgboost_reg_alpha', 0.01, 0.5)
            params['xgboost_reg_lambda'] = trial.suggest_float('xgboost_reg_lambda', 0.5, 2.0)
        else:
            # デフォルト設定
            params['xgboost_n_estimators'] = trial.suggest_int('xgboost_n_estimators', 100, 400)
            params['xgboost_max_depth'] = trial.suggest_int('xgboost_max_depth', 4, 8)
            params['xgboost_learning_rate'] = trial.suggest_float('xgboost_learning_rate', 0.03, 0.12)
            params['xgboost_reg_alpha'] = trial.suggest_float('xgboost_reg_alpha', 0.01, 1.0)
            params['xgboost_reg_lambda'] = trial.suggest_float('xgboost_reg_lambda', 0.5, 3.0)

        # CatBoost パラメータ
        params['catboost_iterations'] = trial.suggest_int('catboost_iterations', 100, 800)
        params['catboost_depth'] = trial.suggest_int('catboost_depth', 4, 10)
        params['catboost_learning_rate'] = trial.suggest_float('catboost_learning_rate', 0.02, 0.15)
        params['catboost_l2_leaf_reg'] = trial.suggest_float('catboost_l2_leaf_reg', 1.0, 10.0)

        # RandomForest パラメータ
        params['rf_n_estimators'] = trial.suggest_int('rf_n_estimators', 50, 300)
        params['rf_max_depth'] = trial.suggest_int('rf_max_depth', 5, 20)
        params['rf_min_samples_split'] = trial.suggest_int('rf_min_samples_split', 2, 10)

        return params

    def _create_ensemble_config(self, params: Dict[str, Any]) -> EnsembleConfig:
        """最適化パラメータからアンサンブル設定作成"""

        config = EnsembleConfig(
            use_xgboost=True,
            use_catboost=True,
            use_random_forest=True,
            use_lstm_transformer=False,
            use_gradient_boosting=False,
            use_svr=False,

            xgboost_params={
                'n_estimators': params.get('xgboost_n_estimators', 300),
                'max_depth': params.get('xgboost_max_depth', 8),
                'learning_rate': params.get('xgboost_learning_rate', 0.05),
                'reg_alpha': params.get('xgboost_reg_alpha', 0.01),
                'reg_lambda': params.get('xgboost_reg_lambda', 1.0),
                'enable_hyperopt': False
            },

            catboost_params={
                'iterations': params.get('catboost_iterations', 500),
                'depth': params.get('catboost_depth', 8),
                'learning_rate': params.get('catboost_learning_rate', 0.05),
                'l2_leaf_reg': params.get('catboost_l2_leaf_reg', 3.0),
                'enable_hyperopt': False,
                'verbose': 0
            },

            random_forest_params={
                'n_estimators': params.get('rf_n_estimators', 200),
                'max_depth': params.get('rf_max_depth', 15),
                'min_samples_split': params.get('rf_min_samples_split', 2),
                'enable_hyperopt': False
            }
        )

        return config

    async def _evaluate_optimized_model(self, ensemble_system: EnsembleSystem,
                                       best_params: Dict[str, Any],
                                       X_train: np.ndarray, y_train: np.ndarray,
                                       X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """最適化モデルの詳細評価"""

        try:
            # 最適パラメータでモデル再訓練
            config = self._create_ensemble_config(best_params)
            optimized_ensemble = EnsembleSystem(config)

            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            optimized_ensemble.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                feature_names=feature_names
            )

            # 予測
            predictions = optimized_ensemble.predict(X_val)

            # 詳細評価指標計算
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            metrics = {
                'r2_score': r2_score(y_val, predictions.final_predictions),
                'rmse': np.sqrt(mean_squared_error(y_val, predictions.final_predictions)),
                'mae': mean_absolute_error(y_val, predictions.final_predictions),
                'direction_accuracy': np.mean(
                    (y_val > 0) == (predictions.final_predictions > 0)
                ) * 100
            }

            return metrics

        except Exception as e:
            logger.error(f"最適化モデル評価エラー: {e}")
            return {}

    def _check_convergence(self) -> bool:
        """最適化収束判定"""
        if not self.study or len(self.study.trials) < 20:
            return False

        # 最新20試行の改善率を確認
        recent_values = [trial.value for trial in self.study.trials[-20:] if trial.value is not None]

        if len(recent_values) < 10:
            return False

        # 上位10%の値の分散が小さければ収束と判定
        top_values = sorted(recent_values, reverse=True)[:max(1, len(recent_values) // 10)]

        if len(top_values) > 1:
            variance = np.var(top_values)
            return variance < 0.001  # 分散が小さければ収束

        return False

    def _save_optimization_result(self, result: OptimizationResult):
        """最適化結果保存"""
        try:
            filename = f"optimization_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_path, filename)

            # JSON用にデータ変換
            result_dict = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'market_regime': result.market_regime.value,
                'timestamp': result.timestamp.isoformat(),
                'model_performance': result.model_performance,
                'convergence_achieved': result.convergence_achieved
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"最適化結果保存: {filepath}")

        except Exception as e:
            logger.error(f"最適化結果保存エラー: {e}")

    async def _send_optimization_notification(self, result: OptimizationResult):
        """最適化結果通知"""
        try:
            notification_data = {
                'optimization_score': result.best_score,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'market_regime': result.market_regime.value,
                'convergence_status': '収束' if result.convergence_achieved else '未収束',
                'performance_summary': ', '.join([
                    f"{k}: {v:.3f}" for k, v in result.model_performance.items()
                ])
            }

            # カスタム通知テンプレート作成
            from .notification_system import NotificationTemplate, NotificationType, NotificationChannel

            template = NotificationTemplate(
                template_id="optimization_result",
                subject_template="[最適化完了] ハイパーパラメータ自動最適化結果 - {market_regime}相場",
                body_template="""
🤖 ハイパーパラメータ自動最適化完了

📊 最適化結果:
- スコア: {optimization_score:.4f}
- 試行回数: {n_trials}回
- 最適化時間: {optimization_time:.1f}秒
- 市場レジーム: {market_regime}
- 収束状況: {convergence_status}

📈 モデル性能:
{performance_summary}

---
Issue #487 Phase 2: 適応的最適化システム
""",
                notification_type=NotificationType.SUCCESS,
                channels=[NotificationChannel.LOG, NotificationChannel.CONSOLE, NotificationChannel.FILE]
            )

            self.notification.templates["optimization_result"] = template
            self.notification.send_notification("optimization_result", notification_data)

        except Exception as e:
            logger.error(f"最適化結果通知エラー: {e}")

    def get_regime_adapted_config(self, regime: MarketRegime = None) -> EnsembleConfig:
        """レジーム適応設定取得"""
        if not regime:
            regime = self.current_regime

        # 最新の最適化結果から該当レジーム設定を取得
        regime_optimizations = [
            result for result in self.optimization_history
            if result.market_regime == regime
        ]

        if regime_optimizations:
            # 最新の最適化結果を使用
            latest_result = max(regime_optimizations, key=lambda x: x.timestamp)
            return self._create_ensemble_config(latest_result.best_params)
        else:
            # デフォルト設定にレジーム適応調整
            return self._get_default_regime_config(regime)

    def _get_default_regime_config(self, regime: MarketRegime) -> EnsembleConfig:
        """レジーム別デフォルト設定"""

        base_config = EnsembleConfig(
            use_xgboost=True,
            use_catboost=True,
            use_random_forest=True,
            use_lstm_transformer=False,
            use_gradient_boosting=False,
            use_svr=False
        )

        if regime == MarketRegime.VOLATILE:
            # 高ボラティリティ: 過学習抑制
            base_config.xgboost_params = {
                'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05,
                'reg_alpha': 0.5, 'reg_lambda': 2.0, 'enable_hyperopt': False
            }
            base_config.catboost_params = {
                'iterations': 300, 'depth': 6, 'learning_rate': 0.03,
                'l2_leaf_reg': 5.0, 'enable_hyperopt': False, 'verbose': 0
            }
        elif regime == MarketRegime.BULL:
            # 強気相場: トレンド追従
            base_config.xgboost_params = {
                'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.08,
                'reg_alpha': 0.1, 'reg_lambda': 1.0, 'enable_hyperopt': False
            }
            base_config.catboost_params = {
                'iterations': 600, 'depth': 8, 'learning_rate': 0.06,
                'l2_leaf_reg': 2.0, 'enable_hyperopt': False, 'verbose': 0
            }
        elif regime == MarketRegime.BEAR:
            # 弱気相場: 安定性重視
            base_config.xgboost_params = {
                'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.04,
                'reg_alpha': 0.3, 'reg_lambda': 1.5, 'enable_hyperopt': False
            }
            base_config.catboost_params = {
                'iterations': 500, 'depth': 7, 'learning_rate': 0.04,
                'l2_leaf_reg': 3.0, 'enable_hyperopt': False, 'verbose': 0
            }

        return base_config


# デバッグ用メイン関数
async def main():
    """デバッグ用メイン"""
    logger.info("適応的最適化システム テスト実行")

    # システム初期化
    optimizer = AdaptiveOptimizationSystem()

    # テスト用市場データ生成
    np.random.seed(42)
    n_samples = 300
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # よりリアルな株価データ生成
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    market_data = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Volume': np.random.lognormal(10, 0.5, n_samples)
    })

    # 市場レジーム検出テスト
    logger.info("市場レジーム検出テスト")
    regime_metrics = optimizer.detect_market_regime(market_data)

    # 検出結果表示
    print("=" * 50)
    print("市場レジーム検出結果")
    print("=" * 50)
    print(f"レジーム: {regime_metrics.regime.value}")
    print(f"信頼度: {regime_metrics.confidence:.2f}")
    print(f"ボラティリティ: {regime_metrics.volatility:.3f}")
    print(f"トレンド強度: {regime_metrics.trend_strength:.3f}")
    print(f"モメンタム: {regime_metrics.momentum:.3f}")
    print(f"継続期間: {regime_metrics.regime_duration_days}日")
    print(f"変化確率: {regime_metrics.transition_probability:.2f}")

    # レジーム適応設定取得
    adapted_config = optimizer.get_regime_adapted_config(regime_metrics.regime)
    print(f"\nレジーム適応設定:")
    print(f"XGBoost設定: {adapted_config.xgboost_params}")
    print(f"CatBoost設定: {adapted_config.catboost_params}")

    logger.info("適応的最適化システム テスト完了")


if __name__ == "__main__":
    asyncio.run(main())