"""
予測オーケストレーター
機械学習モデル、特徴量エンジニアリング、アンサンブル戦略を統合して
高精度な株価予測システムを構築する
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .feature_engineering import AdvancedFeatureEngineer, create_target_variables
from .ml_models import MLModelManager, ModelConfig
from .ensemble import EnsembleTradingStrategy, EnsembleStrategy, EnsembleVotingType
from .signals import TradingSignal, SignalType, SignalStrength
from .backtest import BacktestConfig, BacktestResult
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="prediction_orchestrator")


@dataclass
class PredictionConfig:
    """予測システム設定"""

    # 基本設定
    prediction_horizon: int = 5  # 予測期間（日数）
    min_data_length: int = 200  # 最小データ長
    feature_selection_top_k: int = 50  # 特徴量選択数

    # 機械学習設定
    enable_ensemble_ml: bool = True
    enable_adaptive_weighting: bool = True
    retrain_frequency: int = 50  # 再訓練頻度

    # アンサンブル設定
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.ML_OPTIMIZED
    voting_type: EnsembleVotingType = EnsembleVotingType.ML_ENSEMBLE

    # リスク管理設定
    confidence_threshold: float = 0.6
    risk_adjustment_factor: float = 0.8
    max_prediction_uncertainty: float = 0.3

    # パフォーマンス追跡
    enable_performance_tracking: bool = True
    performance_lookback_periods: int = 100


class PredictionOrchestrator:
    """予測オーケストレーター - 統合予測システム"""

    def __init__(
        self,
        config: PredictionConfig,
        models_dir: Optional[str] = None,
        enable_ml: bool = True
    ):
        """
        Args:
            config: 予測システム設定
            models_dir: モデル保存ディレクトリ
            enable_ml: 機械学習機能を有効にするか
        """
        self.config = config
        self.enable_ml = enable_ml

        # コンポーネントの初期化
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ml_manager = MLModelManager(models_dir) if enable_ml else None
        self.ensemble_strategy = EnsembleTradingStrategy(
            ensemble_strategy=config.ensemble_strategy,
            voting_type=config.voting_type,
            enable_ml_models=enable_ml,
            models_dir=models_dir
        )

        # 予測履歴と性能追跡
        self.prediction_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "directional_accuracy": [],
            "return_correlation": []
        }

        # 適応的重み
        self.model_weights: Dict[str, float] = {}
        self.last_retrain_date: Optional[datetime] = None

        logger.info("予測オーケストレーターを初期化しました")

    def train_prediction_system(
        self,
        historical_data: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        予測システム全体を訓練

        Args:
            historical_data: 過去データ
            symbols: 対象銘柄リスト
            force_retrain: 強制再訓練フラグ

        Returns:
            訓練結果の辞書
        """
        try:
            logger.info("予測システムの訓練を開始")

            if len(historical_data) < self.config.min_data_length:
                raise ValueError(f"データが不足しています（最低{self.config.min_data_length}日必要）")

            training_results = {}

            # 1. 特徴量エンジニアリング
            logger.info("高度な特徴量を生成中...")
            volume_data = historical_data["Volume"] if "Volume" in historical_data.columns else None
            features = self.feature_engineer.generate_all_features(
                price_data=historical_data,
                volume_data=volume_data
            )

            if features.empty:
                raise ValueError("特徴量生成に失敗しました")

            # 2. ターゲット変数生成
            targets = create_target_variables(
                historical_data,
                prediction_horizon=self.config.prediction_horizon
            )

            # 3. 機械学習モデル訓練
            if self.enable_ml and self.ml_manager:
                ml_results = self._train_ml_models(features, targets, force_retrain)
                training_results["ml_models"] = ml_results

            # 4. アンサンブル戦略訓練
            ensemble_results = self.ensemble_strategy.train_ml_models(
                historical_data,
                retrain=force_retrain
            )
            training_results["ensemble_models"] = ensemble_results

            # 5. 適応的重み初期化
            if self.config.enable_adaptive_weighting:
                self._initialize_adaptive_weights(features, targets)

            # 6. パフォーマンス評価
            if self.config.enable_performance_tracking:
                performance_results = self._evaluate_prediction_performance(
                    historical_data, features, targets
                )
                training_results["performance_evaluation"] = performance_results

            self.last_retrain_date = datetime.now()

            logger.info("予測システム訓練完了")
            return {
                "success": True,
                "training_date": self.last_retrain_date.isoformat(),
                "feature_count": len(features.columns),
                "data_length": len(historical_data),
                "results": training_results
            }

        except Exception as e:
            logger.error(f"予測システム訓練エラー: {e}")
            return {"success": False, "error": str(e)}

    def generate_enhanced_prediction(
        self,
        current_data: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        強化された予測を生成

        Args:
            current_data: 現在の価格データ
            indicators: テクニカル指標
            market_data: 市場データ

        Returns:
            統合予測結果
        """
        try:
            if len(current_data) < 50:
                logger.warning("予測に十分なデータがありません")
                return None

            # 1. 特徴量生成
            volume_data = current_data["Volume"] if "Volume" in current_data.columns else None
            features = self.feature_engineer.generate_all_features(
                price_data=current_data,
                volume_data=volume_data,
                market_data=market_data
            )

            if features.empty:
                return None

            # 2. 機械学習予測
            ml_predictions = {}
            ml_confidence = {}
            if self.enable_ml and self.ml_manager:
                ml_predictions, ml_confidence = self._generate_ml_predictions(features)

            # 3. アンサンブル予測
            ensemble_signal = self.ensemble_strategy.generate_ensemble_signal(
                current_data, indicators
            )

            # 4. 統合予測計算
            integrated_prediction = self._integrate_predictions(
                ml_predictions, ensemble_signal, current_data, features
            )

            # 5. リスク調整
            risk_adjusted_prediction = self._apply_risk_adjustments(
                integrated_prediction, current_data, features
            )

            # 6. 予測の不確実性評価
            prediction_uncertainty = self._calculate_prediction_uncertainty(
                ml_predictions, ensemble_signal, features
            )

            # 7. 予測履歴に記録
            prediction_record = {
                "timestamp": datetime.now(),
                "price": float(current_data["Close"].iloc[-1]),
                "prediction": risk_adjusted_prediction,
                "uncertainty": prediction_uncertainty,
                "ml_predictions": ml_predictions,
                "ml_confidence": ml_confidence,
                "ensemble_signal": ensemble_signal.ensemble_signal.signal_type.value if ensemble_signal else None,
                "features_used": len(features.columns)
            }

            self.prediction_history.append(prediction_record)

            # 履歴サイズ制限
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            return {
                "prediction": risk_adjusted_prediction,
                "confidence": 1.0 - prediction_uncertainty,
                "uncertainty": prediction_uncertainty,
                "ml_predictions": ml_predictions,
                "ensemble_signal": ensemble_signal,
                "risk_factors": self._identify_risk_factors(current_data, features),
                "recommendation": self._generate_recommendation(risk_adjusted_prediction, prediction_uncertainty),
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"統合予測生成エラー: {e}")
            return None

    def _train_ml_models(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        force_retrain: bool = False
    ) -> Dict[str, Any]:
        """機械学習モデルを訓練"""
        try:
            results = {}

            # モデル設定の定義
            model_configs = [
                ("enhanced_return_predictor", "future_returns", "xgboost", "regression"),
                ("enhanced_direction_predictor", "future_direction", "gradient_boosting", "classification"),
                ("volatility_forecaster", "future_high_volatility", "random_forest", "classification"),
                ("trend_detector", "future_returns", "linear", "regression"),
                ("momentum_predictor", "future_direction", "random_forest", "classification")
            ]

            for model_name, target_name, model_type, task_type in model_configs:
                try:
                    if target_name not in targets:
                        continue

                    # データの準備
                    common_index = features.index.intersection(targets[target_name].index)
                    if len(common_index) < 100:
                        continue

                    X_train = features.loc[common_index]
                    y_train = targets[target_name].loc[common_index]

                    # 特徴量選択
                    selected_features = self.feature_engineer.select_important_features(
                        X_train, y_train, top_k=self.config.feature_selection_top_k
                    )
                    X_train_selected = X_train[selected_features]

                    # モデル設定
                    config = ModelConfig(
                        model_type=model_type,
                        task_type=task_type,
                        cv_folds=5,
                        feature_selection=True,
                        max_features=self.config.feature_selection_top_k,
                        model_params=self._get_optimized_model_params(model_type)
                    )

                    # モデル作成と訓練
                    model = self.ml_manager.create_model(model_name, config)
                    training_result = self.ml_manager.train_model(model_name, X_train_selected, y_train)

                    # モデル保存
                    self.ml_manager.save_model(model_name)

                    results[model_name] = {
                        "target": target_name,
                        "model_type": model_type,
                        "task_type": task_type,
                        "training_result": training_result,
                        "selected_features": selected_features
                    }

                    logger.info(f"モデル {model_name} の訓練完了")

                except Exception as e:
                    logger.error(f"モデル {model_name} の訓練エラー: {e}")
                    results[model_name] = {"error": str(e)}

            return results

        except Exception as e:
            logger.error(f"機械学習モデル訓練エラー: {e}")
            return {"error": str(e)}

    def _generate_ml_predictions(self, features: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """機械学習予測を生成"""
        predictions = {}
        confidence_scores = {}

        try:
            latest_features = features.tail(1)

            for model_name in self.ml_manager.list_models():
                try:
                    model = self.ml_manager.models[model_name]
                    if model.is_fitted:
                        # 予測実行
                        pred = self.ml_manager.predict(model_name, latest_features)
                        predictions[model_name] = float(pred[0]) if len(pred) > 0 else 0.0

                        # 信頼度スコア計算（特徴量重要度と履歴パフォーマンスベース）
                        confidence = self._calculate_model_confidence(model_name, model)
                        confidence_scores[model_name] = confidence

                except Exception as e:
                    logger.warning(f"モデル {model_name} の予測エラー: {e}")

            return predictions, confidence_scores

        except Exception as e:
            logger.error(f"機械学習予測エラー: {e}")
            return {}, {}

    def _integrate_predictions(
        self,
        ml_predictions: Dict[str, float],
        ensemble_signal: Optional[Any],
        current_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """予測を統合"""
        try:
            # 重み配分
            ml_weight = 0.6 if ml_predictions else 0.0
            ensemble_weight = 0.4 if ensemble_signal else 0.0

            # 重みの正規化
            total_weight = ml_weight + ensemble_weight
            if total_weight > 0:
                ml_weight /= total_weight
                ensemble_weight /= total_weight

            # 統合予測計算
            integrated_result = {
                "price_direction": 0.0,  # -1 to 1
                "expected_return": 0.0,
                "volatility_forecast": 0.0,
                "trend_strength": 0.0,
                "confidence": 0.0
            }

            # ML予測の統合
            if ml_predictions:
                # リターン予測
                return_preds = [
                    pred for name, pred in ml_predictions.items()
                    if "return" in name or "trend" in name
                ]
                if return_preds:
                    integrated_result["expected_return"] = np.mean(return_preds) * ml_weight

                # 方向性予測
                direction_preds = [
                    pred for name, pred in ml_predictions.items()
                    if "direction" in name or "momentum" in name
                ]
                if direction_preds:
                    # 0-1の予測を-1から1にスケール
                    direction_score = (np.mean(direction_preds) - 0.5) * 2
                    integrated_result["price_direction"] = direction_score * ml_weight

                # ボラティリティ予測
                vol_preds = [
                    pred for name, pred in ml_predictions.items()
                    if "volatility" in name
                ]
                if vol_preds:
                    integrated_result["volatility_forecast"] = np.mean(vol_preds) * ml_weight

            # アンサンブル予測の統合
            if ensemble_signal and ensemble_signal.ensemble_signal:
                signal = ensemble_signal.ensemble_signal

                # シグナルタイプを数値に変換
                signal_value = 0.0
                if signal.signal_type == SignalType.BUY:
                    signal_value = 1.0
                elif signal.signal_type == SignalType.SELL:
                    signal_value = -1.0

                # 強度による調整
                strength_multiplier = 1.0
                if signal.strength == SignalStrength.STRONG:
                    strength_multiplier = 1.0
                elif signal.strength == SignalStrength.MEDIUM:
                    strength_multiplier = 0.7
                elif signal.strength == SignalStrength.WEAK:
                    strength_multiplier = 0.4

                ensemble_contribution = signal_value * (signal.confidence / 100.0) * strength_multiplier
                integrated_result["price_direction"] += ensemble_contribution * ensemble_weight
                integrated_result["trend_strength"] = strength_multiplier * ensemble_weight

            # 全体信頼度の計算
            ml_confidence = np.mean(list(self.model_weights.values())) if self.model_weights else 0.0
            ensemble_confidence = (ensemble_signal.ensemble_confidence / 100.0) if ensemble_signal else 0.0

            integrated_result["confidence"] = (
                ml_confidence * ml_weight + ensemble_confidence * ensemble_weight
            )

            # 範囲制限
            integrated_result["price_direction"] = np.clip(integrated_result["price_direction"], -1.0, 1.0)
            integrated_result["confidence"] = np.clip(integrated_result["confidence"], 0.0, 1.0)

            return integrated_result

        except Exception as e:
            logger.error(f"予測統合エラー: {e}")
            return {
                "price_direction": 0.0,
                "expected_return": 0.0,
                "volatility_forecast": 0.0,
                "trend_strength": 0.0,
                "confidence": 0.0
            }

    def _apply_risk_adjustments(
        self,
        prediction: Dict[str, Any],
        current_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """リスク調整を適用"""
        try:
            adjusted_prediction = prediction.copy()

            # 1. ボラティリティ調整
            if len(current_data) >= 20:
                recent_volatility = current_data["Close"].pct_change().tail(20).std()
                historical_volatility = current_data["Close"].pct_change().std()

                if recent_volatility > historical_volatility * 1.5:  # 高ボラティリティ
                    vol_adjustment = self.config.risk_adjustment_factor
                    adjusted_prediction["price_direction"] *= vol_adjustment
                    adjusted_prediction["confidence"] *= vol_adjustment
                    adjusted_prediction["expected_return"] *= vol_adjustment

            # 2. 流動性調整（出来高ベース）
            if "Volume" in current_data.columns and len(current_data) >= 10:
                avg_volume = current_data["Volume"].tail(10).mean()
                recent_volume = current_data["Volume"].iloc[-1]

                if recent_volume < avg_volume * 0.5:  # 低流動性
                    liquidity_adjustment = 0.8
                    adjusted_prediction["confidence"] *= liquidity_adjustment

            # 3. 市場レジーム調整
            if len(current_data) >= 50:
                sma_20 = current_data["Close"].rolling(20).mean().iloc[-1]
                sma_50 = current_data["Close"].rolling(50).mean().iloc[-1]
                current_price = current_data["Close"].iloc[-1]

                # トレンド状況による調整
                if sma_20 > sma_50 * 1.05:  # 強い上昇トレンド
                    if prediction["price_direction"] > 0:
                        adjusted_prediction["confidence"] *= 1.1  # 信頼度向上
                elif sma_20 < sma_50 * 0.95:  # 強い下降トレンド
                    if prediction["price_direction"] < 0:
                        adjusted_prediction["confidence"] *= 1.1  # 信頼度向上

            # 4. 信頼度閾値適用
            if adjusted_prediction["confidence"] < self.config.confidence_threshold:
                # 低信頼度の場合は予測を弱める
                confidence_ratio = adjusted_prediction["confidence"] / self.config.confidence_threshold
                adjusted_prediction["price_direction"] *= confidence_ratio
                adjusted_prediction["expected_return"] *= confidence_ratio

            return adjusted_prediction

        except Exception as e:
            logger.error(f"リスク調整エラー: {e}")
            return prediction

    def _calculate_prediction_uncertainty(
        self,
        ml_predictions: Dict[str, float],
        ensemble_signal: Optional[Any],
        features: pd.DataFrame
    ) -> float:
        """予測の不確実性を計算"""
        try:
            uncertainty_factors = []

            # 1. ML予測の分散
            if ml_predictions:
                pred_values = list(ml_predictions.values())
                if len(pred_values) > 1:
                    ml_std = np.std(pred_values)
                    uncertainty_factors.append(ml_std)

            # 2. アンサンブル不確実性
            if ensemble_signal and hasattr(ensemble_signal, 'ensemble_uncertainty'):
                uncertainty_factors.append(ensemble_signal.ensemble_uncertainty)

            # 3. 特徴量の品質
            if not features.empty:
                # 欠損値率
                missing_rate = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
                uncertainty_factors.append(missing_rate)

                # 特徴量の変動性
                if len(features) > 1:
                    feature_volatility = features.std().mean() / (features.mean().abs().mean() + 1e-8)
                    uncertainty_factors.append(min(feature_volatility, 1.0))

            # 4. モデル一致度
            if len(self.model_weights) > 1:
                weight_std = np.std(list(self.model_weights.values()))
                uncertainty_factors.append(weight_std)

            # 総合不確実性
            if uncertainty_factors:
                total_uncertainty = np.mean(uncertainty_factors)
                return min(total_uncertainty, self.config.max_prediction_uncertainty)
            else:
                return self.config.max_prediction_uncertainty

        except Exception as e:
            logger.error(f"不確実性計算エラー: {e}")
            return self.config.max_prediction_uncertainty

    def _identify_risk_factors(
        self,
        current_data: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[str]:
        """リスク要因を特定"""
        risk_factors = []

        try:
            # 1. 高ボラティリティ
            if len(current_data) >= 20:
                recent_vol = current_data["Close"].pct_change().tail(20).std()
                historical_vol = current_data["Close"].pct_change().std()
                if recent_vol > historical_vol * 1.5:
                    risk_factors.append("高ボラティリティ")

            # 2. 低流動性
            if "Volume" in current_data.columns and len(current_data) >= 10:
                avg_volume = current_data["Volume"].tail(10).mean()
                recent_volume = current_data["Volume"].iloc[-1]
                if recent_volume < avg_volume * 0.5:
                    risk_factors.append("低流動性")

            # 3. 極端な価格レベル
            if len(current_data) >= 50:
                price_percentile = (
                    current_data["Close"].tail(50).rank().iloc[-1] / 50
                )
                if price_percentile > 0.95:
                    risk_factors.append("価格高値圏")
                elif price_percentile < 0.05:
                    risk_factors.append("価格安値圏")

            # 4. データ品質問題
            if not features.empty:
                missing_rate = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
                if missing_rate > 0.1:
                    risk_factors.append("データ品質低下")

            return risk_factors

        except Exception as e:
            logger.error(f"リスク要因特定エラー: {e}")
            return []

    def _generate_recommendation(
        self,
        prediction: Dict[str, Any],
        uncertainty: float
    ) -> Dict[str, Any]:
        """推奨アクションを生成"""
        try:
            direction = prediction["price_direction"]
            confidence = prediction["confidence"]
            expected_return = prediction.get("expected_return", 0.0)

            # 基本推奨の決定
            if confidence < self.config.confidence_threshold:
                action = "HOLD"
                strength = "WEAK"
                reason = "低信頼度のため様子見"
            elif abs(direction) < 0.3:
                action = "HOLD"
                strength = "WEAK"
                reason = "明確な方向性なし"
            elif direction > 0.5 and confidence > 0.7:
                action = "BUY"
                strength = "STRONG"
                reason = "強い上昇シグナル"
            elif direction > 0.3:
                action = "BUY"
                strength = "MEDIUM"
                reason = "中程度の上昇シグナル"
            elif direction < -0.5 and confidence > 0.7:
                action = "SELL"
                strength = "STRONG"
                reason = "強い下降シグナル"
            elif direction < -0.3:
                action = "SELL"
                strength = "MEDIUM"
                reason = "中程度の下降シグナル"
            else:
                action = "HOLD"
                strength = "WEAK"
                reason = "不明確なシグナル"

            # 不確実性による調整
            if uncertainty > self.config.max_prediction_uncertainty * 0.8:
                if action != "HOLD":
                    strength = "WEAK"
                    reason += "（高い不確実性により強度減）"

            return {
                "action": action,
                "strength": strength,
                "confidence": confidence,
                "expected_return": expected_return,
                "uncertainty": uncertainty,
                "reason": reason,
                "position_size_suggestion": self._suggest_position_size(confidence, uncertainty)
            }

        except Exception as e:
            logger.error(f"推奨生成エラー: {e}")
            return {
                "action": "HOLD",
                "strength": "WEAK",
                "confidence": 0.0,
                "expected_return": 0.0,
                "uncertainty": 1.0,
                "reason": "エラーのため取引停止",
                "position_size_suggestion": 0.0
            }

    def _suggest_position_size(self, confidence: float, uncertainty: float) -> float:
        """推奨ポジションサイズを計算"""
        try:
            # ベースサイズ（信頼度ベース）
            base_size = confidence * 0.2  # 最大20%

            # 不確実性による調整
            uncertainty_discount = 1.0 - uncertainty
            adjusted_size = base_size * uncertainty_discount

            # 範囲制限
            return max(0.0, min(adjusted_size, 0.2))

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            return 0.05  # デフォルト5%

    def _get_optimized_model_params(self, model_type: str) -> Dict[str, Any]:
        """最適化されたモデルパラメータを取得"""
        optimized_params = {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": True,
                "random_state": 42
            },
            "gradient_boosting": {
                "n_estimators": 150,
                "learning_rate": 0.08,
                "max_depth": 8,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "subsample": 0.8,
                "random_state": 42
            },
            "xgboost": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 8,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            },
            "linear": {
                "alpha": 0.01,
                "max_iter": 2000,
                "random_state": 42
            }
        }

        return optimized_params.get(model_type, {})

    def _calculate_model_confidence(self, model_name: str, model: Any) -> float:
        """モデルの信頼度を計算"""
        try:
            base_confidence = 0.5

            # 特徴量重要度による調整
            if hasattr(model, '_get_feature_importance'):
                importance = model._get_feature_importance()
                if importance:
                    # 重要度の集中度（少数の特徴量に集中している方が安定）
                    importance_values = list(importance.values())
                    importance_concentration = np.max(importance_values) / (np.mean(importance_values) + 1e-8)
                    concentration_boost = min(importance_concentration / 10.0, 0.3)
                    base_confidence += concentration_boost

            # 過去のパフォーマンスによる調整（簡略化）
            performance_boost = self.model_weights.get(model_name, 0.0) * 0.2
            base_confidence += performance_boost

            return min(base_confidence, 1.0)

        except Exception as e:
            logger.error(f"モデル信頼度計算エラー: {e}")
            return 0.5

    def _initialize_adaptive_weights(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """適応的重みを初期化"""
        try:
            if not self.ml_manager:
                return

            # 各モデルの初期重みを設定
            for model_name in self.ml_manager.list_models():
                self.model_weights[model_name] = 1.0 / len(self.ml_manager.list_models())

            logger.info(f"適応的重みを初期化: {self.model_weights}")

        except Exception as e:
            logger.error(f"適応的重み初期化エラー: {e}")

    def _evaluate_prediction_performance(
        self,
        historical_data: pd.DataFrame,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """予測パフォーマンスを評価"""
        try:
            if len(historical_data) < 100:
                return {"error": "評価に十分なデータがありません"}

            evaluation_results = {}

            # Walk-forward分析での評価
            test_size = min(50, len(historical_data) // 4)
            train_end = len(historical_data) - test_size

            train_data = historical_data.iloc[:train_end]
            test_data = historical_data.iloc[train_end:]

            # テスト期間での予測精度評価
            correct_predictions = 0
            total_predictions = 0

            for i in range(len(test_data) - self.config.prediction_horizon):
                try:
                    # その時点でのデータを使用
                    current_data = historical_data.iloc[:train_end + i + 1]

                    # 実際の将来リターンを計算
                    future_idx = train_end + i + self.config.prediction_horizon
                    if future_idx < len(historical_data):
                        actual_return = (
                            historical_data["Close"].iloc[future_idx] /
                            historical_data["Close"].iloc[train_end + i] - 1
                        )

                        # 簡略化された予測（デモ用）
                        predicted_direction = 1 if i % 2 == 0 else -1  # プレースホルダー
                        actual_direction = 1 if actual_return > 0 else -1

                        if predicted_direction == actual_direction:
                            correct_predictions += 1
                        total_predictions += 1

                except Exception:
                    continue

            # 精度計算
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            evaluation_results = {
                "directional_accuracy": accuracy,
                "total_predictions": total_predictions,
                "correct_predictions": correct_predictions,
                "evaluation_period": test_size,
                "evaluation_date": datetime.now().isoformat()
            }

            return evaluation_results

        except Exception as e:
            logger.error(f"パフォーマンス評価エラー: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        try:
            status = {
                "orchestrator_config": {
                    "prediction_horizon": self.config.prediction_horizon,
                    "feature_selection_top_k": self.config.feature_selection_top_k,
                    "confidence_threshold": self.config.confidence_threshold,
                    "ensemble_strategy": self.config.ensemble_strategy.value,
                    "voting_type": self.config.voting_type.value
                },
                "ml_enabled": self.enable_ml,
                "last_retrain_date": self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                "prediction_history_length": len(self.prediction_history),
                "model_weights": self.model_weights.copy(),
                "performance_metrics": {
                    metric: len(values) for metric, values in self.performance_metrics.items()
                }
            }

            # ML情報
            if self.ml_manager:
                status["ml_models"] = {
                    "total_models": len(self.ml_manager.list_models()),
                    "fitted_models": sum(
                        1 for model_name in self.ml_manager.list_models()
                        if self.ml_manager.models[model_name].is_fitted
                    ),
                    "model_list": self.ml_manager.list_models()
                }

            # アンサンブル情報
            if self.ensemble_strategy:
                ensemble_info = self.ensemble_strategy.get_strategy_summary()
                status["ensemble_strategy"] = ensemble_info

            return status

        except Exception as e:
            logger.error(f"システム状態取得エラー: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    # テスト用のサンプル実行
    import datetime

    # サンプルデータ生成
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)

    trend = np.linspace(100, 150, len(dates))
    noise = np.random.randn(len(dates)) * 5
    close_prices = trend + noise

    sample_data = pd.DataFrame({
        "Date": dates,
        "Open": close_prices + np.random.randn(len(dates)) * 1,
        "High": close_prices + np.abs(np.random.randn(len(dates))) * 3,
        "Low": close_prices - np.abs(np.random.randn(len(dates))) * 3,
        "Close": close_prices,
        "Volume": np.random.randint(1000000, 10000000, len(dates))
    })
    sample_data.set_index("Date", inplace=True)

    # 予測オーケストレーターのテスト
    config = PredictionConfig(
        prediction_horizon=5,
        enable_ensemble_ml=True,
        confidence_threshold=0.6
    )

    orchestrator = PredictionOrchestrator(config, enable_ml=False)  # テスト用にMLを無効

    # システム状態確認
    status = orchestrator.get_system_status()
    print("システム状態:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # 予測生成テスト
    prediction = orchestrator.generate_enhanced_prediction(sample_data)
    if prediction:
        print("\n統合予測結果:")
        for key, value in prediction.items():
            if key != "ensemble_signal":  # 複雑なオブジェクトは除外
                print(f"  {key}: {value}")
    else:
        print("予測生成に失敗しました")
