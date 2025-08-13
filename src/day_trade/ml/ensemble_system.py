#!/usr/bin/env python3
"""
Ensemble Learning System for Stock Prediction

Issue #462: アンサンブル学習システムのメイン実装
複数のベースモデルを統合し、予測精度95%超を目指す
"""

import time
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from .base_models import RandomForestModel, GradientBoostingModel, SVRModel, BaseModelInterface
from .base_models.base_model_interface import ModelPrediction, ModelMetrics
from .stacking_ensemble import StackingEnsemble, StackingConfig
from .dynamic_weighting_system import DynamicWeightingSystem, DynamicWeightingConfig
from .advanced_ml_interface import (
    AdvancedMLEngineInterface,
    LSTMTransformerEngine,
    AdvancedModelType,
    create_advanced_ml_engine
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class EnsembleMethod(Enum):
    """アンサンブル手法"""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    WEIGHTED = "weighted"


@dataclass
class EnsembleConfig:
    """アンサンブル設定"""
    # 使用するモデル
    use_lstm_transformer: bool = True
    use_random_forest: bool = True
    use_gradient_boosting: bool = True
    use_svr: bool = True

    # アンサンブル手法
    ensemble_methods: List[EnsembleMethod] = field(
        default_factory=lambda: [EnsembleMethod.VOTING, EnsembleMethod.WEIGHTED]
    )

    # 重み付け設定
    enable_dynamic_weighting: bool = True
    weight_update_frequency: int = 100  # サンプル数
    performance_window: int = 500  # パフォーマンス評価ウィンドウ

    # スタッキング設定
    enable_stacking: bool = True
    stacking_config: Optional[StackingConfig] = None

    # 動的重み調整設定
    dynamic_weighting_config: Optional[DynamicWeightingConfig] = None

    # 交差検証設定
    cv_folds: int = 5
    train_test_split: float = 0.8

    # パフォーマンス設定
    n_jobs: int = -1
    verbose: bool = True

    # ベースモデルのハイパーパラメータ
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'enable_hyperopt': True
    })
    gradient_boosting_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'enable_hyperopt': True,
        'early_stopping': True
    })
    svr_params: Dict[str, Any] = field(default_factory=lambda: {
        'kernel': 'rbf',
        'enable_hyperopt': True
    })


@dataclass
class EnsemblePrediction:
    """アンサンブル予測結果"""
    final_predictions: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    ensemble_confidence: np.ndarray
    model_weights: Dict[str, float]
    processing_time: float
    method_used: str


class EnsembleSystem:
    """
    アンサンブル学習システム

    複数のベースモデルを統合し、高精度な株価予測を実現
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        初期化

        Args:
            config: アンサンブル設定
        """
        self.config = config or EnsembleConfig()
        self.base_models: Dict[str, BaseModelInterface] = {}
        self.model_weights: Dict[str, float] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.is_trained = False

        # 高度なアンサンブル機能
        self.stacking_ensemble = None
        self.dynamic_weighting = None

        # Issue #473対応: Advanced ML Engine（明確なインターフェース）
        self.advanced_ml_engine: "Optional[AdvancedMLEngineInterface]" = None
        if self.config.use_lstm_transformer:
            try:
                self.advanced_ml_engine = create_advanced_ml_engine(
                    AdvancedModelType.LSTM_TRANSFORMER
                )
                logger.info(f"Advanced ML Engine初期化完了: {self.advanced_ml_engine.get_model_type().value}")

                # 能力情報をログ出力
                capabilities = self.advanced_ml_engine.get_capabilities()
                logger.info(f"Engine能力: シーケンス予測={capabilities.supports_sequence_prediction}, "
                          f"不確実性定量化={capabilities.supports_uncertainty_quantification}, "
                          f"推論目標時間={capabilities.inference_time_target_ms}ms")
            except Exception as e:
                logger.warning(f"Advanced ML Engine初期化失敗: {e}")

        # ベースモデル初期化
        self._initialize_base_models()

        # 高度アンサンブル機能初期化
        self._initialize_advanced_features()

        # パフォーマンスメトリクス
        self.ensemble_metrics = {}

        logger.info(f"アンサンブルシステム初期化完了: {len(self.base_models)}個のベースモデル")

    def _initialize_base_models(self):
        """ベースモデル初期化"""
        try:
            # Random Forest
            if self.config.use_random_forest:
                self.base_models["random_forest"] = RandomForestModel(self.config.random_forest_params)

            # Gradient Boosting
            if self.config.use_gradient_boosting:
                self.base_models["gradient_boosting"] = GradientBoostingModel(self.config.gradient_boosting_params)

            # SVR
            if self.config.use_svr:
                self.base_models["svr"] = SVRModel(self.config.svr_params)

            # 均等重みで初期化
            n_models = len(self.base_models)
            # Issue #473対応: Advanced ML Engine の統合
            if self.advanced_ml_engine and self.advanced_ml_engine.is_trained():
                n_models += 1
                self.model_weights["lstm_transformer"] = 1.0 / n_models

            for model_name in self.base_models.keys():
                self.model_weights[model_name] = 1.0 / n_models

            logger.info(f"初期重み設定: {self.model_weights}")

        except Exception as e:
            logger.error(f"ベースモデル初期化エラー: {e}")
            raise

    def _initialize_advanced_features(self):
        """高度アンサンブル機能初期化"""
        try:
            # スタッキングアンサンブル初期化
            if self.config.enable_stacking and len(self.base_models) >= 2:
                stacking_config = self.config.stacking_config or StackingConfig()
                self.stacking_ensemble = StackingEnsemble(self.base_models, stacking_config)
                logger.info("スタッキングアンサンブル初期化完了")

            # 動的重み調整システム初期化
            if self.config.enable_dynamic_weighting:
                model_names = list(self.base_models.keys())
                # Issue #473対応: Advanced ML Engine の統合
                if self.advanced_ml_engine:
                    model_names.append("lstm_transformer")

                dw_config = self.config.dynamic_weighting_config or DynamicWeightingConfig()
                self.dynamic_weighting = DynamicWeightingSystem(model_names, dw_config)
                logger.info("動的重み調整システム初期化完了")

        except Exception as e:
            logger.warning(f"高度アンサンブル機能初期化エラー: {e}")

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        アンサンブルシステム学習

        Args:
            X: 訓練データの特徴量 (n_samples, n_features)
            y: 訓練データの目標変数 (n_samples,)
            validation_data: 検証データ (X_val, y_val)
            feature_names: 特徴量名リスト

        Returns:
            学習結果辞書
        """
        start_time = time.time()
        logger.info(f"アンサンブル学習開始: データ形状 {X.shape}")

        try:
            # 特徴量名設定
            if feature_names:
                for model in self.base_models.values():
                    model.set_feature_names(feature_names)

            # 各ベースモデルの学習
            model_results = {}

            # 1. Advanced ML Engine学習（Issue #473対応）
            if self.advanced_ml_engine:
                try:
                    logger.info(f"Advanced ML Engine学習開始: {self.advanced_ml_engine.get_model_type().value}")

                    # データ形状の検証
                    if not self.advanced_ml_engine.validate_input_shape(X):
                        logger.warning("Advanced ML Engine: 入力データ形状が不適切です")

                    # 学習実行
                    training_metrics = self.advanced_ml_engine.train(X, y, validation_data)
                    model_results["lstm_transformer"] = {
                        "status": "学習完了",
                        "metrics": training_metrics,
                        "model_type": self.advanced_ml_engine.get_model_type().value
                    }
                    logger.info(f"Advanced ML Engine学習完了: 精度={training_metrics.accuracy:.4f}")

                except Exception as e:
                    logger.warning(f"Advanced ML Engine学習失敗: {e}")
                    model_results["lstm_transformer"] = {"status": "学習失敗", "error": str(e)}

            # 2. 従来MLモデル学習 - Issue #705対応: 並列化実装
            model_results.update(self._parallel_model_training(X, y, validation_data))

            # 3. スタッキングアンサンブル学習
            if self.stacking_ensemble and validation_data:
                logger.info("スタッキングアンサンブル学習開始")
                stacking_results = self.stacking_ensemble.fit(X, y, validation_data)
                model_results["stacking_ensemble"] = stacking_results

            # 4. アンサンブル重み最適化
            if validation_data and self.config.enable_dynamic_weighting:
                self._optimize_ensemble_weights(validation_data[0], validation_data[1])

            # 学習結果まとめ
            training_results = {
                'total_training_time': time.time() - start_time,
                'model_results': model_results,
                'final_weights': self.model_weights.copy(),
                'ensemble_methods': [method.value for method in self.config.ensemble_methods]
            }

            # 検証データでのアンサンブル評価
            if validation_data:
                X_val, y_val = validation_data
                ensemble_metrics = self._evaluate_ensemble(X_val, y_val)
                training_results['ensemble_validation_metrics'] = ensemble_metrics

                logger.info(f"アンサンブル検証RMSE: {ensemble_metrics.get('rmse', 'N/A'):.4f}")
                logger.info(f"アンサンブル Hit Rate: {ensemble_metrics.get('hit_rate', 'N/A'):.3f}")

            self.is_trained = True
            self.performance_history.append(training_results)

            logger.info(f"アンサンブル学習完了: {time.time() - start_time:.2f}秒")
            return training_results

        except Exception as e:
            logger.error(f"アンサンブル学習エラー: {e}")
            raise

    def predict(self, X: np.ndarray, method: Optional[EnsembleMethod] = None) -> EnsemblePrediction:
        """
        アンサンブル予測実行

        Args:
            X: 予測対象の特徴量 (n_samples, n_features)
            method: 使用するアンサンブル手法

        Returns:
            EnsemblePrediction: アンサンブル予測結果
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")

        start_time = time.time()

        try:
            # デフォルト手法選択
            if method is None:
                method = self.config.ensemble_methods[0]

            # 各モデルからの予測収集
            individual_predictions = {}

            # 1. Advanced ML Engine予測（Issue #473対応）
            if self.advanced_ml_engine and self.advanced_ml_engine.is_trained() and "lstm_transformer" in self.model_weights:
                try:
                    # Issue #473対応: 統一インターフェースによる予測
                    if not self.advanced_ml_engine.validate_input_shape(X):
                        logger.warning("Advanced ML Engine: 入力形状が無効")
                        lstm_pred = np.zeros(len(X))  # フォールバック
                    else:
                        transformed_X = self.advanced_ml_engine.prepare_data(X)
                        prediction_result = self.advanced_ml_engine.predict(
                            transformed_X,
                            return_confidence=True,
                            return_attention=False
                        )

                        if hasattr(prediction_result, 'predictions') and prediction_result.predictions is not None:
                            lstm_pred = prediction_result.predictions.flatten()
                            logger.debug(f"Advanced ML Engine予測: 形状{prediction_result.predictions.shape}, "
                                      f"信頼度平均={getattr(prediction_result, 'confidence', 'N/A')}")
                        else:
                            lstm_pred = np.zeros(len(X))

                    individual_predictions["lstm_transformer"] = lstm_pred
                except Exception as e:
                    logger.warning(f"LSTM-Transformer予測失敗: {e}")

            # 2. 従来MLモデル予測 - Issue #705対応: 並列化実装
            parallel_predictions = self._parallel_model_prediction(X)
            individual_predictions.update(parallel_predictions)

            # 3. アンサンブル統合 - Issue #470対応統合実装
            if method == EnsembleMethod.STACKING:
                final_predictions = self._stacking_ensemble(individual_predictions, X)
            else:
                # 統合アンサンブル手法を使用（VOTING/WEIGHTEDを効率的に統合）
                final_predictions = self._unified_ensemble(individual_predictions, method)

            # 信頼度計算
            ensemble_confidence = self._calculate_ensemble_confidence(individual_predictions)

            # 動的重み調整システムへのフィードバック
            if self.dynamic_weighting:
                self.dynamic_weighting.current_weights = self.model_weights.copy()

            processing_time = time.time() - start_time

            return EnsemblePrediction(
                final_predictions=final_predictions,
                individual_predictions=individual_predictions,
                ensemble_confidence=ensemble_confidence,
                model_weights=self.model_weights.copy(),
                processing_time=processing_time,
                method_used=method.value
            )

        except Exception as e:
            logger.error(f"アンサンブル予測エラー: {e}")
            raise

    def _unified_ensemble(self, predictions: Dict[str, np.ndarray], method: EnsembleMethod) -> np.ndarray:
        """
        統合アンサンブル手法 - Issue #470対応

        voting と weighted ensemble を統合した効率的実装

        Args:
            predictions: モデル別予測結果
            method: アンサンブル手法

        Returns:
            統合予測結果
        """
        if not predictions:
            raise ValueError("予測結果が空です")

        # 予測配列とモデル名の準備
        model_names = list(predictions.keys())
        pred_values = list(predictions.values())

        # 形状チェックと標準化
        shapes = [pred.shape for pred in pred_values]
        if len(set(shapes)) > 1:
            logger.warning(f"予測形状が不統一: {shapes}")
            # 最小サイズに合わせる
            min_len = min(pred.shape[0] for pred in pred_values)
            pred_values = [pred[:min_len] for pred in pred_values]

        pred_array = np.array(pred_values)

        if method == EnsembleMethod.VOTING:
            # 投票アンサンブル（効率的な単純平均）
            return np.mean(pred_array, axis=0)

        elif method == EnsembleMethod.WEIGHTED:
            # 重み付きアンサンブル（ベクトル化実装）
            weights = np.array([
                self.model_weights.get(model_name, 1.0)
                for model_name in model_names
            ])

            # 重みの正規化
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(model_names)) / len(model_names)

            # ベクトル化重み付き平均
            return np.average(pred_array, axis=0, weights=weights)

        else:
            # デフォルト: 単純平均
            logger.warning(f"未対応のアンサンブル手法: {method}、単純平均を使用")
            return np.mean(pred_array, axis=0)

    def _voting_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """投票アンサンブル（単純平均）- 後方互換性のため残存"""
        return self._unified_ensemble(predictions, EnsembleMethod.VOTING)

    def _weighted_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """重み付きアンサンブル - 後方互換性のため残存"""
        return self._unified_ensemble(predictions, EnsembleMethod.WEIGHTED)

    def _stacking_ensemble(self, predictions: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """スタッキングアンサンブル（メタ学習）"""
        if self.stacking_ensemble and self.stacking_ensemble.is_fitted:
            try:
                # スタッキングアンサンブルで予測
                stacking_result = self.stacking_ensemble.predict(X)
                return stacking_result.predictions
            except Exception as e:
                logger.warning(f"スタッキング予測失敗: {e}, 重み付きアンサンブルで代替")
                return self._weighted_ensemble(predictions)
        else:
            logger.warning("スタッキングアンサンブル未学習、重み付きアンサンブルで代替")
            return self._weighted_ensemble(predictions)

    def _calculate_ensemble_confidence(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """アンサンブル信頼度計算"""
        if len(predictions) < 2:
            return np.ones(len(list(predictions.values())[0])) * 0.5

        # 予測の分散を信頼度とする（分散が小さいほど信頼度が高い）
        pred_array = np.array(list(predictions.values()))
        prediction_variance = np.var(pred_array, axis=0)

        # 正規化して信頼度に変換（分散が大きいほど信頼度は低い）
        max_var = np.max(prediction_variance)
        if max_var > 0:
            confidence = 1.0 - (prediction_variance / max_var)
        else:
            confidence = np.ones_like(prediction_variance)

        return confidence

    def _optimize_ensemble_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """アンサンブル重み最適化"""
        try:
            from scipy.optimize import minimize

            # 各モデルの予測取得
            model_predictions = {}
            for model_name, model in self.base_models.items():
                if model.is_trained:
                    pred_result = model.predict(X_val)
                    model_predictions[model_name] = pred_result.predictions

            if len(model_predictions) < 2:
                logger.warning("重み最適化に十分なモデルがありません")
                return

            model_names = list(model_predictions.keys())
            pred_matrix = np.array([model_predictions[name] for name in model_names]).T

            # 目的関数：重み付き予測のMSE最小化
            def objective(weights):
                weights = weights / np.sum(weights)  # 正規化
                ensemble_pred = np.dot(pred_matrix, weights)
                mse = np.mean((y_val - ensemble_pred) ** 2)
                return mse

            # 制約：重みの合計=1、各重み>=0
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0.0, 1.0) for _ in model_names]

            # 初期重み（均等）
            initial_weights = np.ones(len(model_names)) / len(model_names)

            # 最適化実行
            result = minimize(objective, initial_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)

            if result.success:
                optimal_weights = result.x / np.sum(result.x)  # 正規化

                # 重み更新
                for i, model_name in enumerate(model_names):
                    self.model_weights[model_name] = optimal_weights[i]

                logger.info(f"重み最適化完了: {dict(zip(model_names, optimal_weights))}")
            else:
                logger.warning("重み最適化失敗、現在の重みを維持")

        except ImportError:
            logger.warning("scipy.optimize未インストール、重み最適化スキップ")
        except Exception as e:
            logger.error(f"重み最適化エラー: {e}")

    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """アンサンブル評価"""
        try:
            ensemble_pred = self.predict(X, method=EnsembleMethod.WEIGHTED)
            y_pred = ensemble_pred.final_predictions

            # 基本指標
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Hit Rate
            if len(y) > 1:
                y_diff = np.diff(y)
                pred_diff = np.diff(y_pred)
                direction_match = np.sign(y_diff) == np.sign(pred_diff)
                hit_rate = np.mean(direction_match)
            else:
                hit_rate = 0.5

            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'hit_rate': hit_rate
            }

        except Exception as e:
            logger.error(f"アンサンブル評価エラー: {e}")
            return {}

    def get_model_performance_comparison(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """個別モデル性能比較"""
        results = []

        # 各ベースモデルの評価
        for model_name, model in self.base_models.items():
            if not model.is_trained:
                continue
            try:
                metrics = model.evaluate(X, y)
                results.append({
                    'model': model_name,
                    'rmse': metrics.rmse,
                    'mae': metrics.mae,
                    'r2_score': metrics.r2_score,
                    'hit_rate': metrics.hit_rate,
                    'weight': self.model_weights.get(model_name, 0.0)
                })
            except Exception as e:
                logger.warning(f"{model_name}評価失敗: {e}")

        # アンサンブル評価
        ensemble_metrics = self._evaluate_ensemble(X, y)
        if ensemble_metrics:
            results.append({
                'model': 'ensemble',
                'rmse': ensemble_metrics['rmse'],
                'mae': ensemble_metrics['mae'],
                'r2_score': ensemble_metrics['r2_score'],
                'hit_rate': ensemble_metrics['hit_rate'],
                'weight': 1.0
            })

        return pd.DataFrame(results).sort_values('rmse')

    def save_ensemble(self, filepath: str, compress: bool = True) -> bool:
        """
        アンサンブルシステム保存 - Issue #706対応最適化版

        Args:
            filepath: 保存先ファイルパス
            compress: 圧縮を有効にするか

        Returns:
            bool: 保存成功フラグ
        """
        try:
            import pickle
            import gzip
            from pathlib import Path

            # ファイル存在確認とディレクトリ作成
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # メタデータのみ保存（軽量化）
            ensemble_data = {
                'config': self.config,
                'model_weights': self.model_weights,
                'performance_history': self.performance_history,
                'is_trained': self.is_trained,
                'ensemble_metrics': self.ensemble_metrics,
                'version': '2.0',  # バージョン管理
                'save_timestamp': time.time()
            }

            # 各モデルは軽量な状態情報のみ保存
            model_data = {}
            for model_name, model in self.base_models.items():
                # 大きなモデルオブジェクトではなく、再構築可能な情報のみ
                model_info = {
                    'model_type': type(model).__name__,
                    'config': model.config,
                    'is_trained': model.is_trained,
                    'feature_names': getattr(model, 'feature_names', []),
                    'training_metrics': getattr(model, 'training_metrics', {}),
                }

                # scikit-learn モデルの場合、学習済みパラメータを抽出
                if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
                    model_info['sklearn_params'] = model.model.get_params()

                model_data[model_name] = model_info

            ensemble_data['models'] = model_data

            # 圧縮保存または通常保存
            if compress:
                with gzip.open(f"{filepath}.gz", 'wb') as f:
                    pickle.dump(ensemble_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_path = f"{filepath}.gz"
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(ensemble_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_path = filepath

            logger.info(f"アンサンブルシステム保存完了: {saved_path}")
            return True

        except Exception as e:
            logger.error(f"アンサンブルシステム保存エラー: {e}")
            return False

    def load_ensemble(self, filepath: str) -> bool:
        """
        アンサンブルシステム読み込み - Issue #706対応新機能

        Args:
            filepath: 読み込みファイルパス

        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            import pickle
            import gzip
            from pathlib import Path

            # ファイル存在確認
            load_path = Path(filepath)
            gz_path = Path(f"{filepath}.gz")

            if gz_path.exists():
                # 圧縮ファイルを読み込み
                with gzip.open(gz_path, 'rb') as f:
                    ensemble_data = pickle.load(f)
                loaded_from = gz_path
            elif load_path.exists():
                # 非圧縮ファイルを読み込み
                with open(load_path, 'rb') as f:
                    ensemble_data = pickle.load(f)
                loaded_from = load_path
            else:
                raise FileNotFoundError(f"アンサンブルファイルが見つかりません: {filepath}")

            # バージョン確認
            version = ensemble_data.get('version', '1.0')
            if version != '2.0':
                logger.warning(f"古いバージョンのアンサンブルファイル: {version}")

            # 基本情報の復元
            self.config = ensemble_data['config']
            self.model_weights = ensemble_data['model_weights']
            self.performance_history = ensemble_data['performance_history']
            self.is_trained = ensemble_data['is_trained']
            self.ensemble_metrics = ensemble_data.get('ensemble_metrics', {})

            # モデルの再構築（軽量版）
            self.base_models = {}
            model_data = ensemble_data['models']

            for model_name, model_info in model_data.items():
                # モデルタイプに基づいて再構築
                model_type = model_info['model_type']
                model_config = model_info['config']

                if model_type == 'RandomForestModel':
                    model = RandomForestModel(model_config)
                elif model_type == 'GradientBoostingModel':
                    model = GradientBoostingModel(model_config)
                elif model_type == 'SVRModel':
                    model = SVRModel(model_config)
                else:
                    logger.warning(f"未対応のモデルタイプ: {model_type}")
                    continue

                # 基本情報の復元
                model.is_trained = model_info['is_trained']
                if 'feature_names' in model_info:
                    model.set_feature_names(model_info['feature_names'])
                model.training_metrics = model_info.get('training_metrics', {})

                self.base_models[model_name] = model

            # サブシステムの初期化（設定のみ）
            self._initialize_subsystems()

            logger.info(f"アンサンブルシステム読み込み完了: {loaded_from}")
            logger.info(f"読み込まれたモデル数: {len(self.base_models)}")
            return True

        except Exception as e:
            logger.error(f"アンサンブルシステム読み込みエラー: {e}")
            return False

    def update_dynamic_weights(self, predictions: Dict[str, np.ndarray],
                              actuals: np.ndarray, timestamp: int = None):
        """
        Issue #472対応: 簡素化された動的重み更新

        DynamicWeightingSystemが内部で完結した重み更新・同期を実行

        Args:
            predictions: モデル別予測値
            actuals: 実際の値
            timestamp: タイムスタンプ
        """
        if self.dynamic_weighting:
            try:
                # Issue #472対応: 一括更新・同期処理
                updated_weights = self.dynamic_weighting.sync_and_update_performance(
                    predictions, actuals, self.model_weights, timestamp
                )

                logger.debug(f"動的重み更新完了: {len(updated_weights)}モデル")

            except Exception as e:
                logger.warning(f"動的重み更新エラー: {e}")

    def create_simplified_weight_updater(self):
        """
        Issue #472対応: 簡潔な重み更新関数の生成

        Returns:
            簡潔な重み更新関数
        """
        if not self.dynamic_weighting:
            return lambda *args, **kwargs: False

        return self.dynamic_weighting.create_weight_updater()

    def get_dynamic_weight_update_strategy(self) -> str:
        """
        Issue #472対応: 動的重み更新戦略の取得

        Returns:
            現在の重み更新戦略の説明
        """
        if not self.dynamic_weighting:
            return "動的重み調整は無効です"

        strategy_info = [
            "統合重み更新戦略:",
            "1. パフォーマンス蓄積",
            "2. 重み再計算（閾値達成時）",
            "3. EnsembleSystem重み直接同期",
            "4. 手動マージ処理の排除"
        ]

        return " → ".join(strategy_info)

    def get_ensemble_info(self) -> Dict[str, Any]:
        """アンサンブル情報取得"""
        info = {
            'is_trained': self.is_trained,
            'n_base_models': len(self.base_models),
            'model_names': list(self.base_models.keys()),
            'model_weights': self.model_weights,
            'ensemble_methods': [method.value for method in self.config.ensemble_methods],
            'performance_history_count': len(self.performance_history)
        }

        # 高度機能の情報追加
        if self.stacking_ensemble:
            info['stacking_info'] = self.stacking_ensemble.get_stacking_info()

        if self.dynamic_weighting:
            info['dynamic_weighting_info'] = self.dynamic_weighting.get_performance_summary()

        return info

    def _parallel_model_training(self, X: np.ndarray, y: np.ndarray,
                                validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        ベースモデルの並列学習 - Issue #705対応

        Args:
            X: 訓練データの特徴量
            y: 訓練データの目標変数
            validation_data: 検証データ

        Returns:
            Dict[str, Any]: 各モデルの学習結果
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
        import multiprocessing as mp

        def train_single_model(args):
            """単一モデルの学習関数"""
            model_name, model, X_data, y_data, val_data = args
            try:
                logger.info(f"{model_name}並列学習開始")
                result = model.fit(X_data, y_data, validation_data=val_data)
                result['training_method'] = 'parallel'
                logger.info(f"{model_name}並列学習完了")
                return model_name, result
            except Exception as e:
                logger.error(f"{model_name}並列学習エラー: {e}")
                return model_name, {"status": "失敗", "error": str(e), "training_method": "parallel"}

        model_results = {}

        if not self.base_models:
            return model_results

        # 並列処理設定
        n_jobs = getattr(self.config, 'n_jobs', -1)
        if n_jobs == -1:
            n_jobs = min(len(self.base_models), mp.cpu_count())
        elif n_jobs <= 0:
            n_jobs = 1

        logger.info(f"並列学習開始: {len(self.base_models)}モデルを{n_jobs}プロセスで実行")

        # 並列学習実行
        training_args = [
            (model_name, model, X, y, validation_data)
            for model_name, model in self.base_models.items()
        ]

        try:
            # ThreadPoolExecutorを使用（scikit-learnモデルはピクル化に問題がある場合があるため）
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_model = {
                    executor.submit(train_single_model, args): args[0]
                    for args in training_args
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result_name, result = future.result(timeout=3600)  # 1時間のタイムアウト
                        model_results[result_name] = result
                    except Exception as e:
                        logger.error(f"{model_name}並列学習例外: {e}")
                        model_results[model_name] = {
                            "status": "失敗",
                            "error": f"並列処理例外: {str(e)}",
                            "training_method": "parallel"
                        }

        except Exception as e:
            logger.warning(f"並列学習フォールバック: {e}")
            # フォールバック: 順次実行
            logger.info("順次学習にフォールバック")
            for model_name, model in self.base_models.items():
                try:
                    logger.info(f"{model_name}学習開始（順次）")
                    result = model.fit(X, y, validation_data=validation_data)
                    result['training_method'] = 'sequential_fallback'
                    model_results[model_name] = result
                    logger.info(f"{model_name}学習完了（順次）")
                except Exception as model_e:
                    logger.error(f"{model_name}学習エラー（順次）: {model_e}")
                    model_results[model_name] = {
                        "status": "失敗",
                        "error": str(model_e),
                        "training_method": "sequential_fallback"
                    }

        return model_results

    def _parallel_model_prediction(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        ベースモデルの並列予測 - Issue #705対応

        Args:
            X: 予測対象の特徴量

        Returns:
            Dict[str, np.ndarray]: 各モデルの予測結果
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp

        def predict_single_model(args):
            """単一モデルの予測関数"""
            model_name, model, X_data = args
            try:
                if not model.is_trained:
                    return model_name, None
                pred_result = model.predict(X_data)
                return model_name, pred_result.predictions
            except Exception as e:
                logger.warning(f"{model_name}並列予測失敗: {e}")
                return model_name, None

        predictions = {}

        # 学習済みモデルのみフィルタリング
        trained_models = {
            name: model for name, model in self.base_models.items()
            if model.is_trained
        }

        if not trained_models:
            return predictions

        # 並列処理設定
        n_jobs = getattr(self.config, 'n_jobs', -1)
        if n_jobs == -1:
            n_jobs = min(len(trained_models), mp.cpu_count())
        elif n_jobs <= 0:
            n_jobs = 1

        # 並列予測実行
        prediction_args = [
            (model_name, model, X)
            for model_name, model in trained_models.items()
        ]

        try:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_model = {
                    executor.submit(predict_single_model, args): args[0]
                    for args in prediction_args
                }

                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result_name, pred = future.result(timeout=300)  # 5分のタイムアウト
                        if pred is not None:
                            predictions[result_name] = pred
                    except Exception as e:
                        logger.warning(f"{model_name}並列予測例外: {e}")

        except Exception as e:
            logger.warning(f"並列予測フォールバック: {e}")
            # フォールバック: 順次実行
            for model_name, model in trained_models.items():
                try:
                    pred_result = model.predict(X)
                    predictions[model_name] = pred_result.predictions
                except Exception as model_e:
                    logger.warning(f"{model_name}予測失敗（順次）: {model_e}")

        return predictions


if __name__ == "__main__":
    # テスト実行
    print("=== Ensemble System テスト ===")

    # テストデータ生成
    np.random.seed(42)
    n_samples, n_features = 1000, 30
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + np.sum(X[:, 5:10]**2, axis=1) + 0.2 * np.random.randn(n_samples)

    # 訓練・検証データ分割
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # アンサンブルシステム初期化
    config = EnsembleConfig(
        use_lstm_transformer=False,  # テスト用に無効化
        enable_dynamic_weighting=True
    )
    ensemble = EnsembleSystem(config)

    # 特徴量名
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # 学習
    print("アンサンブル学習開始...")
    results = ensemble.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          feature_names=feature_names)

    print(f"学習完了: {results['total_training_time']:.2f}秒")
    print(f"最終重み: {results['final_weights']}")

    # 予測
    ensemble_pred = ensemble.predict(X_val, method=EnsembleMethod.WEIGHTED)
    print(f"アンサンブル予測完了: {len(ensemble_pred.final_predictions)} サンプル")

    # 性能比較
    performance_df = ensemble.get_model_performance_comparison(X_val, y_val)
    print("\n=== モデル性能比較 ===")
    print(performance_df)

    # システム情報
    info = ensemble.get_ensemble_info()
    print(f"\nアンサンブル情報: {info}")