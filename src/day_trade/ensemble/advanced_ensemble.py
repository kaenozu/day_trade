#!/usr/bin/env python3
"""
統合アンサンブル予測システム
Advanced Ensemble Prediction System

Issue #762: 高度なアンサンブル予測システムの強化 - Phase 5
"""

import numpy as np
import pandas as pd
import asyncio
import logging
import time
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import pickle
import json

# Machine Learning
try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
except ImportError:
    warnings.warn("scikit-learn not available")

# 内部コンポーネント
from .adaptive_weighting import AdaptiveWeightingEngine
from .meta_learning import MetaLearnerEngine, Task
from .ensemble_optimizer import EnsembleOptimizer
from .performance_analyzer import EnsembleAnalyzer

# ログ設定
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """予測結果"""
    predictions: np.ndarray
    confidence_scores: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    ensemble_weights: np.ndarray
    metadata: Dict[str, Any]
    timestamp: float
    processing_time: float

@dataclass
class EnsembleConfig:
    """アンサンブル設定"""
    enable_adaptive_weighting: bool = True
    enable_meta_learning: bool = True
    enable_optimization: bool = True
    enable_analysis: bool = True

    # 動的重み付け設定
    adaptive_weighting_config: Dict[str, Any] = field(default_factory=lambda: {
        "lookback_window": 252,
        "regime_threshold": 0.05,
        "weight_momentum": 0.9
    })

    # メタ学習設定
    meta_learning_config: Dict[str, Any] = field(default_factory=lambda: {
        "inner_lr": 0.01,
        "outer_lr": 0.001,
        "adaptation_steps": 5
    })

    # 最適化設定
    optimization_config: Dict[str, Any] = field(default_factory=lambda: {
        "optimization_budget": 50,
        "population_size": 20,
        "max_generations": 10
    })

    # 分析設定
    analysis_config: Dict[str, Any] = field(default_factory=lambda: {
        "attribution_method": "shap",
        "decomposition_depth": 2
    })

class AdvancedEnsembleSystem:
    """高度なアンサンブル予測システム"""

    def __init__(self,
                 models: Optional[List[Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 enable_adaptive_weighting: bool = True,
                 enable_meta_learning: bool = True,
                 enable_optimization: bool = True,
                 enable_analysis: bool = True):

        # 設定初期化
        self.config = self._init_config(
            config,
            enable_adaptive_weighting,
            enable_meta_learning,
            enable_optimization,
            enable_analysis
        )

        # ベースモデル
        self.models = models or self._create_default_models()
        self.model_names = [f"model_{i}" for i in range(len(self.models))]

        # コンポーネント初期化
        self._init_components()

        # 状態管理
        self.is_fitted = False
        self.training_history = []
        self.prediction_cache = {}

        logger.info(f"AdvancedEnsembleSystem initialized with {len(self.models)} models")

    def _init_config(self,
                    config: Optional[Dict[str, Any]],
                    enable_adaptive_weighting: bool,
                    enable_meta_learning: bool,
                    enable_optimization: bool,
                    enable_analysis: bool) -> EnsembleConfig:
        """設定初期化"""
        if config is None:
            config = {}

        return EnsembleConfig(
            enable_adaptive_weighting=enable_adaptive_weighting,
            enable_meta_learning=enable_meta_learning,
            enable_optimization=enable_optimization,
            enable_analysis=enable_analysis,
            adaptive_weighting_config=config.get('adaptive_weighting', {}),
            meta_learning_config=config.get('meta_learning', {}),
            optimization_config=config.get('optimization', {}),
            analysis_config=config.get('analysis', {})
        )

    def _create_default_models(self) -> List[Any]:
        """デフォルトモデル作成"""
        try:
            models = [
                LinearRegression(),
                RandomForestRegressor(n_estimators=50, random_state=42),
                GradientBoostingRegressor(n_estimators=50, random_state=42)
            ]
            return models
        except Exception as e:
            logger.warning(f"Could not create sklearn models: {e}")
            return []

    def _init_components(self) -> None:
        """コンポーネント初期化"""
        try:
            # 動的重み付けエンジン
            if self.config.enable_adaptive_weighting:
                self.weighting_engine = AdaptiveWeightingEngine(
                    n_models=len(self.models),
                    **self.config.adaptive_weighting_config
                )
            else:
                self.weighting_engine = None

            # メタ学習エンジン（入力次元は後で設定）
            if self.config.enable_meta_learning:
                self.meta_learner = None  # fit時に初期化
            else:
                self.meta_learner = None

            # アンサンブル最適化エンジン
            if self.config.enable_optimization:
                self.optimizer = EnsembleOptimizer(**self.config.optimization_config)
            else:
                self.optimizer = None

            # パフォーマンス分析エンジン
            if self.config.enable_analysis:
                self.analyzer = EnsembleAnalyzer(**self.config.analysis_config)
            else:
                self.analyzer = None

            # スケーラー
            self.scaler = StandardScaler()

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    async def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AdvancedEnsembleSystem':
        """システム学習"""
        start_time = time.time()

        try:
            logger.info("Starting ensemble system training...")

            # データ前処理
            X_scaled = self.scaler.fit_transform(X)

            # ベースモデル学習
            await self._fit_base_models(X_scaled, y)

            # メタ学習エンジン初期化と学習
            if self.config.enable_meta_learning:
                await self._fit_meta_learner(X_scaled, y)

            # アンサンブル最適化
            if self.config.enable_optimization:
                await self._optimize_ensemble(X_scaled, y)

            # 動的重み付けシステム初期化
            if self.config.enable_adaptive_weighting:
                await self._init_adaptive_weighting(X_scaled, y)

            self.is_fitted = True
            training_time = time.time() - start_time

            logger.info(f"Ensemble system training completed in {training_time:.2f}s")

            # 学習履歴記録
            self.training_history.append({
                'timestamp': time.time(),
                'training_time': training_time,
                'data_shape': X.shape,
                'n_models': len(self.models)
            })

            return self

        except Exception as e:
            logger.error(f"Error in ensemble system training: {e}")
            raise

    async def _fit_base_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """ベースモデル学習"""
        logger.info("Training base models...")

        # 並列学習
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(len(self.models), 4)) as executor:
            tasks = []
            for i, model in enumerate(self.models):
                task = loop.run_in_executor(executor, self._fit_single_model, model, X, y, i)
                tasks.append(task)

            await asyncio.gather(*tasks)

        logger.info("Base models training completed")

    def _fit_single_model(self, model: Any, X: np.ndarray, y: np.ndarray, model_idx: int) -> None:
        """単一モデル学習"""
        try:
            model.fit(X, y.ravel())
            logger.debug(f"Model {model_idx} fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting model {model_idx}: {e}")
            raise

    async def _fit_meta_learner(self, X: np.ndarray, y: np.ndarray) -> None:
        """メタ学習エンジン学習"""
        logger.info("Training meta-learner...")

        try:
            # メタ学習エンジン初期化
            self.meta_learner = MetaLearnerEngine(
                input_dim=X.shape[1],
                **self.config.meta_learning_config
            )

            # タスク生成（データ分割）
            tasks = self._create_meta_learning_tasks(X, y)

            # メタ学習実行
            results = await self.meta_learner.meta_learn_batch(tasks)

            logger.info(f"Meta-learning completed with {len(results)} episodes")

        except Exception as e:
            logger.error(f"Error in meta-learning: {e}")
            # メタ学習失敗時はNoneに設定
            self.meta_learner = None

    def _create_meta_learning_tasks(self, X: np.ndarray, y: np.ndarray, n_tasks: int = 5) -> List[Task]:
        """メタ学習タスク生成"""
        tasks = []
        n_samples = X.shape[0]

        for i in range(n_tasks):
            # ランダムサンプリング
            support_size = min(20, n_samples // 3)
            query_size = min(10, n_samples // 6)

            indices = np.random.permutation(n_samples)
            support_indices = indices[:support_size]
            query_indices = indices[support_size:support_size + query_size]

            support_set = (X[support_indices], y[support_indices])
            query_set = (X[query_indices], y[query_indices])

            task = Task(
                task_id=f"meta_task_{i}",
                support_set=support_set,
                query_set=query_set,
                metadata={'domain': 'financial', 'difficulty': 1.0}
            )
            tasks.append(task)

        return tasks

    async def _optimize_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """アンサンブル最適化"""
        if self.optimizer is None:
            return

        logger.info("Optimizing ensemble...")

        try:
            # ベースモデル予測取得
            predictions = await self._get_base_predictions(X)

            # 最適化実行
            optimal_config = await self.optimizer.optimize_ensemble(
                training_data=(predictions, y),
                validation_data=(predictions[-100:], y[-100:]) if len(y) > 100 else (predictions, y)
            )

            logger.info("Ensemble optimization completed")

        except Exception as e:
            logger.error(f"Error in ensemble optimization: {e}")

    async def _init_adaptive_weighting(self, X: np.ndarray, y: np.ndarray) -> None:
        """動的重み付けシステム初期化"""
        if self.weighting_engine is None:
            return

        logger.info("Initializing adaptive weighting...")

        try:
            # 初期予測とパフォーマンス
            predictions = await self._get_base_predictions(X)

            # 初期重み設定
            await self.weighting_engine.initialize_weights(predictions, y)

            logger.info("Adaptive weighting initialization completed")

        except Exception as e:
            logger.error(f"Error in adaptive weighting initialization: {e}")

    async def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """ベースモデル予測取得"""
        predictions = []

        for model in self.models:
            try:
                pred = model.predict(X)
                predictions.append(pred.reshape(-1, 1))
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                # フォールバック予測
                predictions.append(np.zeros((X.shape[0], 1)))

        return np.hstack(predictions)

    async def predict(self, X: np.ndarray, **kwargs) -> PredictionResult:
        """統合予測"""
        if not self.is_fitted:
            raise ValueError("Ensemble system must be fitted before prediction")

        start_time = time.time()

        try:
            # データ前処理
            X_scaled = self.scaler.transform(X)

            # ベースモデル予測
            base_predictions = await self._get_base_predictions(X_scaled)
            individual_preds = {
                self.model_names[i]: base_predictions[:, i]
                for i in range(len(self.model_names))
            }

            # 重み計算
            weights = await self._compute_ensemble_weights(X_scaled, base_predictions)

            # アンサンブル予測
            ensemble_pred = np.average(base_predictions, axis=1, weights=weights.T)

            # メタ学習予測（利用可能な場合）
            meta_pred = None
            if self.meta_learner is not None:
                try:
                    # 簡単なサポートセット作成（実際のデータ使用）
                    support_set = (X_scaled[:5], ensemble_pred[:5].reshape(-1, 1))
                    meta_pred, _ = await self.meta_learner.predict_with_adaptation(
                        support_set, X_scaled
                    )
                except Exception as e:
                    logger.warning(f"Meta-learning prediction failed: {e}")

            # 最終予測（メタ学習との組み合わせ）
            if meta_pred is not None:
                final_pred = 0.7 * ensemble_pred + 0.3 * meta_pred.flatten()
            else:
                final_pred = ensemble_pred

            # 信頼度スコア計算
            confidence_scores = self._compute_confidence_scores(base_predictions, weights)

            # 処理時間
            processing_time = time.time() - start_time

            # 結果作成
            result = PredictionResult(
                predictions=final_pred.reshape(-1, 1),
                confidence_scores=confidence_scores,
                individual_predictions=individual_preds,
                ensemble_weights=weights,
                metadata={
                    'n_models': len(self.models),
                    'meta_learning_used': meta_pred is not None,
                    'adaptive_weighting_used': self.weighting_engine is not None,
                    'prediction_method': 'advanced_ensemble'
                },
                timestamp=time.time(),
                processing_time=processing_time
            )

            return result

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # フォールバック予測
            return PredictionResult(
                predictions=np.zeros((X.shape[0], 1)),
                confidence_scores=np.zeros(X.shape[0]),
                individual_predictions={},
                ensemble_weights=np.ones(len(self.models)) / len(self.models),
                metadata={'error': str(e)},
                timestamp=time.time(),
                processing_time=time.time() - start_time
            )

    async def _compute_ensemble_weights(self, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """アンサンブル重み計算"""
        if self.weighting_engine is not None:
            try:
                # 市場データダミー作成（実装では実際の市場データを使用）
                market_data = pd.DataFrame({
                    'price': np.random.randn(X.shape[0]) * 0.01 + 100,
                    'volume': np.random.randint(1000, 10000, X.shape[0]),
                    'volatility': np.random.randn(X.shape[0]) * 0.02 + 0.2
                })

                # 適応的重み計算
                weights = await self.weighting_engine.update_weights(
                    market_data, predictions, predictions.mean(axis=1)
                )
                return weights
            except Exception as e:
                logger.warning(f"Adaptive weighting failed: {e}")

        # フォールバック：均等重み
        return np.ones((len(self.models), X.shape[0])) / len(self.models)

    def _compute_confidence_scores(self, predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """信頼度スコア計算"""
        try:
            # 予測の標準偏差（不確実性の指標）
            pred_std = np.std(predictions, axis=1)

            # 重みの集中度（高いほど確信）
            weight_concentration = -np.sum(weights.T * np.log(weights.T + 1e-8), axis=1)

            # 正規化した信頼度（0-1）
            confidence = 1.0 / (1.0 + pred_std) * np.exp(-weight_concentration)

            return np.clip(confidence, 0.0, 1.0)

        except Exception as e:
            logger.warning(f"Confidence score computation failed: {e}")
            return np.ones(predictions.shape[0]) * 0.5

    async def analyze_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """パフォーマンス分析"""
        if not self.is_fitted:
            raise ValueError("System must be fitted before analysis")

        try:
            # 予測実行
            result = await self.predict(X)

            # 基本メトリクス
            mse = mean_squared_error(y, result.predictions)
            mae = mean_absolute_error(y, result.predictions)
            r2 = r2_score(y, result.predictions)

            analysis = {
                'basic_metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mse))
                },
                'ensemble_info': {
                    'n_models': len(self.models),
                    'avg_confidence': float(np.mean(result.confidence_scores)),
                    'processing_time': float(result.processing_time)
                }
            }

            # 詳細分析（アナライザー利用可能時）
            if self.analyzer is not None:
                try:
                    detailed_analysis = await self.analyzer.analyze_ensemble_performance(
                        self, X, y
                    )
                    analysis['detailed_analysis'] = detailed_analysis
                except Exception as e:
                    logger.warning(f"Detailed analysis failed: {e}")

            return analysis

        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return {'error': str(e)}

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            'is_fitted': self.is_fitted,
            'n_models': len(self.models),
            'components': {
                'adaptive_weighting': self.weighting_engine is not None,
                'meta_learning': self.meta_learner is not None,
                'optimization': self.optimizer is not None,
                'analysis': self.analyzer is not None
            },
            'training_history': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }

    def save_system(self, filepath: str) -> None:
        """システム保存"""
        try:
            system_data = {
                'models': self.models,
                'config': self.config,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted,
                'training_history': self.training_history
            }

            with open(filepath, 'wb') as f:
                pickle.dump(system_data, f)

            logger.info(f"System saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving system: {e}")
            raise

    @classmethod
    def load_system(cls, filepath: str) -> 'AdvancedEnsembleSystem':
        """システム読み込み"""
        try:
            with open(filepath, 'rb') as f:
                system_data = pickle.load(f)

            # システム再構築
            system = cls(
                models=system_data['models'],
                config=system_data['config'].__dict__ if hasattr(system_data['config'], '__dict__') else system_data['config']
            )

            system.scaler = system_data['scaler']
            system.is_fitted = system_data['is_fitted']
            system.training_history = system_data['training_history']

            logger.info(f"System loaded from {filepath}")
            return system

        except Exception as e:
            logger.error(f"Error loading system: {e}")
            raise

# 便利関数
async def create_and_train_ensemble(X: np.ndarray,
                                  y: np.ndarray,
                                  models: Optional[List[Any]] = None,
                                  config: Optional[Dict[str, Any]] = None) -> AdvancedEnsembleSystem:
    """アンサンブルシステム作成・学習"""
    system = AdvancedEnsembleSystem(models=models, config=config)
    await system.fit(X, y)
    return system

async def demo_ensemble_system():
    """アンサンブルシステムデモ"""
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, 1)
    y = (X @ true_weights + np.random.randn(n_samples, 1) * 0.1).flatten()

    print("Creating advanced ensemble system...")

    # システム作成・学習
    system = await create_and_train_ensemble(X, y)

    print("System status:")
    print(json.dumps(system.get_system_status(), indent=2))

    # 予測テスト
    X_test = np.random.randn(100, n_features)
    y_test = (X_test @ true_weights + np.random.randn(100, 1) * 0.1).flatten()

    print("\nMaking predictions...")
    result = await system.predict(X_test)

    print(f"Prediction shape: {result.predictions.shape}")
    print(f"Average confidence: {np.mean(result.confidence_scores):.3f}")
    print(f"Processing time: {result.processing_time:.3f}s")

    # パフォーマンス分析
    print("\nAnalyzing performance...")
    analysis = await system.analyze_performance(X_test, y_test)

    print("Performance metrics:")
    if 'basic_metrics' in analysis:
        for metric, value in analysis['basic_metrics'].items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(demo_ensemble_system())