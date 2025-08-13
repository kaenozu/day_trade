#!/usr/bin/env python3
"""
Random Forest Model for Ensemble Learning

Issue #462: Random Forestベースモデルの実装
高い解釈性と安定性を提供する決定木アンサンブル
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base_model_interface import BaseModelInterface, ModelPrediction
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class RandomForestModel(BaseModelInterface):
    """
    Random Forest回帰モデル

    特徴:
    - 高い汎化性能
    - 特徴量重要度の解釈性
    - オーバーフィッティング耐性
    - 並列処理による高速学習
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: モデル設定辞書
        """
        default_config = {
            # Random Forest設定
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,  # 全CPU使用

            # 学習設定
            'enable_hyperopt': True,
            'cv_folds': 5,
            'normalize_features': True,

            # パフォーマンス設定
            'verbose': 0,
            'warm_start': False,
        }

        # 設定をマージ
        final_config = {**default_config, **(config or {})}
        super().__init__("RandomForest", final_config)

        # スケーラー
        self.scaler = StandardScaler() if self.config['normalize_features'] else None
        self.best_params = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Random Forestモデル学習

        Args:
            X: 訓練データの特徴量 (n_samples, n_features)
            y: 訓練データの目標変数 (n_samples,)
            validation_data: 検証データ (X_val, y_val)

        Returns:
            学習結果辞書
        """
        start_time = time.time()
        logger.info(f"Random Forest学習開始: データ形状 {X.shape}")

        try:
            # 特徴量正規化
            if self.scaler is not None:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X.copy()

            # ハイパーパラメータ最適化
            if self.config['enable_hyperopt']:
                logger.info("ハイパーパラメータ最適化開始")
                self.model = self._hyperparameter_optimization(X_scaled, y)
            else:
                # デフォルトパラメータで学習
                self.model = RandomForestRegressor(**self._get_rf_params())
                self.model.fit(X_scaled, y)

            # 検証データでの評価
            training_results = {'training_time': time.time() - start_time}

            if validation_data is not None:
                X_val, y_val = validation_data
                if self.scaler is not None:
                    X_val_scaled = self.scaler.transform(X_val)
                else:
                    X_val_scaled = X_val.copy()

                val_metrics = self.evaluate(X_val_scaled, y_val)
                training_results['validation_metrics'] = val_metrics
                logger.info(f"検証RMSE: {val_metrics.rmse:.4f}, Hit Rate: {val_metrics.hit_rate:.3f}")

            # 学習メトリクス保存
            self.training_metrics = training_results
            self.is_trained = True

            # 特徴量重要度取得
            feature_importance = self.get_feature_importance()
            training_results['feature_importance_top10'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            logger.info(f"Random Forest学習完了: {time.time() - start_time:.2f}秒")
            return training_results

        except Exception as e:
            logger.error(f"Random Forest学習エラー: {e}")
            raise

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        予測実行

        Args:
            X: 予測対象の特徴量 (n_samples, n_features)

        Returns:
            ModelPrediction: 予測結果
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")

        start_time = time.time()

        try:
            # 特徴量正規化
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.copy()

            # 予測実行
            predictions = self.model.predict(X_scaled)

            # 予測分布（信頼区間）計算
            confidence = self._calculate_prediction_confidence(X_scaled)

            # 特徴量重要度
            feature_importance = self.get_feature_importance()

            processing_time = time.time() - start_time

            return ModelPrediction(
                predictions=predictions,
                confidence=confidence,
                feature_importance=feature_importance,
                model_name=self.model_name,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Random Forest予測エラー: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度取得

        Returns:
            特徴量名と重要度のマッピング
        """
        if not self.is_trained:
            return {}

        importances = self.model.feature_importances_

        if len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances))
        else:
            # 特徴量名がない場合は番号で返す
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def _hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        """
        ハイパーパラメータ最適化

        Args:
            X: 学習データ特徴量
            y: 学習データ目標変数

        Returns:
            最適化されたRandomForestRegressor
        """
        # 時系列交差検証
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])

        # 探索パラメータ空間
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Base model
        rf_base = RandomForestRegressor(
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs'],
            bootstrap=self.config['bootstrap']
        )

        # Grid Search
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # 既にRFが並列化されているため
            verbose=self.config['verbose']
        )

        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        logger.info(f"最適パラメータ: {self.best_params}")
        logger.info(f"最適CV Score: {-grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _get_rf_params(self) -> Dict[str, Any]:
        """Random Forestパラメータ取得"""
        return {
            'n_estimators': self.config['n_estimators'],
            'max_depth': self.config['max_depth'],
            'min_samples_split': self.config['min_samples_split'],
            'min_samples_leaf': self.config['min_samples_leaf'],
            'max_features': self.config['max_features'],
            'bootstrap': self.config['bootstrap'],
            'random_state': self.config['random_state'],
            'n_jobs': self.config['n_jobs'],
            'verbose': self.config['verbose'],
            'warm_start': self.config['warm_start']
        }

    def _calculate_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        予測信頼度計算

        Random Forestの各決定木からの予測分散を使用

        Args:
            X: 入力特徴量

        Returns:
            各サンプルの予測信頼度 (標準偏差)
        """
        try:
            # 各決定木からの予測取得
            tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])

            # 予測の標準偏差を信頼度とする
            prediction_std = np.std(tree_predictions, axis=0)

            return prediction_std

        except Exception as e:
            logger.warning(f"信頼度計算エラー: {e}")
            return np.zeros(len(X))

    def get_tree_count(self) -> int:
        """決定木の数を取得"""
        return len(self.model.estimators_) if self.is_trained else 0

    def get_model_complexity(self) -> Dict[str, Any]:
        """モデル複雑度情報取得"""
        if not self.is_trained:
            return {}

        return {
            'n_trees': self.get_tree_count(),
            'max_depth': self.config['max_depth'],
            'total_nodes': sum(tree.tree_.node_count for tree in self.model.estimators_),
            'avg_nodes_per_tree': np.mean([tree.tree_.node_count for tree in self.model.estimators_])
        }


if __name__ == "__main__":
    # テスト実行
    print("=== Random Forest Model テスト ===")

    # テストデータ生成
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(n_samples)

    # 訓練・検証データ分割
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # モデル初期化・学習
    rf_model = RandomForestModel({
        'enable_hyperopt': False,  # テスト用に高速化
        'n_estimators': 50
    })

    # 特徴量名設定
    rf_model.set_feature_names([f"feature_{i}" for i in range(n_features)])

    # 学習
    results = rf_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    print(f"学習完了: {results['training_time']:.2f}秒")

    # 予測
    prediction = rf_model.predict(X_val)
    print(f"予測完了: {len(prediction.predictions)} サンプル")
    print(f"RMSE: {np.sqrt(np.mean((y_val - prediction.predictions)**2)):.4f}")

    # 特徴量重要度
    importance = rf_model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"重要特徴量TOP5: {top_features}")

    # モデル複雑度
    complexity = rf_model.get_model_complexity()
    print(f"モデル複雑度: {complexity}")