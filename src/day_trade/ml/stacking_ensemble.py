#!/usr/bin/env python3
"""
Stacking Ensemble Implementation

Issue #462: 高度スタッキングアンサンブル実装
メタ学習器による複数モデル統合で最高精度を実現
"""

import time
from typing import Dict, List, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Issue #692対応: ベイズ最適化ライブラリ
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna未インストール - GridSearchCVを使用", ImportWarning)

from .base_models.base_model_interface import BaseModelInterface, ModelPrediction
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class StackingConfig:
    """
    スタッキング設定
    
    Issue #485対応: デフォルトパラメータ管理の改善
    """
    # メタ学習器選択
    meta_learner_type: str = "xgboost"  # linear, ridge, lasso, elastic, rf, xgboost, mlp

    # 交差検証設定
    cv_method: str = "timeseries"  # kfold, timeseries
    cv_folds: int = 5

    # メタ学習器パラメータ
    meta_learner_params: Dict[str, Any] = None

    # 特徴量エンジニアリング
    include_base_features: bool = False  # 元特徴量をメタ学習器に含める
    include_prediction_stats: bool = True  # 予測統計量を含める

    # パフォーマンス設定
    enable_hyperopt: bool = True
    normalize_meta_features: bool = True
    verbose: bool = True
    
    # Issue #692対応: ハイパーパラメータ最適化設定
    hyperopt_method: str = "optuna"  # "optuna", "grid_search"
    hyperopt_n_trials: int = 100  # Optuna試行回数
    hyperopt_timeout: Optional[int] = 300  # 最適化タイムアウト（秒）
    hyperopt_early_stopping: bool = True  # 早期停止
    
    # Issue #485対応: メタ学習器デフォルトパラメータの管理
    meta_learner_defaults: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Issue #485対応: デフォルトパラメータの初期化
        """
        if self.meta_learner_defaults is None:
            self.meta_learner_defaults = self._get_default_meta_learner_params()
        
        if self.meta_learner_params is None:
            self.meta_learner_params = {}
    
    def _get_default_meta_learner_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Issue #485対応: メタ学習器タイプ別デフォルトパラメータ定義
        
        Returns:
            各メタ学習器タイプのデフォルトパラメータ辞書
        """
        return {
            "linear": {
                # LinearRegressionにはハイパーパラメータなし
            },
            "ridge": {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            },
            "lasso": {
                'alpha': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            "elastic": {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'random_state': 42
            },
            "rf": {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'n_jobs': -1
            },
            "xgboost": {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'objective': 'reg:squarederror',
                'n_jobs': -1
            },
            "mlp": {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 500,
                'random_state': 42
            }
        }
    
    def get_meta_learner_params(self, learner_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Issue #485対応: 指定学習器の最終パラメータ取得
        
        Args:
            learner_type: メタ学習器タイプ（Noneの場合は設定値を使用）
            
        Returns:
            デフォルト + カスタムパラメータのマージ結果
        """
        target_type = learner_type or self.meta_learner_type
        
        if target_type not in self.meta_learner_defaults:
            logger.warning(f"未知のメタ学習器タイプ: {target_type}")
            return self.meta_learner_params.copy()
            
        # デフォルト + カスタムパラメータのマージ
        default_params = self.meta_learner_defaults[target_type].copy()
        default_params.update(self.meta_learner_params)
        
        return default_params
        
    def update_meta_learner_defaults(self, learner_type: str, params: Dict[str, Any]) -> None:
        """
        Issue #485対応: デフォルトパラメータの更新
        
        Args:
            learner_type: 更新対象のメタ学習器タイプ
            params: 更新パラメータ辞書
        """
        if learner_type not in self.meta_learner_defaults:
            logger.warning(f"新しいメタ学習器タイプ {learner_type} のデフォルトパラメータを追加")
            self.meta_learner_defaults[learner_type] = {}
            
        self.meta_learner_defaults[learner_type].update(params)
        logger.info(f"{learner_type} のデフォルトパラメータを更新: {params}")
        
    @classmethod 
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'StackingConfig':
        """
        Issue #485対応: 辞書からの設定読み込み
        
        Args:
            config_dict: 設定辞書
            
        Returns:
            StackingConfig インスタンス
        """
        return cls(**config_dict)
        
    @classmethod
    def load_from_file(cls, filepath: str) -> 'StackingConfig':
        """
        Issue #485対応: ファイルからの設定読み込み
        
        Args:
            filepath: 設定ファイルパス（JSON形式）
            
        Returns:
            StackingConfig インスタンス
        """
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー {filepath}: {e}")
            logger.info("デフォルト設定を使用します")
            return cls()


class StackingEnsemble:
    """
    スタッキングアンサンブル実装

    2段階学習:
    1. Base Learners: 複数の基底モデルで予測
    2. Meta Learner: Base Learnersの予測を統合
    """

    def __init__(self, base_models: Dict[str, BaseModelInterface],
                 config: Optional[StackingConfig] = None):
        """
        初期化

        Args:
            base_models: ベースモデル辞書
            config: スタッキング設定
        """
        self.base_models = base_models
        self.config = config or StackingConfig()

        # メタ学習器
        self.meta_learner = None
        self.meta_scaler = StandardScaler() if self.config.normalize_meta_features else None

        # 訓練データ保存（メタ学習用）
        self.meta_features_train = None
        self.meta_targets_train = None

        # 学習状態
        self.is_fitted = False
        self.cv_scores = {}
        self.meta_feature_names = []
        
        # Issue #486対応: 元特徴量名の管理
        self.original_feature_names = []  # 元データの特徴量名

        logger.info(f"スタッキングアンサンブル初期化: {len(base_models)}個のベースモデル")

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray,
            validation_data: Optional[Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]] = None) -> Dict[str, Any]:
        """
        スタッキングアンサンブル学習
        
        Issue #486対応: DataFrameの特徴量名を自動検出・保持

        Args:
            X: 訓練データの特徴量（numpy配列またはpandas DataFrame）
            y: 訓練データの目標変数
            validation_data: 検証データ（オプション）

        Returns:
            学習結果辞書
        """
        start_time = time.time()
        
        # Issue #486対応: 特徴量名の自動検出
        if isinstance(X, pd.DataFrame):
            self.original_feature_names = list(X.columns)
            X_array = X.values
            logger.info(f"DataFrame特徴量名を検出: {len(self.original_feature_names)}個")
        else:
            X_array = X
            # numpy配列の場合は汎用名を生成
            self.original_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            logger.debug(f"numpy配列のため汎用特徴量名を生成: {len(self.original_feature_names)}個")
        
        logger.info(f"スタッキング学習開始: データ形状 {X_array.shape}")

        try:
            # Step 1: ベースモデルの交差検証学習
            meta_features = self._generate_meta_features(X_array, y)

            # Step 2: メタ学習器の学習
            self._fit_meta_learner(meta_features, y, X_array if self.config.include_base_features else None)

            # Step 3: ベースモデルの最終学習（全データ）
            self._fit_base_models_final(X_array, y)

            # 学習結果
            training_results = {
                'training_time': time.time() - start_time,
                'meta_learner_type': self.config.meta_learner_type,
                'meta_features_shape': meta_features.shape,
                'cv_scores': self.cv_scores,
                'meta_feature_names': self.meta_feature_names
            }

            # 検証データでの評価
            if validation_data:
                X_val, y_val = validation_data
                # Issue #486対応: DataFrameかnumpy配列かを統一
                if isinstance(X_val, pd.DataFrame):
                    X_val_array = X_val.values
                else:
                    X_val_array = X_val
                    
                val_pred = self.predict(X_val_array)
                val_mse = np.mean((y_val - val_pred.predictions) ** 2)
                val_rmse = np.sqrt(val_mse)

                training_results['validation_rmse'] = val_rmse
                logger.info(f"スタッキング検証RMSE: {val_rmse:.4f}")

            self.is_fitted = True
            logger.info(f"スタッキング学習完了: {time.time() - start_time:.2f}秒")

            return training_results

        except Exception as e:
            logger.error(f"スタッキング学習エラー: {e}")
            raise

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        スタッキング予測

        Args:
            X: 予測データの特徴量

        Returns:
            ModelPrediction: 予測結果
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません")

        start_time = time.time()

        try:
            # Step 1: ベースモデルからの予測
            # Issue #488対応: 学習状態の一貫性検証
            base_predictions = {}
            for name, model in self.base_models.items():
                if model.is_trained:
                    # 学習状態の厳密な検証
                    if not model.validate_training_state(require_trained=True):
                        logger.error(f"{name}: is_trainedがTrueですが学習状態が不整合です")
                        status = model.get_training_status()
                        logger.error(f"{name}学習状態詳細: {status}")
                        raise ValueError(f"ベースモデル {name} の学習状態が不整合です")
                    
                    pred_result = model.predict(X)
                    base_predictions[name] = pred_result.predictions
                else:
                    logger.warning(f"{name}: 未学習のため予測をスキップします")

            # Step 2: メタ特徴量生成
            meta_features = self._create_meta_features_from_predictions(
                base_predictions, X if self.config.include_base_features else None
            )

            # Step 3: メタ学習器で最終予測
            if self.meta_scaler:
                meta_features_scaled = self.meta_scaler.transform(meta_features)
            else:
                meta_features_scaled = meta_features

            final_predictions = self.meta_learner.predict(meta_features_scaled)

            # 信頼度計算（予測分散ベース）
            confidence = self._calculate_stacking_confidence(base_predictions)

            processing_time = time.time() - start_time

            return ModelPrediction(
                predictions=final_predictions,
                confidence=confidence,
                feature_importance=self._get_meta_feature_importance(),
                model_name="StackingEnsemble",
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"スタッキング予測エラー: {e}")
            raise

    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        交差検証によるメタ特徴量生成

        Args:
            X: 訓練データ特徴量
            y: 訓練データ目標変数

        Returns:
            メタ特徴量配列
        """
        logger.info("メタ特徴量生成開始")

        # 交差検証設定
        if self.config.cv_method == "timeseries":
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = KFold(n_splits=self.config.cv_folds, shuffle=False)

        # メタ特徴量初期化
        meta_predictions = {name: np.zeros(len(X)) for name in self.base_models.keys()}

        # 交差検証実行
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"CV Fold {fold + 1}/{self.config.cv_folds}")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # 各ベースモデルの学習・予測
            for name, model in self.base_models.items():
                try:
                    # モデルコピー作成（元モデルを変更しない）
                    model_copy = self._copy_model(model)

                    # Fold学習
                    model_copy.fit(X_fold_train, y_fold_train)

                    # Fold予測
                    pred_result = model_copy.predict(X_fold_val)
                    meta_predictions[name][val_idx] = pred_result.predictions

                    # CV スコア記録
                    fold_mse = np.mean((y_fold_val - pred_result.predictions) ** 2)
                    if name not in self.cv_scores:
                        self.cv_scores[name] = []
                    self.cv_scores[name].append(np.sqrt(fold_mse))

                except Exception as e:
                    logger.warning(f"Fold {fold} {name} エラー: {e}")
                    # エラー時はゼロ埋め
                    meta_predictions[name][val_idx] = 0.0

        # メタ特徴量構築
        meta_features = self._create_meta_features_from_predictions(
            meta_predictions, X if self.config.include_base_features else None
        )

        logger.info(f"メタ特徴量生成完了: {meta_features.shape}")
        return meta_features

    def _create_meta_features_from_predictions(self,
                                             predictions: Dict[str, np.ndarray],
                                             base_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        予測値からメタ特徴量を作成

        Args:
            predictions: モデル名と予測値の辞書
            base_features: 元特徴量（オプション）

        Returns:
            メタ特徴量配列
        """
        feature_list = []
        feature_names = []

        # 1. 基本予測値
        for name, pred in predictions.items():
            feature_list.append(pred.reshape(-1, 1))
            feature_names.append(f"pred_{name}")

        # 2. 予測統計量
        if self.config.include_prediction_stats and len(predictions) > 1:
            pred_array = np.array(list(predictions.values())).T  # (n_samples, n_models)

            # 統計量計算
            pred_mean = np.mean(pred_array, axis=1, keepdims=True)
            pred_std = np.std(pred_array, axis=1, keepdims=True)
            pred_min = np.min(pred_array, axis=1, keepdims=True)
            pred_max = np.max(pred_array, axis=1, keepdims=True)
            pred_median = np.median(pred_array, axis=1, keepdims=True)

            feature_list.extend([pred_mean, pred_std, pred_min, pred_max, pred_median])
            feature_names.extend(['pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_median'])

            # ペアワイズ差分
            model_names = list(predictions.keys())
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    diff = (pred_array[:, i] - pred_array[:, j]).reshape(-1, 1)
                    feature_list.append(diff)
                    feature_names.append(f"diff_{model_names[i]}_{model_names[j]}")

        # 3. 元特徴量（Issue #486対応: 実際の特徴量名を使用）
        if base_features is not None:
            feature_list.append(base_features)
            
            # 実際の特徴量名を使用（利用可能な場合）
            if len(self.original_feature_names) == base_features.shape[1]:
                # 元特徴量名に'base_'プレフィックスを付加
                base_names = [f"base_{name}" for name in self.original_feature_names]
                feature_names.extend(base_names)
                logger.debug(f"元特徴量名を使用: {len(base_names)}個の名前付き特徴量")
            else:
                # フォールバック: 汎用名を使用
                base_names = [f"base_feature_{i}" for i in range(base_features.shape[1])]
                feature_names.extend(base_names)
                logger.warning(
                    f"元特徴量名の数({len(self.original_feature_names)})と"
                    f"特徴量数({base_features.shape[1]})が不一致 - 汎用名を使用"
                )

        # 結合
        meta_features = np.concatenate(feature_list, axis=1)
        self.meta_feature_names = feature_names

        return meta_features

    def _fit_meta_learner(self, meta_features: np.ndarray, targets: np.ndarray,
                         base_features: Optional[np.ndarray] = None):
        """
        Issue #489対応: メタ学習器の学習（ハイパーパラメータ最適化機能付き）

        Args:
            meta_features: メタ特徴量
            targets: 目標変数
            base_features: 元特徴量（オプション）
        """
        logger.info(f"メタ学習器学習開始: {self.config.meta_learner_type}")

        # 前処理
        if self.meta_scaler:
            meta_features_scaled = self.meta_scaler.fit_transform(meta_features)
        else:
            meta_features_scaled = meta_features

        # Issue #489対応: ハイパーパラメータ最適化
        if self.config.enable_hyperopt:
            logger.info("メタ学習器ハイパーパラメータ最適化開始")
            self.meta_learner = self._optimize_meta_learner_hyperparams(
                meta_features_scaled, targets
            )
            logger.info("メタ学習器ハイパーパラメータ最適化完了")
        else:
            # デフォルトパラメータでメタ学習器初期化
            self.meta_learner = self._create_meta_learner()
            
            # 学習
            if hasattr(self.meta_learner, 'fit'):
                self.meta_learner.fit(meta_features_scaled, targets)
            else:
                raise ValueError(f"メタ学習器 {self.config.meta_learner_type} はfitメソッドを持ちません")

        # 学習データ保存
        self.meta_features_train = meta_features_scaled
        self.meta_targets_train = targets

        logger.info("メタ学習器学習完了")

    def _create_meta_learner(self):
        """
        Issue #485対応: 改良されたメタ学習器作成（設定管理強化）
        """
        # 設定から最終パラメータを取得（デフォルト + カスタムパラメータ）
        final_params = self.config.get_meta_learner_params()
        learner_type = self.config.meta_learner_type
        
        logger.debug(f"メタ学習器 {learner_type} を作成: {final_params}")

        if learner_type == "linear":
            return LinearRegression(**final_params)
        elif learner_type == "ridge":
            return Ridge(**final_params)
        elif learner_type == "lasso":
            return Lasso(**final_params)
        elif learner_type == "elastic":
            return ElasticNet(**final_params)
        elif learner_type == "rf":
            return RandomForestRegressor(**final_params)
        elif learner_type == "xgboost":
            return XGBRegressor(**final_params)
        elif learner_type == "mlp":
            return MLPRegressor(**final_params)
        else:
            raise ValueError(f"不明なメタ学習器タイプ: {learner_type}")

    def _optimize_meta_learner_hyperparams(self, meta_features: np.ndarray, 
                                         targets: np.ndarray):
        """
        Issue #692対応: 効率化されたメタ学習器ハイパーパラメータ最適化
        
        Args:
            meta_features: メタ特徴量
            targets: 目標変数
            
        Returns:
            最適化されたメタ学習器
            
        Note:
            - Optuna利用可能時はベイズ最適化で効率化
            - フォールバック時はGridSearchCVを使用
        """
        start_time = time.time()
        
        # Linear回帰はハイパーパラメータがないため、そのまま返す
        if self.config.meta_learner_type == "linear":
            learner = self._create_meta_learner()
            learner.fit(meta_features, targets)
            return learner
        
        # 最適化方法の選択
        if (self.config.hyperopt_method == "optuna" and OPTUNA_AVAILABLE and 
            self.config.meta_learner_type in ["ridge", "lasso", "elastic", "rf", "xgboost", "mlp"]):
            # Issue #692対応: Optunaベイズ最適化
            try:
                optimized_learner = self._optimize_with_optuna(meta_features, targets)
                optimization_method = "Optuna"
            except Exception as e:
                logger.warning(f"Optuna最適化エラー: {e} - GridSearchCVにフォールバック")
                optimized_learner = self._optimize_with_grid_search(meta_features, targets)
                optimization_method = "GridSearchCV"
        else:
            # GridSearchCVフォールバック
            if not OPTUNA_AVAILABLE and self.config.hyperopt_method == "optuna":
                logger.warning("Optuna未利用 - GridSearchCVにフォールバック")
            optimized_learner = self._optimize_with_grid_search(meta_features, targets)
            optimization_method = "GridSearchCV"
        
        optimization_time = time.time() - start_time
        logger.info(f"メタ学習器最適化完了 ({optimization_method}): {optimization_time:.3f}秒")
        
        return optimized_learner

    def _optimize_with_optuna(self, meta_features: np.ndarray, targets: np.ndarray):
        """
        Issue #692対応: Optunaによるベイズ最適化
        
        Args:
            meta_features: メタ特徴量
            targets: 目標変数
            
        Returns:
            最適化されたメタ学習器
        """
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.metrics import mean_squared_error
        
        # 時系列交差検証設定
        cv = TimeSeriesSplit(n_splits=min(5, len(targets) // 20))
        
        def objective(trial):
            """Optuna目的関数"""
            try:
                # メタ学習器タイプ別パラメータ定義
                if self.config.meta_learner_type == "ridge":
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 1000.0, log=True),
                        'random_state': 42
                    }
                    model = Ridge(**params)
                    
                elif self.config.meta_learner_type == "lasso":
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                        'max_iter': trial.suggest_int('max_iter', 500, 2000),
                        'random_state': 42
                    }
                    model = Lasso(**params)
                    
                elif self.config.meta_learner_type == "elastic":
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
                        'max_iter': trial.suggest_int('max_iter', 500, 2000),
                        'random_state': 42
                    }
                    model = ElasticNet(**params)
                    
                elif self.config.meta_learner_type == "rf":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'random_state': 42
                    }
                    model = RandomForestRegressor(**params)
                    
                elif self.config.meta_learner_type == "xgboost":
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
                        'max_depth': trial.suggest_int('max_depth', 2, 12),
                        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
                        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
                        'random_state': 42,
                        'objective': 'reg:squarederror'
                    }
                    model = XGBRegressor(**params)
                    
                elif self.config.meta_learner_type == "mlp":
                    # 隠れ層構造の動的定義
                    n_layers = trial.suggest_int('n_layers', 1, 3)
                    hidden_sizes = []
                    for i in range(n_layers):
                        size = trial.suggest_int(f'n_units_l{i}', 10, 200)
                        hidden_sizes.append(size)
                    
                    params = {
                        'hidden_layer_sizes': tuple(hidden_sizes),
                        'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
                        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                        'max_iter': 1000,
                        'random_state': 42
                    }
                    model = MLPRegressor(**params)
                    
                else:
                    raise ValueError(f"未対応のメタ学習器タイプ: {self.config.meta_learner_type}")
                
                # 交差検証による評価
                cv_scores = cross_val_score(
                    model, meta_features, targets, 
                    cv=cv, scoring='neg_mean_squared_error'
                )
                
                return -cv_scores.mean()  # 最小化するため負の値を返す
                
            except Exception as e:
                # エラー時は大きなペナルティを返す
                logger.debug(f"Optuna trial error: {e}")
                return float('inf')
        
        # Optuna Study作成
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler
        )
        
        # 最適化実行
        study.optimize(
            objective,
            n_trials=self.config.hyperopt_n_trials,
            timeout=self.config.hyperopt_timeout,
            show_progress_bar=False
        )
        
        # 最適パラメータでモデル作成
        best_params = study.best_params
        logger.info(f"Optuna最適パラメータ: {best_params}")
        logger.info(f"最適CVスコア: {study.best_value:.4f} ({study.n_trials}試行)")
        
        # 最適モデルの学習
        if self.config.meta_learner_type == "ridge":
            optimized_model = Ridge(**{k: v for k, v in best_params.items() if k != 'random_state'}, random_state=42)
        elif self.config.meta_learner_type == "lasso":
            optimized_model = Lasso(**{k: v for k, v in best_params.items() if k != 'random_state'}, random_state=42)
        elif self.config.meta_learner_type == "elastic":
            optimized_model = ElasticNet(**{k: v for k, v in best_params.items() if k != 'random_state'}, random_state=42)
        elif self.config.meta_learner_type == "rf":
            optimized_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != 'random_state'}, random_state=42)
        elif self.config.meta_learner_type == "xgboost":
            optimized_model = XGBRegressor(**{k: v for k, v in best_params.items() if k not in ['random_state', 'objective']}, 
                                         random_state=42, objective='reg:squarederror')
        elif self.config.meta_learner_type == "mlp":
            # MLPの隠れ層サイズ復元
            n_layers = best_params['n_layers']
            hidden_layer_sizes = tuple(best_params[f'n_units_l{i}'] for i in range(n_layers))
            mlp_params = {k: v for k, v in best_params.items() if not k.startswith(('n_layers', 'n_units_'))}
            mlp_params['hidden_layer_sizes'] = hidden_layer_sizes
            optimized_model = MLPRegressor(**{k: v for k, v in mlp_params.items() if k != 'random_state'}, 
                                         random_state=42, max_iter=1000)
        
        optimized_model.fit(meta_features, targets)
        return optimized_model

    def _optimize_with_grid_search(self, meta_features: np.ndarray, targets: np.ndarray):
        """
        Issue #692対応: GridSearchCVフォールバック最適化
        
        Args:
            meta_features: メタ特徴量
            targets: 目標変数
            
        Returns:
            最適化されたメタ学習器
        """
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        
        # ハイパーパラメータグリッド定義（従来版を簡略化）
        param_grids = {
            "ridge": {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            "lasso": {
                'alpha': [0.01, 0.1, 1.0, 10.0]
            },
            "elastic": {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9]  # 簡略化
            },
            "rf": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],  # None除去で安定化
                'min_samples_split': [2, 5, 10]
            },
            "xgboost": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]  # subsample除去で簡略化
            },
            "mlp": {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],  # 簡略化
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]  # 簡略化
            }
        }
        
        if self.config.meta_learner_type not in param_grids:
            logger.warning(f"メタ学習器タイプ {self.config.meta_learner_type} のハイパーパラメータ最適化未対応")
            learner = self._create_meta_learner()
            learner.fit(meta_features, targets)
            return learner
            
        # ベースメタ学習器作成
        base_learner = self._create_meta_learner_for_hyperopt()
        param_grid = param_grids[self.config.meta_learner_type]
        
        # 時系列交差検証
        cv = TimeSeriesSplit(n_splits=min(5, len(targets) // 20))
        
        # グリッドサーチ実行
        try:
            grid_search = GridSearchCV(
                base_learner,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(meta_features, targets)
            
            logger.info(f"GridSearch最適パラメータ: {grid_search.best_params_}")
            logger.info(f"最適CVスコア: {-grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.warning(f"GridSearchハイパーパラメータ最適化エラー: {e}")
            logger.info("デフォルトパラメータで学習を続行")
            learner = self._create_meta_learner()
            learner.fit(meta_features, targets)
            return learner
    
    def _create_meta_learner_for_hyperopt(self):
        """ハイパーパラメータ最適化用のベースメタ学習器作成（パラメータなし）"""
        if self.config.meta_learner_type == "ridge":
            return Ridge(random_state=42)
        elif self.config.meta_learner_type == "lasso":
            return Lasso(random_state=42)
        elif self.config.meta_learner_type == "elastic":
            return ElasticNet(random_state=42)
        elif self.config.meta_learner_type == "rf":
            return RandomForestRegressor(random_state=42)
        elif self.config.meta_learner_type == "xgboost":
            return XGBRegressor(random_state=42, objective='reg:squarederror')
        elif self.config.meta_learner_type == "mlp":
            return MLPRegressor(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"不明なメタ学習器タイプ: {self.config.meta_learner_type}")

    def _fit_base_models_final(self, X: np.ndarray, y: np.ndarray):
        """ベースモデルの最終学習（全データ）"""
        logger.info("ベースモデル最終学習開始")

        for name, model in self.base_models.items():
            try:
                # Issue #488対応: 学習状態の一貫性検証
                if not model.is_trained:
                    # 未学習状態の検証
                    if not model.validate_training_state(require_trained=False):
                        logger.warning(f"{name}: is_trainedがFalseですが学習状態が不整合です")
                        status = model.get_training_status()
                        logger.warning(f"{name}学習状態詳細: {status}")
                    
                    model.fit(X, y)
                    
                    # 学習後の状態検証
                    if not model.is_trained:
                        logger.error(f"{name}: fit()完了後もis_trainedがFalseです")
                        status = model.get_training_status()
                        logger.error(f"{name}学習後状態詳細: {status}")
                        raise ValueError(f"ベースモデル {name} の学習に失敗しました")
                        
                logger.info(f"{name} 最終学習完了")
            except Exception as e:
                logger.error(f"{name} 最終学習エラー: {e}")
                # Issue #488対応: 学習失敗時の状態確認
                status = model.get_training_status()
                logger.error(f"{name}エラー時状態詳細: {status}")

    def _copy_model(self, model: BaseModelInterface) -> BaseModelInterface:
        """
        Issue #691対応: ベースモデルコピー最適化
        
        sklearn.base.cloneを活用した効率的なモデルコピーを実現。
        複雑なモデルのインスタンス化オーバーヘッドを削減し、
        メモリ使用量とコピー時間を最適化する。
        
        Args:
            model: コピー対象のベースモデル
            
        Returns:
            コピーされたベースモデルインスタンス
            
        Raises:
            ValueError: コピーに失敗した場合
        """
        try:
            # 1. BaseModelInterfaceレベルでのコピー対応チェック
            if hasattr(model, 'copy') and callable(model.copy):
                logger.debug(f"{model.model_name}: 独自copyメソッドでコピー")
                return model.copy()
            
            # Issue #691対応: sklearn.base.cloneによる効率的コピー
            # 2. sklearn.base.cloneを最優先で使用（最も効率的）
            try:
                from sklearn.base import clone
                
                # BaseModelInterface全体をcloneできるかチェック
                if hasattr(model, 'get_params') and hasattr(model, 'set_params'):
                    try:
                        # BaseModelInterface自体がsklearn互換の場合
                        cloned_model = clone(model)
                        cloned_model.is_trained = False
                        cloned_model.training_metrics = {}
                        cloned_model.model = None  # 未学習状態で初期化
                        logger.debug(f"{model.model_name}: BaseModelInterface直接cloneでコピー完了")
                        return cloned_model
                    except Exception as e:
                        logger.debug(f"{model.model_name}: BaseModelInterface直接clone失敗: {e}, 内部モデルcloneを試行")
                
                # 内部skleanモデルをcloneして新しいBaseModelInterfaceに配置
                if hasattr(model, 'model') and model.model is not None:
                    if hasattr(model.model, 'get_params') and hasattr(model.model, 'set_params'):
                        # Issue #691対応: sklearn互換モデルの効率的clone
                        cloned_sklearn_model = clone(model.model)
                        
                        # 新しいBaseModelInterfaceインスタンス作成（最小限のオーバーヘッド）
                        new_model = self._create_minimal_model_copy(model, cloned_sklearn_model)
                        logger.debug(f"{model.model_name}: sklearn.base.clone最適化コピー完了")
                        return new_model
                    else:
                        logger.debug(f"{model.model_name}: 内部モデルsklearn非互換, 基本コピーにフォールバック")
                
            except ImportError:
                logger.warning(f"{model.model_name}: sklearn未使用環境, 基本コピーを実行")
            
            # 3. 基本コピー（フォールバック）- Issue #691対応でメモリ効率化
            return self._create_basic_model_copy(model)
            
        except Exception as e:
            error_msg = f"モデル {model.model_name} のコピーに失敗しました: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _create_minimal_model_copy(self, original_model: BaseModelInterface, 
                                   cloned_sklearn_model) -> BaseModelInterface:
        """
        Issue #691対応: 最小限のオーバーヘッドでモデルコピー作成
        
        Args:
            original_model: 元のBaseModelInterface
            cloned_sklearn_model: cloneされたsklearnモデル
            
        Returns:
            最適化されたモデルコピー
        """
        import copy
        
        # 最小限の初期化でBaseModelInterfaceインスタンス作成
        model_class = original_model.__class__
        
        # configの効率的コピー（必要最小限のみ）
        config_copy = copy.copy(original_model.config) if hasattr(original_model, 'config') else None
        
        # 新しいインスタンス作成
        new_model = model_class(config_copy)
        
        # Issue #691対応: 効率的属性コピー
        new_model.model = cloned_sklearn_model  # 既にcloneされたモデルを設定
        new_model.is_trained = False  # 未学習状態で初期化
        new_model.training_metrics = {}  # 空の辞書で初期化
        new_model.feature_names = original_model.feature_names.copy() if hasattr(original_model, 'feature_names') and original_model.feature_names else []
        
        # モデル名継承（重要な識別情報）
        if hasattr(original_model, 'model_name'):
            new_model.model_name = original_model.model_name
        
        return new_model
    
    def _create_basic_model_copy(self, original_model: BaseModelInterface) -> BaseModelInterface:
        """
        Issue #691対応: 基本的なモデルコピー（フォールバック用）
        
        Args:
            original_model: 元のBaseModelInterface
            
        Returns:
            基本コピー
        """
        import copy
        
        model_class = original_model.__class__
        
        # 効率的な設定コピー
        if hasattr(original_model, 'config') and original_model.config:
            config_copy = copy.copy(original_model.config)
        else:
            config_copy = None
        
        # 新しいインスタンス作成
        new_model = model_class(config_copy)
        
        # 基本属性のコピー（効率的）
        new_model.feature_names = original_model.feature_names.copy() if hasattr(original_model, 'feature_names') and original_model.feature_names else []
        new_model.is_trained = False
        new_model.training_metrics = {}
        new_model.model = None  # 未学習状態で初期化
        
        # モデル名継承
        if hasattr(original_model, 'model_name'):
            new_model.model_name = original_model.model_name
        
        logger.debug(f"{original_model.model_name}: 基本コピー完了")
        return new_model

    def _calculate_stacking_confidence(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """スタッキング信頼度計算"""
        if len(base_predictions) < 2:
            return np.ones(len(list(base_predictions.values())[0])) * 0.5

        # ベース予測の分散ベース信頼度
        pred_array = np.array(list(base_predictions.values())).T
        pred_variance = np.var(pred_array, axis=1)

        # 正規化（分散が小さいほど信頼度が高い）
        max_var = np.max(pred_variance) if np.max(pred_variance) > 0 else 1.0
        confidence = 1.0 - (pred_variance / max_var)

        return confidence

    def _get_meta_feature_importance(self) -> Dict[str, float]:
        """メタ特徴量重要度取得"""
        if not self.is_fitted or not hasattr(self.meta_learner, 'feature_importances_'):
            return {}

        importances = self.meta_learner.feature_importances_

        if len(self.meta_feature_names) == len(importances):
            return dict(zip(self.meta_feature_names, importances))
        else:
            return {}

    def get_cv_results(self) -> pd.DataFrame:
        """交差検証結果取得"""
        if not self.cv_scores:
            return pd.DataFrame()

        results = []
        for model_name, scores in self.cv_scores.items():
            results.append({
                'model': model_name,
                'mean_rmse': np.mean(scores),
                'std_rmse': np.std(scores),
                'min_rmse': np.min(scores),
                'max_rmse': np.max(scores)
            })

        return pd.DataFrame(results).sort_values('mean_rmse')

    def get_stacking_info(self) -> Dict[str, Any]:
        """スタッキング情報取得"""
        return {
            'is_fitted': self.is_fitted,
            'meta_learner_type': self.config.meta_learner_type,
            'n_base_models': len(self.base_models),
            'base_model_names': list(self.base_models.keys()),
            'meta_feature_count': len(self.meta_feature_names),
            'cv_method': self.config.cv_method,
            'cv_folds': self.config.cv_folds,
            # Issue #486対応: 特徴量名情報の追加
            'original_feature_count': len(self.original_feature_names),
            'has_meaningful_feature_names': len(self.original_feature_names) > 0 and 
                                          not all(name.startswith('feature_') for name in self.original_feature_names),
            'include_base_features': self.config.include_base_features
        }
    
    def get_meta_feature_info(self) -> Dict[str, Any]:
        """
        Issue #486対応: メタ特徴量の詳細情報取得
        
        Returns:
            メタ特徴量の構成と命名に関する情報
        """
        if not self.is_fitted:
            return {'error': 'アンサンブルが未学習です'}
            
        # 特徴量タイプ別の統計
        pred_features = [name for name in self.meta_feature_names if name.startswith('pred_')]
        stat_features = [name for name in self.meta_feature_names if name in ['pred_mean', 'pred_std', 'pred_min', 'pred_max', 'pred_median']]
        diff_features = [name for name in self.meta_feature_names if name.startswith('diff_')]
        base_features = [name for name in self.meta_feature_names if name.startswith('base_')]
        
        return {
            'total_meta_features': len(self.meta_feature_names),
            'feature_types': {
                'prediction_features': len(pred_features),
                'statistical_features': len(stat_features),
                'difference_features': len(diff_features),
                'base_features': len(base_features)
            },
            'original_feature_names': self.original_feature_names,
            'uses_meaningful_names': len(base_features) > 0 and any(
                not name.startswith('base_feature_') for name in base_features
            ),
            'sample_feature_names': {
                'prediction': pred_features[:3] if pred_features else [],
                'statistical': stat_features[:3] if stat_features else [],
                'difference': diff_features[:3] if diff_features else [],
                'base': base_features[:3] if base_features else []
            }
        }
        
    def validate_base_models_consistency(self) -> Dict[str, Any]:
        """
        Issue #488対応: 全ベースモデルの学習状態一貫性検証
        
        Returns:
            検証結果辞書
        """
        validation_results = {
            'consistent_models': [],
            'inconsistent_models': [],
            'total_models': len(self.base_models),
            'trained_count': 0,
            'untrained_count': 0,
            'detailed_status': {}
        }
        
        for name, model in self.base_models.items():
            status = model.get_training_status()
            validation_results['detailed_status'][name] = status
            
            if status['consistent_state']:
                validation_results['consistent_models'].append(name)
            else:
                validation_results['inconsistent_models'].append(name)
                logger.warning(f"{name}: 学習状態が不整合 - {status}")
                
            if model.is_trained:
                validation_results['trained_count'] += 1
            else:
                validation_results['untrained_count'] += 1
        
        # 全体サマリー        
        consistency_rate = len(validation_results['consistent_models']) / validation_results['total_models']
        validation_results['consistency_rate'] = consistency_rate
        validation_results['is_ensemble_ready'] = (
            validation_results['trained_count'] > 0 and 
            len(validation_results['inconsistent_models']) == 0
        )
        
        logger.info(f"ベースモデル一貫性検証完了: {consistency_rate:.2%} ({len(validation_results['consistent_models'])}/{validation_results['total_models']})")
        
        return validation_results


if __name__ == "__main__":
    # テスト実行（プレースホルダー）
    print("=== Stacking Ensemble テスト ===")
    print("実際のテストは統合テストで実行します")