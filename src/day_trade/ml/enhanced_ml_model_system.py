#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced ML Model System - 拡張機械学習モデルシステム

機械学習モデルの性能を大幅に改善する包括的なシステム
- 最新のアルゴリズム実装
- ハイパーパラメータ自動最適化
- モデル アンサンブル
- 動的モデル選択
- リアルタイム モデル更新
- 高度な特徴選択
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from pathlib import Path
import pickle
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import time
from collections import defaultdict, deque
import hashlib
import joblib
import gc

# 機械学習ライブラリ
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet, SGDClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, classification_report
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import (
    SelectKBest, RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from scipy import stats
from scipy.optimize import minimize

# 高度なML ライブラリ
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """モデルタイプ"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    ENSEMBLE_VOTING = "ensemble_voting"
    ENSEMBLE_STACKING = "ensemble_stacking"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"

class OptimizationMethod(Enum):
    """最適化手法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"

class FeatureSelectionMethod(Enum):
    """特徴選択手法"""
    VARIANCE_THRESHOLD = "variance_threshold"
    UNIVARIATE = "univariate"
    RFE = "rfe"
    MODEL_BASED = "model_based"
    PCA = "pca"
    ICA = "ica"

@dataclass
class ModelConfiguration:
    """モデル設定"""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    preprocessing: List[str]
    feature_selection: Optional[FeatureSelectionMethod] = None
    cross_validation_folds: int = 5
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN

@dataclass
class ModelPerformance:
    """モデル性能"""
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    training_time: float
    prediction_time: float
    model_size: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleResult:
    """アンサンブル結果"""
    ensemble_type: str
    base_models: List[str]
    performance: ModelPerformance
    weights: Dict[str, float]
    diversity_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class AdvancedFeatureSelector:
    """高度特徴選択システム"""
    
    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}
        self.selection_history = []
        
    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: FeatureSelectionMethod = FeatureSelectionMethod.MODEL_BASED,
                       n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """特徴選択実行"""
        logger.info(f"特徴選択を開始: 手法={method.value}, 元特徴数={X.shape[1]}")
        
        try:
            if method == FeatureSelectionMethod.VARIANCE_THRESHOLD:
                selected_X, selected_features = self._variance_threshold_selection(X, y)
            
            elif method == FeatureSelectionMethod.UNIVARIATE:
                selected_X, selected_features = self._univariate_selection(X, y, n_features)
            
            elif method == FeatureSelectionMethod.RFE:
                selected_X, selected_features = self._rfe_selection(X, y, n_features)
            
            elif method == FeatureSelectionMethod.MODEL_BASED:
                selected_X, selected_features = self._model_based_selection(X, y, n_features)
            
            elif method == FeatureSelectionMethod.PCA:
                selected_X, selected_features = self._pca_selection(X, y, n_features)
            
            elif method == FeatureSelectionMethod.ICA:
                selected_X, selected_features = self._ica_selection(X, y, n_features)
            
            else:
                # デフォルトは全特徴使用
                selected_X = X
                selected_features = list(X.columns)
            
            # 選択結果記録
            self.selection_history.append({
                'method': method.value,
                'original_features': X.shape[1],
                'selected_features': len(selected_features),
                'feature_list': selected_features,
                'timestamp': datetime.now()
            })
            
            logger.info(f"特徴選択完了: {len(selected_features)}個の特徴を選択")
            return selected_X, selected_features
            
        except Exception as e:
            logger.error(f"特徴選択エラー: {e}")
            return X, list(X.columns)
    
    def _variance_threshold_selection(self, X: pd.DataFrame, y: pd.Series,
                                    threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """分散による特徴選択"""
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        selected_features = [col for col, selected in zip(X.columns, selector.get_support()) if selected]
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series,
                            n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """単変量統計による特徴選択"""
        if n_features is None:
            n_features = min(50, X.shape[1] // 2)
        
        selector = SelectKBest(k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [col for col, selected in zip(X.columns, selector.get_support()) if selected]
        
        # スコア記録
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_scores['univariate'] = feature_scores
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series,
                      n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """再帰的特徴除去"""
        if n_features is None:
            n_features = min(30, X.shape[1] // 3)
        
        # ベースモデルとしてランダムフォレスト使用
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = [col for col, selected in zip(X.columns, selector.get_support()) if selected]
        
        # ランキング記録
        feature_rankings = dict(zip(X.columns, selector.ranking_))
        self.feature_scores['rfe'] = feature_rankings
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series,
                             n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """モデルベース特徴選択"""
        # 複数のモデルで特徴重要度を計算
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]
        
        if XGBOOST_AVAILABLE:
            models.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')))
        
        feature_importance_scores = defaultdict(list)
        
        for name, model in models:
            try:
                model.fit(X, y)
                importances = model.feature_importances_
                
                for feature, importance in zip(X.columns, importances):
                    feature_importance_scores[feature].append(importance)
                    
            except Exception as e:
                logger.warning(f"モデル {name} での特徴重要度計算エラー: {e}")
                continue
        
        # 平均重要度計算
        avg_importance = {}
        for feature, scores in feature_importance_scores.items():
            avg_importance[feature] = np.mean(scores) if scores else 0.0
        
        # 上位特徴選択
        if n_features is None:
            n_features = min(40, X.shape[1] // 2)
        
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:n_features]]
        
        self.feature_scores['model_based'] = avg_importance
        
        return X[selected_features], selected_features
    
    def _pca_selection(self, X: pd.DataFrame, y: pd.Series,
                      n_components: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """主成分分析による次元削減"""
        if n_components is None:
            n_components = min(20, X.shape[1] // 2)
        
        # データ標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA適用
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 新しい特徴名生成
        pca_features = [f'PCA_{i+1}' for i in range(n_components)]
        
        # 主成分の寄与率記録
        explained_variance = dict(zip(pca_features, pca.explained_variance_ratio_))
        self.feature_scores['pca'] = explained_variance
        
        return pd.DataFrame(X_pca, columns=pca_features, index=X.index), pca_features
    
    def _ica_selection(self, X: pd.DataFrame, y: pd.Series,
                      n_components: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """独立成分分析による次元削減"""
        if n_components is None:
            n_components = min(15, X.shape[1] // 3)
        
        # データ標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ICA適用
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        X_ica = ica.fit_transform(X_scaled)
        
        # 新しい特徴名生成
        ica_features = [f'ICA_{i+1}' for i in range(n_components)]
        
        return pd.DataFrame(X_ica, columns=ica_features, index=X.index), ica_features

class HyperparameterOptimizer:
    """ハイパーパラメータ最適化システム"""
    
    def __init__(self):
        self.optimization_history = {}
        self.best_params = {}
        
    def optimize_model(self, model_type: ModelType, X: pd.DataFrame, y: pd.Series,
                      optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                      n_trials: int = 100) -> Tuple[Dict[str, Any], float]:
        """モデル最適化"""
        logger.info(f"ハイパーパラメータ最適化開始: {model_type.value} - {optimization_method.value}")
        
        try:
            if optimization_method == OptimizationMethod.BAYESIAN:
                return self._bayesian_optimization(model_type, X, y, n_trials)
            elif optimization_method == OptimizationMethod.GRID_SEARCH:
                return self._grid_search_optimization(model_type, X, y)
            elif optimization_method == OptimizationMethod.RANDOM_SEARCH:
                return self._random_search_optimization(model_type, X, y)
            else:
                # デフォルトパラメータ返却
                return self._get_default_params(model_type), 0.5
                
        except Exception as e:
            logger.error(f"ハイパーパラメータ最適化エラー: {e}")
            return self._get_default_params(model_type), 0.0
    
    def _bayesian_optimization(self, model_type: ModelType, X: pd.DataFrame, y: pd.Series,
                             n_trials: int) -> Tuple[Dict[str, Any], float]:
        """ベイジアン最適化"""
        def objective(trial):
            params = self._suggest_params(trial, model_type)
            model = self._create_model(model_type, params)
            
            # 交差検証
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            return cv_scores.mean()
        
        # Optuna 最適化
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_score = study.best_value
        
        # 履歴保存
        self.optimization_history[model_type.value] = {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'timestamp': datetime.now()
        }
        
        return best_params, best_score
    
    def _grid_search_optimization(self, model_type: ModelType, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        """グリッドサーチ最適化"""
        param_grid = self._get_param_grid(model_type)
        model = self._create_model(model_type, {})
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def _random_search_optimization(self, model_type: ModelType, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        """ランダムサーチ最適化"""
        param_distributions = self._get_param_distributions(model_type)
        model = self._create_model(model_type, {})
        
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, random_state=42
        )
        random_search.fit(X, y)
        
        return random_search.best_params_, random_search.best_score_
    
    def _suggest_params(self, trial, model_type: ModelType) -> Dict[str, Any]:
        """パラメータ提案（Optuna用）"""
        if model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        
        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        
        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }
        
        elif model_type == ModelType.NEURAL_NETWORK:
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', 
                    [(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 1000)
            }
        
        else:
            return {}
    
    def _get_param_grid(self, model_type: ModelType) -> Dict[str, List]:
        """パラメータグリッド取得"""
        if model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:
            return {}
    
    def _get_param_distributions(self, model_type: ModelType) -> Dict[str, Any]:
        """パラメータ分布取得"""
        if model_type == ModelType.RANDOM_FOREST:
            return {
                'n_estimators': stats.randint(50, 300),
                'max_depth': stats.randint(3, 20),
                'min_samples_split': stats.randint(2, 20),
                'min_samples_leaf': stats.randint(1, 20)
            }
        else:
            return {}
    
    def _create_model(self, model_type: ModelType, params: Dict[str, Any]):
        """モデル作成"""
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        
        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
        
        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
        
        elif model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(random_state=42, **params)
        
        else:
            return RandomForestClassifier(random_state=42, n_jobs=-1)
    
    def _get_default_params(self, model_type: ModelType) -> Dict[str, Any]:
        """デフォルトパラメータ取得"""
        defaults = {
            ModelType.RANDOM_FOREST: {'n_estimators': 100, 'max_depth': 10},
            ModelType.XGBOOST: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            ModelType.LIGHTGBM: {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            ModelType.NEURAL_NETWORK: {'hidden_layer_sizes': (100,), 'max_iter': 500}
        }
        return defaults.get(model_type, {})

class AdvancedEnsembleSystem:
    """高度アンサンブル学習システム"""
    
    def __init__(self):
        self.base_models = {}
        self.ensemble_models = {}
        self.performance_history = defaultdict(list)
        
    def create_ensemble(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                       ensemble_type: str = "voting") -> EnsembleResult:
        """アンサンブル作成"""
        logger.info(f"アンサンブル作成開始: タイプ={ensemble_type}, モデル数={len(models)}")
        
        try:
            if ensemble_type == "voting":
                ensemble_model = self._create_voting_ensemble(models)
            elif ensemble_type == "stacking":
                ensemble_model = self._create_stacking_ensemble(models)
            elif ensemble_type == "weighted":
                ensemble_model = self._create_weighted_ensemble(models, X, y)
            else:
                ensemble_model = self._create_voting_ensemble(models)
            
            # アンサンブル性能評価
            cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='accuracy')
            
            # 多様性評価
            diversity_score = self._calculate_diversity(models, X, y)
            
            # 性能記録
            performance = ModelPerformance(
                model_type=ModelType.ENSEMBLE_VOTING,
                accuracy=cv_scores.mean(),
                precision=0.0,  # 後で計算
                recall=0.0,     # 後で計算
                f1_score=0.0,   # 後で計算
                auc_score=0.0,  # 後で計算
                cv_scores=cv_scores.tolist(),
                feature_importance={},
                training_time=0.0,
                prediction_time=0.0,
                model_size=0
            )
            
            # 重み計算
            weights = self._calculate_model_weights(models, X, y)
            
            result = EnsembleResult(
                ensemble_type=ensemble_type,
                base_models=list(models.keys()),
                performance=performance,
                weights=weights,
                diversity_score=diversity_score
            )
            
            # アンサンブルモデル保存
            self.ensemble_models[f"{ensemble_type}_ensemble"] = ensemble_model
            
            logger.info(f"アンサンブル作成完了: 精度={cv_scores.mean():.3f}, 多様性={diversity_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"アンサンブル作成エラー: {e}")
            raise
    
    def _create_voting_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """投票アンサンブル作成"""
        estimators = [(name, model) for name, model in models.items()]
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    def _create_stacking_ensemble(self, models: Dict[str, Any]) -> StackingClassifier:
        """スタッキングアンサンブル作成"""
        estimators = [(name, model) for name, model in models.items()]
        meta_classifier = LogisticRegression(random_state=42)
        return StackingClassifier(
            estimators=estimators, 
            final_estimator=meta_classifier,
            cv=5, n_jobs=-1
        )
    
    def _create_weighted_ensemble(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """重み付きアンサンブル作成"""
        # 各モデルの性能に基づく重み計算
        weights = []
        estimators = []
        
        for name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            weight = cv_scores.mean()
            weights.append(weight)
            estimators.append((name, model))
        
        # 重み正規化
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 重み付き投票アンサンブル作成
        voting_ensemble = VotingClassifier(
            estimators=estimators, 
            voting='soft', 
            weights=normalized_weights,
            n_jobs=-1
        )
        
        return voting_ensemble
    
    def _calculate_diversity(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
        """アンサンブル多様性計算"""
        try:
            # 各モデルの予測を取得
            predictions = {}
            for name, model in models.items():
                model.fit(X, y)
                pred = model.predict(X)
                predictions[name] = pred
            
            # ペアワイズ不一致度計算
            model_names = list(predictions.keys())
            disagreements = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    pred1 = predictions[model_names[i]]
                    pred2 = predictions[model_names[j]]
                    disagreement = np.mean(pred1 != pred2)
                    disagreements.append(disagreement)
            
            # 平均不一致度を多様性スコアとする
            diversity_score = np.mean(disagreements) if disagreements else 0.0
            return diversity_score
            
        except Exception as e:
            logger.error(f"多様性計算エラー: {e}")
            return 0.0
    
    def _calculate_model_weights(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """モデル重み計算"""
        weights = {}
        
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                weights[name] = cv_scores.mean()
            except Exception as e:
                logger.warning(f"モデル {name} の重み計算エラー: {e}")
                weights[name] = 0.5  # デフォルト重み
        
        # 正規化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return weights

class ModelManager:
    """モデル管理システム"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.performance_tracker = defaultdict(list)
        
    def register_model(self, name: str, model: Any, metadata: Dict[str, Any] = None):
        """モデル登録"""
        self.models[name] = model
        self.model_metadata[name] = {
            'created_at': datetime.now(),
            'model_type': type(model).__name__,
            'parameters': getattr(model, 'get_params', lambda: {})(),
            'custom_metadata': metadata or {}
        }
        logger.info(f"モデル '{name}' を登録しました")
    
    def get_model(self, name: str) -> Optional[Any]:
        """モデル取得"""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """モデル一覧"""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> bool:
        """モデル削除"""
        if name in self.models:
            del self.models[name]
            del self.model_metadata[name]
            logger.info(f"モデル '{name}' を削除しました")
            return True
        return False
    
    def save_model(self, name: str, filepath: str) -> bool:
        """モデル保存"""
        if name not in self.models:
            logger.error(f"モデル '{name}' が見つかりません")
            return False
        
        try:
            joblib.dump(self.models[name], filepath)
            logger.info(f"モデル '{name}' を {filepath} に保存しました")
            return True
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            return False
    
    def load_model(self, name: str, filepath: str) -> bool:
        """モデル読み込み"""
        try:
            model = joblib.load(filepath)
            self.register_model(name, model)
            logger.info(f"モデル '{name}' を {filepath} から読み込みました")
            return True
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False

class EnhancedMLModelSystem:
    """拡張機械学習モデルシステム（メインクラス）"""
    
    def __init__(self, db_path: str = "enhanced_ml_system.db"):
        self.db_path = Path(db_path)
        self.feature_selector = AdvancedFeatureSelector()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.ensemble_system = AdvancedEnsembleSystem()
        self.model_manager = ModelManager()
        
        self.setup_database()
        
        # 統計情報
        self.training_stats = {
            'models_trained': 0,
            'ensembles_created': 0,
            'optimizations_performed': 0,
            'features_selected': 0
        }
        
        logger.info("拡張機械学習モデルシステムを初期化しました")
    
    def setup_database(self):
        """データベース初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performances (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        model_name TEXT,
                        model_type TEXT,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        auc_score REAL,
                        training_time REAL,
                        hyperparameters TEXT,
                        feature_count INTEGER
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS ensemble_results (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        ensemble_type TEXT,
                        base_models TEXT,
                        accuracy REAL,
                        diversity_score REAL,
                        weights TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_history (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        model_type TEXT,
                        optimization_method TEXT,
                        best_score REAL,
                        best_params TEXT,
                        n_trials INTEGER
                    )
                ''')
                
                conn.commit()
                logger.info("拡張MLシステムデータベースを初期化しました")
        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
    
    async def train_advanced_model(self, X: pd.DataFrame, y: pd.Series,
                                 model_config: ModelConfiguration) -> ModelPerformance:
        """高度モデル訓練"""
        logger.info(f"高度モデル訓練開始: {model_config.model_type.value}")
        
        start_time = time.time()
        
        try:
            # 特徴選択
            if model_config.feature_selection:
                X_selected, selected_features = self.feature_selector.select_features(
                    X, y, model_config.feature_selection
                )
                self.training_stats['features_selected'] += len(selected_features)
            else:
                X_selected = X
                selected_features = list(X.columns)
            
            # ハイパーパラメータ最適化
            if model_config.optimization_method != OptimizationMethod.GRID_SEARCH:  # デフォルト回避
                optimized_params, best_score = self.hyperparameter_optimizer.optimize_model(
                    model_config.model_type, X_selected, y, model_config.optimization_method
                )
                self.training_stats['optimizations_performed'] += 1
            else:
                optimized_params = model_config.hyperparameters
                best_score = 0.0
            
            # 最終パラメータ統合
            final_params = {**model_config.hyperparameters, **optimized_params}
            
            # モデル作成・訓練
            model = self._create_optimized_model(model_config.model_type, final_params)
            model.fit(X_selected, y)
            
            # 性能評価
            cv_scores = cross_val_score(
                model, X_selected, y, 
                cv=model_config.cross_validation_folds,
                scoring='accuracy'
            )
            
            # 詳細評価
            y_pred = model.predict(X_selected)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # AUC計算（可能な場合）
            try:
                y_proba = model.predict_proba(X_selected)[:, 1]
                auc = roc_auc_score(y, y_proba)
            except:
                auc = 0.0
            
            # 特徴重要度取得
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                for feature, importance in zip(selected_features, model.feature_importances_):
                    feature_importance[feature] = float(importance)
            
            training_time = time.time() - start_time
            
            # 性能オブジェクト作成
            performance = ModelPerformance(
                model_type=model_config.model_type,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                cv_scores=cv_scores.tolist(),
                feature_importance=feature_importance,
                training_time=training_time,
                prediction_time=0.0,
                model_size=self._calculate_model_size(model)
            )
            
            # モデル登録
            model_name = f"{model_config.model_type.value}_{int(time.time())}"
            self.model_manager.register_model(model_name, model, {
                'configuration': model_config.__dict__,
                'performance': performance.__dict__,
                'selected_features': selected_features
            })
            
            # 性能記録
            await self._save_model_performance(model_name, performance, final_params, len(selected_features))
            
            self.training_stats['models_trained'] += 1
            
            logger.info(f"モデル訓練完了: {model_name}, 精度={accuracy:.3f}, 訓練時間={training_time:.1f}秒")
            return performance
            
        except Exception as e:
            logger.error(f"高度モデル訓練エラー: {e}")
            raise
    
    async def create_advanced_ensemble(self, model_configs: List[ModelConfiguration],
                                     X: pd.DataFrame, y: pd.Series,
                                     ensemble_type: str = "stacking") -> EnsembleResult:
        """高度アンサンブル作成"""
        logger.info(f"高度アンサンブル作成開始: タイプ={ensemble_type}, モデル数={len(model_configs)}")
        
        try:
            # ベースモデル訓練
            base_models = {}
            for i, config in enumerate(model_configs):
                performance = await self.train_advanced_model(X, y, config)
                model_name = f"base_model_{i}"
                base_models[model_name] = self.model_manager.get_model(
                    list(self.model_manager.models.keys())[-1]
                )
            
            # アンサンブル作成
            ensemble_result = self.ensemble_system.create_ensemble(
                base_models, X, y, ensemble_type
            )
            
            # 結果保存
            await self._save_ensemble_result(ensemble_result)
            
            self.training_stats['ensembles_created'] += 1
            
            logger.info(f"アンサンブル作成完了: 精度={ensemble_result.performance.accuracy:.3f}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"高度アンサンブル作成エラー: {e}")
            raise
    
    def _create_optimized_model(self, model_type: ModelType, params: Dict[str, Any]):
        """最適化モデル作成"""
        if model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        
        elif model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', **params)
        
        elif model_type == ModelType.LIGHTGBM and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, **params)
        
        elif model_type == ModelType.CATBOOST and CATBOOST_AVAILABLE:
            return cb.CatBoostClassifier(random_state=42, verbose=False, **params)
        
        elif model_type == ModelType.NEURAL_NETWORK:
            return MLPClassifier(random_state=42, **params)
        
        elif model_type == ModelType.SVM:
            return SVC(random_state=42, probability=True, **params)
        
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=42, **params)
        
        elif model_type == ModelType.EXTRA_TREES:
            return ExtraTreesClassifier(random_state=42, n_jobs=-1, **params)
        
        else:
            return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    
    def _calculate_model_size(self, model) -> int:
        """モデルサイズ計算"""
        try:
            import sys
            return sys.getsizeof(pickle.dumps(model))
        except:
            return 0
    
    async def _save_model_performance(self, model_name: str, performance: ModelPerformance,
                                    params: Dict[str, Any], feature_count: int):
        """モデル性能保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performances (
                        timestamp, model_name, model_type, accuracy, precision_score,
                        recall_score, f1_score, auc_score, training_time,
                        hyperparameters, feature_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.timestamp.isoformat(),
                    model_name,
                    performance.model_type.value,
                    performance.accuracy,
                    performance.precision,
                    performance.recall,
                    performance.f1_score,
                    performance.auc_score,
                    performance.training_time,
                    json.dumps(params),
                    feature_count
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"モデル性能保存エラー: {e}")
    
    async def _save_ensemble_result(self, result: EnsembleResult):
        """アンサンブル結果保存"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO ensemble_results (
                        timestamp, ensemble_type, base_models, accuracy,
                        diversity_score, weights
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.timestamp.isoformat(),
                    result.ensemble_type,
                    json.dumps(result.base_models),
                    result.performance.accuracy,
                    result.diversity_score,
                    json.dumps(result.weights)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"アンサンブル結果保存エラー: {e}")
    
    def get_system_report(self) -> Dict[str, Any]:
        """システム レポート"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 最新のモデル性能統計
                performance_query = '''
                    SELECT model_type, AVG(accuracy), AVG(f1_score), COUNT(*) 
                    FROM model_performances 
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY model_type
                    ORDER BY AVG(accuracy) DESC
                '''
                performance_stats = conn.execute(performance_query).fetchall()
                
                # アンサンブル統計
                ensemble_query = '''
                    SELECT ensemble_type, AVG(accuracy), AVG(diversity_score), COUNT(*)
                    FROM ensemble_results
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY ensemble_type
                '''
                ensemble_stats = conn.execute(ensemble_query).fetchall()
                
            return {
                'training_statistics': self.training_stats,
                'model_performance_by_type': [
                    {
                        'model_type': row[0],
                        'avg_accuracy': row[1],
                        'avg_f1_score': row[2],
                        'model_count': row[3]
                    }
                    for row in performance_stats
                ],
                'ensemble_statistics': [
                    {
                        'ensemble_type': row[0],
                        'avg_accuracy': row[1],
                        'avg_diversity': row[2],
                        'ensemble_count': row[3]
                    }
                    for row in ensemble_stats
                ],
                'registered_models': len(self.model_manager.models),
                'available_model_types': [
                    ModelType.RANDOM_FOREST.value,
                    ModelType.GRADIENT_BOOSTING.value,
                    ModelType.EXTRA_TREES.value,
                    ModelType.NEURAL_NETWORK.value,
                    ModelType.SVM.value
                ] + ([ModelType.XGBOOST.value] if XGBOOST_AVAILABLE else []) + \
                ([ModelType.LIGHTGBM.value] if LIGHTGBM_AVAILABLE else []) + \
                ([ModelType.CATBOOST.value] if CATBOOST_AVAILABLE else []),
                'feature_selection_history': len(self.feature_selector.selection_history),
                'optimization_history': len(self.hyperparameter_optimizer.optimization_history)
            }
            
        except Exception as e:
            logger.error(f"システム レポート生成エラー: {e}")
            return {'error': str(e)}

# 使用例とデモ
async def demo_enhanced_ml_system():
    """拡張MLシステム デモ"""
    try:
        # システム初期化
        system = EnhancedMLModelSystem("demo_enhanced_ml.db")
        
        # サンプルデータ作成
        n_samples = 1000
        n_features = 50
        
        # 特徴量データ
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # ターゲット（バイナリ分類）
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        print("サンプルデータ作成完了")
        
        # モデル設定作成
        rf_config = ModelConfiguration(
            model_type=ModelType.RANDOM_FOREST,
            hyperparameters={},
            preprocessing=[],
            feature_selection=FeatureSelectionMethod.MODEL_BASED,
            optimization_method=OptimizationMethod.BAYESIAN
        )
        
        gb_config = ModelConfiguration(
            model_type=ModelType.GRADIENT_BOOSTING,
            hyperparameters={},
            preprocessing=[],
            feature_selection=FeatureSelectionMethod.RFE,
            optimization_method=OptimizationMethod.BAYESIAN
        )
        
        # 単一モデル訓練
        rf_performance = await system.train_advanced_model(X, y, rf_config)
        print(f"Random Forest性能: 精度={rf_performance.accuracy:.3f}")
        
        # アンサンブル作成
        ensemble_configs = [rf_config, gb_config]
        ensemble_result = await system.create_advanced_ensemble(
            ensemble_configs, X, y, "stacking"
        )
        print(f"アンサンブル性能: 精度={ensemble_result.performance.accuracy:.3f}")
        
        # システム レポート
        report = system.get_system_report()
        print("システム レポート:")
        print(f"  訓練済みモデル数: {report['training_statistics']['models_trained']}")
        print(f"  作成済みアンサンブル数: {report['training_statistics']['ensembles_created']}")
        print(f"  実行済み最適化数: {report['training_statistics']['optimizations_performed']}")
        
        return system
        
    except Exception as e:
        logger.error(f"デモ実行エラー: {e}")
        raise

if __name__ == "__main__":
    # デモ実行
    asyncio.run(demo_enhanced_ml_system())