#!/usr/bin/env python3
"""
Support Vector Regression Model for Ensemble Learning

Issue #462: SVRベースモデルの実装
非線形パターンと高次元データに強いサポートベクター回帰
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

from .base_model_interface import BaseModelInterface, ModelPrediction
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SVRModel(BaseModelInterface):
    """
    Support Vector Regression モデル
    
    特徴:
    - 非線形パターン学習
    - 高次元データ対応
    - ロバストな予測
    - カーネル関数による柔軟性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: モデル設定辞書
        """
        default_config = {
            # SVR設定
            'kernel': 'rbf',  # rbf, linear, poly, sigmoid
            'C': 1.0,         # 正則化パラメータ
            'epsilon': 0.1,   # イプシロン管回帰
            'gamma': 'scale', # カーネル係数
            'degree': 3,      # 多項式カーネル次数
            'coef0': 0.0,     # 独立項
            'shrinking': True,
            'cache_size': 200,  # MB
            'max_iter': -1,
            
            # 学習設定
            'enable_hyperopt': True,
            'cv_folds': 5,
            'scaler_type': 'standard',  # standard, robust
            
            # パフォーマンス設定
            'verbose': False,
            'tol': 1e-3,
        }
        
        # 設定をマージ
        final_config = {**default_config, **(config or {})}
        super().__init__("SVR", final_config)
        
        # スケーラー選択
        if self.config['scaler_type'] == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
        self.best_params = {}
        self.pipeline = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        SVRモデル学習
        
        Args:
            X: 訓練データの特徴量 (n_samples, n_features)
            y: 訓練データの目標変数 (n_samples,)
            validation_data: 検証データ (X_val, y_val)
            
        Returns:
            学習結果辞書
        """
        start_time = time.time()
        logger.info(f"SVR学習開始: データ形状 {X.shape}")
        
        try:
            # パイプライン作成（前処理 + SVR）
            if self.config['enable_hyperopt']:
                logger.info("ハイパーパラメータ最適化開始")
                self.pipeline = self._hyperparameter_optimization(X, y)
                self.model = self.pipeline.named_steps['svr']
            else:
                # デフォルトパラメータで学習
                self.pipeline = Pipeline([
                    ('scaler', self.scaler),
                    ('svr', SVR(**self._get_svr_params()))
                ])
                self.pipeline.fit(X, y)
                self.model = self.pipeline.named_steps['svr']
            
            # 学習結果の記録
            training_results = {
                'training_time': time.time() - start_time,
                'n_support_vectors': self.model.n_support_,
                'support_vector_ratio': len(self.model.support_) / len(X),
                'dual_coef_norm': np.linalg.norm(self.model.dual_coef_) if hasattr(self.model, 'dual_coef_') else None
            }
            
            # 検証データでの評価
            if validation_data is not None:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                training_results['validation_metrics'] = val_metrics
                logger.info(f"検証RMSE: {val_metrics.rmse:.4f}, Hit Rate: {val_metrics.hit_rate:.3f}")
            
            # 学習メトリクス保存
            self.training_metrics = training_results
            self.is_trained = True
            
            logger.info(f"SVR学習完了: {time.time() - start_time:.2f}秒")
            logger.info(f"サポートベクター数: {self.model.n_support_} ({training_results['support_vector_ratio']:.1%})")
            
            return training_results
            
        except Exception as e:
            logger.error(f"SVR学習エラー: {e}")
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
            # パイプラインを使用して予測（前処理込み）
            predictions = self.pipeline.predict(X)
            
            # 決定関数による信頼度推定
            confidence = self._calculate_prediction_confidence(X)
            
            # 特徴量重要度（SVRでは利用不可）
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
            logger.error(f"SVR予測エラー: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度取得
        
        SVRでは特徴量重要度が直接取得できないため、
        線形カーネルの場合のみ重みを返す
        
        Returns:
            特徴量名と重要度のマッピング（線形カーネル時のみ）
        """
        if not self.is_trained:
            return {}
        
        try:
            # 線形カーネルの場合のみ重みを取得可能
            if self.config['kernel'] == 'linear' and hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])  # 重みの絶対値
                
                if len(self.feature_names) == len(importances):
                    return dict(zip(self.feature_names, importances))
                else:
                    return {f"feature_{i}": imp for i, imp in enumerate(importances)}
            else:
                logger.debug("非線形カーネルのため特徴量重要度は利用できません")
                return {}
                
        except Exception as e:
            logger.warning(f"特徴量重要度取得エラー: {e}")
            return {}
    
    def _hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> Pipeline:
        """
        ハイパーパラメータ最適化
        
        Args:
            X: 学習データ特徴量
            y: 学習データ目標変数
            
        Returns:
            最適化されたPipeline
        """
        # 時系列交差検証
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        # パイプライン構築
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('svr', SVR())
        ])
        
        # 探索パラメータ空間
        if self.config['kernel'] == 'rbf':
            param_grid = {
                'svr__C': [0.1, 1.0, 10.0, 100.0],
                'svr__epsilon': [0.01, 0.1, 0.2, 0.5],
                'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            }
        elif self.config['kernel'] == 'linear':
            param_grid = {
                'svr__C': [0.1, 1.0, 10.0, 100.0],
                'svr__epsilon': [0.01, 0.1, 0.2, 0.5]
            }
        elif self.config['kernel'] == 'poly':
            param_grid = {
                'svr__C': [0.1, 1.0, 10.0],
                'svr__epsilon': [0.01, 0.1, 0.2],
                'svr__degree': [2, 3, 4],
                'svr__gamma': ['scale', 'auto', 0.01, 0.1]
            }
        else:  # sigmoid
            param_grid = {
                'svr__C': [0.1, 1.0, 10.0],
                'svr__epsilon': [0.01, 0.1, 0.2],
                'svr__gamma': ['scale', 'auto', 0.01, 0.1],
                'svr__coef0': [0.0, 0.1, 1.0]
            }
        
        # 基本パラメータ設定
        for param_set in param_grid.values() if isinstance(param_grid, dict) else [param_grid]:
            if isinstance(param_set, dict):
                for key in param_set.keys():
                    if key not in ['svr__C', 'svr__epsilon', 'svr__gamma', 'svr__degree', 'svr__coef0']:
                        param_grid[f'svr__{key.replace("svr__", "")}'] = getattr(self.config, key.replace("svr__", ""), None)
        
        # その他のSVRパラメータ設定
        base_params = {
            'svr__kernel': self.config['kernel'],
            'svr__shrinking': self.config['shrinking'],
            'svr__cache_size': self.config['cache_size'],
            'svr__max_iter': self.config['max_iter'],
            'svr__tol': self.config['tol'],
            'svr__verbose': self.config['verbose']
        }
        
        pipeline.set_params(**base_params)
        
        # Grid Search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=1,  # SVRは既に最適化されているため
            verbose=1 if self.config['verbose'] else 0
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        logger.info(f"最適パラメータ: {self.best_params}")
        logger.info(f"最適CV Score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _get_svr_params(self) -> Dict[str, Any]:
        """SVRパラメータ取得"""
        return {
            'kernel': self.config['kernel'],
            'C': self.config['C'],
            'epsilon': self.config['epsilon'],
            'gamma': self.config['gamma'],
            'degree': self.config['degree'],
            'coef0': self.config['coef0'],
            'shrinking': self.config['shrinking'],
            'cache_size': self.config['cache_size'],
            'max_iter': self.config['max_iter'],
            'tol': self.config['tol'],
            'verbose': self.config['verbose']
        }
    
    def _calculate_prediction_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        予測信頼度計算
        
        SVRでは決定関数の値を使用して信頼度を推定
        
        Args:
            X: 入力特徴量
            
        Returns:
            各サンプルの予測信頼度
        """
        try:
            # 前処理済みデータを取得
            X_scaled = self.pipeline.named_steps['scaler'].transform(X)
            
            # 決定関数の値を信頼度として使用
            if hasattr(self.model, 'decision_function'):
                decision_scores = np.abs(self.model.decision_function(X_scaled))
                # 正規化（0-1範囲）
                if decision_scores.max() > decision_scores.min():
                    confidence = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
                else:
                    confidence = np.ones_like(decision_scores) * 0.5
            else:
                # フォールバック: 一定値
                confidence = np.ones(len(X)) * 0.5
            
            return confidence
            
        except Exception as e:
            logger.warning(f"信頼度計算エラー: {e}")
            return np.ones(len(X)) * 0.5
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """サポートベクター情報取得"""
        if not self.is_trained:
            return {}
        
        try:
            return {
                'n_support_vectors': self.model.n_support_,
                'support_vector_indices': self.model.support_.tolist() if hasattr(self.model, 'support_') else [],
                'dual_coefficients_norm': (
                    float(np.linalg.norm(self.model.dual_coef_)) 
                    if hasattr(self.model, 'dual_coef_') else None
                ),
                'intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None
            }
        except Exception as e:
            logger.warning(f"サポートベクター情報取得エラー: {e}")
            return {}
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """モデル複雑度情報取得"""
        if not self.is_trained:
            return {}
        
        support_info = self.get_support_vectors_info()
        
        return {
            'kernel': self.config['kernel'],
            'C': self.config['C'],
            'epsilon': self.config['epsilon'],
            'n_support_vectors': support_info.get('n_support_vectors', 0),
            'support_vector_ratio': (
                self.training_metrics.get('support_vector_ratio', 0.0)
                if self.training_metrics else 0.0
            )
        }
    
    def predict_with_margin(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        マージン付き予測
        
        Args:
            X: 入力特徴量
            
        Returns:
            予測値とマージン（信頼区間）のタプル
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")
        
        try:
            predictions = self.pipeline.predict(X)
            margins = np.full_like(predictions, self.config['epsilon'])
            
            return predictions, margins
            
        except Exception as e:
            logger.error(f"マージン付き予測エラー: {e}")
            raise


if __name__ == "__main__":
    # テスト実行
    print("=== Support Vector Regression Model テスト ===")
    
    # テストデータ生成（非線形パターン）
    np.random.seed(42)
    n_samples, n_features = 1000, 15
    X = np.random.randn(n_samples, n_features)
    # 非線形関数
    y = (np.sum(X[:, :3]**2, axis=1) + 
         np.sin(X[:, 3]) * X[:, 4] + 
         0.2 * np.random.randn(n_samples))
    
    # 訓練・検証データ分割
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # モデル初期化・学習
    svr_model = SVRModel({
        'enable_hyperopt': False,  # テスト用に高速化
        'kernel': 'rbf',
        'C': 10.0,
        'epsilon': 0.1
    })
    
    # 特徴量名設定
    svr_model.set_feature_names([f"feature_{i}" for i in range(n_features)])
    
    # 学習
    results = svr_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    print(f"学習完了: {results['training_time']:.2f}秒")
    print(f"サポートベクター比率: {results['support_vector_ratio']:.1%}")
    
    # 予測
    prediction = svr_model.predict(X_val)
    print(f"予測完了: {len(prediction.predictions)} サンプル")
    print(f"RMSE: {np.sqrt(np.mean((y_val - prediction.predictions)**2)):.4f}")
    
    # マージン付き予測
    pred_with_margin, margins = svr_model.predict_with_margin(X_val[:10])
    print(f"マージン付き予測例: {pred_with_margin[:3]}")
    print(f"マージン: {margins[:3]}")
    
    # モデル複雑度
    complexity = svr_model.get_model_complexity()
    print(f"モデル複雑度: {complexity}")
    
    # サポートベクター情報
    sv_info = svr_model.get_support_vectors_info()
    print(f"サポートベクター数: {sv_info.get('n_support_vectors', 0)}")