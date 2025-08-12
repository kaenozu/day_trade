#!/usr/bin/env python3
"""
Gradient Boosting Model for Ensemble Learning

Issue #462: Gradient Boostingベースモデルの実装
高精度な予測能力を持つ勾配ブースティング
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from .base_model_interface import BaseModelInterface, ModelPrediction
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class GradientBoostingModel(BaseModelInterface):
    """
    Gradient Boosting回帰モデル
    
    特徴:
    - 高い予測精度
    - 逐次的特徴量学習
    - 残差に基づく学習
    - 早期停止による過学習防止
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化
        
        Args:
            config: モデル設定辞書
        """
        default_config = {
            # Gradient Boosting設定
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'max_features': 'sqrt',
            'subsample': 0.8,
            'random_state': 42,
            
            # 学習設定
            'enable_hyperopt': True,
            'cv_folds': 5,
            'normalize_features': True,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            
            # パフォーマンス設定
            'verbose': 0,
            'warm_start': False,
        }
        
        # 設定をマージ
        final_config = {**default_config, **(config or {})}
        super().__init__("GradientBoosting", final_config)
        
        # スケーラー
        self.scaler = StandardScaler() if self.config['normalize_features'] else None
        self.best_params = {}
        self.feature_importances_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Gradient Boostingモデル学習
        
        Args:
            X: 訓練データの特徴量 (n_samples, n_features)
            y: 訓練データの目標変数 (n_samples,)
            validation_data: 検証データ (X_val, y_val)
            
        Returns:
            学習結果辞書
        """
        start_time = time.time()
        logger.info(f"Gradient Boosting学習開始: データ形状 {X.shape}")
        
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
                self.model = GradientBoostingRegressor(**self._get_gbm_params())
                self.model.fit(X_scaled, y)
            
            # 学習履歴の記録
            training_results = {
                'training_time': time.time() - start_time,
                'n_estimators_used': self.model.n_estimators_,
                'train_score': self.model.train_score_[-1] if hasattr(self.model, 'train_score_') else None,
                'oob_improvement': getattr(self.model, 'oob_improvement_', None)
            }
            
            # 検証データでの評価
            if validation_data is not None:
                X_val, y_val = validation_data
                if self.scaler is not None:
                    X_val_scaled = self.scaler.transform(X_val)
                else:
                    X_val_scaled = X_val.copy()
                    
                val_metrics = self.evaluate(X_val_scaled, y_val)
                training_results['validation_metrics'] = val_metrics
                logger.info(f"検証RMSE: {val_metrics.rmse:.4f}, Hit Rate: {val_metrics.hit_rate:.3f}")
            
            # 学習曲線の記録
            if hasattr(self.model, 'train_score_'):
                training_results['learning_curve'] = {
                    'train_scores': self.model.train_score_.tolist(),
                    'iterations': list(range(1, len(self.model.train_score_) + 1))
                }
            
            # 学習メトリクス保存
            self.training_metrics = training_results
            self.is_trained = True
            
            # 特徴量重要度取得
            feature_importance = self.get_feature_importance()
            training_results['feature_importance_top10'] = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            
            logger.info(f"Gradient Boosting学習完了: {time.time() - start_time:.2f}秒")
            return training_results
            
        except Exception as e:
            logger.error(f"Gradient Boosting学習エラー: {e}")
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
            
            # 段階別予測（学習の進行による予測変化）
            staged_predictions = self._get_staged_predictions(X_scaled)
            confidence = self._calculate_prediction_confidence(staged_predictions)
            
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
            logger.error(f"Gradient Boosting予測エラー: {e}")
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
    
    def _hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """
        ハイパーパラメータ最適化
        
        Args:
            X: 学習データ特徴量
            y: 学習データ目標変数
            
        Returns:
            最適化されたGradientBoostingRegressor
        """
        # 時系列交差検証
        tscv = TimeSeriesSplit(n_splits=self.config['cv_folds'])
        
        # 探索パラメータ空間（段階的最適化）
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Base model
        gbm_base = GradientBoostingRegressor(
            random_state=self.config['random_state'],
            verbose=self.config['verbose']
        )
        
        # Grid Search
        grid_search = GridSearchCV(
            gbm_base,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=self.config['verbose']
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        logger.info(f"最適パラメータ: {self.best_params}")
        logger.info(f"最適CV Score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _get_gbm_params(self) -> Dict[str, Any]:
        """Gradient Boostingパラメータ取得"""
        params = {
            'n_estimators': self.config['n_estimators'],
            'learning_rate': self.config['learning_rate'],
            'max_depth': self.config['max_depth'],
            'min_samples_split': self.config['min_samples_split'],
            'min_samples_leaf': self.config['min_samples_leaf'],
            'max_features': self.config['max_features'],
            'subsample': self.config['subsample'],
            'random_state': self.config['random_state'],
            'verbose': self.config['verbose'],
            'warm_start': self.config['warm_start']
        }
        
        # 早期停止設定
        if self.config['early_stopping']:
            params.update({
                'validation_fraction': self.config['validation_fraction'],
                'n_iter_no_change': self.config['n_iter_no_change']
            })
        
        return params
    
    def _get_staged_predictions(self, X: np.ndarray, n_stages: int = 10) -> np.ndarray:
        """
        段階別予測取得（学習の進行状況別）
        
        Args:
            X: 入力特徴量
            n_stages: 段階数
            
        Returns:
            各段階での予測結果 (n_stages, n_samples)
        """
        try:
            # 等間隔で段階を設定
            n_estimators = self.model.n_estimators_
            stages = np.linspace(
                max(1, n_estimators // n_stages),
                n_estimators,
                n_stages,
                dtype=int
            )
            
            staged_preds = []
            for stage in stages:
                pred = self.model.predict(X)  # TODO: staged_predict実装
                staged_preds.append(pred)
            
            return np.array(staged_preds)
            
        except Exception as e:
            logger.warning(f"段階別予測取得エラー: {e}")
            # フォールバック: 単一予測を複製
            single_pred = self.model.predict(X)
            return np.tile(single_pred, (n_stages, 1))
    
    def _calculate_prediction_confidence(self, staged_predictions: np.ndarray) -> np.ndarray:
        """
        予測信頼度計算
        
        段階別予測の分散を使用して信頼度を算出
        
        Args:
            staged_predictions: 段階別予測結果 (n_stages, n_samples)
            
        Returns:
            各サンプルの予測信頼度（標準偏差）
        """
        try:
            if staged_predictions.shape[0] > 1:
                # 段階間の標準偏差を信頼度とする
                confidence = np.std(staged_predictions, axis=0)
            else:
                # 単一予測の場合はデフォルト信頼度
                confidence = np.ones(staged_predictions.shape[1]) * 0.1
            
            return confidence
            
        except Exception as e:
            logger.warning(f"信頼度計算エラー: {e}")
            return np.zeros(staged_predictions.shape[1])
    
    def get_learning_curve(self) -> Dict[str, Any]:
        """学習曲線取得"""
        if not self.is_trained or not hasattr(self.model, 'train_score_'):
            return {}
        
        return {
            'train_scores': self.model.train_score_.tolist(),
            'iterations': list(range(1, len(self.model.train_score_) + 1)),
            'n_estimators_used': self.model.n_estimators_,
            'oob_improvement': getattr(self.model, 'oob_improvement_', [])
        }
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """モデル複雑度情報取得"""
        if not self.is_trained:
            return {}
        
        return {
            'n_estimators': self.model.n_estimators_,
            'max_depth': self.config['max_depth'],
            'learning_rate': self.config['learning_rate'],
            'subsample': self.config['subsample'],
            'final_train_score': (
                self.model.train_score_[-1] if hasattr(self.model, 'train_score_') 
                else None
            )
        }
    
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """
        学習曲線のプロット
        
        Args:
            save_path: 保存パス（Noneの場合は表示のみ）
        """
        try:
            import matplotlib.pyplot as plt
            
            curve_data = self.get_learning_curve()
            if not curve_data:
                logger.warning("学習曲線データが存在しません")
                return
            
            plt.figure(figsize=(10, 6))
            plt.plot(curve_data['iterations'], curve_data['train_scores'], 
                    label='Training Score', marker='o', markersize=3)
            
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Negative Log Likelihood')
            plt.title(f'{self.model_name} Learning Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"学習曲線保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlibが利用できません")
        except Exception as e:
            logger.error(f"学習曲線プロットエラー: {e}")


if __name__ == "__main__":
    # テスト実行
    print("=== Gradient Boosting Model テスト ===")
    
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
    gbm_model = GradientBoostingModel({
        'enable_hyperopt': False,  # テスト用に高速化
        'n_estimators': 100,
        'early_stopping': False
    })
    
    # 特徴量名設定
    gbm_model.set_feature_names([f"feature_{i}" for i in range(n_features)])
    
    # 学習
    results = gbm_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    print(f"学習完了: {results['training_time']:.2f}秒")
    
    # 予測
    prediction = gbm_model.predict(X_val)
    print(f"予測完了: {len(prediction.predictions)} サンプル")
    print(f"RMSE: {np.sqrt(np.mean((y_val - prediction.predictions)**2)):.4f}")
    
    # 特徴量重要度
    importance = gbm_model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"重要特徴量TOP5: {top_features}")
    
    # モデル複雑度
    complexity = gbm_model.get_model_complexity()
    print(f"モデル複雑度: {complexity}")
    
    # 学習曲線
    learning_curve = gbm_model.get_learning_curve()
    if learning_curve:
        print(f"最終学習スコア: {learning_curve['train_scores'][-1]:.4f}")