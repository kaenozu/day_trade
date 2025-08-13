#!/usr/bin/env python3
"""
Base Model Interface for Ensemble Learning

統一されたインターフェースによりアンサンブル学習での
異なるモデルの管理を簡素化
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ModelPrediction:
    """モデル予測結果"""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_name: str = "unknown"
    processing_time: float = 0.0


@dataclass
class ModelMetrics:
    """
    モデル評価指標

    Issue #493対応: 金融系指標の実装とドキュメント化

    Attributes:
        mse: 平均二乗誤差
        rmse: 平均二乗平方根誤差
        mae: 平均絶対誤差
        r2_score: 決定係数（予測精度）
        hit_rate: 方向性予測精度（0.0-1.0）
        sharpe_ratio: シャープレシオ（リスク調整後リターン、計算不可時はNone）
        max_drawdown: 最大ドローダウン（最大下落率、負の値、計算不可時はNone）
    """
    mse: float
    rmse: float
    mae: float
    r2_score: float
    hit_rate: float  # 方向性予測精度
    sharpe_ratio: Optional[float] = None  # リスク調整後リターン指標
    max_drawdown: Optional[float] = None  # 最大下落率（負の値）


class BaseModelInterface(ABC):
    """
    すべてのベースモデルが実装すべき統一インターフェース

    アンサンブル学習での一貫性を保つため、すべてのモデルが
    この基底クラスを継承し、共通のメソッドを実装する必要がある

    Issue #494対応: 統一ロギングポリシー
    ==========================================

    ロギングレベル使用ガイドライン:

    - INFO: システムの主要イベント・状態変化
      * モデル初期化、学習完了、保存/読み込み成功
      * 重要な計算結果・メトリクス

    - WARNING: 期待される動作からの逸脱（処理は継続）
      * データ不足によるフォールバック
      * 設定不備による代替処理

    - ERROR: システムエラー・処理失敗
      * 例外発生・処理中断
      * 必須データの欠損

    - DEBUG: 内部状態・詳細処理情報
      * パラメータ設定、中間計算結果
      * デバッグ用の詳細情報
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            model_name: モデル名
            config: モデル設定辞書
        """
        self.model_name = model_name
        self.config = config or {}
        # Issue #488対応: is_trainedプロパティの明確な定義
        self.is_trained = False  # fit()成功完了でTrue、初期化時や学習失敗時はFalse
        self.model = None
        self.training_metrics = {}
        self.feature_names = []

        logger.info(f"{model_name}モデル初期化")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        モデル学習

        Args:
            X: 訓練データの特徴量
            y: 訓練データの目標変数
            validation_data: 検証データ (X_val, y_val)

        Returns:
            学習結果辞書（メトリクス、学習時間等）
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """
        予測実行

        Args:
            X: 予測対象の特徴量

        Returns:
            ModelPrediction: 予測結果とメタ情報
        """
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """
        特徴量重要度取得
        
        Issue #495対応: 抽象メソッドから具体実装に変更
        
        全てのベースモデルが特徴量重要度を提供するわけではないため、
        抽象メソッドから外し、各モデルの実装クラスで可能な場合に
        オーバーライドする形に変更。
        
        デフォルト実装では空の辞書を返すため、特徴量重要度を
        提供できないモデルでもエラーが発生しない。
        
        Returns:
            特徴量名と重要度のマッピング
            特徴量重要度を提供できない場合は空の辞書
        """
        if not self.is_trained:
            logger.debug(f"{self.model_name}: 未学習モデルのため特徴量重要度を取得できません")
            return {}
        
        logger.debug(f"{self.model_name}: このモデルタイプでは特徴量重要度をサポートしていません")
        return {}
    
    def has_feature_importance(self) -> bool:
        """
        特徴量重要度提供可否チェック
        
        Issue #495対応: 特徴量重要度提供可否の明示的チェック
        
        Returns:
            特徴量重要度を提供できる場合True、できない場合False
        """
        if not self.is_trained:
            return False
        
        # デフォルト実装では空の辞書を返すだけなので、提供不可
        # 実装クラスで特徴量重要度を提供する場合はオーバーライドする
        return len(self.get_feature_importance()) > 0
    
    def _create_feature_importance_dict(self, importances: np.ndarray, 
                                       feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Issue #495対応: 特徴量重要度辞書作成ヘルパーメソッド
        
        各実装クラスで特徴量重要度辞書を作成する際の共通ロジック
        
        Args:
            importances: 特徴量重要度配列
            feature_names: 特徴量名リスト（未指定時はself.feature_namesを使用）
            
        Returns:
            特徴量名と重要度のマッピング
        """
        if len(importances) == 0:
            return {}
        
        # 特徴量名の決定
        if feature_names is None:
            feature_names = self.feature_names if hasattr(self, 'feature_names') and self.feature_names else []
        
        # 特徴量名と重要度の数が合わない場合はgeneric名を使用
        if len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
            logger.warning(f"{self.model_name}: 特徴量名数不一致のためgeneric名を使用")
        
        # 重要度でソート（降順）
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return dict(importance_pairs)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        モデル評価

        Args:
            X: 評価データの特徴量
            y: 評価データの目標変数

        Returns:
            ModelMetrics: 評価指標
        """
        try:
            prediction = self.predict(X)
            y_pred = prediction.predictions

            # 基本指標計算
            mse = np.mean((y - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y - y_pred))

            # R²スコア
            y_mean = np.mean(y)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Hit Rate（方向性予測精度）- Issue #492対応
            hit_rate = self._calculate_hit_rate(y, y_pred)

            # Issue #493対応: 金融系評価指標の計算
            sharpe_ratio = self._calculate_sharpe_ratio(y, y_pred)
            max_drawdown = self._calculate_max_drawdown(y_pred)

            return ModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2_score,
                hit_rate=hit_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )

        except Exception as e:
            logger.error(f"{self.model_name}評価エラー: {e}", exc_info=True)

            # エラーの詳細情報をログ出力 - Issue #492対応
            logger.error(
                f"評価データ情報 - X: {X.shape if hasattr(X, 'shape') else type(X)}, "
                f"y: {y.shape if hasattr(y, 'shape') else type(y)}"
            )

            # エラー時はデフォルト値を返す
            return ModelMetrics(
                mse=float('inf'),
                rmse=float('inf'),
                mae=float('inf'),
                r2_score=-1.0,
                hit_rate=0.5,
                sharpe_ratio=None,  # Issue #493対応: エラー時も明示的に設定
                max_drawdown=None   # Issue #493対応: エラー時も明示的に設定
            )

    def save_model(self, filepath: str, compression: bool = True) -> bool:
        """
        モデル保存
        
        Issue #693対応: モデルタイプに応じた最適化保存
        
        Args:
            filepath: 保存先パス
            compression: 圧縮保存を使用するかどうか
            
        Returns:
            保存成功フラグ
            
        Note:
            - scikit-learn/XGBoost: joblibによる高速保存
            - PyTorch: 専用保存メソッド
            - その他: pickle保存（フォールバック）
        """
        import time
        start_time = time.time()
        
        try:
            # Issue #693対応: モデルタイプ検出と最適化保存方法選択
            save_method = self._detect_optimal_save_method(self.model)
            
            if save_method == "joblib":
                success = self._save_with_joblib(filepath, compression)
            elif save_method == "xgboost":
                success = self._save_with_xgboost(filepath)
            elif save_method == "pytorch":
                success = self._save_with_pytorch(filepath)
            elif save_method == "tensorflow":
                success = self._save_with_tensorflow(filepath)
            else:
                # フォールバック: pickle保存
                success = self._save_with_pickle(filepath)
                
            if success:
                save_time = time.time() - start_time
                logger.info(f"{self.model_name}モデル保存完了 ({save_method}): {filepath} ({save_time:.3f}秒)")
                
            return success

        except Exception as e:
            logger.error(f"{self.model_name}モデル保存エラー: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        モデル読み込み
        
        Issue #693対応: モデルタイプに応じた最適化読み込み
        
        Args:
            filepath: 読み込み元パス
            
        Returns:
            読み込み成功フラグ
            
        Note:
            - ファイル拡張子とメタデータからモデル形式を自動検出
            - 対応形式: joblib, XGBoost, PyTorch, TensorFlow, pickle
        """
        import time
        import os
        start_time = time.time()
        
        try:
            # ファイル存在確認
            if not os.path.exists(filepath):
                logger.error(f"{self.model_name}: モデルファイルが見つかりません - {filepath}")
                return False
            
            # Issue #693対応: ファイル形式の自動検出
            load_method = self._detect_optimal_load_method(filepath)
            
            if load_method == "joblib":
                success = self._load_with_joblib(filepath)
            elif load_method == "xgboost":
                success = self._load_with_xgboost(filepath)
            elif load_method == "pytorch":
                success = self._load_with_pytorch(filepath)
            elif load_method == "tensorflow":
                success = self._load_with_tensorflow(filepath)
            else:
                # フォールバック: pickle読み込み
                success = self._load_with_pickle(filepath)
                
            if success:
                load_time = time.time() - start_time
                # Issue #494対応: システムの重要操作成功 = INFO
                logger.info(f"{self.model_name}モデル読み込み完了 ({load_method}): {filepath} ({load_time:.3f}秒)")
                
            return success

        except Exception as e:
            # Issue #494対応: システムエラー = ERROR
            logger.error(f"{self.model_name}モデル読み込みエラー: {e}")
            return False

    def _detect_optimal_save_method(self, model) -> str:
        """
        Issue #693対応: モデルタイプに基づく最適保存方法の検出
        
        Args:
            model: 保存対象のモデル
            
        Returns:
            最適保存方法名
        """
        if model is None:
            return "pickle"
        
        model_type_name = type(model).__name__
        model_module = getattr(type(model), '__module__', '')
        
        try:
            # XGBoost検出
            if hasattr(model, 'save_model') and ('xgboost' in model_module or 'xgb' in model_type_name.lower()):
                return "xgboost"
            
            # PyTorch検出
            if hasattr(model, 'state_dict') and 'torch' in model_module:
                return "pytorch"
            
            # TensorFlow/Keras検出
            if (hasattr(model, 'save') and 
                ('tensorflow' in model_module or 'keras' in model_module)):
                return "tensorflow"
            
            # scikit-learn系検出
            if (hasattr(model, 'get_params') and hasattr(model, 'fit') and
                ('sklearn' in model_module or hasattr(model, 'predict'))):
                return "joblib"
            
        except Exception as e:
            logger.debug(f"モデルタイプ検出エラー: {e}")
        
        # フォールバック
        return "pickle"

    def _detect_optimal_load_method(self, filepath: str) -> str:
        """
        Issue #693対応: ファイル拡張子に基づく最適読み込み方法の検出
        
        Args:
            filepath: 読み込み対象ファイルパス
            
        Returns:
            最適読み込み方法名
        """
        import os
        
        # ファイル拡張子による判定
        _, ext = os.path.splitext(filepath.lower())
        
        if ext in ['.joblib', '.pkl.gz', '.pkl.bz2']:
            return "joblib"
        elif ext in ['.json', '.ubj'] or 'xgb' in filepath.lower():
            return "xgboost"  
        elif ext in ['.pt', '.pth', '.pytorch']:
            return "pytorch"
        elif ext in ['.h5', '.tf', '.pb'] or 'model' in filepath:
            return "tensorflow"
        else:
            # pickle形式と仮定
            return "pickle"

    def _save_with_joblib(self, filepath: str, compression: bool = True) -> bool:
        """Issue #693対応: joblib保存"""
        try:
            import joblib
            
            # メタデータ付きで保存
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names,
                'save_format': 'joblib'
            }
            
            # 圧縮設定
            compress_level = 3 if compression else 0
            joblib.dump(model_data, filepath, compress=compress_level)
            
            return True
            
        except ImportError:
            logger.warning("joblib未インストール - pickle保存にフォールバック")
            return self._save_with_pickle(filepath)
        except Exception as e:
            logger.error(f"joblib保存エラー: {e}")
            return False

    def _save_with_xgboost(self, filepath: str) -> bool:
        """Issue #693対応: XGBoost専用保存"""
        try:
            # XGBoostモデル本体を専用形式で保存
            self.model.save_model(filepath)
            
            # メタデータを別ファイルで保存
            import json
            metadata_path = filepath + '.meta'
            metadata = {
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names,
                'save_format': 'xgboost'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"XGBoost保存エラー: {e}")
            return False

    def _save_with_pytorch(self, filepath: str) -> bool:
        """Issue #693対応: PyTorch専用保存"""
        try:
            import torch
            
            # PyTorchモデルの状態辞書と追加情報を保存
            save_data = {
                'model_state_dict': self.model.state_dict(),
                'model_class': type(self.model).__name__,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names,
                'save_format': 'pytorch'
            }
            
            torch.save(save_data, filepath)
            return True
            
        except ImportError:
            logger.warning("PyTorch未インストール - pickle保存にフォールバック")
            return self._save_with_pickle(filepath)
        except Exception as e:
            logger.error(f"PyTorch保存エラー: {e}")
            return False

    def _save_with_tensorflow(self, filepath: str) -> bool:
        """Issue #693対応: TensorFlow/Keras専用保存"""
        try:
            # TensorFlow/Kerasモデル保存
            self.model.save(filepath)
            
            # メタデータを別ファイルで保存
            import json
            metadata_path = filepath + '.meta'
            metadata = {
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names,
                'save_format': 'tensorflow'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"TensorFlow保存エラー: {e}")
            return False

    def _save_with_pickle(self, filepath: str) -> bool:
        """Issue #693対応: pickle保存（フォールバック）"""
        try:
            import pickle
            
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'feature_names': self.feature_names,
                'save_format': 'pickle'
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"pickle保存エラー: {e}")
            return False

    def _load_with_joblib(self, filepath: str) -> bool:
        """Issue #693対応: joblib読み込み"""
        try:
            import joblib
            
            model_data = joblib.load(filepath)
            self._restore_model_data(model_data)
            
            return True
            
        except ImportError:
            logger.warning("joblib未インストール - pickle読み込みにフォールバック")
            return self._load_with_pickle(filepath)
        except Exception as e:
            logger.error(f"joblib読み込みエラー: {e}")
            return False

    def _load_with_xgboost(self, filepath: str) -> bool:
        """Issue #693対応: XGBoost専用読み込み"""
        try:
            import xgboost as xgb
            import json
            import os
            
            # XGBoostモデル読み込み
            # モデルタイプを推測（分類器 or 回帰器）
            try:
                self.model = xgb.XGBRegressor()
                self.model.load_model(filepath)
            except:
                self.model = xgb.XGBClassifier()
                self.model.load_model(filepath)
            
            # メタデータ読み込み
            metadata_path = filepath + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self._restore_model_data(metadata)
            else:
                # メタデータがない場合のフォールバック
                self.is_trained = True
                
            return True
            
        except ImportError:
            logger.warning("XGBoost未インストール - pickle読み込みにフォールバック")
            return self._load_with_pickle(filepath)
        except Exception as e:
            logger.error(f"XGBoost読み込みエラー: {e}")
            return False

    def _load_with_pytorch(self, filepath: str) -> bool:
        """Issue #693対応: PyTorch専用読み込み"""
        try:
            import torch
            
            # PyTorchデータ読み込み
            save_data = torch.load(filepath, map_location='cpu')
            
            # モデルクラス情報があれば使用（なければskip）
            if 'model_state_dict' in save_data and self.model is not None:
                self.model.load_state_dict(save_data['model_state_dict'])
            else:
                logger.warning("PyTorchモデル構造情報不足 - state_dictのみ保存")
                self.model = save_data.get('model_state_dict')
            
            # メタデータ復元
            self._restore_model_data(save_data)
            
            return True
            
        except ImportError:
            logger.warning("PyTorch未インストール - pickle読み込みにフォールバック")
            return self._load_with_pickle(filepath)
        except Exception as e:
            logger.error(f"PyTorch読み込みエラー: {e}")
            return False

    def _load_with_tensorflow(self, filepath: str) -> bool:
        """Issue #693対応: TensorFlow/Keras専用読み込み"""
        try:
            import tensorflow as tf
            import json
            import os
            
            # TensorFlow/Kerasモデル読み込み
            self.model = tf.keras.models.load_model(filepath)
            
            # メタデータ読み込み
            metadata_path = filepath + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self._restore_model_data(metadata)
            else:
                # メタデータがない場合のフォールバック
                self.is_trained = True
                
            return True
            
        except ImportError:
            logger.warning("TensorFlow未インストール - pickle読み込みにフォールバック")
            return self._load_with_pickle(filepath)
        except Exception as e:
            logger.error(f"TensorFlow読み込みエラー: {e}")
            return False

    def _load_with_pickle(self, filepath: str) -> bool:
        """Issue #693対応: pickle読み込み（フォールバック）"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self._restore_model_data(model_data)
            
            return True
            
        except Exception as e:
            logger.error(f"pickle読み込みエラー: {e}")
            return False

    def _restore_model_data(self, model_data: dict):
        """
        Issue #693対応: モデルデータ復元共通処理
        
        Args:
            model_data: 復元するモデルデータ辞書
        """
        if not isinstance(model_data, dict):
            logger.warning("モデルデータが辞書形式ではありません")
            return
        
        # モデル実体の復元（PyTorchの場合は特別処理済み）
        if 'model' in model_data:
            self.model = model_data['model']
        
        # 基本属性の復元
        self.model_name = model_data.get('model_name', self.model_name)
        self.config = model_data.get('config', {})
        self.is_trained = model_data.get('is_trained', False)
        self.training_metrics = model_data.get('training_metrics', {})
        self.feature_names = model_data.get('feature_names', [])

    def _calculate_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray,
                           min_samples: int = 2, zero_threshold: float = 1e-6) -> float:
        """
        改善されたHit Rate（方向性予測精度）計算 - Issue #492対応

        Args:
            y_true: 実際の値
            y_pred: 予測値
            min_samples: hit_rate計算に必要な最小サンプル数
            zero_threshold: 0値と判定する閾値

        Returns:
            hit_rate: 方向性予測精度（0.0-1.0）

        Note:
            - データ数が少ない場合は適切なデフォルト値を返す
            - np.sign()の0値を適切に処理
            - 連続する変化量での方向性一致率を計算
        """
        try:
            # データが不十分な場合
            if len(y_true) < min_samples or len(y_pred) < min_samples:
                # Issue #494対応: 期待される動作からの逸脱 = WARNING
                logger.warning(
                    f"{self.model_name}: Hit rate計算データ不足 "
                    f"(実際: {len(y_true)}, 予測: {len(y_pred)}, 必要: {min_samples}) "
                    f"- デフォルト値0.5を返します"
                )
                return 0.5

            # 前日比（変化量）を計算
            y_diff = np.diff(y_true)
            pred_diff = np.diff(y_pred)

            # 変化量データが不十分な場合
            if len(y_diff) == 0 or len(pred_diff) == 0:
                # Issue #494対応: 期待される動作からの逸脱 = WARNING
                logger.warning(f"{self.model_name}: Hit rate計算で変化量データなし - デフォルト値0.5を返します")
                return 0.5

            # 方向性を判定（0値の適切な処理）
            actual_direction = self._get_direction(y_diff, zero_threshold)
            predicted_direction = self._get_direction(pred_diff, zero_threshold)

            # 無効なサンプル（どちらも0の場合）を除外
            valid_mask = ~((actual_direction == 0) & (predicted_direction == 0))

            if not np.any(valid_mask):
                # Issue #494対応: 期待される動作からの逸脱 = WARNING
                logger.warning(f"{self.model_name}: Hit rate計算で有効な方向性データなし - デフォルト値0.5を返します")
                return 0.5

            # 有効なサンプルでの方向性一致率
            valid_actual = actual_direction[valid_mask]
            valid_predicted = predicted_direction[valid_mask]

            # 方向性一致
            direction_match = valid_actual == valid_predicted
            hit_rate = np.mean(direction_match)

            # Issue #494対応: 内部計算結果・詳細処理状況 = DEBUG
            logger.debug(
                f"{self.model_name}: Hit rate計算完了 - "
                f"hit_rate={hit_rate:.3f}, 有効サンプル={np.sum(valid_mask)}/{len(y_diff)}"
            )

            return float(hit_rate)

        except Exception as e:
            # Issue #494対応: システムエラー = ERROR（例外詳細付き）
            logger.error(f"{self.model_name}: Hit rate計算でエラーが発生しました: {e}", exc_info=True)
            return 0.5

    def _get_direction(self, values: np.ndarray, zero_threshold: float) -> np.ndarray:
        """
        値の方向性を判定（0値の適切な処理）

        Args:
            values: 判定対象の値
            zero_threshold: 0値と判定する閾値

        Returns:
            direction: 方向性 (1: 上昇, -1: 下降, 0: 変化なし)
        """
        # 絶対値が閾値以下の場合は0（変化なし）とする
        direction = np.zeros_like(values, dtype=int)

        # 上昇判定
        direction[values > zero_threshold] = 1

        # 下降判定
        direction[values < -zero_threshold] = -1

        return direction

    def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray,
                               risk_free_rate: float = 0.0, min_samples: int = 30) -> Optional[float]:
        """
        Issue #493対応: Sharpe Ratio計算

        金融分析でのリスク調整後リターンを測定する指標

        Args:
            y_true: 実際のリターン/価格変動
            y_pred: 予測リターン/価格変動
            risk_free_rate: リスクフリーレート（デフォルト0%）
            min_samples: 計算に必要な最小サンプル数

        Returns:
            Sharpe Ratio（リスク調整後リターン）、計算不可能な場合はNone

        Note:
            - リターンの平均を期待リターンとして使用
            - リターンの標準偏差をリスクとして使用
            - Sharpe = (期待リターン - リスクフリーレート) / リスクの標準偏差
        """
        try:
            # データ数が不十分な場合
            if len(y_true) < min_samples or len(y_pred) < min_samples:
                logger.debug(
                    f"{self.model_name}: Sharpe Ratio計算データ不足 "
                    f"(サンプル数: {min(len(y_true), len(y_pred))}, 必要: {min_samples})"
                )
                return None

            # リターン系列から変化率を計算（価格→リターン変換）
            if np.all(y_pred == y_pred[0]):
                # 全予測が同一の場合（リターン=0）
                logger.debug(f"{self.model_name}: 予測値が一定のためSharpe Ratio計算不可")
                return None

            # 予測リターンの統計量計算
            pred_returns = np.diff(y_pred) / np.abs(y_pred[:-1] + 1e-8)  # 相対変化率
            pred_returns = pred_returns[np.isfinite(pred_returns)]  # 無限値除去

            if len(pred_returns) == 0:
                logger.debug(f"{self.model_name}: 有効な予測リターンなし")
                return None

            mean_return = np.mean(pred_returns)
            return_std = np.std(pred_returns)

            # 標準偏差が0の場合（リスクなし）
            if return_std == 0 or np.isclose(return_std, 0):
                logger.debug(f"{self.model_name}: 予測リターンの標準偏差が0のためSharpe Ratio計算不可")
                return None

            # Sharpe Ratio計算
            excess_return = mean_return - risk_free_rate
            sharpe_ratio = excess_return / return_std

            # 数値の有効性チェック
            if np.isfinite(sharpe_ratio):
                logger.debug(f"{self.model_name}: Sharpe Ratio計算完了 - {sharpe_ratio:.4f}")
                return float(sharpe_ratio)
            else:
                logger.debug(f"{self.model_name}: Sharpe Ratio計算結果が無効値")
                return None

        except Exception as e:
            logger.debug(f"{self.model_name}: Sharpe Ratio計算エラー: {e}")
            return None

    def _calculate_max_drawdown(self, predictions: np.ndarray, min_samples: int = 10) -> Optional[float]:
        """
        Issue #493対応: Maximum Drawdown計算

        予測による投資パフォーマンスの最大下落幅を計算

        Args:
            predictions: 予測値系列
            min_samples: 計算に必要な最小サンプル数

        Returns:
            Max Drawdown（最大下落率、負の値）、計算不可能な場合はNone

        Note:
            - 予測値を累積リターンとして扱い、ピークからの最大下落を計算
            - Max Drawdown = (谷値 - 直前ピーク値) / 直前ピーク値
            - 通常は負の値で表現される
        """
        try:
            # データ数が不十分な場合
            if len(predictions) < min_samples:
                logger.debug(
                    f"{self.model_name}: Max Drawdown計算データ不足 "
                    f"(サンプル数: {len(predictions)}, 必要: {min_samples})"
                )
                return None

            # 予測値を累積リターンに変換
            cumulative_returns = np.cumsum(predictions)

            # ランニング最大値（ピーク値）を計算
            running_max = np.maximum.accumulate(cumulative_returns)

            # ドローダウン計算: (現在値 - ピーク値) / ピーク値
            drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)  # ゼロ除算回避

            # 無限値・NaN値の除去
            valid_drawdown = drawdown[np.isfinite(drawdown)]

            if len(valid_drawdown) == 0:
                logger.debug(f"{self.model_name}: 有効なドローダウンなし")
                return None

            # 最大ドローダウン（最も負の値）
            max_drawdown = np.min(valid_drawdown)

            # 数値の有効性チェック
            if np.isfinite(max_drawdown):
                logger.debug(f"{self.model_name}: Max Drawdown計算完了 - {max_drawdown:.4f}")
                return float(max_drawdown)
            else:
                logger.debug(f"{self.model_name}: Max Drawdown計算結果が無効値")
                return None

        except Exception as e:
            logger.debug(f"{self.model_name}: Max Drawdown計算エラー: {e}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報取得

        Returns:
            モデル情報辞書
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10]  # 最初の10個のみ
        }

    def set_feature_names(self, feature_names: List[str]):
        """
        Issue #494対応: 特徴量名設定

        Args:
            feature_names: 設定する特徴量名のリスト
        """
        self.feature_names = feature_names
        # Issue #494対応: 内部状態変更 = DEBUG
        logger.debug(f"{self.model_name}: 特徴量名を設定しました - {len(feature_names)}個")
    
    def validate_training_state(self, require_trained: bool = True) -> bool:
        """
        Issue #488対応: 学習状態の検証
        
        Args:
            require_trained: Trueなら学習済み状態を要求、Falseなら未学習状態を要求
            
        Returns:
            学習状態が期待通りならTrue、そうでなければFalse
            
        Raises:
            ValueError: 学習状態が期待と異なり、かつ厳密検証モードの場合
        """
        actual_trained = self.is_trained and self.model is not None
        
        if require_trained:
            if not actual_trained:
                logger.warning(
                    f"{self.model_name}: 学習済み状態が期待されていますが、"
                    f"is_trained={self.is_trained}, model={'存在' if self.model else 'None'}"
                )
                return False
        else:
            if actual_trained:
                logger.warning(
                    f"{self.model_name}: 未学習状態が期待されていますが、"
                    f"is_trained={self.is_trained}, model={'存在' if self.model else 'None'}"
                )
                return False
        
        return True
        
    def get_training_status(self) -> Dict[str, Any]:
        """
        Issue #488対応: 詳細な学習状態情報取得
        
        Returns:
            学習状態の詳細情報
        """
        return {
            'is_trained': self.is_trained,
            'has_model': self.model is not None,
            'has_training_metrics': len(self.training_metrics) > 0,
            'has_feature_names': len(self.feature_names) > 0,
            'model_type': type(self.model).__name__ if self.model else None,
            'consistent_state': self.is_trained == (self.model is not None)
        }

    def __str__(self) -> str:
        """文字列表現"""
        status = "Trained" if self.is_trained else "Untrained"
        return f"{self.model_name} ({status})"

    def __repr__(self) -> str:
        """詳細文字列表現"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', trained={self.is_trained})"

    def copy(self) -> 'BaseModelInterface':
        """
        Issue #483対応: モデルの安全なコピー作成
        
        継承クラスでオーバーライドして、モデル固有のコピーロジックを実装可能
        
        Returns:
            未学習状態の新しいモデルインスタンス
            
        Note:
            - デフォルト実装では未学習状態でコピーを作成
            - 学習済み状態を保持したい場合は継承クラスでオーバーライド
        """
        try:
            # 新しいインスタンス作成
            model_class = self.__class__
            new_model = model_class(self.model_name, self.config.copy() if self.config else None)
            
            # 基本属性のコピー
            new_model.feature_names = self.feature_names.copy() if self.feature_names else []
            new_model.is_trained = False  # 未学習状態で初期化
            new_model.training_metrics = {}  # 学習メトリクスクリア
            new_model.model = None  # モデル実体クリア
            
            # 設定の深いコピー
            if self.config:
                import copy
                new_model.config = copy.deepcopy(self.config)
            
            logger.debug(f"{self.model_name}: copyメソッドでコピー完了")
            return new_model
            
        except Exception as e:
            logger.error(f"{self.model_name}: copyメソッド失敗: {e}")
            raise