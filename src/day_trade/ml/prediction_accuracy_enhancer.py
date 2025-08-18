#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Accuracy Enhancer - 予測精度向上システム

Issue #885対応：予測精度向上のための包括的アプローチ実装
機械学習モデルの予測精度を向上させるための包括的なシステム

主要機能：
1. データ品質の監視と改善
2. 高度な特徴量エンジニアリング
3. モデル選択とアンサンブル
4. ハイパーパラメータ最適化の統合
5. 堅牢な検証戦略
6. 過学習防止
7. コンセプトドリフト対応
8. ドメイン知識の活用
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import warnings
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

# 機械学習ライブラリ
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 既存システムとの統合
try:
    from enhanced_feature_engineering import EnhancedFeatureEngineering
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from hyperparameter_optimizer import HyperparameterOptimizer
    HYPERPARAMETER_OPTIMIZER_AVAILABLE = True
except ImportError:
    HYPERPARAMETER_OPTIMIZER_AVAILABLE = False

try:
    from real_data_provider_v2 import MultiSourceDataProvider
    REAL_DATA_PROVIDER_AVAILABLE = True
except ImportError:
    REAL_DATA_PROVIDER_AVAILABLE = False


class DataQualityLevel(Enum):
    """データ品質レベル"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ModelComplexity(Enum):
    """モデル複雑度"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ValidationStrategy(Enum):
    """検証戦略"""
    TIME_SERIES_CV = "time_series_cv"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"


@dataclass
class DataQualityMetrics:
    """データ品質メトリクス"""
    completeness_score: float  # 完全性スコア (0-1)
    consistency_score: float   # 一貫性スコア (0-1)
    accuracy_score: float      # 正確性スコア (0-1)
    timeliness_score: float    # 適時性スコア (0-1)
    relevance_score: float     # 関連性スコア (0-1)
    overall_quality: DataQualityLevel
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FeatureImportanceMetrics:
    """特徴量重要度メトリクス"""
    feature_name: str
    importance_score: float
    stability_score: float      # 重要度の安定性
    correlation_with_target: float
    information_gain: float
    permutation_importance: float


@dataclass
class ModelPerformanceMetrics:
    """モデル性能メトリクス"""
    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    rmse: float
    r2_score: float
    stability_score: float       # 性能の安定性
    overfitting_score: float     # 過学習スコア
    drift_detection_score: float # ドリフト検知スコア


@dataclass
class EnhancementConfiguration:
    """予測精度向上設定"""
    # データ品質設定
    min_data_quality_threshold: float = 0.7
    data_quality_check_enabled: bool = True

    # 特徴量エンジニアリング設定
    max_features: int = 100
    feature_selection_enabled: bool = True
    correlation_threshold: float = 0.95

    # モデル設定
    ensemble_enabled: bool = True
    max_ensemble_models: int = 5

    # 検証設定
    validation_strategy: ValidationStrategy = ValidationStrategy.TIME_SERIES_CV
    cv_folds: int = 5

    # 過学習防止設定
    early_stopping_enabled: bool = True
    regularization_enabled: bool = True

    # ドリフト検知設定
    drift_detection_enabled: bool = True
    drift_detection_window: int = 100
    drift_threshold: float = 0.05


class DataQualityAnalyzer:
    """データ品質分析クラス"""

    def __init__(self, config: EnhancementConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def analyze_data_quality(self, data: pd.DataFrame, target_column: str) -> DataQualityMetrics:
        """データ品質の包括的分析"""
        try:
            # 完全性チェック
            completeness = self._calculate_completeness(data)

            # 一貫性チェック
            consistency = self._calculate_consistency(data)

            # 正確性チェック
            accuracy = self._calculate_accuracy(data, target_column)

            # 適時性チェック
            timeliness = self._calculate_timeliness(data)

            # 関連性チェック
            relevance = self._calculate_relevance(data, target_column)

            # 総合品質レベル決定
            overall_score = (completeness + consistency + accuracy + timeliness + relevance) / 5
            quality_level = self._determine_quality_level(overall_score)

            # 課題と推奨事項の生成
            issues, recommendations = self._generate_quality_recommendations(
                completeness, consistency, accuracy, timeliness, relevance
            )

            return DataQualityMetrics(
                completeness_score=completeness,
                consistency_score=consistency,
                accuracy_score=accuracy,
                timeliness_score=timeliness,
                relevance_score=relevance,
                overall_quality=quality_level,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"データ品質分析エラー: {e}")
            return DataQualityMetrics(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                timeliness_score=0.0,
                relevance_score=0.0,
                overall_quality=DataQualityLevel.POOR,
                issues=[f"分析エラー: {e}"],
                recommendations=["データの基本的な検証を実施してください"]
            )

    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """完全性スコア計算"""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        return max(0.0, 1.0 - (missing_cells / total_cells))

    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """一貫性スコア計算"""
        consistency_issues = 0
        total_checks = 0

        # 数値列のデータ型一貫性チェック
        for col in data.select_dtypes(include=[np.number]).columns:
            total_checks += 1
            # 異常に大きな値や小さな値の検出
            if not data[col].empty:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((data[col] < (q1 - 3 * iqr)) | (data[col] > (q3 + 3 * iqr))).sum()
                if outliers > len(data) * 0.1:  # 10%以上が外れ値
                    consistency_issues += 1

        return max(0.0, 1.0 - (consistency_issues / max(total_checks, 1)))

    def _calculate_accuracy(self, data: pd.DataFrame, target_column: str) -> float:
        """正確性スコア計算"""
        accuracy_issues = 0
        total_checks = 0

        # 基本的な正確性チェック
        for col in data.columns:
            if col == target_column:
                continue

            total_checks += 1

            # 負の価格や出来高の検出
            if 'price' in col.lower() or 'volume' in col.lower():
                if (data[col] < 0).any():
                    accuracy_issues += 1

            # ゼロ除算の可能性チェック
            if data[col].dtype in [np.float64, np.int64]:
                if (data[col] == 0).sum() > len(data) * 0.5:  # 50%以上がゼロ
                    accuracy_issues += 0.5

        return max(0.0, 1.0 - (accuracy_issues / max(total_checks, 1)))

    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """適時性スコア計算"""
        if 'timestamp' in data.columns or data.index.name == 'timestamp':
            try:
                # 最新データの新しさをチェック
                if 'timestamp' in data.columns:
                    latest_time = pd.to_datetime(data['timestamp']).max()
                else:
                    latest_time = data.index.max()

                time_diff = (datetime.now() - latest_time).total_seconds() / 3600  # 時間単位

                if time_diff < 1:
                    return 1.0
                elif time_diff < 24:
                    return 0.8
                elif time_diff < 168:  # 1週間
                    return 0.6
                else:
                    return 0.4

            except Exception:
                return 0.5

        return 0.5  # タイムスタンプ情報がない場合

    def _calculate_relevance(self, data: pd.DataFrame, target_column: str) -> float:
        """関連性スコア計算"""
        if not SCIPY_AVAILABLE or target_column not in data.columns:
            return 0.5

        try:
            target = data[target_column].dropna()
            relevance_scores = []

            for col in data.columns:
                if col == target_column:
                    continue

                col_data = data[col].dropna()
                if len(col_data) > 10:
                    # 相関計算
                    correlation, _ = pearsonr(
                        col_data[:min(len(col_data), len(target))],
                        target[:min(len(col_data), len(target))]
                    )
                    relevance_scores.append(abs(correlation))

            return np.mean(relevance_scores) if relevance_scores else 0.5

        except Exception:
            return 0.5

    def _determine_quality_level(self, overall_score: float) -> DataQualityLevel:
        """総合品質レベル決定"""
        if overall_score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif overall_score >= 0.7:
            return DataQualityLevel.GOOD
        elif overall_score >= 0.5:
            return DataQualityLevel.FAIR
        else:
            return DataQualityLevel.POOR

    def _generate_quality_recommendations(self, completeness: float, consistency: float,
                                        accuracy: float, timeliness: float,
                                        relevance: float) -> Tuple[List[str], List[str]]:
        """品質に基づく課題と推奨事項の生成"""
        issues = []
        recommendations = []

        if completeness < 0.8:
            issues.append("データの欠損率が高い")
            recommendations.append("欠損データの補完または除外を検討")

        if consistency < 0.7:
            issues.append("データの一貫性に問題がある")
            recommendations.append("外れ値の検出と処理を実施")

        if accuracy < 0.8:
            issues.append("データの正確性に疑問がある")
            recommendations.append("データソースの検証と清浄化を実施")

        if timeliness < 0.6:
            issues.append("データが古すぎる")
            recommendations.append("より新しいデータソースの確保を検討")

        if relevance < 0.5:
            issues.append("ターゲットとの関連性が低い特徴量が存在")
            recommendations.append("特徴量選択と特徴量エンジニアリングを実施")

        return issues, recommendations


class AdvancedFeatureSelector:
    """高度な特徴量選択クラス"""

    def __init__(self, config: EnhancementConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def select_features(self, X: pd.DataFrame, y: pd.Series,
                            task_type: str = 'classification') -> Tuple[pd.DataFrame, List[FeatureImportanceMetrics]]:
        """高度な特徴量選択の実行"""
        try:
            # 1. 相関による特徴量除去
            X_filtered = self._remove_correlated_features(X)

            # 2. 統計的特徴量選択
            X_selected = self._statistical_feature_selection(X_filtered, y, task_type)

            # 3. 機械学習ベースの特徴量選択
            X_final, importance_metrics = await self._ml_based_feature_selection(
                X_selected, y, task_type
            )

            # 4. 特徴量重要度の安定性評価
            stable_features = self._evaluate_feature_stability(X_final, y, importance_metrics)

            return X_final[stable_features], importance_metrics

        except Exception as e:
            self.logger.error(f"特徴量選択エラー: {e}")
            return X, []

    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """高相関特徴量の除去"""
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # 高相関の特徴量ペアを特定
        high_corr_features = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > self.config.correlation_threshold)
        ]

        self.logger.info(f"高相関により除去される特徴量: {len(high_corr_features)}")
        return X.drop(columns=high_corr_features)

    def _statistical_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                     task_type: str) -> pd.DataFrame:
        """統計的特徴量選択"""
        if not SKLEARN_AVAILABLE:
            return X

        try:
            if task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')

            selector.fit(X, y)

            # 上位特徴量を選択
            k_best = min(self.config.max_features, len(X.columns))
            top_features = X.columns[np.argsort(selector.scores_)[-k_best:]]

            return X[top_features]

        except Exception as e:
            self.logger.warning(f"統計的特徴量選択でエラー: {e}")
            return X

    async def _ml_based_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                        task_type: str) -> Tuple[pd.DataFrame, List[FeatureImportanceMetrics]]:
        """機械学習ベースの特徴量選択"""
        importance_metrics = []

        if not SKLEARN_AVAILABLE:
            return X, importance_metrics

        try:
            # Random Forestによる特徴量重要度
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X, y)

            # 特徴量重要度メトリクスの作成
            for i, feature in enumerate(X.columns):
                if SCIPY_AVAILABLE:
                    correlation, _ = pearsonr(X[feature], y)
                else:
                    correlation = 0.0

                importance_metrics.append(FeatureImportanceMetrics(
                    feature_name=feature,
                    importance_score=model.feature_importances_[i],
                    stability_score=1.0,  # 後で計算
                    correlation_with_target=abs(correlation),
                    information_gain=model.feature_importances_[i],
                    permutation_importance=model.feature_importances_[i]
                ))

            # 重要度順にソート
            importance_metrics.sort(key=lambda x: x.importance_score, reverse=True)

            # 上位特徴量を選択
            top_k = min(self.config.max_features, len(importance_metrics))
            selected_features = [m.feature_name for m in importance_metrics[:top_k]]

            return X[selected_features], importance_metrics

        except Exception as e:
            self.logger.error(f"ML特徴量選択エラー: {e}")
            return X, importance_metrics

    def _evaluate_feature_stability(self, X: pd.DataFrame, y: pd.Series,
                                  importance_metrics: List[FeatureImportanceMetrics]) -> List[str]:
        """特徴量重要度の安定性評価"""
        stable_features = []

        if not SKLEARN_AVAILABLE or len(X) < 100:
            return list(X.columns)

        try:
            # 複数のサブサンプルで重要度を計算
            n_samples = min(5, len(X) // 20)
            stability_scores = defaultdict(list)

            for _ in range(n_samples):
                # サブサンプル作成
                sample_indices = np.random.choice(len(X), size=len(X)//2, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]

                # Random Forestで重要度計算
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_sample, y_sample)

                for i, feature in enumerate(X_sample.columns):
                    stability_scores[feature].append(model.feature_importances_[i])

            # 安定性スコア計算（標準偏差の逆数）
            for feature, scores in stability_scores.items():
                if len(scores) > 1:
                    stability = 1.0 / (np.std(scores) + 1e-8)
                    if stability > 0.5:  # 閾値以上の安定性
                        stable_features.append(feature)

            # 最低限の特徴量数を確保
            if len(stable_features) < 10:
                stable_features = list(X.columns)[:20]

            return stable_features

        except Exception as e:
            self.logger.warning(f"安定性評価エラー: {e}")
            return list(X.columns)


class ConceptDriftDetector:
    """コンセプトドリフト検知クラス"""

    def __init__(self, config: EnhancementConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reference_data = None
        self.drift_history = deque(maxlen=config.drift_detection_window)

    async def detect_drift(self, current_data: pd.DataFrame,
                         reference_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """コンセプトドリフトの検知"""
        if reference_data is not None:
            self.reference_data = reference_data

        if self.reference_data is None:
            return {"drift_detected": False, "drift_score": 0.0, "analysis": "参照データなし"}

        try:
            # 統計的ドリフト検知
            statistical_drift = self._statistical_drift_detection(current_data)

            # 分布ドリフト検知
            distribution_drift = self._distribution_drift_detection(current_data)

            # 総合ドリフトスコア
            overall_drift_score = (statistical_drift + distribution_drift) / 2

            # ドリフト判定
            drift_detected = overall_drift_score > self.config.drift_threshold

            # 履歴に追加
            self.drift_history.append({
                'timestamp': datetime.now(),
                'drift_score': overall_drift_score,
                'detected': drift_detected
            })

            return {
                "drift_detected": drift_detected,
                "drift_score": overall_drift_score,
                "statistical_drift": statistical_drift,
                "distribution_drift": distribution_drift,
                "analysis": self._generate_drift_analysis(overall_drift_score, drift_detected)
            }

        except Exception as e:
            self.logger.error(f"ドリフト検知エラー: {e}")
            return {"drift_detected": False, "drift_score": 0.0, "analysis": f"エラー: {e}"}

    def _statistical_drift_detection(self, current_data: pd.DataFrame) -> float:
        """統計的ドリフト検知"""
        if not SCIPY_AVAILABLE:
            return 0.0

        drift_scores = []

        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in self.reference_data.columns:
                try:
                    # Kolmogorov-Smirnov検定
                    ref_values = self.reference_data[column].dropna()
                    curr_values = current_data[column].dropna()

                    if len(ref_values) > 10 and len(curr_values) > 10:
                        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
                        drift_scores.append(statistic)

                except Exception as e:
                    self.logger.debug(f"列{column}の統計検定エラー: {e}")
                    continue

        return np.mean(drift_scores) if drift_scores else 0.0

    def _distribution_drift_detection(self, current_data: pd.DataFrame) -> float:
        """分布ドリフト検知"""
        drift_scores = []

        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in self.reference_data.columns:
                try:
                    ref_values = self.reference_data[column].dropna()
                    curr_values = current_data[column].dropna()

                    if len(ref_values) > 10 and len(curr_values) > 10:
                        # 平均と標準偏差の変化
                        ref_mean, ref_std = ref_values.mean(), ref_values.std()
                        curr_mean, curr_std = curr_values.mean(), curr_values.std()

                        mean_change = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                        std_change = abs(curr_std - ref_std) / (ref_std + 1e-8)

                        drift_scores.append((mean_change + std_change) / 2)

                except Exception as e:
                    self.logger.debug(f"列{column}の分布検定エラー: {e}")
                    continue

        return np.mean(drift_scores) if drift_scores else 0.0

    def _generate_drift_analysis(self, drift_score: float, drift_detected: bool) -> str:
        """ドリフト分析結果の生成"""
        if drift_detected:
            return f"コンセプトドリフトが検知されました (スコア: {drift_score:.3f}). モデルの再学習を検討してください。"
        else:
            return f"データは安定しています (スコア: {drift_score:.3f}). 現在のモデルを継続使用可能です。"


class PredictionAccuracyEnhancer:
    """予測精度向上メインクラス"""

    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)

        # 設定読み込み
        self.config = self._load_configuration(config_path)

        # コンポーネント初期化
        self.data_quality_analyzer = DataQualityAnalyzer(self.config)
        self.feature_selector = AdvancedFeatureSelector(self.config)
        self.drift_detector = ConceptDriftDetector(self.config)

        # 外部システムとの統合
        self.feature_engineer = None
        self.hyperparameter_optimizer = None
        self.data_provider = None

        self._initialize_external_systems()

        # 内部状態
        self.enhancement_history = []
        self.current_best_models = {}

        self.logger.info("Prediction Accuracy Enhancer initialized")

    def _load_configuration(self, config_path: Optional[Path]) -> EnhancementConfiguration:
        """設定の読み込み"""
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                return EnhancementConfiguration(**config_data)
            except Exception as e:
                self.logger.warning(f"設定読み込みエラー: {e}. デフォルト設定を使用")

        return EnhancementConfiguration()

    def _initialize_external_systems(self):
        """外部システムの初期化"""
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                self.feature_engineer = EnhancedFeatureEngineering()
                self.logger.info("特徴量エンジニアリングシステム統合完了")

            if HYPERPARAMETER_OPTIMIZER_AVAILABLE:
                self.hyperparameter_optimizer = HyperparameterOptimizer()
                self.logger.info("ハイパーパラメータ最適化システム統合完了")

            if REAL_DATA_PROVIDER_AVAILABLE:
                self.data_provider = MultiSourceDataProvider()
                self.logger.info("データプロバイダーシステム統合完了")

        except Exception as e:
            self.logger.warning(f"外部システム統合エラー: {e}")

    async def enhance_prediction_accuracy(self, symbol: str,
                                        training_data: pd.DataFrame,
                                        target_column: str,
                                        task_type: str = 'classification') -> Dict[str, Any]:
        """予測精度向上の包括的実行"""
        try:
            self.logger.info(f"予測精度向上開始: {symbol}")

            enhancement_result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'steps_completed': [],
                'improvements': {},
                'recommendations': [],
                'final_models': {}
            }

            # 1. データ品質分析
            self.logger.info("Step 1: データ品質分析")
            quality_metrics = await self.data_quality_analyzer.analyze_data_quality(
                training_data, target_column
            )
            enhancement_result['data_quality'] = quality_metrics
            enhancement_result['steps_completed'].append('data_quality_analysis')

            # データ品質が低い場合は警告
            if quality_metrics.overall_quality == DataQualityLevel.POOR:
                self.logger.warning("データ品質が低いため、精度向上が制限される可能性があります")
                enhancement_result['recommendations'].extend(quality_metrics.recommendations)

            # 2. 高度な特徴量エンジニアリング
            if self.feature_engineer:
                self.logger.info("Step 2: 高度な特徴量エンジニアリング")
                enhanced_features = await self._apply_feature_engineering(training_data, symbol)
                if enhanced_features is not None:
                    training_data = pd.concat([training_data, enhanced_features], axis=1)
                    enhancement_result['steps_completed'].append('feature_engineering')

            # 3. 特徴量選択
            self.logger.info("Step 3: 高度な特徴量選択")
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]

            X_selected, feature_importance = await self.feature_selector.select_features(
                X, y, task_type
            )
            enhancement_result['feature_importance'] = feature_importance
            enhancement_result['selected_features'] = list(X_selected.columns)
            enhancement_result['steps_completed'].append('feature_selection')

            # 4. モデル最適化（ハイパーパラメータ最適化統合）
            if self.hyperparameter_optimizer:
                self.logger.info("Step 4: ハイパーパラメータ最適化")
                optimized_models = await self._optimize_models(X_selected, y, task_type)
                enhancement_result['optimized_models'] = optimized_models
                enhancement_result['steps_completed'].append('hyperparameter_optimization')

            # 5. アンサンブルモデル構築
            if self.config.ensemble_enabled:
                self.logger.info("Step 5: アンサンブルモデル構築")
                ensemble_result = await self._build_ensemble_model(X_selected, y, task_type)
                enhancement_result['ensemble_model'] = ensemble_result
                enhancement_result['steps_completed'].append('ensemble_modeling')

            # 6. 堅牢な検証
            self.logger.info("Step 6: 堅牢な検証")
            validation_result = await self._robust_validation(X_selected, y, task_type)
            enhancement_result['validation_result'] = validation_result
            enhancement_result['steps_completed'].append('robust_validation')

            # 7. コンセプトドリフト検知設定
            if self.config.drift_detection_enabled:
                self.logger.info("Step 7: ドリフト検知設定")
                await self.drift_detector.detect_drift(training_data)
                enhancement_result['steps_completed'].append('drift_detection_setup')

            # 8. 最終的な推奨事項生成
            enhancement_result['recommendations'].extend(
                self._generate_final_recommendations(enhancement_result)
            )

            # 履歴に追加
            self.enhancement_history.append(enhancement_result)

            self.logger.info(f"予測精度向上完了: {symbol}")
            return enhancement_result

        except Exception as e:
            self.logger.error(f"予測精度向上エラー: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': str(e),
                'steps_completed': [],
                'recommendations': ['エラーが発生しました。データとシステムの確認をしてください。']
            }

    async def _apply_feature_engineering(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """特徴量エンジニアリングの適用"""
        try:
            if hasattr(self.feature_engineer, 'calculate_enhanced_features'):
                enhanced_features = await self.feature_engineer.calculate_enhanced_features(data, symbol)
                return enhanced_features
            return None
        except Exception as e:
            self.logger.warning(f"特徴量エンジニアリングエラー: {e}")
            return None

    async def _optimize_models(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """モデル最適化の実行"""
        optimization_results = {}

        try:
            if hasattr(self.hyperparameter_optimizer, 'optimize_all_models'):
                # ハイパーパラメータ最適化システムとの統合
                y_dict = {'price_direction': y} if task_type == 'classification' else {'price_regression': y}
                results = await self.hyperparameter_optimizer.optimize_all_models(
                    'optimization', X, y_dict
                )
                optimization_results.update(results)

            return optimization_results

        except Exception as e:
            self.logger.warning(f"モデル最適化エラー: {e}")
            return {}

    async def _build_ensemble_model(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """アンサンブルモデルの構築"""
        try:
            if not SKLEARN_AVAILABLE:
                return {"status": "sklearn not available"}

            # 複数のモデルを訓練
            models = []

            if task_type == 'classification':
                models.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))
                if XGBOOST_AVAILABLE:
                    models.append(('xgb', xgb.XGBClassifier(random_state=42)))
            else:
                models.append(('rf', RandomForestRegressor(n_estimators=100, random_state=42)))
                if XGBOOST_AVAILABLE:
                    models.append(('xgb', xgb.XGBRegressor(random_state=42)))

            # 各モデルの性能評価
            model_scores = {}
            for name, model in models:
                try:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2')
                    model_scores[name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'model': model
                    }
                except Exception as e:
                    self.logger.warning(f"モデル{name}の評価エラー: {e}")
                    continue

            return {
                'models': model_scores,
                'ensemble_strategy': 'weighted_average',
                'num_models': len(model_scores)
            }

        except Exception as e:
            self.logger.error(f"アンサンブルモデル構築エラー: {e}")
            return {"error": str(e)}

    async def _robust_validation(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
        """堅牢な検証の実行"""
        try:
            validation_results = {}

            if not SKLEARN_AVAILABLE:
                return {"status": "sklearn not available"}

            # 時系列交差検証
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)

            # Random Forestでの検証
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = 'r2'

            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

            validation_results['time_series_cv'] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores.tolist()
            }

            # 過学習検知
            model.fit(X, y)
            train_score = model.score(X, y)
            overfitting_indicator = train_score - np.mean(scores)

            validation_results['overfitting_analysis'] = {
                'train_score': train_score,
                'cv_mean_score': np.mean(scores),
                'overfitting_indicator': overfitting_indicator,
                'overfitting_detected': overfitting_indicator > 0.1
            }

            return validation_results

        except Exception as e:
            self.logger.error(f"堅牢な検証エラー: {e}")
            return {"error": str(e)}

    def _generate_final_recommendations(self, enhancement_result: Dict[str, Any]) -> List[str]:
        """最終的な推奨事項の生成"""
        recommendations = []

        # データ品質に基づく推奨
        if 'data_quality' in enhancement_result:
            quality = enhancement_result['data_quality']
            if quality.overall_quality == DataQualityLevel.POOR:
                recommendations.append("データ品質の改善が急務です")
            elif quality.overall_quality == DataQualityLevel.FAIR:
                recommendations.append("データ品質の向上により、さらなる精度改善が期待できます")

        # 特徴量に基づく推奨
        if 'selected_features' in enhancement_result:
            num_features = len(enhancement_result['selected_features'])
            if num_features < 10:
                recommendations.append("特徴量が少ないため、追加の特徴量エンジニアリングを検討してください")
            elif num_features > 50:
                recommendations.append("特徴量が多すぎる可能性があります。さらなる特徴量選択を検討してください")

        # 検証結果に基づく推奨
        if 'validation_result' in enhancement_result:
            validation = enhancement_result['validation_result']
            if 'overfitting_analysis' in validation and validation['overfitting_analysis'].get('overfitting_detected'):
                recommendations.append("過学習が検知されました。正則化やデータ拡張を検討してください")

        # 一般的な推奨事項
        recommendations.append("定期的なモデル性能監視とドリフト検知を実施してください")
        recommendations.append("新しいデータでの継続的な再学習を計画してください")

        return recommendations

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """予測精度向上の実行履歴サマリー"""
        if not self.enhancement_history:
            return {"status": "No enhancement history available"}

        return {
            "total_enhancements": len(self.enhancement_history),
            "recent_enhancements": self.enhancement_history[-5:],
            "common_recommendations": self._get_common_recommendations(),
            "system_integrations": {
                "feature_engineering": FEATURE_ENGINEERING_AVAILABLE,
                "hyperparameter_optimization": HYPERPARAMETER_OPTIMIZER_AVAILABLE,
                "data_provider": REAL_DATA_PROVIDER_AVAILABLE
            }
        }

    def _get_common_recommendations(self) -> List[str]:
        """共通の推奨事項の抽出"""
        all_recommendations = []
        for enhancement in self.enhancement_history:
            all_recommendations.extend(enhancement.get('recommendations', []))

        # 頻出推奨事項の特定
        from collections import Counter
        recommendation_counts = Counter(all_recommendations)
        common_recommendations = [
            rec for rec, count in recommendation_counts.most_common(5)
        ]

        return common_recommendations


# ファクトリー関数
def create_prediction_accuracy_enhancer(config_path: Optional[str] = None) -> PredictionAccuracyEnhancer:
    """
    PredictionAccuracyEnhancerインスタンスの作成

    Args:
        config_path: 設定ファイルパス

    Returns:
        PredictionAccuracyEnhancerインスタンス
    """
    config_path_obj = Path(config_path) if config_path else None
    return PredictionAccuracyEnhancer(config_path=config_path_obj)


# グローバルインスタンス
try:
    prediction_accuracy_enhancer = PredictionAccuracyEnhancer()
    logging.info("Global prediction accuracy enhancer initialized")
except Exception as e:
    logging.warning(f"Global prediction accuracy enhancer initialization failed: {e}")
    prediction_accuracy_enhancer = None


if __name__ == "__main__":
    # 基本動作確認は別ファイルに分離
    pass