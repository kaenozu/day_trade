#!/usr/bin/env python3
"""
包括的予測評価システム
Issue #870: 予測精度向上のための包括的提案

全ての精度向上コンポーネントを統合し、包括的な評価・比較を実行
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib

# 実装したシステムのインポート
try:
    from advanced_feature_selector import create_advanced_feature_selector, FeatureCategory
    FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    FEATURE_SELECTOR_AVAILABLE = False

try:
    from advanced_ensemble_system import create_advanced_ensemble_system, EnsembleMethod
    ENSEMBLE_SYSTEM_AVAILABLE = True
except ImportError:
    ENSEMBLE_SYSTEM_AVAILABLE = False

try:
    from .hybrid_lstm_transformer import HybridLSTMTransformer as create_hybrid_timeseries_predictor
    HYBRID_PREDICTOR_AVAILABLE = True
except ImportError:
    HYBRID_PREDICTOR_AVAILABLE = False

try:
    from meta_learning_system import create_meta_learning_system, TaskType
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False


class EvaluationMetric(Enum):
    """評価指標"""
    MSE = "mse"                              # 平均二乗誤差
    RMSE = "rmse"                            # 平均二乗誤差平方根
    MAE = "mae"                              # 平均絶対誤差
    R2 = "r2"                                # 決定係数
    EXPLAINED_VARIANCE = "explained_variance" # 説明分散
    MAX_ERROR = "max_error"                  # 最大誤差
    DIRECTIONAL_ACCURACY = "directional_accuracy"  # 方向性精度
    PROFIT_POTENTIAL = "profit_potential"     # 利益ポテンシャル


class ComponentType(Enum):
    """コンポーネントタイプ"""
    BASELINE = "baseline"                    # ベースライン
    FEATURE_SELECTION = "feature_selection"  # 特徴量選択
    ENSEMBLE = "ensemble"                    # アンサンブル
    HYBRID_TIMESERIES = "hybrid_timeseries"  # ハイブリッド時系列
    META_LEARNING = "meta_learning"          # メタラーニング
    INTEGRATED = "integrated"                # 統合システム


@dataclass
class ComponentResult:
    """コンポーネント結果"""
    component_type: ComponentType
    component_name: str
    predictions: np.ndarray
    training_time: float
    prediction_time: float
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    model_details: Dict[str, Any] = field(default_factory=dict)
    memory_usage: float = 0.0


@dataclass
class EvaluationReport:
    """評価レポート"""
    timestamp: datetime
    dataset_info: Dict[str, Any]
    baseline_performance: Dict[str, float]
    component_results: List[ComponentResult]
    improvement_analysis: Dict[str, Any]
    statistical_significance: Dict[str, Any]
    recommendations: List[str]


class MetricsCalculator:
    """指標計算器"""

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """全指標計算"""
        metrics = {}

        try:
            # 基本回帰指標
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred)

            # 方向性精度（価格変動方向の予測精度）
            metrics['directional_accuracy'] = MetricsCalculator._directional_accuracy(y_true, y_pred)

            # 利益ポテンシャル（予測に基づく投資戦略の理論利益）
            metrics['profit_potential'] = MetricsCalculator._profit_potential(y_true, y_pred)

            # 安定性指標
            metrics['prediction_stability'] = MetricsCalculator._prediction_stability(y_pred)

        except Exception as e:
            logging.warning(f"指標計算エラー: {e}")
            metrics = {'error': str(e)}

        return metrics

    @staticmethod
    def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """方向性精度計算"""
        if len(y_true) < 2:
            return 0.0

        # 実際の変化方向
        true_direction = np.sign(np.diff(y_true))

        # 予測の変化方向
        pred_direction = np.sign(np.diff(y_pred))

        # 方向一致率
        correct_directions = np.sum(true_direction == pred_direction)
        total_directions = len(true_direction)

        return correct_directions / total_directions if total_directions > 0 else 0.0

    @staticmethod
    def _profit_potential(y_true: np.ndarray, y_pred: np.ndarray,
                         transaction_cost: float = 0.001) -> float:
        """利益ポテンシャル計算"""
        if len(y_true) < 2:
            return 0.0

        # 実際のリターン
        true_returns = np.diff(y_true) / y_true[:-1]

        # 予測に基づく投資判断（正の予測で買い、負の予測で売り）
        pred_signals = np.sign(np.diff(y_pred))

        # 戦略リターン（取引コスト考慮）
        strategy_returns = pred_signals * true_returns - np.abs(pred_signals) * transaction_cost

        # 累積リターン
        cumulative_return = np.prod(1 + strategy_returns) - 1

        return cumulative_return

    @staticmethod
    def _prediction_stability(y_pred: np.ndarray) -> float:
        """予測安定性計算"""
        if len(y_pred) < 2:
            return 1.0

        # 予測値の変動係数の逆数で安定性を表現
        pred_std = np.std(y_pred)
        pred_mean = np.mean(np.abs(y_pred))

        if pred_mean == 0:
            return 0.0

        cv = pred_std / pred_mean
        stability = 1.0 / (1.0 + cv)

        return stability


class BaselineModelFactory:
    """ベースラインモデル工場"""

    @staticmethod
    def create_simple_models() -> Dict[str, Any]:
        """シンプルなベースラインモデル作成"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge

        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42
            )
        }

        return models


class ComponentEvaluator:
    """コンポーネント評価器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def evaluate_baseline(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series) -> ComponentResult:
        """ベースライン評価"""
        self.logger.info("ベースライン評価開始")

        # 最良のベースラインモデルを選択
        baseline_models = BaselineModelFactory.create_simple_models()
        best_score = -np.inf
        best_model = None
        best_name = ""

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        for name, model in baseline_models.items():
            try:
                # 交差検証
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='r2')
                avg_score = np.mean(scores)

                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    best_name = name

            except Exception as e:
                self.logger.warning(f"ベースライン {name} 評価失敗: {e}")
                continue

        if best_model is None:
            # フォールバック
            best_model = RandomForestRegressor(n_estimators=10, random_state=42)
            best_name = "fallback_rf"

        # 訓練と予測
        start_time = datetime.now()
        best_model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        start_time = datetime.now()
        predictions = best_model.predict(X_test_scaled)
        prediction_time = (datetime.now() - start_time).total_seconds()

        # 指標計算
        metrics = MetricsCalculator.calculate_all_metrics(y_test.values, predictions)

        return ComponentResult(
            component_type=ComponentType.BASELINE,
            component_name=f"baseline_{best_name}",
            predictions=predictions,
            training_time=training_time,
            prediction_time=prediction_time,
            metrics=metrics,
            model_details={'cv_score': best_score, 'model_type': best_name}
        )

    def evaluate_feature_selection(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 price_data: pd.DataFrame) -> Optional[ComponentResult]:
        """特徴量選択評価"""
        if not FEATURE_SELECTOR_AVAILABLE:
            self.logger.warning("特徴量選択システムが利用できません")
            return None

        self.logger.info("特徴量選択システム評価開始")

        try:
            # 特徴量選択実行
            selector = create_advanced_feature_selector(max_features=30)

            start_time = datetime.now()
            selected_X_train, selection_info = selector.select_features(
                X_train, y_train, price_data, method='ensemble'
            )
            feature_selection_time = (datetime.now() - start_time).total_seconds()

            # テストデータの特徴量選択
            selected_features = selection_info['selected_features']
            selected_X_test = X_test[selected_features]

            # ベースラインモデルで予測
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            X_train_scaled = self.scaler.fit_transform(selected_X_train)
            X_test_scaled = self.scaler.transform(selected_X_test)

            start_time = datetime.now()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            start_time = datetime.now()
            predictions = model.predict(X_test_scaled)
            prediction_time = (datetime.now() - start_time).total_seconds()

            # 指標計算
            metrics = MetricsCalculator.calculate_all_metrics(y_test.values, predictions)

            # 特徴量重要度
            feature_importance = dict(zip(
                selected_features,
                model.feature_importances_
            ))

            return ComponentResult(
                component_type=ComponentType.FEATURE_SELECTION,
                component_name="advanced_feature_selection",
                predictions=predictions,
                training_time=training_time + feature_selection_time,
                prediction_time=prediction_time,
                metrics=metrics,
                feature_importance=feature_importance,
                model_details={
                    'selected_features_count': len(selected_features),
                    'selection_ratio': selection_info.get('selection_ratio', 0),
                    'market_regime': selection_info.get('market_regime', 'unknown')
                }
            )

        except Exception as e:
            self.logger.error(f"特徴量選択評価エラー: {e}")
            return None

    def evaluate_ensemble_system(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> Optional[ComponentResult]:
        """アンサンブルシステム評価"""
        if not ENSEMBLE_SYSTEM_AVAILABLE:
            self.logger.warning("アンサンブルシステムが利用できません")
            return None

        self.logger.info("アンサンブルシステム評価開始")

        try:
            # アンサンブルシステム作成
            ensemble_system = create_advanced_ensemble_system(
                method=EnsembleMethod.STACKING,
                cv_folds=3
            )

            # 訓練
            start_time = datetime.now()
            ensemble_system.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            # 予測
            start_time = datetime.now()
            predictions = ensemble_system.predict(X_test)
            prediction_time = (datetime.now() - start_time).total_seconds()

            # 指標計算
            metrics = MetricsCalculator.calculate_all_metrics(y_test.values, predictions)

            # システム情報取得
            summary = ensemble_system.get_ensemble_summary()

            return ComponentResult(
                component_type=ComponentType.ENSEMBLE,
                component_name="advanced_ensemble_stacking",
                predictions=predictions,
                training_time=training_time,
                prediction_time=prediction_time,
                metrics=metrics,
                model_details={
                    'ensemble_models': summary.get('ensemble_models', []),
                    'best_ensemble': summary.get('best_ensemble', 'unknown'),
                    'performance': summary.get('performance', {})
                }
            )

        except Exception as e:
            self.logger.error(f"アンサンブル評価エラー: {e}")
            return None

    def evaluate_hybrid_timeseries(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series) -> Optional[ComponentResult]:
        """ハイブリッド時系列評価"""
        if not HYBRID_PREDICTOR_AVAILABLE:
            self.logger.warning("ハイブリッド時系列システムが利用できません")
            return None

        self.logger.info("ハイブリッド時系列システム評価開始")

        try:
            # ハイブリッド予測システム作成
            hybrid_predictor = create_hybrid_timeseries_predictor()

            # 時系列データ準備
            time_series_data = y_train.values.reshape(-1, 1)

            # 訓練
            start_time = datetime.now()
            hybrid_predictor.fit(time_series_data, X_train.values)
            training_time = (datetime.now() - start_time).total_seconds()

            # 予測
            start_time = datetime.now()
            predictions = hybrid_predictor.predict(
                steps=len(X_test),
                X_future=X_test.values,
                uncertainty=False
            )
            prediction_time = (datetime.now() - start_time).total_seconds()

            # 指標計算
            metrics = MetricsCalculator.calculate_all_metrics(y_test.values, predictions)

            return ComponentResult(
                component_type=ComponentType.HYBRID_TIMESERIES,
                component_name="hybrid_timeseries_predictor",
                predictions=predictions,
                training_time=training_time,
                prediction_time=prediction_time,
                metrics=metrics,
                model_details={
                    'predictor_type': 'hybrid_state_space_lstm_ml',
                    'uncertainty_quantification': True
                }
            )

        except Exception as e:
            self.logger.error(f"ハイブリッド時系列評価エラー: {e}")
            return None

    def evaluate_meta_learning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series,
                             price_data: pd.DataFrame) -> Optional[ComponentResult]:
        """メタラーニング評価"""
        if not META_LEARNING_AVAILABLE:
            self.logger.warning("メタラーニングシステムが利用できません")
            return None

        self.logger.info("メタラーニングシステム評価開始")

        try:
            # メタラーニングシステム作成
            meta_system = create_meta_learning_system(repository_size=50)

            # 訓練・予測
            start_time = datetime.now()
            model, predictions, result_info = meta_system.fit_predict(
                X_train, y_train, price_data,
                task_type=TaskType.REGRESSION,
                X_predict=X_test
            )
            total_time = (datetime.now() - start_time).total_seconds()

            # 指標計算
            metrics = MetricsCalculator.calculate_all_metrics(y_test.values, predictions)

            # 学習洞察取得
            insights = meta_system.get_learning_insights()

            return ComponentResult(
                component_type=ComponentType.META_LEARNING,
                component_name="meta_learning_system",
                predictions=predictions,
                training_time=result_info.get('training_time', 0),
                prediction_time=total_time - result_info.get('training_time', 0),
                metrics=metrics,
                model_details={
                    'selected_model': result_info.get('model_type', 'unknown'),
                    'market_condition': result_info.get('market_condition', 'unknown'),
                    'repository_size': insights.get('model_repository_size', 0),
                    'total_episodes': insights.get('total_episodes', 0)
                }
            )

        except Exception as e:
            self.logger.error(f"メタラーニング評価エラー: {e}")
            return None


class ComprehensivePredictionEvaluator:
    """包括的予測評価器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.component_evaluator = ComponentEvaluator()
        self.evaluation_history = []

    def run_comprehensive_evaluation(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series,
                                   price_data: pd.DataFrame,
                                   save_results: bool = True) -> EvaluationReport:
        """包括的評価実行"""
        self.logger.info("包括的予測精度評価開始")

        # データセット情報
        dataset_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'target_mean': float(y_train.mean()),
            'target_std': float(y_train.std()),
            'evaluation_timestamp': datetime.now()
        }

        # 各コンポーネント評価
        results = []

        # 1. ベースライン評価
        baseline_result = self.component_evaluator.evaluate_baseline(
            X_train, y_train, X_test, y_test
        )
        results.append(baseline_result)

        # 2. 特徴量選択システム評価
        feature_result = self.component_evaluator.evaluate_feature_selection(
            X_train, y_train, X_test, y_test, price_data
        )
        if feature_result:
            results.append(feature_result)

        # 3. アンサンブルシステム評価
        ensemble_result = self.component_evaluator.evaluate_ensemble_system(
            X_train, y_train, X_test, y_test
        )
        if ensemble_result:
            results.append(ensemble_result)

        # 4. ハイブリッド時系列評価
        hybrid_result = self.component_evaluator.evaluate_hybrid_timeseries(
            X_train, y_train, X_test, y_test
        )
        if hybrid_result:
            results.append(hybrid_result)

        # 5. メタラーニング評価
        meta_result = self.component_evaluator.evaluate_meta_learning(
            X_train, y_train, X_test, y_test, price_data
        )
        if meta_result:
            results.append(meta_result)

        # ベースライン性能
        baseline_performance = baseline_result.metrics

        # 改善分析
        improvement_analysis = self._analyze_improvements(baseline_result, results)

        # 統計的有意性検定
        statistical_significance = self._test_statistical_significance(
            y_test.values, baseline_result, results
        )

        # 推奨事項生成
        recommendations = self._generate_recommendations(
            baseline_result, results, improvement_analysis
        )

        # 評価レポート作成
        report = EvaluationReport(
            timestamp=datetime.now(),
            dataset_info=dataset_info,
            baseline_performance=baseline_performance,
            component_results=results,
            improvement_analysis=improvement_analysis,
            statistical_significance=statistical_significance,
            recommendations=recommendations
        )

        # 結果保存
        if save_results:
            self._save_evaluation_results(report)

        # 履歴追加
        self.evaluation_history.append(report)

        self.logger.info("包括的評価完了")
        return report

    def _analyze_improvements(self, baseline: ComponentResult,
                            results: List[ComponentResult]) -> Dict[str, Any]:
        """改善分析"""
        analysis = {
            'accuracy_improvements': {},
            'performance_gains': {},
            'efficiency_analysis': {},
            'best_performing_component': None,
            'cumulative_improvement': 0.0
        }

        baseline_r2 = baseline.metrics.get('r2', 0)
        best_r2 = baseline_r2
        best_component = None

        for result in results:
            if result.component_type == ComponentType.BASELINE:
                continue

            component_r2 = result.metrics.get('r2', 0)
            improvement = ((component_r2 - baseline_r2) / max(abs(baseline_r2), 1e-6)) * 100

            analysis['accuracy_improvements'][result.component_name] = improvement
            analysis['performance_gains'][result.component_name] = {
                'r2_improvement': improvement,
                'mse_reduction': ((baseline.metrics.get('mse', 1) - result.metrics.get('mse', 1)) /
                                baseline.metrics.get('mse', 1)) * 100,
                'directional_accuracy': result.metrics.get('directional_accuracy', 0) * 100
            }

            # 効率性分析
            training_efficiency = result.metrics.get('r2', 0) / max(result.training_time, 1e-6)
            prediction_efficiency = result.metrics.get('r2', 0) / max(result.prediction_time, 1e-6)

            analysis['efficiency_analysis'][result.component_name] = {
                'training_efficiency': training_efficiency,
                'prediction_efficiency': prediction_efficiency,
                'total_time': result.training_time + result.prediction_time
            }

            # 最良コンポーネント特定
            if component_r2 > best_r2:
                best_r2 = component_r2
                best_component = result.component_name

        analysis['best_performing_component'] = best_component
        analysis['cumulative_improvement'] = ((best_r2 - baseline_r2) / max(abs(baseline_r2), 1e-6)) * 100

        return analysis

    def _test_statistical_significance(self, y_true: np.ndarray,
                                     baseline: ComponentResult,
                                     results: List[ComponentResult]) -> Dict[str, Any]:
        """統計的有意性検定"""
        significance = {}

        try:
            from scipy import stats

            baseline_errors = np.abs(y_true - baseline.predictions)

            for result in results:
                if result.component_type == ComponentType.BASELINE:
                    continue

                component_errors = np.abs(y_true - result.predictions)

                # Wilcoxon符号付き順位検定
                statistic, p_value = stats.wilcoxon(
                    baseline_errors, component_errors, alternative='greater'
                )

                significance[result.component_name] = {
                    'p_value': p_value,
                    'statistically_significant': p_value < 0.05,
                    'effect_size': (np.mean(baseline_errors) - np.mean(component_errors)) / np.std(baseline_errors)
                }

        except ImportError:
            self.logger.warning("SciPy利用不可：統計的検定をスキップ")

        except Exception as e:
            self.logger.warning(f"統計的検定エラー: {e}")

        return significance

    def _generate_recommendations(self, baseline: ComponentResult,
                                results: List[ComponentResult],
                                improvement_analysis: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        best_component = improvement_analysis.get('best_performing_component')
        cumulative_improvement = improvement_analysis.get('cumulative_improvement', 0)

        # 精度改善に基づく推奨
        if cumulative_improvement > 10:
            recommendations.append(
                f"顕著な精度向上（{cumulative_improvement:.1f}%）が確認されました。"
                f"{best_component}の本格導入を推奨します。"
            )
        elif cumulative_improvement > 5:
            recommendations.append(
                f"中程度の精度向上（{cumulative_improvement:.1f}%）が確認されました。"
                f"継続的な改善と監視を推奨します。"
            )
        else:
            recommendations.append(
                "精度向上が限定的です。データ品質の改善や"
                "追加的な特徴量エンジニアリングを検討してください。"
            )

        # 効率性に基づく推奨
        efficiency_analysis = improvement_analysis.get('efficiency_analysis', {})
        if efficiency_analysis:
            most_efficient = max(
                efficiency_analysis.items(),
                key=lambda x: x[1]['training_efficiency']
            )[0]
            recommendations.append(
                f"計算効率の観点から{most_efficient}が最適です。"
            )

        # コンポーネント固有の推奨
        for result in results:
            if result.component_type == ComponentType.FEATURE_SELECTION:
                if result.metrics.get('r2', 0) > baseline.metrics.get('r2', 0):
                    recommendations.append(
                        "特徴量選択による精度向上が確認されました。"
                        "定期的な特徴量再選択の実装を推奨します。"
                    )

            elif result.component_type == ComponentType.ENSEMBLE:
                if result.metrics.get('directional_accuracy', 0) > 0.6:
                    recommendations.append(
                        "アンサンブル手法が良好な方向性予測を示しています。"
                        "取引戦略への組み込みを検討してください。"
                    )

        return recommendations

    def _save_evaluation_results(self, report: EvaluationReport) -> None:
        """評価結果保存"""
        try:
            # 結果ディレクトリ作成
            results_dir = Path("evaluation_results")
            results_dir.mkdir(exist_ok=True)

            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")

            # JSON形式でレポート保存
            report_file = results_dir / f"comprehensive_evaluation_{timestamp}.json"

            # レポートを辞書に変換
            report_dict = {
                'timestamp': report.timestamp.isoformat(),
                'dataset_info': report.dataset_info,
                'baseline_performance': report.baseline_performance,
                'component_results': [
                    {
                        'component_type': r.component_type.value,
                        'component_name': r.component_name,
                        'metrics': r.metrics,
                        'training_time': r.training_time,
                        'prediction_time': r.prediction_time,
                        'model_details': r.model_details
                    }
                    for r in report.component_results
                ],
                'improvement_analysis': report.improvement_analysis,
                'statistical_significance': report.statistical_significance,
                'recommendations': report.recommendations
            }

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)

            # 可視化結果保存
            self._create_visualizations(report, results_dir / f"plots_{timestamp}")

            self.logger.info(f"評価結果保存完了: {report_file}")

        except Exception as e:
            self.logger.error(f"結果保存エラー: {e}")

    def _create_visualizations(self, report: EvaluationReport, plots_dir: Path) -> None:
        """可視化作成"""
        try:
            plots_dir.mkdir(exist_ok=True)
            plt.style.use('default')

            # 1. 精度比較バープロット
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # R2スコア比較
            components = [r.component_name for r in report.component_results]
            r2_scores = [r.metrics.get('r2', 0) for r in report.component_results]

            ax1.bar(range(len(components)), r2_scores)
            ax1.set_xlabel('Components')
            ax1.set_ylabel('R² Score')
            ax1.set_title('R² Score Comparison')
            ax1.set_xticks(range(len(components)))
            ax1.set_xticklabels(components, rotation=45, ha='right')

            # 方向性精度比較
            directional_acc = [r.metrics.get('directional_accuracy', 0) * 100
                             for r in report.component_results]

            ax2.bar(range(len(components)), directional_acc)
            ax2.set_xlabel('Components')
            ax2.set_ylabel('Directional Accuracy (%)')
            ax2.set_title('Directional Accuracy Comparison')
            ax2.set_xticks(range(len(components)))
            ax2.set_xticklabels(components, rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(plots_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 改善分析ヒートマップ
            improvement_data = report.improvement_analysis.get('accuracy_improvements', {})
            if improvement_data:
                fig, ax = plt.subplots(figsize=(10, 6))

                components = list(improvement_data.keys())
                improvements = list(improvement_data.values())

                colors = ['red' if x < 0 else 'green' for x in improvements]
                bars = ax.bar(range(len(components)), improvements, color=colors, alpha=0.7)

                ax.set_xlabel('Components')
                ax.set_ylabel('Improvement (%)')
                ax.set_title('Accuracy Improvement vs Baseline')
                ax.set_xticks(range(len(components)))
                ax.set_xticklabels(components, rotation=45, ha='right')
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

                # 値をバーの上に表示
                for bar, value in zip(bars, improvements):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                           f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

                plt.tight_layout()
                plt.savefig(plots_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()

            self.logger.info(f"可視化作成完了: {plots_dir}")

        except Exception as e:
            self.logger.warning(f"可視化作成エラー: {e}")

    def print_summary_report(self, report: EvaluationReport) -> None:
        """要約レポート出力"""
        print("\n" + "="*80)
        print("包括的予測精度評価レポート")
        print("="*80)

        print(f"\n評価日時: {report.timestamp}")
        print(f"データセット: 訓練{report.dataset_info['train_samples']}件, "
              f"テスト{report.dataset_info['test_samples']}件, "
              f"特徴量{report.dataset_info['features']}個")

        print(f"\nベースライン性能:")
        baseline_perf = report.baseline_performance
        print(f"  R² Score: {baseline_perf.get('r2', 0):.4f}")
        print(f"  RMSE: {baseline_perf.get('rmse', 0):.4f}")
        print(f"  方向性精度: {baseline_perf.get('directional_accuracy', 0)*100:.1f}%")

        print(f"\nコンポーネント別結果:")
        for result in report.component_results:
            if result.component_type == ComponentType.BASELINE:
                continue

            print(f"\n  {result.component_name}:")
            print(f"    R² Score: {result.metrics.get('r2', 0):.4f}")
            print(f"    RMSE: {result.metrics.get('rmse', 0):.4f}")
            print(f"    方向性精度: {result.metrics.get('directional_accuracy', 0)*100:.1f}%")
            print(f"    訓練時間: {result.training_time:.2f}秒")
            print(f"    予測時間: {result.prediction_time:.2f}秒")

        print(f"\n改善分析:")
        improvement = report.improvement_analysis
        print(f"  最良コンポーネント: {improvement.get('best_performing_component', 'N/A')}")
        print(f"  累積改善率: {improvement.get('cumulative_improvement', 0):.1f}%")

        accuracy_improvements = improvement.get('accuracy_improvements', {})
        for component, improvement_pct in accuracy_improvements.items():
            print(f"  {component}: {improvement_pct:+.1f}%")

        print(f"\n推奨事項:")
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"  {i}. {recommendation}")

        print("\n" + "="*80)


def create_comprehensive_evaluator() -> ComprehensivePredictionEvaluator:
    """包括的評価器作成"""
    return ComprehensivePredictionEvaluator()


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)

    # サンプルデータ作成
    np.random.seed(42)
    n_samples, n_features = 1000, 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # 複雑な非線形関係
    y = (X['feature_0'] * 2 +
         X['feature_1'] ** 2 * 0.5 +
         X['feature_2'] * X['feature_3'] * 0.3 +
         np.sin(X['feature_4']) * 1.5 +
         np.log(np.abs(X['feature_5']) + 1) * 0.8 +
         np.random.randn(n_samples) * 0.1)

    # 価格データシミュレーション
    price_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(200) * 0.02) + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })

    # データ分割
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 包括的評価実行
    evaluator = create_comprehensive_evaluator()

    report = evaluator.run_comprehensive_evaluation(
        X_train, y_train, X_test, y_test, price_data
    )

    # 要約レポート出力
    evaluator.print_summary_report(report)

    print("包括的予測精度評価システムのテスト完了")