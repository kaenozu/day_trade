#!/usr/bin/env python3
"""
Accuracy Benchmark System for Ensemble Learning

Issue #462対応: アンサンブル学習システムの予測精度95%超達成のための
包括的ベンチマークと改善提案システム
"""

import time
import warnings
from typing import Dict, List, Any, Tuple, Optional, NamedTuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .ensemble_system import EnsembleSystem, EnsembleConfig, EnsembleMethod
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AccuracyMetrics(NamedTuple):
    """精度メトリクス"""
    mse: float
    rmse: float
    mae: float
    r2_score: float
    mape: float
    explained_variance: float
    hit_rate: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy_percentage: float


@dataclass
class BenchmarkConfig:
    """ベンチマーク設定"""
    # データ設定
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15

    # 時系列交差検証
    n_splits: int = 5
    gap: int = 0  # 学習・検証間のギャップ

    # 特徴量設定
    n_features: int = 20
    sequence_length: int = 30  # LSTM用シーケンス長

    # パフォーマンス評価
    calculate_financial_metrics: bool = True
    generate_detailed_report: bool = True

    # 最適化設定
    enable_hyperparameter_tuning: bool = True
    max_optimization_time: int = 300  # 秒


class AccuracyBenchmark:
    """
    アンサンブル学習システムの精度ベンチマーク

    現在の予測精度を測定し、95%達成に向けた改善提案を提供
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        初期化

        Args:
            config: ベンチマーク設定
        """
        self.config = config or BenchmarkConfig()
        self.benchmark_results: Dict[str, Any] = {}
        self.improvement_suggestions: List[str] = []

        logger.info(f"AccuracyBenchmark初期化完了: {self.config}")

    def generate_synthetic_stock_data(self, n_samples: int = 5000,
                                    add_noise: bool = True,
                                    trend_strength: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        高品質な合成株価データ生成

        Args:
            n_samples: サンプル数
            add_noise: ノイズ追加フラグ
            trend_strength: トレンド強度

        Returns:
            特徴量、目標変数、特徴量名のタプル
        """
        np.random.seed(42)  # 再現性確保

        # 時系列インデックス
        time_idx = np.arange(n_samples)

        # 基本的なトレンド・季節性
        trend = trend_strength * (time_idx / n_samples)
        seasonal = 0.1 * np.sin(2 * np.pi * time_idx / 252)  # 年次季節性
        weekly = 0.05 * np.sin(2 * np.pi * time_idx / 5)     # 週次パターン

        # 技術指標ベースの特徴量生成
        features = []
        feature_names = []

        # 1. 価格関連特徴量
        base_price = 100 + trend + seasonal + weekly
        if add_noise:
            base_price += np.random.normal(0, 0.5, n_samples)

        # 移動平均
        for window in [5, 10, 20]:
            ma = pd.Series(base_price).rolling(window).mean().fillna(method='bfill').values
            features.append(ma)
            feature_names.append(f'MA_{window}')

        # 2. ボラティリティ特徴量
        returns = np.diff(base_price, prepend=base_price[0])
        for window in [5, 10, 20]:
            vol = pd.Series(returns).rolling(window).std().fillna(0).values
            features.append(vol)
            feature_names.append(f'VOL_{window}')

        # 3. モメンタム特徴量
        for lag in [1, 5, 10]:
            momentum = np.concatenate([
                np.zeros(lag),
                np.diff(base_price, lag)
            ])[:n_samples]
            features.append(momentum)
            feature_names.append(f'MOMENTUM_{lag}')

        # 4. RSI風指標
        rsi_like = np.tanh(returns / np.std(returns)) * 50 + 50
        features.append(rsi_like)
        feature_names.append('RSI_LIKE')

        # 5. ランダム特徴量（ノイズテスト用）
        for i in range(3):
            noise_feature = np.random.normal(0, 1, n_samples)
            features.append(noise_feature)
            feature_names.append(f'NOISE_{i}')

        # 特徴量マトリックス作成
        X = np.column_stack(features)

        # 目標変数：翌日リターン予測
        y = np.concatenate([
            returns[1:],
            [returns[-1]]  # 最後の値は前の値をコピー
        ])

        # 正規化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        logger.info(f"合成データ生成完了: {X.shape}, target range: [{y.min():.4f}, {y.max():.4f}]")

        return X, y, feature_names

    def calculate_comprehensive_metrics(self, y_true: np.ndarray,
                                      y_pred: np.ndarray) -> AccuracyMetrics:
        """
        包括的精度メトリクス計算

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            AccuracyMetrics: 包括的な精度指標
        """
        # 基本的な回帰指標
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE（ゼロ除算対策）
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # 分散説明率
        explained_var = explained_variance_score(y_true, y_pred)

        # Hit Rate（方向性予測精度）
        if len(y_true) > 1:
            true_directions = np.sign(np.diff(y_true))
            pred_directions = np.sign(np.diff(y_pred))
            hit_rate = np.mean(true_directions == pred_directions)
        else:
            hit_rate = 0.5

        # 金融指標
        sharpe_ratio = self._calculate_sharpe_ratio(y_pred, y_true)
        max_drawdown = self._calculate_max_drawdown(y_pred)

        # 総合精度パーセンテージ（複数指標の組み合わせ）
        accuracy_components = [
            max(0, r2 * 100),  # R² to percentage
            max(0, (1 - rmse / np.std(y_true)) * 100),  # RMSE normalized
            hit_rate * 100,  # Hit rate as percentage
            max(0, (1 - mape / 100) * 100)  # MAPE inverted
        ]
        accuracy_percentage = np.mean(accuracy_components)

        return AccuracyMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            mape=mape,
            explained_variance=explained_var,
            hit_rate=hit_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            accuracy_percentage=min(99.99, accuracy_percentage)  # Cap at 99.99%
        )

    def _calculate_sharpe_ratio(self, predictions: np.ndarray,
                               actual: np.ndarray,
                               risk_free_rate: float = 0.02) -> float:
        """Sharpe比計算"""
        try:
            strategy_returns = predictions - np.mean(predictions)
            if np.std(strategy_returns) == 0:
                return 0.0
            sharpe = (np.mean(strategy_returns) - risk_free_rate) / np.std(strategy_returns)
            return sharpe
        except:
            return 0.0

    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """最大ドローダウン計算"""
        try:
            cumulative = np.cumsum(values)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            return np.max(drawdown) / (np.max(running_max) + 1e-8) * 100
        except:
            return 0.0

    def run_cross_validation_benchmark(self, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str],
                                     ensemble_config: Optional[EnsembleConfig] = None) -> Dict[str, Any]:
        """
        時系列交差検証ベンチマーク実行

        Args:
            X: 特徴量データ
            y: 目標変数
            feature_names: 特徴量名
            ensemble_config: アンサンブル設定

        Returns:
            ベンチマーク結果
        """
        if ensemble_config is None:
            ensemble_config = EnsembleConfig()

        logger.info(f"交差検証ベンチマーク開始: {X.shape}, CV={self.config.n_splits}分割")

        # 時系列分割
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            gap=self.config.gap
        )

        cv_results = []
        fold_predictions = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}/{self.config.n_splits} 実行中...")

            # データ分割
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 検証データ分割
            val_size = int(len(train_idx) * 0.2)
            if val_size > 0:
                X_val = X_train[-val_size:]
                y_val = y_train[-val_size:]
                X_train = X_train[:-val_size]
                y_train = y_train[:-val_size]
                validation_data = (X_val, y_val)
            else:
                validation_data = None

            try:
                # アンサンブルシステム作成・学習
                ensemble = EnsembleSystem(ensemble_config)

                start_time = time.time()
                train_results = ensemble.fit(X_train, y_train,
                                           validation_data=validation_data,
                                           feature_names=feature_names)
                training_time = time.time() - start_time

                # 予測実行
                prediction = ensemble.predict(X_test, method=EnsembleMethod.WEIGHTED)

                # メトリクス計算
                metrics = self.calculate_comprehensive_metrics(y_test, prediction.final_predictions)

                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'training_time': training_time,
                    'prediction_time': prediction.processing_time,
                    'metrics': metrics._asdict(),
                    'individual_predictions': prediction.individual_predictions,
                    'model_weights': prediction.model_weights,
                    'ensemble_confidence': np.mean(prediction.ensemble_confidence)
                }

                cv_results.append(fold_result)
                fold_predictions.append({
                    'y_true': y_test,
                    'y_pred': prediction.final_predictions,
                    'fold': fold + 1
                })

                logger.info(f"Fold {fold + 1} 完了: 精度={metrics.accuracy_percentage:.2f}%, "
                          f"Hit Rate={metrics.hit_rate:.3f}")

            except Exception as e:
                logger.error(f"Fold {fold + 1} エラー: {e}")
                cv_results.append({
                    'fold': fold + 1,
                    'error': str(e),
                    'metrics': None
                })

        # 結果集計
        successful_folds = [r for r in cv_results if 'error' not in r]

        if successful_folds:
            # 平均メトリクス計算
            avg_metrics = {}
            for metric_name in successful_folds[0]['metrics'].keys():
                values = [fold['metrics'][metric_name] for fold in successful_folds]
                avg_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

            # 全体結果
            benchmark_result = {
                'cross_validation_summary': {
                    'n_successful_folds': len(successful_folds),
                    'n_total_folds': self.config.n_splits,
                    'success_rate': len(successful_folds) / self.config.n_splits,
                    'avg_metrics': avg_metrics,
                    'individual_fold_results': cv_results,
                    'fold_predictions': fold_predictions
                },
                'ensemble_config': ensemble_config.__dict__,
                'benchmark_config': self.config.__dict__,
                'execution_timestamp': time.time()
            }

            # 改善提案生成
            self._generate_improvement_suggestions(avg_metrics, successful_folds)
            benchmark_result['improvement_suggestions'] = self.improvement_suggestions

            logger.info(f"交差検証完了: 平均精度={avg_metrics['accuracy_percentage']['mean']:.2f}% "
                       f"(±{avg_metrics['accuracy_percentage']['std']:.2f}%)")
        else:
            benchmark_result = {
                'error': '全てのFoldで実行に失敗しました',
                'failed_results': cv_results
            }

        return benchmark_result

    def _generate_improvement_suggestions(self, avg_metrics: Dict[str, Dict[str, float]],
                                        fold_results: List[Dict[str, Any]]):
        """改善提案生成"""
        suggestions = []

        current_accuracy = avg_metrics['accuracy_percentage']['mean']
        target_accuracy = 95.0

        suggestions.append(f"現在の平均精度: {current_accuracy:.2f}%")
        suggestions.append(f"目標精度: {target_accuracy:.2f}%")
        suggestions.append(f"改善必要量: {target_accuracy - current_accuracy:.2f}%")

        # R²スコアベースの提案
        r2_score = avg_metrics['r2_score']['mean']
        if r2_score < 0.9:
            suggestions.append("✦ R²スコア改善提案:")
            suggestions.append("  - より複雑な特徴量エンジニアリング")
            suggestions.append("  - ハイパーパラメータ最適化強化")
            suggestions.append("  - 追加の非線形モデル導入")

        # Hit Rate改善提案
        hit_rate = avg_metrics['hit_rate']['mean']
        if hit_rate < 0.85:
            suggestions.append("✦ Hit Rate改善提案:")
            suggestions.append("  - 方向性予測に特化したモデル追加")
            suggestions.append("  - 分類ベースのアンサンブル導入")
            suggestions.append("  - 時系列パターン認識モデル強化")

        # RMSE改善提案
        rmse_normalized = 1 - avg_metrics['rmse']['mean'] / np.sqrt(avg_metrics['mse']['mean'])
        if rmse_normalized < 0.9:
            suggestions.append("✦ RMSE改善提案:")
            suggestions.append("  - データ前処理パイプライン改善")
            suggestions.append("  - 外れ値除去アルゴリズム導入")
            suggestions.append("  - より高精度な基底モデル追加")

        # アンサンブル重み分析
        weight_analysis = self._analyze_model_weights(fold_results)
        if weight_analysis:
            suggestions.extend(weight_analysis)

        # 95%達成のための具体的提案
        suggestions.append("\n🎯 95%精度達成のための優先順位:")
        accuracy_gap = target_accuracy - current_accuracy

        if accuracy_gap > 10:
            suggestions.append("1. 【高優先度】基底モデルの大幅強化")
            suggestions.append("   - XGBoost/CatBoost追加")
            suggestions.append("   - Neural Network系モデル改良")
            suggestions.append("2. 【高優先度】特徴量エンジニアリング")
            suggestions.append("   - 高次特徴量作成")
            suggestions.append("   - 外部データソース統合")
        elif accuracy_gap > 5:
            suggestions.append("1. 【中優先度】ハイパーパラメータ最適化")
            suggestions.append("   - Optuna/Hyperopt導入")
            suggestions.append("   - Grid Search強化")
            suggestions.append("2. 【中優先度】アンサンブル手法改良")
            suggestions.append("   - Stacking層数増加")
            suggestions.append("   - 動的重み調整改善")
        else:
            suggestions.append("1. 【低優先度】微調整最適化")
            suggestions.append("   - 学習率調整")
            suggestions.append("   - 正則化パラメータ調整")

        self.improvement_suggestions = suggestions

    def _analyze_model_weights(self, fold_results: List[Dict[str, Any]]) -> List[str]:
        """モデル重み分析と提案"""
        suggestions = []

        try:
            # 各モデルの平均重み計算
            all_weights = {}
            for result in fold_results:
                if 'model_weights' in result:
                    for model, weight in result['model_weights'].items():
                        if model not in all_weights:
                            all_weights[model] = []
                        all_weights[model].append(weight)

            if all_weights:
                suggestions.append("✦ モデル重み分析:")
                avg_weights = {model: np.mean(weights) for model, weights in all_weights.items()}

                # 重みが低いモデルの特定
                sorted_weights = sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)

                for model, weight in sorted_weights:
                    suggestions.append(f"  - {model}: {weight:.3f}")

                # 最も重みの低いモデルに対する提案
                lowest_weight_model = sorted_weights[-1][0]
                if sorted_weights[-1][1] < 0.1:
                    suggestions.append(f"  ⚠ {lowest_weight_model} の重みが低い - パフォーマンス改善が必要")

        except Exception as e:
            logger.warning(f"重み分析エラー: {e}")

        return suggestions

    def generate_benchmark_report(self, results: Dict[str, Any],
                                output_path: Optional[str] = None) -> str:
        """
        ベンチマークレポート生成

        Args:
            results: ベンチマーク結果
            output_path: 出力パス

        Returns:
            レポート文字列
        """
        report_lines = []

        # ヘッダー
        report_lines.extend([
            "=" * 80,
            "ENSEMBLE LEARNING ACCURACY BENCHMARK REPORT",
            f"Issue #462: 予測精度95%超達成のためのベンチマーク分析",
            "=" * 80,
            f"実行時刻: {pd.Timestamp.now()}",
            ""
        ])

        # エラーチェック
        if 'error' in results:
            report_lines.extend([
                "❌ ベンチマーク実行エラー",
                f"エラー内容: {results['error']}",
                ""
            ])
            return "\n".join(report_lines)

        # サマリー情報
        cv_summary = results['cross_validation_summary']
        avg_metrics = cv_summary['avg_metrics']

        report_lines.extend([
            "📊 BENCHMARK SUMMARY",
            "-" * 50,
            f"交差検証Fold数: {cv_summary['n_successful_folds']}/{cv_summary['n_total_folds']}",
            f"成功率: {cv_summary['success_rate']:.1%}",
            ""
        ])

        # 精度メトリクス
        report_lines.extend([
            "🎯 ACCURACY METRICS",
            "-" * 50
        ])

        key_metrics = ['accuracy_percentage', 'r2_score', 'hit_rate', 'rmse', 'mae']
        for metric in key_metrics:
            if metric in avg_metrics:
                m = avg_metrics[metric]
                report_lines.append(f"{metric.upper():20s}: {m['mean']:6.3f} ± {m['std']:5.3f} "
                                  f"[{m['min']:6.3f}, {m['max']:6.3f}]")

        report_lines.append("")

        # 95%達成評価
        current_accuracy = avg_metrics['accuracy_percentage']['mean']
        target_accuracy = 95.0
        gap = target_accuracy - current_accuracy

        report_lines.extend([
            "🎯 95% ACCURACY TARGET ANALYSIS",
            "-" * 50,
            f"現在の精度:     {current_accuracy:6.2f}%",
            f"目標精度:       {target_accuracy:6.2f}%",
            f"必要改善量:     {gap:6.2f}%",
            f"達成度:         {(current_accuracy/target_accuracy)*100:6.1f}%",
            ""
        ])

        # 達成可能性評価
        if gap <= 1:
            status = "✅ ほぼ達成 - 微調整で95%達成可能"
        elif gap <= 3:
            status = "🟡 有望 - 中程度の改善で95%達成可能"
        elif gap <= 8:
            status = "🟠 課題 - 大幅改善が必要だが達成可能"
        else:
            status = "🔴 困難 - 根本的なアプローチ変更が必要"

        report_lines.extend([
            f"達成可能性評価: {status}",
            ""
        ])

        # 改善提案
        if 'improvement_suggestions' in results:
            report_lines.extend([
                "💡 IMPROVEMENT SUGGESTIONS",
                "-" * 50
            ])
            for suggestion in results['improvement_suggestions']:
                report_lines.append(suggestion)
            report_lines.append("")

        # 個別Fold詳細
        if self.config.generate_detailed_report:
            report_lines.extend([
                "📋 DETAILED FOLD RESULTS",
                "-" * 50
            ])

            for fold_result in cv_summary['individual_fold_results']:
                if 'error' not in fold_result:
                    metrics = fold_result['metrics']
                    report_lines.extend([
                        f"Fold {fold_result['fold']}:",
                        f"  Accuracy: {metrics['accuracy_percentage']:6.2f}%",
                        f"  Hit Rate: {metrics['hit_rate']:6.3f}",
                        f"  R² Score: {metrics['r2_score']:6.3f}",
                        f"  Training Time: {fold_result['training_time']:6.1f}s",
                        ""
                    ])

        # フッター
        report_lines.extend([
            "=" * 80,
            "📝 RECOMMENDATIONS FOR 95% ACCURACY",
            "-" * 50,
            "1. ハイパーパラメータ最適化強化",
            "2. 高度な特徴量エンジニアリング導入",
            "3. より強力な基底モデル追加",
            "4. アンサンブル手法の改良",
            "5. データ品質向上とノイズ除去",
            "=" * 80
        ])

        report_text = "\n".join(report_lines)

        # ファイル出力
        if output_path:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"ベンチマークレポート保存: {output_path}")
            except Exception as e:
                logger.error(f"レポート保存エラー: {e}")

        return report_text

    def run_full_benchmark(self, data_size: int = 3000,
                         ensemble_configs: Optional[List[EnsembleConfig]] = None,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        フルベンチマーク実行

        Args:
            data_size: データサイズ
            ensemble_configs: テストするアンサンブル設定のリスト
            output_dir: 出力ディレクトリ

        Returns:
            統合ベンチマーク結果
        """
        logger.info(f"フルベンチマーク開始: データサイズ={data_size}")

        # デフォルト設定
        if ensemble_configs is None:
            ensemble_configs = [
                # 基本設定
                EnsembleConfig(
                    use_lstm_transformer=False,  # 高速化のため無効
                    use_random_forest=True,
                    use_gradient_boosting=True,
                    use_svr=True,
                    enable_stacking=False,
                    enable_dynamic_weighting=False
                ),
                # 高精度設定
                EnsembleConfig(
                    use_lstm_transformer=False,  # 高速化のため無効
                    use_random_forest=True,
                    use_gradient_boosting=True,
                    use_svr=True,
                    enable_stacking=True,
                    enable_dynamic_weighting=True
                )
            ]

        # 合成データ生成
        X, y, feature_names = self.generate_synthetic_stock_data(n_samples=data_size)

        # 各設定でベンチマーク実行
        all_results = {}
        best_config = None
        best_accuracy = 0

        for i, config in enumerate(ensemble_configs):
            config_name = f"config_{i+1}"
            logger.info(f"設定 {config_name} ベンチマーク開始...")

            try:
                results = self.run_cross_validation_benchmark(X, y, feature_names, config)

                if 'cross_validation_summary' in results:
                    accuracy = results['cross_validation_summary']['avg_metrics']['accuracy_percentage']['mean']
                    logger.info(f"設定 {config_name} 完了: 精度={accuracy:.2f}%")

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_config = config_name
                else:
                    logger.warning(f"設定 {config_name} 失敗")

                all_results[config_name] = results

            except Exception as e:
                logger.error(f"設定 {config_name} エラー: {e}")
                all_results[config_name] = {'error': str(e)}

        # 統合結果作成
        full_benchmark_result = {
            'individual_config_results': all_results,
            'best_configuration': best_config,
            'best_accuracy': best_accuracy,
            'data_info': {
                'n_samples': data_size,
                'n_features': len(feature_names),
                'feature_names': feature_names
            },
            'benchmark_config': self.config.__dict__,
            'execution_info': {
                'timestamp': time.time(),
                'n_configs_tested': len(ensemble_configs)
            }
        }

        # 統合レポート生成
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 各設定のレポート
            for config_name, results in all_results.items():
                if 'error' not in results:
                    report_path = output_dir / f"benchmark_report_{config_name}.txt"
                    self.generate_benchmark_report(results, str(report_path))

            # 統合レポート
            summary_report = self._generate_summary_report(full_benchmark_result)
            with open(output_dir / "benchmark_summary.txt", 'w', encoding='utf-8') as f:
                f.write(summary_report)

        logger.info(f"フルベンチマーク完了: 最高精度={best_accuracy:.2f}% (設定: {best_config})")

        return full_benchmark_result

    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """統合レポート生成"""
        lines = [
            "=" * 80,
            "ENSEMBLE LEARNING FULL BENCHMARK SUMMARY",
            f"Issue #462: 95%精度達成のための包括的ベンチマーク",
            "=" * 80,
            f"実行時刻: {pd.Timestamp.now()}",
            f"テスト設定数: {results['execution_info']['n_configs_tested']}",
            "",
            "🏆 BEST PERFORMANCE",
            "-" * 50,
            f"最高精度: {results['best_accuracy']:.2f}%",
            f"最適設定: {results['best_configuration']}",
            "",
            "📈 CONFIGURATION COMPARISON",
            "-" * 50
        ]

        # 各設定の比較
        for config_name, config_results in results['individual_config_results'].items():
            if 'error' not in config_results and 'cross_validation_summary' in config_results:
                cv_summary = config_results['cross_validation_summary']
                accuracy = cv_summary['avg_metrics']['accuracy_percentage']['mean']
                accuracy_std = cv_summary['avg_metrics']['accuracy_percentage']['std']
                hit_rate = cv_summary['avg_metrics']['hit_rate']['mean']

                lines.extend([
                    f"{config_name.upper()}:",
                    f"  精度: {accuracy:6.2f}% ± {accuracy_std:4.2f}%",
                    f"  Hit Rate: {hit_rate:6.3f}",
                    f"  成功率: {cv_summary['success_rate']:.1%}",
                    ""
                ])

        # 95%達成のための最終提案
        best_accuracy = results['best_accuracy']
        gap_to_95 = 95.0 - best_accuracy

        lines.extend([
            "🎯 95% ACCURACY ACHIEVEMENT PLAN",
            "-" * 50,
            f"現在の最高精度: {best_accuracy:6.2f}%",
            f"目標との差:     {gap_to_95:6.2f}%",
            ""
        ])

        if gap_to_95 <= 0:
            lines.append("🎉 95%精度達成済み！")
        elif gap_to_95 <= 2:
            lines.extend([
                "✅ 95%達成まであとわずか！",
                "推奨アクション:",
                "1. ハイパーパラメータの細かい調整",
                "2. 特徴量の微調整",
                "3. データクリーニングの改善"
            ])
        elif gap_to_95 <= 5:
            lines.extend([
                "🟡 中程度の改善で95%達成可能",
                "推奨アクション:",
                "1. より強力なベースモデルの追加",
                "2. アンサンブル手法の改良",
                "3. 特徴量エンジニアリングの強化"
            ])
        else:
            lines.extend([
                "🔴 大幅な改善が必要",
                "推奨アクション:",
                "1. 根本的なアーキテクチャ変更",
                "2. 外部データソースの統合",
                "3. より高度な深層学習モデルの導入"
            ])

        lines.extend([
            "",
            "=" * 80,
            "詳細な改善提案は各設定の個別レポートを参照してください。",
            "=" * 80
        ])

        return "\n".join(lines)


def run_accuracy_benchmark_demo():
    """ベンチマークデモ実行"""
    print("=" * 60)
    print("Issue #462: アンサンブル学習精度ベンチマーク")
    print("=" * 60)

    try:
        # ベンチマーク設定
        benchmark_config = BenchmarkConfig(
            n_splits=3,  # デモ用に高速化
            generate_detailed_report=True
        )

        benchmark = AccuracyBenchmark(benchmark_config)

        print("ベンチマーク実行中...")

        # フルベンチマーク実行
        results = benchmark.run_full_benchmark(
            data_size=1000,  # デモ用に小さなサイズ
            output_dir="benchmark_reports"
        )

        # 結果表示
        best_accuracy = results['best_accuracy']
        best_config = results['best_configuration']

        print(f"\nベンチマーク完了!")
        print(f"最高精度: {best_accuracy:.2f}%")
        print(f"最適設定: {best_config}")
        print(f"95%達成まで: {95.0 - best_accuracy:.2f}%の改善が必要")

        if best_accuracy >= 95.0:
            print("95%精度達成！")
        elif best_accuracy >= 90.0:
            print("90%超達成 - もう少しで95%！")
        else:
            print("さらなる改善が必要")

        print(f"\n詳細レポート: benchmark_reports/ ディレクトリを確認してください")

        return True

    except Exception as e:
        print(f"ベンチマークエラー: {e}")
        return False


if __name__ == "__main__":
    success = run_accuracy_benchmark_demo()
    exit(0 if success else 1)