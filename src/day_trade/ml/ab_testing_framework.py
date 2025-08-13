#!/usr/bin/env python3
"""
ML Model A/B Testing Framework

Issue #733対応: 機械学習モデルのA/Bテスト実装
データ駆動型意思決定によるモデル最適化とリスク軽減

主要機能:
- 実験グループ管理（A/Bグループ定義・データ配分）
- トラフィック分割（銘柄・時間・セグメント別）
- 統計的有意性検定（t検定・カイ二乗検定）
- レポート・視覚化
- ロールアウト・ロールバック
"""

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class ExperimentStatus(Enum):
    """実験ステータス"""
    DRAFT = "draft"  # 下書き
    ACTIVE = "active"  # 実行中
    PAUSED = "paused"  # 一時停止
    COMPLETED = "completed"  # 完了
    CANCELLED = "cancelled"  # キャンセル


class TrafficSplitStrategy(Enum):
    """トラフィック分割戦略"""
    RANDOM = "random"  # ランダム分割
    HASH_BASED = "hash_based"  # ハッシュベース分割
    TIME_BASED = "time_based"  # 時間ベース分割
    SYMBOL_BASED = "symbol_based"  # 銘柄ベース分割
    SEGMENT_BASED = "segment_based"  # セグメントベース分割


@dataclass
class ExperimentGroup:
    """実験グループ設定"""
    group_id: str
    name: str
    description: str
    traffic_percentage: float  # トラフィック配分割合（0.0-1.0）
    model_config: Dict[str, Any]  # モデル設定
    feature_config: Dict[str, Any] = field(default_factory=dict)  # 特徴量設定
    is_control: bool = False  # コントロールグループかどうか
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentConfig:
    """A/Bテスト実験設定"""
    experiment_id: str
    name: str
    description: str
    groups: List[ExperimentGroup]
    traffic_split_strategy: TrafficSplitStrategy = TrafficSplitStrategy.RANDOM
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    success_metrics: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall"])
    minimum_sample_size: int = 1000  # 最小サンプルサイズ
    significance_level: float = 0.05  # 有意水準
    power: float = 0.8  # 検定力
    status: ExperimentStatus = ExperimentStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """実験結果データ"""
    result_id: str
    experiment_id: str
    group_id: str
    symbol: str
    timestamp: datetime
    prediction: float
    actual_value: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalTestResult:
    """統計検定結果"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: Optional[float] = None
    power: Optional[float] = None
    sample_size_a: int = 0
    sample_size_b: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """実験レポート"""
    experiment_id: str
    experiment_name: str
    report_generated_at: datetime
    duration_days: int
    total_samples: int
    group_metrics: Dict[str, Dict[str, float]]  # グループ別メトリクス
    statistical_tests: List[StatisticalTestResult]
    recommendations: List[str]  # 推奨事項
    summary: str
    confidence_level: float
    winner_group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrafficSplitter:
    """トラフィック分割システム"""

    def __init__(self, experiment_config: ExperimentConfig):
        """初期化"""
        self.config = experiment_config
        self.strategy = experiment_config.traffic_split_strategy

        # トラフィック配分の検証
        total_percentage = sum(group.traffic_percentage for group in self.config.groups)
        if abs(total_percentage - 1.0) > 0.001:
            raise ValueError(f"トラフィック配分の合計が100%になりません: {total_percentage*100:.1f}%")

    def assign_to_group(self, identifier: str, context: Dict[str, Any] = None) -> str:
        """
        識別子に基づいてグループを割り当て

        Args:
            identifier: 分割に使用する識別子（銘柄コード、ユーザーID等）
            context: 追加のコンテキスト情報

        Returns:
            割り当てられたグループID
        """
        context = context or {}

        if self.strategy == TrafficSplitStrategy.RANDOM:
            return self._random_assignment()
        elif self.strategy == TrafficSplitStrategy.HASH_BASED:
            return self._hash_based_assignment(identifier)
        elif self.strategy == TrafficSplitStrategy.TIME_BASED:
            return self._time_based_assignment(context.get('timestamp', datetime.now()))
        elif self.strategy == TrafficSplitStrategy.SYMBOL_BASED:
            return self._symbol_based_assignment(identifier)
        else:
            return self._random_assignment()

    def _random_assignment(self) -> str:
        """ランダムグループ割り当て"""
        rand_val = np.random.random()
        cumulative = 0.0

        for group in self.config.groups:
            cumulative += group.traffic_percentage
            if rand_val <= cumulative:
                return group.group_id

        # フォールバック：最後のグループ
        return self.config.groups[-1].group_id

    def _hash_based_assignment(self, identifier: str) -> str:
        """ハッシュベースグループ割り当て"""
        # 一貫した分割のためのハッシュベース実装
        hash_input = f"{self.config.experiment_id}_{identifier}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0  # 0.0-1.0に正規化

        cumulative = 0.0
        for group in self.config.groups:
            cumulative += group.traffic_percentage
            if normalized_hash <= cumulative:
                return group.group_id

        return self.config.groups[-1].group_id

    def _time_based_assignment(self, timestamp: datetime) -> str:
        """時間ベースグループ割り当て"""
        # 時間に基づく周期的な分割
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()

        # 簡単な時間ベース分割（カスタマイズ可能）
        time_hash = (hour_of_day + day_of_week * 24) % len(self.config.groups)
        return self.config.groups[time_hash].group_id

    def _symbol_based_assignment(self, symbol: str) -> str:
        """銘柄ベースグループ割り当て"""
        # 銘柄コードに基づく一貫した分割
        if symbol.isdigit():
            symbol_num = int(symbol)
            group_index = symbol_num % len(self.config.groups)
            return self.config.groups[group_index].group_id
        else:
            # 非数値銘柄の場合はハッシュベース
            return self._hash_based_assignment(symbol)


class StatisticalAnalyzer:
    """統計分析エンジン"""

    def __init__(self, significance_level: float = 0.05):
        """初期化"""
        self.significance_level = significance_level

    def perform_t_test(self, group_a_data: np.ndarray, group_b_data: np.ndarray,
                      alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        独立二標本t検定の実行

        Args:
            group_a_data: グループAのデータ
            group_b_data: グループBのデータ
            alternative: 対立仮説 ('two-sided', 'less', 'greater')

        Returns:
            統計検定結果
        """
        # t検定の実行
        statistic, p_value = stats.ttest_ind(group_a_data, group_b_data, alternative=alternative)

        # 効果量（Cohen's d）の計算
        pooled_std = np.sqrt(((len(group_a_data) - 1) * np.var(group_a_data, ddof=1) +
                            (len(group_b_data) - 1) * np.var(group_b_data, ddof=1)) /
                           (len(group_a_data) + len(group_b_data) - 2))
        effect_size = (np.mean(group_a_data) - np.mean(group_b_data)) / pooled_std if pooled_std > 0 else 0

        # 信頼区間の計算
        ci_low, ci_high = self._calculate_confidence_interval(group_a_data, group_b_data)

        return StatisticalTestResult(
            test_name="independent_t_test",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=p_value < self.significance_level,
            confidence_interval=(ci_low, ci_high),
            effect_size=float(effect_size),
            sample_size_a=len(group_a_data),
            sample_size_b=len(group_b_data),
            metadata={
                'alternative': alternative,
                'group_a_mean': float(np.mean(group_a_data)),
                'group_b_mean': float(np.mean(group_b_data)),
                'group_a_std': float(np.std(group_a_data, ddof=1)),
                'group_b_std': float(np.std(group_b_data, ddof=1))
            }
        )

    def perform_chi_squared_test(self, observed_a: np.ndarray, observed_b: np.ndarray) -> StatisticalTestResult:
        """
        カイ二乗検定の実行

        Args:
            observed_a: グループAの観測度数
            observed_b: グループBの観測度数

        Returns:
            統計検定結果
        """
        # 分割表の作成
        contingency_table = np.array([observed_a, observed_b])

        # カイ二乗検定の実行
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # Cramer's V (効果量) の計算
        n = np.sum(contingency_table)
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

        return StatisticalTestResult(
            test_name="chi_squared_test",
            statistic=float(chi2_stat),
            p_value=float(p_value),
            is_significant=p_value < self.significance_level,
            confidence_interval=(0.0, 1.0),  # カイ二乗では信頼区間は通常計算しない
            effect_size=float(cramers_v),
            sample_size_a=int(np.sum(observed_a)),
            sample_size_b=int(np.sum(observed_b)),
            metadata={
                'degrees_of_freedom': int(dof),
                'expected_frequencies': expected.tolist()
            }
        )

    def _calculate_confidence_interval(self, group_a: np.ndarray, group_b: np.ndarray,
                                     confidence: float = 0.95) -> Tuple[float, float]:
        """平均の差の信頼区間を計算"""
        mean_diff = np.mean(group_a) - np.mean(group_b)

        # プールされた標準誤差の計算
        se_pooled = np.sqrt(np.var(group_a, ddof=1)/len(group_a) + np.var(group_b, ddof=1)/len(group_b))

        # 自由度
        df = len(group_a) + len(group_b) - 2

        # t値
        t_critical = stats.t.ppf((1 + confidence) / 2, df)

        # 信頼区間
        margin_error = t_critical * se_pooled

        return (mean_diff - margin_error, mean_diff + margin_error)


class ABTestingFramework:
    """A/Bテストフレームワーク メインクラス"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        初期化

        Args:
            storage_path: 実験データ保存パス
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/ab_testing")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.traffic_splitters: Dict[str, TrafficSplitter] = {}
        self.results_storage: Dict[str, List[ExperimentResult]] = {}
        self.analyzer = StatisticalAnalyzer()

        logger.info(f"A/Bテストフレームワーク初期化完了: {self.storage_path}")

    async def create_experiment(self, config: ExperimentConfig) -> bool:
        """
        新しい実験を作成

        Args:
            config: 実験設定

        Returns:
            作成成功かどうか
        """
        try:
            # 設定検証
            if not self._validate_experiment_config(config):
                return False

            # 実験IDの重複チェック
            if config.experiment_id in self.active_experiments:
                logger.warning(f"実験ID {config.experiment_id} は既に存在します")
                return False

            # トラフィック分割器の作成
            traffic_splitter = TrafficSplitter(config)

            # 実験の登録
            self.active_experiments[config.experiment_id] = config
            self.traffic_splitters[config.experiment_id] = traffic_splitter
            self.results_storage[config.experiment_id] = []

            # 設定の永続化
            await self._save_experiment_config(config)

            logger.info(f"実験 {config.experiment_id} ({config.name}) を作成しました")
            return True

        except Exception as e:
            logger.error(f"実験作成エラー: {e}")
            return False

    async def assign_to_experiment(self, experiment_id: str, identifier: str,
                                 context: Dict[str, Any] = None) -> Optional[str]:
        """
        実験グループへの割り当て

        Args:
            experiment_id: 実験ID
            identifier: 分割識別子
            context: コンテキスト

        Returns:
            割り当てられたグループID
        """
        if experiment_id not in self.active_experiments:
            logger.warning(f"実験 {experiment_id} が見つかりません")
            return None

        experiment = self.active_experiments[experiment_id]

        # 実験が有効かチェック
        if experiment.status != ExperimentStatus.ACTIVE:
            return None

        # 実験期間内かチェック
        now = datetime.now()
        if experiment.end_date and now > experiment.end_date:
            return None

        traffic_splitter = self.traffic_splitters[experiment_id]
        group_id = traffic_splitter.assign_to_group(identifier, context)

        logger.debug(f"実験 {experiment_id}: {identifier} → グループ {group_id}")
        return group_id

    async def record_result(self, result: ExperimentResult) -> bool:
        """
        実験結果の記録

        Args:
            result: 実験結果

        Returns:
            記録成功かどうか
        """
        try:
            if result.experiment_id not in self.active_experiments:
                logger.warning(f"実験 {result.experiment_id} が見つかりません")
                return False

            # 結果の保存
            self.results_storage[result.experiment_id].append(result)

            # 定期的な永続化（1000件ごと）
            if len(self.results_storage[result.experiment_id]) % 1000 == 0:
                await self._save_experiment_results(result.experiment_id)

            return True

        except Exception as e:
            logger.error(f"結果記録エラー: {e}")
            return False

    async def generate_report(self, experiment_id: str) -> Optional[ExperimentReport]:
        """
        実験レポートの生成

        Args:
            experiment_id: 実験ID

        Returns:
            実験レポート
        """
        try:
            if experiment_id not in self.active_experiments:
                logger.warning(f"実験 {experiment_id} が見つかりません")
                return None

            experiment = self.active_experiments[experiment_id]
            results = self.results_storage.get(experiment_id, [])

            if not results:
                logger.warning(f"実験 {experiment_id} に結果データがありません")
                return None

            # グループ別メトリクス計算
            group_metrics = self._calculate_group_metrics(results)

            # 統計検定の実行
            statistical_tests = await self._perform_statistical_tests(results, experiment)

            # レポート生成
            report = ExperimentReport(
                experiment_id=experiment_id,
                experiment_name=experiment.name,
                report_generated_at=datetime.now(),
                duration_days=(datetime.now() - experiment.start_date).days,
                total_samples=len(results),
                group_metrics=group_metrics,
                statistical_tests=statistical_tests,
                recommendations=self._generate_recommendations(statistical_tests, group_metrics),
                summary=self._generate_summary(statistical_tests, group_metrics),
                confidence_level=1 - experiment.significance_level,
                winner_group_id=self._determine_winner(statistical_tests, group_metrics)
            )

            # レポートの永続化
            await self._save_report(report)

            logger.info(f"実験 {experiment_id} のレポートを生成しました")
            return report

        except Exception as e:
            logger.error(f"レポート生成エラー: {e}")
            return None

    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """実験設定の検証"""
        # グループ数のチェック
        if len(config.groups) < 2:
            logger.error("実験には最低2つのグループが必要です")
            return False

        # トラフィック配分の検証
        total_percentage = sum(group.traffic_percentage for group in config.groups)
        if abs(total_percentage - 1.0) > 0.001:
            logger.error(f"トラフィック配分の合計が100%になりません: {total_percentage*100:.1f}%")
            return False

        # グループIDの重複チェック
        group_ids = [group.group_id for group in config.groups]
        if len(group_ids) != len(set(group_ids)):
            logger.error("重複するグループIDが存在します")
            return False

        # コントロールグループのチェック
        control_groups = [group for group in config.groups if group.is_control]
        if len(control_groups) != 1:
            logger.error("コントロールグループは1つのみ指定してください")
            return False

        return True

    def _calculate_group_metrics(self, results: List[ExperimentResult]) -> Dict[str, Dict[str, float]]:
        """グループ別メトリクス計算"""
        group_data = {}

        # グループ別にデータを分類
        for result in results:
            if result.group_id not in group_data:
                group_data[result.group_id] = {
                    'predictions': [],
                    'actual_values': [],
                    'latencies': [],
                    'count': 0
                }

            group_data[result.group_id]['predictions'].append(result.prediction)
            if result.actual_value is not None:
                group_data[result.group_id]['actual_values'].append(result.actual_value)
            group_data[result.group_id]['latencies'].append(result.latency_ms)
            group_data[result.group_id]['count'] += 1

        # メトリクス計算
        metrics = {}
        for group_id, data in group_data.items():
            predictions = np.array(data['predictions'])
            actual_values = np.array(data['actual_values'])
            latencies = np.array(data['latencies'])

            group_metrics = {
                'sample_size': data['count'],
                'avg_prediction': float(np.mean(predictions)),
                'prediction_std': float(np.std(predictions)),
                'avg_latency_ms': float(np.mean(latencies)),
                'latency_std_ms': float(np.std(latencies))
            }

            # 実測値がある場合のみ精度計算
            if len(actual_values) > 0:
                mae = np.mean(np.abs(predictions[:len(actual_values)] - actual_values))
                mse = np.mean((predictions[:len(actual_values)] - actual_values) ** 2)
                rmse = np.sqrt(mse)

                group_metrics.update({
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'accuracy_samples': len(actual_values)
                })

            metrics[group_id] = group_metrics

        return metrics

    async def _perform_statistical_tests(self, results: List[ExperimentResult],
                                       experiment: ExperimentConfig) -> List[StatisticalTestResult]:
        """統計検定の実行"""
        test_results = []

        # グループ別にデータを分類
        group_data = {}
        for result in results:
            if result.group_id not in group_data:
                group_data[result.group_id] = []
            group_data[result.group_id].append(result)

        # コントロールグループを特定
        control_group_id = None
        for group in experiment.groups:
            if group.is_control:
                control_group_id = group.group_id
                break

        if not control_group_id or control_group_id not in group_data:
            logger.warning("コントロールグループのデータが見つかりません")
            return test_results

        control_data = group_data[control_group_id]
        control_predictions = np.array([r.prediction for r in control_data])
        control_latencies = np.array([r.latency_ms for r in control_data])

        # 各テストグループとコントロールの比較
        for group_id, test_data in group_data.items():
            if group_id == control_group_id:
                continue

            test_predictions = np.array([r.prediction for r in test_data])
            test_latencies = np.array([r.latency_ms for r in test_data])

            # 予測値のt検定
            if len(control_predictions) >= 30 and len(test_predictions) >= 30:
                prediction_test = self.analyzer.perform_t_test(
                    control_predictions, test_predictions
                )
                prediction_test.metadata['metric'] = 'predictions'
                prediction_test.metadata['groups'] = f"{control_group_id}_vs_{group_id}"
                test_results.append(prediction_test)

            # レイテンシのt検定
            if len(control_latencies) >= 30 and len(test_latencies) >= 30:
                latency_test = self.analyzer.perform_t_test(
                    control_latencies, test_latencies
                )
                latency_test.metadata['metric'] = 'latency_ms'
                latency_test.metadata['groups'] = f"{control_group_id}_vs_{group_id}"
                test_results.append(latency_test)

        return test_results

    def _generate_recommendations(self, tests: List[StatisticalTestResult],
                                metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []

        significant_tests = [test for test in tests if test.is_significant]

        if not significant_tests:
            recommendations.append("統計的に有意な差は検出されませんでした。")
            recommendations.append("サンプルサイズを増やすか、実験期間を延長することを検討してください。")
        else:
            for test in significant_tests:
                metric = test.metadata.get('metric', '不明')
                groups = test.metadata.get('groups', '不明')

                if test.effect_size and abs(test.effect_size) > 0.5:  # 大きな効果量
                    recommendations.append(
                        f"{metric}において{groups}間で大きな効果量({test.effect_size:.3f})の"
                        f"統計的有意差が検出されました（p={test.p_value:.4f}）"
                    )
                elif test.effect_size and abs(test.effect_size) > 0.2:  # 中程度の効果量
                    recommendations.append(
                        f"{metric}において{groups}間で中程度の効果量({test.effect_size:.3f})の"
                        f"統計的有意差が検出されました（p={test.p_value:.4f}）"
                    )

        # パフォーマンス比較
        if len(metrics) >= 2:
            group_ids = list(metrics.keys())
            best_latency_group = min(group_ids, key=lambda g: metrics[g].get('avg_latency_ms', float('inf')))
            recommendations.append(f"レイテンシが最も低いのは {best_latency_group} グループです。")

        return recommendations

    def _generate_summary(self, tests: List[StatisticalTestResult],
                        metrics: Dict[str, Dict[str, float]]) -> str:
        """サマリー生成"""
        total_samples = sum(m.get('sample_size', 0) for m in metrics.values())
        significant_tests = len([test for test in tests if test.is_significant])

        summary = f"実験サマリー: "
        summary += f"総サンプル数 {total_samples}, "
        summary += f"統計検定 {len(tests)} 件実行, "
        summary += f"有意差検出 {significant_tests} 件"

        if significant_tests > 0:
            summary += f" - 統計的に有意な差が検出されました。"
        else:
            summary += f" - 統計的に有意な差は検出されませんでした。"

        return summary

    def _determine_winner(self, tests: List[StatisticalTestResult],
                        metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
        """勝者グループの決定"""
        # 簡単な勝者決定ロジック（カスタマイズ可能）
        significant_tests = [test for test in tests if test.is_significant]

        if not significant_tests:
            return None

        # 効果量が最大のテストを基準に勝者を決定
        best_test = max(significant_tests, key=lambda t: abs(t.effect_size) if t.effect_size else 0)

        if best_test.metadata.get('groups'):
            groups_str = best_test.metadata['groups']
            # "groupA_vs_groupB" から勝者を抽出（簡易実装）
            if best_test.effect_size > 0:
                return groups_str.split('_vs_')[0]  # 最初のグループが勝者
            else:
                return groups_str.split('_vs_')[1]  # 2番目のグループが勝者

        return None

    async def _save_experiment_config(self, config: ExperimentConfig):
        """実験設定の永続化"""
        config_path = self.storage_path / f"experiment_{config.experiment_id}_config.json"

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)

    async def _save_experiment_results(self, experiment_id: str):
        """実験結果の永続化"""
        results_path = self.storage_path / f"experiment_{experiment_id}_results.json"
        results = self.results_storage.get(experiment_id, [])

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in results], f, ensure_ascii=False, indent=2, default=str)

    async def _save_report(self, report: ExperimentReport):
        """レポートの永続化"""
        report_path = self.storage_path / f"experiment_{report.experiment_id}_report.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)


# ユーティリティ関数
def create_simple_ab_experiment(experiment_name: str,
                               control_config: Dict[str, Any],
                               test_config: Dict[str, Any],
                               traffic_split: float = 0.5) -> ExperimentConfig:
    """
    シンプルなA/B実験設定の作成ヘルパー

    Args:
        experiment_name: 実験名
        control_config: コントロールグループ設定
        test_config: テストグループ設定
        traffic_split: テストグループのトラフィック割合（0.0-1.0）

    Returns:
        実験設定
    """
    experiment_id = f"ab_test_{uuid.uuid4().hex[:8]}"

    control_group = ExperimentGroup(
        group_id="control",
        name="Control Group",
        description="コントロールグループ",
        traffic_percentage=1.0 - traffic_split,
        model_config=control_config,
        is_control=True
    )

    test_group = ExperimentGroup(
        group_id="test",
        name="Test Group",
        description="テストグループ",
        traffic_percentage=traffic_split,
        model_config=test_config,
        is_control=False
    )

    return ExperimentConfig(
        experiment_id=experiment_id,
        name=experiment_name,
        description=f"A/B Test: {experiment_name}",
        groups=[control_group, test_group],
        traffic_split_strategy=TrafficSplitStrategy.HASH_BASED
    )


async def main():
    """デモンストレーション"""
    logger.info("A/Bテストフレームワーク デモンストレーション開始")

    # フレームワークの初期化
    framework = ABTestingFramework("data/ab_testing_demo")

    # シンプルなA/B実験の作成
    experiment_config = create_simple_ab_experiment(
        experiment_name="RandomForest vs GradientBoosting Comparison",
        control_config={"model_type": "RandomForest", "n_estimators": 100},
        test_config={"model_type": "GradientBoosting", "n_estimators": 100},
        traffic_split=0.5
    )

    # 実験の作成と開始
    success = await framework.create_experiment(experiment_config)
    if success:
        experiment_config.status = ExperimentStatus.ACTIVE
        logger.info(f"実験 {experiment_config.experiment_id} を開始しました")

    logger.info("A/Bテストフレームワーク デモンストレーション完了")


if __name__ == "__main__":
    asyncio.run(main())