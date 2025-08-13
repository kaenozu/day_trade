#!/usr/bin/env python3
"""
ML Experimentation Platform - 統合実験プラットフォーム

Issue #733対応: A/Bテストとデプロイメント管理の統合システム
機械学習モデルのライフサイクル全体をサポート

主要機能:
- 実験設計・実行（A/Bテスト）
- モデルデプロイメント管理（ロールアウト/ロールバック）
- 実験結果とデプロイメントメトリクスの統合分析
- 自動化されたモデル選択・配信
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from .ab_testing_framework import (
    ABTestingFramework, ExperimentConfig, ExperimentResult,
    ExperimentReport, create_simple_ab_experiment,
    ExperimentStatus, TrafficSplitStrategy
)
from .model_deployment_manager import (
    ModelDeploymentManager, DeploymentConfig, ModelVersion,
    DeploymentMetrics, create_canary_deployment,
    DeploymentStatus, DeploymentStrategy
)
from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


@dataclass
class ExperimentDeploymentPair:
    """実験-デプロイメントペア"""
    pair_id: str
    experiment_id: str
    deployment_id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # "active", "completed", "failed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceComparison:
    """モデルパフォーマンス比較結果"""
    comparison_id: str
    experiment_id: str
    control_group_id: str
    test_group_id: str
    metric_comparisons: Dict[str, Dict[str, float]]  # メトリック比較
    statistical_significance: Dict[str, bool]  # 統計的有意性
    recommendation: str  # 推奨事項
    confidence_score: float  # 信頼度
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutoDeploymentDecision:
    """自動デプロイメント判定結果"""
    decision_id: str
    experiment_id: str
    winning_model_version_id: str
    deploy_recommended: bool
    deployment_strategy: DeploymentStrategy
    confidence_level: float
    reasons: List[str]
    risk_assessment: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)


class ModelPerformanceAnalyzer:
    """モデルパフォーマンス分析"""

    def __init__(self):
        """初期化"""
        self.comparison_threshold = {
            'accuracy_improvement': 0.01,  # 1%以上の精度改善
            'latency_degradation': 0.1,    # 10%以内のレイテンシ劣化
            'error_rate_increase': 0.005,  # 0.5%以内のエラー率増加
        }

    def analyze_experiment_results(self, experiment_report: ExperimentReport) -> ModelPerformanceComparison:
        """
        実験結果の詳細分析

        Args:
            experiment_report: 実験レポート

        Returns:
            パフォーマンス比較結果
        """
        comparison_id = f"perf_comp_{uuid.uuid4().hex[:8]}"

        # コントロールグループとテストグループの特定
        control_group_id = None
        test_group_id = None

        for group_id, metrics in experiment_report.group_metrics.items():
            if "control" in group_id.lower():
                control_group_id = group_id
            else:
                test_group_id = group_id

        if not control_group_id or not test_group_id:
            # 最初の2つのグループを使用
            group_ids = list(experiment_report.group_metrics.keys())
            control_group_id = group_ids[0] if len(group_ids) > 0 else "unknown"
            test_group_id = group_ids[1] if len(group_ids) > 1 else "unknown"

        # メトリクス比較の計算
        metric_comparisons = self._calculate_metric_comparisons(
            experiment_report.group_metrics, control_group_id, test_group_id
        )

        # 統計的有意性の判定
        statistical_significance = {}
        for test in experiment_report.statistical_tests:
            metric = test.metadata.get('metric', 'unknown')
            statistical_significance[metric] = test.is_significant

        # 推奨事項の生成
        recommendation = self._generate_performance_recommendation(
            metric_comparisons, statistical_significance
        )

        # 信頼度スコアの計算
        confidence_score = self._calculate_confidence_score(
            experiment_report.statistical_tests, metric_comparisons
        )

        return ModelPerformanceComparison(
            comparison_id=comparison_id,
            experiment_id=experiment_report.experiment_id,
            control_group_id=control_group_id,
            test_group_id=test_group_id,
            metric_comparisons=metric_comparisons,
            statistical_significance=statistical_significance,
            recommendation=recommendation,
            confidence_score=confidence_score
        )

    def _calculate_metric_comparisons(self, group_metrics: Dict[str, Dict[str, float]],
                                    control_id: str, test_id: str) -> Dict[str, Dict[str, float]]:
        """メトリクス比較の計算"""
        comparisons = {}

        control_metrics = group_metrics.get(control_id, {})
        test_metrics = group_metrics.get(test_id, {})

        for metric_name in control_metrics.keys():
            if metric_name in test_metrics:
                control_value = control_metrics[metric_name]
                test_value = test_metrics[metric_name]

                if control_value != 0:
                    relative_change = (test_value - control_value) / control_value
                    absolute_change = test_value - control_value
                else:
                    relative_change = 0.0
                    absolute_change = test_value

                comparisons[metric_name] = {
                    'control_value': control_value,
                    'test_value': test_value,
                    'absolute_change': absolute_change,
                    'relative_change': relative_change,
                    'improvement_percentage': relative_change * 100
                }

        return comparisons

    def _generate_performance_recommendation(self, metric_comparisons: Dict[str, Dict[str, float]],
                                           statistical_significance: Dict[str, bool]) -> str:
        """パフォーマンス推奨事項の生成"""
        recommendations = []

        # 精度改善のチェック
        if 'accuracy' in metric_comparisons:
            accuracy_change = metric_comparisons['accuracy']['relative_change']
            is_significant = statistical_significance.get('accuracy', False)

            if accuracy_change > self.comparison_threshold['accuracy_improvement'] and is_significant:
                recommendations.append(f"精度が{accuracy_change*100:.2f}%向上（統計的有意）")
            elif accuracy_change > 0 and is_significant:
                recommendations.append(f"精度が{accuracy_change*100:.2f}%向上（統計的有意、小幅改善）")

        # レイテンシ悪化のチェック
        if 'avg_latency_ms' in metric_comparisons:
            latency_change = metric_comparisons['avg_latency_ms']['relative_change']

            if latency_change > self.comparison_threshold['latency_degradation']:
                recommendations.append(f"レイテンシが{latency_change*100:.1f}%悪化（要注意）")

        # エラー率増加のチェック
        if 'error_rate' in metric_comparisons:
            error_change = metric_comparisons['error_rate']['absolute_change']

            if error_change > self.comparison_threshold['error_rate_increase']:
                recommendations.append(f"エラー率が{error_change*100:.3f}%増加（要注意）")

        if not recommendations:
            recommendations.append("大きな性能変化は検出されませんでした")

        return "; ".join(recommendations)

    def _calculate_confidence_score(self, statistical_tests, metric_comparisons) -> float:
        """信頼度スコアの計算"""
        if not statistical_tests:
            return 0.5

        # 統計的有意なテストの割合と効果量を考慮
        significant_tests = [test for test in statistical_tests if test.is_significant]
        significance_ratio = len(significant_tests) / len(statistical_tests)

        # 効果量の平均
        effect_sizes = [abs(test.effect_size) for test in significant_tests if test.effect_size]
        avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0

        # 信頼度スコア計算（0.0-1.0）
        confidence_score = min(1.0, significance_ratio * 0.7 + min(avg_effect_size, 1.0) * 0.3)

        return float(confidence_score)


class AutoDeploymentDecisionEngine:
    """自動デプロイメント判定エンジン"""

    def __init__(self):
        """初期化"""
        self.deployment_criteria = {
            'min_confidence_score': 0.7,  # 最小信頼度
            'min_accuracy_improvement': 0.02,  # 最小精度改善（2%）
            'max_latency_degradation': 0.15,  # 最大レイテンシ劣化許容（15%）
            'max_error_rate_increase': 0.01,  # 最大エラー率増加許容（1%）
        }

        self.risk_factors = {
            'high_traffic': 0.3,  # 高トラフィック時のリスク
            'recent_deployments': 0.2,  # 最近のデプロイメントのリスク
            'complex_model': 0.1,  # 複雑なモデルのリスク
        }

    def make_deployment_decision(self, performance_comparison: ModelPerformanceComparison,
                               deployment_context: Dict[str, Any] = None) -> AutoDeploymentDecision:
        """
        自動デプロイメント判定

        Args:
            performance_comparison: パフォーマンス比較結果
            deployment_context: デプロイメントコンテキスト

        Returns:
            デプロイメント判定結果
        """
        deployment_context = deployment_context or {}
        decision_id = f"deploy_decision_{uuid.uuid4().hex[:8]}"

        # 基本的な判定ロジック
        should_deploy = True
        reasons = []
        confidence_level = performance_comparison.confidence_score

        # 信頼度チェック
        if confidence_level < self.deployment_criteria['min_confidence_score']:
            should_deploy = False
            reasons.append(f"信頼度不足 ({confidence_level:.3f} < {self.deployment_criteria['min_confidence_score']})")

        # 精度改善チェック
        accuracy_improvement = 0.0
        if 'accuracy' in performance_comparison.metric_comparisons:
            accuracy_improvement = performance_comparison.metric_comparisons['accuracy']['relative_change']

            if accuracy_improvement < self.deployment_criteria['min_accuracy_improvement']:
                should_deploy = False
                reasons.append(f"精度改善不足 ({accuracy_improvement*100:.2f}% < {self.deployment_criteria['min_accuracy_improvement']*100:.1f}%)")
            else:
                reasons.append(f"精度改善確認 (+{accuracy_improvement*100:.2f}%)")

        # レイテンシ悪化チェック
        if 'avg_latency_ms' in performance_comparison.metric_comparisons:
            latency_change = performance_comparison.metric_comparisons['avg_latency_ms']['relative_change']

            if latency_change > self.deployment_criteria['max_latency_degradation']:
                should_deploy = False
                reasons.append(f"レイテンシ悪化過大 ({latency_change*100:.1f}% > {self.deployment_criteria['max_latency_degradation']*100:.1f}%)")

        # エラー率増加チェック
        if 'error_rate' in performance_comparison.metric_comparisons:
            error_rate_change = performance_comparison.metric_comparisons['error_rate']['absolute_change']

            if error_rate_change > self.deployment_criteria['max_error_rate_increase']:
                should_deploy = False
                reasons.append(f"エラー率増加過大 ({error_rate_change*100:.3f}% > {self.deployment_criteria['max_error_rate_increase']*100:.3f}%)")

        # リスク評価
        risk_assessment = self._assess_deployment_risk(deployment_context)

        # デプロイメント戦略の決定
        deployment_strategy = self._choose_deployment_strategy(
            should_deploy, confidence_level, risk_assessment
        )

        # 最終判定（リスクが高い場合は慎重にする）
        if should_deploy and risk_assessment.get('overall_risk', 0.0) > 0.7:
            deployment_strategy = DeploymentStrategy.CANARY
            reasons.append("高リスクのためカナリアデプロイメントを推奨")

        if not reasons:
            reasons.append("全ての条件を満たしています")

        return AutoDeploymentDecision(
            decision_id=decision_id,
            experiment_id=performance_comparison.experiment_id,
            winning_model_version_id=performance_comparison.test_group_id,
            deploy_recommended=should_deploy,
            deployment_strategy=deployment_strategy,
            confidence_level=confidence_level,
            reasons=reasons,
            risk_assessment=risk_assessment
        )

    def _assess_deployment_risk(self, context: Dict[str, Any]) -> Dict[str, float]:
        """デプロイメントリスクの評価"""
        risk_scores = {}

        # トラフィックボリュームリスク
        traffic_volume = context.get('daily_request_volume', 0)
        if traffic_volume > 1000000:  # 100万リクエスト/日
            risk_scores['traffic_volume'] = 0.4
        elif traffic_volume > 100000:  # 10万リクエスト/日
            risk_scores['traffic_volume'] = 0.2
        else:
            risk_scores['traffic_volume'] = 0.1

        # 最近のデプロイメント履歴リスク
        recent_deployments = context.get('recent_deployments_count', 0)
        risk_scores['deployment_frequency'] = min(0.5, recent_deployments * 0.1)

        # モデル複雑さリスク
        model_complexity = context.get('model_complexity_score', 0.5)
        risk_scores['model_complexity'] = model_complexity * 0.3

        # 全体リスクスコア
        overall_risk = np.mean(list(risk_scores.values())) if risk_scores else 0.5
        risk_scores['overall_risk'] = overall_risk

        return risk_scores

    def _choose_deployment_strategy(self, should_deploy: bool, confidence_level: float,
                                  risk_assessment: Dict[str, float]) -> DeploymentStrategy:
        """デプロイメント戦略の選択"""
        if not should_deploy:
            return DeploymentStrategy.CANARY  # 保守的な戦略

        overall_risk = risk_assessment.get('overall_risk', 0.5)

        if confidence_level > 0.9 and overall_risk < 0.3:
            return DeploymentStrategy.ROLLING  # 高信頼度・低リスク
        elif confidence_level > 0.8 and overall_risk < 0.5:
            return DeploymentStrategy.BLUE_GREEN  # 中高信頼度・中リスク
        else:
            return DeploymentStrategy.CANARY  # その他はカナリア


class MLExperimentationPlatform:
    """ML実験プラットフォーム"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        初期化

        Args:
            storage_path: データ保存パス
        """
        self.storage_path = Path(storage_path) if storage_path else Path("data/ml_experimentation")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # コアコンポーネントの初期化
        self.ab_testing_framework = ABTestingFramework(str(self.storage_path / "ab_testing"))
        self.deployment_manager = ModelDeploymentManager(str(self.storage_path / "deployments"))
        self.performance_analyzer = ModelPerformanceAnalyzer()
        self.decision_engine = AutoDeploymentDecisionEngine()

        # 実験-デプロイメントペア管理
        self.experiment_deployment_pairs: Dict[str, ExperimentDeploymentPair] = {}
        self.performance_comparisons: Dict[str, ModelPerformanceComparison] = {}
        self.deployment_decisions: Dict[str, AutoDeploymentDecision] = {}

        logger.info(f"ML実験プラットフォーム初期化完了: {self.storage_path}")

    async def create_model_comparison_experiment(self, experiment_name: str,
                                               control_model: Dict[str, Any],
                                               test_model: Dict[str, Any],
                                               traffic_split: float = 0.5,
                                               experiment_duration_hours: int = 24) -> Optional[str]:
        """
        モデル比較実験の作成

        Args:
            experiment_name: 実験名
            control_model: コントロールモデル設定
            test_model: テストモデル設定
            traffic_split: テストモデルへのトラフィック割合
            experiment_duration_hours: 実験期間（時間）

        Returns:
            実験ID
        """
        try:
            # A/B実験の作成
            experiment_config = create_simple_ab_experiment(
                experiment_name=experiment_name,
                control_config=control_model,
                test_config=test_model,
                traffic_split=traffic_split
            )

            # 実験期間の設定
            experiment_config.end_date = datetime.now() + timedelta(hours=experiment_duration_hours)

            # A/Bテストフレームワークに登録
            success = await self.ab_testing_framework.create_experiment(experiment_config)

            if success:
                # 実験を開始
                experiment_config.status = ExperimentStatus.ACTIVE

                logger.info(f"モデル比較実験 {experiment_config.experiment_id} を作成・開始しました")
                return experiment_config.experiment_id

            return None

        except Exception as e:
            logger.error(f"モデル比較実験作成エラー: {e}")
            return None

    async def record_model_prediction(self, experiment_id: str, symbol: str,
                                    prediction: float, actual_value: Optional[float] = None,
                                    latency_ms: float = 0.0, metadata: Dict[str, Any] = None) -> bool:
        """
        モデル予測結果の記録

        Args:
            experiment_id: 実験ID
            symbol: 銘柄コード
            prediction: 予測値
            actual_value: 実測値（オプション）
            latency_ms: レイテンシ（ミリ秒）
            metadata: メタデータ

        Returns:
            記録成功かどうか
        """
        try:
            # 実験グループの割り当て
            group_id = await self.ab_testing_framework.assign_to_experiment(
                experiment_id, symbol, {"timestamp": datetime.now()}
            )

            if not group_id:
                return False

            # 実験結果の記録
            result = ExperimentResult(
                result_id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                group_id=group_id,
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=prediction,
                actual_value=actual_value,
                latency_ms=latency_ms,
                metadata=metadata or {}
            )

            success = await self.ab_testing_framework.record_result(result)

            # 対応するデプロイメントのメトリクス記録
            if experiment_id in self.experiment_deployment_pairs:
                await self._update_deployment_metrics(experiment_id, result)

            return success

        except Exception as e:
            logger.error(f"モデル予測結果記録エラー: {e}")
            return False

    async def analyze_experiment_performance(self, experiment_id: str) -> Optional[ModelPerformanceComparison]:
        """
        実験パフォーマンスの分析

        Args:
            experiment_id: 実験ID

        Returns:
            パフォーマンス比較結果
        """
        try:
            # 実験レポートの生成
            experiment_report = await self.ab_testing_framework.generate_report(experiment_id)

            if not experiment_report:
                logger.warning(f"実験 {experiment_id} のレポートを生成できませんでした")
                return None

            # パフォーマンス分析
            performance_comparison = self.performance_analyzer.analyze_experiment_results(experiment_report)

            # 結果の保存
            self.performance_comparisons[performance_comparison.comparison_id] = performance_comparison
            await self._save_performance_comparison(performance_comparison)

            logger.info(f"実験 {experiment_id} のパフォーマンス分析完了")
            return performance_comparison

        except Exception as e:
            logger.error(f"実験パフォーマンス分析エラー: {e}")
            return None

    async def get_deployment_recommendation(self, experiment_id: str,
                                         deployment_context: Dict[str, Any] = None) -> Optional[AutoDeploymentDecision]:
        """
        デプロイメント推奨事項の取得

        Args:
            experiment_id: 実験ID
            deployment_context: デプロイメントコンテキスト

        Returns:
            デプロイメント判定結果
        """
        try:
            # パフォーマンス分析の実行
            performance_comparison = await self.analyze_experiment_performance(experiment_id)

            if not performance_comparison:
                return None

            # デプロイメント判定
            deployment_decision = self.decision_engine.make_deployment_decision(
                performance_comparison, deployment_context
            )

            # 結果の保存
            self.deployment_decisions[deployment_decision.decision_id] = deployment_decision
            await self._save_deployment_decision(deployment_decision)

            logger.info(f"実験 {experiment_id} のデプロイメント推奨分析完了")
            logger.info(f"推奨: {'デプロイ実行' if deployment_decision.deploy_recommended else 'デプロイ見送り'}")

            return deployment_decision

        except Exception as e:
            logger.error(f"デプロイメント推奨分析エラー: {e}")
            return None

    async def execute_auto_deployment(self, deployment_decision: AutoDeploymentDecision,
                                    model_version: ModelVersion) -> Optional[str]:
        """
        自動デプロイメントの実行

        Args:
            deployment_decision: デプロイメント判定結果
            model_version: デプロイするモデルバージョン

        Returns:
            デプロイメントID
        """
        try:
            if not deployment_decision.deploy_recommended:
                logger.info("デプロイメント推奨されていないため、デプロイメントをスキップします")
                return None

            # モデルバージョンの登録
            await self.deployment_manager.register_model_version(model_version)

            # デプロイメント設定の作成
            deployment_config = self._create_deployment_config(deployment_decision, model_version)

            # デプロイメントの作成・開始
            success = await self.deployment_manager.create_deployment(deployment_config)

            if success:
                await self.deployment_manager.start_deployment(deployment_config.deployment_id)

                # 実験-デプロイメントペアの作成
                pair = ExperimentDeploymentPair(
                    pair_id=str(uuid.uuid4()),
                    experiment_id=deployment_decision.experiment_id,
                    deployment_id=deployment_config.deployment_id,
                    name=f"Auto Deployment from Experiment {deployment_decision.experiment_id}",
                    description=f"自動デプロイメント: {'; '.join(deployment_decision.reasons)}"
                )

                self.experiment_deployment_pairs[pair.pair_id] = pair
                await self._save_experiment_deployment_pair(pair)

                logger.info(f"自動デプロイメント {deployment_config.deployment_id} を開始しました")
                return deployment_config.deployment_id

            return None

        except Exception as e:
            logger.error(f"自動デプロイメント実行エラー: {e}")
            return None

    def get_active_experiments(self) -> List[str]:
        """アクティブな実験一覧を取得"""
        # A/Bテストフレームワークからアクティブな実験を取得
        # 実装簡略化のため、ここでは空のリストを返す
        return []

    def get_active_deployments(self) -> List[str]:
        """アクティブなデプロイメント一覧を取得"""
        return self.deployment_manager.list_active_deployments()

    def get_experiment_deployment_pairs(self) -> Dict[str, ExperimentDeploymentPair]:
        """実験-デプロイメントペア一覧を取得"""
        return self.experiment_deployment_pairs.copy()

    async def _update_deployment_metrics(self, experiment_id: str, result: ExperimentResult):
        """デプロイメントメトリクスの更新"""
        # 対応するデプロイメントIDを取得
        deployment_id = None
        for pair in self.experiment_deployment_pairs.values():
            if pair.experiment_id == experiment_id:
                deployment_id = pair.deployment_id
                break

        if not deployment_id:
            return

        # デプロイメントメトリクスの作成・記録
        metrics = DeploymentMetrics(
            timestamp=result.timestamp,
            deployment_id=deployment_id,
            traffic_percentage=0.5,  # 簡易実装
            request_count=1,
            error_count=1 if result.actual_value and abs(result.prediction - result.actual_value) > 0.1 else 0,
            error_rate=1.0 if result.actual_value and abs(result.prediction - result.actual_value) > 0.1 else 0.0,
            avg_latency_ms=result.latency_ms,
            p99_latency_ms=result.latency_ms * 1.5,  # 推定
            accuracy=1.0 - abs(result.prediction - result.actual_value) if result.actual_value else None
        )

        await self.deployment_manager.record_metrics(deployment_id, metrics)

    def _create_deployment_config(self, decision: AutoDeploymentDecision,
                                model_version: ModelVersion) -> DeploymentConfig:
        """デプロイメント設定の作成"""
        deployment_id = f"auto_deploy_{uuid.uuid4().hex[:8]}"

        # リスクレベルに基づくロールアウト戦略の調整
        risk_level = decision.risk_assessment.get('overall_risk', 0.5)

        if decision.deployment_strategy == DeploymentStrategy.CANARY:
            if risk_level > 0.7:
                # 高リスク：より慎重なカナリア
                rollout_increments = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
                increment_duration = 60  # 60分
            else:
                # 標準カナリア
                rollout_increments = [0.05, 0.2, 0.5, 1.0]
                increment_duration = 30  # 30分
        else:
            rollout_increments = [1.0]  # 即座に全切り替え
            increment_duration = 1

        return DeploymentConfig(
            deployment_id=deployment_id,
            name=f"Auto Deployment - {model_version.name}",
            source_version=model_version,
            strategy=decision.deployment_strategy,
            rollout_increments=rollout_increments,
            increment_duration_minutes=increment_duration,
            auto_rollback_enabled=True,
            error_rate_threshold=0.02,  # 2%
            latency_p99_threshold_ms=500.0,
            accuracy_threshold=0.85
        )

    async def _save_performance_comparison(self, comparison: ModelPerformanceComparison):
        """パフォーマンス比較結果の永続化"""
        comparison_path = self.storage_path / f"performance_comparison_{comparison.comparison_id}.json"

        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(comparison), f, ensure_ascii=False, indent=2, default=str)

    async def _save_deployment_decision(self, decision: AutoDeploymentDecision):
        """デプロイメント判定結果の永続化"""
        decision_path = self.storage_path / f"deployment_decision_{decision.decision_id}.json"

        with open(decision_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(decision), f, ensure_ascii=False, indent=2, default=str)

    async def _save_experiment_deployment_pair(self, pair: ExperimentDeploymentPair):
        """実験-デプロイメントペアの永続化"""
        pair_path = self.storage_path / f"experiment_deployment_pair_{pair.pair_id}.json"

        with open(pair_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(pair), f, ensure_ascii=False, indent=2, default=str)


# メインのユーティリティ関数
def create_ml_experiment_platform(storage_path: Optional[str] = None) -> MLExperimentationPlatform:
    """
    ML実験プラットフォームの作成

    Args:
        storage_path: データ保存パス

    Returns:
        ML実験プラットフォーム
    """
    return MLExperimentationPlatform(storage_path)


async def run_complete_ml_experiment_workflow():
    """完全なML実験ワークフローのデモ"""
    logger.info("ML実験プラットフォーム 完全ワークフローデモ開始")

    # プラットフォームの初期化
    platform = create_ml_experiment_platform("data/ml_experimentation_demo")

    # 1. モデル比較実験の作成
    experiment_id = await platform.create_model_comparison_experiment(
        experiment_name="RandomForest vs XGBoost Performance Test",
        control_model={"model_type": "RandomForest", "n_estimators": 100},
        test_model={"model_type": "XGBoost", "n_estimators": 100},
        traffic_split=0.5,
        experiment_duration_hours=1
    )

    if experiment_id:
        logger.info(f"実験 {experiment_id} を開始しました")

        # 2. シミュレートされた予測結果の記録
        symbols = ["7203", "8306", "9984", "6758", "4689"]

        for i in range(50):
            symbol = np.random.choice(symbols)
            prediction = np.random.normal(100.0, 10.0)
            actual_value = prediction + np.random.normal(0, 5.0)
            latency_ms = np.random.uniform(20.0, 100.0)

            await platform.record_model_prediction(
                experiment_id=experiment_id,
                symbol=symbol,
                prediction=prediction,
                actual_value=actual_value,
                latency_ms=latency_ms
            )

            if i % 10 == 0:
                await asyncio.sleep(0.1)  # 少し待機

        # 3. 実験パフォーマンスの分析
        performance_comparison = await platform.analyze_experiment_performance(experiment_id)

        if performance_comparison:
            logger.info(f"パフォーマンス分析完了: {performance_comparison.recommendation}")

            # 4. デプロイメント推奨事項の取得
            deployment_context = {
                'daily_request_volume': 50000,
                'recent_deployments_count': 1,
                'model_complexity_score': 0.4
            }

            deployment_decision = await platform.get_deployment_recommendation(
                experiment_id, deployment_context
            )

            if deployment_decision:
                logger.info(f"デプロイメント推奨: {deployment_decision.deploy_recommended}")
                logger.info(f"推奨理由: {'; '.join(deployment_decision.reasons)}")

                # 5. 自動デプロイメントの実行（推奨される場合）
                if deployment_decision.deploy_recommended:
                    test_model_version = ModelVersion(
                        version_id="xgboost_v1.0.0",
                        name="XGBoost v1.0.0",
                        description="A/Bテストで勝利したXGBoostモデル",
                        model_path="/models/xgboost_v1.0.0.joblib",
                        config={"model_type": "XGBoost", "n_estimators": 100}
                    )

                    deployment_id = await platform.execute_auto_deployment(
                        deployment_decision, test_model_version
                    )

                    if deployment_id:
                        logger.info(f"自動デプロイメント {deployment_id} を開始しました")

                        # 短時間待機してデプロイメント状態を確認
                        await asyncio.sleep(2)

                        deployment_state = platform.deployment_manager.get_deployment_status(deployment_id)
                        if deployment_state:
                            logger.info(f"デプロイメント状態: {deployment_state.status.value}")
                            logger.info(f"現在のトラフィック: {deployment_state.current_traffic_percentage*100:.1f}%")

    # 6. プラットフォーム状態のサマリー
    active_experiments = platform.get_active_experiments()
    active_deployments = platform.get_active_deployments()
    experiment_deployment_pairs = platform.get_experiment_deployment_pairs()

    logger.info(f"プラットフォーム状態:")
    logger.info(f"  - アクティブな実験: {len(active_experiments)}")
    logger.info(f"  - アクティブなデプロイメント: {len(active_deployments)}")
    logger.info(f"  - 実験-デプロイメントペア: {len(experiment_deployment_pairs)}")

    logger.info("ML実験プラットフォーム 完全ワークフローデモ完了")


async def main():
    """メインデモンストレーション"""
    await run_complete_ml_experiment_workflow()


if __name__ == "__main__":
    asyncio.run(main())