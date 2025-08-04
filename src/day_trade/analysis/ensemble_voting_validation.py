"""
アンサンブル投票アルゴリズム検証システム

投票ロジックの詳細検証、重み付けアルゴリズムの分析、
戦略合意形成プロセスの検証を実行する。
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging_config import get_context_logger
from ..utils.performance_analyzer import profile_performance
from .ensemble import EnsembleVotingType
from .signals import SignalStrength, SignalType, TradingSignal

warnings.filterwarnings("ignore")
logger = get_context_logger(__name__)


@dataclass
class VotingTestScenario:
    """投票テストシナリオ"""

    scenario_name: str
    description: str
    strategy_signals: List[Tuple[str, TradingSignal]]  # (strategy_name, signal)
    strategy_weights: Dict[str, float]
    expected_result: Dict[str, Any]
    voting_type: EnsembleVotingType
    tolerance: float = 0.1  # 許容誤差


@dataclass
class VotingValidationResult:
    """投票検証結果"""

    scenario_name: str
    voting_type: str
    success: bool
    actual_signal_type: Optional[SignalType] = None
    expected_signal_type: Optional[SignalType] = None
    actual_confidence: float = 0.0
    expected_confidence: float = 0.0
    confidence_error: float = 0.0
    weight_consistency: bool = True
    calculation_details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class VotingAnalysisReport:
    """投票分析レポート"""

    analysis_timestamp: datetime
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int

    # 投票タイプ別結果
    soft_voting_results: Dict[str, int]
    hard_voting_results: Dict[str, int]
    weighted_average_results: Dict[str, int]

    # 精度分析
    avg_confidence_error: float
    max_confidence_error: float
    signal_type_accuracy: float

    # 重み付け分析
    weight_consistency_rate: float
    weight_distribution_analysis: Dict[str, float]

    # 推奨事項
    voting_algorithm_recommendations: List[str]
    identified_issues: List[str]


class EnsembleVotingValidator:
    """アンサンブル投票アルゴリズム検証器"""

    def __init__(self):
        self.test_scenarios = []
        self.validation_results = []
        self.mathematical_precision = 1e-6

        logger.info(
            "アンサンブル投票アルゴリズム検証器初期化完了",
            section="voting_validator_init",
        )

    @profile_performance
    def run_comprehensive_voting_validation(self) -> VotingAnalysisReport:
        """包括的投票検証実行"""

        logger.info(
            "包括的投票アルゴリズム検証開始", section="comprehensive_voting_validation"
        )

        try:
            # 1. テストシナリオ生成
            self._generate_voting_test_scenarios()

            # 2. 各投票アルゴリズムの検証
            soft_voting_results = self._validate_soft_voting()
            hard_voting_results = self._validate_hard_voting()
            weighted_average_results = self._validate_weighted_average()

            # 3. 重み付けロジック検証
            weight_validation_results = self._validate_weight_logic()

            # 4. エッジケース検証
            edge_case_results = self._validate_edge_cases()

            # 5. 数学的正確性検証
            mathematical_validation = self._validate_mathematical_correctness()

            # 6. 結果分析
            analysis_report = self._analyze_voting_results(
                soft_voting_results,
                hard_voting_results,
                weighted_average_results,
                weight_validation_results,
                edge_case_results,
                mathematical_validation,
            )

            logger.info(
                "包括的投票アルゴリズム検証完了",
                section="comprehensive_voting_validation",
                total_scenarios=analysis_report.total_scenarios,
                passed_scenarios=analysis_report.passed_scenarios,
                success_rate=analysis_report.passed_scenarios
                / analysis_report.total_scenarios
                * 100
                if analysis_report.total_scenarios > 0
                else 0,
            )

            return analysis_report

        except Exception as e:
            logger.error(
                "包括的投票検証エラー",
                section="comprehensive_voting_validation",
                error=str(e),
            )
            raise

    def _generate_voting_test_scenarios(self):
        """投票テストシナリオ生成"""

        self.test_scenarios = []

        # 1. 基本的な合意シナリオ
        self.test_scenarios.extend(self._create_consensus_scenarios())

        # 2. 分裂シナリオ
        self.test_scenarios.extend(self._create_split_scenarios())

        # 3. 重み付けテストシナリオ
        self.test_scenarios.extend(self._create_weighted_scenarios())

        # 4. エッジケースシナリオ
        self.test_scenarios.extend(self._create_edge_case_scenarios())

        logger.info(
            f"投票テストシナリオ生成完了: {len(self.test_scenarios)}件",
            section="voting_scenario_generation",
        )

    def _create_consensus_scenarios(self) -> List[VotingTestScenario]:
        """合意シナリオ作成"""

        scenarios = []

        # 全戦略買い合意
        buy_signals = [
            ("strategy_1", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 80.0)),
            ("strategy_2", TradingSignal(SignalType.BUY, SignalStrength.MEDIUM, 75.0)),
            ("strategy_3", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 85.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="unanimous_buy_consensus",
                description="全戦略による買い合意",
                strategy_signals=buy_signals,
                strategy_weights={
                    "strategy_1": 0.4,
                    "strategy_2": 0.3,
                    "strategy_3": 0.3,
                },
                expected_result={
                    "signal_type": SignalType.BUY,
                    "confidence_range": (75.0, 85.0),
                },
                voting_type=EnsembleVotingType.SOFT_VOTING,
            )
        )

        # 全戦略売り合意
        sell_signals = [
            ("strategy_1", TradingSignal(SignalType.SELL, SignalStrength.STRONG, 90.0)),
            ("strategy_2", TradingSignal(SignalType.SELL, SignalStrength.MEDIUM, 70.0)),
            ("strategy_3", TradingSignal(SignalType.SELL, SignalStrength.STRONG, 85.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="unanimous_sell_consensus",
                description="全戦略による売り合意",
                strategy_signals=sell_signals,
                strategy_weights={
                    "strategy_1": 0.4,
                    "strategy_2": 0.3,
                    "strategy_3": 0.3,
                },
                expected_result={
                    "signal_type": SignalType.SELL,
                    "confidence_range": (70.0, 90.0),
                },
                voting_type=EnsembleVotingType.SOFT_VOTING,
            )
        )

        return scenarios

    def _create_split_scenarios(self) -> List[VotingTestScenario]:
        """分裂シナリオ作成"""

        scenarios = []

        # 2対1の分裂（買い優勢）
        split_signals_buy = [
            ("strategy_1", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 80.0)),
            ("strategy_2", TradingSignal(SignalType.BUY, SignalStrength.MEDIUM, 70.0)),
            ("strategy_3", TradingSignal(SignalType.SELL, SignalStrength.WEAK, 60.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="buy_majority_split",
                description="買い優勢の2対1分裂",
                strategy_signals=split_signals_buy,
                strategy_weights={
                    "strategy_1": 0.33,
                    "strategy_2": 0.33,
                    "strategy_3": 0.34,
                },
                expected_result={
                    "signal_type": SignalType.BUY,
                    "confidence_range": (60.0, 80.0),
                },
                voting_type=EnsembleVotingType.HARD_VOTING,
            )
        )

        # 1対1対1の完全分裂
        complete_split_signals = [
            ("strategy_1", TradingSignal(SignalType.BUY, SignalStrength.MEDIUM, 70.0)),
            ("strategy_2", TradingSignal(SignalType.SELL, SignalStrength.MEDIUM, 75.0)),
            ("strategy_3", TradingSignal(SignalType.HOLD, SignalStrength.WEAK, 50.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="complete_split",
                description="完全分裂シナリオ",
                strategy_signals=complete_split_signals,
                strategy_weights={
                    "strategy_1": 0.33,
                    "strategy_2": 0.33,
                    "strategy_3": 0.34,
                },
                expected_result={
                    "signal_type": SignalType.HOLD,  # 分裂時のデフォルト
                    "confidence_range": (40.0, 60.0),
                },
                voting_type=EnsembleVotingType.HARD_VOTING,
            )
        )

        return scenarios

    def _create_weighted_scenarios(self) -> List[VotingTestScenario]:
        """重み付けテストシナリオ作成"""

        scenarios = []

        # 高重み戦略による影響力テスト
        weighted_signals = [
            (
                "high_weight_strategy",
                TradingSignal(SignalType.BUY, SignalStrength.STRONG, 85.0),
            ),
            (
                "low_weight_strategy_1",
                TradingSignal(SignalType.SELL, SignalStrength.MEDIUM, 70.0),
            ),
            (
                "low_weight_strategy_2",
                TradingSignal(SignalType.SELL, SignalStrength.WEAK, 60.0),
            ),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="high_weight_dominance",
                description="高重み戦略の支配力テスト",
                strategy_signals=weighted_signals,
                strategy_weights={
                    "high_weight_strategy": 0.7,
                    "low_weight_strategy_1": 0.15,
                    "low_weight_strategy_2": 0.15,
                },
                expected_result={
                    "signal_type": SignalType.BUY,  # 高重み戦略が勝利
                    "confidence_range": (70.0, 85.0),
                },
                voting_type=EnsembleVotingType.WEIGHTED_AVERAGE,
            )
        )

        return scenarios

    def _create_edge_case_scenarios(self) -> List[VotingTestScenario]:
        """エッジケースシナリオ作成"""

        scenarios = []

        # 信頼度ゼロのシグナル
        zero_confidence_signals = [
            ("strategy_1", TradingSignal(SignalType.BUY, SignalStrength.WEAK, 0.0)),
            ("strategy_2", TradingSignal(SignalType.SELL, SignalStrength.WEAK, 0.0)),
            ("strategy_3", TradingSignal(SignalType.HOLD, SignalStrength.WEAK, 0.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="zero_confidence_signals",
                description="信頼度ゼロシグナルの処理",
                strategy_signals=zero_confidence_signals,
                strategy_weights={
                    "strategy_1": 0.33,
                    "strategy_2": 0.33,
                    "strategy_3": 0.34,
                },
                expected_result={
                    "signal_type": SignalType.HOLD,
                    "confidence_range": (0.0, 10.0),
                },
                voting_type=EnsembleVotingType.SOFT_VOTING,
            )
        )

        # 極端に高い信頼度
        extreme_confidence_signals = [
            ("strategy_1", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 100.0)),
            ("strategy_2", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 100.0)),
            ("strategy_3", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 100.0)),
        ]

        scenarios.append(
            VotingTestScenario(
                scenario_name="extreme_confidence_signals",
                description="極端に高い信頼度の処理",
                strategy_signals=extreme_confidence_signals,
                strategy_weights={
                    "strategy_1": 0.33,
                    "strategy_2": 0.33,
                    "strategy_3": 0.34,
                },
                expected_result={
                    "signal_type": SignalType.BUY,
                    "confidence_range": (95.0, 100.0),
                },
                voting_type=EnsembleVotingType.SOFT_VOTING,
            )
        )

        return scenarios

    def _validate_soft_voting(self) -> List[VotingValidationResult]:
        """ソフト投票検証"""

        results = []
        soft_voting_scenarios = [
            s
            for s in self.test_scenarios
            if s.voting_type == EnsembleVotingType.SOFT_VOTING
        ]

        for scenario in soft_voting_scenarios:
            result = self._execute_voting_test(scenario)
            results.append(result)

        logger.info(
            f"ソフト投票検証完了: {len(results)}件",
            section="soft_voting_validation",
            passed=sum(1 for r in results if r.success),
        )

        return results

    def _validate_hard_voting(self) -> List[VotingValidationResult]:
        """ハード投票検証"""

        results = []
        hard_voting_scenarios = [
            s
            for s in self.test_scenarios
            if s.voting_type == EnsembleVotingType.HARD_VOTING
        ]

        for scenario in hard_voting_scenarios:
            result = self._execute_voting_test(scenario)
            results.append(result)

        logger.info(
            f"ハード投票検証完了: {len(results)}件",
            section="hard_voting_validation",
            passed=sum(1 for r in results if r.success),
        )

        return results

    def _validate_weighted_average(self) -> List[VotingValidationResult]:
        """重み付け平均検証"""

        results = []
        weighted_scenarios = [
            s
            for s in self.test_scenarios
            if s.voting_type == EnsembleVotingType.WEIGHTED_AVERAGE
        ]

        for scenario in weighted_scenarios:
            result = self._execute_voting_test(scenario)
            results.append(result)

        logger.info(
            f"重み付け平均検証完了: {len(results)}件",
            section="weighted_average_validation",
            passed=sum(1 for r in results if r.success),
        )

        return results

    def _execute_voting_test(
        self, scenario: VotingTestScenario
    ) -> VotingValidationResult:
        """投票テスト実行"""

        try:
            # 模擬的な投票計算実行
            actual_result = self._simulate_voting_calculation(scenario)

            # 結果検証
            success = True

            # シグナルタイプ検証
            expected_signal_type = scenario.expected_result.get("signal_type")
            if (
                expected_signal_type
                and actual_result["signal_type"] != expected_signal_type
            ):
                success = False

            # 信頼度範囲検証
            expected_confidence_range = scenario.expected_result.get("confidence_range")
            actual_confidence = actual_result["confidence"]

            if expected_confidence_range and not (
                expected_confidence_range[0]
                <= actual_confidence
                <= expected_confidence_range[1]
            ):
                success = False

            # 重み一貫性検証
            weight_consistency = self._validate_weight_consistency(
                scenario.strategy_weights
            )

            confidence_error = 0.0
            if expected_confidence_range:
                expected_mid = (
                    expected_confidence_range[0] + expected_confidence_range[1]
                ) / 2
                confidence_error = abs(actual_confidence - expected_mid)

            return VotingValidationResult(
                scenario_name=scenario.scenario_name,
                voting_type=scenario.voting_type.value,
                success=success,
                actual_signal_type=actual_result["signal_type"],
                expected_signal_type=expected_signal_type,
                actual_confidence=actual_confidence,
                expected_confidence=expected_confidence_range[0]
                if expected_confidence_range
                else 0,
                confidence_error=confidence_error,
                weight_consistency=weight_consistency,
                calculation_details=actual_result.get("details", {}),
            )

        except Exception as e:
            return VotingValidationResult(
                scenario_name=scenario.scenario_name,
                voting_type=scenario.voting_type.value,
                success=False,
                error_message=str(e),
            )

    def _simulate_voting_calculation(
        self, scenario: VotingTestScenario
    ) -> Dict[str, Any]:
        """投票計算シミュレーション"""

        signals = scenario.strategy_signals
        weights = scenario.strategy_weights
        voting_type = scenario.voting_type

        if voting_type == EnsembleVotingType.SOFT_VOTING:
            return self._calculate_soft_voting(signals, weights)
        elif voting_type == EnsembleVotingType.HARD_VOTING:
            return self._calculate_hard_voting(signals, weights)
        elif voting_type == EnsembleVotingType.WEIGHTED_AVERAGE:
            return self._calculate_weighted_average(signals, weights)
        else:
            raise ValueError(f"Unsupported voting type: {voting_type}")

    def _calculate_soft_voting(
        self, signals: List[Tuple[str, TradingSignal]], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """ソフト投票計算"""

        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0

        for strategy_name, signal in signals:
            weight = weights.get(strategy_name, 0.0)
            confidence_weight = signal.confidence * weight

            if signal.signal_type == SignalType.BUY:
                buy_score += confidence_weight
            elif signal.signal_type == SignalType.SELL:
                sell_score += confidence_weight
            else:  # HOLD
                hold_score += confidence_weight

            total_weight += weight

        # 正規化
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
            hold_score /= total_weight

        # 最高スコアのシグナルタイプを選択
        max_score = max(buy_score, sell_score, hold_score)

        if max_score == buy_score:
            signal_type = SignalType.BUY
            confidence = buy_score
        elif max_score == sell_score:
            signal_type = SignalType.SELL
            confidence = sell_score
        else:
            signal_type = SignalType.HOLD
            confidence = hold_score

        return {
            "signal_type": signal_type,
            "confidence": confidence,
            "details": {
                "buy_score": buy_score,
                "sell_score": sell_score,
                "hold_score": hold_score,
                "total_weight": total_weight,
            },
        }

    def _calculate_hard_voting(
        self, signals: List[Tuple[str, TradingSignal]], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """ハード投票計算"""

        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        vote_weights = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_confidence = 0.0

        for strategy_name, signal in signals:
            weight = weights.get(strategy_name, 0.0)

            if signal.signal_type == SignalType.BUY:
                buy_votes += 1
                vote_weights["buy"] += weight
            elif signal.signal_type == SignalType.SELL:
                sell_votes += 1
                vote_weights["sell"] += weight
            else:  # HOLD
                hold_votes += 1
                vote_weights["hold"] += weight

            total_confidence += signal.confidence

        # 最多投票のシグナルタイプを選択
        if buy_votes > sell_votes and buy_votes > hold_votes:
            signal_type = SignalType.BUY
            confidence = vote_weights["buy"] / max(sum(vote_weights.values()), 1) * 100
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            signal_type = SignalType.SELL
            confidence = vote_weights["sell"] / max(sum(vote_weights.values()), 1) * 100
        else:
            signal_type = SignalType.HOLD
            confidence = vote_weights["hold"] / max(sum(vote_weights.values()), 1) * 100

        # 同票の場合はHOLD
        if (
            buy_votes == sell_votes
            or (buy_votes == hold_votes and buy_votes > sell_votes)
            or (sell_votes == hold_votes and sell_votes > buy_votes)
        ):
            signal_type = SignalType.HOLD
            confidence = total_confidence / len(signals) if signals else 50.0

        return {
            "signal_type": signal_type,
            "confidence": confidence,
            "details": {
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "hold_votes": hold_votes,
                "vote_weights": vote_weights,
            },
        }

    def _calculate_weighted_average(
        self, signals: List[Tuple[str, TradingSignal]], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """重み付け平均計算"""

        # シグナルタイプを数値化（BUY=1, HOLD=0, SELL=-1）
        signal_values = []
        confidence_values = []
        weight_values = []

        for strategy_name, signal in signals:
            weight = weights.get(strategy_name, 0.0)

            if signal.signal_type == SignalType.BUY:
                signal_value = 1.0
            elif signal.signal_type == SignalType.SELL:
                signal_value = -1.0
            else:  # HOLD
                signal_value = 0.0

            signal_values.append(signal_value)
            confidence_values.append(signal.confidence)
            weight_values.append(weight)

        # 重み付け平均計算
        if sum(weight_values) > 0:
            weighted_signal = sum(
                sv * cv * wv
                for sv, cv, wv in zip(signal_values, confidence_values, weight_values)
            )
            weighted_signal /= sum(
                cv * wv for cv, wv in zip(confidence_values, weight_values)
            )

            weighted_confidence = sum(
                cv * wv for cv, wv in zip(confidence_values, weight_values)
            )
            weighted_confidence /= sum(weight_values)
        else:
            weighted_signal = 0.0
            weighted_confidence = 50.0

        # 数値化されたシグナルを元のタイプに変換
        if weighted_signal > 0.1:
            signal_type = SignalType.BUY
        elif weighted_signal < -0.1:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        return {
            "signal_type": signal_type,
            "confidence": abs(weighted_confidence),
            "details": {
                "weighted_signal_value": weighted_signal,
                "signal_values": signal_values,
                "confidence_values": confidence_values,
                "weight_values": weight_values,
            },
        }

    def _validate_weight_consistency(self, weights: Dict[str, float]) -> bool:
        """重み一貫性検証"""

        # 重みの合計が1に近いかチェック
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > self.mathematical_precision:
            return False

        # 負の重みがないかチェック
        return not any(w < 0 for w in weights.values())

    def _validate_weight_logic(self) -> List[VotingValidationResult]:
        """重み付けロジック検証"""

        results = []

        # 重み正規化テスト
        test_weights = {"s1": 0.2, "s2": 0.3, "s3": 0.6}  # 合計1.1
        normalized = self._normalize_weights(test_weights)

        normalization_success = (
            abs(sum(normalized.values()) - 1.0) < self.mathematical_precision
        )

        results.append(
            VotingValidationResult(
                scenario_name="weight_normalization",
                voting_type="weight_logic",
                success=normalization_success,
                calculation_details={
                    "original": test_weights,
                    "normalized": normalized,
                },
            )
        )

        return results

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """重み正規化"""

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}
        else:
            # 全て0の場合は均等割り当て
            n = len(weights)
            return {k: 1.0 / n for k in weights} if n > 0 else {}

    def _validate_edge_cases(self) -> List[VotingValidationResult]:
        """エッジケース検証"""

        results = []
        edge_case_scenarios = [
            s
            for s in self.test_scenarios
            if "zero_confidence" in s.scenario_name
            or "extreme_confidence" in s.scenario_name
        ]

        for scenario in edge_case_scenarios:
            result = self._execute_voting_test(scenario)
            results.append(result)

        return results

    def _validate_mathematical_correctness(self) -> List[VotingValidationResult]:
        """数学的正確性検証"""

        results = []

        # 可換性テスト（順序を変えても結果が同じ）
        original_signals = [
            ("s1", TradingSignal(SignalType.BUY, SignalStrength.STRONG, 80.0)),
            ("s2", TradingSignal(SignalType.SELL, SignalStrength.MEDIUM, 70.0)),
        ]
        reversed_signals = list(reversed(original_signals))
        weights = {"s1": 0.6, "s2": 0.4}

        original_result = self._calculate_soft_voting(original_signals, weights)
        reversed_result = self._calculate_soft_voting(reversed_signals, weights)

        commutativity_success = (
            original_result["signal_type"] == reversed_result["signal_type"]
            and abs(original_result["confidence"] - reversed_result["confidence"])
            < self.mathematical_precision
        )

        results.append(
            VotingValidationResult(
                scenario_name="commutativity_test",
                voting_type="mathematical_correctness",
                success=commutativity_success,
                calculation_details={
                    "original_result": original_result,
                    "reversed_result": reversed_result,
                },
            )
        )

        return results

    def _analyze_voting_results(
        self,
        soft_voting_results: List[VotingValidationResult],
        hard_voting_results: List[VotingValidationResult],
        weighted_average_results: List[VotingValidationResult],
        weight_validation_results: List[VotingValidationResult],
        edge_case_results: List[VotingValidationResult],
        mathematical_validation: List[VotingValidationResult],
    ) -> VotingAnalysisReport:
        """投票結果分析"""

        all_results = (
            soft_voting_results
            + hard_voting_results
            + weighted_average_results
            + weight_validation_results
            + edge_case_results
            + mathematical_validation
        )

        total_scenarios = len(all_results)
        passed_scenarios = sum(1 for r in all_results if r.success)
        failed_scenarios = total_scenarios - passed_scenarios

        # 投票タイプ別結果
        soft_voting_stats = {
            "passed": sum(1 for r in soft_voting_results if r.success),
            "failed": len(soft_voting_results)
            - sum(1 for r in soft_voting_results if r.success),
        }
        hard_voting_stats = {
            "passed": sum(1 for r in hard_voting_results if r.success),
            "failed": len(hard_voting_results)
            - sum(1 for r in hard_voting_results if r.success),
        }
        weighted_average_stats = {
            "passed": sum(1 for r in weighted_average_results if r.success),
            "failed": len(weighted_average_results)
            - sum(1 for r in weighted_average_results if r.success),
        }

        # 精度分析
        confidence_errors = [
            r.confidence_error for r in all_results if r.confidence_error > 0
        ]
        avg_confidence_error = np.mean(confidence_errors) if confidence_errors else 0.0
        max_confidence_error = np.max(confidence_errors) if confidence_errors else 0.0

        signal_type_matches = sum(
            1
            for r in all_results
            if r.actual_signal_type == r.expected_signal_type
            and r.expected_signal_type is not None
        )
        signal_type_accuracy = (
            signal_type_matches
            / len([r for r in all_results if r.expected_signal_type is not None])
            * 100
            if len([r for r in all_results if r.expected_signal_type is not None]) > 0
            else 0
        )

        # 重み付け分析
        weight_consistency_rate = (
            sum(1 for r in all_results if r.weight_consistency) / total_scenarios * 100
            if total_scenarios > 0
            else 0
        )

        # 推奨事項生成
        recommendations = self._generate_voting_recommendations(all_results)
        issues = self._identify_voting_issues(all_results)

        return VotingAnalysisReport(
            analysis_timestamp=datetime.now(),
            total_scenarios=total_scenarios,
            passed_scenarios=passed_scenarios,
            failed_scenarios=failed_scenarios,
            soft_voting_results=soft_voting_stats,
            hard_voting_results=hard_voting_stats,
            weighted_average_results=weighted_average_stats,
            avg_confidence_error=avg_confidence_error,
            max_confidence_error=max_confidence_error,
            signal_type_accuracy=signal_type_accuracy,
            weight_consistency_rate=weight_consistency_rate,
            weight_distribution_analysis={},
            voting_algorithm_recommendations=recommendations,
            identified_issues=issues,
        )

    def _generate_voting_recommendations(
        self, results: List[VotingValidationResult]
    ) -> List[str]:
        """投票推奨事項生成"""

        recommendations = []

        success_rate = (
            sum(1 for r in results if r.success) / len(results) * 100 if results else 0
        )

        if success_rate > 95:
            recommendations.append("投票アルゴリズムは非常に高い精度で動作している")
        elif success_rate > 85:
            recommendations.append(
                "投票アルゴリズムは良好に動作しているが、細部の改善の余地がある"
            )
        else:
            recommendations.append("投票アルゴリズムの大幅な改善が必要")

        # 信頼度エラー分析
        confidence_errors = [
            r.confidence_error for r in results if r.confidence_error > 0
        ]
        if confidence_errors:
            avg_error = np.mean(confidence_errors)
            if avg_error > 10:
                recommendations.append("信頼度計算の精度向上が必要")

        return recommendations

    def _identify_voting_issues(
        self, results: List[VotingValidationResult]
    ) -> List[str]:
        """投票問題特定"""

        issues = []

        failed_results = [r for r in results if not r.success]
        if failed_results:
            issues.append(f"{len(failed_results)}件の投票テストが失敗")

        # エラーメッセージ分析
        error_messages = [r.error_message for r in failed_results if r.error_message]
        if error_messages:
            issues.append("投票計算中にエラーが発生")

        return issues

    def generate_detailed_voting_report(self, report: VotingAnalysisReport) -> str:
        """詳細投票検証レポート生成"""

        lines = [
            "=" * 80,
            "アンサンブル投票アルゴリズム詳細検証レポート",
            "=" * 80,
            "",
            f"分析日時: {report.analysis_timestamp}",
            f"総シナリオ数: {report.total_scenarios}",
            f"成功: {report.passed_scenarios}, 失敗: {report.failed_scenarios}",
            f"成功率: {report.passed_scenarios / report.total_scenarios * 100:.1f}%"
            if report.total_scenarios > 0
            else "成功率: N/A",
            "",
            "投票アルゴリズム別結果:",
            "-" * 50,
            f"ソフト投票: 成功={report.soft_voting_results['passed']}, 失敗={report.soft_voting_results['failed']}",
            f"ハード投票: 成功={report.hard_voting_results['passed']}, 失敗={report.hard_voting_results['failed']}",
            f"重み付け平均: 成功={report.weighted_average_results['passed']}, 失敗={report.weighted_average_results['failed']}",
            "",
            "精度分析:",
            "-" * 30,
            f"平均信頼度誤差: {report.avg_confidence_error:.2f}%",
            f"最大信頼度誤差: {report.max_confidence_error:.2f}%",
            f"シグナルタイプ精度: {report.signal_type_accuracy:.1f}%",
            f"重み一貫性率: {report.weight_consistency_rate:.1f}%",
            "",
            "推奨事項:",
            "-" * 30,
        ]

        for recommendation in report.voting_algorithm_recommendations:
            lines.append(f"- {recommendation}")

        if report.identified_issues:
            lines.extend(["", "特定された問題:", "-" * 30])
            for issue in report.identified_issues:
                lines.append(f"- {issue}")

        lines.extend(["", "=" * 80])

        return "\n".join(lines)


# 使用例とデモ
if __name__ == "__main__":
    logger.info("アンサンブル投票アルゴリズム検証デモ開始", section="demo")

    try:
        # 投票検証器作成
        validator = EnsembleVotingValidator()

        # 包括的投票検証実行
        analysis_report = validator.run_comprehensive_voting_validation()

        # 詳細レポート生成
        detailed_report = validator.generate_detailed_voting_report(analysis_report)

        logger.info(
            "アンサンブル投票アルゴリズム検証デモ完了",
            section="demo",
            total_scenarios=analysis_report.total_scenarios,
            success_rate=analysis_report.passed_scenarios
            / analysis_report.total_scenarios
            * 100
            if analysis_report.total_scenarios > 0
            else 0,
        )

        print(detailed_report)

    except Exception as e:
        logger.error(f"投票検証デモエラー: {e}", section="demo")

    finally:
        # リソースクリーンアップ
        import gc

        gc.collect()
