#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction Strategies - 予測戦略モジュール

複数の予測戦略とモデル選択アルゴリズムの実装
Issue #855関連：柔軟性向上のための戦略パターン
"""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from src.day_trade.utils.encoding_fix import apply_windows_encoding_fix

# Windows環境対応
apply_windows_encoding_fix()

logger = logging.getLogger(__name__)


@dataclass
class PredictionCandidate:
    """予測候補"""
    prediction: Any
    system_id: str
    confidence: float
    response_time: float
    accuracy_history: float = 0.0
    weight: float = 1.0


class PredictionStrategy(ABC):
    """予測戦略の基底クラス"""

    @abstractmethod
    async def execute(
            self, candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """戦略を実行して最終予測を選択"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """戦略名を取得"""
        pass


class SingleBestStrategy(PredictionStrategy):
    """最高性能システム単体予測戦略"""

    def __init__(self, selection_criteria: str = "confidence"):
        """
        Args:
            selection_criteria: 選択基準
                ('confidence', 'accuracy', 'response_time')
        """
        self.selection_criteria = selection_criteria

    async def execute(
            self, candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """最高スコアの候補を選択"""
        if not candidates:
            raise ValueError("予測候補が空です")

        valid_candidates = [c for c in candidates if c.prediction is not None]
        if not valid_candidates:
            raise ValueError("有効な予測候補がありません")

        if self.selection_criteria == "confidence":
            best = max(valid_candidates, key=lambda c: c.confidence)
        elif self.selection_criteria == "accuracy":
            best = max(valid_candidates, key=lambda c: c.accuracy_history)
        elif self.selection_criteria == "response_time":
            best = min(valid_candidates, key=lambda c: c.response_time)
        else:
            best = max(valid_candidates, key=lambda c: c.confidence)

        criteria_value = getattr(best, self.selection_criteria)
        logger.debug(
            f"SingleBest選択: {best.system_id} "
            f"({self.selection_criteria}={criteria_value})"
        )
        return best

    def get_strategy_name(self) -> str:
        return f"single_best_{self.selection_criteria}"


class WeightedEnsembleStrategy(PredictionStrategy):
    """重み付きアンサンブル予測戦略"""

    def __init__(self, weighting_method: str = "confidence_based"):
        """
        Args:
            weighting_method: 重み付け方法
                ('confidence_based', 'accuracy_based', 'equal')
        """
        self.weighting_method = weighting_method

    async def execute(
            self, candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """アンサンブル予測を実行"""
        valid_candidates = [c for c in candidates if c.prediction is not None]
        if not valid_candidates:
            raise ValueError("有効な予測候補がありません")

        if len(valid_candidates) == 1:
            return valid_candidates[0]

        # 予測値の数値化とアンサンブル
        ensemble_prediction = await self._compute_ensemble(valid_candidates)

        # アンサンブル結果の信頼度計算
        ensemble_confidence = self._compute_ensemble_confidence(
            valid_candidates
        )

        # 平均レスポンス時間
        avg_response_time = np.mean([
            c.response_time for c in valid_candidates
        ])

        ensemble_candidate = PredictionCandidate(
            prediction=ensemble_prediction,
            system_id="ensemble",
            confidence=ensemble_confidence,
            response_time=avg_response_time,
            accuracy_history=np.mean([
                c.accuracy_history for c in valid_candidates
            ]),
            weight=1.0
        )

        logger.debug(f"アンサンブル予測: {len(valid_candidates)}候補から生成")
        return ensemble_candidate

    async def _compute_ensemble(
            self, candidates: List[PredictionCandidate]
    ) -> Any:
        """アンサンブル予測の計算"""
        # 重み計算
        weights = self._calculate_weights(candidates)

        try:
            # 数値予測の場合
            predictions = []
            for candidate in candidates:
                pred = candidate.prediction
                if isinstance(pred, dict):
                    # 辞書形式の予測から数値を抽出
                    if 'prediction' in pred:
                        pred = pred['prediction']
                    elif 'value' in pred:
                        pred = pred['value']
                    elif 'price' in pred:
                        pred = pred['price']

                if isinstance(pred, (int, float)):
                    predictions.append(float(pred))
                elif hasattr(pred, '__float__'):
                    predictions.append(float(pred))

            if predictions:
                # 重み付き平均
                weighted_prediction = np.average(predictions, weights=weights)
                logger.debug(
                    f"数値アンサンブル: {predictions} -> {weighted_prediction}"
                )
                return weighted_prediction

        except Exception as e:
            logger.debug(f"数値アンサンブル失敗: {e}")

        # 数値化できない場合は信頼度最高の予測を返す
        best_candidate = max(candidates, key=lambda c: c.confidence)
        return best_candidate.prediction

    def _calculate_weights(
            self, candidates: List[PredictionCandidate]
    ) -> List[float]:
        """候補の重み計算"""
        if self.weighting_method == "confidence_based":
            weights = [c.confidence for c in candidates]
        elif self.weighting_method == "accuracy_based":
            weights = [c.accuracy_history for c in candidates]
        elif self.weighting_method == "equal":
            weights = [1.0] * len(candidates)
        else:
            weights = [c.confidence for c in candidates]

        # 正規化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(candidates)] * len(candidates)

        return weights

    def _compute_ensemble_confidence(
            self, candidates: List[PredictionCandidate]
    ) -> float:
        """アンサンブル信頼度の計算"""
        confidences = [c.confidence for c in candidates]

        # 信頼度の調和平均（保守的な推定）
        harmonic_mean = len(confidences) / sum(
            1.0 / max(c, 0.01) for c in confidences
        )

        # 候補数による信頼度向上（最大1.2倍）
        diversity_bonus = min(1.2, 1.0 + 0.05 * (len(candidates) - 1))

        ensemble_confidence = min(0.95, harmonic_mean * diversity_bonus)
        return ensemble_confidence

    def get_strategy_name(self) -> str:
        return f"weighted_ensemble_{self.weighting_method}"


class AdaptiveStrategy(PredictionStrategy):
    """適応的予測戦略"""

    def __init__(self, performance_history: Dict[str, List[float]] = None,
                 adaptation_window: int = 20):
        """
        Args:
            performance_history: システム別性能履歴
            adaptation_window: 適応ウィンドウサイズ
        """
        self.performance_history = performance_history or {}
        self.adaptation_window = adaptation_window

    async def execute(
            self, candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """性能履歴に基づく適応的選択"""
        valid_candidates = [c for c in candidates if c.prediction is not None]
        if not valid_candidates:
            raise ValueError("有効な予測候補がありません")

        # 性能履歴による重み付け
        weighted_candidates = []
        for candidate in valid_candidates:
            adaptive_weight = self._calculate_adaptive_weight(
                candidate.system_id
            )
            weighted_candidates.append((candidate, adaptive_weight))

        # 重み付きランダム選択（確率的）
        total_weight = sum(weight for _, weight in weighted_candidates)
        if total_weight == 0:
            # フォールバック: 信頼度ベース選択
            selected = max(valid_candidates, key=lambda c: c.confidence)
        else:
            # 重み付き確率選択
            import random
            rand_val = random.random() * total_weight
            cumulative_weight = 0
            selected = valid_candidates[0]  # デフォルト

            for candidate, weight in weighted_candidates:
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    selected = candidate
                    break

        logger.debug(f"適応的選択: {selected.system_id}")
        return selected

    def _calculate_adaptive_weight(self, system_id: str) -> float:
        """システムの適応的重み計算"""
        if system_id not in self.performance_history:
            return 1.0  # 履歴がない場合はデフォルト重み

        history = self.performance_history[system_id]
        if not history:
            return 1.0

        # 最近のパフォーマンスを重視
        recent_history = history[-self.adaptation_window:]

        if len(recent_history) < 3:
            return 1.0  # データ不足

        # 性能トレンドの計算
        recent_avg = np.mean(recent_history)
        overall_avg = np.mean(history)

        # トレンドボーナス（最近の性能が全体平均より良い場合）
        trend_bonus = max(0.5, min(2.0, recent_avg / max(overall_avg, 0.1)))

        # 安定性ボーナス（性能の分散が小さい場合）
        stability_bonus = 1.0 / (1.0 + np.std(recent_history))

        adaptive_weight = trend_bonus * stability_bonus
        return adaptive_weight

    def update_performance_history(self, system_id: str, performance: float):
        """性能履歴の更新"""
        if system_id not in self.performance_history:
            self.performance_history[system_id] = []

        self.performance_history[system_id].append(performance)

        # ウィンドウサイズを超えた場合は古いデータを削除
        max_history = self.adaptation_window * 3  # 適応ウィンドウの3倍まで保持
        if len(self.performance_history[system_id]) > max_history:
            self.performance_history[system_id] = (
                self.performance_history[system_id][-max_history:]
            )

    def get_strategy_name(self) -> str:
        return "adaptive"


class FallbackChainStrategy(PredictionStrategy):
    """フォールバックチェーン戦略"""

    def __init__(self, fallback_order: List[str] = None,
                 confidence_threshold: float = 0.6):
        """
        Args:
            fallback_order: フォールバック順序
            confidence_threshold: 信頼度閾値
        """
        self.fallback_order = fallback_order or []
        self.confidence_threshold = confidence_threshold

    async def execute(
            self, candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """フォールバックチェーンによる選択"""
        valid_candidates = [c for c in candidates if c.prediction is not None]
        if not valid_candidates:
            raise ValueError("有効な予測候補がありません")

        # 候補をシステムIDでマップ化
        candidate_map = {c.system_id: c for c in valid_candidates}

        # フォールバック順序に従って選択
        for system_id in self.fallback_order:
            if system_id in candidate_map:
                candidate = candidate_map[system_id]
                if candidate.confidence >= self.confidence_threshold:
                    logger.debug(
                        f"フォールバックチェーン選択: {system_id} "
                        f"(信頼度: {candidate.confidence})"
                    )
                    return candidate

        # フォールバック順序で適切な候補が見つからない場合
        # 信頼度最高の候補を選択
        best_candidate = max(valid_candidates, key=lambda c: c.confidence)
        logger.debug(
            f"フォールバックチェーン: 最高信頼度候補を選択 "
            f"{best_candidate.system_id}"
        )
        return best_candidate

    def get_strategy_name(self) -> str:
        return "fallback_chain"


class StrategyManager:
    """予測戦略管理クラス"""

    def __init__(self):
        self.strategies: Dict[str, PredictionStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """デフォルト戦略の登録"""
        self.register_strategy(
            "single_best_confidence", SingleBestStrategy("confidence")
        )
        self.register_strategy(
            "single_best_accuracy", SingleBestStrategy("accuracy")
        )
        self.register_strategy(
            "single_best_speed", SingleBestStrategy("response_time")
        )

        self.register_strategy(
            "ensemble_confidence", WeightedEnsembleStrategy("confidence_based")
        )
        self.register_strategy(
            "ensemble_accuracy", WeightedEnsembleStrategy("accuracy_based")
        )
        self.register_strategy(
            "ensemble_equal", WeightedEnsembleStrategy("equal")
        )

        self.register_strategy("adaptive", AdaptiveStrategy())
        self.register_strategy("fallback_chain", FallbackChainStrategy())

    def register_strategy(self, name: str, strategy: PredictionStrategy):
        """戦略の登録"""
        self.strategies[name] = strategy
        logger.debug(f"予測戦略を登録: {name}")

    def get_strategy(self, name: str) -> Optional[PredictionStrategy]:
        """戦略の取得"""
        return self.strategies.get(name)

    def list_strategies(self) -> List[str]:
        """利用可能な戦略のリスト"""
        return list(self.strategies.keys())

    async def execute_strategy(
            self, strategy_name: str,
            candidates: List[PredictionCandidate]
    ) -> PredictionCandidate:
        """指定戦略の実行"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"未知の戦略: {strategy_name}")

        return await strategy.execute(candidates)


# グローバル戦略マネージャーインスタンス
strategy_manager = StrategyManager()


# ユーティリティ関数
def create_prediction_candidate(
        prediction: Any, system_id: str,
        confidence: float = 0.7,
        response_time: float = 1.0
) -> PredictionCandidate:
    """PredictionCandidateの作成ヘルパー"""
    return PredictionCandidate(
        prediction=prediction,
        system_id=system_id,
        confidence=confidence,
        response_time=response_time
    )


if __name__ == "__main__":
    # 戦略テスト
    async def test_strategies():
        logger.info("予測戦略テスト開始")

        # テスト候補作成
        candidates = [
            create_prediction_candidate(100.5, "system_a", 0.8, 0.5),
            create_prediction_candidate(101.2, "system_b", 0.7, 1.0),
            create_prediction_candidate(99.8, "system_c", 0.9, 1.5),
        ]

        # 各戦略のテスト
        for strategy_name in strategy_manager.list_strategies():
            try:
                result = await strategy_manager.execute_strategy(
                    strategy_name, candidates
                )
                logger.info(
                    f"{strategy_name}: {result.system_id} "
                    f"(予測: {result.prediction}, 信頼度: {result.confidence})"
                )
            except Exception as e:
                logger.error(f"{strategy_name} エラー: {e}")

    logging.basicConfig(
        level=logging.INFO, format='%(levelname)s: %(message)s'
    )
    asyncio.run(test_strategies())
