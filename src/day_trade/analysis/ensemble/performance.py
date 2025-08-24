"""
戦略パフォーマンス管理モジュール
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__, component="ensemble_performance")


@dataclass
class StrategyPerformance:
    """戦略パフォーマンス記録"""

    strategy_name: str
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    success_rate: float = 0.0
    average_confidence: float = 0.0
    average_return: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None

    def update_performance(
        self, success: bool, confidence: float, return_rate: float = 0.0
    ) -> None:
        """パフォーマンスを更新"""
        self.total_signals += 1
        if success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1

        self.success_rate = (
            self.successful_signals / self.total_signals
            if self.total_signals > 0
            else 0.0
        )

        # 移動平均で信頼度と収益率を更新
        alpha = 0.1  # 学習率
        if self.total_signals == 1:
            # 初回は直接設定
            self.average_confidence = confidence
        else:
            self.average_confidence = (
                1 - alpha
            ) * self.average_confidence + alpha * confidence
        self.average_return = (1 - alpha) * self.average_return + alpha * return_rate

        self.last_updated = datetime.now()


class PerformanceManager:
    """パフォーマンス管理クラス"""

    def __init__(self, performance_file: str = None):
        """
        Args:
            performance_file: パフォーマンス記録ファイルパス
        """
        self.performance_file = performance_file
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self._load_performance_history()

    def _load_performance_history(self) -> None:
        """パフォーマンス履歴をロード"""
        if not self.performance_file:
            return

        try:
            performance_path = Path(self.performance_file)
            if performance_path.exists():
                with open(performance_path, encoding="utf-8") as f:
                    data = json.load(f)

                for strategy_name, perf_data in data.items():
                    self.strategy_performance[strategy_name] = StrategyPerformance(
                        strategy_name=strategy_name,
                        total_signals=perf_data.get("total_signals", 0),
                        successful_signals=perf_data.get("successful_signals", 0),
                        failed_signals=perf_data.get("failed_signals", 0),
                        success_rate=perf_data.get("success_rate", 0.0),
                        average_confidence=perf_data.get("average_confidence", 0.0),
                        average_return=perf_data.get("average_return", 0.0),
                        sharpe_ratio=perf_data.get("sharpe_ratio", 0.0),
                        last_updated=(
                            datetime.fromisoformat(perf_data["last_updated"])
                            if perf_data.get("last_updated")
                            else None
                        ),
                    )

                logger.info(
                    f"パフォーマンス履歴をロード: {len(self.strategy_performance)} 戦略"
                )
        except Exception as e:
            logger.warning(f"パフォーマンス履歴ロードエラー: {e}")

    def _save_performance_history(self) -> None:
        """パフォーマンス履歴を保存"""
        if not self.performance_file:
            return

        try:
            data = {}
            for strategy_name, perf in self.strategy_performance.items():
                data[strategy_name] = {
                    "total_signals": perf.total_signals,
                    "successful_signals": perf.successful_signals,
                    "failed_signals": perf.failed_signals,
                    "success_rate": perf.success_rate,
                    "average_confidence": perf.average_confidence,
                    "average_return": perf.average_return,
                    "sharpe_ratio": perf.sharpe_ratio,
                    "last_updated": (
                        perf.last_updated.isoformat() if perf.last_updated else None
                    ),
                }

            performance_path = Path(self.performance_file)
            performance_path.parent.mkdir(parents=True, exist_ok=True)

            with open(performance_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug("パフォーマンス履歴を保存")
        except Exception as e:
            logger.error(f"パフォーマンス履歴保存エラー: {e}")

    def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        confidence: float,
        return_rate: float = 0.0,
    ) -> None:
        """戦略パフォーマンスを更新"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name
            )

        self.strategy_performance[strategy_name].update_performance(
            success, confidence, return_rate
        )
        self._save_performance_history()

    def update_adaptive_weights(self, strategies: Dict[str, object]) -> Dict[str, float]:
        """適応型戦略の重みを更新"""
        if not self.strategy_performance:
            # パフォーマンス履歴がない場合は均等重み
            return {name: 1.0 / len(strategies) for name in strategies}

        # パフォーマンスベースの重み計算
        total_score = 0.0
        strategy_scores = {}

        for strategy_name in strategies:
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]

                # 複合スコア計算（成功率 + シャープレシオ + 最新性）
                recency_factor = 1.0
                if perf.last_updated:
                    days_old = (datetime.now() - perf.last_updated).days
                    recency_factor = max(
                        0.1, 1.0 - days_old / 365.0
                    )  # 1年で0.1まで減衰

                score = (
                    perf.success_rate * 0.4
                    + max(0, perf.sharpe_ratio) * 0.3
                    + max(0, perf.average_return) * 0.2
                    + recency_factor * 0.1
                )

                strategy_scores[strategy_name] = max(0.01, score)  # 最小重み保証
                total_score += strategy_scores[strategy_name]
            else:
                strategy_scores[strategy_name] = 0.2  # デフォルト重み
                total_score += 0.2

        # 正規化
        normalized_weights = {}
        if total_score > 0:
            for strategy_name in strategy_scores:
                normalized_weights[strategy_name] = (
                    strategy_scores[strategy_name] / total_score
                )
        else:
            # フォールバック: 均等重み
            normalized_weights = {name: 1.0 / len(strategies) for name in strategies}

        logger.debug(f"適応型重み更新: {normalized_weights}")
        return normalized_weights

    def get_performance_summary(self) -> Dict[str, float]:
        """パフォーマンスサマリーを取得"""
        if not self.strategy_performance:
            return {"avg_success_rate": 0.0, "total_strategies": 0}

        success_rates = [perf.success_rate for perf in self.strategy_performance.values()]
        return {
            "avg_success_rate": np.mean(success_rates),
            "total_strategies": len(self.strategy_performance),
            "max_success_rate": max(success_rates) if success_rates else 0.0,
            "min_success_rate": min(success_rates) if success_rates else 0.0,
        }

    def get_strategy_performance(self, strategy_name: str) -> StrategyPerformance:
        """指定戦略のパフォーマンスを取得"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(strategy_name)
        return self.strategy_performance[strategy_name]

    def reset_performance(self, strategy_name: str = None) -> None:
        """パフォーマンスをリセット"""
        if strategy_name:
            if strategy_name in self.strategy_performance:
                self.strategy_performance[strategy_name] = StrategyPerformance(
                    strategy_name
                )
        else:
            self.strategy_performance.clear()

        self._save_performance_history()
        logger.info(f"パフォーマンスリセット: {strategy_name or '全戦略'}")