"""
アンサンブル取引戦略
複数の戦略を組み合わせて最適化されたシグナルを生成する
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from .signals import TradingSignalGenerator, TradingSignal, SignalType, SignalStrength

logger = logging.getLogger(__name__)


class EnsembleVotingType(Enum):
    """アンサンブル投票タイプ"""

    SOFT_VOTING = "soft"  # 信頼度による重み付け投票
    HARD_VOTING = "hard"  # 多数決投票
    WEIGHTED_AVERAGE = "weighted"  # 重み付け平均


class EnsembleStrategy(Enum):
    """アンサンブル戦略タイプ"""

    CONSERVATIVE = "conservative"  # 保守的（合意重視）
    AGGRESSIVE = "aggressive"  # 積極的（機会重視）
    BALANCED = "balanced"  # バランス型
    ADAPTIVE = "adaptive"  # 適応型（パフォーマンス最適化）


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
    ):
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
        self.average_confidence = (
            1 - alpha
        ) * self.average_confidence + alpha * confidence
        self.average_return = (1 - alpha) * self.average_return + alpha * return_rate

        self.last_updated = datetime.now()


@dataclass
class EnsembleSignal:
    """アンサンブルシグナル"""

    ensemble_signal: TradingSignal
    strategy_signals: List[Tuple[str, TradingSignal]]  # (strategy_name, signal)
    voting_scores: Dict[str, float]  # 各戦略の投票スコア
    ensemble_confidence: float
    strategy_weights: Dict[str, float]  # 各戦略の重み
    voting_type: EnsembleVotingType
    meta_features: Dict[str, Any]  # メタ特徴量


class EnsembleTradingStrategy:
    """アンサンブル取引戦略"""

    def __init__(
        self,
        ensemble_strategy: EnsembleStrategy = EnsembleStrategy.BALANCED,
        voting_type: EnsembleVotingType = EnsembleVotingType.SOFT_VOTING,
        performance_file: Optional[str] = None,
    ):
        """
        Args:
            ensemble_strategy: アンサンブル戦略タイプ
            voting_type: 投票方式
            performance_file: パフォーマンス記録ファイル
        """
        self.ensemble_strategy = ensemble_strategy
        self.voting_type = voting_type
        self.performance_file = performance_file

        # 個別戦略の初期化
        self.strategies = self._initialize_strategies()

        # パフォーマンス履歴
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        self._load_performance_history()

        # 動的重み
        self.strategy_weights = self._initialize_weights()

        # メタ学習のための特徴量
        self.meta_features = {}

    def _initialize_strategies(self) -> Dict[str, TradingSignalGenerator]:
        """個別戦略を初期化"""
        strategies = {}

        # 1. 保守的RSI戦略
        conservative_strategy = TradingSignalGenerator()
        conservative_strategy.clear_rules()
        from .signals import (
            RSIOversoldRule,
            RSIOverboughtRule,
            MACDCrossoverRule,
            MACDDeathCrossRule,
        )

        conservative_strategy.add_buy_rule(RSIOversoldRule(threshold=20, weight=2.0))
        conservative_strategy.add_buy_rule(MACDCrossoverRule(weight=1.5))
        conservative_strategy.add_sell_rule(RSIOverboughtRule(threshold=80, weight=2.0))
        conservative_strategy.add_sell_rule(MACDDeathCrossRule(weight=1.5))
        strategies["conservative_rsi"] = conservative_strategy

        # 2. 積極的モメンタム戦略
        momentum_strategy = TradingSignalGenerator()
        momentum_strategy.clear_rules()
        from .signals import BollingerBandRule, PatternBreakoutRule, VolumeSpikeBuyRule

        momentum_strategy.add_buy_rule(BollingerBandRule(position="lower", weight=1.5))
        momentum_strategy.add_buy_rule(
            PatternBreakoutRule(direction="upward", weight=2.5)
        )
        momentum_strategy.add_buy_rule(VolumeSpikeBuyRule(weight=2.0))
        momentum_strategy.add_sell_rule(BollingerBandRule(position="upper", weight=1.5))
        momentum_strategy.add_sell_rule(
            PatternBreakoutRule(direction="downward", weight=2.5)
        )
        strategies["aggressive_momentum"] = momentum_strategy

        # 3. トレンドフォロー戦略
        trend_strategy = TradingSignalGenerator()
        trend_strategy.clear_rules()
        from .signals import GoldenCrossRule, DeadCrossRule

        trend_strategy.add_buy_rule(GoldenCrossRule(weight=3.0))
        trend_strategy.add_buy_rule(MACDCrossoverRule(weight=2.0))
        trend_strategy.add_sell_rule(DeadCrossRule(weight=3.0))
        trend_strategy.add_sell_rule(MACDDeathCrossRule(weight=2.0))
        strategies["trend_following"] = trend_strategy

        # 4. 平均回帰戦略
        mean_reversion_strategy = TradingSignalGenerator()
        mean_reversion_strategy.clear_rules()
        mean_reversion_strategy.add_buy_rule(RSIOversoldRule(threshold=30, weight=2.0))
        mean_reversion_strategy.add_buy_rule(
            BollingerBandRule(position="lower", weight=2.5)
        )
        mean_reversion_strategy.add_sell_rule(
            RSIOverboughtRule(threshold=70, weight=2.0)
        )
        mean_reversion_strategy.add_sell_rule(
            BollingerBandRule(position="upper", weight=2.5)
        )
        strategies["mean_reversion"] = mean_reversion_strategy

        # 5. デフォルト統合戦略
        default_strategy = TradingSignalGenerator()  # 既存のデフォルトルール
        strategies["default_integrated"] = default_strategy

        return strategies

    def _initialize_weights(self) -> Dict[str, float]:
        """戦略の初期重みを設定"""
        if self.ensemble_strategy == EnsembleStrategy.CONSERVATIVE:
            return {
                "conservative_rsi": 0.3,
                "aggressive_momentum": 0.1,
                "trend_following": 0.2,
                "mean_reversion": 0.3,
                "default_integrated": 0.1,
            }
        elif self.ensemble_strategy == EnsembleStrategy.AGGRESSIVE:
            return {
                "conservative_rsi": 0.1,
                "aggressive_momentum": 0.35,
                "trend_following": 0.3,
                "mean_reversion": 0.15,
                "default_integrated": 0.1,
            }
        elif self.ensemble_strategy == EnsembleStrategy.BALANCED:
            return {
                "conservative_rsi": 0.2,
                "aggressive_momentum": 0.25,
                "trend_following": 0.25,
                "mean_reversion": 0.2,
                "default_integrated": 0.1,
            }
        else:  # ADAPTIVE
            # 初期は均等、パフォーマンスに基づいて動的調整
            return {name: 0.2 for name in self.strategies.keys()}

    def _load_performance_history(self):
        """パフォーマンス履歴をロード"""
        if not self.performance_file:
            return

        try:
            performance_path = Path(self.performance_file)
            if performance_path.exists():
                with open(performance_path, "r", encoding="utf-8") as f:
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

    def _save_performance_history(self):
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

    def generate_ensemble_signal(
        self,
        df: pd.DataFrame,
        indicators: Optional[pd.DataFrame] = None,
        patterns: Optional[Dict] = None,
    ) -> Optional[EnsembleSignal]:
        """
        アンサンブルシグナルを生成

        Args:
            df: 価格データのDataFrame
            indicators: テクニカル指標のDataFrame
            patterns: チャートパターン認識結果

        Returns:
            EnsembleSignal or None
        """
        try:
            # 各戦略からシグナルを取得
            strategy_signals = []
            for strategy_name, strategy in self.strategies.items():
                try:
                    signal = strategy.generate_signal(df, indicators, patterns)
                    if signal:
                        strategy_signals.append((strategy_name, signal))
                except Exception as e:
                    logger.warning(f"戦略 {strategy_name} でエラー: {e}")

            if not strategy_signals:
                return None

            # メタ特徴量を計算
            meta_features = self._calculate_meta_features(df, indicators, patterns)

            # 動的重み調整（適応型の場合）
            if self.ensemble_strategy == EnsembleStrategy.ADAPTIVE:
                self._update_adaptive_weights()

            # アンサンブル投票を実行
            ensemble_result = self._perform_ensemble_voting(
                strategy_signals, meta_features
            )

            if ensemble_result:
                ensemble_signal, voting_scores, ensemble_confidence = ensemble_result

                return EnsembleSignal(
                    ensemble_signal=ensemble_signal,
                    strategy_signals=strategy_signals,
                    voting_scores=voting_scores,
                    ensemble_confidence=ensemble_confidence,
                    strategy_weights=self.strategy_weights.copy(),
                    voting_type=self.voting_type,
                    meta_features=meta_features,
                )

            return None

        except Exception as e:
            logger.error(f"アンサンブルシグナル生成エラー: {e}")
            return None

    def _calculate_meta_features(
        self, df: pd.DataFrame, indicators: pd.DataFrame, patterns: Dict
    ) -> Dict[str, Any]:
        """メタ特徴量を計算"""
        try:
            meta_features = {}

            # 市場状況の特徴量
            if len(df) >= 20:
                # ボラティリティ
                returns = df["Close"].pct_change().dropna()
                if len(returns) > 0:
                    meta_features["volatility"] = returns.std() * np.sqrt(
                        252
                    )  # 年率ボラティリティ
                    meta_features["mean_return"] = returns.mean()

                # トレンド強度
                if len(df) >= 50:
                    sma_20 = df["Close"].rolling(20).mean()
                    sma_50 = df["Close"].rolling(50).mean()
                    if not sma_20.empty and not sma_50.empty:
                        meta_features["trend_strength"] = (
                            sma_20.iloc[-1] / sma_50.iloc[-1] - 1
                        ) * 100

                # 価格位置（過去のレンジ内での位置）
                high_20 = df["High"].rolling(20).max().iloc[-1]
                low_20 = df["Low"].rolling(20).min().iloc[-1]
                current_price = df["Close"].iloc[-1]
                if high_20 != low_20:
                    meta_features["price_position"] = (current_price - low_20) / (
                        high_20 - low_20
                    )

            # テクニカル指標の状況
            if indicators is not None and not indicators.empty:
                if "RSI" in indicators.columns:
                    meta_features["rsi_level"] = indicators["RSI"].iloc[-1]

                if "MACD" in indicators.columns and "MACD_Signal" in indicators.columns:
                    macd_diff = (
                        indicators["MACD"].iloc[-1] - indicators["MACD_Signal"].iloc[-1]
                    )
                    meta_features["macd_divergence"] = macd_diff

            # 出来高の特徴量
            if "Volume" in df.columns and len(df) >= 10:
                avg_volume = df["Volume"].rolling(10).mean().iloc[-1]
                current_volume = df["Volume"].iloc[-1]
                meta_features["volume_ratio"] = (
                    current_volume / avg_volume if avg_volume > 0 else 1.0
                )

            return meta_features

        except Exception as e:
            logger.error(f"メタ特徴量計算エラー: {e}")
            return {}

    def _perform_ensemble_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float]]:
        """アンサンブル投票を実行"""
        try:
            if self.voting_type == EnsembleVotingType.SOFT_VOTING:
                return self._soft_voting(strategy_signals, meta_features)
            elif self.voting_type == EnsembleVotingType.HARD_VOTING:
                return self._hard_voting(strategy_signals, meta_features)
            else:  # WEIGHTED_AVERAGE
                return self._weighted_average_voting(strategy_signals, meta_features)

        except Exception as e:
            logger.error(f"アンサンブル投票エラー: {e}")
            return None

    def _soft_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float]]:
        """ソフト投票（信頼度による重み付け投票）"""
        voting_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_weight = 0.0
        strategy_contributions = {}

        for strategy_name, signal in strategy_signals:
            _strategy_weight = self.strategy_weights.get(
                strategy_name, 0.2
            )  # Renamed variable

            # パフォーマンスによる重み調整
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                performance_multiplier = 0.5 + perf.success_rate  # 0.5-1.5の範囲
                _strategy_weight *= performance_multiplier  # Renamed variable

            weighted_confidence = (
                signal.confidence * _strategy_weight
            )  # Renamed variable
            voting_scores[signal.signal_type.value] += weighted_confidence
            total_weight += _strategy_weight  # Renamed variable

            strategy_contributions[strategy_name] = weighted_confidence

        if total_weight == 0:
            return None

        # 正規化
        for signal_type in voting_scores:
            voting_scores[signal_type] /= total_weight

        # 最高スコアのシグナルタイプを決定
        best_signal_type = max(voting_scores, key=voting_scores.get)
        best_score = voting_scores[best_signal_type]

        # 閾値チェック
        confidence_threshold = self._get_confidence_threshold()
        if best_score < confidence_threshold:
            best_signal_type = "hold"
            best_score = 0.0

        # 強度を決定
        if best_score >= 70:
            strength = SignalStrength.STRONG
        elif best_score >= 40:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        # 理由をまとめる
        reasons = []
        for strategy_name, signal in strategy_signals:
            if signal.signal_type.value == best_signal_type:
                reasons.extend(
                    [f"{strategy_name}: {reason}" for reason in signal.reasons]
                )

        if not reasons:
            reasons = [f"アンサンブル投票結果: {best_signal_type}"]

        # 最新の価格とタイムスタンプ
        latest_signal = strategy_signals[0][1]  # 最初のシグナルから取得

        ensemble_signal = TradingSignal(
            signal_type=SignalType(best_signal_type),
            strength=strength,
            confidence=best_score,
            reasons=reasons,
            conditions_met={},
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
        )

        return ensemble_signal, strategy_contributions, best_score

    def _hard_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float]]:
        """ハード投票（多数決投票）"""
        vote_counts = {"buy": 0, "sell": 0, "hold": 0}
        strategy_contributions = {}

        for strategy_name, signal in strategy_signals:
            _strategy_weight = self.strategy_weights.get(
                strategy_name, 0.2
            )  # Renamed variable

            # パフォーマンスによる重み調整
            if strategy_name in self.strategy_performance:
                perf = self.strategy_performance[strategy_name]
                if perf.success_rate < 0.3:  # 成功率が低い戦略は投票権を減らす
                    continue

            vote_counts[signal.signal_type.value] += 1
            strategy_contributions[strategy_name] = 1.0

        if sum(vote_counts.values()) == 0:
            return None

        # 最多得票のシグナルタイプを決定
        best_signal_type = max(vote_counts, key=vote_counts.get)
        vote_count = vote_counts[best_signal_type]

        # 過半数を取得した場合のみ有効
        total_votes = sum(vote_counts.values())
        if vote_count / total_votes < 0.5:
            best_signal_type = "hold"

        # 信頼度は参加戦略の平均信頼度
        confidences = [
            signal.confidence
            for strategy_name, signal in strategy_signals
            if signal.signal_type.value == best_signal_type
        ]
        ensemble_confidence = np.mean(confidences) if confidences else 0.0

        # 強度を決定（投票数に基づく）
        if vote_count >= len(strategy_signals) * 0.8:
            strength = SignalStrength.STRONG
        elif vote_count >= len(strategy_signals) * 0.6:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        # 理由をまとめる
        reasons = [f"多数決投票: {vote_count}/{total_votes} 票獲得"]

        # 最新の価格とタイムスタンプ
        latest_signal = strategy_signals[0][1]

        ensemble_signal = TradingSignal(
            signal_type=SignalType(best_signal_type),
            strength=strength,
            confidence=ensemble_confidence,
            reasons=reasons,
            conditions_met={},
            timestamp=latest_signal.timestamp,
            price=latest_signal.price,
        )

        return ensemble_signal, strategy_contributions, ensemble_confidence

    def _weighted_average_voting(
        self,
        strategy_signals: List[Tuple[str, TradingSignal]],
        meta_features: Dict[str, Any],
    ) -> Optional[Tuple[TradingSignal, Dict[str, float], float]]:
        """重み付け平均投票"""
        # ソフト投票の変種として実装
        return self._soft_voting(strategy_signals, meta_features)

    def _get_confidence_threshold(self) -> float:
        """信頼度閾値を取得"""
        if self.ensemble_strategy == EnsembleStrategy.CONSERVATIVE:
            return 60.0
        elif self.ensemble_strategy == EnsembleStrategy.AGGRESSIVE:
            return 30.0
        elif self.ensemble_strategy == EnsembleStrategy.BALANCED:
            return 45.0
        else:  # ADAPTIVE
            # 過去のパフォーマンスに基づいて動的調整
            avg_success_rate = (
                np.mean(
                    [perf.success_rate for perf in self.strategy_performance.values()]
                )
                if self.strategy_performance
                else 0.0
            )
            return 30.0 + (70.0 - 30.0) * (1 - avg_success_rate)

    def _update_adaptive_weights(self):
        """適応型戦略の重みを更新"""
        if not self.strategy_performance:
            return

        # パフォーマンスベースの重み計算
        total_score = 0.0
        strategy_scores = {}

        for strategy_name in self.strategies.keys():
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
        if total_score > 0:
            for strategy_name in strategy_scores:
                self.strategy_weights[strategy_name] = (
                    strategy_scores[strategy_name] / total_score
                )

        logger.debug(f"適応型重み更新: {self.strategy_weights}")

    def update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        confidence: float,
        return_rate: float = 0.0,
    ):
        """戦略パフォーマンスを更新"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = StrategyPerformance(
                strategy_name
            )

        self.strategy_performance[strategy_name].update_performance(
            success, confidence, return_rate
        )
        self._save_performance_history()

    def get_strategy_summary(self) -> Dict[str, Any]:
        """戦略サマリーを取得"""
        return {
            "ensemble_strategy": self.ensemble_strategy.value,
            "voting_type": self.voting_type.value,
            "strategy_weights": self.strategy_weights,
            "strategy_count": len(self.strategies),
            "performance_records": len(self.strategy_performance),
            "avg_success_rate": (
                np.mean(
                    [perf.success_rate for perf in self.strategy_performance.values()]
                )
                if self.strategy_performance
                else 0.0
            ),
        }


# 使用例
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime

    # サンプルデータ作成
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    np.random.seed(42)

    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close_prices = trend + noise

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close_prices + np.random.randn(100) * 0.5,
            "High": close_prices + np.abs(np.random.randn(100)) * 2,
            "Low": close_prices - np.abs(np.random.randn(100)) * 2,
            "Close": close_prices,
            "Volume": np.random.randint(1000000, 5000000, 100),
        }
    )
    df.set_index("Date", inplace=True)

    # アンサンブル戦略テスト
    ensemble = EnsembleTradingStrategy(
        ensemble_strategy=EnsembleStrategy.BALANCED,
        voting_type=EnsembleVotingType.SOFT_VOTING,
    )

    # シグナル生成
    ensemble_signal = ensemble.generate_ensemble_signal(df)

    if ensemble_signal:
        signal = ensemble_signal.ensemble_signal
        print(f"アンサンブルシグナル: {signal.signal_type.value.upper()}")
        print(f"強度: {signal.strength.value}")
        print(f"信頼度: {signal.confidence:.1f}%")
        print(f"価格: {signal.price:.2f}")

        print("\n戦略別貢献度:")
        for strategy_name, score in ensemble_signal.voting_scores.items():
            print(f"  {strategy_name}: {score:.2f}")

        print("\n戦略重み:")
        for strategy_name, weight in ensemble_signal.strategy_weights.items():
            print(f"  {strategy_name}: {weight:.2f}")

        print("\nメタ特徴量:")
        for feature, value in ensemble_signal.meta_features.items():
            print(f"  {feature}: {value}")
    else:
        print("アンサンブルシグナルなし")

    # 戦略サマリー
    print("\n戦略サマリー:")
    summary = ensemble.get_strategy_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
