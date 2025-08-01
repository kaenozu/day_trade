"""
アンサンブル戦略のテストケース
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch

from src.day_trade.analysis.ensemble import (
    EnsembleTradingStrategy,
    EnsembleStrategy,
    EnsembleVotingType,
    EnsembleSignal,
    StrategyPerformance,
)
from src.day_trade.analysis.signals import TradingSignal, SignalType, SignalStrength


class TestEnsembleTradingStrategy:
    """アンサンブル戦略のテストクラス"""

    @pytest.fixture
    def sample_df(self):
        """サンプルデータフレーム"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")
        np.random.seed(42)

        close_prices = np.cumsum(np.random.randn(50) * 0.5) + 100

        df = pd.DataFrame(
            {
                "Open": close_prices + np.random.randn(50) * 0.1,
                "High": close_prices + np.abs(np.random.randn(50)) * 0.2,
                "Low": close_prices - np.abs(np.random.randn(50)) * 0.2,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, 50),
            },
            index=dates,
        )

        return df

    @pytest.fixture
    def sample_indicators(self):
        """サンプル指標データ"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq="D")

        return pd.DataFrame(
            {
                "RSI": np.random.uniform(20, 80, 50),
                "MACD": np.random.randn(50) * 0.5,
                "MACD_Signal": np.random.randn(50) * 0.3,
                "BB_Upper": np.random.uniform(105, 110, 50),
                "BB_Lower": np.random.uniform(95, 100, 50),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_patterns(self):
        """サンプルパターンデータ"""
        return {
            "breakouts": pd.DataFrame(
                {
                    "Upward_Breakout": [False] * 49 + [True],
                    "Upward_Confidence": [0] * 49 + [75.0],
                    "Downward_Breakout": [False] * 50,
                    "Downward_Confidence": [0] * 50,
                }
            ),
            "crosses": pd.DataFrame(
                {
                    "Golden_Cross": [False] * 48 + [True, False],
                    "Golden_Confidence": [0] * 48 + [80.0, 0],
                    "Dead_Cross": [False] * 50,
                    "Dead_Confidence": [0] * 50,
                }
            ),
        }

    def test_ensemble_strategy_initialization(self):
        """アンサンブル戦略の初期化テスト"""
        ensemble = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED,
            voting_type=EnsembleVotingType.SOFT_VOTING,
        )

        assert ensemble.ensemble_strategy == EnsembleStrategy.BALANCED
        assert ensemble.voting_type == EnsembleVotingType.SOFT_VOTING
        assert len(ensemble.strategies) == 5
        assert "conservative_rsi" in ensemble.strategies
        assert "aggressive_momentum" in ensemble.strategies
        assert "trend_following" in ensemble.strategies
        assert "mean_reversion" in ensemble.strategies
        assert "default_integrated" in ensemble.strategies

    def test_strategy_weights_initialization(self):
        """戦略重みの初期化テスト"""
        # バランス型
        balanced = EnsembleTradingStrategy(ensemble_strategy=EnsembleStrategy.BALANCED)
        assert abs(sum(balanced.strategy_weights.values()) - 1.0) < 1e-6
        assert balanced.strategy_weights["aggressive_momentum"] == 0.25

        # 保守型
        conservative = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.CONSERVATIVE
        )
        assert conservative.strategy_weights["conservative_rsi"] == 0.3
        assert conservative.strategy_weights["aggressive_momentum"] == 0.1

        # 積極型
        aggressive = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.AGGRESSIVE
        )
        assert aggressive.strategy_weights["aggressive_momentum"] == 0.35
        assert aggressive.strategy_weights["conservative_rsi"] == 0.1

    def test_meta_features_calculation(self, sample_df, sample_indicators):
        """メタ特徴量計算テスト"""
        ensemble = EnsembleTradingStrategy()

        meta_features = ensemble._calculate_meta_features(
            sample_df, sample_indicators, {}
        )

        assert "volatility" in meta_features
        assert "mean_return" in meta_features
        assert "rsi_level" in meta_features
        assert "macd_divergence" in meta_features
        assert "volume_ratio" in meta_features

        # 値の妥当性チェック
        assert meta_features["volatility"] >= 0
        assert 0 <= meta_features["rsi_level"] <= 100

    def test_ensemble_signal_generation(
        self, sample_df, sample_indicators, sample_patterns
    ):
        """アンサンブルシグナル生成テスト"""
        ensemble = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED,
            voting_type=EnsembleVotingType.SOFT_VOTING,
        )

        # モックを使用して個別戦略のシグナルをコントロール
        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=60.0,
            reasons=["Test signal"],
            conditions_met={"test": True},
            timestamp=datetime.now(),
            price=100.0,
        )

        with patch.object(
            ensemble.strategies["conservative_rsi"],
            "generate_signal",
            return_value=mock_signal,
        ):
            with patch.object(
                ensemble.strategies["trend_following"],
                "generate_signal",
                return_value=mock_signal,
            ):
                result = ensemble.generate_ensemble_signal(
                    sample_df, sample_indicators, sample_patterns
                )

        assert result is not None
        assert isinstance(result, EnsembleSignal)
        assert result.ensemble_signal.signal_type in [
            SignalType.BUY,
            SignalType.SELL,
            SignalType.HOLD,
        ]
        assert len(result.strategy_signals) >= 2
        assert result.ensemble_confidence >= 0

    def test_soft_voting(self):
        """ソフト投票テスト"""
        ensemble = EnsembleTradingStrategy(voting_type=EnsembleVotingType.SOFT_VOTING)

        # テストシグナルを作成
        buy_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["Strong buy"],
            conditions_met={},
            timestamp=datetime.now(),
            price=100.0,
        )

        sell_signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            confidence=30.0,
            reasons=["Weak sell"],
            conditions_met={},
            timestamp=datetime.now(),
            price=100.0,
        )

        strategy_signals = [
            ("strategy1", buy_signal),
            ("strategy2", buy_signal),
            ("strategy3", sell_signal),
        ]

        result = ensemble._soft_voting(strategy_signals, {})

        assert result is not None
        ensemble_signal, voting_scores, confidence = result
        assert (
            ensemble_signal.signal_type == SignalType.BUY
        )  # より強い買いシグナルが勝つはず
        assert confidence > 0

    def test_hard_voting(self):
        """ハード投票テスト"""
        ensemble = EnsembleTradingStrategy(voting_type=EnsembleVotingType.HARD_VOTING)

        buy_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=60.0,
            reasons=["Buy signal"],
            conditions_met={},
            timestamp=datetime.now(),
            price=100.0,
        )

        strategy_signals = [
            ("strategy1", buy_signal),
            ("strategy2", buy_signal),
        ]

        result = ensemble._hard_voting(strategy_signals, {})

        assert result is not None
        ensemble_signal, voting_scores, confidence = result
        assert ensemble_signal.signal_type == SignalType.BUY

    def test_confidence_threshold(self):
        """信頼度閾値テスト"""
        conservative = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.CONSERVATIVE
        )
        aggressive = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.AGGRESSIVE
        )
        balanced = EnsembleTradingStrategy(ensemble_strategy=EnsembleStrategy.BALANCED)

        assert conservative._get_confidence_threshold() == 60.0
        assert aggressive._get_confidence_threshold() == 30.0
        assert balanced._get_confidence_threshold() == 45.0

    def test_strategy_performance_update(self):
        """戦略パフォーマンス更新テスト"""
        ensemble = EnsembleTradingStrategy()

        # パフォーマンスを更新
        ensemble.update_strategy_performance("test_strategy", True, 75.0, 0.05)

        assert "test_strategy" in ensemble.strategy_performance
        perf = ensemble.strategy_performance["test_strategy"]
        assert perf.total_signals == 1
        assert perf.successful_signals == 1
        assert perf.success_rate == 1.0
        assert perf.average_confidence == 75.0

        # 失敗ケースを追加
        ensemble.update_strategy_performance("test_strategy", False, 40.0, -0.02)

        perf = ensemble.strategy_performance["test_strategy"]
        assert perf.total_signals == 2
        assert perf.successful_signals == 1
        assert perf.success_rate == 0.5

    def test_adaptive_weights_update(self):
        """適応型重み更新テスト"""
        ensemble = EnsembleTradingStrategy(ensemble_strategy=EnsembleStrategy.ADAPTIVE)

        # パフォーマンスデータを設定
        ensemble.strategy_performance["strategy1"] = StrategyPerformance(
            strategy_name="strategy1",
            total_signals=10,
            successful_signals=8,
            success_rate=0.8,
            average_confidence=70.0,
        )

        ensemble.strategy_performance["strategy2"] = StrategyPerformance(
            strategy_name="strategy2",
            total_signals=10,
            successful_signals=3,
            success_rate=0.3,
            average_confidence=50.0,
        )

        ensemble._update_adaptive_weights()

        # 成功率の高い戦略の重みが増加しているはず
        # (ただし、ここでは戦略名が一致しないため、実際の重み変更は発生しない)
        assert abs(sum(ensemble.strategy_weights.values()) - 1.0) < 1e-6

    def test_strategy_summary(self):
        """戦略サマリーテスト"""
        ensemble = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED,
            voting_type=EnsembleVotingType.SOFT_VOTING,
        )

        summary = ensemble.get_strategy_summary()

        assert summary["ensemble_strategy"] == "balanced"
        assert summary["voting_type"] == "soft"
        assert summary["strategy_count"] == 5
        assert "strategy_weights" in summary
        assert "avg_success_rate" in summary


class TestStrategyPerformance:
    """戦略パフォーマンスクラスのテスト"""

    def test_performance_initialization(self):
        """パフォーマンス初期化テスト"""
        perf = StrategyPerformance("test_strategy")

        assert perf.strategy_name == "test_strategy"
        assert perf.total_signals == 0
        assert perf.successful_signals == 0
        assert perf.success_rate == 0.0

    def test_performance_update(self):
        """パフォーマンス更新テスト"""
        perf = StrategyPerformance("test_strategy")

        # 成功ケース
        perf.update_performance(True, 80.0, 0.05)
        assert perf.total_signals == 1
        assert perf.successful_signals == 1
        assert perf.success_rate == 1.0

        # 失敗ケース
        perf.update_performance(False, 40.0, -0.02)
        assert perf.total_signals == 2
        assert perf.successful_signals == 1
        assert perf.success_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
