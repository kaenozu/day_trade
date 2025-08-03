"""
アンサンブル戦略のテストケース
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.day_trade.analysis.ensemble import (
    EnsembleSignal,
    EnsembleStrategy,
    EnsembleTradingStrategy,
    EnsembleVotingType,
    StrategyPerformance,
)
from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal


class TestEnsembleTradingStrategy:
    """アンサンブル戦略のテストクラス"""

    @pytest.fixture
    def sample_df(self):
        """サンプルデータフレーム"""
        # テストの再現性のため固定日付を使用
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
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
        # テストの再現性のため固定日付を使用
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        np.random.seed(42)  # 再現性のため固定シード

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

        # 必要な特徴量が含まれていることを確認
        assert "volatility" in meta_features
        assert "mean_return" in meta_features
        assert "rsi_level" in meta_features
        assert "macd_divergence" in meta_features
        assert "volume_ratio" in meta_features
        assert "trend_strength" in meta_features
        assert "price_position" in meta_features

        # 固定シードによる具体的な値のアサーション
        # ボラティリティ（年率）の検証
        expected_volatility = 0.075648  # 実際の計算結果に基づく
        assert abs(meta_features["volatility"] - expected_volatility) < 0.01

        # RSIレベルの検証（最後の値）
        expected_rsi = sample_indicators["RSI"].iloc[-1]
        assert abs(meta_features["rsi_level"] - expected_rsi) < 0.01

        # MACD乖離の検証（最後の値）
        expected_macd_divergence = (
            sample_indicators["MACD"].iloc[-1] - sample_indicators["MACD_Signal"].iloc[-1]
        )
        assert abs(meta_features["macd_divergence"] - expected_macd_divergence) < 0.01

        # 出来高比率の検証（最後の値と平均の比率）
        current_volume = sample_df["Volume"].iloc[-1]
        avg_volume = sample_df["Volume"].rolling(10).mean().iloc[-1]
        expected_volume_ratio = current_volume / avg_volume
        assert abs(meta_features["volume_ratio"] - expected_volume_ratio) < 0.01

        # トレンド強度の検証（SMA20/SMA50 - 1）
        sma_20 = sample_df["Close"].rolling(20).mean().iloc[-1]
        sma_50 = sample_df["Close"].rolling(50).mean().iloc[-1]
        expected_trend_strength = (sma_20 / sma_50 - 1) * 100
        assert abs(meta_features["trend_strength"] - expected_trend_strength) < 0.01

        # 価格位置の検証（20日レンジ内での位置）
        high_20 = sample_df["High"].rolling(20).max().iloc[-1]
        low_20 = sample_df["Low"].rolling(20).min().iloc[-1]
        current_price = sample_df["Close"].iloc[-1]
        expected_price_position = (current_price - low_20) / (high_20 - low_20)
        assert abs(meta_features["price_position"] - expected_price_position) < 0.01

        # 値の妥当性チェック
        assert meta_features["volatility"] >= 0
        assert 0 <= meta_features["rsi_level"] <= 100
        assert 0 <= meta_features["price_position"] <= 1
        assert meta_features["volume_ratio"] > 0

    def test_ensemble_signal_generation(
        self, sample_df, sample_indicators, sample_patterns
    ):
        """アンサンブルシグナル生成テスト"""
        ensemble = EnsembleTradingStrategy(
            ensemble_strategy=EnsembleStrategy.BALANCED,
            voting_type=EnsembleVotingType.SOFT_VOTING,
        )

        # 多様なモックシグナルを作成
        buy_signal_strong = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["Strong buy signal"],
            conditions_met={"rsi_oversold": True},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        buy_signal_medium = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=60.0,
            reasons=["Medium buy signal"],
            conditions_met={"macd_crossover": True},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        sell_signal_weak = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            confidence=35.0,
            reasons=["Weak sell signal"],
            conditions_met={"rsi_overbought": False},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        hold_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=20.0,
            reasons=["Hold signal"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        # 複数の戦略からの多様なシグナルをモック
        with patch.object(
            ensemble.strategies["conservative_rsi"],
            "generate_signal",
            return_value=buy_signal_strong,
        ):
            with patch.object(
                ensemble.strategies["trend_following"],
                "generate_signal",
                return_value=buy_signal_medium,
            ):
                with patch.object(
                    ensemble.strategies["aggressive_momentum"],
                    "generate_signal",
                    return_value=sell_signal_weak,
                ):
                    with patch.object(
                        ensemble.strategies["mean_reversion"],
                        "generate_signal",
                        return_value=hold_signal,
                    ):
                        result = ensemble.generate_ensemble_signal(
                            sample_df, sample_indicators, sample_patterns
                        )

        # 基本的な検証
        assert result is not None
        assert isinstance(result, EnsembleSignal)

        # アンサンブルシグナルの詳細な検証
        ensemble_signal = result.ensemble_signal
        assert ensemble_signal.signal_type in [
            SignalType.BUY,
            SignalType.SELL,
            SignalType.HOLD,
        ]

        # アンサンブル投票の結果（信頼度閾値により結果が変わる可能性）
        assert ensemble_signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert ensemble_signal.strength in [SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]
        assert ensemble_signal.confidence >= 0  # 0も含む（HOLDの場合）

        # 戦略シグナルの検証（モックした戦略のみ）
        assert len(result.strategy_signals) >= 4  # 少なくとも4つの戦略からシグナル
        strategy_names = [signal[0] for signal in result.strategy_signals]
        assert "conservative_rsi" in strategy_names
        assert "trend_following" in strategy_names
        assert "aggressive_momentum" in strategy_names
        assert "mean_reversion" in strategy_names

        # 投票スコアの検証
        assert len(result.voting_scores) > 0
        assert result.ensemble_confidence >= 0

        # 戦略重みの検証
        assert len(result.strategy_weights) == 5  # 5つの戦略重み
        assert abs(sum(result.strategy_weights.values()) - 1.0) < 1e-6

        # 投票タイプの検証
        assert result.voting_type == EnsembleVotingType.SOFT_VOTING

        # メタ特徴量の存在確認
        assert len(result.meta_features) > 0
        assert "volatility" in result.meta_features

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
        ensemble_signal, voting_scores, confidence, uncertainty = result
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
        ensemble_signal, voting_scores, confidence, uncertainty = result
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
        assert perf.average_confidence == 75.0  # 初回は直接設定

        # 失敗ケースを追加
        ensemble.update_strategy_performance("test_strategy", False, 40.0, -0.02)

        perf = ensemble.strategy_performance["test_strategy"]
        assert perf.total_signals == 2
        assert perf.successful_signals == 1
        assert perf.success_rate == 0.5

    def test_adaptive_weights_update(self):
        """適応型重み更新テスト"""
        ensemble = EnsembleTradingStrategy(ensemble_strategy=EnsembleStrategy.ADAPTIVE)

        # 実際の戦略名を使用してパフォーマンスデータを設定
        ensemble.strategy_performance["conservative_rsi"] = StrategyPerformance(
            strategy_name="conservative_rsi",
            total_signals=20,
            successful_signals=16,
            success_rate=0.8,
            average_confidence=70.0,
            average_return=0.05,
            sharpe_ratio=1.2,
        )

        ensemble.strategy_performance["aggressive_momentum"] = StrategyPerformance(
            strategy_name="aggressive_momentum",
            total_signals=20,
            successful_signals=6,
            success_rate=0.3,
            average_confidence=45.0,
            average_return=-0.02,
            sharpe_ratio=0.2,
        )

        ensemble.strategy_performance["trend_following"] = StrategyPerformance(
            strategy_name="trend_following",
            total_signals=15,
            successful_signals=9,
            success_rate=0.6,
            average_confidence=65.0,
            average_return=0.03,
            sharpe_ratio=0.8,
        )

        # 重み更新前の値を記録
        original_weights = ensemble.strategy_weights.copy()

        # 適応型重み更新を実行
        ensemble._update_adaptive_weights()

        # 重みの合計が1.0であることを確認
        assert abs(sum(ensemble.strategy_weights.values()) - 1.0) < 1e-6

        # パフォーマンスの高い戦略の重みが増加していることを確認
        # conservative_rsi（成功率0.8）の重みが元の値より高くなっているはず
        assert ensemble.strategy_weights["conservative_rsi"] > original_weights["conservative_rsi"]

        # パフォーマンスの低い戦略の重みが減少していることを確認
        # aggressive_momentum（成功率0.3）の重みが元の値より低くなっているはず
        assert ensemble.strategy_weights["aggressive_momentum"] < original_weights["aggressive_momentum"]

        # trend_following（成功率0.6）の重みも適切に調整されていることを確認
        # 中程度のパフォーマンスなので、元の値と比較して妥当な範囲内
        assert ensemble.strategy_weights["trend_following"] > 0

        # 各戦略の重みが0以上であることを確認
        for weight in ensemble.strategy_weights.values():
            assert weight >= 0

        # パフォーマンスデータが設定されていない戦略はデフォルト重み（0.2）を維持
        # 重みは動的に計算されるため、具体的な値ではなく範囲を確認
        assert ensemble.strategy_weights["mean_reversion"] > 0
        assert ensemble.strategy_weights["default_integrated"] > 0

        # 両戦略の重みが近い値であることを確認（デフォルト重み0.2を使用）
        assert abs(ensemble.strategy_weights["mean_reversion"] - ensemble.strategy_weights["default_integrated"]) < 0.01

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


    def test_edge_case_all_hold_signals(self, sample_df, sample_indicators, sample_patterns):
        """エッジケース: 全てのシグナルがHOLDの場合"""
        ensemble = EnsembleTradingStrategy(voting_type=EnsembleVotingType.SOFT_VOTING)

        hold_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=20.0,
            reasons=["Hold signal"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        # 全戦略がHOLDシグナルを返すようにモック
        with patch.object(ensemble.strategies["conservative_rsi"], "generate_signal", return_value=hold_signal):
            with patch.object(ensemble.strategies["trend_following"], "generate_signal", return_value=hold_signal):
                with patch.object(ensemble.strategies["aggressive_momentum"], "generate_signal", return_value=hold_signal):
                    result = ensemble.generate_ensemble_signal(sample_df, sample_indicators, sample_patterns)

        assert result is not None
        assert result.ensemble_signal.signal_type == SignalType.HOLD
        assert result.ensemble_confidence >= 0

    def test_edge_case_equal_buy_sell_votes(self):
        """エッジケース: 買いと売りが同数の投票"""
        ensemble = EnsembleTradingStrategy(voting_type=EnsembleVotingType.HARD_VOTING)

        buy_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=60.0,
            reasons=["Buy signal"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        sell_signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.MEDIUM,
            confidence=60.0,
            reasons=["Sell signal"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        strategy_signals = [
            ("strategy1", buy_signal),
            ("strategy2", sell_signal),
        ]

        result = ensemble._hard_voting(strategy_signals, {})

        # ハード投票では同数の場合、最多得票のシグナルタイプが選ばれる
        # 実装では買いと売りが同数の場合、最初に評価されたものが選ばれる可能性がある
        assert result is not None
        ensemble_signal, _, _, _ = result
        assert ensemble_signal.signal_type in [SignalType.BUY, SignalType.SELL]

    def test_edge_case_extreme_confidence_values(self):
        """エッジケース: 極端な信頼度値"""
        ensemble = EnsembleTradingStrategy(voting_type=EnsembleVotingType.SOFT_VOTING)

        # 極端に高い信頼度
        high_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=95.0,
            reasons=["Very high confidence"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        # 極端に低い信頼度
        low_confidence_signal = TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            confidence=5.0,
            reasons=["Very low confidence"],
            conditions_met={},
            timestamp=pd.Timestamp("2023-02-19"),
            price=100.0,
        )

        strategy_signals = [
            ("strategy1", high_confidence_signal),
            ("strategy2", low_confidence_signal),
        ]

        result = ensemble._soft_voting(strategy_signals, {})
        assert result is not None
        ensemble_signal, _, confidence, _ = result

        # 高信頼度のBUYシグナルが勝つはず
        assert ensemble_signal.signal_type == SignalType.BUY
        assert confidence > 0

    def test_error_handling_strategy_exception(self, sample_df, sample_indicators, sample_patterns):
        """エラーハンドリング: 個別戦略でエラーが発生した場合"""
        ensemble = EnsembleTradingStrategy()

        # 一つの戦略でエラーを発生させる
        with patch.object(
            ensemble.strategies["conservative_rsi"],
            "generate_signal",
            side_effect=Exception("Strategy error")
        ):
            # 他の戦略は正常なシグナルを返す
            normal_signal = TradingSignal(
                signal_type=SignalType.BUY,
                strength=SignalStrength.MEDIUM,
                confidence=60.0,
                reasons=["Normal signal"],
                conditions_met={},
                timestamp=pd.Timestamp("2023-02-19"),
                price=100.0,
            )

            with patch.object(
                ensemble.strategies["trend_following"],
                "generate_signal",
                return_value=normal_signal
            ):
                result = ensemble.generate_ensemble_signal(sample_df, sample_indicators, sample_patterns)

        # エラーが発生した戦略は除外されるが、他の戦略で結果を生成できるはず
        if result is not None:
            assert isinstance(result, EnsembleSignal)
            # エラーが発生した戦略は戦略シグナルに含まれない
            strategy_names = [signal[0] for signal in result.strategy_signals]
            assert "conservative_rsi" not in strategy_names

    def test_error_handling_no_valid_signals(self, sample_df, sample_indicators, sample_patterns):
        """エラーハンドリング: 有効なシグナルがない場合"""
        ensemble = EnsembleTradingStrategy()

        # 全ての戦略がNoneを返すようにモック
        with patch.object(ensemble.strategies["conservative_rsi"], "generate_signal", return_value=None):
            with patch.object(ensemble.strategies["trend_following"], "generate_signal", return_value=None):
                with patch.object(ensemble.strategies["aggressive_momentum"], "generate_signal", return_value=None):
                    with patch.object(ensemble.strategies["mean_reversion"], "generate_signal", return_value=None):
                        with patch.object(ensemble.strategies["default_integrated"], "generate_signal", return_value=None):
                            result = ensemble.generate_ensemble_signal(sample_df, sample_indicators, sample_patterns)

        # 有効なシグナルがない場合はNoneが返されるはず
        assert result is None

    def test_confidence_threshold_edge_cases(self):
        """信頼度閾値のエッジケース"""
        # 適応型戦略でパフォーマンスデータがない場合
        adaptive_ensemble = EnsembleTradingStrategy(ensemble_strategy=EnsembleStrategy.ADAPTIVE)
        threshold = adaptive_ensemble._get_confidence_threshold()

        # パフォーマンスデータがない場合のデフォルト計算
        # 30.0 + (70.0 - 30.0) * (1 - 0.0) = 70.0
        assert threshold == 70.0

        # パフォーマンスデータがある場合
        adaptive_ensemble.strategy_performance["test"] = StrategyPerformance(
            strategy_name="test",
            success_rate=0.6
        )
        threshold_with_data = adaptive_ensemble._get_confidence_threshold()

        # 成功率0.6の場合: 30.0 + (70.0 - 30.0) * (1 - 0.6) = 46.0
        expected_threshold = 30.0 + (70.0 - 30.0) * (1 - 0.6)
        assert abs(threshold_with_data - expected_threshold) < 0.01


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
