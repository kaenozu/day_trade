"""
予測オーケストレーターのテスト
統合予測システムの包括的なテスト
"""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.day_trade.analysis.prediction_orchestrator import (
    PredictionConfig,
    PredictionOrchestrator,
)
from src.day_trade.analysis.signals import SignalStrength, SignalType, TradingSignal


class TestPredictionOrchestrator:
    """予測オーケストレーターテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        np.random.seed(42)

        # トレンドのある価格データ
        trend = np.linspace(100, 150, len(dates))
        noise = np.random.randn(len(dates)) * 3
        close_prices = trend + noise

        return pd.DataFrame(
            {
                "Date": dates,
                "Open": close_prices + np.random.randn(len(dates)) * 1,
                "High": close_prices + np.abs(np.random.randn(len(dates))) * 2,
                "Low": close_prices - np.abs(np.random.randn(len(dates))) * 2,
                "Close": close_prices,
                "Volume": np.random.randint(1000000, 5000000, len(dates)),
            }
        ).set_index("Date")

    @pytest.fixture
    def prediction_config(self):
        """予測設定"""
        return PredictionConfig(
            prediction_horizon=5,
            min_data_length=100,
            feature_selection_top_k=20,
            confidence_threshold=0.6,
            enable_ensemble_ml=True,
            enable_adaptive_weighting=True,
        )

    @pytest.fixture
    def orchestrator_no_ml(self, prediction_config):
        """機械学習なしのオーケストレーター"""
        return PredictionOrchestrator(prediction_config, enable_ml=False)

    @pytest.fixture
    def orchestrator_with_ml(self, prediction_config):
        """機械学習ありのオーケストレーター（モック）"""
        with patch("src.day_trade.analysis.prediction_orchestrator.MLModelManager"):
            orchestrator = PredictionOrchestrator(prediction_config, enable_ml=True)
            orchestrator.ml_manager = Mock()
            orchestrator.ml_manager.list_models.return_value = [
                "enhanced_return_predictor",
                "enhanced_direction_predictor",
                "volatility_forecaster",
            ]
            return orchestrator

    def test_orchestrator_initialization(self, prediction_config):
        """オーケストレーター初期化テスト"""
        orchestrator = PredictionOrchestrator(prediction_config, enable_ml=False)

        assert orchestrator.config == prediction_config
        assert not orchestrator.enable_ml
        assert orchestrator.feature_engineer is not None
        assert orchestrator.ensemble_strategy is not None
        assert orchestrator.prediction_history == []
        assert len(orchestrator.performance_metrics) > 0

    def test_orchestrator_initialization_with_ml(self, prediction_config):
        """機械学習ありの初期化テスト"""
        with patch("src.day_trade.analysis.prediction_orchestrator.MLModelManager"):
            orchestrator = PredictionOrchestrator(prediction_config, enable_ml=True)

            assert orchestrator.enable_ml
            assert orchestrator.ml_manager is not None

    def test_generate_enhanced_prediction_no_ml(self, orchestrator_no_ml, sample_data):
        """機械学習なしでの統合予測生成テスト"""
        # アンサンブル戦略のモック
        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=75.0,
            reasons=["テスト理由"],
            conditions_met={},
            timestamp=datetime.now(),
            price=120.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 75.0
        mock_ensemble_signal.ensemble_uncertainty = 0.2

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            prediction = orchestrator_no_ml.generate_enhanced_prediction(sample_data)

            assert prediction is not None
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "uncertainty" in prediction
            assert "recommendation" in prediction
            assert "timestamp" in prediction

            # 予測内容の検証
            pred = prediction["prediction"]
            assert "price_direction" in pred
            assert "expected_return" in pred
            assert "confidence" in pred
            assert -1.0 <= pred["price_direction"] <= 1.0
            assert 0.0 <= pred["confidence"] <= 1.0

    def test_generate_enhanced_prediction_with_ml(
        self, orchestrator_with_ml, sample_data
    ):
        """機械学習ありでの統合予測生成テスト"""
        # ML予測のモック
        orchestrator_with_ml.ml_manager.models = {
            "enhanced_return_predictor": Mock(is_fitted=True),
            "enhanced_direction_predictor": Mock(is_fitted=True),
            "volatility_forecaster": Mock(is_fitted=True),
        }

        orchestrator_with_ml.ml_manager.predict.side_effect = [
            [0.05],  # リターン予測
            [0.7],  # 方向性予測
            [0.3],  # ボラティリティ予測
        ]

        # アンサンブルシグナルのモック
        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=80.0,
            reasons=["強い買いシグナル"],
            conditions_met={},
            timestamp=datetime.now(),
            price=125.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 80.0
        mock_ensemble_signal.ensemble_uncertainty = 0.15

        with patch.object(
            orchestrator_with_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            # モデル重みを設定
            orchestrator_with_ml.model_weights = {
                "enhanced_return_predictor": 0.8,
                "enhanced_direction_predictor": 0.9,
                "volatility_forecaster": 0.7,
            }

            prediction = orchestrator_with_ml.generate_enhanced_prediction(sample_data)

            assert prediction is not None
            assert "ml_predictions" in prediction
            assert "ensemble_signal" in prediction

            # ML予測の検証
            ml_preds = prediction["ml_predictions"]
            assert len(ml_preds) > 0

            # 統合予測の検証
            pred = prediction["prediction"]
            assert pred["price_direction"] != 0.0  # ML + アンサンブルの統合効果
            assert pred["confidence"] > 0.0

    def test_insufficient_data_handling(self, orchestrator_no_ml):
        """データ不足時の処理テスト"""
        # 短すぎるデータ
        short_data = pd.DataFrame(
            {"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]}
        )

        prediction = orchestrator_no_ml.generate_enhanced_prediction(short_data)
        assert prediction is None

    def test_risk_adjustment_high_volatility(self, orchestrator_no_ml, sample_data):
        """高ボラティリティ時のリスク調整テスト"""
        # 高ボラティリティデータを作成
        volatile_data = sample_data.copy()
        volatile_data.loc[volatile_data.index[-20:], "Close"] = volatile_data.loc[
            volatile_data.index[-20:], "Close"
        ] * (1 + np.random.randn(20) * 0.1)

        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=90.0,
            reasons=["強い買いシグナル"],
            conditions_met={},
            timestamp=datetime.now(),
            price=140.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 90.0
        mock_ensemble_signal.ensemble_uncertainty = 0.1

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            prediction = orchestrator_no_ml.generate_enhanced_prediction(volatile_data)

            assert prediction is not None
            # 高ボラティリティによるリスク調整が適用されることを確認
            pred = prediction["prediction"]
            assert pred["confidence"] < 0.9  # 元の90%より下がっているはず

    def test_risk_factors_identification(self, orchestrator_no_ml, sample_data):
        """リスク要因特定テスト"""
        # 低流動性データを作成
        low_liquidity_data = sample_data.copy()
        low_liquidity_data.iloc[-1, low_liquidity_data.columns.get_loc("Volume")] = (
            100000  # 平均より大幅に少ない
        )

        mock_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=50.0,
            reasons=["中立"],
            conditions_met={},
            timestamp=datetime.now(),
            price=130.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 50.0
        mock_ensemble_signal.ensemble_uncertainty = 0.2

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            prediction = orchestrator_no_ml.generate_enhanced_prediction(
                low_liquidity_data
            )

            assert prediction is not None
            risk_factors = prediction["risk_factors"]
            assert isinstance(risk_factors, list)
            # 低流動性が検出されることを期待
            assert any("流動性" in factor for factor in risk_factors)

    def test_recommendation_generation(self, orchestrator_no_ml, sample_data):
        """推奨アクション生成テスト"""
        # 強い買いシグナルのテスト
        strong_buy_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=85.0,
            reasons=["強い上昇トレンド"],
            conditions_met={},
            timestamp=datetime.now(),
            price=135.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = strong_buy_signal
        mock_ensemble_signal.ensemble_confidence = 85.0
        mock_ensemble_signal.ensemble_uncertainty = 0.1

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            prediction = orchestrator_no_ml.generate_enhanced_prediction(sample_data)

            assert prediction is not None
            recommendation = prediction["recommendation"]

            assert recommendation["action"] in ["BUY", "SELL", "HOLD"]
            assert recommendation["strength"] in ["STRONG", "MEDIUM", "WEAK"]
            assert 0.0 <= recommendation["confidence"] <= 1.0
            assert 0.0 <= recommendation["position_size_suggestion"] <= 1.0
            assert isinstance(recommendation["reason"], str)

    def test_prediction_uncertainty_calculation(
        self, orchestrator_with_ml, sample_data
    ):
        """予測不確実性計算テスト"""
        # 分散の大きいML予測を設定
        orchestrator_with_ml.ml_manager.models = {
            "model1": Mock(is_fitted=True),
            "model2": Mock(is_fitted=True),
            "model3": Mock(is_fitted=True),
        }

        orchestrator_with_ml.ml_manager.predict.side_effect = [
            [0.1],  # 大きく異なる予測値
            [-0.05],
            [0.08],
        ]

        mock_signal = TradingSignal(
            signal_type=SignalType.HOLD,
            strength=SignalStrength.WEAK,
            confidence=50.0,
            reasons=["中立"],
            conditions_met={},
            timestamp=datetime.now(),
            price=130.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 50.0
        mock_ensemble_signal.ensemble_uncertainty = 0.3  # 高い不確実性

        with patch.object(
            orchestrator_with_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            orchestrator_with_ml.model_weights = {
                "model1": 0.3,
                "model2": 0.4,
                "model3": 0.3,
            }

            prediction = orchestrator_with_ml.generate_enhanced_prediction(sample_data)

            assert prediction is not None
            uncertainty = prediction["uncertainty"]
            assert 0.0 <= uncertainty <= 1.0
            # 分散の大きい予測により不確実性が高くなることを確認
            assert uncertainty > 0.1

    def test_position_size_suggestion(self, orchestrator_no_ml, sample_data):
        """ポジションサイズ提案テスト"""
        # 高信頼度・低不確実性のケース
        high_confidence_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=90.0,
            reasons=["極めて強い買いシグナル"],
            conditions_met={},
            timestamp=datetime.now(),
            price=140.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = high_confidence_signal
        mock_ensemble_signal.ensemble_confidence = 90.0
        mock_ensemble_signal.ensemble_uncertainty = 0.05  # 低い不確実性

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            prediction = orchestrator_no_ml.generate_enhanced_prediction(sample_data)

            assert prediction is not None
            position_size = prediction["recommendation"]["position_size_suggestion"]

            # 高信頼度・低不確実性では大きめのポジションサイズを提案
            assert position_size > 0.1
            assert position_size <= 0.2  # 最大制限内

    def test_system_status_reporting(self, orchestrator_no_ml):
        """システム状態レポートテスト"""
        status = orchestrator_no_ml.get_system_status()

        assert "orchestrator_config" in status
        assert "ml_enabled" in status
        assert "prediction_history_length" in status
        assert "model_weights" in status
        assert "performance_metrics" in status

        # 設定値の確認
        config = status["orchestrator_config"]
        assert (
            config["prediction_horizon"] == orchestrator_no_ml.config.prediction_horizon
        )
        assert (
            config["confidence_threshold"]
            == orchestrator_no_ml.config.confidence_threshold
        )

        # ML無効の確認
        assert not status["ml_enabled"]

    def test_prediction_history_management(self, orchestrator_no_ml, sample_data):
        """予測履歴管理テスト"""
        initial_history_length = len(orchestrator_no_ml.prediction_history)

        mock_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            confidence=70.0,
            reasons=["テスト"],
            conditions_met={},
            timestamp=datetime.now(),
            price=125.0,
        )

        mock_ensemble_signal = Mock()
        mock_ensemble_signal.ensemble_signal = mock_signal
        mock_ensemble_signal.ensemble_confidence = 70.0
        mock_ensemble_signal.ensemble_uncertainty = 0.2

        with patch.object(
            orchestrator_no_ml.ensemble_strategy,
            "generate_ensemble_signal",
            return_value=mock_ensemble_signal,
        ):
            # 複数回予測を実行
            for _ in range(3):
                prediction = orchestrator_no_ml.generate_enhanced_prediction(
                    sample_data
                )
                assert prediction is not None

            # 履歴が増加していることを確認
            assert (
                len(orchestrator_no_ml.prediction_history) == initial_history_length + 3
            )

            # 履歴の内容確認
            latest_record = orchestrator_no_ml.prediction_history[-1]
            assert "timestamp" in latest_record
            assert "price" in latest_record
            assert "prediction" in latest_record
            assert "uncertainty" in latest_record

    def test_error_handling(self, orchestrator_no_ml):
        """エラーハンドリングテスト"""
        # 無効なデータでの予測
        invalid_data = pd.DataFrame()

        prediction = orchestrator_no_ml.generate_enhanced_prediction(invalid_data)
        assert prediction is None

        # NaNを含むデータでの予測
        nan_data = pd.DataFrame(
            {
                "Close": [100, np.nan, 102, 103, 104],
                "Volume": [1000, 1100, np.nan, 1300, 1400],
            }
        )

        prediction = orchestrator_no_ml.generate_enhanced_prediction(nan_data)
        # エラーが発生してもクラッシュしないことを確認
        # 結果は None または 有効な辞書
        assert prediction is None or isinstance(prediction, dict)

    def test_optimized_model_params(self, orchestrator_no_ml):
        """最適化されたモデルパラメータテスト"""
        # 各モデルタイプのパラメータ取得
        rf_params = orchestrator_no_ml._get_optimized_model_params("random_forest")
        gb_params = orchestrator_no_ml._get_optimized_model_params("gradient_boosting")
        xgb_params = orchestrator_no_ml._get_optimized_model_params("xgboost")
        linear_params = orchestrator_no_ml._get_optimized_model_params("linear")

        # パラメータの存在確認
        assert isinstance(rf_params, dict)
        assert isinstance(gb_params, dict)
        assert isinstance(xgb_params, dict)
        assert isinstance(linear_params, dict)

        # Random Forestの主要パラメータ確認
        assert "n_estimators" in rf_params
        assert "max_depth" in rf_params
        assert "random_state" in rf_params

        # 不明なモデルタイプの処理
        unknown_params = orchestrator_no_ml._get_optimized_model_params("unknown_model")
        assert unknown_params == {}


if __name__ == "__main__":
    pytest.main([__file__])
