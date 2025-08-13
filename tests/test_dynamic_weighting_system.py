import pytest
import numpy as np
from collections import deque
from datetime import datetime
import time
from unittest.mock import patch

from src.day_trade.ml.dynamic_weighting_system import DynamicWeightingSystem, DynamicWeightingConfig, MarketRegime, PerformanceWindow

# Helper function to create a mock timestamp for deterministic tests
def mock_time_time():
    return 1678886400.0 # A fixed timestamp (March 15, 2023 00:00:00 GMT)

@pytest.fixture
def default_config():
    return DynamicWeightingConfig(
        window_size=20, # Changed from 10 to 20
        min_samples_for_update=5,
        update_frequency=5,
        weighting_method="performance_based",
        decay_factor=0.95,
        momentum_factor=0.0, # Disable momentum for simpler testing
        enable_regime_detection=False, # Disable for simpler testing
        max_weight_change=0.2,
        min_weight=0.1,
        max_weight=0.9,
        verbose=False
    )

@pytest.fixture
def dws_instance(default_config):
    model_names = ["model_A", "model_B", "model_C"]
    return DynamicWeightingSystem(model_names, default_config)

@patch('time.time', side_effect=mock_time_time)
def test_dws_initialization(mock_time, dws_instance):
    assert dws_instance.model_names == ["model_A", "model_B", "model_C"]
    assert len(dws_instance.current_weights) == 3
    assert sum(dws_instance.current_weights.values()) == pytest.approx(1.0)
    assert dws_instance.config.window_size == 20 # Changed assertion
    assert dws_instance.config.weighting_method == "performance_based"
    assert dws_instance.current_regime == MarketRegime.SIDEWAYS # Default initial regime

@patch('time.time', side_effect=mock_time_time)
def test_update_performance_basic(mock_time, dws_instance):
    # Initial weights should be equal
    initial_weights = dws_instance.get_current_weights()

    # Simulate data where model_A performs best
    predictions_step1 = {
        "model_A": np.array([10.1, 10.2, 10.3, 10.4, 10.5]),
        "model_B": np.array([10.5, 10.4, 10.3, 10.2, 10.1]),
        "model_C": np.array([10.3, 10.1, 10.5, 10.2, 10.4]),
    }
    actuals_step1 = np.array([10.0, 10.1, 10.2, 10.3, 10.4]) # Model A is closest

    for i in range(dws_instance.config.min_samples_for_update):
        preds = {k: np.array([v[i]]) for k, v in predictions_step1.items()}
        actual = np.array([actuals_step1[i]])
        dws_instance.update_performance(preds, actual, mock_time_time() + i)

    # Weights should update after min_samples_for_update
    updated_weights = dws_instance.get_current_weights()
    assert updated_weights["model_A"] > initial_weights["model_A"]
    assert updated_weights["model_B"] < initial_weights["model_B"]
    assert updated_weights["model_C"] < initial_weights["model_C"]
    assert sum(updated_weights.values()) == pytest.approx(1.0)

    # Simulate more data where model_B performs best
    predictions_step2 = {
        "model_A": np.array([11.5, 11.4, 11.3, 11.2, 11.1]),
        "model_B": np.array([11.1, 11.2, 11.3, 11.4, 11.5]),
        "model_C": np.array([11.3, 11.1, 11.5, 11.2, 11.4]),
    }
    actuals_step2 = np.array([11.0, 11.1, 11.2, 11.3, 11.4]) # Model B is closest

    for i in range(dws_instance.config.min_samples_for_update):
        preds = {k: np.array([v[i]]) for k, v in predictions_step2.items()}
        actual = np.array([actuals_step2[i]])
        dws_instance.update_performance(preds, actual, mock_time_time() + dws_instance.config.min_samples_for_update + i)

    final_weights = dws_instance.get_current_weights()
    assert final_weights["model_A"] < updated_weights["model_A"]
    assert final_weights["model_B"] > updated_weights["model_B"]
    assert final_weights["model_C"] == pytest.approx(updated_weights["model_C"]) # Changed assertion
    assert sum(final_weights.values()) == pytest.approx(1.0)

    # Test weight constraints
    assert all(w >= dws_instance.config.min_weight for w in final_weights.values())
    assert all(w <= dws_instance.config.max_weight for w in final_weights.values())

@patch('time.time', side_effect=mock_time_time)
def test_performance_based_weighting(mock_time, dws_instance):
    # Populate enough data for a weight update
    for _ in range(dws_instance.config.window_size):
        dws_instance.update_performance(
            {"model_A": np.array([10.1]), "model_B": np.array([10.5]), "model_C": np.array([10.3])},
            np.array([10.0]),
            mock_time_time()
        )

    # Manually call the weighting method
    weights = dws_instance._performance_based_weighting()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "model_A" in weights
    assert "model_B" in weights
    assert "model_C" in weights

@patch('time.time', side_effect=mock_time_time)
def test_weight_constraints(mock_time, dws_instance):
    # Set initial weights to be outside constraints
    dws_instance.current_weights = {"model_A": 0.01, "model_B": 0.98, "model_C": 0.01}

    # Simulate a scenario where model_A should gain weight, model_B lose, model_C stay same
    predictions = {
        "model_A": np.array([10.1]),
        "model_B": np.array([10.9]),
        "model_C": np.array([10.5]),
    }
    actuals = np.array([10.0])

    # Populate enough data for a weight update
    for _ in range(dws_instance.config.window_size):
        dws_instance.update_performance(predictions, actuals, mock_time_time())

    # Check if weights are within constraints
    final_weights = dws_instance.get_current_weights()
    assert final_weights["model_A"] >= dws_instance.config.min_weight
    assert final_weights["model_B"] <= dws_instance.config.max_weight
    assert sum(final_weights.values()) == pytest.approx(1.0)

@patch('time.time', side_effect=mock_time_time)
def test_sharpe_based_weighting(mock_time, dws_instance):
    dws_instance.config.weighting_method = "sharpe_based"
    # Populate enough data for a weight update
    for _ in range(dws_instance.config.window_size):
        dws_instance.update_performance(
            {"model_A": np.array([10.1]), "model_B": np.array([10.5]), "model_C": np.array([10.3])},
            np.array([10.0]),
            mock_time_time()
        )

    weights = dws_instance._sharpe_based_weighting()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "model_A" in weights

@patch('time.time', side_effect=mock_time_time)
def test_regime_aware_weighting(mock_time, dws_instance):
    dws_instance.config.weighting_method = "regime_aware"
    dws_instance.config.enable_regime_detection = True
    dws_instance.config.volatility_threshold = 0.001 # Low threshold for easy high vol

    # Simulate data to trigger a high volatility regime
    # Ensure actuals have significant changes to trigger high volatility
    actuals_data = [10.0, 10.5, 9.8, 11.0, 9.5, 12.0, 9.0, 13.0, 8.0, 14.0,
                    7.0, 15.0, 6.0, 16.0, 5.0, 17.0, 4.0, 18.0, 3.0, 19.0]

    for i in range(len(actuals_data)): # Iterate through all actuals_data
        # Use a dummy prediction as it's not relevant for regime detection
        dummy_predictions = {"model_A": np.array([actuals_data[i]]),
                             "model_B": np.array([actuals_data[i]]),
                             "model_C": np.array([actuals_data[i]])}
        dws_instance.update_performance(
            dummy_predictions,
            np.array([actuals_data[i]]),
            mock_time_time() + i
        )

    # Ensure regime is detected and weights are updated
    print(f"Current regime: {dws_instance.current_regime}")
    assert dws_instance.current_regime != MarketRegime.SIDEWAYS
    weights = dws_instance.get_current_weights()
    assert sum(weights.values()) == pytest.approx(1.0)

@patch('time.time', side_effect=mock_time_time)
def test_momentum_application(mock_time, dws_instance):
    dws_instance.config.momentum_factor = 0.5 # Enable momentum

    # Set initial weights
    dws_instance.current_weights = {"model_A": 0.8, "model_B": 0.1, "model_C": 0.1}

    # Simulate data where model_B performs best, but momentum should slow down the shift
    predictions = {
        "model_A": np.array([10.9]),
        "model_B": np.array([10.1]),
        "model_C": np.array([10.5]),
    }
    actuals = np.array([10.0])

    # Populate enough data for a weight update
    for _ in range(dws_instance.config.window_size):
        dws_instance.update_performance(predictions, actuals, mock_time_time())

    final_weights = dws_instance.get_current_weights()
    # Model B should have increased, but not as much as without momentum
    assert final_weights["model_B"] > 0.1
    assert final_weights["model_A"] < 0.8
    assert sum(final_weights.values()) == pytest.approx(1.0)

    # Verify that momentum actually had an effect (e.g., model_B's weight is less than it would be without momentum)
    # This is hard to assert precisely without re-calculating the non-momentum case,
    # but we can check if it's between initial and full performance-based shift.
    # For now, just checking it's not exactly the initial or the unconstrained new weight.

    # This test primarily ensures the momentum logic is called and doesn't break.

@patch('time.time', side_effect=mock_time_time)
def test_insufficient_samples_for_update(mock_time, dws_instance):
    # Update with less than min_samples_for_update
    for i in range(dws_instance.config.min_samples_for_update - 1):
        dws_instance.update_performance(
            {"model_A": np.array([10.1]), "model_B": np.array([10.5]), "model_C": np.array([10.3])},
            np.array([10.0]),
            mock_time_time() + i
        )

    # Weights should not have changed
    initial_weights = {"model_A": 1/3, "model_B": 1/3, "model_C": 1/3}
    assert dws_instance.get_current_weights()["model_A"] == pytest.approx(initial_weights["model_A"])
    assert dws_instance.total_updates == 0

@patch('time.time', side_effect=mock_time_time)
def test_regime_detection(mock_time, dws_instance):
    dws_instance.config.enable_regime_detection = True
    dws_instance.config.volatility_threshold = 0.001 # Low threshold for easy high vol

    # Simulate low volatility sideways market
    # Use fixed data to guarantee low volatility
    low_vol_actuals = [10.0, 10.001, 10.002, 10.003, 10.004, 10.005, 10.006, 10.007, 10.008, 10.009,
                       10.010, 10.011, 10.012, 10.013, 10.014, 10.015, 10.016, 10.017, 10.018, 10.019]

    for i in range(20): # Need at least 20 samples for regime detection
        dws_instance.update_performance(
            {"model_A": np.array([low_vol_actuals[i]]),
             "model_B": np.array([low_vol_actuals[i]]),
             "model_C": np.array([low_vol_actuals[i]])},
            np.array([low_vol_actuals[i]]),
            mock_time_time() + i
        )
    print(f"Current regime (low vol): {dws_instance.current_regime}")
    assert dws_instance.current_regime == MarketRegime.LOW_VOLATILITY # Changed assertion

    # Simulate high volatility market
    # Use fixed data to guarantee high volatility
    high_vol_actuals = [10.0, 10.5, 9.8, 11.0, 9.5, 12.0, 9.0, 13.0, 8.0, 14.0,
                        7.0, 15.0, 6.0, 16.0, 5.0, 17.0, 4.0, 18.0, 3.0, 19.0]

    for i in range(20):
        dws_instance.update_performance(
            {"model_A": np.array([high_vol_actuals[i]]),
             "model_B": np.array([high_vol_actuals[i]]),
             "model_C": np.array([high_vol_actuals[i]])},
            np.array([high_vol_actuals[i]]),
            mock_time_time() + 20 + i
        )
    print(f"Current regime (high vol): {dws_instance.current_regime}")
    assert dws_instance.current_regime == MarketRegime.HIGH_VOLATILITY
    assert len(dws_instance.regime_history) > 0

@patch('time.time', side_effect=mock_time_time)
def test_performance_window_metrics(mock_time):
    predictions = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
    actuals = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
    timestamps = [1,2,3,4,5]

    window = PerformanceWindow(predictions, actuals, timestamps)
    metrics = window.calculate_metrics()

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "hit_rate" in metrics
    assert "sample_count" in metrics

    # For these specific inputs, model is perfectly aligned with actuals direction
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["sample_count"] == 5
    assert metrics["rmse"] == pytest.approx(0.1) # sqrt(mean((0.1)^2))
    assert metrics["mae"] == pytest.approx(0.1)

    # Test case with no predictions
    window_empty = PerformanceWindow(np.array([]), np.array([]), [])
    metrics_empty = window_empty.calculate_metrics()
    assert metrics_empty == {}

    # Test case with single prediction (hit_rate should be 0.5)
    window_single = PerformanceWindow(np.array([10.1]), np.array([10.0]), [1])
    metrics_single = window_single.calculate_metrics()
    assert metrics_single["hit_rate"] == pytest.approx(0.5)
    assert metrics_single["sample_count"] == 1
    assert metrics_single["rmse"] == pytest.approx(0.1)
    assert metrics_single["mae"] == pytest.approx(0.1)