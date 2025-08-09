#!/usr/bin/env python3
"""
Volatility Prediction System テスト
Issue #315 Phase 4: ボラティリティ予測システム実装テスト

GARCH モデル、VIX指標、動的リスク調整の総合テスト
"""

import asyncio
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルート追加
sys.path.insert(0, str(Path(__file__).parent))

async def test_volatility_system_initialization():
    """Volatility Prediction System初期化テスト"""
    print("\n=== Volatility Prediction System初期化テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=True,
            enable_parallel=True,
            garch_model_type="GARCH",
            forecast_horizon=5,
            var_confidence=0.95,
            max_concurrent=5
        )
        print("[OK] VolatilityPredictionSystem initialization success")

        # 設定確認
        assert volatility_system.enable_cache == True, "Cache should be enabled"
        assert volatility_system.enable_parallel == True, "Parallel should be enabled"
        assert volatility_system.garch_model_type == "GARCH", f"GARCH model should be GARCH, got {volatility_system.garch_model_type}"
        assert volatility_system.forecast_horizon == 5, f"Forecast horizon should be 5, got {volatility_system.forecast_horizon}"
        assert volatility_system.var_confidence == 0.95, f"VaR confidence should be 0.95, got {volatility_system.var_confidence}"

        print(f"[OK] Cache enabled: {volatility_system.enable_cache}")
        print(f"[OK] Parallel enabled: {volatility_system.enable_parallel}")
        print(f"[OK] GARCH model type: {volatility_system.garch_model_type}")
        print(f"[OK] Forecast horizon: {volatility_system.forecast_horizon} days")
        print(f"[OK] VaR confidence: {volatility_system.var_confidence:.1%}")
        print(f"[OK] Max concurrent: {volatility_system.max_concurrent}")

        return True

    except Exception as e:
        print(f"[ERROR] Volatility system initialization test failed: {e}")
        traceback.print_exc()
        return False

async def test_garch_volatility_prediction():
    """GARCH ボラティリティ予測テスト"""
    print("\n=== GARCH ボラティリティ予測テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # 長期テストデータ生成（GARCH用に100日以上）
        dates = pd.date_range(start='2024-01-01', periods=150, freq='D')
        np.random.seed(42)

        # ボラティリティクラスタリングを持つ時系列生成
        base_price = 2500
        volatility_regime = np.random.choice([0.01, 0.02, 0.03], size=150, p=[0.4, 0.4, 0.2])
        returns = []

        for i in range(150):
            if i == 0:
                returns.append(0.0)
            else:
                ret = np.random.normal(0.0005, volatility_regime[i])
                returns.append(ret)

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
            'High': [p * np.random.uniform(1.002, 1.015) for p in prices],
            'Low': [p * np.random.uniform(0.985, 0.998) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(600000, 2200000, 150),
        }, index=dates)

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=False,  # テスト用にキャッシュ無効
            enable_parallel=False,
            garch_model_type="GARCH"
        )
        print("[OK] GARCH volatility system initialization success")

        # GARCH予測実行
        garch_result = await volatility_system.predict_garch_volatility(test_data, "TEST_GARCH")

        # 結果検証
        assert hasattr(garch_result, 'symbol'), "GARCH result missing symbol"
        assert garch_result.symbol == "TEST_GARCH", f"Symbol mismatch: {garch_result.symbol}"
        assert hasattr(garch_result, 'current_volatility'), "GARCH result missing current_volatility"
        assert hasattr(garch_result, 'forecast_volatility'), "GARCH result missing forecast_volatility"
        assert garch_result.current_volatility > 0, f"Invalid current volatility: {garch_result.current_volatility}"
        assert garch_result.forecast_volatility > 0, f"Invalid forecast volatility: {garch_result.forecast_volatility}"
        assert hasattr(garch_result, 'volatility_trend'), "GARCH result missing volatility_trend"
        assert garch_result.volatility_trend in ['increasing', 'decreasing', 'stable'], f"Invalid trend: {garch_result.volatility_trend}"
        assert hasattr(garch_result, 'confidence_interval'), "GARCH result missing confidence_interval"
        assert len(garch_result.confidence_interval) == 2, "Confidence interval should have 2 values"

        print(f"[OK] Current volatility: {garch_result.current_volatility:.1%}")
        print(f"[OK] Forecast volatility: {garch_result.forecast_volatility:.1%}")
        print(f"[OK] Volatility trend: {garch_result.volatility_trend}")
        print(f"[OK] Model type: {garch_result.model_type}")
        print(f"[OK] Confidence interval: ({garch_result.confidence_interval[0]:.1%}, {garch_result.confidence_interval[1]:.1%})")
        print(f"[OK] Processing time: {garch_result.processing_time:.3f}s")

        # モデルパラメータ確認
        if garch_result.model_params:
            print(f"[OK] Model parameters: {len(garch_result.model_params)} parameters")
            for param, value in garch_result.model_params.items():
                print(f"     {param}: {value:.6f}")

        return True

    except Exception as e:
        print(f"[ERROR] GARCH volatility prediction test failed: {e}")
        traceback.print_exc()
        return False

async def test_vix_risk_assessment():
    """VIX リスク評価テスト"""
    print("\n=== VIX リスク評価テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(100)

        # 高ボラティリティ期間を含むデータ
        base_price = 3000
        high_vol_periods = [30, 60]  # 30日目と60日目から高ボラ
        prices = [base_price]

        for i in range(99):
            if i in high_vol_periods or (i > 30 and i < 40) or (i > 60 and i < 70):
                daily_ret = np.random.normal(0, 0.035)  # 高ボラティリティ
            else:
                daily_ret = np.random.normal(0.001, 0.015)  # 通常ボラティリティ

            new_price = prices[-1] * (1 + daily_ret)
            prices.append(new_price)

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
            'High': [p * np.random.uniform(1.005, 1.025) for p in prices],
            'Low': [p * np.random.uniform(0.975, 0.995) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(400000, 1800000, 100),
        }, index=dates)

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=False,
            enable_parallel=False
        )
        print("[OK] VIX risk system initialization success")

        # VIXリスク評価実行
        vix_result = await volatility_system.assess_vix_risk(test_data, "TEST_VIX")

        # 結果検証
        assert hasattr(vix_result, 'symbol'), "VIX result missing symbol"
        assert vix_result.symbol == "TEST_VIX", f"Symbol mismatch: {vix_result.symbol}"
        assert hasattr(vix_result, 'vix_level'), "VIX result missing vix_level"
        assert vix_result.vix_level > 0, f"Invalid VIX level: {vix_result.vix_level}"
        assert hasattr(vix_result, 'risk_regime'), "VIX result missing risk_regime"
        assert vix_result.risk_regime in ['low', 'normal', 'high', 'extreme'], f"Invalid risk regime: {vix_result.risk_regime}"
        assert hasattr(vix_result, 'market_fear_indicator'), "VIX result missing market_fear_indicator"
        assert 0 <= vix_result.market_fear_indicator <= 1, f"Invalid market fear: {vix_result.market_fear_indicator}"
        assert hasattr(vix_result, 'correlation_with_vix'), "VIX result missing correlation"
        assert -1 <= vix_result.correlation_with_vix <= 1, f"Invalid correlation: {vix_result.correlation_with_vix}"
        assert hasattr(vix_result, 'risk_adjustment_factor'), "VIX result missing risk_adjustment_factor"
        assert vix_result.risk_adjustment_factor > 0, f"Invalid risk adjustment: {vix_result.risk_adjustment_factor}"
        assert hasattr(vix_result, 'recommended_action'), "VIX result missing recommended_action"
        assert vix_result.recommended_action in ['increase', 'maintain', 'reduce', 'avoid'], f"Invalid action: {vix_result.recommended_action}"

        print(f"[OK] VIX level: {vix_result.vix_level:.1f}")
        print(f"[OK] Risk regime: {vix_result.risk_regime}")
        print(f"[OK] Market fear indicator: {vix_result.market_fear_indicator:.1%}")
        print(f"[OK] Correlation with VIX: {vix_result.correlation_with_vix:.3f}")
        print(f"[OK] Risk adjustment factor: {vix_result.risk_adjustment_factor:.3f}")
        print(f"[OK] Recommended action: {vix_result.recommended_action}")
        print(f"[OK] Processing time: {vix_result.processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"[ERROR] VIX risk assessment test failed: {e}")
        traceback.print_exc()
        return False

async def test_dynamic_risk_metrics():
    """動的リスク指標テスト"""
    print("\n=== 動的リスク指標テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # テストデータ生成
        dates = pd.date_range(start='2024-01-01', periods=120, freq='D')
        np.random.seed(200)

        # ドローダウン期間を含むデータ
        base_price = 2800
        prices = [base_price]

        for i in range(119):
            # 40-60日目にドローダウン期間
            if 40 <= i <= 60:
                daily_ret = np.random.normal(-0.005, 0.025)  # 下落トレンド + 高ボラ
            else:
                daily_ret = np.random.normal(0.001, 0.018)   # 通常期間

            new_price = max(prices[-1] * (1 + daily_ret), base_price * 0.7)  # 最低価格制限
            prices.append(new_price)

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
            'High': [p * np.random.uniform(1.005, 1.020) for p in prices],
            'Low': [p * np.random.uniform(0.980, 0.998) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(500000, 2000000, 120),
        }, index=dates)

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=False,
            var_confidence=0.95
        )
        print("[OK] Dynamic risk system initialization success")

        # 事前にGARCH・VIX結果を取得
        garch_result = await volatility_system.predict_garch_volatility(test_data, "TEST_RISK")
        vix_result = await volatility_system.assess_vix_risk(test_data, "TEST_RISK")
        print(f"[OK] Prerequisites obtained: GARCH={garch_result.volatility_trend}, VIX={vix_result.risk_regime}")

        # 動的リスク指標計算
        risk_metrics = await volatility_system.calculate_dynamic_risk_metrics(
            test_data, "TEST_RISK", garch_result, vix_result
        )

        # 結果検証
        assert hasattr(risk_metrics, 'symbol'), "Risk metrics missing symbol"
        assert risk_metrics.symbol == "TEST_RISK", f"Symbol mismatch: {risk_metrics.symbol}"
        assert hasattr(risk_metrics, 'current_var'), "Risk metrics missing current_var"
        assert 0 < risk_metrics.current_var < 1, f"Invalid VaR: {risk_metrics.current_var}"
        assert hasattr(risk_metrics, 'expected_shortfall'), "Risk metrics missing expected_shortfall"
        assert risk_metrics.expected_shortfall >= risk_metrics.current_var, "ES should be >= VaR"
        assert hasattr(risk_metrics, 'max_drawdown_forecast'), "Risk metrics missing max_drawdown_forecast"
        assert 0 < risk_metrics.max_drawdown_forecast < 1, f"Invalid drawdown forecast: {risk_metrics.max_drawdown_forecast}"
        assert hasattr(risk_metrics, 'position_size_multiplier'), "Risk metrics missing position_size_multiplier"
        assert risk_metrics.position_size_multiplier > 0, f"Invalid position multiplier: {risk_metrics.position_size_multiplier}"
        assert hasattr(risk_metrics, 'risk_budget_allocation'), "Risk metrics missing risk_budget_allocation"
        assert 0 < risk_metrics.risk_budget_allocation <= 1, f"Invalid risk budget: {risk_metrics.risk_budget_allocation}"
        assert hasattr(risk_metrics, 'stop_loss_level'), "Risk metrics missing stop_loss_level"
        assert hasattr(risk_metrics, 'take_profit_level'), "Risk metrics missing take_profit_level"

        print(f"[OK] Current VaR (95%): {risk_metrics.current_var:.1%}")
        print(f"[OK] Expected Shortfall: {risk_metrics.expected_shortfall:.1%}")
        print(f"[OK] Max Drawdown forecast: {risk_metrics.max_drawdown_forecast:.1%}")
        print(f"[OK] Sharpe ratio forecast: {risk_metrics.sharpe_ratio_forecast:.3f}")
        print(f"[OK] Position size multiplier: {risk_metrics.position_size_multiplier:.3f}x")
        print(f"[OK] Risk budget allocation: {risk_metrics.risk_budget_allocation:.1%}")
        print(f"[OK] Stop loss level: ¥{risk_metrics.stop_loss_level:.0f}")
        print(f"[OK] Take profit level: ¥{risk_metrics.take_profit_level:.0f}")
        print(f"[OK] Processing time: {risk_metrics.processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"[ERROR] Dynamic risk metrics test failed: {e}")
        traceback.print_exc()
        return False

async def test_integrated_volatility_forecast():
    """統合ボラティリティ予測テスト"""
    print("\n=== 統合ボラティリティ予測テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # 総合的なテストデータ（長期間・複雑なパターン）
        dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
        np.random.seed(300)

        # 複数レジームを含む時系列
        base_price = 2400
        prices = [base_price]
        regimes = [
            (0, 40, 0.001, 0.015),    # 通常期間
            (40, 80, -0.002, 0.025),  # 下落・高ボラ期間
            (80, 120, 0.002, 0.020),  # 回復期間
            (120, 180, 0.0005, 0.012) # 安定期間
        ]

        for i in range(179):
            current_regime = None
            for start, end, mean_ret, vol in regimes:
                if start <= i < end:
                    current_regime = (mean_ret, vol)
                    break

            if current_regime:
                daily_ret = np.random.normal(current_regime[0], current_regime[1])
            else:
                daily_ret = np.random.normal(0.001, 0.018)

            new_price = prices[-1] * (1 + daily_ret)
            prices.append(new_price)

        test_data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.998, 1.002) for p in prices],
            'High': [p * np.random.uniform(1.005, 1.020) for p in prices],
            'Low': [p * np.random.uniform(0.980, 0.998) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(600000, 2500000, 180),
        }, index=dates)

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=True,
            enable_parallel=True,
            forecast_horizon=5
        )
        print("[OK] Integrated volatility system initialization success")

        # 統合予測実行
        integrated_result = await volatility_system.integrated_volatility_forecast(
            test_data, "TEST_INTEGRATED"
        )

        # 結果検証
        assert hasattr(integrated_result, 'symbol'), "Integrated result missing symbol"
        assert integrated_result.symbol == "TEST_INTEGRATED", f"Symbol mismatch: {integrated_result.symbol}"
        assert hasattr(integrated_result, 'garch_result'), "Integrated result missing garch_result"
        assert hasattr(integrated_result, 'vix_assessment'), "Integrated result missing vix_assessment"
        assert hasattr(integrated_result, 'risk_metrics'), "Integrated result missing risk_metrics"
        assert hasattr(integrated_result, 'final_volatility_forecast'), "Integrated result missing final_volatility_forecast"
        assert integrated_result.final_volatility_forecast > 0, f"Invalid volatility forecast: {integrated_result.final_volatility_forecast}"
        assert hasattr(integrated_result, 'integrated_risk_score'), "Integrated result missing integrated_risk_score"
        assert 0 <= integrated_result.integrated_risk_score <= 1, f"Invalid risk score: {integrated_result.integrated_risk_score}"
        assert hasattr(integrated_result, 'recommended_position_size'), "Integrated result missing recommended_position_size"
        assert 0 < integrated_result.recommended_position_size <= 1, f"Invalid position size: {integrated_result.recommended_position_size}"
        assert hasattr(integrated_result, 'confidence_level'), "Integrated result missing confidence_level"
        assert 0 < integrated_result.confidence_level <= 1, f"Invalid confidence: {integrated_result.confidence_level}"

        print(f"[OK] Final volatility forecast: {integrated_result.final_volatility_forecast:.1%}")
        print(f"[OK] Integrated risk score: {integrated_result.integrated_risk_score:.3f}")
        print(f"[OK] Recommended position size: {integrated_result.recommended_position_size:.1%}")
        print(f"[OK] Risk-adjusted return forecast: {integrated_result.risk_adjusted_return_forecast:.1%}")
        print(f"[OK] Confidence level: {integrated_result.confidence_level:.1%}")
        print(f"[OK] Processing time: {integrated_result.processing_time:.3f}s")

        # コンポーネント結果確認
        garch = integrated_result.garch_result
        vix = integrated_result.vix_assessment
        risk = integrated_result.risk_metrics

        print("\n[COMPONENTS] Sub-system results:")
        print(f"  GARCH: {garch.volatility_trend} trend, {garch.forecast_volatility:.1%} vol")
        print(f"  VIX: {vix.risk_regime} regime, {vix.recommended_action} action")
        print(f"  Risk: {risk.current_var:.1%} VaR, {risk.position_size_multiplier:.2f}x position")

        return True

    except Exception as e:
        print(f"[ERROR] Integrated volatility forecast test failed: {e}")
        traceback.print_exc()
        return False

async def test_batch_volatility_analysis():
    """バッチボラティリティ分析テスト"""
    print("\n=== バッチボラティリティ分析テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # 複数銘柄データ
        symbols = ["STOCK_A", "STOCK_B", "STOCK_C"]
        batch_data = {}

        for i, symbol in enumerate(symbols):
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            np.random.seed(400 + i * 10)

            base_price = 2000 + i * 500
            # 異なるボラティリティ特性
            vol_multipliers = [1.0, 1.5, 0.8]  # 通常、高ボラ、低ボラ
            vol_mult = vol_multipliers[i]

            prices = [base_price]
            for j in range(99):
                daily_ret = np.random.normal(0.001, 0.020 * vol_mult)
                new_price = prices[-1] * (1 + daily_ret)
                prices.append(new_price)

            test_data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'High': [p * np.random.uniform(1.005, 1.018) for p in prices],
                'Low': [p * np.random.uniform(0.982, 0.999) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(300000, 1500000, 100),
            }, index=dates)

            batch_data[symbol] = test_data

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=True,
            enable_parallel=False,  # テスト用に順次実行
            max_concurrent=3
        )
        print("[OK] Batch volatility system initialization success")

        # バッチ分析実行
        batch_results = {}
        for symbol, data in batch_data.items():
            result = await volatility_system.integrated_volatility_forecast(data, symbol)
            batch_results[symbol] = result

            print(f"[OK] {symbol} processed:")
            print(f"     Volatility forecast: {result.final_volatility_forecast:.1%}")
            print(f"     Risk score: {result.integrated_risk_score:.3f}")
            print(f"     Position size: {result.recommended_position_size:.1%}")
            print(f"     Confidence: {result.confidence_level:.1%}")

        # バッチ結果検証
        assert len(batch_results) == len(symbols), f"Results count mismatch: {len(batch_results)} vs {len(symbols)}"

        for symbol in symbols:
            assert symbol in batch_results, f"Missing results for {symbol}"
            result = batch_results[symbol]

            assert result.symbol == symbol, f"Symbol mismatch: {result.symbol}"
            assert result.final_volatility_forecast > 0, f"Invalid volatility for {symbol}"
            assert 0 <= result.integrated_risk_score <= 1, f"Invalid risk score for {symbol}"
            assert 0 < result.recommended_position_size <= 1, f"Invalid position size for {symbol}"

        print(f"[OK] Batch analysis completed: {len(batch_results)} symbols")

        # 相対比較表示
        print("\n[COMPARISON] Relative risk ranking:")
        sorted_symbols = sorted(
            batch_results.items(),
            key=lambda x: x[1].integrated_risk_score,
            reverse=True
        )
        for i, (symbol, result) in enumerate(sorted_symbols):
            rank = i + 1
            print(f"  {rank}. {symbol}: Risk={result.integrated_risk_score:.3f}, Vol={result.final_volatility_forecast:.1%}")

        return True

    except Exception as e:
        print(f"[ERROR] Batch volatility analysis test failed: {e}")
        traceback.print_exc()
        return False

async def test_performance_monitoring():
    """パフォーマンス監視テスト"""
    print("\n=== パフォーマンス監視テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # システム初期化
        volatility_system = VolatilityPredictionSystem(
            enable_cache=True,
            enable_parallel=True
        )

        # 初期統計取得
        initial_stats = volatility_system.get_performance_stats()

        # 統計項目検証
        required_keys = [
            'total_forecasts', 'cache_hit_rate', 'garch_forecasts',
            'vix_assessments', 'avg_processing_time', 'system_status'
        ]

        for key in required_keys:
            assert key in initial_stats, f"Missing stats key: {key}"

        print("[OK] Performance stats structure validation passed")
        print(f"[STATS] Total forecasts: {initial_stats['total_forecasts']}")
        print(f"[STATS] Cache hit rate: {initial_stats['cache_hit_rate']:.1%}")
        print(f"[STATS] GARCH forecasts: {initial_stats['garch_forecasts']}")
        print(f"[STATS] VIX assessments: {initial_stats['vix_assessments']}")

        # システム状態確認
        system_status = initial_stats['system_status']
        print(f"[SYSTEM] Cache enabled: {system_status['cache_enabled']}")
        print(f"[SYSTEM] Parallel enabled: {system_status['parallel_enabled']}")
        print(f"[SYSTEM] GARCH model: {system_status['garch_model']}")
        print(f"[SYSTEM] ARCH available: {system_status['arch_available']}")
        print(f"[SYSTEM] Scikit-learn available: {system_status['sklearn_available']}")

        # リスクパラメータ確認
        risk_params = initial_stats['risk_parameters']
        print(f"[RISK] Base position size: {risk_params['base_position_size']:.1%}")
        print(f"[RISK] Max position size: {risk_params['max_position_size']:.1%}")
        print(f"[RISK] Max leverage: {risk_params['max_leverage']:.1f}x")

        # 最適化効果確認
        benefits = initial_stats['optimization_benefits']
        print("[OPTIMIZATION] Benefits:")
        for benefit, description in benefits.items():
            print(f"  {benefit}: {description}")

        return True

    except Exception as e:
        print(f"[ERROR] Performance monitoring test failed: {e}")
        traceback.print_exc()
        return False

async def test_risk_scenario_analysis():
    """リスクシナリオ分析テスト"""
    print("\n=== リスクシナリオ分析テスト ===")

    try:
        from src.day_trade.risk.volatility_prediction_system import (
            VolatilityPredictionSystem,
        )

        # 極端なリスクシナリオデータ生成
        scenarios = {
            "low_risk": {"mean": 0.001, "vol": 0.010, "description": "低リスク期間"},
            "normal_risk": {"mean": 0.001, "vol": 0.020, "description": "通常リスク期間"},
            "high_risk": {"mean": -0.001, "vol": 0.040, "description": "高リスク期間"},
            "extreme_risk": {"mean": -0.003, "vol": 0.060, "description": "極端リスク期間"}
        }

        volatility_system = VolatilityPredictionSystem(
            enable_cache=False,
            forecast_horizon=3
        )

        scenario_results = {}

        for scenario_name, params in scenarios.items():
            # シナリオデータ生成
            dates = pd.date_range(start='2024-01-01', periods=80, freq='D')
            np.random.seed(500)

            base_price = 2500
            prices = [base_price]

            for i in range(79):
                daily_ret = np.random.normal(params["mean"], params["vol"])
                new_price = max(prices[-1] * (1 + daily_ret), base_price * 0.5)
                prices.append(new_price)

            scenario_data = pd.DataFrame({
                'Open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'High': [p * np.random.uniform(1.005, 1.015) for p in prices],
                'Low': [p * np.random.uniform(0.985, 0.999) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(400000, 2000000, 80),
            }, index=dates)

            # 統合予測実行
            result = await volatility_system.integrated_volatility_forecast(
                scenario_data, f"SCENARIO_{scenario_name.upper()}"
            )

            scenario_results[scenario_name] = {
                'result': result,
                'params': params
            }

            print(f"[OK] {scenario_name} scenario ({params['description']}):")
            print(f"     Volatility forecast: {result.final_volatility_forecast:.1%}")
            print(f"     Risk score: {result.integrated_risk_score:.3f}")
            print(f"     Position size: {result.recommended_position_size:.1%}")
            print(f"     VIX risk regime: {result.vix_assessment.risk_regime}")
            print(f"     Max drawdown forecast: {result.risk_metrics.max_drawdown_forecast:.1%}")

        # シナリオ間比較検証
        risk_scores = [data['result'].integrated_risk_score for data in scenario_results.values()]
        position_sizes = [data['result'].recommended_position_size for data in scenario_results.values()]

        # リスクスコアが適切に段階的に上昇することを確認
        assert risk_scores[0] < risk_scores[1] < risk_scores[2], "Risk scores should increase with risk level"

        # ポジションサイズが適切に段階的に減少することを確認
        assert position_sizes[0] >= position_sizes[1] >= position_sizes[2], "Position sizes should decrease with risk level"

        print("\n[VALIDATION] Risk ranking verification:")
        for i, (scenario, data) in enumerate(scenario_results.items()):
            print(f"  {i+1}. {scenario}: Risk={data['result'].integrated_risk_score:.3f}, Pos={data['result'].recommended_position_size:.1%}")

        print("[OK] Risk scenario differentiation working correctly")

        return True

    except Exception as e:
        print(f"[ERROR] Risk scenario analysis test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """メインテスト実行"""
    print("Volatility Prediction System（統合最適化版）テスト開始")
    print("=" * 80)

    test_results = []

    # 各テスト実行
    test_results.append(("システム初期化", await test_volatility_system_initialization()))
    test_results.append(("GARCH ボラティリティ予測", await test_garch_volatility_prediction()))
    test_results.append(("VIX リスク評価", await test_vix_risk_assessment()))
    test_results.append(("動的リスク指標", await test_dynamic_risk_metrics()))
    test_results.append(("統合ボラティリティ予測", await test_integrated_volatility_forecast()))
    test_results.append(("バッチボラティリティ分析", await test_batch_volatility_analysis()))
    test_results.append(("パフォーマンス監視", await test_performance_monitoring()))
    test_results.append(("リスクシナリオ分析", await test_risk_scenario_analysis()))

    # 結果サマリー
    print("\n" + "=" * 80)
    print("=== テスト結果サマリー ===")

    passed = 0
    for name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1

    success_rate = passed / len(test_results) * 100
    print(f"\n成功率: {passed}/{len(test_results)} ({success_rate:.1f}%)")

    if passed == len(test_results):
        print("[SUCCESS] 全テスト成功！Volatility Prediction System準備完了")
        print("\nIssue #315 Phase 4実装成果:")
        print("🎯 GARCH ボラティリティモデル（時系列ボラティリティ予測）")
        print("📊 VIX指標リスク評価（市場恐怖指数・リスクレジーム判定）")
        print("⚖️  動的リスク調整（VaR、Expected Shortfall、ポジションサイジング）")
        print("🔄 統合ボラティリティ予測（GARCH + VIX + リスク指標統合）")
        print("📈 リスクシナリオ対応（低リスク～極端リスク適応的調整）")
        print("🛡️  最大ドローダウン20%削減・シャープレシオ0.3向上目標達成基盤")
        print("🚀 統合最適化基盤フル活用（Issues #322-325 + Phase 1-3統合）")
        return True
    else:
        print("[WARNING] 一部テスト失敗 - 要修正")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nテスト中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n致命的エラー: {e}")
        traceback.print_exc()
        sys.exit(1)
