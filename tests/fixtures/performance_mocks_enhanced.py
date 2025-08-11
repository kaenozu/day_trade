#!/usr/bin/env python3
"""
パフォーマンステスト用強化モックシステム
Issue #375: テスト速度改善のための重い処理モック化

重い処理（ML訓練、バックテスト、大規模I/O）を高速化するモック実装
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List
from unittest.mock import Mock

import numpy as np
import pandas as pd


@dataclass
class MockPerformanceResult:
    """モック性能結果"""

    execution_time_ms: float
    success_rate: float
    data_points: int
    mock_used: bool = True


class MLModelMocks:
    """機械学習モデル用高速モック"""

    @staticmethod
    def create_fast_ml_model_manager() -> Mock:
        """高速MLモデルマネージャーモック作成"""
        mock = Mock()
        mock.name = "FastMLModelManager"

        # 訓練の高速モック（実際の訓練をスキップ）
        def fast_train_model(model_name: str, X: pd.DataFrame, y: pd.Series, **kwargs):
            """高速訓練モック（0.001秒で完了）"""
            time.sleep(0.001)  # 最小限の遅延
            return {
                "model_name": model_name,
                "training_samples": len(X),
                "features": len(X.columns),
                "training_time_ms": 1.0,
                "accuracy": np.random.uniform(0.8, 0.95),
                "status": "trained",
            }

        mock.train_model.side_effect = fast_train_model

        # 予測の高速モック
        def fast_predict(model_name: str, X: pd.DataFrame, **kwargs):
            """高速予測モック"""
            time.sleep(0.0001)  # 極小遅延
            # 現実的な予測値を生成
            predictions = np.random.normal(0.05, 0.1, len(X))  # 5%±10%のリターン予測
            return predictions

        mock.predict.side_effect = fast_predict

        # モデル保存・読み込みの高速モック
        def fast_save_model(model_name: str, path: str, **kwargs):
            """高速モデル保存モック"""
            time.sleep(0.001)
            return {"status": "saved", "path": path, "size_mb": 1.2}

        def fast_load_model(model_name: str, path: str, **kwargs):
            """高速モデル読み込みモック"""
            time.sleep(0.001)
            return {
                "status": "loaded",
                "model_name": model_name,
                "accuracy": np.random.uniform(0.8, 0.95),
            }

        mock.save_model.side_effect = fast_save_model
        mock.load_model.side_effect = fast_load_model

        # バッチ処理のモック
        def fast_batch_predict(models: List[str], X: pd.DataFrame, **kwargs):
            """高速バッチ予測モック"""
            time.sleep(0.001)
            results = {}
            for model_name in models:
                results[model_name] = np.random.normal(0.05, 0.1, len(X))
            return results

        mock.batch_predict.side_effect = fast_batch_predict

        return mock

    @staticmethod
    def create_fast_feature_engineering_manager() -> Mock:
        """高速特徴量エンジニアリングマネージャーモック"""
        mock = Mock()
        mock.name = "FastFeatureEngineeringManager"

        def fast_generate_features(data: pd.DataFrame, **kwargs):
            """高速特徴量生成モック"""
            time.sleep(0.001)

            # 基本的な特徴量を高速生成
            features = data.copy()

            # テクニカル指標風の特徴量
            features["sma_20"] = data.get("Close", data.iloc[:, 0]).rolling(20).mean()
            features["rsi"] = np.random.uniform(20, 80, len(data))
            features["macd"] = np.random.normal(0, 0.5, len(data))
            features["volatility"] = np.random.uniform(0.01, 0.05, len(data))
            features["volume_sma"] = data.get("Volume", 1000000)

            return features

        mock.generate_features.side_effect = fast_generate_features

        def fast_calculate_indicators(
            data: pd.DataFrame, indicators: List[str], **kwargs
        ):
            """高速指標計算モック"""
            time.sleep(0.001)
            result = {}

            for indicator in indicators:
                if "sma" in indicator.lower():
                    result[indicator] = data.iloc[:, 0].rolling(20).mean()
                elif "rsi" in indicator.lower():
                    result[indicator] = np.random.uniform(20, 80, len(data))
                elif "macd" in indicator.lower():
                    result[indicator] = np.random.normal(0, 0.5, len(data))
                else:
                    result[indicator] = np.random.normal(0, 1, len(data))

            return result

        mock.calculate_indicators.side_effect = fast_calculate_indicators

        return mock


class BacktestMocks:
    """バックテスト用高速モック"""

    @staticmethod
    def create_fast_backtest_engine() -> Mock:
        """高速バックテストエンジンモック"""
        mock = Mock()
        mock.name = "FastBacktestEngine"

        def fast_run_backtest(
            data: pd.DataFrame,
            strategy: Callable,
            start_date: str = None,
            end_date: str = None,
            initial_capital: float = 1000000,
            **kwargs,
        ):
            """高速バックテスト実行モック"""
            time.sleep(0.005)  # 5ms遅延

            # 現実的な結果を生成
            days = len(data) if hasattr(data, "__len__") else 252
            final_value = initial_capital * np.random.uniform(0.9, 1.3)  # -10%〜+30%
            total_return = (final_value - initial_capital) / initial_capital

            # 取引統計
            num_trades = int(days / 10)  # 10日に1回取引
            winning_trades = int(num_trades * np.random.uniform(0.4, 0.7))

            result = {
                "initial_capital": initial_capital,
                "final_value": final_value,
                "total_return": total_return,
                "annualized_return": total_return * (252 / days) if days > 0 else 0,
                "sharpe_ratio": np.random.uniform(-0.5, 2.0),
                "max_drawdown": -np.random.uniform(0.05, 0.25),
                "volatility": np.random.uniform(0.15, 0.35),
                "num_trades": num_trades,
                "winning_trades": winning_trades,
                "win_rate": winning_trades / num_trades if num_trades > 0 else 0,
                "execution_time_ms": 5.0,
                "data_points": days,
            }

            return result

        mock.run_backtest.side_effect = fast_run_backtest

        # 複数戦略バックテストのモック
        def fast_multi_strategy_backtest(
            strategies: List[Callable], data: pd.DataFrame, **kwargs
        ):
            """高速マルチ戦略バックテストモック"""
            time.sleep(0.002 * len(strategies))  # 戦略数に比例

            results = {}
            for i, strategy in enumerate(strategies):
                strategy_name = (
                    f"strategy_{i}"
                    if hasattr(strategy, "__name__")
                    else strategy.__name__
                )
                results[strategy_name] = fast_run_backtest(data, strategy, **kwargs)

            return results

        mock.multi_strategy_backtest.side_effect = fast_multi_strategy_backtest

        return mock

    @staticmethod
    def create_fast_portfolio_optimizer() -> Mock:
        """高速ポートフォリオ最適化モック"""
        mock = Mock()
        mock.name = "FastPortfolioOptimizer"

        def fast_optimize_portfolio(assets: List[str], returns: pd.DataFrame, **kwargs):
            """高速ポートフォリオ最適化モック"""
            time.sleep(0.003)  # 3ms遅延

            # 等重みをベースにしたランダムな重み
            n_assets = len(assets)
            weights = np.random.dirichlet(np.ones(n_assets))  # 合計1になる正の重み

            result = {
                "assets": assets,
                "weights": dict(zip(assets, weights)),
                "expected_return": np.random.uniform(0.05, 0.15),
                "volatility": np.random.uniform(0.12, 0.25),
                "sharpe_ratio": np.random.uniform(0.5, 1.8),
                "optimization_time_ms": 3.0,
            }

            return result

        mock.optimize_portfolio.side_effect = fast_optimize_portfolio

        return mock


class DatabaseMocks:
    """データベース用高速モック"""

    @staticmethod
    def create_fast_database_manager() -> Mock:
        """高速データベースマネージャーモック"""
        mock = Mock()
        mock.name = "FastDatabaseManager"

        # インメモリ「データベース」
        in_memory_db = {"stocks": [], "trades": [], "portfolios": [], "alerts": []}

        def fast_execute_query(query: str, params: Dict = None, **kwargs):
            """高速クエリ実行モック"""
            time.sleep(0.0001)  # 0.1ms遅延

            query_lower = query.lower()

            # SELECT操作
            if query_lower.startswith("select"):
                if "stock" in query_lower:
                    return [
                        {"symbol": "7203", "name": "トヨタ", "price": 2000},
                        {"symbol": "6758", "name": "ソニー", "price": 3000},
                    ]
                elif "trade" in query_lower:
                    return [
                        {"id": 1, "symbol": "7203", "quantity": 100, "price": 2000},
                        {"id": 2, "symbol": "6758", "quantity": 50, "price": 3000},
                    ]
                else:
                    return []

            # INSERT/UPDATE/DELETE操作
            elif any(op in query_lower for op in ["insert", "update", "delete"]):
                return {"affected_rows": np.random.randint(1, 10)}

            return []

        mock.execute_query.side_effect = fast_execute_query

        def fast_bulk_insert(table: str, data: List[Dict], **kwargs):
            """高速バルクインサートモック"""
            time.sleep(0.001)  # 1ms遅延
            in_memory_db[table].extend(data)
            return {"inserted_rows": len(data)}

        mock.bulk_insert.side_effect = fast_bulk_insert

        # 接続テスト
        mock.test_connection.return_value = {"status": "connected", "latency_ms": 0.1}

        return mock


class IntegrationTestMocks:
    """統合テスト用高速モック"""

    @staticmethod
    def create_fast_comprehensive_system() -> Dict[str, Mock]:
        """包括的高速システムモック"""
        return {
            "ml_manager": MLModelMocks.create_fast_ml_model_manager(),
            "feature_manager": MLModelMocks.create_fast_feature_engineering_manager(),
            "backtest_engine": BacktestMocks.create_fast_backtest_engine(),
            "portfolio_optimizer": BacktestMocks.create_fast_portfolio_optimizer(),
            "database_manager": DatabaseMocks.create_fast_database_manager(),
        }

    @staticmethod
    def create_fast_analysis_engine() -> Mock:
        """高速分析エンジンモック"""
        mock = Mock()
        mock.name = "FastAnalysisEngine"

        def fast_comprehensive_analysis(symbols: List[str], **kwargs):
            """高速包括分析モック"""
            time.sleep(0.01)  # 10ms遅延

            results = {}
            for symbol in symbols:
                results[symbol] = {
                    "technical_analysis": {
                        "trend": np.random.choice(["bullish", "bearish", "neutral"]),
                        "strength": np.random.uniform(0, 1),
                        "rsi": np.random.uniform(20, 80),
                        "macd_signal": np.random.choice(["buy", "sell", "hold"]),
                    },
                    "fundamental_analysis": {
                        "score": np.random.uniform(1, 10),
                        "pe_ratio": np.random.uniform(10, 30),
                        "dividend_yield": np.random.uniform(0, 0.06),
                    },
                    "ml_prediction": {
                        "expected_return": np.random.normal(0.05, 0.1),
                        "confidence": np.random.uniform(0.6, 0.9),
                        "recommendation": np.random.choice(["buy", "sell", "hold"]),
                    },
                }

            return {
                "analysis_results": results,
                "execution_time_ms": 10.0,
                "symbols_analyzed": len(symbols),
                "success_rate": 1.0,
            }

        mock.comprehensive_analysis.side_effect = fast_comprehensive_analysis

        return mock


def create_performance_mock_decorator(max_execution_time_ms: float = 10.0):
    """性能制限付きモックデコレータ作成"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()

            # 最大実行時間を制限
            time.sleep(min(max_execution_time_ms / 1000, 0.001))

            result = func(*args, **kwargs)

            execution_time = (time.perf_counter() - start_time) * 1000

            # 結果に性能情報を追加
            if isinstance(result, dict):
                result["mock_execution_time_ms"] = execution_time
                result["mock_performance_limited"] = True

            return result

        return wrapper

    return decorator


# エクスポート用ファクトリ関数


def create_fast_test_environment(
    include_ml: bool = True,
    include_backtest: bool = True,
    include_database: bool = True,
    include_integration: bool = True,
    max_execution_time_ms: float = 10.0,
) -> Dict[str, Any]:
    """高速テスト環境作成"""

    environment = {
        "performance_limit_ms": max_execution_time_ms,
        "created_at": datetime.now(),
        "mock_types": [],
    }

    if include_ml:
        environment["ml_manager"] = MLModelMocks.create_fast_ml_model_manager()
        environment[
            "feature_manager"
        ] = MLModelMocks.create_fast_feature_engineering_manager()
        environment["mock_types"].append("ml_models")

    if include_backtest:
        environment["backtest_engine"] = BacktestMocks.create_fast_backtest_engine()
        environment[
            "portfolio_optimizer"
        ] = BacktestMocks.create_fast_portfolio_optimizer()
        environment["mock_types"].append("backtest")

    if include_database:
        environment["database_manager"] = DatabaseMocks.create_fast_database_manager()
        environment["mock_types"].append("database")

    if include_integration:
        environment[
            "analysis_engine"
        ] = IntegrationTestMocks.create_fast_analysis_engine()
        environment["mock_types"].append("integration")

    return environment


def measure_mock_performance(
    mock_func: Callable, *args, **kwargs
) -> MockPerformanceResult:
    """モック性能測定"""
    start_time = time.perf_counter()

    try:
        result = mock_func(*args, **kwargs)
        success = True
    except Exception:
        result = None
        success = False

    execution_time = (time.perf_counter() - start_time) * 1000

    data_points = 0
    if hasattr(result, "__len__"):
        data_points = len(result)
    elif isinstance(result, dict) and "data_points" in result:
        data_points = result["data_points"]

    return MockPerformanceResult(
        execution_time_ms=execution_time,
        success_rate=1.0 if success else 0.0,
        data_points=data_points,
        mock_used=True,
    )


if __name__ == "__main__":
    # テスト実行
    print("=== Issue #375 高速モックシステムテスト ===")

    # 高速テスト環境作成
    env = create_fast_test_environment()

    print(f"作成されたモック: {', '.join(env['mock_types'])}")

    # MLモデルテスト
    if "ml_models" in env["mock_types"]:
        print("\n1. MLモデルモックテスト")

        test_data = pd.DataFrame(
            {"feature1": np.random.randn(1000), "feature2": np.random.randn(1000)}
        )
        test_target = pd.Series(np.random.randn(1000))

        # 訓練性能測定
        ml_perf = measure_mock_performance(
            env["ml_manager"].train_model, "test_model", test_data, test_target
        )
        print(f"  ML訓練モック: {ml_perf.execution_time_ms:.2f}ms")

        # 予測性能測定
        pred_perf = measure_mock_performance(
            env["ml_manager"].predict, "test_model", test_data
        )
        print(f"  ML予測モック: {pred_perf.execution_time_ms:.2f}ms")

    # バックテストテスト
    if "backtest" in env["mock_types"]:
        print("\n2. バックテストモックテスト")

        test_data = pd.DataFrame(
            {
                "Close": np.random.randn(252) + 100,
                "Volume": np.random.randint(1000, 10000, 252),
            }
        )

        def dummy_strategy(data):
            return {"action": "hold", "quantity": 0}

        backtest_perf = measure_mock_performance(
            env["backtest_engine"].run_backtest, test_data, dummy_strategy
        )
        print(f"  バックテストモック: {backtest_perf.execution_time_ms:.2f}ms")

    # データベーステスト
    if "database" in env["mock_types"]:
        print("\n3. データベースモックテスト")

        db_perf = measure_mock_performance(
            env["database_manager"].execute_query, "SELECT * FROM stocks"
        )
        print(f"  DBクエリモック: {db_perf.execution_time_ms:.2f}ms")

    print(f"\n=== 全モック実行時間制限: {env['performance_limit_ms']}ms以下 ===")
    print("Issue #375 高速モックシステム動作確認完了")
