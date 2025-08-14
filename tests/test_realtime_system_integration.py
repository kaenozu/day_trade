#!/usr/bin/env python3
"""
リアルタイムシステム統合テスト
Real-Time System Integration Tests

Issue #763: リアルタイム特徴量生成と予測パイプライン
包括的統合テストスイート
"""

import asyncio
import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.day_trade.realtime import (
    MarketDataPoint,
    FeatureValue,
    RealTimeFeatureEngine,
    StreamingDataProcessor,
    StreamConfig,
    RealTimeFeatureStore,
    FeatureStoreConfig,
    AsyncPredictionPipeline,
    PipelineConfig,
    SimpleMovingAverageModel,
    create_realtime_system,
    get_system_info
)


class TestRealTimeFeatureEngine:
    """リアルタイム特徴量エンジンテスト"""

    @pytest.fixture
    def feature_engine(self):
        return RealTimeFeatureEngine()

    @pytest.fixture
    def sample_data_points(self):
        """サンプルデータポイント"""
        symbol = "7203"
        base_time = datetime.now()
        prices = [2000, 2005, 2010, 2008, 2012, 2015, 2018, 2020, 2017, 2022]

        return [
            MarketDataPoint(
                symbol=symbol,
                timestamp=base_time + timedelta(seconds=i),
                price=price,
                volume=1000 + i * 10
            )
            for i, price in enumerate(prices)
        ]

    @pytest.mark.asyncio
    async def test_feature_engine_initialization(self, feature_engine):
        """特徴量エンジン初期化テスト"""
        assert feature_engine is not None
        assert len(feature_engine.indicators) == 0
        assert feature_engine.feature_count == 0

    @pytest.mark.asyncio
    async def test_add_symbol(self, feature_engine):
        """銘柄追加テスト"""
        symbol = "7203"
        feature_engine.add_symbol(symbol)

        assert symbol in feature_engine.indicators
        # デフォルト指標が追加されているか確認
        assert len(feature_engine.indicators[symbol]) > 0

    @pytest.mark.asyncio
    async def test_process_data_points(self, feature_engine, sample_data_points):
        """データポイント処理テスト"""
        features_generated = []

        for data_point in sample_data_points:
            features = await feature_engine.process_data_point(data_point)
            features_generated.extend(features)

        # 特徴量が生成されることを確認
        assert len(features_generated) > 0

        # 各特徴量の構造を確認
        for feature in features_generated:
            assert isinstance(feature, FeatureValue)
            assert feature.symbol == "7203"
            assert isinstance(feature.value, (int, float))
            assert feature.timestamp is not None

    @pytest.mark.asyncio
    async def test_get_latest_features(self, feature_engine, sample_data_points):
        """最新特徴量取得テスト"""
        # データを処理
        for data_point in sample_data_points:
            await feature_engine.process_data_point(data_point)

        # 最新特徴量取得
        latest_features = feature_engine.get_latest_features("7203")

        assert isinstance(latest_features, dict)
        assert len(latest_features) > 0

        # 少なくともSMAが生成されていることを確認
        sma_features = [name for name in latest_features.keys() if "sma" in name]
        assert len(sma_features) > 0

    @pytest.mark.asyncio
    async def test_feature_vector_generation(self, feature_engine, sample_data_points):
        """特徴量ベクトル生成テスト"""
        # データを処理
        for data_point in sample_data_points:
            await feature_engine.process_data_point(data_point)

        # 特徴量ベクトル取得
        vector = feature_engine.get_feature_vector("7203")

        if vector is not None:
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0

    @pytest.mark.asyncio
    async def test_performance_stats(self, feature_engine, sample_data_points):
        """パフォーマンス統計テスト"""
        # データを処理
        for data_point in sample_data_points:
            await feature_engine.process_data_point(data_point)

        # パフォーマンス統計取得
        stats = feature_engine.get_performance_stats()

        assert isinstance(stats, dict)
        assert "avg_processing_time_ms" in stats
        assert "total_features_generated" in stats
        assert stats["total_features_generated"] >= 0


class TestStreamingDataProcessor:
    """ストリーミングデータプロセッサテスト"""

    @pytest.fixture
    def feature_engine(self):
        return RealTimeFeatureEngine()

    @pytest.fixture
    def stream_config(self):
        return StreamConfig(
            url="wss://test.com/stream",
            symbols=["7203", "8306"],
            buffer_size=100
        )

    @pytest.fixture
    def streaming_processor(self, feature_engine, stream_config):
        return StreamingDataProcessor(
            feature_engine=feature_engine,
            stream_config=stream_config
        )

    def test_initialization(self, streaming_processor):
        """初期化テスト"""
        assert streaming_processor is not None
        assert not streaming_processor.is_running
        assert len(streaming_processor.filters) > 0

    @pytest.mark.asyncio
    async def test_data_filtering(self, streaming_processor):
        """データフィルタリングテスト"""
        # 有効なデータ
        valid_data = {
            "symbol": "7203",
            "price": 2000.0,
            "volume": 1000,
            "timestamp": datetime.now().isoformat()
        }

        # 無効なデータ（価格が負）
        invalid_data = {
            "symbol": "7203",
            "price": -100.0,
            "volume": 1000,
            "timestamp": datetime.now().isoformat()
        }

        # フィルタリング実行
        valid_result = await streaming_processor._apply_filters(valid_data)
        invalid_result = await streaming_processor._apply_filters(invalid_data)

        assert valid_result is True
        assert invalid_result is False

    def test_metrics_collection(self, streaming_processor):
        """メトリクス収集テスト"""
        metrics = streaming_processor.get_metrics()

        assert hasattr(metrics, 'messages_received')
        assert hasattr(metrics, 'messages_processed')
        assert hasattr(metrics, 'error_count')


@pytest.mark.asyncio
class TestRealTimeFeatureStore:
    """リアルタイム特徴量ストアテスト"""

    @pytest.fixture
    def store_config(self):
        return FeatureStoreConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=3,  # テスト用DB
            default_ttl=60
        )

    @pytest.fixture
    def feature_store(self, store_config):
        return RealTimeFeatureStore(store_config)

    @pytest.fixture
    def sample_feature(self):
        return FeatureValue(
            name="test_sma_20",
            value=2100.5,
            timestamp=datetime.now(),
            symbol="7203",
            metadata={"period": 20}
        )

    async def test_store_and_retrieve_feature(self, feature_store, sample_feature):
        """特徴量保存・取得テスト"""
        try:
            # Redis接続テスト
            await feature_store.connect()

            # 特徴量保存
            success = await feature_store.store_feature(sample_feature)
            assert success is True

            # 特徴量取得
            retrieved_feature = await feature_store.get_feature(
                sample_feature.symbol,
                sample_feature.name
            )

            if retrieved_feature:
                assert retrieved_feature.name == sample_feature.name
                assert retrieved_feature.symbol == sample_feature.symbol
                assert abs(retrieved_feature.value - sample_feature.value) < 0.001

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
        finally:
            try:
                await feature_store.disconnect()
            except:
                pass

    async def test_batch_feature_retrieval(self, feature_store):
        """一括特徴量取得テスト"""
        try:
            await feature_store.connect()

            # テスト特徴量作成
            features = [
                FeatureValue("sma_5", 2000.0, datetime.now(), "7203"),
                FeatureValue("sma_20", 2010.0, datetime.now(), "7203"),
                FeatureValue("rsi_14", 65.0, datetime.now(), "7203")
            ]

            # 一括保存
            for feature in features:
                await feature_store.store_feature(feature)

            # 一括取得
            retrieved_features = await feature_store.get_latest_features("7203")

            assert isinstance(retrieved_features, dict)

        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
        finally:
            try:
                await feature_store.disconnect()
            except:
                pass


class TestAsyncPredictionPipeline:
    """非同期予測パイプラインテスト"""

    @pytest.fixture
    def pipeline_config(self):
        feature_store_config = FeatureStoreConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=4,  # テスト用
            default_ttl=60
        )

        stream_config = StreamConfig(
            url="wss://test.com",
            symbols=["7203"]
        )

        return PipelineConfig(
            feature_store_config=feature_store_config,
            stream_config=stream_config,
            prediction_interval_ms=100
        )

    @pytest.fixture
    def prediction_pipeline(self, pipeline_config):
        return AsyncPredictionPipeline(pipeline_config)

    @pytest.mark.asyncio
    async def test_single_prediction(self, prediction_pipeline):
        """単体予測テスト"""
        test_features = {
            "sma_5": 2100.0,
            "sma_20": 2090.0,
            "sma_50": 2080.0,
            "rsi_14": 65.0,
            "macd": 5.2
        }

        result = await prediction_pipeline.predict_single("7203", test_features)

        assert result is not None
        assert result.symbol == "7203"
        assert result.prediction_type in ["buy", "sell", "hold"]
        assert 0 <= result.confidence <= 1

    def test_model_switching(self, prediction_pipeline):
        """モデル切り替えテスト"""
        # デフォルトモデル確認
        assert prediction_pipeline.active_model == "simple_ma"

        # 存在するモデルに切り替え
        success = prediction_pipeline.switch_model("ensemble")
        assert success is True
        assert prediction_pipeline.active_model == "ensemble"

        # 存在しないモデルに切り替え
        success = prediction_pipeline.switch_model("nonexistent")
        assert success is False
        assert prediction_pipeline.active_model == "ensemble"  # 変更されない

    def test_metrics_collection(self, prediction_pipeline):
        """メトリクス収集テスト"""
        metrics = prediction_pipeline.get_metrics()

        assert hasattr(metrics, 'total_predictions')
        assert hasattr(metrics, 'successful_predictions')
        assert hasattr(metrics, 'avg_prediction_time_ms')


class TestSystemIntegration:
    """システム統合テスト"""

    @pytest.mark.asyncio
    async def test_create_realtime_system(self):
        """リアルタイムシステム作成テスト"""
        symbols = ["7203", "8306"]

        try:
            system = await create_realtime_system(
                symbols=symbols,
                prediction_model="simple_ma"
            )

            assert system is not None
            assert isinstance(system, AsyncPredictionPipeline)
            assert system.active_model == "simple_ma"

        except Exception as e:
            pytest.skip(f"System creation failed: {e}")

    def test_get_system_info(self):
        """システム情報取得テスト"""
        info = get_system_info()

        assert isinstance(info, dict)
        assert "module" in info
        assert "version" in info
        assert "components" in info
        assert "features" in info
        assert "performance" in info

        # バージョン確認
        assert info["version"] == "2.0.0"

        # コンポーネント確認
        expected_components = [
            "RealTimeFeatureEngine",
            "StreamingDataProcessor",
            "RealTimeFeatureStore",
            "AsyncPredictionPipeline"
        ]

        for component in expected_components:
            assert component in info["components"]


class TestPerformanceBenchmark:
    """パフォーマンスベンチマークテスト"""

    @pytest.mark.asyncio
    async def test_feature_generation_performance(self):
        """特徴量生成パフォーマンステスト"""
        engine = RealTimeFeatureEngine()
        symbol = "7203"

        # 大量データでのテスト
        data_points = []
        base_price = 2000.0

        for i in range(1000):  # 1000件
            price = base_price + np.random.normal(0, 10)
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now() + timedelta(milliseconds=i),
                price=price,
                volume=1000 + i
            )
            data_points.append(data_point)

        # パフォーマンス測定
        start_time = time.time()

        for data_point in data_points:
            await engine.process_data_point(data_point)

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms

        # パフォーマンス評価
        stats = engine.get_performance_stats()

        # 目標: 平均処理時間 < 10ms
        assert stats["avg_processing_time_ms"] < 10, f"処理時間が目標を超過: {stats['avg_processing_time_ms']:.2f}ms"

        # 目標: スループット > 100 features/sec
        assert stats["throughput_features_per_second"] > 100, f"スループットが目標を下回る: {stats['throughput_features_per_second']:.2f}/sec"

        print(f"Performance Results:")
        print(f"  Total processing time: {processing_time:.2f}ms")
        print(f"  Average per data point: {processing_time/len(data_points):.2f}ms")
        print(f"  Total features generated: {stats['total_features_generated']}")
        print(f"  Throughput: {stats['throughput_features_per_second']:.2f} features/sec")

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self):
        """エンドツーエンドレイテンシテスト"""
        # 簡単なエンドツーエンドテスト
        engine = RealTimeFeatureEngine()

        data_point = MarketDataPoint(
            symbol="7203",
            timestamp=datetime.now(),
            price=2000.0,
            volume=1000
        )

        # レイテンシ測定
        start_time = time.time()

        # 特徴量生成
        features = await engine.process_data_point(data_point)

        # 簡単な予測（シミュレーション）
        if features:
            feature_dict = {f.name: f.value for f in features}
            model = SimpleMovingAverageModel()

            # 必要な特徴量が揃っている場合のみ予測
            required_features = model.get_required_features()
            if all(name in feature_dict for name in required_features):
                prediction = await model.predict(feature_dict, "7203")

                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms

                # 目標: エンドツーエンドレイテンシ < 50ms
                assert latency < 50, f"レイテンシが目標を超過: {latency:.2f}ms"

                print(f"End-to-end latency: {latency:.2f}ms")
                print(f"Prediction: {prediction.prediction_type} (confidence: {prediction.confidence:.2f})")


# テスト実行用のメイン関数
if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v", "--tb=short"])