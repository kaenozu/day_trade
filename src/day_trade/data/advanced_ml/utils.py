#!/usr/bin/env python3
"""
Advanced ML Engine Utilities Module

ユーティリティ関数とファクトリ関数
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...ml.hybrid_lstm_transformer import HybridModelConfig
from ...utils.logging_config import get_context_logger
from .config import ModelConfig
from .ml_engine import AdvancedMLEngine
from .next_gen_engine import NextGenAITradingEngine

logger = get_context_logger(__name__)


def create_advanced_ml_engine(config_dict: Optional[Dict] = None) -> AdvancedMLEngine:
    """Advanced ML Engine インスタンス作成"""
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        config = ModelConfig()

    return AdvancedMLEngine(config)


def create_next_gen_engine(config: Optional[Dict] = None) -> NextGenAITradingEngine:
    """次世代AIエンジンファクトリ関数"""
    if config:
        hybrid_config = HybridModelConfig(**config)
    else:
        hybrid_config = HybridModelConfig()

    return NextGenAITradingEngine(hybrid_config)


async def extract_fft_features_async(
    prices: pd.Series, n_features: int = 10
) -> List[pd.Series]:
    """
    非同期FFT特徴量抽出 - Issue #707対応

    CPU集約的な処理を非同期処理で最適化
    """

    def _compute_fft_chunk(chunk_data):
        """FFTチャンク計算"""
        try:
            fft_chunk = np.fft.fft(chunk_data)
            return np.abs(fft_chunk[1 : n_features + 1])
        except Exception as e:
            logger.warning(f"FFTチャンク計算エラー: {e}")
            return np.zeros(n_features)

    try:
        prices_clean = prices.dropna()
        if len(prices_clean) < n_features:
            return [
                pd.Series([0.0] * len(prices), index=prices.index)
                for _ in range(n_features)
            ]

        # チャンクサイズ計算
        chunk_size = max(256, len(prices_clean) // 4)  # 最小256、最大4分割
        chunks = [
            prices_clean.values[i : i + chunk_size]
            for i in range(0, len(prices_clean), chunk_size)
        ]

        # 非同期でFFT処理
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = [
                loop.run_in_executor(executor, _compute_fft_chunk, chunk)
                for chunk in chunks
                if len(chunk) >= n_features
            ]

            if tasks:
                results = await asyncio.gather(*tasks)
                # 結果をマージ（平均）
                amplitudes = np.mean(results, axis=0)
            else:
                # フォールバック: 同期処理
                fft = np.fft.fft(prices_clean.values)
                amplitudes = np.abs(fft[1 : n_features + 1])

        # 効率的な特徴量系列作成
        fft_features = [
            pd.Series(
                [amplitude] * len(prices), index=prices.index, dtype=np.float32
            )
            for amplitude in amplitudes
        ]

        return fft_features

    except Exception as e:
        logger.warning(f"非同期FFT特徴量抽出エラー: {e}")
        # フォールバック: ゼロ特徴量
        return [
            pd.Series([0.0] * len(prices), index=prices.index)
            for _ in range(n_features)
        ]


async def measure_inference_time_async(
    hybrid_model, test_data: pd.DataFrame, n_iterations: int = 10
) -> Optional[float]:
    """
    非同期推論時間測定 - Issue #707対応

    Args:
        hybrid_model: ハイブリッドモデル
        test_data: テストデータ
        n_iterations: 測定回数

    Returns:
        平均推論時間(ms)、エラー時はNone
    """

    async def async_inference():
        """非同期推論実行"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            start = time.time()
            try:
                result = await loop.run_in_executor(
                    executor, hybrid_model.predict, test_data.tail(10)
                )
                return (time.time() - start) * 1000  # ms変換
            except Exception as e:
                logger.warning(f"非同期推論時間測定エラー: {e}")
                return None

    try:
        # 非同期推論時間測定
        tasks = [async_inference() for _ in range(n_iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 有効な結果のみ抽出
        inference_times = [
            result
            for result in results
            if isinstance(result, (int, float)) and result is not None
        ]

        if inference_times:
            avg_time = np.mean(inference_times)
            logger.info(
                f"非同期推論時間測定完了: {len(inference_times)}回測定、"
                f"平均{avg_time:.2f}ms"
            )
            return avg_time
        else:
            return None

    except Exception as e:
        logger.warning(f"非同期推論時間測定失敗: {e}")
        return None


def validate_model_config(config: ModelConfig) -> bool:
    """モデル設定の検証"""
    try:
        # 基本パラメータ検証
        if config.sequence_length <= 0:
            logger.error("sequence_lengthは正の値である必要があります")
            return False

        if config.num_features <= 0:
            logger.error("num_featuresは正の値である必要があります")
            return False

        if config.batch_size <= 0:
            logger.error("batch_sizeは正の値である必要があります")
            return False

        if not (0.0 < config.learning_rate < 1.0):
            logger.error("learning_rateは0と1の間である必要があります")
            return False

        # LSTM設定検証
        if config.lstm_hidden_size <= 0:
            logger.error("lstm_hidden_sizeは正の値である必要があります")
            return False

        if config.lstm_num_layers <= 0:
            logger.error("lstm_num_layersは正の値である必要があります")
            return False

        if not (0.0 <= config.lstm_dropout <= 1.0):
            logger.error("lstm_dropoutは0と1の間である必要があります")
            return False

        # Transformer設定検証
        if config.transformer_d_model <= 0:
            logger.error("transformer_d_modelは正の値である必要があります")
            return False

        if config.transformer_nhead <= 0:
            logger.error("transformer_nheadは正の値である必要があります")
            return False

        if config.transformer_d_model % config.transformer_nhead != 0:
            logger.error(
                "transformer_d_modelはtransformer_nheadで割り切れる必要があります"
            )
            return False

        logger.info("モデル設定検証完了: すべてのパラメータが有効です")
        return True

    except Exception as e:
        logger.error(f"モデル設定検証エラー: {e}")
        return False


def optimize_config_for_hardware(config: ModelConfig) -> ModelConfig:
    """ハードウェアに最適化された設定を生成"""
    try:
        import psutil

        # メモリ使用量を取得
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)

        # CPU数を取得
        cpu_count = psutil.cpu_count()

        logger.info(
            f"システム情報: メモリ {available_memory_gb:.1f}GB, "
            f"CPU {cpu_count}コア"
        )

        # メモリ制約に基づく最適化
        if available_memory_gb < 4:
            # 低メモリ環境
            config.batch_size = min(config.batch_size, 16)
            config.sequence_length = min(config.sequence_length, 500)
            config.lstm_hidden_size = min(config.lstm_hidden_size, 128)
            config.transformer_d_model = min(config.transformer_d_model, 256)
            logger.info("低メモリ環境に最適化")

        elif available_memory_gb < 8:
            # 中メモリ環境
            config.batch_size = min(config.batch_size, 32)
            config.sequence_length = min(config.sequence_length, 750)
            config.lstm_hidden_size = min(config.lstm_hidden_size, 256)
            config.transformer_d_model = min(config.transformer_d_model, 512)
            logger.info("中メモリ環境に最適化")

        else:
            # 高メモリ環境
            logger.info("十分なメモリがあります。設定はそのまま使用")

        # CPU数に基づく最適化
        if cpu_count <= 2:
            # 低CPU環境
            config.num_epochs = min(config.num_epochs, 50)
            logger.info("低CPU環境: エポック数を削減")

        return config

    except ImportError:
        logger.warning("psutilが利用できません。最適化をスキップ")
        return config
    except Exception as e:
        logger.warning(f"ハードウェア最適化エラー: {e}")
        return config


def get_recommended_config(data_size: int, feature_count: int) -> ModelConfig:
    """データサイズに基づく推奨設定を生成"""
    config = ModelConfig()

    try:
        # データサイズに基づく調整
        if data_size < 1000:
            # 小規模データ
            config.sequence_length = min(100, data_size // 4)
            config.batch_size = 8
            config.num_epochs = 50
            config.lstm_hidden_size = 64
            config.transformer_d_model = 128
            logger.info("小規模データ用設定を適用")

        elif data_size < 10000:
            # 中規模データ
            config.sequence_length = min(200, data_size // 10)
            config.batch_size = 16
            config.num_epochs = 75
            config.lstm_hidden_size = 128
            config.transformer_d_model = 256
            logger.info("中規模データ用設定を適用")

        else:
            # 大規模データ
            config.sequence_length = min(500, data_size // 20)
            config.batch_size = 32
            config.num_epochs = 100
            config.lstm_hidden_size = 256
            config.transformer_d_model = 512
            logger.info("大規模データ用設定を適用")

        # 特徴量数に基づく調整
        config.num_features = min(feature_count, config.num_features)

        # ハードウェア最適化
        config = optimize_config_for_hardware(config)

        return config

    except Exception as e:
        logger.error(f"推奨設定生成エラー: {e}")
        return ModelConfig()  # デフォルト設定を返す


def benchmark_model_performance(
    model_engine: AdvancedMLEngine, test_data: pd.DataFrame, iterations: int = 100
) -> Dict[str, float]:
    """モデル性能ベンチマーク"""
    try:
        if model_engine.model is None:
            logger.error("モデルが初期化されていません")
            return {}

        # テストデータ準備
        X_test, _ = model_engine.prepare_data(test_data)

        if len(X_test) == 0:
            logger.error("テストデータが空です")
            return {}

        # ウォームアップ
        for _ in range(5):
            _ = model_engine.predict(X_test[:1])

        # ベンチマーク実行
        inference_times = []
        memory_usage = []

        for i in range(iterations):
            start_time = time.time()

            try:
                _ = model_engine.predict(X_test[:1])
                inference_time = (time.time() - start_time) * 1000  # ms変換
                inference_times.append(inference_time)

                # メモリ使用量測定（オプション）
                try:
                    import psutil

                    process = psutil.Process()
                    memory_usage.append(process.memory_info().rss / 1024**2)  # MB
                except ImportError:
                    pass

            except Exception as e:
                logger.warning(f"ベンチマーク反復 {i} でエラー: {e}")

        if not inference_times:
            return {}

        # 統計計算
        results = {
            "avg_inference_time_ms": np.mean(inference_times),
            "median_inference_time_ms": np.median(inference_times),
            "min_inference_time_ms": np.min(inference_times),
            "max_inference_time_ms": np.max(inference_times),
            "std_inference_time_ms": np.std(inference_times),
            "throughput_predictions_per_sec": 1000 / np.mean(inference_times),
            "iterations_completed": len(inference_times),
        }

        if memory_usage:
            results.update(
                {
                    "avg_memory_usage_mb": np.mean(memory_usage),
                    "max_memory_usage_mb": np.max(memory_usage),
                    "min_memory_usage_mb": np.min(memory_usage),
                }
            )

        logger.info(f"ベンチマーク完了: {len(inference_times)} 回実行")
        logger.info(f"平均推論時間: {results['avg_inference_time_ms']:.2f}ms")

        return results

    except Exception as e:
        logger.error(f"ベンチマーク実行エラー: {e}")
        return {}