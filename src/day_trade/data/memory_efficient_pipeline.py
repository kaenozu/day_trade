#!/usr/bin/env python3
"""
メモリ効率データパイプライン
Issue #314: TOPIX500全銘柄対応

大規模データの効率的なストリーミング処理システム
"""

import gc
import weakref
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

try:
    from ..utils.logging_config import get_context_logger
except ImportError:

    def get_context_logger(name):
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


logger = get_context_logger(__name__)


@dataclass
class DataChunk:
    """データチャンク"""

    symbol: str
    data: pd.DataFrame
    metadata: Dict[str, Any]
    processing_stage: str = "raw"
    memory_size: int = 0

    def __post_init__(self):
        if self.memory_size == 0:
            self.memory_size = self.data.memory_usage(deep=True).sum()


class DataProcessor(ABC):
    """データ処理抽象基底クラス"""

    @abstractmethod
    def process(self, chunk: DataChunk) -> DataChunk:
        """
        データチャンク処理

        Args:
            chunk: 入力データチャンク

        Returns:
            処理済みデータチャンク
        """
        pass

    @abstractmethod
    def get_memory_usage(self) -> int:
        """メモリ使用量取得"""
        pass


class TechnicalIndicatorProcessor(DataProcessor):
    """テクニカル指標処理プロセッサ"""

    def __init__(self, indicators: List[str] = None):
        """
        初期化

        Args:
            indicators: 計算するテクニカル指標リスト
        """
        self.indicators = indicators or ["sma_5", "sma_20", "rsi", "macd"]
        self._temp_data = {}

    def process(self, chunk: DataChunk) -> DataChunk:
        """テクニカル指標計算"""
        try:
            df = chunk.data.copy()

            # SMA計算
            if "sma_5" in self.indicators:
                df["sma_5"] = df["Close"].rolling(window=5).mean()
            if "sma_20" in self.indicators:
                df["sma_20"] = df["Close"].rolling(window=20).mean()

            # RSI計算
            if "rsi" in self.indicators:
                delta = df["Close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df["rsi"] = 100 - (100 / (1 + rs))

            # MACD計算
            if "macd" in self.indicators:
                ema12 = df["Close"].ewm(span=12).mean()
                ema26 = df["Close"].ewm(span=26).mean()
                df["macd"] = ema12 - ema26
                df["macd_signal"] = df["macd"].ewm(span=9).mean()

            # NaN値処理
            df = df.fillna(method="ffill").fillna(0)

            processed_chunk = DataChunk(
                symbol=chunk.symbol,
                data=df,
                metadata={**chunk.metadata, "indicators_added": self.indicators},
                processing_stage="technical_indicators",
                memory_size=df.memory_usage(deep=True).sum(),
            )

            return processed_chunk

        except Exception as e:
            logger.error(f"テクニカル指標計算エラー ({chunk.symbol}): {e}")
            return chunk

    def get_memory_usage(self) -> int:
        """メモリ使用量取得"""
        return sum(df.memory_usage(deep=True).sum() for df in self._temp_data.values())


class StatisticalFeatureProcessor(DataProcessor):
    """統計特徴量処理プロセッサ"""

    def __init__(self, lookback_windows: List[int] = None):
        """
        初期化

        Args:
            lookback_windows: 統計計算期間リスト
        """
        self.lookback_windows = lookback_windows or [5, 10, 20]

    def process(self, chunk: DataChunk) -> DataChunk:
        """統計特徴量計算"""
        try:
            df = chunk.data.copy()

            # リターン系特徴量
            df["returns"] = df["Close"].pct_change()

            for window in self.lookback_windows:
                # ボラティリティ
                df[f"volatility_{window}"] = df["returns"].rolling(window).std()

                # 価格レンジ
                df[f"high_low_ratio_{window}"] = (
                    df["High"].rolling(window).max() - df["Low"].rolling(window).min()
                ) / df["Close"]

                # 出来高統計
                df[f"volume_mean_{window}"] = df["Volume"].rolling(window).mean()
                df[f"volume_std_{window}"] = df["Volume"].rolling(window).std()

            # NaN値処理
            df = df.fillna(method="ffill").fillna(0)

            processed_chunk = DataChunk(
                symbol=chunk.symbol,
                data=df,
                metadata={**chunk.metadata, "statistical_features_added": True},
                processing_stage="statistical_features",
                memory_size=df.memory_usage(deep=True).sum(),
            )

            return processed_chunk

        except Exception as e:
            logger.error(f"統計特徴量計算エラー ({chunk.symbol}): {e}")
            return chunk

    def get_memory_usage(self) -> int:
        """メモリ使用量取得"""
        return 0  # 一時データなし


class MemoryEfficientCache:
    """メモリ効率キャッシュシステム"""

    def __init__(self, max_memory_mb: int = 256):
        """
        初期化

        Args:
            max_memory_mb: 最大メモリ使用量（MB）
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.access_order = deque()
        self.memory_usage = 0
        self.weak_refs = weakref.WeakValueDictionary()

    def get(self, key: str) -> Optional[DataChunk]:
        """キャッシュからデータ取得"""
        if key in self.cache:
            # LRU更新
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, chunk: DataChunk):
        """キャッシュにデータ保存"""
        # メモリ制限チェック
        chunk_size = chunk.memory_size

        while self.memory_usage + chunk_size > self.max_memory_bytes and self.access_order:
            # 最も古いアイテムを削除
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                old_chunk = self.cache.pop(oldest_key)
                self.memory_usage -= old_chunk.memory_size
                del old_chunk
                gc.collect()

        # 新しいデータを追加
        if chunk_size <= self.max_memory_bytes:
            self.cache[key] = chunk
            self.access_order.append(key)
            self.memory_usage += chunk_size

    def clear(self):
        """キャッシュクリア"""
        self.cache.clear()
        self.access_order.clear()
        self.memory_usage = 0
        gc.collect()

    def get_memory_info(self) -> Dict[str, Any]:
        """メモリ情報取得"""
        return {
            "current_usage_mb": self.memory_usage / 1024 / 1024,
            "max_usage_mb": self.max_memory_bytes / 1024 / 1024,
            "usage_ratio": self.memory_usage / self.max_memory_bytes,
            "cached_items": len(self.cache),
        }


class StreamingDataPipeline:
    """ストリーミングデータパイプライン"""

    def __init__(
        self,
        processors: List[DataProcessor] = None,
        cache_size_mb: int = 256,
        chunk_size: int = 100,
    ):
        """
        初期化

        Args:
            processors: データ処理プロセッサリスト
            cache_size_mb: キャッシュサイズ（MB）
            chunk_size: チャンクサイズ
        """
        self.processors = processors or []
        self.cache = MemoryEfficientCache(cache_size_mb)
        self.chunk_size = chunk_size
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss

        logger.info(f"ストリーミングパイプライン初期化: {len(self.processors)}プロセッサ")

    def add_processor(self, processor: DataProcessor):
        """プロセッサ追加"""
        self.processors.append(processor)
        logger.info(f"プロセッサ追加: {processor.__class__.__name__}")

    @contextmanager
    def memory_monitor(self, operation_name: str):
        """メモリ監視コンテキストマネージャー"""
        start_memory = self.get_current_memory_usage()
        try:
            yield
        finally:
            end_memory = self.get_current_memory_usage()
            memory_delta = end_memory - start_memory
            if memory_delta > 50:  # 50MB以上の増加で警告
                logger.warning(f"{operation_name}: メモリ増加 {memory_delta:.1f}MB")

    def get_current_memory_usage(self) -> float:
        """現在のメモリ使用量取得（MB）"""
        current = self.process.memory_info().rss
        return (current - self.initial_memory) / 1024 / 1024

    def process_data_stream(
        self, data_generator: Iterator[Tuple[str, pd.DataFrame]]
    ) -> Generator[DataChunk, None, None]:
        """
        データストリーム処理

        Args:
            data_generator: (symbol, data)の形式のデータジェネレータ

        Yields:
            処理済みデータチャンク
        """
        processed_count = 0

        for symbol, raw_data in data_generator:
            with self.memory_monitor(f"Processing {symbol}"):
                # キャッシュチェック
                cache_key = f"{symbol}_processed"
                cached_chunk = self.cache.get(cache_key)

                if cached_chunk:
                    logger.debug(f"キャッシュヒット: {symbol}")
                    yield cached_chunk
                    continue

                # 初期チャンク作成
                chunk = DataChunk(
                    symbol=symbol,
                    data=raw_data,
                    metadata={"source": "raw", "timestamp": pd.Timestamp.now()},
                    processing_stage="raw",
                )

                # 処理パイプライン実行
                for processor in self.processors:
                    try:
                        chunk = processor.process(chunk)

                        # メモリ制限チェック
                        if self.get_current_memory_usage() > 800:  # 800MB制限
                            logger.warning("メモリ制限接近 - ガベージコレクション実行")
                            gc.collect()

                    except Exception as e:
                        logger.error(f"処理エラー ({symbol}, {processor.__class__.__name__}): {e}")
                        continue

                # キャッシュに保存
                self.cache.put(cache_key, chunk)

                processed_count += 1

                if processed_count % 50 == 0:
                    logger.info(f"処理進捗: {processed_count}件完了")
                    cache_info = self.cache.get_memory_info()
                    logger.info(f"キャッシュ使用率: {cache_info['usage_ratio']:.1%}")

                yield chunk

    def process_batch(self, symbol_data_pairs: List[Tuple[str, pd.DataFrame]]) -> List[DataChunk]:
        """
        バッチ処理

        Args:
            symbol_data_pairs: (symbol, data)のペアリスト

        Returns:
            処理済みデータチャンクリスト
        """

        def data_generator():
            yield from symbol_data_pairs

        results = []
        for chunk in self.process_data_stream(data_generator()):
            results.append(chunk)

        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """パイプライン統計情報取得"""
        processor_memory = sum(p.get_memory_usage() for p in self.processors)
        cache_info = self.cache.get_memory_info()

        return {
            "total_memory_usage_mb": self.get_current_memory_usage(),
            "processor_memory_mb": processor_memory / 1024 / 1024,
            "cache_info": cache_info,
            "active_processors": len(self.processors),
            "processor_types": [p.__class__.__name__ for p in self.processors],
        }

    def cleanup(self):
        """リソースクリーンアップ"""
        self.cache.clear()
        for processor in self.processors:
            if hasattr(processor, "cleanup"):
                processor.cleanup()
        gc.collect()
        logger.info("パイプラインクリーンアップ完了")


class DataStreamGenerator:
    """データストリームジェネレータ"""

    def __init__(self, symbols: List[str], data_source=None):
        """
        初期化

        Args:
            symbols: 銘柄リスト
            data_source: データソース（Noneの場合はモックデータ）
        """
        self.symbols = symbols
        self.data_source = data_source

    def generate_mock_data(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """モックデータ生成"""
        dates = pd.date_range(start="2023-01-01", periods=days)
        np.random.seed(hash(symbol) % 10000)  # 銘柄ごとに異なるシード

        # 基準価格を銘柄ごとに設定
        base_price = 1000 + (hash(symbol) % 3000)

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(base_price * 0.95, base_price * 1.05, days),
                "High": np.random.uniform(base_price * 1.00, base_price * 1.08, days),
                "Low": np.random.uniform(base_price * 0.92, base_price * 1.00, days),
                "Close": np.random.uniform(base_price * 0.95, base_price * 1.05, days),
                "Volume": np.random.randint(100000, 5000000, days),
            },
            index=dates,
        )

        return data

    def stream_data(self) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """データストリーム生成"""
        for symbol in self.symbols:
            try:
                if self.data_source:
                    data = self.data_source.fetch_stock_data(symbol)
                else:
                    data = self.generate_mock_data(symbol)

                if data is not None and not data.empty:
                    yield symbol, data
                else:
                    logger.warning(f"データなし: {symbol}")

            except Exception as e:
                logger.error(f"データ生成エラー ({symbol}): {e}")


if __name__ == "__main__":
    print("=== メモリ効率データパイプライン テスト ===")

    try:
        # テスト銘柄
        test_symbols = [
            "7203",
            "8306",
            "9984",
            "6758",
            "4689",
            "8058",
            "8031",
            "4568",
            "9501",
            "8801",
        ]

        print("1. パイプライン初期化...")
        pipeline = StreamingDataPipeline(
            processors=[
                TechnicalIndicatorProcessor(["sma_5", "sma_20", "rsi"]),
                StatisticalFeatureProcessor([5, 20]),
            ],
            cache_size_mb=128,
            chunk_size=50,
        )

        print("2. データストリーム生成...")
        data_generator = DataStreamGenerator(test_symbols)

        print("3. ストリーミング処理実行...")
        processed_chunks = []

        start_time = pd.Timestamp.now()
        for i, chunk in enumerate(pipeline.process_data_stream(data_generator.stream_data())):
            processed_chunks.append(chunk)

            if i % 3 == 0:
                stats = pipeline.get_pipeline_stats()
                print(f"   処理中: {chunk.symbol} (メモリ: {stats['total_memory_usage_mb']:.1f}MB)")

        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()

        print("4. 結果サマリー...")
        print(f"   処理銘柄数: {len(processed_chunks)}")
        print(f"   処理時間: {processing_time:.1f}秒")
        print(f"   スループット: {len(processed_chunks)/processing_time:.1f} 銘柄/秒")

        # パイプライン統計
        final_stats = pipeline.get_pipeline_stats()
        print(f"   最終メモリ使用量: {final_stats['total_memory_usage_mb']:.1f}MB")
        print(f"   キャッシュ使用率: {final_stats['cache_info']['usage_ratio']:.1%}")

        print("5. サンプル処理結果確認...")
        if processed_chunks:
            sample_chunk = processed_chunks[0]
            print(f"   サンプル銘柄: {sample_chunk.symbol}")
            print(f"   データ行数: {len(sample_chunk.data)}")
            print(f"   列数: {len(sample_chunk.data.columns)}")
            print(f"   処理ステージ: {sample_chunk.processing_stage}")
            print(f"   メモリサイズ: {sample_chunk.memory_size / 1024:.1f}KB")

        print("6. クリーンアップ...")
        pipeline.cleanup()

        print("\n[OK] メモリ効率データパイプライン テスト完了！")

    except Exception as e:
        print(f"[NG] テストエラー: {e}")
        import traceback

        traceback.print_exc()
