#!/usr/bin/env python3
"""
多角的データ収集システム - メインデータ管理

Issue #322: ML Data Shortage Problem Resolution
多角的データ収集管理システムのメイン実装
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .models import CollectedData
from .news_collector import NewsDataCollector
from .sentiment_analyzer import SentimentAnalyzer
from .macro_collector import MacroEconomicCollector

try:
    from ...utils.logging_config import get_context_logger
    from ...utils.unified_cache_manager import (
        UnifiedCacheManager,
        generate_unified_cache_key,
    )
    from ..stock_fetcher import StockFetcher
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO)

    def get_context_logger(name):
        return logging.getLogger(name)

    # キャッシュマネージャーのモック
    class UnifiedCacheManager:
        def __init__(self, **kwargs):
            pass

        def get(self, key, default=None):
            return default

        def put(self, key, value, **kwargs):
            return True

    def generate_unified_cache_key(*args, **kwargs):
        return f"data_key_{hash(str(args) + str(kwargs))}"

    # ストックフェッチャーのモック
    class StockFetcher:
        def fetch_stock_data(self, symbol, days=100):
            import pandas as pd
            dates = pd.date_range(start="2024-01-01", periods=days)
            return pd.DataFrame(
                {
                    "Open": np.random.uniform(2000, 3000, days),
                    "High": np.random.uniform(2100, 3100, days),
                    "Low": np.random.uniform(1900, 2900, days),
                    "Close": np.random.uniform(2000, 3000, days),
                    "Volume": np.random.randint(1000000, 10000000, days),
                },
                index=dates,
            )


logger = get_context_logger(__name__)


class MultiSourceDataManager:
    """多角的データ収集管理システム"""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_ttl_minutes: int = 10,
        max_concurrent: int = 10,
    ):
        """
        初期化

        Args:
            enable_cache: キャッシュ機能有効化
            cache_ttl_minutes: キャッシュTTL（分）
            max_concurrent: 最大同時実行数
        """
        # Issue #324統合キャッシュ連携
        if enable_cache:
            try:
                self.cache_manager = UnifiedCacheManager(
                    l1_memory_mb=32,  # データ収集用ホットキャッシュ
                    l2_memory_mb=128,  # データ統合用ウォームキャッシュ
                    l3_disk_mb=256,  # 履歴データ用コールドキャッシュ
                )
                self.cache_enabled = True
                logger.info("統合キャッシュシステム連携完了")
            except Exception as e:
                logger.warning(f"キャッシュ初期化失敗: {e}")
                self.cache_manager = None
                self.cache_enabled = False
        else:
            self.cache_manager = None
            self.cache_enabled = False

        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_concurrent = max_concurrent

        # データ収集器初期化
        self.collectors = {
            "price": StockFetcher(),
            "news": NewsDataCollector(),
            "sentiment": SentimentAnalyzer(),
            "macro": MacroEconomicCollector(),
        }

        # 統計情報
        self.collection_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "cache_hits": 0,
            "avg_collection_time": 0.0,
            "last_collection_time": None,
        }

        logger.info("多角的データ管理システム初期化完了")
        logger.info(f"  - データ収集器: {len(self.collectors)}種類")
        logger.info(f"  - キャッシュ: {'有効' if self.cache_enabled else '無効'}")
        logger.info(f"  - 最大同時実行: {max_concurrent}")

    async def collect_comprehensive_data(self, symbol: str) -> Dict[str, CollectedData]:
        """
        包括的データ収集

        Args:
            symbol: 銘柄シンボル

        Returns:
            Dict[str, CollectedData]: 収集されたデータ辞書
        """
        collection_start = time.time()

        try:
            # キャッシュチェック
            if self.cache_enabled:
                cached_data = self._check_comprehensive_cache(symbol)
                if cached_data:
                    self.collection_stats["cache_hits"] += 1
                    logger.info(f"包括データキャッシュヒット: {symbol}")
                    return cached_data

            # 並列データ収集
            collected_data = {}

            # セマフォで同時実行数制限
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def collect_with_semaphore(collector_name: str, collector):
                async with semaphore:
                    return await self._collect_single_source(
                        collector_name, collector, symbol
                    )

            # 全データ収集器を並列実行
            tasks = [
                collect_with_semaphore(name, collector)
                for name, collector in self.collectors.items()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 結果統合
            for i, (collector_name, _) in enumerate(self.collectors.items()):
                if i < len(results) and not isinstance(results[i], Exception):
                    collected_data[collector_name] = results[i]
                else:
                    logger.error(
                        f"データ収集失敗 {collector_name}: {results[i] if i < len(results) else 'Unknown'}"
                    )

            # 相互依存データの統合処理
            collected_data = self._integrate_cross_dependencies(collected_data)

            # キャッシュ保存
            if self.cache_enabled and collected_data:
                self._save_comprehensive_cache(symbol, collected_data)

            # 統計更新
            collection_time = time.time() - collection_start
            self._update_collection_stats(len(collected_data), collection_time)

            logger.info(
                f"包括データ収集完了 {symbol}: {len(collected_data)}種類 ({collection_time:.2f}秒)"
            )
            return collected_data

        except Exception as e:
            logger.error(f"包括データ収集エラー {symbol}: {e}")
            return {}

    async def _collect_single_source(
        self, collector_name: str, collector, symbol: str
    ) -> Optional[CollectedData]:
        """
        単一ソースデータ収集

        Args:
            collector_name: 収集器名
            collector: データ収集器
            symbol: 銘柄シンボル

        Returns:
            Optional[CollectedData]: 収集データ
        """
        try:
            # データ収集実行
            if collector_name == "sentiment":
                # センチメント分析はニュースデータに依存
                news_data = []  # 実際にはニュース収集結果を渡す
                data = await collector.collect_data(symbol, news_data=news_data)
            else:
                data = await collector.collect_data(symbol)

            return data

        except Exception as e:
            logger.error(f"単一ソース収集エラー {collector_name} {symbol}: {e}")
            return None

    def _integrate_cross_dependencies(
        self, collected_data: Dict[str, CollectedData]
    ) -> Dict[str, CollectedData]:
        """
        相互依存データの統合処理

        Args:
            collected_data: 収集されたデータ辞書

        Returns:
            Dict[str, CollectedData]: 統合されたデータ辞書
        """
        try:
            # ニュースデータがある場合、センチメント分析を更新
            if "news" in collected_data and "sentiment" in collected_data:
                news_data = collected_data["news"].data

                # センチメント分析を再実行（ニュースデータ付き）
                sentiment_analyzer = self.collectors.get("sentiment")
                if sentiment_analyzer and news_data:
                    # 同期版でセンチメント更新
                    updated_sentiment = self._update_sentiment_with_news(
                        collected_data["sentiment"], news_data
                    )
                    collected_data["sentiment"] = updated_sentiment

            # マクロ経済データとセクター分析の統合
            if "macro" in collected_data:
                macro_data = collected_data["macro"].data
                symbol = collected_data["macro"].symbol

                # セクター影響度の詳細計算
                enhanced_macro = self._enhance_macro_with_sector_analysis(
                    macro_data, symbol
                )
                collected_data["macro"].data = enhanced_macro

            return collected_data

        except Exception as e:
            logger.error(f"相互依存統合エラー: {e}")
            return collected_data

    def _update_sentiment_with_news(
        self, sentiment_data: CollectedData, news_data: list
    ) -> CollectedData:
        """
        ニュース付きセンチメント更新

        Args:
            sentiment_data: センチメントデータ
            news_data: ニュースデータリスト

        Returns:
            CollectedData: 更新されたセンチメントデータ
        """
        try:
            sentiment_analyzer = SentimentAnalyzer()

            # ニュース記事のセンチメント分析
            article_sentiments = []
            for article in news_data:
                text = article.get("title", "") + " " + article.get("summary", "")
                sentiment = sentiment_analyzer._analyze_sentiment(text)
                article_sentiments.append(sentiment)

            # 総合センチメント更新
            if article_sentiments:
                avg_sentiment = np.mean([s["compound"] for s in article_sentiments])
                positive_ratio = np.mean(
                    [1 if s["compound"] > 0.1 else 0 for s in article_sentiments]
                )
                negative_ratio = np.mean(
                    [1 if s["compound"] < -0.1 else 0 for s in article_sentiments]
                )

                sentiment_data.data.update(
                    {
                        "overall_sentiment": avg_sentiment,
                        "positive_ratio": positive_ratio,
                        "negative_ratio": negative_ratio,
                        "neutral_ratio": 1.0 - positive_ratio - negative_ratio,
                        "article_sentiments": article_sentiments,
                        "news_integrated": True,
                    }
                )

                # 品質スコア再計算
                sentiment_data.quality_score = (
                    sentiment_analyzer._evaluate_sentiment_quality(sentiment_data.data)
                )

            return sentiment_data

        except Exception as e:
            logger.error(f"センチメント更新エラー: {e}")
            return sentiment_data

    def _enhance_macro_with_sector_analysis(
        self, macro_data: Dict, symbol: str
    ) -> Dict:
        """
        マクロ経済データのセクター強化

        Args:
            macro_data: マクロ経済データ
            symbol: 銘柄シンボル

        Returns:
            Dict: 強化されたマクロ経済データ
        """
        try:
            # より詳細なセクター分析
            sector_sensitivity = self._calculate_detailed_sector_sensitivity(symbol)

            # マクロ指標との相関分析
            correlation_scores = self._calculate_macro_correlations(
                macro_data, sector_sensitivity
            )

            # 予測影響度計算
            impact_prediction = self._predict_macro_impact(
                macro_data, correlation_scores
            )

            macro_data.update(
                {
                    "sector_sensitivity": sector_sensitivity,
                    "correlation_scores": correlation_scores,
                    "impact_prediction": impact_prediction,
                    "enhanced": True,
                }
            )

            return macro_data

        except Exception as e:
            logger.error(f"マクロデータ強化エラー: {e}")
            return macro_data

    def _calculate_detailed_sector_sensitivity(self, symbol: str) -> Dict[str, float]:
        """詳細セクター感度計算"""
        # セクターマッピング（実際にはマスタデータベースから取得）
        sector_mappings = {
            "7203": {"sector": "automotive", "sub_sector": "manufacturing"},
            "8306": {"sector": "banking", "sub_sector": "commercial_banking"},
            "9984": {"sector": "retail", "sub_sector": "ecommerce"},
        }

        sector_info = sector_mappings.get(
            symbol, {"sector": "general", "sub_sector": "general"}
        )

        # セクター感度定義
        sensitivity_profiles = {
            "automotive": {
                "interest_rate": -0.6,
                "exchange_rate": 0.8,
                "oil_price": -0.7,
                "consumer_confidence": 0.9,
                "industrial_production": 0.8,
            },
            "banking": {
                "interest_rate": 0.9,
                "exchange_rate": 0.2,
                "inflation": 0.4,
                "credit_spread": -0.8,
                "regulatory_risk": -0.6,
            },
            "retail": {
                "interest_rate": -0.5,
                "consumer_confidence": 0.9,
                "employment": 0.8,
                "disposable_income": 0.9,
                "inflation": -0.6,
            },
            "general": {"interest_rate": 0.0, "exchange_rate": 0.0, "inflation": 0.0},
        }

        return sensitivity_profiles.get(
            sector_info["sector"], sensitivity_profiles["general"]
        )

    def _calculate_macro_correlations(
        self, macro_data: Dict, sector_sensitivity: Dict
    ) -> Dict[str, float]:
        """マクロ相関分析"""
        correlations = {}

        for macro_indicator, value in macro_data.items():
            if macro_indicator in sector_sensitivity and isinstance(
                value, (int, float)
            ):
                # 相関強度計算
                sensitivity = sector_sensitivity[macro_indicator]
                correlation = sensitivity * np.tanh(abs(value) / 10)  # 正規化
                correlations[macro_indicator] = correlation

        return correlations

    def _predict_macro_impact(
        self, macro_data: Dict, correlations: Dict
    ) -> Dict[str, Any]:
        """マクロ影響予測"""
        impact_scores = []

        for indicator, correlation in correlations.items():
            if indicator in macro_data:
                value = macro_data[indicator]
                if isinstance(value, (int, float)):
                    impact = (
                        correlation * (value - 0) / max(abs(value), 1)
                    )  # 正規化影響度
                    impact_scores.append(impact)

        if impact_scores:
            overall_impact = np.mean(impact_scores)
            impact_volatility = np.std(impact_scores)
        else:
            overall_impact = 0.0
            impact_volatility = 0.0

        return {
            "overall_impact": overall_impact,
            "impact_volatility": impact_volatility,
            "risk_level": (
                "high"
                if impact_volatility > 0.5
                else "medium" if impact_volatility > 0.2 else "low"
            ),
            "confidence": min(len(impact_scores) / 5, 1.0),
        }

    def _check_comprehensive_cache(
        self, symbol: str
    ) -> Optional[Dict[str, CollectedData]]:
        """包括キャッシュチェック"""
        if not self.cache_enabled:
            return None

        cache_key = generate_unified_cache_key(
            "multi_data",
            "comprehensive",
            symbol,
            time_bucket_minutes=self.cache_ttl_minutes,
        )

        cached_data = self.cache_manager.get(cache_key)
        return cached_data

    def _save_comprehensive_cache(self, symbol: str, data: Dict[str, CollectedData]):
        """包括キャッシュ保存"""
        if not self.cache_enabled:
            return

        cache_key = generate_unified_cache_key(
            "multi_data",
            "comprehensive",
            symbol,
            time_bucket_minutes=self.cache_ttl_minutes,
        )

        # 高優先度でキャッシュ保存（複合データは重要）
        success = self.cache_manager.put(cache_key, data, priority=9.0)

        if success:
            logger.debug(f"包括データキャッシュ保存: {symbol}")

    def _update_collection_stats(self, successful_count: int, collection_time: float):
        """収集統計更新"""
        self.collection_stats["total_requests"] += 1
        self.collection_stats["successful_requests"] += 1 if successful_count > 0 else 0

        # 移動平均で平均時間更新
        current_avg = self.collection_stats["avg_collection_time"]
        total_requests = self.collection_stats["total_requests"]

        self.collection_stats["avg_collection_time"] = (
            current_avg * (total_requests - 1) + collection_time
        ) / total_requests
        self.collection_stats["last_collection_time"] = datetime.now()

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """包括統計情報取得"""
        stats = {
            "collection_stats": self.collection_stats.copy(),
            "collector_health": {},
            "cache_stats": {},
            "system_status": "active",
        }

        # 各収集器のヘルス状態
        for name, collector in self.collectors.items():
            try:
                stats["collector_health"][name] = collector.get_health_status()
            except Exception as e:
                stats["collector_health"][name] = {"status": "error", "error": str(e)}

        # キャッシュ統計
        if self.cache_enabled and hasattr(
            self.cache_manager, "get_comprehensive_stats"
        ):
            try:
                stats["cache_stats"] = self.cache_manager.get_comprehensive_stats()
            except Exception as e:
                stats["cache_stats"] = {"error": str(e)}

        return stats

    async def shutdown(self):
        """システムシャットダウン"""
        logger.info("多角的データ管理システムシャットダウン開始")

        # 非同期セッションクリーンアップ
        for collector in self.collectors.values():
            if hasattr(collector, "session") and collector.session:
                try:
                    await collector.session.close()
                except Exception as e:
                    logger.warning(f"セッションクローズエラー: {e}")

        logger.info("多角的データ管理システムシャットダウン完了")