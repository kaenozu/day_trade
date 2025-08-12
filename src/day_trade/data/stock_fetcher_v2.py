"""
統合株価データ取得システム（リファクタリング版）

モジュラー設計による高性能・高信頼性データ取得インターフェース
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..utils.logging_config import get_context_logger
from .cache import CachePerformanceMonitor
from .fetchers import BulkFetcher, YFinanceFetcher

logger = get_context_logger(__name__)


class StockFetcher:
    """
    統合株価データ取得クラス（リファクタリング版）

    全ての株価データ取得機能を統合した包括的なインターフェース
    """

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        max_workers: int = 10,
        enable_performance_monitoring: bool = True,
    ):
        """
        初期化

        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
            max_workers: 最大並行ワーカー数
            enable_performance_monitoring: パフォーマンス監視有効化
        """
        # コア機能の初期化
        self.yfinance_fetcher = YFinanceFetcher(
            cache_size=cache_size,
            price_cache_ttl=price_cache_ttl,
            historical_cache_ttl=historical_cache_ttl,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )

        self.bulk_fetcher = BulkFetcher(
            base_fetcher=self.yfinance_fetcher,
            max_workers=max_workers,
        )

        # パフォーマンス監視
        if enable_performance_monitoring:
            self.performance_monitor = CachePerformanceMonitor()
        else:
            self.performance_monitor = None

        # 設定情報の保存
        self.configuration = {
            "cache_size": cache_size,
            "price_cache_ttl": price_cache_ttl,
            "historical_cache_ttl": historical_cache_ttl,
            "retry_count": retry_count,
            "retry_delay": retry_delay,
            "max_workers": max_workers,
            "performance_monitoring": enable_performance_monitoring,
        }

        logger.info(
            f"StockFetcher v2 初期化完了: "
            f"cache_size={cache_size}, workers={max_workers}, "
            f"monitoring={enable_performance_monitoring}"
        )

    # ========== 単一銘柄データ取得 ==========

    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        現在価格取得

        Args:
            code: 銘柄コード

        Returns:
            価格情報辞書
        """
        try:
            result = self.yfinance_fetcher.get_current_price(code)

            # パフォーマンス監視
            if self.performance_monitor:
                self.performance_monitor.record_function_call(
                    "get_current_price",
                    cache_hit=(result is not None),
                    response_time=0.0,  # 実際の実装では応答時間を測定
                )

            return result

        except Exception as e:
            logger.error(f"現在価格取得エラー: {code} - {e}")
            if self.performance_monitor:
                self.performance_monitor.record_function_call(
                    "get_current_price", cache_hit=False, response_time=0.0, error=e
                )
            return None

    def get_historical_data(
        self,
        code: str,
        period: str = "1mo",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        履歴データ取得

        Args:
            code: 銘柄コード
            period: 期間
            interval: 間隔

        Returns:
            履歴データのDataFrame
        """
        return self.yfinance_fetcher.get_historical_data(code, period, interval)

    def get_historical_data_range(
        self,
        code: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """日付範囲指定での履歴データ取得"""
        return self.yfinance_fetcher.get_historical_data_range(code, start_date, end_date, interval)

    def get_company_info(self, code: str) -> Optional[Dict[str, Any]]:
        """企業情報取得"""
        return self.yfinance_fetcher.get_company_info(code)

    # ========== 複数銘柄データ取得 ==========

    def get_realtime_data(self, codes: List[str]) -> Dict[str, Dict[str, float]]:
        """複数銘柄のリアルタイムデータを取得"""
        return self.yfinance_fetcher.get_realtime_data(codes)

    def bulk_get_current_prices(
        self,
        codes: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_optimization: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        複数銘柄の現在価格を一括取得

        Args:
            codes: 銘柄コードリスト
            progress_callback: 進捗コールバック関数
            use_optimization: 最適化機能使用

        Returns:
            銘柄別価格データ辞書
        """
        if use_optimization and len(codes) > 100:
            return self.bulk_fetcher.bulk_get_current_prices_optimized(codes)
        else:
            return self.bulk_fetcher.bulk_get_current_prices(codes, progress_callback)

    def bulk_get_company_info(
        self,
        codes: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """複数銘柄の企業情報を一括取得"""
        return self.bulk_fetcher.bulk_get_company_info(codes, progress_callback)

    # ========== キャッシュ・パフォーマンス管理 ==========

    def clear_all_caches(self) -> None:
        """全てのキャッシュをクリア"""
        self.yfinance_fetcher.clear_all_caches()
        logger.info("全キャッシュクリア完了")

    def get_cache_performance_report(self) -> Dict[str, Any]:
        """包括的なキャッシュパフォーマンスレポートを取得"""
        try:
            yfinance_report = self.yfinance_fetcher.get_cache_performance_report()
            bulk_stats = self.bulk_fetcher.get_bulk_stats()

            report = {
                "yfinance_cache_performance": yfinance_report,
                "bulk_processing_stats": bulk_stats,
                "system_configuration": self.configuration,
                "report_timestamp": datetime.now().isoformat(),
            }

            # パフォーマンス監視データを追加
            if self.performance_monitor:
                report["performance_monitoring"] = {
                    "overall_performance": self.performance_monitor.get_overall_performance(),
                    "top_functions": self.performance_monitor.get_top_functions(5),
                    "recent_alerts": self.performance_monitor.check_alerts(),
                }

            return report

        except Exception as e:
            logger.error(f"パフォーマンスレポート取得エラー: {e}")
            return {"error": str(e)}

    def optimize_performance(self) -> Dict[str, Any]:
        """パフォーマンス最適化の実行と提案"""
        try:
            recommendations = []
            applied_optimizations = []

            # YFinanceキャッシュ最適化
            yf_optimization = self.yfinance_fetcher.optimize_cache_settings()
            if yf_optimization.get("recommendations"):
                recommendations.extend(yf_optimization["recommendations"])

            # バルク処理設定の最適化（平均的な処理量を想定）
            avg_symbols = 100  # 平均処理銘柄数の想定
            bulk_optimization = self.bulk_fetcher.optimize_batch_settings(avg_symbols)

            if bulk_optimization.get("recommendations"):
                recommendations.extend(bulk_optimization["recommendations"])
                # 最適化設定を自動適用
                self.bulk_fetcher.apply_optimized_settings(bulk_optimization)
                applied_optimizations.append("バルク処理設定の最適化を適用")

            optimization_result = {
                "recommendations": recommendations,
                "applied_optimizations": applied_optimizations,
                "yfinance_cache_analysis": yf_optimization,
                "bulk_processing_analysis": bulk_optimization,
                "optimization_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"パフォーマンス最適化完了: {len(applied_optimizations)}項目適用")
            return optimization_result

        except Exception as e:
            logger.error(f"パフォーマンス最適化エラー: {e}")
            return {"error": str(e)}

    # ========== システム監視・管理 ==========

    def health_check(self) -> Dict[str, Any]:
        """システム全体のヘルスチェック"""
        try:
            yf_health = self.yfinance_fetcher.health_check()
            bulk_health = self.bulk_fetcher.health_check()

            # 統合ヘルススコア計算
            overall_score = (yf_health["score"] + bulk_health["score"]) / 2
            all_issues = yf_health["issues"] + bulk_health["issues"]

            overall_status = "healthy"
            if overall_score < 60:
                overall_status = "critical"
            elif overall_score < 80:
                overall_status = "warning"

            health_report = {
                "overall_status": overall_status,
                "overall_score": overall_score,
                "all_issues": all_issues,
                "component_health": {
                    "yfinance_fetcher": yf_health,
                    "bulk_fetcher": bulk_health,
                },
                "system_info": {
                    "configuration": self.configuration,
                    "uptime_info": "N/A",  # 実装に応じて追加
                },
                "health_check_timestamp": datetime.now().isoformat(),
            }

            # パフォーマンス監視アラート
            if self.performance_monitor:
                recent_alerts = self.performance_monitor.check_alerts()
                if recent_alerts:
                    health_report["performance_alerts"] = recent_alerts

            return health_report

        except Exception as e:
            logger.error(f"ヘルスチェックエラー: {e}")
            return {
                "overall_status": "error",
                "overall_score": 0,
                "error": str(e),
                "health_check_timestamp": datetime.now().isoformat(),
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """システム統計情報の取得"""
        try:
            yf_stats = self.yfinance_fetcher.get_performance_summary()
            bulk_stats = self.bulk_fetcher.get_bulk_stats()

            system_stats = {
                "yfinance_performance": yf_stats,
                "bulk_processing": bulk_stats,
                "configuration": self.configuration,
                "stats_timestamp": datetime.now().isoformat(),
            }

            # パフォーマンス監視統計
            if self.performance_monitor:
                system_stats["performance_monitoring"] = {
                    "overall_stats": self.performance_monitor.get_overall_performance(),
                    "function_stats": {
                        func: self.performance_monitor.get_function_performance(func)
                        for func in [
                            "get_current_price",
                            "get_historical_data",
                            "get_company_info",
                        ]
                    },
                    "performance_trend": self.performance_monitor.get_performance_trend(30),
                }

            return system_stats

        except Exception as e:
            logger.error(f"システム統計取得エラー: {e}")
            return {"error": str(e)}

    def reset_all_stats(self) -> None:
        """全統計情報のリセット"""
        try:
            self.yfinance_fetcher.reset_stats()
            self.bulk_fetcher.reset_bulk_stats()

            if self.performance_monitor:
                self.performance_monitor.reset_stats()

            logger.info("全統計情報リセット完了")

        except Exception as e:
            logger.error(f"統計リセットエラー: {e}")

    # ========== 便利メソッド ==========

    def format_symbol(self, code: str, market: str = "T") -> str:
        """銘柄コード形式変換（公開メソッド）"""
        return self.yfinance_fetcher._format_symbol(code, market)

    def validate_symbol(self, symbol: str) -> bool:
        """銘柄コード検証"""
        try:
            self.yfinance_fetcher._validate_symbol(symbol)
            return True
        except Exception:
            return False

    def get_retry_stats(self) -> Dict[str, Any]:
        """リトライ統計の取得"""
        return self.yfinance_fetcher.get_retry_stats()

    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        # 必要に応じてクリーンアップ処理
        logger.info("StockFetcher終了")

    def __repr__(self) -> str:
        """オブジェクトの文字列表現"""
        return (
            f"StockFetcher(cache_size={self.configuration['cache_size']}, "
            f"workers={self.configuration['max_workers']}, "
            f"monitoring={'enabled' if self.performance_monitor else 'disabled'})"
        )
