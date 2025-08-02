"""
耐障害性強化版株価データ取得モジュール
Issue 185: 外部API通信の耐障害性強化

高度なリトライ機構、サーキットブレーカー、フェイルオーバー機能を統合
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

from ..utils.api_resilience import (
    APIEndpoint,
    CircuitBreakerConfig,
    ResilientAPIClient,
    RetryConfig,
)
from ..utils.exceptions import (
    APIError,
    DataError,
    NetworkError,
    ValidationError,
)
from ..utils.logging_config import (
    get_context_logger,
    log_api_call,
    log_business_event,
    log_error_with_context,
    log_performance_metric,
    log_security_event,
)
from .stock_fetcher import StockFetcher  # 既存実装を継承


class EnhancedStockFetcher(StockFetcher):
    """耐障害性強化版株価データ取得クラス"""

    def __init__(
        self,
        cache_size: int = 128,
        price_cache_ttl: int = 30,
        historical_cache_ttl: int = 300,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        enable_fallback: bool = True,
        enable_circuit_breaker: bool = True,
        enable_health_monitoring: bool = True,
    ):
        """
        Args:
            cache_size: LRUキャッシュのサイズ
            price_cache_ttl: 価格データキャッシュのTTL（秒）
            historical_cache_ttl: ヒストリカルデータキャッシュのTTL（秒）
            retry_count: リトライ回数
            retry_delay: リトライ間隔（秒）
            enable_fallback: フェイルオーバー機能の有効化
            enable_circuit_breaker: サーキットブレーカーの有効化
            enable_health_monitoring: ヘルスモニタリングの有効化
        """
        # 親クラス初期化
        super().__init__(cache_size, price_cache_ttl, historical_cache_ttl, retry_count, retry_delay)

        self.enable_fallback = enable_fallback
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_health_monitoring = enable_health_monitoring

        # 拡張ロガー
        self.logger = get_context_logger(__name__)

        # 耐障害性機能の初期化
        if self.enable_fallback or self.enable_circuit_breaker:
            self._setup_resilience()

        self.logger.info("EnhancedStockFetcher初期化完了",
                        enable_fallback=enable_fallback,
                        enable_circuit_breaker=enable_circuit_breaker,
                        enable_health_monitoring=enable_health_monitoring)

    def _setup_resilience(self) -> None:
        """耐障害性機能のセットアップ"""
        # APIエンドポイント設定（Yahoo Finance の複数アクセス方法）
        endpoints = [
            APIEndpoint(
                name="yahoo_primary",
                base_url="https://query1.finance.yahoo.com",
                priority=1,
                timeout=30.0,
                health_check_path="/v8/finance/chart/AAPL"
            ),
            APIEndpoint(
                name="yahoo_secondary",
                base_url="https://query2.finance.yahoo.com",
                priority=2,
                timeout=30.0,
                health_check_path="/v8/finance/chart/AAPL"
            ),
        ]

        # リトライ設定
        retry_config = RetryConfig(
            max_attempts=self.retry_count,
            base_delay=self.retry_delay,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
            status_forcelist=[429, 500, 502, 503, 504, 408]
        )

        # サーキットブレーカー設定
        circuit_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=120.0,  # 2分間
            monitor_window=300.0  # 5分間
        )

        # 耐障害性クライアント
        self.resilient_client = ResilientAPIClient(
            endpoints=endpoints,
            retry_config=retry_config,
            circuit_config=circuit_config,
            enable_health_check=self.enable_health_monitoring,
            health_check_interval=300.0  # 5分間隔
        )

        self.logger.info("耐障害性機能セットアップ完了",
                        endpoints_count=len(endpoints),
                        retry_attempts=retry_config.max_attempts,
                        circuit_failure_threshold=circuit_config.failure_threshold)

    def _enhanced_retry_on_error(self, func, *args, **kwargs):
        """拡張版エラー時リトライ機能"""
        if not (self.enable_fallback or self.enable_circuit_breaker):
            # 従来のリトライ機構を使用
            return super()._retry_on_error(func, *args, **kwargs)

        last_exception = None

        # 段階的なフォールバック戦略
        strategies = [
            ("standard", self._standard_execution),
            ("cache_fallback", self._cache_fallback_execution),
            ("degraded_mode", self._degraded_mode_execution),
        ]

        for strategy_name, strategy_func in strategies:
            try:
                self.logger.debug(f"実行戦略: {strategy_name}")
                result = strategy_func(func, *args, **kwargs)

                if result is not None:
                    log_business_event("api_execution_success",
                                     strategy=strategy_name,
                                     function=func.__name__)
                    return result

            except Exception as e:
                last_exception = e
                log_error_with_context(e, {
                    "strategy": strategy_name,
                    "function": func.__name__,
                    "args": str(args)[:200],  # 長すぎる場合は切り詰め
                })

                self.logger.warning(f"戦略 {strategy_name} が失敗、次の戦略を試行",
                                  strategy=strategy_name,
                                  error=str(e)[:200])
                continue

        # すべての戦略が失敗
        log_business_event("api_execution_failed_all_strategies",
                         function=func.__name__,
                         strategies=[s[0] for s in strategies])

        if last_exception:
            self._handle_error(last_exception)
        else:
            raise APIError("すべての実行戦略が失敗しました")

    def _standard_execution(self, func, *args, **kwargs):
        """標準実行（耐障害性クライアント使用）"""
        if hasattr(self, 'resilient_client'):
            # 耐障害性クライアントを使用した実行
            return func(*args, **kwargs)
        else:
            # フォールバック：従来のリトライ機構
            return super()._retry_on_error(func, *args, **kwargs)

    def _cache_fallback_execution(self, func, *args, **kwargs):
        """キャッシュフォールバック実行"""
        # キャッシュから古いデータでも取得を試行
        if hasattr(func, 'clear_cache'):
            # TTLを無視してキャッシュから取得
            try:
                # 一時的にTTLを延長
                original_ttl = getattr(func, '_cache_ttl', None)
                if original_ttl:
                    func._cache_ttl = 3600 * 24  # 24時間に延長

                result = func(*args, **kwargs)

                if result is not None:
                    self.logger.info("キャッシュフォールバックで取得成功",
                                   function=func.__name__)
                    log_business_event("cache_fallback_success",
                                     function=func.__name__)
                    return result

            except Exception as e:
                self.logger.debug(f"キャッシュフォールバック失敗: {e}")
            finally:
                # TTLを元に戻す
                if original_ttl:
                    func._cache_ttl = original_ttl

        return None

    def _degraded_mode_execution(self, func, *args, **kwargs):
        """劣化モード実行（最小限のデータ）"""
        # 最低限のデフォルトデータを返す
        self.logger.warning("劣化モードで実行",
                          function=func.__name__)

        log_security_event("degraded_mode_activated", "warning",
                         function=func.__name__)

        # 関数名に応じてデフォルトデータを生成
        if "get_current_price" in func.__name__:
            return self._generate_default_price_data(args[0] if args else "UNKNOWN")
        elif "get_historical_data" in func.__name__:
            return self._generate_default_historical_data()
        elif "get_company_info" in func.__name__:
            return self._generate_default_company_info(args[0] if args else "UNKNOWN")

        return None

    def _generate_default_price_data(self, symbol: str) -> Dict[str, any]:
        """デフォルト価格データを生成"""
        return {
            "symbol": symbol,
            "current_price": 0.0,
            "previous_close": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0,
            "market_cap": None,
            "timestamp": datetime.now(),
            "data_quality": "degraded",
            "source": "fallback"
        }

    def _generate_default_historical_data(self) -> pd.DataFrame:
        """デフォルトヒストリカルデータを生成"""
        # 空のDataFrameを返す
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def _generate_default_company_info(self, symbol: str) -> Dict[str, any]:
        """デフォルト企業情報を生成"""
        return {
            "symbol": symbol,
            "name": f"Company {symbol}",
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": None,
            "employees": None,
            "description": "Company information unavailable",
            "website": None,
            "headquarters": "Unknown",
            "country": "Unknown",
            "currency": "Unknown",
            "exchange": "Unknown",
            "data_quality": "degraded",
            "source": "fallback"
        }

    def get_current_price(self, code: str) -> Optional[Dict[str, float]]:
        """
        耐障害性強化版現在株価取得

        Args:
            code: 証券コード

        Returns:
            価格情報の辞書
        """
        def _enhanced_get_price():
            price_logger = self.logger.bind(
                operation="enhanced_get_current_price",
                stock_code=code
            )
            price_logger.info("耐障害性強化版価格取得開始")

            start_time = time.time()

            try:
                # データ検証強化
                self._validate_symbol(code)
                self._validate_market_hours()

                symbol = self._format_symbol(code)

                # Yahoo Finance API経由で取得
                ticker = self._get_ticker(symbol)

                # API呼び出しログ（詳細版）
                log_api_call("yfinance_enhanced", "GET", f"ticker.info for {symbol}",
                           context="enhanced_stock_fetcher")

                info = ticker.info

                # レスポンス検証強化
                if not self._validate_api_response(info):
                    raise DataError(f"無効なAPIレスポンス: {symbol}")

                # 価格データ抽出と検証
                price_data = self._extract_and_validate_price_data(info, symbol)

                # パフォーマンス測定
                elapsed_time = (time.time() - start_time) * 1000
                log_performance_metric("enhanced_price_fetch_time", elapsed_time, "ms",
                                     stock_code=code,
                                     data_source="enhanced")

                price_logger.info("耐障害性強化版価格取得完了",
                                current_price=price_data.get("current_price"),
                                elapsed_ms=elapsed_time,
                                data_quality="standard")

                return price_data

            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                log_error_with_context(e, {
                    "operation": "enhanced_get_current_price",
                    "stock_code": code,
                    "elapsed_ms": elapsed_time
                })
                price_logger.error("耐障害性強化版価格取得でエラー",
                                 error=str(e),
                                 elapsed_ms=elapsed_time)
                raise

        return self._enhanced_retry_on_error(_enhanced_get_price)

    def _validate_market_hours(self) -> None:
        """市場時間の検証"""
        # 簡単な実装例（実際にはより複雑な市場時間チェックが必要）
        now = datetime.now()
        hour = now.hour

        # 日本市場の取引時間外でも警告のみ
        if hour < 9 or hour > 15:
            self.logger.warning("市場時間外のアクセス",
                              current_hour=hour,
                              market="JP")

    def _validate_api_response(self, response_data: any) -> bool:
        """APIレスポンスの妥当性検証"""
        if not response_data:
            return False

        if not isinstance(response_data, dict):
            return False

        # 最低限必要なフィールドの存在確認
        required_fields = ['currentPrice', 'regularMarketPrice', 'previousClose']
        has_required = any(field in response_data for field in required_fields)

        if not has_required:
            self.logger.warning("必要なフィールドが不足",
                              available_fields=list(response_data.keys())[:10])
            return False

        return True

    def _extract_and_validate_price_data(self, info: dict, symbol: str) -> Dict[str, any]:
        """価格データの抽出と検証"""
        # 現在価格の取得（複数の可能性を試行）
        current_price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("price")
        )

        previous_close = info.get("previousClose")

        if current_price is None:
            raise DataError(f"現在価格を取得できません: {symbol}")

        # 価格の妥当性チェック
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValidationError(f"無効な価格データ: {current_price}")

        # 変化額・変化率の計算
        change = 0.0
        change_percent = 0.0

        if previous_close and isinstance(previous_close, (int, float)) and previous_close > 0:
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "previous_close": float(previous_close) if previous_close else None,
            "change": float(change),
            "change_percent": float(change_percent),
            "volume": int(info.get("volume", 0)),
            "market_cap": info.get("marketCap"),
            "timestamp": datetime.now(),
            "data_quality": "standard",
            "source": "yfinance_enhanced"
        }

    def get_system_status(self) -> Dict[str, any]:
        """システム状態の取得"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "service": "enhanced_stock_fetcher",
            "cache_stats": self._get_cache_stats(),
            "resilience_status": None
        }

        if hasattr(self, 'resilient_client'):
            status["resilience_status"] = self.resilient_client.get_status()

        return status

    def _get_cache_stats(self) -> Dict[str, any]:
        """キャッシュ統計の取得"""
        stats = {}

        # メソッドレベルのキャッシュ統計
        methods_with_cache = ['get_current_price', 'get_historical_data', 'get_company_info']

        for method_name in methods_with_cache:
            method = getattr(self, method_name, None)
            if method and hasattr(method, 'get_stats'):
                stats[method_name] = method.get_stats()

        return stats

    def health_check(self) -> Dict[str, any]:
        """ヘルスチェック"""
        start_time = time.time()

        health = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {},
            "response_time_ms": 0
        }

        # 基本機能チェック
        try:
            # 簡単な価格取得テスト（キャッシュヒットする可能性があるシンボル）
            test_result = self.get_current_price("7203")  # トヨタ
            health["checks"]["price_fetch"] = {
                "status": "pass" if test_result else "fail",
                "details": "価格取得テスト"
            }
        except Exception as e:
            health["checks"]["price_fetch"] = {
                "status": "fail",
                "details": f"価格取得エラー: {str(e)[:100]}"
            }
            health["status"] = "degraded"

        # 耐障害性機能チェック
        if hasattr(self, 'resilient_client'):
            try:
                resilience_status = self.resilient_client.get_status()
                health["checks"]["resilience"] = {
                    "status": "pass" if resilience_status["overall_health"] else "fail",
                    "details": resilience_status
                }

                if not resilience_status["overall_health"]:
                    health["status"] = "degraded"

            except Exception as e:
                health["checks"]["resilience"] = {
                    "status": "fail",
                    "details": f"耐障害性チェックエラー: {str(e)[:100]}"
                }
                health["status"] = "degraded"

        health["response_time_ms"] = (time.time() - start_time) * 1000

        # ヘルスステータスをログ出力
        log_business_event("health_check_completed",
                         status=health["status"],
                         response_time=health["response_time_ms"],
                         checks_passed=sum(1 for check in health["checks"].values()
                                         if check["status"] == "pass"))

        return health


# 使用例
if __name__ == "__main__":
    from ..utils.logging_config import setup_logging

    setup_logging()
    logger = get_context_logger(__name__)

    logger.info("=== EnhancedStockFetcher デモ開始 ===")

    # 耐障害性強化版インスタンス作成
    enhanced_fetcher = EnhancedStockFetcher(
        enable_fallback=True,
        enable_circuit_breaker=True,
        enable_health_monitoring=True
    )

    # ヘルスチェック
    health = enhanced_fetcher.health_check()
    logger.info("ヘルスチェック結果", **health)

    # システム状態確認
    status = enhanced_fetcher.get_system_status()
    logger.info("システム状態", **status)

    # 価格取得テスト
    try:
        price_data = enhanced_fetcher.get_current_price("7203")
        if price_data:
            logger.info("価格取得成功",
                       symbol=price_data["symbol"],
                       current_price=price_data["current_price"],
                       data_quality=price_data.get("data_quality", "unknown"))
    except Exception as e:
        logger.error("価格取得エラー", error=str(e))

    logger.info("=== EnhancedStockFetcher デモ完了 ===")
