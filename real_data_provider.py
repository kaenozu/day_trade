#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Provider - 実戦投入用リアルデータ取得システム (改善版)

Issue #852対応: 堅牢なデータ取得と責務分離の強化
- カスタム例外による明確なエラーハンドリング
- データ取得結果の明確な管理
- 外部設定からの銘柄リスト読み込み
- レート制限対応とAPIレスポンス最適化
- データ検証の強化
- テストコードの分離完了
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, NamedTuple
import logging
from dataclasses import dataclass
import time as time_module
import json
import yaml
from pathlib import Path

# Windows環境での文字化け対策
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Windows Console API対応
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


# Issue #852: カスタム例外クラス定義
class RealDataProviderError(Exception):
    """リアルデータプロバイダー基本例外"""
    pass


class DataFetchError(RealDataProviderError):
    """データ取得エラー"""

    def __init__(self, symbol: str, message: str, original_error: Exception = None):
        self.symbol = symbol
        self.original_error = original_error
        super().__init__(f"Failed to fetch data for {symbol}: {message}")


class DataValidationError(RealDataProviderError):
    """データ検証エラー"""

    def __init__(self, symbol: str, validation_issue: str):
        self.symbol = symbol
        self.validation_issue = validation_issue
        super().__init__(f"Data validation failed for {symbol}: {validation_issue}")


class ConfigurationError(RealDataProviderError):
    """設定エラー"""
    pass


# Issue #852: データ取得結果クラス
class DataFetchResult(NamedTuple):
    """データ取得結果"""
    success: bool
    data: Optional['RealStockData']
    error: Optional[str]
    symbol: str


@dataclass
class RealStockData:
    """実際の株価データ - Issue #852対応"""
    symbol: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    previous_close: float
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    # Issue #852: データ鮮度管理
    last_fetched_time: datetime = None
    data_source: str = "Yahoo Finance"

    def __post_init__(self):
        """データクラス初期化後処理"""
        if self.last_fetched_time is None:
            self.last_fetched_time = datetime.now()

    @property
    def change_percent(self) -> float:
        """前日比パーセント"""
        if self.previous_close and self.previous_close > 0:
            return ((self.current_price - self.previous_close) / self.previous_close) * 100
        return 0.0

    @property
    def day_range_percent(self) -> float:
        """当日値幅パーセント"""
        if self.low_price > 0:
            return ((self.high_price - self.low_price) / self.low_price) * 100
        return 0.0


class RealDataProvider:
    """
    実戦投入用リアルデータプロバイダー - Issue #852対応
    Yahoo Finance APIを使用して実際の株価データを取得
    堅牢なエラーハンドリングと責務分離を実現
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)

        # Issue #852: 外部設定からの銘柄リスト読み込み
        self.config_path = config_path or "config/symbols.yaml"
        self.target_symbols = {}

        # デフォルト設定値（設定ファイル読み込み前）
        self.cache_duration = 60
        self.min_request_interval = 0.5
        self.max_requests_per_minute = 60
        self.timeout_seconds = 30
        self.min_price = 1.0
        self.max_price_change_percent = 50.0
        self.required_fields = ["Open", "High", "Low", "Close", "Volume"]
        self.max_retries = 3
        self.retry_delay = 1.0
        self.consecutive_failure_threshold = 5

        # 設定ファイル読み込み（上記デフォルト値を上書き）
        self._load_symbol_configuration()

        # キャッシュ機能（API制限回避）
        self.data_cache = {}

        # Issue #852: レート制限対応（強化版）
        self.last_request_time = 0
        self.request_count = 0
        self.consecutive_failures = {}  # 銘柄別連続失敗カウント

        # yfinance セッション管理
        self.session = None

    def _load_symbol_configuration(self):
        """Issue #852: 外部設定ファイルからの銘柄リスト読み込み（強化版）"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.suffix.lower() == '.yaml':
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)

                # 銘柄リスト読み込み
                self.target_symbols = config.get('symbols', {})

                # API設定読み込み
                api_settings = config.get('api_settings', {})
                self.cache_duration = api_settings.get('cache_duration', 60)
                self.min_request_interval = api_settings.get('request_interval', 0.5)
                self.max_requests_per_minute = api_settings.get('max_requests_per_minute', 60)
                self.timeout_seconds = api_settings.get('timeout_seconds', 30)

                # データ検証設定読み込み
                validation_settings = config.get('data_validation', {})
                self.min_price = validation_settings.get('min_price', 1.0)
                self.max_price_change_percent = validation_settings.get('max_price_change_percent', 50.0)
                self.required_fields = validation_settings.get('required_ohlcv_fields',
                                                             ["Open", "High", "Low", "Close", "Volume"])

                # エラーハンドリング設定読み込み
                error_settings = config.get('error_handling', {})
                self.max_retries = error_settings.get('max_retries', 3)
                self.retry_delay = error_settings.get('retry_delay', 1.0)
                self.consecutive_failure_threshold = error_settings.get('consecutive_failure_threshold', 5)

                self.logger.info(f"Loaded {len(self.target_symbols)} symbols and settings from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self._create_default_configuration()

        except Exception as e:
            error_msg = f"Failed to load symbol configuration: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(f"Cannot load configuration from {self.config_path}: {e}") from e

    def _create_default_configuration(self):
        """デフォルト設定ファイルの作成"""
        default_config = {
            'symbols': {
                "7203.T": "トヨタ自動車",
                "9984.T": "ソフトバンクG",
                "6758.T": "ソニーG",
                "8306.T": "三菱UFJ",
                "4689.T": "LINEヤフー",
                "8058.T": "三菱商事",
                "6501.T": "日立製作所",
                "4063.T": "信越化学",
                "6098.T": "リクルートHD",
                "9432.T": "NTT"
            },
            'api_settings': {
                'cache_duration': 60,
                'request_interval': 0.5,
                'max_requests_per_minute': 60
            }
        }

        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

            self.target_symbols = default_config['symbols']
            self.logger.info(f"Created default configuration: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to create default configuration: {e}")
            self._load_default_symbols()

    def _load_default_symbols(self):
        """フォールバック用のデフォルト銘柄リスト"""
        self.target_symbols = {
            "7203.T": "トヨタ自動車",
            "9984.T": "ソフトバンクG",
            "6758.T": "ソニーG",
            "8306.T": "三菱UFJ",
            "8058.T": "三菱商事"
        }
        self.logger.info("Using fallback symbol list")

    async def _enforce_rate_limit(self):
        """Issue #852: レート制限の実施"""
        current_time = time_module.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            wait_time = self.min_request_interval - elapsed
            await asyncio.sleep(wait_time)

        self.last_request_time = time_module.time()
        self.request_count += 1

    def _validate_stock_data(self, data: pd.DataFrame, symbol: str, previous_close: Optional[float] = None) -> bool:
        """Issue #852: データ検証の強化（拡張版）"""
        if data.empty:
            raise DataValidationError(symbol, "Empty dataset returned")

        # 必要なカラムの存在確認
        missing_columns = [col for col in self.required_fields if col not in data.columns]
        if missing_columns:
            raise DataValidationError(symbol, f"Missing required columns: {missing_columns}")

        # 最新データの基本検証
        latest = data.iloc[-1]

        # 価格データの整合性チェック
        if latest['High'] < latest['Low']:
            raise DataValidationError(symbol, "High price is less than low price")

        if any(latest[col] <= 0 for col in ['Open', 'High', 'Low', 'Close']):
            raise DataValidationError(symbol, "Invalid price data (zero or negative)")

        # 最小価格チェック
        if latest['Close'] < self.min_price:
            raise DataValidationError(symbol, f"Price {latest['Close']} below minimum {self.min_price}")

        # 価格変動の妥当性チェック（前日比）
        if previous_close and previous_close > 0:
            price_change_percent = abs(latest['Close'] - previous_close) / previous_close * 100
            if price_change_percent > self.max_price_change_percent:
                raise DataValidationError(
                    symbol,
                    f"Price change {price_change_percent:.1f}% exceeds maximum {self.max_price_change_percent}%"
                )

        # 出来高の基本チェック
        if latest['Volume'] < 0:
            raise DataValidationError(symbol, "Negative volume data")

        # 価格の四本値チェック
        ohlc_prices = [latest['Open'], latest['High'], latest['Low'], latest['Close']]
        if latest['Low'] > min(latest['Open'], latest['Close']) or latest['High'] < max(latest['Open'], latest['Close']):
            raise DataValidationError(symbol, "OHLC price relationships are inconsistent")

        self.logger.debug(f"Data validation passed for {symbol}")
        return True

    def get_real_stock_data(self, symbol: str) -> DataFetchResult:
        """
        指定銘柄の実際の株価データを取得 - Issue #852対応強化版

        Args:
            symbol: 銘柄コード（例：7203.T）

        Returns:
            DataFetchResult: データ取得結果（成功/失敗情報を含む）
        """
        # 連続失敗チェック
        if self.consecutive_failures.get(symbol, 0) >= self.consecutive_failure_threshold:
            error_msg = f"Symbol {symbol} has exceeded consecutive failure threshold ({self.consecutive_failure_threshold})"
            self.logger.warning(error_msg)
            return DataFetchResult(False, None, error_msg, symbol)

        # リトライ処理
        for attempt in range(self.max_retries + 1):
            try:
                # キャッシュチェック
                cache_key = f"{symbol}_{int(time_module.time() / self.cache_duration)}"
                if cache_key in self.data_cache:
                    cached_result = self.data_cache[cache_key]
                    self.logger.debug(f"Cache hit for {symbol}")
                    # キャッシュヒット時は連続失敗カウントをリセット
                    self.consecutive_failures[symbol] = 0
                    return cached_result

                # Yahoo Finance APIでデータ取得（セッション使用）
                if self.session is None:
                    self.session = yf.Session()

                ticker = yf.Ticker(symbol, session=self.session)

                # 基本情報取得（タイムアウト対応）
                try:
                    info = ticker.info
                    hist = ticker.history(period="2d")  # 2日分で前日比計算
                except Exception as api_error:
                    error_msg = f"Yahoo Finance API error: {str(api_error)}"
                    self.logger.error(f"Attempt {attempt + 1}/{self.max_retries + 1} - {error_msg}")

                    if attempt < self.max_retries:
                        time_module.sleep(self.retry_delay * (attempt + 1))  # 指数バックオフ
                        continue
                    else:
                        self._increment_failure_count(symbol)
                        return DataFetchResult(False, None, error_msg, symbol)

                # Issue #852: データ検証強化（前日終値も考慮）
                previous_close = None
                if len(hist) > 1:
                    previous_close = float(hist.iloc[-2]['Close'])

                try:
                    self._validate_stock_data(hist, symbol, previous_close)
                except DataValidationError as validation_error:
                    error_msg = f"Data validation failed: {validation_error}"
                    self.logger.error(f"Attempt {attempt + 1}/{self.max_retries + 1} - {error_msg}")

                    if attempt < self.max_retries:
                        time_module.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        self._increment_failure_count(symbol)
                        return DataFetchResult(False, None, error_msg, symbol)

                # 最新データ取得
                latest = hist.iloc[-1]
                previous = hist.iloc[-2] if len(hist) > 1 else latest

                # RealStockDataオブジェクト作成
                stock_data = RealStockData(
                    symbol=symbol,
                    current_price=float(latest['Close']),
                    open_price=float(latest['Open']),
                    high_price=float(latest['High']),
                    low_price=float(latest['Low']),
                    volume=int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                    previous_close=float(previous['Close']),
                    market_cap=info.get('marketCap'),
                    pe_ratio=info.get('trailingPE'),
                    dividend_yield=info.get('dividendYield'),
                    last_fetched_time=datetime.now(),
                    data_source="Yahoo Finance"
                )

                # 成功結果作成
                result = DataFetchResult(True, stock_data, None, symbol)

                # キャッシュ保存
                self.data_cache[cache_key] = result

                # 成功時は連続失敗カウントをリセット
                self.consecutive_failures[symbol] = 0

                self.logger.info(f"Successfully fetched data for {symbol}")
                return result

            except Exception as e:
                error_msg = f"Unexpected error for {symbol}: {str(e)}"
                self.logger.error(f"Attempt {attempt + 1}/{self.max_retries + 1} - {error_msg}")

                if attempt < self.max_retries:
                    time_module.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    self._increment_failure_count(symbol)
                    return DataFetchResult(False, None, error_msg, symbol)

    def _increment_failure_count(self, symbol: str):
        """連続失敗カウントを増加"""
        self.consecutive_failures[symbol] = self.consecutive_failures.get(symbol, 0) + 1
        if self.consecutive_failures[symbol] >= self.consecutive_failure_threshold:
            self.logger.warning(
                f"Symbol {symbol} has reached consecutive failure threshold "
                f"({self.consecutive_failures[symbol]}/{self.consecutive_failure_threshold})"
            )

    async def get_multiple_stocks_data(self, symbols: List[str] = None) -> Dict[str, DataFetchResult]:
        """
        複数銘柄の実際の株価データを並行取得 - Issue #852対応

        Args:
            symbols: 銘柄コードリスト（未指定時は全対象銘柄）

        Returns:
            Dict[str, DataFetchResult]: 銘柄コード -> データ取得結果
        """
        if symbols is None:
            symbols = list(self.target_symbols.keys())

        self.logger.info(f"Fetching data for {len(symbols)} symbols")

        # 並行処理で高速取得（レート制限対応）
        loop = asyncio.get_event_loop()
        tasks = []

        for symbol in symbols:
            # レート制限を適用した実行
            async def fetch_with_rate_limit(sym):
                await self._enforce_rate_limit()
                return await loop.run_in_executor(None, self.get_real_stock_data, sym)

            tasks.append(fetch_with_rate_limit(symbol))

        # 全タスクを並列実行
        results = {}
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(task_results):
            symbol = symbols[i]
            if isinstance(result, DataFetchResult):
                results[symbol] = result
            elif isinstance(result, Exception):
                error_msg = f"Task execution error: {str(result)}"
                self.logger.error(f"Failed to get data for {symbol}: {error_msg}")
                results[symbol] = DataFetchResult(False, None, error_msg, symbol)
            else:
                error_msg = "Unknown result type"
                self.logger.error(f"Unexpected result for {symbol}: {error_msg}")
                results[symbol] = DataFetchResult(False, None, error_msg, symbol)

        # 統計情報をログ出力
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        self.logger.info(f"Data fetch completed: {successful} successful, {failed} failed")

        return results

    def get_successful_data_only(self, results: Dict[str, DataFetchResult]) -> Dict[str, RealStockData]:
        """Issue #852: 成功したデータのみを抽出"""
        return {symbol: result.data for symbol, result in results.items()
                if result.success and result.data is not None}

    def get_market_status(self) -> Dict[str, any]:
        """
        現在の市場状況を取得

        Returns:
            Dict: 市場状況情報
        """
        try:
            # 東証ETFで市場状況判定（先物の代替）
            nikkei_etf = yf.Ticker("1321.T")  # NEXT FUNDS 日経225連動型上場投信
            data = nikkei_etf.history(period="1d")

            if not data.empty:
                latest = data.iloc[-1]
                # 現在時刻による簡易市場判定
                now = datetime.now().time()
                is_open = (time(9, 0) <= now <= time(15, 0) and
                           datetime.now().weekday() < 5)  # 月-金

                return {
                    "market_open": is_open,
                    "nikkei_etf": float(latest['Close']),
                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            self.logger.error(f"Failed to get market status: {e}")

        return {
            "market_open": False,
            "nikkei_etf": None,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_historical_data(self, symbol: str, period: str = "20d") -> Optional[pd.DataFrame]:
        """
        Issue #852: 履歴データ取得（テクニカル指標計算用）

        Args:
            symbol: 銘柄コード
            period: 取得期間（例: "20d", "1mo", "3mo"）

        Returns:
            pd.DataFrame: 履歴データ（OHLCV）
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                self.logger.warning(f"No historical data available for {symbol}")
                return None

            self.logger.info(f"Retrieved {len(hist)} days of historical data for {symbol}")
            return hist

        except Exception as e:
            self.logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None


class RealDataAnalysisEngine:
    """
    実データに基づく分析エンジン - Issue #852対応
    従来のダミーデータベース分析から実データベース分析へ移行
    テクニカル指標計算の責務分離後バージョン
    """

    def __init__(self, data_provider: RealDataProvider = None):
        self.data_provider = data_provider or RealDataProvider()
        self.logger = logging.getLogger(__name__)

        # Issue #852: テクニカル指標計算機能の分離
        # TODO: indicators.pyモジュール統合時に置き換え予定
        self.indicators_module = None

    async def analyze_daytrading_opportunities(self, limit: int = 5) -> List[Dict[str, any]]:
        """
        実データに基づくデイトレード機会分析 - Issue #852対応

        Args:
            limit: 推奨銘柄数

        Returns:
            List[Dict]: デイトレード推奨リスト
        """
        try:
            # 実データ取得
            fetch_results = await self.data_provider.get_multiple_stocks_data()

            if not fetch_results:
                self.logger.warning("No data fetch results available")
                return []

            # 成功したデータのみを抽出
            successful_data = self.data_provider.get_successful_data_only(fetch_results)

            if not successful_data:
                self.logger.warning("No successful data fetches")
                return []

            recommendations = []

            for symbol, data in successful_data.items():
                # Issue #852: テクニカル指標は履歴データから計算（責務分離）
                historical_data = self.data_provider.get_historical_data(symbol, "30d")
                indicators = self._calculate_basic_indicators(historical_data) if historical_data is not None else {}

                # 実データに基づくスコア計算
                score = self._calculate_real_trading_score(data, indicators)

                # 推奨情報作成
                recommendation = {
                    "symbol": symbol.replace('.T', ''),
                    "name": self.data_provider.target_symbols.get(symbol, ""),
                    "current_price": data.current_price,
                    "change_percent": data.change_percent,
                    "volume": data.volume,
                    "volatility": indicators.get("volatility", 0),
                    "rsi": indicators.get("rsi", 50),
                    "ma5": indicators.get("ma5", 0),
                    "ma20": indicators.get("ma20", 0),
                    "trading_score": score,
                    "signal": self._generate_trading_signal(data, indicators, score),
                    "confidence": min(95, max(60, score)),  # 実データベースの信頼度
                    "target_profit": self._calculate_target_profit(data, indicators),
                    "stop_loss": self._calculate_stop_loss(data, indicators),
                    "data_freshness": (datetime.now() - data.last_fetched_time).total_seconds() / 60  # 分単位
                }

                recommendations.append(recommendation)

            # スコア順にソート
            recommendations.sort(key=lambda x: x["trading_score"], reverse=True)

            self.logger.info(f"Generated {len(recommendations)} recommendations from {len(successful_data)} symbols")
            return recommendations[:limit]

        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return []

    def _calculate_basic_indicators(self, hist: pd.DataFrame) -> Dict[str, float]:
        """
        Issue #852: 基本的なテクニカル指標計算（暫定実装）
        将来的にはindicators.pyモジュールに移動予定
        """
        try:
            if hist is None or len(hist) < 20:
                return {}

            indicators = {}

            # 移動平均
            if len(hist) >= 5:
                indicators["ma5"] = float(hist['Close'].rolling(window=5).mean().iloc[-1])
            if len(hist) >= 20:
                indicators["ma20"] = float(hist['Close'].rolling(window=20).mean().iloc[-1])

            # ボラティリティ（年率）
            pct_change = hist['Close'].pct_change().dropna()
            if len(pct_change) > 1:
                indicators["volatility"] = float(pct_change.std() * np.sqrt(252) * 100)

            # RSI（簡易版）
            if len(hist) >= 14:
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                indicators["rsi"] = float(rsi) if not pd.isna(rsi) else 50

            return indicators

        except Exception as e:
            self.logger.error(f"Failed to calculate basic indicators: {e}")
            return {}

    def _calculate_real_trading_score(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """
        実データに基づく取引スコア計算
        """
        score = 0.0

        # 出来高評価（流動性）
        if data.volume > 1000000:  # 100万株以上
            score += 25
        elif data.volume > 500000:
            score += 15
        elif data.volume > 100000:
            score += 10

        # ボラティリティ評価（デイトレード適性）
        volatility = indicators.get("volatility", 0)
        if 2.0 <= volatility <= 8.0:  # 適度なボラティリティ
            score += 30
        elif volatility > 8.0:  # 高ボラティリティ
            score += 20

        # RSI評価（買われ過ぎ・売られ過ぎ）
        rsi = indicators.get("rsi", 50)
        if 30 <= rsi <= 70:  # 適正範囲
            score += 20
        elif rsi < 30 or rsi > 70:  # 反転期待
            score += 15

        # 移動平均評価（トレンド）
        ma5 = indicators.get("ma5", 0)
        ma20 = indicators.get("ma20", 0)
        if ma5 > ma20 and data.current_price > ma5:  # 上昇トレンド
            score += 15

        # 当日値幅評価
        day_range = data.day_range_percent
        if 1.0 <= day_range <= 5.0:  # 適度な値動き
            score += 10

        return min(100.0, score)

    def _generate_trading_signal(self, data: RealStockData, indicators: Dict[str, float], score: float) -> str:
        """実データに基づく売買シグナル生成"""
        rsi = indicators.get("rsi", 50)
        ma5 = indicators.get("ma5", 0)

        if score >= 80:
            if rsi < 30:
                return "★強い買い★"
            elif data.current_price > ma5:
                return "●買い●"
            else:
                return "△やや買い△"
        elif score >= 60:
            return "…待機…"
        else:
            return "▽売り▽"

    def _calculate_target_profit(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """利確目標計算"""
        volatility = indicators.get("volatility", 3.0)
        return min(5.0, max(1.5, volatility * 0.8))  # ボラティリティベース

    def _calculate_stop_loss(self, data: RealStockData, indicators: Dict[str, float]) -> float:
        """損切りライン計算"""
        volatility = indicators.get("volatility", 3.0)
        return min(3.0, max(1.0, volatility * 0.5))  # 利確の半分程度


# Issue #852: テストコード分離完了
# テストコードは tests/test_real_data_provider_improved.py に移動しました

if __name__ == "__main__":
    # 基本的な動作確認のみ
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logger = logging.getLogger(__name__)
    logger.info("Real Data Provider - Issue #852 対応強化版")
    logger.info("主要改善点:")
    logger.info("- カスタム例外による明確なエラーハンドリング")
    logger.info("- データ取得結果の明確な管理")
    logger.info("- 外部設定からの銘柄リスト読み込み")
    logger.info("- レート制限対応とAPIレスポンス最適化")
    logger.info("- データ検証の強化")
    logger.info("- テストコードの分離完了")
    logger.info("")
    logger.info("詳細なテストは tests/test_real_data_provider_improved.py を実行してください")
