"""
銘柄スクリーニング機能（拡張版）
ストラテジーパターンと設定外部化によるリファクタリング版
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from functools import lru_cache

import pandas as pd

from ..data.stock_fetcher import StockFetcher
from ..utils.logging_config import get_context_logger
from ..utils.formatters import format_currency, format_percentage, format_volume
from .indicators import TechnicalIndicators
from .signals import TradingSignalGenerator
from .screening_strategies import ScreeningStrategyFactory
from .screening_config import get_screening_config, ScreeningConfig

logger = get_context_logger(__name__, component="stock_screener_enhanced")


class ScreenerCondition(Enum):
    """スクリーニング条件の種類"""

    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    GOLDEN_CROSS = "golden_cross"
    DEAD_CROSS = "dead_cross"
    BOLLINGER_SQUEEZE = "bollinger_squeeze"
    BOLLINGER_BREAKOUT = "bollinger_breakout"
    VOLUME_SPIKE = "volume_spike"
    PRICE_NEAR_SUPPORT = "price_near_support"
    PRICE_NEAR_RESISTANCE = "price_near_resistance"
    STRONG_MOMENTUM = "strong_momentum"
    REVERSAL_PATTERN = "reversal_pattern"


@dataclass
class ScreenerCriteria:
    """スクリーニング基準"""

    condition: ScreenerCondition
    threshold: Optional[float] = None
    lookback_days: int = 20
    weight: float = 1.0
    description: str = ""


@dataclass
class ScreenerResult:
    """スクリーニング結果"""

    symbol: str
    score: float
    matched_conditions: List[ScreenerCondition]
    technical_data: Dict[str, Any]
    signal_data: Optional[Dict[str, Any]] = None
    last_price: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None


class EnhancedStockScreener:
    """拡張版銘柄スクリーニングクラス"""

    def __init__(
        self,
        stock_fetcher: Optional[StockFetcher] = None,
        config: Optional[ScreeningConfig] = None
    ):
        """
        Args:
            stock_fetcher: 株価データ取得インスタンス
            config: スクリーニング設定
        """
        self.stock_fetcher = stock_fetcher or StockFetcher()
        self.signal_generator = TradingSignalGenerator()
        self.config = config or get_screening_config()
        self.strategy_factory = ScreeningStrategyFactory()

        # データキャッシュ（LRUキャッシュで最適化）
        self._data_cache = {}
        self._cache_timestamps = {}

        logger.info("拡張版スクリーナーを初期化")

    @lru_cache(maxsize=100)
    def _get_default_criteria_for_condition(self, condition: ScreenerCondition) -> ScreenerCriteria:
        """条件のデフォルト基準を取得（キャッシュ付き）"""
        threshold = self.config.get_threshold(condition.value)
        lookback_days = self.config.get_lookback_days(condition.value)

        return ScreenerCriteria(
            condition=condition,
            threshold=threshold,
            lookback_days=lookback_days,
            weight=1.0,
            description=f"{condition.value}条件"
        )

    def get_default_criteria(self) -> List[ScreenerCriteria]:
        """デフォルトスクリーニング基準を取得（設定ファイルベース）"""
        return [
            ScreenerCriteria(
                ScreenerCondition.RSI_OVERSOLD,
                threshold=self.config.get_threshold("RSI_OVERSOLD"),
                weight=1.0,
                description="RSI過売り",
            ),
            ScreenerCriteria(
                ScreenerCondition.GOLDEN_CROSS,
                weight=2.0,
                description="ゴールデンクロス発生",
            ),
            ScreenerCriteria(
                ScreenerCondition.VOLUME_SPIKE,
                threshold=self.config.get_threshold("VOLUME_SPIKE"),
                weight=1.5,
                description="出来高急増",
            ),
            ScreenerCriteria(
                ScreenerCondition.STRONG_MOMENTUM,
                threshold=self.config.get_threshold("STRONG_MOMENTUM"),
                weight=1.2,
                description="強い上昇モメンタム",
            ),
        ]

    def screen_stocks(
        self,
        symbols: List[str],
        criteria: Optional[List[ScreenerCriteria]] = None,
        min_score: float = 0.0,
        max_results: int = 50,
        period: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[ScreenerResult]:
        """
        銘柄スクリーニングを実行（最適化版）

        Args:
            symbols: 対象銘柄コードリスト
            criteria: スクリーニング基準
            min_score: 最小スコア閾値
            max_results: 最大結果数
            period: データ取得期間
            use_cache: キャッシュを使用するか

        Returns:
            スクリーニング結果リスト（スコア順）
        """
        if criteria is None:
            criteria = self.get_default_criteria()

        if period is None:
            period = self.config.get_data_period()

        logger.info(f"拡張版スクリーニング開始: {len(symbols)}銘柄, {len(criteria)}条件")

        results = []
        max_workers = self.config.get_max_workers()

        # バルクデータ取得の実装（効率化）
        if use_cache:
            self._preload_data(symbols, period)

        # 並列処理で各銘柄を評価
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._evaluate_symbol_enhanced,
                    symbol,
                    criteria,
                    period,
                    use_cache
                ): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result and result.score >= min_score:
                        results.append(result)
                        logger.debug(f"銘柄 {symbol}: スコア {result.score:.2f}")
                except Exception as e:
                    logger.error(f"銘柄 {symbol} の評価でエラー: {e}")

        # スコア順でソート
        results.sort(key=lambda x: x.score, reverse=True)

        # 結果数を制限
        if max_results > 0:
            results = results[:max_results]

        logger.info(f"拡張版スクリーニング完了: {len(results)}銘柄が条件を満たしました")
        return results

    def _preload_data(self, symbols: List[str], period: str):
        """データの事前読み込み（バルク取得の代替）"""
        logger.debug(f"データ事前読み込み開始: {len(symbols)}銘柄")

        # バルク取得可能な場合は実装
        bulk_method = getattr(self.stock_fetcher, 'get_bulk_historical_data', None)
        if bulk_method and callable(bulk_method):
            try:
                bulk_data = bulk_method(symbols, period=period, interval="1d")
                for symbol, data in bulk_data.items():
                    if data is not None and not data.empty:
                        cache_key = f"{symbol}_{period}"
                        self._data_cache[cache_key] = data
                        self._cache_timestamps[cache_key] = datetime.now()
                logger.debug("バルクデータ取得完了")
                return
            except Exception as e:
                logger.debug(f"バルクデータ取得失敗、個別取得にフォールバック: {e}")

        # 個別取得でのプリロード
        cache_expiry = timedelta(minutes=30)
        current_time = datetime.now()

        for symbol in symbols:
            cache_key = f"{symbol}_{period}"

            # キャッシュが有効かチェック
            if (cache_key in self._data_cache and
                cache_key in self._cache_timestamps and
                current_time - self._cache_timestamps[cache_key] < cache_expiry):
                continue

            try:
                data = self.stock_fetcher.get_historical_data(symbol, period=period, interval="1d")
                if data is not None and not data.empty:
                    self._data_cache[cache_key] = data
                    self._cache_timestamps[cache_key] = current_time
            except Exception as e:
                logger.debug(f"データプリロード失敗 ({symbol}): {e}")

    def _evaluate_symbol_enhanced(
        self,
        symbol: str,
        criteria: List[ScreenerCriteria],
        period: str,
        use_cache: bool = True
    ) -> Optional[ScreenerResult]:
        """
        個別銘柄の評価（拡張版）

        Args:
            symbol: 銘柄コード
            criteria: スクリーニング基準
            period: データ取得期間
            use_cache: キャッシュを使用するか

        Returns:
            スクリーニング結果
        """
        try:
            # データ取得（キャッシュ使用）
            df = self._get_data_with_cache(symbol, period, use_cache)

            if df is None or df.empty or len(df) < self.config.get_min_data_points():
                logger.debug(f"銘柄 {symbol}: データ不足")
                return None

            # テクニカル指標計算
            indicators = TechnicalIndicators.calculate_all(df)

            # 各条件を戦略パターンで評価
            matched_conditions = []
            total_score = 0.0
            total_weight = 0.0

            for criterion in criteria:
                strategy = self.strategy_factory.get_strategy(criterion.condition.value)
                if not strategy:
                    logger.warning(f"未実装の条件: {criterion.condition.value}")
                    continue

                meets_condition, condition_score = strategy.evaluate(
                    df=df,
                    indicators=indicators,
                    threshold=criterion.threshold,
                    lookback_days=criterion.lookback_days
                )

                if meets_condition:
                    matched_conditions.append(criterion.condition)
                    total_score += condition_score * criterion.weight

                total_weight += criterion.weight

            # 正規化されたスコア
            final_score = total_score / total_weight if total_weight > 0 else 0.0

            # 最低1つの条件を満たしている場合のみ結果を返す
            if not matched_conditions:
                return None

            # 基本情報を収集
            last_price = float(df["Close"].iloc[-1])
            volume = int(df["Volume"].iloc[-1])

            # テクニカルデータの要約（改良版）
            technical_data = self._summarize_technical_data_enhanced(df, indicators)

            return ScreenerResult(
                symbol=symbol,
                score=final_score,
                matched_conditions=matched_conditions,
                technical_data=technical_data,
                last_price=last_price,
                volume=volume,
            )

        except Exception as e:
            logger.error(f"銘柄 {symbol} の評価中にエラー: {e}")
            return None

    def _get_data_with_cache(self, symbol: str, period: str, use_cache: bool) -> Optional[pd.DataFrame]:
        """キャッシュを使用したデータ取得"""
        cache_key = f"{symbol}_{period}"

        if use_cache and cache_key in self._data_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < timedelta(minutes=30):
                return self._data_cache[cache_key]

        # キャッシュにない場合は取得
        try:
            df = self.stock_fetcher.get_historical_data(symbol, period=period, interval="1d")
            if df is not None and not df.empty:
                if use_cache:
                    self._data_cache[cache_key] = df
                    self._cache_timestamps[cache_key] = datetime.now()

                    # キャッシュサイズ制限
                    if len(self._data_cache) > self.config.get_cache_size():
                        self._cleanup_cache()

                return df
        except Exception as e:
            logger.debug(f"データ取得エラー ({symbol}): {e}")

        return None

    def _cleanup_cache(self):
        """古いキャッシュエントリを削除"""
        current_time = datetime.now()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > timedelta(hours=2)
        ]

        for key in expired_keys:
            self._data_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)

        logger.debug(f"キャッシュクリーンアップ: {len(expired_keys)}エントリを削除")

    def _summarize_technical_data_enhanced(
        self, df: pd.DataFrame, indicators: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        テクニカルデータの要約（拡張版）

        Args:
            df: 価格データ
            indicators: テクニカル指標

        Returns:
            要約データ
        """
        summary = {}

        try:
            # 基本統計
            current_price = float(df["Close"].iloc[-1])

            # 52週高安値の計算（実際の52週間で）
            if self.config.should_use_actual_52_weeks():
                weeks_52_ago = datetime.now() - timedelta(weeks=52)
                try:
                    # インデックスがDatetimeIndexの場合
                    if hasattr(df.index, 'to_pydatetime'):
                        df_52w = df[df.index >= weeks_52_ago]
                    else:
                        # フォールバック：利用可能なデータを使用
                        df_52w = df.iloc[-min(252, len(df)):]  # 約1年分

                    if len(df_52w) > 0:
                        high_52w = float(df_52w["High"].max())
                        low_52w = float(df_52w["Low"].min())
                    else:
                        high_52w = float(df["High"].max())
                        low_52w = float(df["Low"].min())
                except Exception:
                    # エラー時のフォールバック
                    high_52w = float(df["High"].max())
                    low_52w = float(df["Low"].min())
            else:
                high_52w = float(df["High"].max())
                low_52w = float(df["Low"].min())

            summary.update(
                {
                    "current_price": current_price,
                    "high_52w": high_52w,
                    "low_52w": low_52w,
                    "price_position": (current_price - low_52w)
                    / (high_52w - low_52w)
                    * 100
                    if high_52w != low_52w
                    else 50,
                    "volume_avg_20d": int(df["Volume"].iloc[-20:].mean()),
                    "price_change_1d": (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                    / df["Close"].iloc[-2]
                    * 100
                    if len(df) >= 2
                    else 0,
                    "price_change_5d": (
                        (df["Close"].iloc[-1] - df["Close"].iloc[-6])
                        / df["Close"].iloc[-6]
                        * 100
                        if len(df) >= 6
                        else 0
                    ),
                    "price_change_20d": (
                        (df["Close"].iloc[-1] - df["Close"].iloc[-21])
                        / df["Close"].iloc[-21]
                        * 100
                        if len(df) >= 21
                        else 0
                    ),
                }
            )

            # テクニカル指標（改良版）
            if "RSI" in indicators.columns:
                rsi = indicators["RSI"].iloc[-1]
                if pd.notna(rsi):
                    summary["rsi"] = float(rsi)

            if "MACD" in indicators.columns:
                macd = indicators["MACD"].iloc[-1]
                if pd.notna(macd):
                    summary["macd"] = float(macd)

            if "SMA_20" in indicators.columns:
                sma20 = indicators["SMA_20"].iloc[-1]
                if pd.notna(sma20):
                    summary["sma_20"] = float(sma20)
                    summary["price_vs_sma20"] = (current_price - sma20) / sma20 * 100

            if "SMA_50" in indicators.columns:
                sma50 = indicators["SMA_50"].iloc[-1]
                if pd.notna(sma50):
                    summary["sma_50"] = float(sma50)
                    summary["price_vs_sma50"] = (current_price - sma50) / sma50 * 100

            # ボラティリティ指標
            if len(df) >= 20:
                volatility = df["Close"].pct_change().iloc[-20:].std() * 100
                summary["volatility_20d"] = float(volatility)

        except Exception as e:
            logger.debug(f"テクニカルデータ要約エラー: {e}")

        return summary

    def create_custom_screener(
        self, name: str, criteria: List[ScreenerCriteria], description: str = ""
    ) -> Callable:
        """
        カスタムスクリーナーを作成

        Args:
            name: スクリーナー名
            criteria: スクリーニング基準
            description: 説明

        Returns:
            カスタムスクリーナー関数
        """

        def custom_screener(symbols: List[str], **kwargs) -> List[ScreenerResult]:
            return self.screen_stocks(symbols, criteria, **kwargs)

        custom_screener.__name__ = name
        custom_screener.__doc__ = description or f"カスタムスクリーナー: {name}"

        return custom_screener

    def get_predefined_screeners(self) -> Dict[str, Callable]:
        """
        事前定義されたスクリーナーを取得（設定ファイルベース）

        Returns:
            スクリーナー辞書
        """
        screeners = {}
        predefined = self.config.get_all_predefined_screeners()

        for name, config in predefined.items():
            criteria = []
            for criterion_config in config.get("criteria", []):
                condition = ScreenerCondition(criterion_config["condition"])
                criteria.append(
                    ScreenerCriteria(
                        condition=condition,
                        threshold=criterion_config.get("threshold"),
                        weight=criterion_config.get("weight", 1.0),
                        description=criterion_config.get("description", ""),
                    )
                )

            screeners[name] = self.create_custom_screener(
                f"{name}_screener", criteria, config.get("description", "")
            )

        return screeners

    def clear_cache(self):
        """キャッシュをクリア"""
        self._data_cache.clear()
        self._cache_timestamps.clear()
        logger.info("データキャッシュをクリア")

    def get_cache_info(self) -> Dict[str, Any]:
        """キャッシュ情報を取得"""
        return {
            "cache_size": len(self._data_cache),
            "max_cache_size": self.config.get_cache_size(),
            "oldest_entry": min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            "newest_entry": max(self._cache_timestamps.values()) if self._cache_timestamps else None,
        }


def create_enhanced_screening_report(
    results: List[ScreenerResult],
    config: Optional[ScreeningConfig] = None
) -> str:
    """
    スクリーニング結果の拡張レポート生成（フォーマッター活用）

    Args:
        results: スクリーニング結果
        config: スクリーニング設定

    Returns:
        レポート文字列
    """
    if not results:
        return "スクリーニング条件を満たす銘柄が見つかりませんでした。"

    if config is None:
        config = get_screening_config()

    use_formatters = config.should_use_formatters()
    currency_precision = config.get_currency_precision()
    percentage_precision = config.get_percentage_precision()
    volume_compact = config.should_use_compact_volume()

    report_lines = [f"=== 拡張版銘柄スクリーニング結果 ({len(results)}銘柄) ===", ""]

    for i, result in enumerate(results, 1):
        # 基本情報
        price_str = (
            format_currency(result.last_price, decimal_places=currency_precision)
            if use_formatters and result.last_price
            else f"¥{result.last_price:,.0f}" if result.last_price else "N/A"
        )

        volume_str = (
            format_volume(result.volume)
            if use_formatters and volume_compact and result.volume
            else f"{result.volume:,}" if result.volume else "N/A"
        )

        report_lines.extend(
            [
                f"{i}. 銘柄コード: {result.symbol}",
                f"   スコア: {result.score:.2f}",
                f"   現在価格: {price_str}",
                f"   出来高: {volume_str}",
                f"   マッチした条件: {', '.join([c.value for c in result.matched_conditions])}",
            ]
        )

        # テクニカルデータの詳細表示
        if result.technical_data:
            if "rsi" in result.technical_data:
                report_lines.append(f"   RSI: {result.technical_data['rsi']:.1f}")

            if "price_change_1d" in result.technical_data:
                change = result.technical_data["price_change_1d"]
                change_str = (
                    format_percentage(change, decimal_places=percentage_precision)
                    if use_formatters
                    else f"{change:+.2f}%"
                )
                report_lines.append(f"   1日変化率: {change_str}")

            if "price_position" in result.technical_data:
                position = result.technical_data["price_position"]
                report_lines.append(f"   52週レンジ位置: {position:.1f}%")

            if "volatility_20d" in result.technical_data:
                vol = result.technical_data["volatility_20d"]
                vol_str = (
                    format_percentage(vol, decimal_places=1, show_sign=False)
                    if use_formatters
                    else f"{vol:.1f}%"
                )
                report_lines.append(f"   20日ボラティリティ: {vol_str}")

        report_lines.append("")

    return "\n".join(report_lines)
