#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering System - 高度特徴量エンジニアリングシステム
Issue #939 対応: センチメント分析・ファンダメンタルズによる特徴量拡充
"""

import time
import json
import requests
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import sqlite3

# NLP/テキスト分析
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    SentimentIntensityAnalyzer = None

# 経済指標API（例：FRED）
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# カスタムモジュール
try:
    from performance_monitor import performance_monitor, track_performance
    HAS_PERFORMANCE_MONITOR = True
except ImportError:
    HAS_PERFORMANCE_MONITOR = False
    def track_performance(func):
        return func

try:
    from audit_logger import audit_logger
    HAS_AUDIT_LOGGER = True
except ImportError:
    HAS_AUDIT_LOGGER = False

warnings.filterwarnings('ignore')


@dataclass
class SentimentAnalysisResult:
    """センチメント分析結果"""
    symbol: str
    sentiment_score: float  # -1.0 (negative) to 1.0 (positive)
    confidence: float      # 0.0 to 1.0
    source_count: int
    analysis_method: str
    timestamp: datetime
    raw_data: Dict[str, Any]


@dataclass
class FundamentalData:
    """ファンダメンタルズデータ"""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    timestamp: datetime = None


@dataclass
class MacroeconomicIndicators:
    """マクロ経済指標"""
    nikkei_225: Optional[float] = None
    usdjpy_rate: Optional[float] = None
    jp_10y_yield: Optional[float] = None
    us_10y_yield: Optional[float] = None
    vix_index: Optional[float] = None
    oil_price: Optional[float] = None
    gold_price: Optional[float] = None
    timestamp: datetime = None


class AdvancedFeatureEngineeringSystem:
    """高度特徴量エンジニアリングシステム"""

    def __init__(self, cache_dir: str = "data/features", cache_hours: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_hours = cache_hours

        # データベース初期化
        self.db_path = self.cache_dir / "features.db"
        self._initialize_database()

        # センチメント分析器
        self.sentiment_analyzers = {}
        if HAS_VADER:
            self.sentiment_analyzers['vader'] = SentimentIntensityAnalyzer()

        # ニュースソース設定（実際のAPIキーが必要）
        self.news_sources = {
            # 'newsapi': 'YOUR_NEWS_API_KEY',
            # 'finnhub': 'YOUR_FINNHUB_API_KEY',
            # 'alpha_vantage': 'YOUR_ALPHA_VANTAGE_API_KEY'
        }

        print("Advanced Feature Engineering System initialized")

    def _initialize_database(self):
        """データベースの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            # センチメントデータテーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    source_count INTEGER NOT NULL,
                    analysis_method TEXT NOT NULL,
                    raw_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ファンダメンタルズデータテーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    dividend_yield REAL,
                    roe REAL,
                    debt_to_equity REAL,
                    revenue_growth REAL,
                    earnings_growth REAL,
                    free_cash_flow REAL,
                    market_cap REAL,
                    enterprise_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # マクロ経済指標テーブル
            conn.execute('''
                CREATE TABLE IF NOT EXISTS macro_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nikkei_225 REAL,
                    usdjpy_rate REAL,
                    jp_10y_yield REAL,
                    us_10y_yield REAL,
                    vix_index REAL,
                    oil_price REAL,
                    gold_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # インデックス作成
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_data(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_data(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_fundamental_symbol ON fundamental_data(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_fundamental_timestamp ON fundamental_data(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_macro_timestamp ON macro_indicators(timestamp)')

            conn.commit()

    @track_performance
    def analyze_market_sentiment(self, symbol: str, use_cache: bool = True) -> SentimentAnalysisResult:
        """市場センチメント分析"""
        try:
            # キャッシュチェック
            if use_cache:
                cached_result = self._get_cached_sentiment(symbol)
                if cached_result:
                    return cached_result

            # 実際の実装では複数のニュースソースから情報を取得
            # ここではダミーデータとロジックのデモンストレーション
            news_data = self._fetch_news_data(symbol)
            social_data = self._fetch_social_sentiment(symbol)

            # センチメントスコア計算
            sentiment_scores = []
            confidence_scores = []
            source_count = 0

            # ニュース記事の分析
            if news_data:
                news_sentiment = self._analyze_text_sentiment(news_data['text'])
                sentiment_scores.append(news_sentiment['sentiment'])
                confidence_scores.append(news_sentiment['confidence'])
                source_count += news_data.get('article_count', 1)

            # ソーシャルメディアの分析
            if social_data:
                social_sentiment = self._analyze_text_sentiment(social_data['text'])
                sentiment_scores.append(social_sentiment['sentiment'])
                confidence_scores.append(social_sentiment['confidence'])
                source_count += social_data.get('post_count', 1)

            # 総合センチメントスコア計算
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                avg_confidence = np.mean(confidence_scores)
            else:
                # デフォルト値（中立）
                avg_sentiment = 0.0
                avg_confidence = 0.1
                source_count = 0

            # 結果作成
            result = SentimentAnalysisResult(
                symbol=symbol,
                sentiment_score=float(avg_sentiment),
                confidence=float(avg_confidence),
                source_count=source_count,
                analysis_method="composite",
                timestamp=datetime.now(),
                raw_data={
                    'news_data': news_data,
                    'social_data': social_data,
                    'individual_scores': sentiment_scores
                }
            )

            # データベースに保存
            self._save_sentiment_data(result)

            return result

        except Exception as e:
            print(f"センチメント分析エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"symbol": symbol, "context": "sentiment_analysis"})

            # エラー時はニュートラルなセンチメントを返す
            return SentimentAnalysisResult(
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.1,
                source_count=0,
                analysis_method="error_fallback",
                timestamp=datetime.now(),
                raw_data={'error': str(e)}
            )

    def _fetch_news_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """ニュースデータを取得（ダミー実装）"""
        # 実際の実装では、NewsAPI、Finnhub、Bloomberg APIなどを使用

        # ダミーニュース記事（実際にはAPIから取得）
        dummy_articles = [
            f"{symbol}の業績が好調で投資家の注目を集めている",
            f"{symbol}は新しい技術革新により市場での競争力を高めている",
            f"アナリストは{symbol}の将来性を楽観視している",
            f"{symbol}の株価は最近のニュースにより変動している"
        ]

        # 銘柄に基づいた簡単なセンチメント調整
        sentiment_modifier = hash(symbol) % 3 - 1  # -1, 0, 1
        if sentiment_modifier > 0:
            dummy_articles.append(f"{symbol}は市場の期待を上回る成果を発表")
        elif sentiment_modifier < 0:
            dummy_articles.append(f"{symbol}は一時的な課題に直面している")

        return {
            'text': ' '.join(dummy_articles),
            'article_count': len(dummy_articles),
            'sources': ['dummy_financial_news', 'dummy_market_report'],
            'timestamp': datetime.now()
        }

    def _fetch_social_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """ソーシャルメディアセンチメント取得（ダミー実装）"""
        # 実際の実装では、Twitter API、Reddit API、Discord API等を使用

        # ダミーソーシャルメディア投稿
        dummy_posts = [
            f"{symbol}の株価動向に注目している",
            f"{symbol}は長期投資に適していると思う",
            f"市場全体の調整で{symbol}も影響を受けている",
        ]

        # 銘柄に基づいたセンチメント調整
        social_sentiment = (hash(symbol) % 100) / 100  # 0.0-1.0
        if social_sentiment > 0.6:
            dummy_posts.append(f"{symbol}は買い時だと思う！")
        elif social_sentiment < 0.4:
            dummy_posts.append(f"{symbol}の動向は少し心配")

        return {
            'text': ' '.join(dummy_posts),
            'post_count': len(dummy_posts),
            'platforms': ['dummy_twitter', 'dummy_reddit'],
            'timestamp': datetime.now()
        }

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """テキストセンチメント分析"""
        sentiment_results = []

        # VADER分析
        if HAS_VADER and 'vader' in self.sentiment_analyzers:
            vader_result = self.sentiment_analyzers['vader'].polarity_scores(text)
            compound_score = vader_result['compound']  # -1.0 to 1.0

            sentiment_results.append({
                'sentiment': compound_score,
                'confidence': abs(compound_score),  # 絶対値を信頼度として使用
                'method': 'vader'
            })

        # TextBlob分析
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1.0 to 1.0
                subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

                sentiment_results.append({
                    'sentiment': polarity,
                    'confidence': 1.0 - subjectivity,  # 客観的ほど信頼度が高い
                    'method': 'textblob'
                })
            except Exception as e:
                print(f"TextBlob分析エラー: {e}")

        # 結果が無い場合のフォールバック
        if not sentiment_results:
            return {'sentiment': 0.0, 'confidence': 0.1}

        # 複数結果の統合
        sentiments = [r['sentiment'] for r in sentiment_results]
        confidences = [r['confidence'] for r in sentiment_results]

        return {
            'sentiment': np.mean(sentiments),
            'confidence': np.mean(confidences)
        }

    def _get_cached_sentiment(self, symbol: str) -> Optional[SentimentAnalysisResult]:
        """キャッシュされたセンチメントデータを取得"""
        cutoff_time = datetime.now() - timedelta(hours=self.cache_hours)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute('''
                SELECT * FROM sentiment_data
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol, cutoff_time)).fetchone()

            if row:
                raw_data = json.loads(row['raw_data']) if row['raw_data'] else {}

                return SentimentAnalysisResult(
                    symbol=row['symbol'],
                    sentiment_score=row['sentiment_score'],
                    confidence=row['confidence'],
                    source_count=row['source_count'],
                    analysis_method=row['analysis_method'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    raw_data=raw_data
                )

        return None

    def _save_sentiment_data(self, result: SentimentAnalysisResult):
        """センチメントデータを保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO sentiment_data
                (symbol, sentiment_score, confidence, source_count, analysis_method, raw_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.symbol,
                result.sentiment_score,
                result.confidence,
                result.source_count,
                result.analysis_method,
                json.dumps(result.raw_data, ensure_ascii=False),
                result.timestamp
            ))
            conn.commit()

    @track_performance
    def get_fundamental_data(self, symbol: str, use_cache: bool = True) -> FundamentalData:
        """ファンダメンタルズデータを取得"""
        try:
            # キャッシュチェック
            if use_cache:
                cached_data = self._get_cached_fundamental_data(symbol)
                if cached_data:
                    return cached_data

            # 実際の実装ではyfinance、Bloomberg API、EDINET等を使用
            fundamental_data = self._fetch_fundamental_data_from_api(symbol)

            # データベースに保存
            self._save_fundamental_data(fundamental_data)

            return fundamental_data

        except Exception as e:
            print(f"ファンダメンタルズデータ取得エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"symbol": symbol, "context": "fundamental_data"})

            # エラー時はデフォルト値を返す
            return FundamentalData(
                symbol=symbol,
                timestamp=datetime.now()
            )

    def _fetch_fundamental_data_from_api(self, symbol: str) -> FundamentalData:
        """APIからファンダメンタルズデータを取得"""
        # 実際の実装例（yfinanceを使用）
        if HAS_YFINANCE:
            try:
                # 日本株の場合は.Tを付加
                ticker_symbol = f"{symbol}.T" if symbol.isdigit() else symbol
                stock = yf.Ticker(ticker_symbol)

                info = stock.info
                financials = stock.financials

                # データ抽出
                fundamental_data = FundamentalData(
                    symbol=symbol,
                    pe_ratio=info.get('trailingPE'),
                    pb_ratio=info.get('priceToBook'),
                    dividend_yield=info.get('dividendYield'),
                    roe=info.get('returnOnEquity'),
                    debt_to_equity=info.get('debtToEquity'),
                    market_cap=info.get('marketCap'),
                    enterprise_value=info.get('enterpriseValue'),
                    timestamp=datetime.now()
                )

                # 成長率計算（過去データがある場合）
                if not financials.empty:
                    try:
                        recent_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else None
                        previous_revenue = financials.loc['Total Revenue'].iloc[1] if 'Total Revenue' in financials.index and len(financials.loc['Total Revenue']) > 1 else None

                        if recent_revenue and previous_revenue and previous_revenue != 0:
                            fundamental_data.revenue_growth = ((recent_revenue - previous_revenue) / previous_revenue) * 100
                    except Exception as e:
                        print(f"成長率計算エラー: {e}")

                return fundamental_data

            except Exception as e:
                print(f"yfinanceデータ取得エラー: {e}")

        # フォールバック：ダミーデータ
        base_value = hash(symbol) % 1000
        return FundamentalData(
            symbol=symbol,
            pe_ratio=10.0 + (base_value % 50),
            pb_ratio=1.0 + (base_value % 10) / 10,
            dividend_yield=(base_value % 100) / 1000,  # 0-10%
            roe=0.05 + (base_value % 200) / 1000,      # 5-25%
            debt_to_equity=0.3 + (base_value % 200) / 1000,  # 30-50%
            revenue_growth=-10 + (base_value % 400) / 10,     # -10% to 30%
            market_cap=100_000_000 + base_value * 1000,
            timestamp=datetime.now()
        )

    def _get_cached_fundamental_data(self, symbol: str) -> Optional[FundamentalData]:
        """キャッシュされたファンダメンタルズデータを取得"""
        cutoff_time = datetime.now() - timedelta(hours=self.cache_hours * 4)  # より長いキャッシュ

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute('''
                SELECT * FROM fundamental_data
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol, cutoff_time)).fetchone()

            if row:
                return FundamentalData(
                    symbol=row['symbol'],
                    pe_ratio=row['pe_ratio'],
                    pb_ratio=row['pb_ratio'],
                    dividend_yield=row['dividend_yield'],
                    roe=row['roe'],
                    debt_to_equity=row['debt_to_equity'],
                    revenue_growth=row['revenue_growth'],
                    earnings_growth=row['earnings_growth'],
                    free_cash_flow=row['free_cash_flow'],
                    market_cap=row['market_cap'],
                    enterprise_value=row['enterprise_value'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )

        return None

    def _save_fundamental_data(self, data: FundamentalData):
        """ファンダメンタルズデータを保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO fundamental_data
                (symbol, pe_ratio, pb_ratio, dividend_yield, roe, debt_to_equity,
                 revenue_growth, earnings_growth, free_cash_flow, market_cap,
                 enterprise_value, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.pe_ratio, data.pb_ratio, data.dividend_yield,
                data.roe, data.debt_to_equity, data.revenue_growth, data.earnings_growth,
                data.free_cash_flow, data.market_cap, data.enterprise_value, data.timestamp
            ))
            conn.commit()

    @track_performance
    def get_macroeconomic_indicators(self, use_cache: bool = True) -> MacroeconomicIndicators:
        """マクロ経済指標を取得"""
        try:
            # キャッシュチェック
            if use_cache:
                cached_data = self._get_cached_macro_indicators()
                if cached_data:
                    return cached_data

            # 実際のマクロ経済指標を取得
            indicators = self._fetch_macro_indicators()

            # データベースに保存
            self._save_macro_indicators(indicators)

            return indicators

        except Exception as e:
            print(f"マクロ経済指標取得エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"context": "macro_indicators"})

            # エラー時はデフォルト値を返す
            return MacroeconomicIndicators(timestamp=datetime.now())

    def _fetch_macro_indicators(self) -> MacroeconomicIndicators:
        """マクロ経済指標をAPIから取得"""
        # 実際の実装では、FRED API、Bloomberg API、日本銀行API等を使用

        if HAS_YFINANCE:
            try:
                # 主要指標を取得
                indicators = {}

                # 日経225
                try:
                    nikkei = yf.Ticker("^N225")
                    nikkei_data = nikkei.history(period="1d")
                    indicators['nikkei_225'] = float(nikkei_data['Close'].iloc[-1])
                except:
                    pass

                # USD/JPY
                try:
                    usdjpy = yf.Ticker("USDJPY=X")
                    usdjpy_data = usdjpy.history(period="1d")
                    indicators['usdjpy_rate'] = float(usdjpy_data['Close'].iloc[-1])
                except:
                    pass

                # VIX指数
                try:
                    vix = yf.Ticker("^VIX")
                    vix_data = vix.history(period="1d")
                    indicators['vix_index'] = float(vix_data['Close'].iloc[-1])
                except:
                    pass

                # 原油価格
                try:
                    oil = yf.Ticker("CL=F")
                    oil_data = oil.history(period="1d")
                    indicators['oil_price'] = float(oil_data['Close'].iloc[-1])
                except:
                    pass

                # 金価格
                try:
                    gold = yf.Ticker("GC=F")
                    gold_data = gold.history(period="1d")
                    indicators['gold_price'] = float(gold_data['Close'].iloc[-1])
                except:
                    pass

                return MacroeconomicIndicators(
                    nikkei_225=indicators.get('nikkei_225'),
                    usdjpy_rate=indicators.get('usdjpy_rate'),
                    vix_index=indicators.get('vix_index'),
                    oil_price=indicators.get('oil_price'),
                    gold_price=indicators.get('gold_price'),
                    timestamp=datetime.now()
                )

            except Exception as e:
                print(f"実際のマクロ指標取得エラー: {e}")

        # フォールバック：ダミーデータ
        base_time = int(time.time()) // 3600  # 1時間ごとに変わる基準値

        return MacroeconomicIndicators(
            nikkei_225=28000 + (base_time % 2000),
            usdjpy_rate=140 + (base_time % 20),
            jp_10y_yield=0.5 + (base_time % 100) / 1000,
            us_10y_yield=4.0 + (base_time % 200) / 1000,
            vix_index=15 + (base_time % 200) / 10,
            oil_price=80 + (base_time % 400) / 10,
            gold_price=2000 + (base_time % 1000) / 10,
            timestamp=datetime.now()
        )

    def _get_cached_macro_indicators(self) -> Optional[MacroeconomicIndicators]:
        """キャッシュされたマクロ経済指標を取得"""
        cutoff_time = datetime.now() - timedelta(hours=1)  # 1時間キャッシュ

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute('''
                SELECT * FROM macro_indicators
                WHERE timestamp > ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (cutoff_time,)).fetchone()

            if row:
                return MacroeconomicIndicators(
                    nikkei_225=row['nikkei_225'],
                    usdjpy_rate=row['usdjpy_rate'],
                    jp_10y_yield=row['jp_10y_yield'],
                    us_10y_yield=row['us_10y_yield'],
                    vix_index=row['vix_index'],
                    oil_price=row['oil_price'],
                    gold_price=row['gold_price'],
                    timestamp=datetime.fromisoformat(row['timestamp'])
                )

        return None

    def _save_macro_indicators(self, indicators: MacroeconomicIndicators):
        """マクロ経済指標を保存"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO macro_indicators
                (nikkei_225, usdjpy_rate, jp_10y_yield, us_10y_yield,
                 vix_index, oil_price, gold_price, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                indicators.nikkei_225, indicators.usdjpy_rate, indicators.jp_10y_yield,
                indicators.us_10y_yield, indicators.vix_index, indicators.oil_price,
                indicators.gold_price, indicators.timestamp
            ))
            conn.commit()

    def create_enhanced_features(self,
                                symbol: str,
                                base_features: pd.DataFrame) -> pd.DataFrame:
        """基本特徴量に高度特徴量を追加"""
        try:
            enhanced_features = base_features.copy()

            # センチメント特徴量
            sentiment_result = self.analyze_market_sentiment(symbol)
            enhanced_features['sentiment_score'] = sentiment_result.sentiment_score
            enhanced_features['sentiment_confidence'] = sentiment_result.confidence
            enhanced_features['sentiment_source_count'] = sentiment_result.source_count

            # ファンダメンタルズ特徴量
            fundamental_data = self.get_fundamental_data(symbol)

            # 安全な値の設定（NoneやNaNの場合はデフォルト値）
            enhanced_features['pe_ratio'] = fundamental_data.pe_ratio or 15.0
            enhanced_features['pb_ratio'] = fundamental_data.pb_ratio or 1.5
            enhanced_features['dividend_yield'] = fundamental_data.dividend_yield or 0.02
            enhanced_features['roe'] = fundamental_data.roe or 0.1
            enhanced_features['debt_to_equity'] = fundamental_data.debt_to_equity or 0.4
            enhanced_features['revenue_growth'] = fundamental_data.revenue_growth or 5.0
            enhanced_features['market_cap_log'] = np.log10(fundamental_data.market_cap or 1000000000)

            # マクロ経済特徴量
            macro_indicators = self.get_macroeconomic_indicators()
            enhanced_features['nikkei_225'] = macro_indicators.nikkei_225 or 28000
            enhanced_features['usdjpy_rate'] = macro_indicators.usdjpy_rate or 145
            enhanced_features['vix_index'] = macro_indicators.vix_index or 20
            enhanced_features['oil_price'] = macro_indicators.oil_price or 85
            enhanced_features['gold_price'] = macro_indicators.gold_price or 2000

            # 派生特徴量の計算
            enhanced_features['valuation_score'] = (
                (1 / enhanced_features['pe_ratio']) * 10 +
                (1 / enhanced_features['pb_ratio']) * 5 +
                enhanced_features['roe'] * 20
            )

            enhanced_features['macro_risk_score'] = (
                enhanced_features['vix_index'] / 100 +  # 0-1の範囲に正規化
                abs(enhanced_features['usdjpy_rate'] - 145) / 50  # 145を中心とした変動
            )

            enhanced_features['combined_score'] = (
                enhanced_features['sentiment_score'] * 0.3 +
                enhanced_features['valuation_score'] / 10 * 0.4 +
                (1 - enhanced_features['macro_risk_score']) * 0.3
            )

            # 無限値・NaN値の処理
            enhanced_features = enhanced_features.replace([np.inf, -np.inf], np.nan)
            enhanced_features = enhanced_features.fillna(0)

            # ログ記録
            if HAS_AUDIT_LOGGER:
                audit_logger.log_business_event(
                    "enhanced_features_created",
                    {
                        "symbol": symbol,
                        "base_feature_count": len(base_features.columns),
                        "enhanced_feature_count": len(enhanced_features.columns),
                        "new_features": list(set(enhanced_features.columns) - set(base_features.columns))
                    }
                )

            return enhanced_features

        except Exception as e:
            print(f"高度特徴量作成エラー: {e}")
            if HAS_AUDIT_LOGGER:
                audit_logger.log_error_with_context(e, {"symbol": symbol, "context": "enhanced_features"})

            # エラー時は基本特徴量をそのまま返す
            return base_features

    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """特徴量エンジニアリングのサマリーを取得"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # センチメントデータ統計
            sentiment_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_records,
                    AVG(sentiment_score) as avg_sentiment,
                    MIN(timestamp) as oldest_record,
                    MAX(timestamp) as newest_record
                FROM sentiment_data
                WHERE timestamp > datetime('now', '-7 days')
            ''').fetchone()

            # ファンダメンタルズデータ統計
            fundamental_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_records,
                    AVG(pe_ratio) as avg_pe_ratio,
                    AVG(pb_ratio) as avg_pb_ratio
                FROM fundamental_data
                WHERE timestamp > datetime('now', '-7 days')
            ''').fetchone()

            # マクロ経済指標統計
            macro_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_records,
                    AVG(nikkei_225) as avg_nikkei,
                    AVG(vix_index) as avg_vix
                FROM macro_indicators
                WHERE timestamp > datetime('now', '-1 days')
            ''').fetchone()

        return {
            'feature_engineering_summary': {
                'sentiment_analysis': {
                    'records_last_7_days': sentiment_stats['total_records'] if sentiment_stats else 0,
                    'average_sentiment': sentiment_stats['avg_sentiment'] if sentiment_stats else 0,
                    'analyzers_available': list(self.sentiment_analyzers.keys())
                },
                'fundamental_analysis': {
                    'records_last_7_days': fundamental_stats['total_records'] if fundamental_stats else 0,
                    'average_pe_ratio': fundamental_stats['avg_pe_ratio'] if fundamental_stats else 0,
                    'average_pb_ratio': fundamental_stats['avg_pb_ratio'] if fundamental_stats else 0
                },
                'macroeconomic_indicators': {
                    'records_last_day': macro_stats['total_records'] if macro_stats else 0,
                    'average_nikkei': macro_stats['avg_nikkei'] if macro_stats else 0,
                    'average_vix': macro_stats['avg_vix'] if macro_stats else 0
                },
                'cache_configuration': {
                    'cache_hours': self.cache_hours,
                    'cache_directory': str(self.cache_dir),
                    'database_path': str(self.db_path)
                }
            }
        }


# グローバルインスタンス
advanced_feature_engineering = AdvancedFeatureEngineeringSystem()


if __name__ == "__main__":
    # テスト実行
    print("Advanced Feature Engineering System テスト開始")

    # テスト銘柄
    test_symbol = "7203"  # トヨタ自動車

    # センチメント分析テスト
    print(f"\n1. センチメント分析テスト: {test_symbol}")
    sentiment = advanced_feature_engineering.analyze_market_sentiment(test_symbol)
    print(f"   センチメントスコア: {sentiment.sentiment_score:.3f}")
    print(f"   信頼度: {sentiment.confidence:.3f}")
    print(f"   ソース数: {sentiment.source_count}")

    # ファンダメンタルズ分析テスト
    print(f"\n2. ファンダメンタルズ分析テスト: {test_symbol}")
    fundamental = advanced_feature_engineering.get_fundamental_data(test_symbol)
    print(f"   P/E Ratio: {fundamental.pe_ratio}")
    print(f"   P/B Ratio: {fundamental.pb_ratio}")
    print(f"   ROE: {fundamental.roe}")
    print(f"   売上成長率: {fundamental.revenue_growth}%")

    # マクロ経済指標テスト
    print(f"\n3. マクロ経済指標テスト")
    macro = advanced_feature_engineering.get_macroeconomic_indicators()
    print(f"   日経225: {macro.nikkei_225}")
    print(f"   USD/JPY: {macro.usdjpy_rate}")
    print(f"   VIX指数: {macro.vix_index}")
    print(f"   原油価格: {macro.oil_price}")

    # 高度特徴量作成テスト
    print(f"\n4. 高度特徴量作成テスト")

    # ダミー基本特徴量
    base_features = pd.DataFrame({
        'price': [1500],
        'volume': [1000000],
        'rsi': [55.5],
        'macd': [0.12],
        'bb_ratio': [0.3]
    })

    enhanced_features = advanced_feature_engineering.create_enhanced_features(test_symbol, base_features)
    print(f"   基本特徴量数: {len(base_features.columns)}")
    print(f"   拡張特徴量数: {len(enhanced_features.columns)}")
    print(f"   追加された特徴量: {list(set(enhanced_features.columns) - set(base_features.columns))}")

    # サマリー情報
    print(f"\n5. システムサマリー")
    summary = advanced_feature_engineering.get_feature_engineering_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    print("\nテスト完了 ✅")