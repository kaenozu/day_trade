#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Search Engine - 高速検索・フィルタエンジン
Issue #949対応: 高速検索 + 複合フィルタ + インテリジェント検索
"""

import re
import json
import sqlite3
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 全文検索（オプショナル）
try:
    import whoosh
    from whoosh.index import create_index, open_index
    from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, NUMERIC, ID
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.scoring import BM25F
    HAS_WHOOSH = True
except ImportError:
    HAS_WHOOSH = False

# 機械学習（オプショナル）
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class SearchType(Enum):
    """検索タイプ"""
    SYMBOL = "SYMBOL"           # 銘柄検索
    COMPANY = "COMPANY"         # 企業名検索
    SECTOR = "SECTOR"           # セクター検索
    AI_SIGNAL = "AI_SIGNAL"     # AI信号検索
    NEWS = "NEWS"               # ニュース検索
    ANALYSIS = "ANALYSIS"       # 分析データ検索
    PERFORMANCE = "PERFORMANCE" # パフォーマンス検索
    GLOBAL = "GLOBAL"           # 全域検索


class FilterOperator(Enum):
    """フィルタ演算子"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IN = "in"
    NOT_IN = "not_in"
    RANGE = "range"
    REGEX = "regex"


class SortOrder(Enum):
    """ソート順"""
    ASC = "asc"
    DESC = "desc"


@dataclass
class SearchFilter:
    """検索フィルタ"""
    field: str
    operator: FilterOperator
    value: Union[str, int, float, List[Any]]
    case_sensitive: bool = False


@dataclass
class SortCriteria:
    """ソート条件"""
    field: str
    order: SortOrder = SortOrder.ASC
    weight: float = 1.0


@dataclass
class SearchRequest:
    """検索リクエスト"""
    query: str
    search_type: SearchType = SearchType.GLOBAL
    filters: List[SearchFilter] = None
    sort_criteria: List[SortCriteria] = None
    limit: int = 100
    offset: int = 0
    highlight: bool = True
    fuzzy_search: bool = False
    search_fields: List[str] = None


@dataclass
class SearchResult:
    """検索結果"""
    item_id: str
    item_type: str
    title: str
    content: str
    score: float
    highlighted_content: str
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class SearchResponse:
    """検索レスポンス"""
    results: List[SearchResult]
    total_count: int
    search_time_ms: float
    filters_applied: List[SearchFilter]
    suggestions: List[str] = None
    facets: Dict[str, Any] = None


class SearchIndex:
    """検索インデックス"""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.index_path = f"data/search_indexes/{index_name}"
        
        if HAS_WHOOSH:
            self._setup_whoosh_index()
        else:
            self._setup_sqlite_index()
        
        self.tfidf_vectorizer = None
        if HAS_SKLEARN:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # 日本語対応のためNone
                ngram_range=(1, 2)
            )
    
    def _setup_whoosh_index(self):
        """Whoosh検索インデックス設定"""
        import os
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # スキーマ定義
        schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, phrase=True),
            content=TEXT(stored=True),
            item_type=KEYWORD(stored=True),
            symbol=KEYWORD(stored=True),
            sector=KEYWORD(stored=True),
            price=NUMERIC(stored=True),
            confidence=NUMERIC(stored=True),
            timestamp=DATETIME(stored=True),
            metadata=TEXT(stored=True)
        )
        
        try:
            self.whoosh_index = open_index(self.index_path)
        except:
            self.whoosh_index = create_index(schema, self.index_path)
    
    def _setup_sqlite_index(self):
        """SQLite検索インデックス設定"""
        self.db_path = f"{self.index_path}.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_items (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                item_type TEXT,
                symbol TEXT,
                sector TEXT,
                price REAL,
                confidence REAL,
                timestamp DATETIME,
                metadata TEXT
            )
        """)
        
        # 全文検索用仮想テーブル（FTS5）
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_items_fts USING fts5(
                id, title, content, item_type, symbol,
                content='search_items'
            )
        """)
        
        # インデックス作成
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_item_type ON search_items(item_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON search_items(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON search_items(timestamp)")
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc_id: str, title: str, content: str, 
                    item_type: str, **metadata):
        """ドキュメント追加"""
        if HAS_WHOOSH:
            self._add_whoosh_document(doc_id, title, content, item_type, **metadata)
        else:
            self._add_sqlite_document(doc_id, title, content, item_type, **metadata)
    
    def _add_whoosh_document(self, doc_id: str, title: str, content: str,
                           item_type: str, **metadata):
        """Whooshドキュメント追加"""
        writer = self.whoosh_index.writer()
        
        writer.add_document(
            id=doc_id,
            title=title,
            content=content,
            item_type=item_type,
            symbol=metadata.get('symbol', ''),
            sector=metadata.get('sector', ''),
            price=metadata.get('price', 0.0),
            confidence=metadata.get('confidence', 0.0),
            timestamp=metadata.get('timestamp', datetime.now()),
            metadata=json.dumps(metadata)
        )
        
        writer.commit()
    
    def _add_sqlite_document(self, doc_id: str, title: str, content: str,
                           item_type: str, **metadata):
        """SQLiteドキュメント追加"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO search_items 
            (id, title, content, item_type, symbol, sector, price, confidence, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, title, content, item_type,
            metadata.get('symbol', ''),
            metadata.get('sector', ''),
            metadata.get('price', 0.0),
            metadata.get('confidence', 0.0),
            metadata.get('timestamp', datetime.now()),
            json.dumps(metadata)
        ))
        
        # FTSテーブルも更新
        cursor.execute("""
            INSERT OR REPLACE INTO search_items_fts (id, title, content, item_type, symbol)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, title, content, item_type, metadata.get('symbol', '')))
        
        conn.commit()
        conn.close()
    
    def search(self, request: SearchRequest) -> SearchResponse:
        """検索実行"""
        start_time = time.time()
        
        if HAS_WHOOSH:
            results = self._whoosh_search(request)
        else:
            results = self._sqlite_search(request)
        
        search_time = (time.time() - start_time) * 1000
        
        # 結果に順位付け
        if request.sort_criteria:
            results = self._sort_results(results, request.sort_criteria)
        
        # ハイライト処理
        if request.highlight:
            self._highlight_results(results, request.query)
        
        # ページング
        total_count = len(results)
        start_idx = request.offset
        end_idx = start_idx + request.limit
        paged_results = results[start_idx:end_idx]
        
        return SearchResponse(
            results=paged_results,
            total_count=total_count,
            search_time_ms=search_time,
            filters_applied=request.filters or [],
            suggestions=self._generate_suggestions(request.query),
            facets=self._generate_facets(results)
        )
    
    def _whoosh_search(self, request: SearchRequest) -> List[SearchResult]:
        """Whoosh検索実行"""
        with self.whoosh_index.searcher(weighting=BM25F()) as searcher:
            
            # クエリパーサー作成
            if request.search_fields:
                parser = MultifieldParser(request.search_fields, self.whoosh_index.schema)
            else:
                parser = MultifieldParser(["title", "content", "symbol"], self.whoosh_index.schema)
            
            # クエリ解析
            if request.fuzzy_search:
                query_str = f"{request.query}~2"  # 編集距離2の曖昧検索
            else:
                query_str = request.query
            
            query = parser.parse(query_str)
            
            # フィルタ適用
            if request.filters:
                filter_query = self._build_whoosh_filter(request.filters)
                if filter_query:
                    query = query & filter_query
            
            # 検索実行
            whoosh_results = searcher.search(query, limit=None)
            
            # 結果変換
            results = []
            for hit in whoosh_results:
                metadata = json.loads(hit['metadata']) if hit['metadata'] else {}
                
                result = SearchResult(
                    item_id=hit['id'],
                    item_type=hit['item_type'],
                    title=hit['title'],
                    content=hit['content'],
                    score=hit.score,
                    highlighted_content="",
                    metadata=metadata,
                    timestamp=hit['timestamp']
                )
                results.append(result)
            
            return results
    
    def _sqlite_search(self, request: SearchRequest) -> List[SearchResult]:
        """SQLite検索実行"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FTS検索クエリ構築
        if request.query.strip():
            fts_query = f"""
                SELECT si.*, rank
                FROM search_items_fts 
                JOIN search_items si ON search_items_fts.id = si.id
                WHERE search_items_fts MATCH ?
                ORDER BY rank
            """
            cursor.execute(fts_query, (request.query,))
        else:
            # 全件取得
            cursor.execute("SELECT *, 1.0 as rank FROM search_items")
        
        rows = cursor.fetchall()
        conn.close()
        
        # 結果変換
        results = []
        for row in rows:
            metadata = json.loads(row[9]) if row[9] else {}
            
            result = SearchResult(
                item_id=row[0],
                item_type=row[3],
                title=row[1],
                content=row[2],
                score=float(row[-1]),  # rank
                highlighted_content="",
                metadata=metadata,
                timestamp=datetime.fromisoformat(row[8]) if row[8] else datetime.now()
            )
            results.append(result)
        
        # フィルタ適用
        if request.filters:
            results = self._apply_filters(results, request.filters)
        
        return results
    
    def _apply_filters(self, results: List[SearchResult], 
                      filters: List[SearchFilter]) -> List[SearchResult]:
        """フィルタ適用"""
        filtered_results = []
        
        for result in results:
            matches = True
            
            for filter_item in filters:
                field_value = self._get_field_value(result, filter_item.field)
                
                if not self._match_filter(field_value, filter_item):
                    matches = False
                    break
            
            if matches:
                filtered_results.append(result)
        
        return filtered_results
    
    def _get_field_value(self, result: SearchResult, field: str) -> Any:
        """フィールド値取得"""
        if hasattr(result, field):
            return getattr(result, field)
        elif field in result.metadata:
            return result.metadata[field]
        else:
            return None
    
    def _match_filter(self, field_value: Any, filter_item: SearchFilter) -> bool:
        """フィルタマッチング"""
        if field_value is None:
            return False
        
        operator = filter_item.operator
        filter_value = filter_item.value
        
        # 文字列の場合の大小文字処理
        if isinstance(field_value, str) and isinstance(filter_value, str):
            if not filter_item.case_sensitive:
                field_value = field_value.lower()
                filter_value = filter_value.lower()
        
        if operator == FilterOperator.EQUALS:
            return field_value == filter_value
        elif operator == FilterOperator.NOT_EQUALS:
            return field_value != filter_value
        elif operator == FilterOperator.GREATER_THAN:
            return field_value > filter_value
        elif operator == FilterOperator.LESS_THAN:
            return field_value < filter_value
        elif operator == FilterOperator.GREATER_EQUAL:
            return field_value >= filter_value
        elif operator == FilterOperator.LESS_EQUAL:
            return field_value <= filter_value
        elif operator == FilterOperator.CONTAINS:
            return str(filter_value) in str(field_value)
        elif operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(filter_value))
        elif operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(filter_value))
        elif operator == FilterOperator.IN:
            return field_value in filter_value
        elif operator == FilterOperator.NOT_IN:
            return field_value not in filter_value
        elif operator == FilterOperator.RANGE:
            return filter_value[0] <= field_value <= filter_value[1]
        elif operator == FilterOperator.REGEX:
            return bool(re.search(str(filter_value), str(field_value)))
        
        return False
    
    def _sort_results(self, results: List[SearchResult], 
                     criteria: List[SortCriteria]) -> List[SearchResult]:
        """結果ソート"""
        def sort_key(result):
            key_values = []
            for criterion in criteria:
                value = self._get_field_value(result, criterion.field)
                if value is None:
                    value = 0 if isinstance(value, (int, float)) else ""
                
                if criterion.order == SortOrder.DESC:
                    if isinstance(value, (int, float)):
                        value = -value
                    else:
                        # 文字列の場合は逆順ソートが複雑
                        pass
                
                key_values.append(value)
            
            return key_values
        
        return sorted(results, key=sort_key)
    
    def _highlight_results(self, results: List[SearchResult], query: str):
        """検索結果ハイライト"""
        if not query.strip():
            return
        
        query_terms = query.lower().split()
        
        for result in results:
            highlighted = result.content
            
            for term in query_terms:
                if len(term) < 2:  # 短すぎる語句はスキップ
                    continue
                
                # 大小文字を無視してハイライト
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted = pattern.sub(f"<mark>{term}</mark>", highlighted)
            
            result.highlighted_content = highlighted
    
    def _generate_suggestions(self, query: str) -> List[str]:
        """検索候補生成"""
        # 簡単な候補生成（実際はより高度なアルゴリズムを使用）
        suggestions = []
        
        if query:
            # よくある検索語句
            common_terms = [
                "トヨタ", "ソフトバンク", "三菱UFJ", "ソニー", "ZHD",
                "7203", "9984", "8306", "6758", "4689",
                "BUY", "SELL", "HOLD", "高信頼度", "低リスク"
            ]
            
            query_lower = query.lower()
            for term in common_terms:
                if query_lower in term.lower() and term.lower() != query_lower:
                    suggestions.append(term)
            
            # 最大5件まで
            suggestions = suggestions[:5]
        
        return suggestions
    
    def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Any]:
        """ファセット生成"""
        facets = {}
        
        # アイテムタイプファセット
        item_types = Counter([result.item_type for result in results])
        facets['item_type'] = dict(item_types.most_common())
        
        # セクターファセット
        sectors = []
        for result in results:
            sector = result.metadata.get('sector')
            if sector:
                sectors.append(sector)
        
        facets['sector'] = dict(Counter(sectors).most_common())
        
        # 価格範囲ファセット
        prices = [result.metadata.get('price', 0) for result in results if result.metadata.get('price', 0) > 0]
        if prices:
            facets['price_range'] = {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) / len(prices)
            }
        
        return facets


class AdvancedSearchEngine:
    """高度検索エンジン"""
    
    def __init__(self):
        self.indexes = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # メインインデックス作成
        self.main_index = SearchIndex("main")
        self.indexes["main"] = self.main_index
        
        # データ投入
        self._populate_sample_data()
        
        # 検索統計
        self.search_stats = {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'popular_queries': Counter(),
            'failed_searches': 0
        }
    
    def _populate_sample_data(self):
        """サンプルデータ投入"""
        # 株式データ
        stocks = [
            {"symbol": "7203", "name": "トヨタ自動車", "sector": "自動車", "price": 1850.5},
            {"symbol": "8306", "name": "三菱UFJフィナンシャル・グループ", "sector": "金融", "price": 642.1},
            {"symbol": "9984", "name": "ソフトバンクグループ", "sector": "通信", "price": 4985.0},
            {"symbol": "6758", "name": "ソニーグループ", "sector": "電機", "price": 10450.0},
            {"symbol": "4689", "name": "Z Holdings", "sector": "情報・通信", "price": 385.4}
        ]
        
        for stock in stocks:
            self.main_index.add_document(
                doc_id=f"stock_{stock['symbol']}",
                title=f"{stock['name']} ({stock['symbol']})",
                content=f"銘柄コード: {stock['symbol']} 企業名: {stock['name']} セクター: {stock['sector']} 現在価格: ¥{stock['price']}",
                item_type="STOCK",
                symbol=stock['symbol'],
                sector=stock['sector'],
                price=stock['price'],
                timestamp=datetime.now()
            )
        
        # AI分析データ
        ai_analyses = [
            {"symbol": "7203", "signal": "BUY", "confidence": 0.85, "reason": "技術指標改善"},
            {"symbol": "8306", "signal": "HOLD", "confidence": 0.72, "reason": "市場環境待機"},
            {"symbol": "9984", "signal": "SELL", "confidence": 0.91, "reason": "過熱感あり"}
        ]
        
        for analysis in ai_analyses:
            self.main_index.add_document(
                doc_id=f"ai_{analysis['symbol']}",
                title=f"AI分析: {analysis['symbol']} - {analysis['signal']}",
                content=f"銘柄: {analysis['symbol']} 推奨: {analysis['signal']} 信頼度: {analysis['confidence']:.1%} 理由: {analysis['reason']}",
                item_type="AI_ANALYSIS",
                symbol=analysis['symbol'],
                signal=analysis['signal'],
                confidence=analysis['confidence'],
                timestamp=datetime.now()
            )
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """非同期検索"""
        try:
            self.search_stats['total_searches'] += 1
            self.search_stats['popular_queries'][request.query] += 1
            
            # 検索実行
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor, 
                self.main_index.search, 
                request
            )
            
            # 統計更新
            self.search_stats['avg_response_time'] = (
                self.search_stats['avg_response_time'] * 0.9 + 
                response.search_time_ms * 0.1
            )
            
            return response
            
        except Exception as e:
            self.search_stats['failed_searches'] += 1
            logging.error(f"Search failed: {e}")
            
            return SearchResponse(
                results=[],
                total_count=0,
                search_time_ms=0.0,
                filters_applied=[],
                suggestions=[]
            )
    
    def search_sync(self, request: SearchRequest) -> SearchResponse:
        """同期検索"""
        return self.main_index.search(request)
    
    def add_document(self, doc_id: str, title: str, content: str,
                    item_type: str, **metadata):
        """ドキュメント追加"""
        self.main_index.add_document(doc_id, title, content, item_type, **metadata)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """検索統計取得"""
        return {
            'total_searches': self.search_stats['total_searches'],
            'avg_response_time_ms': self.search_stats['avg_response_time'],
            'popular_queries': dict(self.search_stats['popular_queries'].most_common(10)),
            'failed_searches': self.search_stats['failed_searches'],
            'success_rate': 1.0 - (self.search_stats['failed_searches'] / max(1, self.search_stats['total_searches']))
        }
    
    def create_saved_search(self, name: str, request: SearchRequest) -> str:
        """保存検索作成"""
        saved_search_id = hashlib.md5(f"{name}_{datetime.now()}".encode()).hexdigest()[:16]
        
        # 保存検索をファイルに保存（簡略版）
        saved_searches_path = "data/saved_searches.json"
        os.makedirs(os.path.dirname(saved_searches_path), exist_ok=True)
        
        saved_searches = {}
        if os.path.exists(saved_searches_path):
            with open(saved_searches_path, 'r', encoding='utf-8') as f:
                saved_searches = json.load(f)
        
        saved_searches[saved_search_id] = {
            'name': name,
            'request': asdict(request),
            'created_at': datetime.now().isoformat()
        }
        
        with open(saved_searches_path, 'w', encoding='utf-8') as f:
            json.dump(saved_searches, f, default=str, ensure_ascii=False, indent=2)
        
        return saved_search_id


# グローバルインスタンス
advanced_search_engine = AdvancedSearchEngine()


def search(query: str, search_type: SearchType = SearchType.GLOBAL,
          filters: List[SearchFilter] = None, limit: int = 20) -> SearchResponse:
    """簡易検索API"""
    request = SearchRequest(
        query=query,
        search_type=search_type,
        filters=filters or [],
        limit=limit
    )
    return advanced_search_engine.search_sync(request)


def search_stocks(query: str, min_price: float = None, max_price: float = None) -> List[SearchResult]:
    """株式検索"""
    filters = []
    if min_price is not None:
        filters.append(SearchFilter("price", FilterOperator.GREATER_EQUAL, min_price))
    if max_price is not None:
        filters.append(SearchFilter("price", FilterOperator.LESS_EQUAL, max_price))
    
    request = SearchRequest(
        query=query,
        search_type=SearchType.SYMBOL,
        filters=filters,
        limit=50
    )
    
    response = advanced_search_engine.search_sync(request)
    return response.results


def search_ai_signals(signal_type: str = None, min_confidence: float = None) -> List[SearchResult]:
    """AI信号検索"""
    filters = []
    if signal_type:
        filters.append(SearchFilter("signal", FilterOperator.EQUALS, signal_type))
    if min_confidence is not None:
        filters.append(SearchFilter("confidence", FilterOperator.GREATER_EQUAL, min_confidence))
    
    request = SearchRequest(
        query="",
        search_type=SearchType.AI_SIGNAL,
        filters=filters,
        sort_criteria=[SortCriteria("confidence", SortOrder.DESC)],
        limit=100
    )
    
    response = advanced_search_engine.search_sync(request)
    return response.results


async def test_search_engine():
    """検索エンジンテスト"""
    print("=== Advanced Search Engine Test ===")
    
    # 基本検索テスト
    print("1. Basic search test:")
    results = search("トヨタ", limit=5)
    print(f"Query: 'トヨタ' - Found {results.total_count} results in {results.search_time_ms:.1f}ms")
    for result in results.results[:3]:
        print(f"  - {result.title} (score: {result.score:.3f})")
    
    # フィルタ検索テスト
    print("\n2. Filter search test:")
    stock_results = search_stocks("", min_price=1000, max_price=5000)
    print(f"Stocks between ¥1000-5000: {len(stock_results)} found")
    for result in stock_results[:3]:
        price = result.metadata.get('price', 0)
        print(f"  - {result.title}: ¥{price}")
    
    # AI信号検索テスト
    print("\n3. AI signal search test:")
    ai_results = search_ai_signals(min_confidence=0.8)
    print(f"High confidence AI signals: {len(ai_results)} found")
    for result in ai_results:
        confidence = result.metadata.get('confidence', 0)
        print(f"  - {result.title}: {confidence:.1%} confidence")
    
    # 検索統計
    print("\n4. Search statistics:")
    stats = advanced_search_engine.get_search_statistics()
    print(f"Total searches: {stats['total_searches']}")
    print(f"Average response time: {stats['avg_response_time_ms']:.1f}ms")
    print(f"Success rate: {stats['success_rate']:.1%}")
    
    print("\nSearch engine test completed!")


if __name__ == "__main__":
    import os
    os.makedirs('data/search_indexes', exist_ok=True)
    
    import asyncio
    asyncio.run(test_search_engine())