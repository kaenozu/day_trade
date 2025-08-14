#!/usr/bin/env python3
"""
Data Management Service - Microservice
Day Trade ML System - Data Fetching & Management API

マイクロサービス: データ管理サービス
責任範囲: データ取得、前処理、キャッシュ管理
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
import json
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# プロジェクトモジュールインポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenTelemetry設定
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=14268,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Prometheus メトリクス
DATA_REQUESTS = Counter('data_requests_total', 'Total data requests', ['method', 'status'])
DATA_FETCH_LATENCY = Histogram('data_fetch_duration_seconds', 'Data fetch request duration')
CACHE_SIZE = Gauge('data_cache_size_bytes', 'Current cache size in bytes')
CACHE_HITS = Counter('data_cache_hits_total', 'Data cache hits')
CACHE_MISSES = Counter('data_cache_misses_total', 'Data cache misses')
ACTIVE_CONNECTIONS = Gauge('data_active_connections', 'Number of active data connections')

# Pydantic Models
class StockDataRequest(BaseModel):
    symbol: str = Field(..., description="株式銘柄コード", regex="^[A-Z0-9]{1,10}$")
    period: Optional[str] = Field("1mo", description="取得期間")
    interval: Optional[str] = Field("1d", description="データ間隔")
    include_features: Optional[bool] = Field(True, description="テクニカル指標計算フラグ")
    cache_enabled: Optional[bool] = Field(True, description="キャッシュ使用フラグ")

    @validator('period')
    def validate_period(cls, v):
        valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if v not in valid_periods:
            raise ValueError(f"無効な期間です。有効な値: {valid_periods}")
        return v

    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]
        if v not in valid_intervals:
            raise ValueError(f"無効な間隔です。有効な値: {valid_intervals}")
        return v

class BatchDataRequest(BaseModel):
    symbols: List[str] = Field(..., description="株式銘柄コードリスト")
    period: Optional[str] = Field("1mo", description="取得期間")
    interval: Optional[str] = Field("1d", description="データ間隔")
    include_features: Optional[bool] = Field(True, description="テクニカル指標計算フラグ")
    parallel_processing: Optional[bool] = Field(True, description="並列処理フラグ")

    @validator('symbols')
    def validate_symbols(cls, v):
        if len(v) > 50:
            raise ValueError("一度に処理できる銘柄数は50個までです")
        return v

class StockPrice(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

class StockDataPoint(BaseModel):
    price: StockPrice
    indicators: Optional[TechnicalIndicators] = None

class StockDataResponse(BaseModel):
    symbol: str
    period: str
    interval: str
    data_points: List[StockDataPoint]
    total_points: int
    cache_hit: bool = False
    fetch_timestamp: datetime
    metadata: Optional[Dict] = None

class BatchDataResponse(BaseModel):
    job_id: str
    total_symbols: int
    successful_fetches: int
    failed_fetches: int
    data: List[StockDataResponse]
    processing_time_seconds: float
    timestamp: datetime

class CacheStats(BaseModel):
    total_entries: int
    size_bytes: int
    hit_rate: float
    ttl_average_seconds: int
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    cache_stats: CacheStats
    active_connections: int
    timestamp: datetime

# FastAPI アプリケーション
app = FastAPI(
    title="Data Management Service",
    description="Day Trade ML System - Data Fetching & Management API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ミドルウェア設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus メトリクス エンドポイント
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# グローバル変数
redis_client: Optional[redis.Redis] = None
app_start_time = datetime.utcnow()

# 依存性注入
async def get_redis_client() -> redis.Redis:
    """Redis依存性注入"""
    global redis_client
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redisクライアントが初期化されていません"
        )
    return redis_client

# ヘルパー関数
async def generate_cache_key(symbol: str, period: str, interval: str) -> str:
    """キャッシュキー生成"""
    return f"stock_data:{symbol}:{period}:{interval}"

async def fetch_stock_data_external(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """外部APIからデータ取得（模擬実装）"""
    # 実際の実装では yfinance や他のAPIを使用
    await asyncio.sleep(0.1)  # 外部API呼び出しシミュレート

    # 模擬データ生成
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'Open': [100 + i + np.random.normal(0, 2) for i in range(30)],
        'High': [102 + i + np.random.normal(0, 2) for i in range(30)],
        'Low': [98 + i + np.random.normal(0, 2) for i in range(30)],
        'Close': [101 + i + np.random.normal(0, 2) for i in range(30)],
        'Volume': [1000000 + np.random.randint(0, 500000) for _ in range(30)]
    }

    df = pd.DataFrame(data, index=dates)
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標計算"""
    # RSI計算（簡易版）
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 移動平均
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # ボリンジャーバンド
    df['BB_Middle'] = df['SMA_20']
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std * 2)

    return df

# API エンドポイント

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時初期化"""
    global redis_client

    logger.info("Data Management Service starting up...")

    try:
        # Redis接続
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", "redis_password_2025")

        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Redis接続テスト
        await redis_client.ping()

        # メトリクス初期化
        ACTIVE_CONNECTIONS.set(0)

        logger.info("Data Management Service startup completed")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時クリーンアップ"""
    global redis_client

    logger.info("Data Management Service shutting down...")

    if redis_client:
        await redis_client.close()

    logger.info("Data Management Service shutdown completed")

@app.get("/health", response_model=HealthResponse)
async def health_check(
    redis_conn: redis.Redis = Depends(get_redis_client)
):
    """ヘルスチェック"""
    try:
        # Redis接続確認
        await redis_conn.ping()

        # キャッシュ統計取得
        cache_info = await redis_conn.info('memory')
        cache_size = cache_info.get('used_memory', 0)

        # 簡易キャッシュ統計
        cache_stats = CacheStats(
            total_entries=100,  # 実際は redis_conn.dbsize()
            size_bytes=cache_size,
            hit_rate=0.85,
            ttl_average_seconds=300,
            oldest_entry=datetime.utcnow() - timedelta(hours=1),
            newest_entry=datetime.utcnow()
        )

        uptime = (datetime.utcnow() - app_start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=uptime,
            cache_stats=cache_stats,
            active_connections=1,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"サービス不正常: {str(e)}"
        )

@app.get("/api/v1/data/stocks/{symbol}", response_model=StockDataResponse)
async def get_stock_data(
    symbol: str,
    period: str = Query("1mo", description="取得期間"),
    interval: str = Query("1d", description="データ間隔"),
    include_features: bool = Query(True, description="テクニカル指標計算フラグ"),
    cache_enabled: bool = Query(True, description="キャッシュ使用フラグ"),
    redis_conn: redis.Redis = Depends(get_redis_client)
):
    """株式データ取得"""
    with tracer.start_as_current_span("data_stock_fetch") as span:
        span.set_attributes({
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "include_features": include_features,
            "cache_enabled": cache_enabled
        })

        start_time = datetime.utcnow()

        try:
            with DATA_FETCH_LATENCY.time():
                # キャッシュ確認
                cache_hit = False
                if cache_enabled:
                    cache_key = await generate_cache_key(symbol, period, interval)
                    cached_data = await redis_conn.get(cache_key)

                    if cached_data:
                        cache_hit = True
                        CACHE_HITS.inc()
                        span.set_attribute("cache_hit", True)

                        cached_result = json.loads(cached_data)
                        DATA_REQUESTS.labels(method="get_stock_data", status="cache_hit").inc()

                        return StockDataResponse(**cached_result)

                CACHE_MISSES.inc()
                ACTIVE_CONNECTIONS.inc()

                try:
                    # 外部APIからデータ取得
                    df = await fetch_stock_data_external(symbol, period, interval)

                    # テクニカル指標計算
                    if include_features:
                        df = calculate_technical_indicators(df)

                    # レスポンス構築
                    data_points = []
                    for idx, row in df.iterrows():
                        price = StockPrice(
                            timestamp=idx,
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=int(row['Volume'])
                        )

                        indicators = None
                        if include_features:
                            indicators = TechnicalIndicators(
                                rsi=row.get('RSI'),
                                macd=row.get('MACD'),
                                macd_signal=row.get('MACD_Signal'),
                                bb_upper=row.get('BB_Upper'),
                                bb_middle=row.get('BB_Middle'),
                                bb_lower=row.get('BB_Lower'),
                                sma_20=row.get('SMA_20'),
                                ema_12=row.get('EMA_12'),
                                ema_26=row.get('EMA_26')
                            )

                        data_points.append(StockDataPoint(
                            price=price,
                            indicators=indicators
                        ))

                    response = StockDataResponse(
                        symbol=symbol,
                        period=period,
                        interval=interval,
                        data_points=data_points,
                        total_points=len(data_points),
                        cache_hit=cache_hit,
                        fetch_timestamp=datetime.utcnow(),
                        metadata={
                            "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                            "data_source": "external_api"
                        }
                    )

                    # キャッシュ保存
                    if cache_enabled:
                        await redis_conn.setex(
                            cache_key,
                            300,  # 5分TTL
                            json.dumps(response.dict(), default=str)
                        )

                    DATA_REQUESTS.labels(method="get_stock_data", status="success").inc()

                    return response

                finally:
                    ACTIVE_CONNECTIONS.dec()

        except Exception as e:
            logger.error(f"Stock data fetch error for {symbol}: {str(e)}")
            span.set_attribute("error", str(e))
            DATA_REQUESTS.labels(method="get_stock_data", status="error").inc()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"データ取得エラー: {str(e)}"
            )

@app.post("/api/v1/data/batch-fetch", response_model=BatchDataResponse)
async def batch_fetch_data(
    request: BatchDataRequest,
    redis_conn: redis.Redis = Depends(get_redis_client)
):
    """バッチデータ取得"""
    with tracer.start_as_current_span("data_batch_fetch") as span:
        span.set_attributes({
            "symbols_count": len(request.symbols),
            "parallel_processing": request.parallel_processing
        })

        start_time = datetime.utcnow()
        job_id = f"batch_data_{int(start_time.timestamp())}"

        try:
            data_responses = []
            successful_fetches = 0
            failed_fetches = 0

            # 並列処理または順次処理
            if request.parallel_processing:
                tasks = []
                for symbol in request.symbols:
                    task = asyncio.create_task(
                        get_stock_data(
                            symbol=symbol,
                            period=request.period,
                            interval=request.interval,
                            include_features=request.include_features,
                            redis_conn=redis_conn
                        )
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        failed_fetches += 1
                    else:
                        data_responses.append(result)
                        successful_fetches += 1
            else:
                for symbol in request.symbols:
                    try:
                        result = await get_stock_data(
                            symbol=symbol,
                            period=request.period,
                            interval=request.interval,
                            include_features=request.include_features,
                            redis_conn=redis_conn
                        )
                        data_responses.append(result)
                        successful_fetches += 1
                    except Exception as e:
                        logger.error(f"Batch fetch error for {symbol}: {str(e)}")
                        failed_fetches += 1

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            span.set_attributes({
                "successful_fetches": successful_fetches,
                "failed_fetches": failed_fetches,
                "processing_time": processing_time
            })

            DATA_REQUESTS.labels(method="batch_fetch", status="success").inc()

            return BatchDataResponse(
                job_id=job_id,
                total_symbols=len(request.symbols),
                successful_fetches=successful_fetches,
                failed_fetches=failed_fetches,
                data=data_responses,
                processing_time_seconds=processing_time,
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Batch fetch error: {str(e)}")
            span.set_attribute("error", str(e))
            DATA_REQUESTS.labels(method="batch_fetch", status="error").inc()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"バッチデータ取得エラー: {str(e)}"
            )

@app.get("/api/v1/data/cache", response_model=CacheStats)
async def get_cache_stats(
    redis_conn: redis.Redis = Depends(get_redis_client)
):
    """キャッシュ統計取得"""
    try:
        # Redis統計取得
        info = await redis_conn.info()
        memory_info = await redis_conn.info('memory')

        total_entries = info.get('db0', {}).get('keys', 0)
        size_bytes = memory_info.get('used_memory', 0)

        # ヒット率計算（実際の実装ではより正確な計算が必要）
        hit_rate = 0.85  # 模擬値

        return CacheStats(
            total_entries=total_entries,
            size_bytes=size_bytes,
            hit_rate=hit_rate,
            ttl_average_seconds=300,
            oldest_entry=datetime.utcnow() - timedelta(hours=1),
            newest_entry=datetime.utcnow()
        )

    except Exception as e:
        logger.error(f"Cache stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"キャッシュ統計取得エラー: {str(e)}"
        )

@app.delete("/api/v1/data/cache")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="削除パターン（省略時は全削除）"),
    redis_conn: redis.Redis = Depends(get_redis_client)
):
    """キャッシュクリア"""
    try:
        if pattern:
            # パターンマッチでキー削除
            keys = await redis_conn.keys(pattern)
            if keys:
                await redis_conn.delete(*keys)
                deleted_count = len(keys)
            else:
                deleted_count = 0
        else:
            # 全キャッシュクリア
            await redis_conn.flushdb()
            deleted_count = "all"

        return {
            "status": "success",
            "deleted_keys": deleted_count,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"キャッシュクリアエラー: {str(e)}"
        )

if __name__ == "__main__":
    # 開発環境での実行
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )