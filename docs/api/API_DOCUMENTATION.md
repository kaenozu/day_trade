# 日間取引システム API ドキュメント

## 目次
1. [概要](#概要)
2. [認証](#認証)
3. [レスポンス形式](#レスポンス形式)
4. [エラーハンドリング](#エラーハンドリング)
5. [Core APIs](#core-apis)
6. [Trading APIs](#trading-apis)
7. [Portfolio APIs](#portfolio-apis)
8. [Market Data APIs](#market-data-apis)
9. [Configuration APIs](#configuration-apis)
10. [Monitoring APIs](#monitoring-apis)

## 概要

日間取引システムは、統一アーキテクチャフレームワークに基づいて構築されたRESTful APIを提供します。すべてのAPIは統一されたレスポンス形式とエラーハンドリングを使用しています。

### ベースURL
```
Production: https://api.daytrading.example.com/v1
Development: http://localhost:8000/v1
```

### Content-Type
```
application/json
```

## 認証

### JWT Bearer Token
すべてのAPI呼び出しにはJWTトークンが必要です。

```http
Authorization: Bearer <jwt_token>
```

### 認証エンドポイント

#### POST /auth/login
ユーザーログイン

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "expires_in": 3600,
    "user_id": "user123"
  },
  "metadata": {
    "timestamp": "2025-08-17T10:00:00Z",
    "operation_id": "login_001"
  }
}
```

## レスポンス形式

### 統一レスポンス形式（OperationResult）

すべてのAPIレスポンスは統一されたOperationResult形式を使用します：

```json
{
  "success": boolean,
  "data": object | array | null,
  "error": {
    "code": "string",
    "message": "string",
    "details": object
  } | null,
  "metadata": {
    "timestamp": "string",
    "operation_id": "string",
    "execution_time_ms": number,
    "source": "string"
  }
}
```

### 成功レスポンス例
```json
{
  "success": true,
  "data": {
    "id": "12345",
    "name": "Example Data"
  },
  "error": null,
  "metadata": {
    "timestamp": "2025-08-17T10:00:00Z",
    "operation_id": "op_001",
    "execution_time_ms": 25.5,
    "source": "trading_service"
  }
}
```

### エラーレスポンス例
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "無効な取引量です",
    "details": {
      "field": "quantity",
      "value": -10,
      "constraint": "must_be_positive"
    }
  },
  "metadata": {
    "timestamp": "2025-08-17T10:00:00Z",
    "operation_id": "op_002",
    "execution_time_ms": 15.2,
    "source": "trading_service"
  }
}
```

## エラーハンドリング

### エラーカテゴリ

| カテゴリ | HTTPステータス | 説明 |
|---------|---------------|------|
| `VALIDATION_ERROR` | 400 | 入力検証エラー |
| `AUTHENTICATION_ERROR` | 401 | 認証エラー |
| `AUTHORIZATION_ERROR` | 403 | 認可エラー |
| `NOT_FOUND_ERROR` | 404 | リソース未発見 |
| `BUSINESS_LOGIC_ERROR` | 422 | ビジネスルール違反 |
| `INFRASTRUCTURE_ERROR` | 500 | インフラストラクチャエラー |
| `EXTERNAL_SERVICE_ERROR` | 502 | 外部サービスエラー |

### リトライ可能エラー

以下のエラーはクライアント側でリトライ可能です：

- `INFRASTRUCTURE_ERROR` (5xx系)
- `EXTERNAL_SERVICE_ERROR` (502, 503, 504)
- `TIMEOUT_ERROR` (408)

推奨リトライ戦略：指数バックオフ（1s, 2s, 4s, 8s, 16s）

## Core APIs

### システム管理API

#### GET /health
ヘルスチェック

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 86400,
    "components": {
      "database": "healthy",
      "redis": "healthy",
      "external_services": "healthy"
    }
  }
}
```

#### GET /metrics
システムメトリクス（要管理者権限）

**Response:**
```json
{
  "success": true,
  "data": {
    "performance": {
      "average_response_time_ms": 125.5,
      "throughput_req_per_sec": 850,
      "error_rate_percent": 0.1
    },
    "resources": {
      "cpu_usage_percent": 45.2,
      "memory_usage_percent": 62.8,
      "disk_usage_percent": 35.1
    }
  }
}
```

## Trading APIs

### 取引実行API

#### POST /trading/execute
取引実行

**Request:**
```json
{
  "portfolio_id": "portfolio_123",
  "symbol": "7203",
  "direction": "buy",
  "quantity": 100,
  "order_type": "market",
  "price": 1500.0,
  "stop_loss": 1400.0,
  "take_profit": 1600.0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "trade_id": "trade_456",
    "executed_price": 1500.0,
    "executed_quantity": 100,
    "commission": 150.0,
    "timestamp": "2025-08-17T10:05:00Z",
    "status": "executed"
  }
}
```

#### GET /trading/trades
取引履歴取得

**Query Parameters:**
- `portfolio_id` (required): ポートフォリオID
- `symbol` (optional): 銘柄コード
- `from_date` (optional): 開始日 (ISO 8601)
- `to_date` (optional): 終了日 (ISO 8601)
- `limit` (optional): 取得件数上限 (default: 100)
- `offset` (optional): オフセット (default: 0)

**Response:**
```json
{
  "success": true,
  "data": {
    "trades": [
      {
        "trade_id": "trade_456",
        "symbol": "7203",
        "direction": "buy",
        "quantity": 100,
        "price": 1500.0,
        "timestamp": "2025-08-17T10:05:00Z",
        "status": "executed"
      }
    ],
    "total_count": 1,
    "has_more": false
  }
}
```

#### DELETE /trading/trades/{trade_id}
取引キャンセル

**Response:**
```json
{
  "success": true,
  "data": {
    "trade_id": "trade_456",
    "status": "cancelled",
    "cancelled_at": "2025-08-17T10:10:00Z"
  }
}
```

## Portfolio APIs

### ポートフォリオ管理API

#### GET /portfolios
ポートフォリオ一覧取得

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolios": [
      {
        "portfolio_id": "portfolio_123",
        "name": "メインポートフォリオ",
        "total_value": 1000000.0,
        "cash_balance": 100000.0,
        "unrealized_pnl": 50000.0,
        "created_at": "2025-01-01T00:00:00Z"
      }
    ]
  }
}
```

#### GET /portfolios/{portfolio_id}
ポートフォリオ詳細取得

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "portfolio_123",
    "name": "メインポートフォリオ",
    "total_value": 1000000.0,
    "cash_balance": 100000.0,
    "positions": [
      {
        "symbol": "7203",
        "quantity": 100,
        "average_price": 1500.0,
        "current_price": 1550.0,
        "unrealized_pnl": 5000.0
      }
    ],
    "performance": {
      "total_return_percent": 5.0,
      "daily_pnl": 1000.0,
      "weekly_pnl": 3000.0,
      "monthly_pnl": 5000.0
    }
  }
}
```

#### POST /portfolios
ポートフォリオ作成

**Request:**
```json
{
  "name": "新しいポートフォリオ",
  "initial_cash": 1000000.0,
  "risk_tolerance": 0.1,
  "strategy": "day_trading"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "portfolio_789",
    "name": "新しいポートフォリオ",
    "initial_cash": 1000000.0,
    "created_at": "2025-08-17T10:15:00Z"
  }
}
```

## Market Data APIs

### 市場データAPI

#### GET /market/quote/{symbol}
リアルタイム株価取得

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "7203",
    "price": 1550.0,
    "change": 50.0,
    "change_percent": 3.33,
    "volume": 1000000,
    "bid": 1549.0,
    "ask": 1551.0,
    "timestamp": "2025-08-17T10:20:00Z"
  }
}
```

#### GET /market/history/{symbol}
過去データ取得

**Query Parameters:**
- `period` (required): 期間 (1d, 5d, 1mo, 3mo, 6mo, 1y)
- `interval` (optional): 間隔 (1m, 5m, 15m, 30m, 1h, 1d) (default: 1d)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "7203",
    "period": "1mo",
    "interval": "1d",
    "data": [
      {
        "timestamp": "2025-07-17T00:00:00Z",
        "open": 1500.0,
        "high": 1600.0,
        "low": 1480.0,
        "close": 1550.0,
        "volume": 2000000
      }
    ]
  }
}
```

#### GET /market/watchlist
ウォッチリスト取得

**Response:**
```json
{
  "success": true,
  "data": {
    "symbols": [
      {
        "symbol": "7203",
        "name": "トヨタ自動車",
        "price": 1550.0,
        "change_percent": 3.33,
        "alert_enabled": true
      }
    ]
  }
}
```

## Configuration APIs

### 設定管理API

#### GET /config/profiles
設定プロファイル一覧取得

**Response:**
```json
{
  "success": true,
  "data": {
    "profiles": ["default", "aggressive", "conservative"],
    "active_profile": "default"
  }
}
```

#### PUT /config/profiles/{profile_name}
設定プロファイル変更

**Request:**
```json
{
  "profile_name": "aggressive"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "active_profile": "aggressive",
    "changed_at": "2025-08-17T10:25:00Z"
  }
}
```

#### GET /config/settings
設定値取得

**Query Parameters:**
- `keys` (optional): 取得する設定キー（カンマ区切り）

**Response:**
```json
{
  "success": true,
  "data": {
    "trading.max_positions": 5,
    "trading.risk_tolerance": 0.1,
    "notifications.email_enabled": true
  }
}
```

#### PUT /config/settings
設定値更新

**Request:**
```json
{
  "trading.max_positions": 10,
  "trading.risk_tolerance": 0.15
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "updated_keys": ["trading.max_positions", "trading.risk_tolerance"],
    "updated_at": "2025-08-17T10:30:00Z"
  }
}
```

## Monitoring APIs

### 監視・分析API

#### GET /monitoring/performance
パフォーマンス分析

**Query Parameters:**
- `component` (optional): 対象コンポーネント
- `hours` (optional): 分析期間（時間）(default: 1)

**Response:**
```json
{
  "success": true,
  "data": {
    "component": "trading_service",
    "average_execution_time_ms": 125.5,
    "max_memory_usage_mb": 512.8,
    "benchmark_results": {
      "execute_trade": {
        "benchmark_response_time_ms": 200.0,
        "actual_response_time_ms": 125.5,
        "meets_benchmark": true
      }
    },
    "suggestions": [
      "パフォーマンスは良好です"
    ]
  }
}
```

#### GET /monitoring/errors
エラー分析

**Query Parameters:**
- `hours` (optional): 分析期間（時間）(default: 24)

**Response:**
```json
{
  "success": true,
  "data": {
    "total_errors": 5,
    "error_rate_percent": 0.1,
    "errors_by_category": {
      "VALIDATION_ERROR": 3,
      "BUSINESS_LOGIC_ERROR": 2
    },
    "top_errors": [
      {
        "message": "無効な取引量です",
        "count": 3,
        "last_occurrence": "2025-08-17T10:35:00Z"
      }
    ]
  }
}
```

#### POST /monitoring/optimize
自動最適化実行（要管理者権限）

**Request:**
```json
{
  "component": "trading_service"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimizations_applied": ["memory_cleanup"],
    "memory_freed_mb": 25.6,
    "suggestions": [
      "キャッシュ効率が向上しました"
    ]
  }
}
```

## WebSocket APIs

### リアルタイムデータ配信

#### 接続
```
wss://api.daytrading.example.com/ws
```

#### 認証
```json
{
  "type": "auth",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 株価配信購読
```json
{
  "type": "subscribe",
  "channel": "quotes",
  "symbols": ["7203", "6758"]
}
```

#### 配信データ形式
```json
{
  "type": "quote",
  "symbol": "7203",
  "price": 1550.0,
  "change": 50.0,
  "timestamp": "2025-08-17T10:40:00Z"
}
```

## SDKサンプル

### Python SDK使用例

```python
from day_trade_sdk import DayTradeClient

# クライアント初期化
client = DayTradeClient(
    base_url="https://api.daytrading.example.com/v1",
    api_key="your_api_key"
)

# 認証
auth_result = client.auth.login("username", "password")
if auth_result.success:
    client.set_token(auth_result.data["access_token"])

# 取引実行
trade_result = client.trading.execute_trade(
    portfolio_id="portfolio_123",
    symbol="7203",
    direction="buy",
    quantity=100,
    order_type="market"
)

if trade_result.success:
    print(f"取引成功: {trade_result.data['trade_id']}")
else:
    print(f"取引失敗: {trade_result.error['message']}")
```

### JavaScript SDK使用例

```javascript
import { DayTradeClient } from '@daytrading/sdk';

// クライアント初期化
const client = new DayTradeClient({
  baseUrl: 'https://api.daytrading.example.com/v1',
  apiKey: 'your_api_key'
});

// 認証
const authResult = await client.auth.login('username', 'password');
if (authResult.success) {
  client.setToken(authResult.data.access_token);
}

// ポートフォリオ取得
const portfolioResult = await client.portfolios.get('portfolio_123');
if (portfolioResult.success) {
  console.log('ポートフォリオ:', portfolioResult.data);
}
```

## レート制限

| エンドポイント | 制限 | ウィンドウ |
|---------------|------|---------|
| 認証API | 10 req/min | ユーザー |
| 取引API | 100 req/min | ユーザー |
| 市場データAPI | 1000 req/min | ユーザー |
| 監視API | 60 req/min | ユーザー |

制限に達した場合、HTTP 429が返されます。

## バージョニング

APIはセマンティックバージョニングを使用します：

- メジャーバージョン: 破壊的変更
- マイナーバージョン: 下位互換性のある機能追加
- パッチバージョン: 下位互換性のあるバグ修正

現在のバージョン: `v1.0.0`

---

**ドキュメント版数**: v1.0  
**最終更新**: 2025-08-17  
**API版数**: v1.0.0