"""
外部APIクライアント列挙型定義
API プロバイダー、データ型、リクエストメソッドの定義
"""

from enum import Enum


class APIProvider(Enum):
    """APIプロバイダー"""

    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    IEX_CLOUD = "iex_cloud"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    MOCK_PROVIDER = "mock_provider"  # 開発・テスト用


class DataType(Enum):
    """データタイプ"""

    STOCK_PRICE = "stock_price"
    MARKET_INDEX = "market_index"
    COMPANY_INFO = "company_info"
    FINANCIAL_STATEMENT = "financial_statement"
    NEWS = "news"
    ECONOMIC_INDICATOR = "economic_indicator"
    EXCHANGE_RATE = "exchange_rate"
    SECTOR_PERFORMANCE = "sector_performance"


class RequestMethod(Enum):
    """HTTPリクエストメソッド"""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"