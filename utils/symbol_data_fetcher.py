import logging
from typing import Optional

# Assuming these are available in the main environment or will be passed
# from src.day_trade.utils.yfinance_import import get_yfinance
# from src.day_trade.data.symbol_names import get_symbol_name

logger = logging.getLogger('daytrade')

_company_name_cache = {}

async def get_company_name_from_yfinance(symbol: str) -> Optional[str]:
    """
    銘柄辞書またはyfinanceから会社名を取得（キャッシュ付き）
    """
    # キャッシュチェック
    if symbol in _company_name_cache:
        return _company_name_cache[symbol]

    # まず銘柄辞書から取得を試行
    try:
        from src.day_trade.data.symbol_names import get_symbol_name
        symbol_name = get_symbol_name(symbol)
        logger.debug(f"get_company_name_from_yfinance: {symbol} -> get_symbol_name returned: {repr(symbol_name)}")
        if symbol_name:
            _company_name_cache[symbol] = symbol_name
            return symbol_name
    except ImportError:
        logger.debug("src.day_trade.data.symbol_names.get_symbol_name not available.")
    except Exception as e:
        logger.warning(f"Failed to get company name for {symbol} from symbol_names: {e}")

    # yfinanceが利用可能かチェック
    try:
        from src.day_trade.utils.yfinance_import import get_yfinance, is_yfinance_available
        if not is_yfinance_available():
            logger.warning("yfinance is not available. Cannot fetch company name.")
            return None

        yf, _ = get_yfinance()
        if not yf:
            return None

        # 日本株の場合は.Tを付加
        symbol_yf = symbol
        if symbol.isdigit() and len(symbol) == 4:
            symbol_yf = f"{symbol}.T"

        ticker = yf.Ticker(symbol_yf)
        info = ticker.info
        name = info.get('longName') or info.get('shortName')
        if name:
            _company_name_cache[symbol] = name
        return name
    except Exception as e:
        logger.warning(f"Failed to get company name for {symbol} from yfinance: {e}")
        return None
