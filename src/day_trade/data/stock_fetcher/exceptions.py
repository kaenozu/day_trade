"""
株価データ取得用カスタム例外クラス
"""

from ...utils.exceptions import (
    APIError,
    DataError,
    ValidationError,
)


class StockFetcherError(APIError):
    """
    株価データ取得エラー（下位互換性のため）

    株価データ取得に関する一般的なエラーを表現します。
    他のより具体的な例外が適用できない場合に使用されます。

    Examples:
        >>> raise StockFetcherError("データ取得に失敗しました")
        >>> raise StockFetcherError(
        ...     message="API制限に達しました",
        ...     error_code="API_LIMIT_EXCEEDED",
        ...     details={"retry_after": 3600}
        ... )
    """

    pass


class InvalidSymbolError(ValidationError):
    """
    無効なシンボルエラー（下位互換性のため）

    提供されたシンボル（銘柄コード）が無効な形式であったり、
    存在しない場合に発生するエラーです。

    Examples:
        >>> raise InvalidSymbolError("無効なシンボル: XYZ123")
        >>> raise InvalidSymbolError(
        ...     message="シンボルが見つかりません",
        ...     error_code="SYMBOL_NOT_FOUND",
        ...     details={"provided_symbol": "INVALID"}
        ... )
    """

    pass


class DataNotFoundError(DataError):
    """
    データが見つからないエラー（下位互換性のため）

    要求されたデータが見つからない、または利用できない場合に
    発生するエラーです。

    Examples:
        >>> raise DataNotFoundError("指定期間のデータが見つかりません")
        >>> raise DataNotFoundError(
        ...     message="ヒストリカルデータなし",
        ...     error_code="NO_HISTORICAL_DATA",
        ...     details={
        ...         "symbol": "7203",
        ...         "period": "1y",
        ...         "requested_date": "2024-01-01"
        ...     }
        ... )
    """

    pass