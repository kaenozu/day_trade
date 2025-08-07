"""
テスト用カスタムアサート関数

Phase 5: テストコード共通化の一環として、
20箇所での重複するアサート処理を統合し、テストの保守性を向上させます。

主な機能:
- DataFrame比較: 浮動小数点誤差を考慮した比較
- 取引データ検証: 取引の整合性チェック
- ポートフォリオ状態検証: ポートフォリオの妥当性チェック
- エラーハンドリング検証: 例外処理の検証
"""

import math
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pytest


def assert_dataframe_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    tolerance: float = 1e-6,
    check_dtype: bool = True,
    check_index: bool = True,
    check_columns: bool = True,
    check_names: bool = True,
    ignore_columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
) -> None:
    """
    DataFrameの詳細比較（浮動小数点誤差を考慮）

    Args:
        df1: 比較対象DataFrame 1
        df2: 比較対象DataFrame 2
        tolerance: 数値比較の許容誤差
        check_dtype: データ型をチェックするか
        check_index: インデックスをチェックするか
        check_columns: カラムをチェックするか
        check_names: 名前をチェックするか
        ignore_columns: 無視するカラム
        sort_by: ソートに使用するカラム

    Raises:
        AssertionError: DataFrameが等しくない場合
    """
    # None チェック
    if df1 is None and df2 is None:
        return
    if df1 is None or df2 is None:
        pytest.fail(
            f"One DataFrame is None: df1={df1 is not None}, df2={df2 is not None}"
        )

    # 型チェック
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        pytest.fail(
            f"Both objects must be DataFrames: df1={type(df1)}, df2={type(df2)}"
        )

    # コピーを作成（元データを変更しないため）
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # 無視するカラムを除外
    if ignore_columns:
        df1_copy = df1_copy.drop(columns=ignore_columns, errors="ignore")
        df2_copy = df2_copy.drop(columns=ignore_columns, errors="ignore")

    # ソート
    if sort_by and sort_by in df1_copy.columns and sort_by in df2_copy.columns:
        df1_copy = df1_copy.sort_values(sort_by).reset_index(drop=True)
        df2_copy = df2_copy.sort_values(sort_by).reset_index(drop=True)

    # 形状チェック
    if df1_copy.shape != df2_copy.shape:
        pytest.fail(f"DataFrame shapes differ: {df1_copy.shape} vs {df2_copy.shape}")

    # カラムチェック
    if check_columns and not df1_copy.columns.equals(df2_copy.columns):
        diff_cols = set(df1_copy.columns).symmetric_difference(set(df2_copy.columns))
        pytest.fail(f"DataFrame columns differ. Unique columns: {diff_cols}")

    # インデックスチェック
    if check_index and not df1_copy.index.equals(df2_copy.index):
        pytest.fail("DataFrame indices are not equal")

    # データチェック
    for col in df1_copy.columns:
        if col not in df2_copy.columns:
            continue

        series1 = df1_copy[col]
        series2 = df2_copy[col]

        # データ型チェック
        if (
            check_dtype and not check_names and series1.dtype.kind != series2.dtype.kind
        ):  # 名前チェック無効時はdtype比較を緩和
            pytest.fail(
                f"Column '{col}' dtypes differ: {series1.dtype} vs {series2.dtype}"
            )

        # 数値データの場合は許容誤差での比較
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(
            series2
        ):
            # NaNの処理
            nan_mask1 = pd.isna(series1)
            nan_mask2 = pd.isna(series2)

            if not nan_mask1.equals(nan_mask2):
                pytest.fail(f"Column '{col}' NaN positions differ")

            # 数値比較（NaN以外）
            non_nan_mask = ~(nan_mask1 | nan_mask2)
            if non_nan_mask.any():
                values1 = series1[non_nan_mask]
                values2 = series2[non_nan_mask]

                if not np.allclose(values1, values2, atol=tolerance, rtol=tolerance):
                    max_diff = np.max(np.abs(values1 - values2))
                    pytest.fail(
                        f"Column '{col}' values differ beyond tolerance {tolerance}. Max diff: {max_diff}"
                    )

        else:
            # 非数値データは厳密比較
            if not series1.equals(series2):
                diff_indices = series1 != series2
                if diff_indices.any():
                    first_diff_idx = diff_indices.idxmax()
                    pytest.fail(
                        f"Column '{col}' differs at index {first_diff_idx}: "
                        f"'{series1.iloc[first_diff_idx]}' vs '{series2.iloc[first_diff_idx]}'"
                    )


def assert_trade_equal(
    trade1: Dict[str, Any],
    trade2: Dict[str, Any],
    tolerance: float = 1e-6,
    ignore_fields: Optional[List[str]] = None,
) -> None:
    """
    取引データの詳細比較

    Args:
        trade1: 比較対象取引 1
        trade2: 比較対象取引 2
        tolerance: 数値比較の許容誤差
        ignore_fields: 無視するフィールド

    Raises:
        AssertionError: 取引データが等しくない場合
    """
    if ignore_fields is None:
        ignore_fields = ["timestamp", "trade_id"]  # デフォルトで無視するフィールド

    # 基本チェック
    if not isinstance(trade1, dict) or not isinstance(trade2, dict):
        pytest.fail(
            f"Both objects must be dictionaries: {type(trade1)}, {type(trade2)}"
        )

    # フィールドの比較
    fields1 = set(trade1.keys()) - set(ignore_fields)
    fields2 = set(trade2.keys()) - set(ignore_fields)

    if fields1 != fields2:
        diff_fields = fields1.symmetric_difference(fields2)
        pytest.fail(f"Trade fields differ. Unique fields: {diff_fields}")

    # 各フィールドの値比較
    for field in fields1:
        value1 = trade1[field]
        value2 = trade2[field]

        # 数値比較
        if isinstance(value1, (int, float, Decimal)) and isinstance(
            value2, (int, float, Decimal)
        ):
            if not math.isclose(
                float(value1), float(value2), abs_tol=tolerance, rel_tol=tolerance
            ):
                pytest.fail(f"Trade field '{field}' differs: {value1} vs {value2}")

        # 文字列比較
        elif isinstance(value1, str) and isinstance(value2, str):
            if value1 != value2:
                pytest.fail(f"Trade field '{field}' differs: '{value1}' vs '{value2}'")

        # その他のデータ型
        else:
            if value1 != value2:
                pytest.fail(f"Trade field '{field}' differs: {value1} vs {value2}")


def assert_portfolio_state(
    portfolio: Dict[str, Any],
    expected_cash: Optional[Union[float, Decimal]] = None,
    expected_positions: Optional[Dict[str, int]] = None,
    expected_total_value: Optional[Union[float, Decimal]] = None,
    tolerance: float = 1e-2,
    allow_partial_positions: bool = False,
) -> None:
    """
    ポートフォリオ状態の検証

    Args:
        portfolio: ポートフォリオデータ
        expected_cash: 期待される現金残高
        expected_positions: 期待されるポジション
        expected_total_value: 期待される総資産価値
        tolerance: 数値比較の許容誤差
        allow_partial_positions: 部分的なポジション指定を許可するか

    Raises:
        AssertionError: ポートフォリオ状態が期待と異なる場合
    """
    if not isinstance(portfolio, dict):
        pytest.fail(f"Portfolio must be a dictionary, got {type(portfolio)}")

    # 必須フィールドのチェック
    required_fields = ["cash_balance", "positions"]
    for field in required_fields:
        if field not in portfolio:
            pytest.fail(f"Portfolio missing required field: {field}")

    # 現金残高のチェック
    if expected_cash is not None:
        actual_cash = portfolio["cash_balance"]
        if isinstance(actual_cash, str):
            actual_cash = float(actual_cash)

        if not math.isclose(
            float(actual_cash), float(expected_cash), abs_tol=tolerance
        ):
            pytest.fail(f"Cash balance differs: {actual_cash} vs {expected_cash}")

    # ポジションのチェック
    if expected_positions is not None:
        actual_positions = portfolio["positions"]

        if not allow_partial_positions and set(actual_positions.keys()) != set(
            expected_positions.keys()
        ):
            # 完全一致チェック
            diff_symbols = set(actual_positions.keys()).symmetric_difference(
                set(expected_positions.keys())
            )
            pytest.fail(f"Position symbols differ. Unique symbols: {diff_symbols}")

        # 数量チェック
        for symbol, expected_qty in expected_positions.items():
            if symbol not in actual_positions:
                if not allow_partial_positions:
                    pytest.fail(
                        f"Position for symbol '{symbol}' not found in portfolio"
                    )
                continue

            actual_qty = actual_positions[symbol]
            if isinstance(actual_qty, dict):
                actual_qty = actual_qty.get("quantity", 0)

            if actual_qty != expected_qty:
                pytest.fail(
                    f"Position quantity for '{symbol}' differs: {actual_qty} vs {expected_qty}"
                )

    # 総資産価値のチェック
    if expected_total_value is not None:
        if "total_value" in portfolio:
            actual_total_value = portfolio["total_value"]
        else:
            # 計算して求める
            cash = float(portfolio["cash_balance"])
            positions_value = 0
            for pos_data in portfolio["positions"].values():
                if isinstance(pos_data, dict):
                    if "current_value" in pos_data:
                        positions_value += float(pos_data["current_value"])
                    elif "quantity" in pos_data and "current_price" in pos_data:
                        positions_value += float(pos_data["quantity"]) * float(
                            pos_data["current_price"]
                        )
            actual_total_value = cash + positions_value

        if not math.isclose(
            float(actual_total_value), float(expected_total_value), abs_tol=tolerance
        ):
            pytest.fail(
                f"Total portfolio value differs: {actual_total_value} vs {expected_total_value}"
            )


def assert_error_handling(
    func,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    expected_exception: type = Exception,
    expected_message: Optional[str] = None,
    message_contains: Optional[str] = None,
) -> None:
    """
    エラーハンドリングの検証

    Args:
        func: テスト対象関数
        args: 関数の引数
        kwargs: 関数のキーワード引数
        expected_exception: 期待される例外タイプ
        expected_message: 期待される例外メッセージ（完全一致）
        message_contains: 例外メッセージに含まれるべき文字列

    Raises:
        AssertionError: 期待される例外が発生しない場合
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    try:
        result = func(*args, **kwargs)
        pytest.fail(
            f"Expected {expected_exception.__name__} but function returned: {result}"
        )

    except expected_exception as e:
        # 例外メッセージのチェック
        if expected_message is not None and str(e) != expected_message:
            pytest.fail(
                f"Exception message differs: '{str(e)}' vs '{expected_message}'"
            )

        if message_contains is not None and message_contains not in str(e):
            pytest.fail(
                f"Exception message '{str(e)}' does not contain '{message_contains}'"
            )

    except Exception as e:
        pytest.fail(
            f"Expected {expected_exception.__name__} but got {type(e).__name__}: {str(e)}"
        )


def assert_numeric_close(
    actual: Union[float, int, Decimal],
    expected: Union[float, int, Decimal],
    tolerance: float = 1e-6,
    message: Optional[str] = None,
) -> None:
    """
    数値の近似比較

    Args:
        actual: 実際の値
        expected: 期待値
        tolerance: 許容誤差
        message: カスタムメッセージ

    Raises:
        AssertionError: 値が許容誤差内で等しくない場合
    """
    if not math.isclose(
        float(actual), float(expected), abs_tol=tolerance, rel_tol=tolerance
    ):
        diff = abs(float(actual) - float(expected))
        error_message = f"Values not close: {actual} vs {expected} (diff: {diff}, tolerance: {tolerance})"
        if message:
            error_message = f"{message}. {error_message}"
        pytest.fail(error_message)


def assert_list_contains_items(
    actual_list: List[Any],
    expected_items: List[Any],
    exact_match: bool = True,
    allow_extra_items: bool = False,
) -> None:
    """
    リスト内容の検証

    Args:
        actual_list: 実際のリスト
        expected_items: 期待される要素
        exact_match: 完全一致を要求するか
        allow_extra_items: 追加要素を許可するか

    Raises:
        AssertionError: リストの内容が期待と異なる場合
    """
    if not isinstance(actual_list, list):
        pytest.fail(f"Expected list, got {type(actual_list)}")

    if exact_match and not allow_extra_items:
        if len(actual_list) != len(expected_items):
            pytest.fail(
                f"List lengths differ: {len(actual_list)} vs {len(expected_items)}"
            )

        for i, (actual, expected) in enumerate(zip(actual_list, expected_items)):
            if actual != expected:
                pytest.fail(f"List item {i} differs: {actual} vs {expected}")

    else:
        # 期待される要素が全て含まれているかチェック
        missing_items = []
        for expected_item in expected_items:
            if expected_item not in actual_list:
                missing_items.append(expected_item)

        if missing_items:
            pytest.fail(f"Missing items in list: {missing_items}")

        if not allow_extra_items:
            extra_items = []
            for actual_item in actual_list:
                if actual_item not in expected_items:
                    extra_items.append(actual_item)

            if extra_items:
                pytest.fail(f"Extra items in list: {extra_items}")


def assert_datetime_close(
    actual: pd.Timestamp,
    expected: pd.Timestamp,
    tolerance_seconds: int = 60,
    message: Optional[str] = None,
) -> None:
    """
    日時の近似比較

    Args:
        actual: 実際の日時
        expected: 期待される日時
        tolerance_seconds: 許容誤差（秒）
        message: カスタムメッセージ

    Raises:
        AssertionError: 日時が許容誤差内で等しくない場合
    """
    if not isinstance(actual, pd.Timestamp):
        actual = pd.Timestamp(actual)
    if not isinstance(expected, pd.Timestamp):
        expected = pd.Timestamp(expected)

    diff_seconds = abs((actual - expected).total_seconds())

    if diff_seconds > tolerance_seconds:
        error_message = f"Timestamps not close: {actual} vs {expected} (diff: {diff_seconds}s, tolerance: {tolerance_seconds}s)"
        if message:
            error_message = f"{message}. {error_message}"
        pytest.fail(error_message)
