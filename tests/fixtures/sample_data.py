"""
テスト用サンプルデータ生成関数

Phase 5: テストコード共通化の一環として、
15箇所での重複するサンプルデータ生成を統合し、テストの保守性を向上させます。

主な機能:
- 株価データ生成: 再現可能で現実的なOHLCVデータ
- 取引データ生成: 様々なシナリオの取引履歴
- マーケットデータ生成: 市場全体の統計データ
- ポートフォリオデータ生成: 複数銘柄のポートフォリオデータ
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def create_sample_stock_data(
    symbol: str = "7203",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    periods: int = 100,
    base_price: float = 1000.0,
    volatility: float = 0.02,
    trend: float = 0.0,
    include_gaps: bool = False,
    include_splits: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    現実的な株価データを生成

    Args:
        symbol: 銘柄コード
        start_date: 開始日
        end_date: 終了日
        periods: データ期間（日数）
        base_price: 基準価格
        volatility: ボラティリティ（日次）
        trend: トレンド（年率）
        include_gaps: ギャップを含むか
        include_splits: 株式分割を含むか
        seed: 乱数シード

    Returns:
        OHLCV形式のDataFrame
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        # 銘柄ごとに一意のシードを生成
        np.random.seed(hash(symbol) % 2**32)

    # 日付範囲の設定
    if end_date is None:
        end_date = datetime.now().date()
    if start_date is None:
        start_date = end_date - timedelta(days=periods)

    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")[:periods]

    # 価格生成（幾何ブラウン運動）
    dt = 1 / 252  # 営業日ベース
    drift = trend * dt
    diffusion = volatility * np.sqrt(dt)

    # リターン生成
    returns = np.random.normal(drift, diffusion, len(dates))

    # トレンド成分を追加
    if trend != 0:
        trend_component = np.linspace(0, trend * len(dates) / 252, len(dates))
        returns += trend_component / len(dates)

    # 価格パスの生成
    log_prices = np.log(base_price) + np.cumsum(returns)
    close_prices = np.exp(log_prices)

    # OHLC の生成
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    # 日中のボラティリティを考慮したHigh/Low
    intraday_volatility = volatility * 0.5
    daily_ranges = np.abs(np.random.normal(0, intraday_volatility, len(dates)))

    high_prices = np.maximum(open_prices, close_prices) + daily_ranges * close_prices
    low_prices = np.minimum(open_prices, close_prices) - daily_ranges * close_prices

    # ボリューム生成（価格変動と相関）
    volume_base = 1000000
    volume_volatility = 0.5
    price_changes = np.abs(np.diff(np.log(close_prices)))
    price_changes = np.concatenate([[0], price_changes])

    # 価格変動が大きいほどボリューム増加
    volume_multiplier = 1 + price_changes * 10
    volume = (
        volume_base
        * volume_multiplier
        * (1 + np.random.normal(0, volume_volatility, len(dates)))
    ).astype(int)
    volume = np.maximum(volume, 10000)  # 最小ボリューム

    # ギャップの追加
    if include_gaps:
        gap_indices = np.random.choice(
            len(dates), size=max(1, len(dates) // 20), replace=False
        )
        for idx in gap_indices:
            if idx > 0:
                gap_size = np.random.normal(0, volatility * 2)
                gap_multiplier = 1 + gap_size
                open_prices[idx] = close_prices[idx - 1] * gap_multiplier
                close_prices[idx] = open_prices[idx] * (1 + returns[idx])
                high_prices[idx] = (
                    max(open_prices[idx], close_prices[idx])
                    + daily_ranges[idx] * close_prices[idx]
                )
                low_prices[idx] = (
                    min(open_prices[idx], close_prices[idx])
                    - daily_ranges[idx] * close_prices[idx]
                )

    # 株式分割の追加
    if include_splits:
        split_indices = np.random.choice(
            len(dates), size=max(1, len(dates) // 100), replace=False
        )
        for idx in split_indices:
            if idx > 10:  # 最初の10日は除外
                split_ratio = np.random.choice([2, 3, 1.5])  # 2:1, 3:1, 3:2分割

                # 分割後は価格が調整される
                open_prices[idx:] /= split_ratio
                high_prices[idx:] /= split_ratio
                low_prices[idx:] /= split_ratio
                close_prices[idx:] /= split_ratio
                volume[idx:] = (volume[idx:] * split_ratio).astype(int)

    # データフレームの作成
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_prices, 2),
            "High": np.round(high_prices, 2),
            "Low": np.round(low_prices, 2),
            "Close": np.round(close_prices, 2),
            "Volume": volume,
        }
    )

    # データの整合性チェック
    df["High"] = np.maximum.reduce([df["High"], df["Open"], df["Close"]])
    df["Low"] = np.minimum.reduce([df["Low"], df["Open"], df["Close"]])

    df.set_index("Date", inplace=True)
    return df


def create_sample_trade_data(
    symbols: List[str] = None,
    num_trades: int = 50,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    initial_cash: Decimal = Decimal("1000000"),
    include_partial_fills: bool = False,
    include_cancelled_orders: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    取引履歴データを生成

    Args:
        symbols: 対象銘柄リスト
        num_trades: 取引回数
        start_date: 開始日
        end_date: 終了日
        initial_cash: 初期資金
        include_partial_fills: 部分約定を含むか
        include_cancelled_orders: キャンセル注文を含むか
        seed: 乱数シード

    Returns:
        取引履歴DataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    if symbols is None:
        symbols = ["7203", "6758", "9984", "8306", "4503"]

    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # 取引日時の生成
    trade_dates = pd.date_range(start=start_date, end=end_date, freq="B")

    trades = []
    cash_balance = initial_cash
    positions = {}

    for i in range(num_trades):
        trade_date = np.random.choice(trade_dates)
        symbol = np.random.choice(symbols)

        # 銘柄ごとの基準価格（株価データと整合性を保つ）
        base_prices = {
            "7203": 2000,
            "6758": 3000,
            "9984": 8000,
            "8306": 4000,
            "4503": 5000,
        }
        base_price = base_prices.get(symbol, 1000)

        # 価格に変動を追加
        price_variation = np.random.normal(0, 0.02)
        price = base_price * (1 + price_variation)

        # 取引タイプの決定
        if symbol in positions and positions[symbol] > 0 and np.random.random() > 0.6:
            # 売り注文（ポジションがある場合）
            side = "SELL"
            max_quantity = min(positions[symbol], 1000)
            quantity = (
                np.random.randint(100, max_quantity + 1, dtype=int)
                if max_quantity >= 100
                else positions[symbol]
            )
        else:
            # 買い注文
            side = "BUY"
            max_affordable = min(int(cash_balance / price), 1000)
            if max_affordable < 100:
                continue  # 資金不足の場合はスキップ
            quantity = np.random.randint(100, max_affordable + 1, dtype=int)

        # 注文タイプ
        order_type = np.random.choice(["MARKET", "LIMIT"], p=[0.7, 0.3])

        # 約定ステータス
        if include_cancelled_orders and np.random.random() < 0.1:
            status = "CANCELLED"
            executed_quantity = 0
        elif include_partial_fills and np.random.random() < 0.2:
            status = "PARTIALLY_FILLED"
            executed_quantity = int(quantity * np.random.uniform(0.3, 0.9))
        else:
            status = "FILLED"
            executed_quantity = quantity

        # 約定価格（指値注文の場合は若干の差を追加）
        if order_type == "LIMIT":
            if side == "BUY":
                executed_price = price * np.random.uniform(0.995, 1.000)
            else:
                executed_price = price * np.random.uniform(1.000, 1.005)
        else:
            executed_price = price * np.random.uniform(0.999, 1.001)

        # 手数料計算
        commission_rate = 0.001  # 0.1%
        commission = executed_quantity * executed_price * commission_rate

        # ポジションと資金の更新
        if status in ["FILLED", "PARTIALLY_FILLED"] and executed_quantity > 0:
            if side == "BUY":
                cash_balance -= executed_quantity * executed_price + commission
                positions[symbol] = positions.get(symbol, 0) + executed_quantity
            else:
                cash_balance += executed_quantity * executed_price - commission
                positions[symbol] = positions.get(symbol, 0) - executed_quantity

        trade = {
            "trade_id": f"TRADE_{i+1:04d}",
            "timestamp": trade_date,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "executed_quantity": executed_quantity,
            "price": round(price, 2),
            "executed_price": round(executed_price, 2),
            "commission": round(commission, 2),
            "status": status,
            "cash_balance": float(cash_balance),
        }

        trades.append(trade)

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def create_sample_market_data(
    num_symbols: int = 50,
    date: Optional[datetime] = None,
    include_indices: bool = True,
    include_sectors: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    市場全体のスナップショットデータを生成

    Args:
        num_symbols: 銘柄数
        date: データ日付
        include_indices: 指数を含むか
        include_sectors: セクター情報を含むか
        seed: 乱数シード

    Returns:
        市場データDataFrame
    """
    if seed is not None:
        np.random.seed(seed)

    if date is None:
        date = datetime.now().date()

    # 実際の銘柄コードっぽいものを生成
    symbols = []
    for i in range(num_symbols):
        if i < 10:
            # 主要銘柄
            major_symbols = [
                "7203",
                "6758",
                "9984",
                "8306",
                "4503",
                "6861",
                "8035",
                "4568",
                "6502",
                "7974",
            ]
            symbols.append(major_symbols[i])
        else:
            # ランダムな4桁コード
            symbols.append(f"{np.random.randint(1000, 9999)}")

    # 基本的な市場データ
    market_data = []

    # セクター定義
    sectors = (
        [
            "自動車",
            "テクノロジー",
            "金融",
            "製薬",
            "小売",
            "エネルギー",
            "不動産",
            "食品",
            "化学",
            "機械",
        ]
        if include_sectors
        else ["一般"]
    )

    for symbol in symbols:
        # 基準価格の設定
        price_ranges = {
            "7203": (1800, 2200),  # トヨタ
            "6758": (2800, 3200),  # ソニー
            "9984": (7000, 9000),  # ソフトバンクG
            "8306": (3500, 4500),  # 三菱UFJ
            "4503": (4800, 5200),  # アステラス
        }

        if symbol in price_ranges:
            price = np.random.uniform(*price_ranges[symbol])
        else:
            price = np.random.lognormal(np.log(1000), 0.5)

        # 日次変動
        daily_change_pct = np.random.normal(0, 0.02)
        daily_change = price * daily_change_pct

        # ボリューム
        volume = int(np.random.lognormal(13, 1))  # 対数正規分布

        # 時価総額（適当な発行済み株式数で計算）
        shares_outstanding = np.random.randint(100000, 10000000) * 1000
        market_cap = price * shares_outstanding

        data = {
            "symbol": symbol,
            "date": date,
            "current_price": round(price, 2),
            "daily_change": round(daily_change, 2),
            "daily_change_pct": round(daily_change_pct * 100, 2),
            "volume": volume,
            "market_cap": int(market_cap),
            "shares_outstanding": shares_outstanding,
        }

        if include_sectors:
            data["sector"] = np.random.choice(sectors)

        market_data.append(data)

    df = pd.DataFrame(market_data)

    # 指数データの追加
    if include_indices:
        # 日経平均の模擬データ
        nikkei_value = np.random.normal(28000, 1000)
        nikkei_change = np.random.normal(0, 200)

        # TOPIX
        topix_value = np.random.normal(2000, 100)
        topix_change = np.random.normal(0, 20)

        indices_data = [
            {
                "symbol": "N225",
                "date": date,
                "current_price": round(nikkei_value, 2),
                "daily_change": round(nikkei_change, 2),
                "daily_change_pct": round(nikkei_change / nikkei_value * 100, 2),
                "volume": 0,
                "market_cap": 0,
                "shares_outstanding": 0,
                "sector": "指数" if include_sectors else "一般",
            },
            {
                "symbol": "TOPIX",
                "date": date,
                "current_price": round(topix_value, 2),
                "daily_change": round(topix_change, 2),
                "daily_change_pct": round(topix_change / topix_value * 100, 2),
                "volume": 0,
                "market_cap": 0,
                "shares_outstanding": 0,
                "sector": "指数" if include_sectors else "一般",
            },
        ]

        df = pd.concat([df, pd.DataFrame(indices_data)], ignore_index=True)

    return df


def create_sample_portfolio_data(
    symbols: List[str] = None,
    initial_cash: Decimal = Decimal("1000000"),
    num_positions: int = 5,
    include_history: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    ポートフォリオデータを生成

    Args:
        symbols: 保有銘柄リスト
        initial_cash: 初期資金
        num_positions: ポジション数
        include_history: 履歴を含むか
        seed: 乱数シード

    Returns:
        ポートフォリオデータの辞書
    """
    if seed is not None:
        np.random.seed(seed)

    if symbols is None:
        symbols = ["7203", "6758", "9984", "8306", "4503", "6861", "8035", "4568"]

    # ポジションの生成
    selected_symbols = np.random.choice(
        symbols, size=min(num_positions, len(symbols)), replace=False
    )

    positions = []
    total_invested = Decimal("0")

    for symbol in selected_symbols:
        # 基準価格
        base_prices = {
            "7203": 2000,
            "6758": 3000,
            "9984": 8000,
            "8306": 4000,
            "4503": 5000,
        }
        avg_price = base_prices.get(symbol, np.random.uniform(500, 5000))

        # 現在価格（±10%の変動）
        current_price = avg_price * np.random.uniform(0.9, 1.1)

        # 保有数量
        quantity = np.random.randint(100, 1000)

        # 投資額と時価
        invested_amount = Decimal(str(avg_price)) * Decimal(str(quantity))
        current_value = Decimal(str(current_price)) * Decimal(str(quantity))

        # 損益
        unrealized_pnl = current_value - invested_amount
        unrealized_pnl_pct = float(unrealized_pnl / invested_amount * 100)

        position = {
            "symbol": symbol,
            "quantity": quantity,
            "avg_price": round(avg_price, 2),
            "current_price": round(current_price, 2),
            "invested_amount": float(invested_amount),
            "current_value": float(current_value),
            "unrealized_pnl": float(unrealized_pnl),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        }

        positions.append(position)
        total_invested += invested_amount

    # 現金残高
    cash_balance = initial_cash - total_invested

    # 総資産価値
    total_portfolio_value = cash_balance + sum(
        Decimal(str(p["current_value"])) for p in positions
    )

    # 全体の損益
    total_unrealized_pnl = sum(Decimal(str(p["unrealized_pnl"])) for p in positions)
    total_return_pct = (
        float(total_unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
    )

    portfolio = {
        "cash_balance": float(cash_balance),
        "total_invested": float(total_invested),
        "total_portfolio_value": float(total_portfolio_value),
        "total_unrealized_pnl": float(total_unrealized_pnl),
        "total_return_pct": round(total_return_pct, 2),
        "positions": positions,
        "last_updated": datetime.now(),
    }

    # 履歴データの生成
    if include_history:
        history_dates = pd.date_range(end=datetime.now().date(), periods=30, freq="D")

        portfolio_history = []
        for date in history_dates:
            # 過去のポートフォリオ価値（ランダムウォーク）
            daily_return = np.random.normal(0, 0.01)
            portfolio_value = float(total_portfolio_value) * (
                1 + daily_return * (30 - len(portfolio_history)) / 30
            )

            portfolio_history.append(
                {
                    "date": date.date(),
                    "portfolio_value": round(portfolio_value, 2),
                    "cash_balance": float(cash_balance),
                    "daily_return_pct": round(daily_return * 100, 2),
                }
            )

        portfolio["history"] = portfolio_history

    return portfolio
