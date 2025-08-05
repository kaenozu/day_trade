"""
表示フォーマット用ユーティリティ
高度なCLI表示機能とASCIIチャート描画
"""

from typing import Any, Dict, List, Union

import pandas as pd
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def format_currency(
    amount: Union[int, float], currency: str = "¥", decimal_places: int = 0
) -> str:
    """
    通貨フォーマット

    Args:
        amount: 金額
        currency: 通貨記号
        decimal_places: 小数点以下の桁数

    Returns:
        フォーマット済み金額文字列
    """
    if amount is None:
        return "N/A"

    if decimal_places == 0:
        return f"{currency}{amount:,.0f}"
    else:
        return f"{currency}{amount:,.{decimal_places}f}"


def format_percentage(
    value: Union[int, float], decimal_places: int = 2, show_sign: bool = True
) -> str:
    """
    パーセンテージフォーマット

    Args:
        value: 値
        decimal_places: 小数点以下の桁数
        show_sign: 符号を表示するか

    Returns:
        フォーマット済みパーセンテージ文字列
    """
    if value is None:
        return "N/A"

    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value:.{decimal_places}f}%"


def format_volume(volume: Union[int, float]) -> str:
    """
    出来高フォーマット

    Args:
        volume: 出来高

    Returns:
        フォーマット済み出来高文字列
    """
    if volume is None:
        return "N/A"

    volume = int(volume)
    if volume >= 1_000_000_000:
        return f"{volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"{volume / 1_000:.1f}K"
    else:
        return f"{volume:,}"


def get_change_color(value: Union[int, float]) -> str:
    """
    変化値に基づく色を取得

    Args:
        value: 変化値

    Returns:
        色名
    """
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "white"


def create_stock_info_table(stock_data: dict) -> Table:
    """
    株価情報テーブルを作成

    Args:
        stock_data: 株価データ

    Returns:
        Richテーブル
    """
    table = Table(title=f"銘柄情報: {stock_data.get('symbol', 'N/A')}")
    table.add_column("項目", style="cyan", no_wrap=True)
    table.add_column("値", style="white")

    # 現在価格
    current_price = stock_data.get("current_price")
    if current_price:
        table.add_row("現在値", format_currency(current_price))

    # 前日終値
    previous_close = stock_data.get("previous_close")
    if previous_close:
        table.add_row("前日終値", format_currency(previous_close))

    # 前日比
    change = stock_data.get("change")
    change_percent = stock_data.get("change_percent")
    if change is not None and change_percent is not None:
        change_color = get_change_color(change)
        change_text = f"[{change_color}]{format_currency(change, currency='¥', decimal_places=0)} ({format_percentage(change_percent)})[/{change_color}]"
        table.add_row("前日比", change_text)

    # 出来高
    volume = stock_data.get("volume")
    if volume:
        table.add_row("出来高", format_volume(volume))

    return table


def create_company_info_table(company_data: dict) -> Table:
    """
    企業情報テーブルを作成

    Args:
        company_data: 企業データ

    Returns:
        Richテーブル
    """
    table = Table(title="企業情報")
    table.add_column("項目", style="cyan", no_wrap=True)
    table.add_column("値", style="white")

    # 企業名
    name = company_data.get("name")
    if name:
        table.add_row("企業名", name)

    # セクター
    sector = company_data.get("sector")
    if sector:
        table.add_row("セクター", sector)

    # 業種
    industry = company_data.get("industry")
    if industry:
        table.add_row("業種", industry)

    # 時価総額
    market_cap = company_data.get("market_cap")
    if market_cap:
        table.add_row("時価総額", format_currency(market_cap))

    # 従業員数
    employees = company_data.get("employees")
    if employees:
        table.add_row("従業員数", f"{employees:,}人")

    return table


def create_historical_data_table(
    df: pd.DataFrame, symbol: str, period: str, interval: str, max_rows: int = 10
) -> Table:
    """
    ヒストリカルデータテーブルを作成

    Args:
        df: ヒストリカルデータ
        symbol: 銘柄コード
        period: 期間
        interval: 間隔
        max_rows: 最大表示行数

    Returns:
        Richテーブル
    """
    table = Table(title=f"{symbol} - 過去 {period} ({interval})")
    table.add_column("日時", style="cyan")
    table.add_column("始値", justify="right")
    table.add_column("高値", justify="right")
    table.add_column("安値", justify="right")
    table.add_column("終値", justify="right")
    table.add_column("出来高", justify="right")

    # 最新のmax_rows件を表示
    display_df = df.tail(max_rows)

    for idx, row in display_df.iterrows():
        # 前日比で色分け
        color = "white"
        if len(df) > 1 and idx > df.index[0]:
            try:
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                prev_close = df.loc[prev_idx, "Close"]
                color = get_change_color(row["Close"] - prev_close)
            except (KeyError, IndexError):
                pass

        table.add_row(
            str(idx.strftime("%Y-%m-%d %H:%M")),
            format_currency(row["Open"]),
            format_currency(row["High"]),
            format_currency(row["Low"]),
            f"[{color}]{format_currency(row['Close'])}[/{color}]",
            format_volume(row["Volume"]),
        )

    return table


def create_watchlist_table(watchlist_data: dict) -> Table:
    """
    ウォッチリストテーブルを作成

    Args:
        watchlist_data: ウォッチリストデータ

    Returns:
        Richテーブル
    """
    table = Table(title="ウォッチリスト")
    table.add_column("コード", style="cyan")
    table.add_column("現在値", justify="right")
    table.add_column("前日比", justify="right")
    table.add_column("前日比率", justify="right")
    table.add_column("出来高", justify="right")

    for code, data in watchlist_data.items():
        if data:
            change_color = get_change_color(data.get("change", 0))
            table.add_row(
                code,
                format_currency(data.get("current_price")),
                f"[{change_color}]{format_currency(data.get('change', 0))}[/{change_color}]",
                f"[{change_color}]{format_percentage(data.get('change_percent', 0))}[/{change_color}]",
                format_volume(data.get("volume")),
            )

    return table


def create_error_panel(
    message: str, title: str = "エラー", solutions: list = None, show_emoji: bool = True
) -> Panel:
    """
    ユーザーフレンドリーなエラーパネルを作成

    Args:
        message: エラーメッセージ
        title: パネルタイトル
        solutions: 解決策のリスト
        show_emoji: 絵文字を表示するか

    Returns:
        Richパネル
    """
    # 基本的なエラーメッセージを作成
    content_lines = []

    # 絵文字付きメッセージ
    emoji = "❌ " if show_emoji else ""
    content_lines.append(f"[bold red]{emoji}{message}[/bold red]")

    # 解決策がある場合は追加
    if solutions:
        content_lines.extend(["", "[bold yellow]💡 解決方法:[/bold yellow]"])

        for i, solution in enumerate(solutions, 1):
            content_lines.append(f"  {i}. {solution}")

    content = "\n".join(content_lines)

    return Panel(
        content,
        title=f"[bold red]{title}[/bold red]",
        border_style="red",
        padding=(1, 2),
    )


def create_success_panel(message: str, title: str = "成功") -> Panel:
    """
    成功パネルを作成

    Args:
        message: 成功メッセージ
        title: パネルタイトル

    Returns:
        Richパネル
    """
    return Panel(Text(message, style="green"), title=title, border_style="green")


def create_warning_panel(message: str, title: str = "警告") -> Panel:
    """
    警告パネルを作成

    Args:
        message: 警告メッセージ
        title: パネルタイトル

    Returns:
        Richパネル
    """
    return Panel(Text(message, style="yellow"), title=title, border_style="yellow")


def create_info_panel(message: str, title: str = "情報") -> Panel:
    """
    情報パネルを作成

    Args:
        message: 情報メッセージ
        title: パネルタイトル

    Returns:
        Richパネル
    """
    return Panel(Text(message, style="blue"), title=title, border_style="blue")


# ========== 高度なフォーマッタ機能 ==========


def format_large_number(number: Union[int, float], precision: int = 1) -> str:
    """
    大きな数値を適切にフォーマット

    Args:
        number: 数値
        precision: 小数点以下桁数

    Returns:
        フォーマット済み数値文字列
    """
    if number is None:
        return "N/A"

    abs_number = abs(number)
    sign = "-" if number < 0 else ""

    if abs_number >= 1_000_000_000_000:  # 1兆以上
        return f"{sign}{abs_number / 1_000_000_000_000:.{precision}f}T"
    elif abs_number >= 1_000_000_000:  # 10億以上
        return f"{sign}{abs_number / 1_000_000_000:.{precision}f}B"
    elif abs_number >= 1_000_000:  # 100万以上
        return f"{sign}{abs_number / 1_000_000:.{precision}f}M"
    elif abs_number >= 1_000:  # 1000以上
        return f"{sign}{abs_number / 1_000:.{precision}f}K"
    else:
        return f"{sign}{abs_number:.{precision}f}"


def create_ascii_chart(
    data: List[float], width: int = 60, height: int = 10, title: str = "Chart"
) -> str:
    """
    ASCIIチャートを作成

    Args:
        data: チャートデータ
        width: チャート幅
        height: チャート高さ
        title: チャートタイトル

    Returns:
        ASCIIチャート文字列
    """
    if not data or len(data) < 2:
        return f"{title}\n[No data to display]"

    # データの正規化
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return f"{title}\n[Data has no variation]"

    # チャート作成
    chart_lines = []

    # タイトル
    chart_lines.append(f"{title}")
    chart_lines.append("=" * width)

    # Y軸ラベル用のスペースを確保
    y_label_width = 8
    chart_width = width - y_label_width - 1

    # データを幅に合わせてサンプリング
    if len(data) > chart_width:
        step = len(data) / chart_width
        sampled_data = [data[int(i * step)] for i in range(chart_width)]
    else:
        sampled_data = data + [data[-1]] * (chart_width - len(data))

    # 各行を作成
    for row in range(height):
        # Y軸の値（上から下へ）
        y_value = max_val - (row / (height - 1)) * (max_val - min_val)
        y_label = f"{y_value:8.2f}"

        # チャート部分
        line = ""
        for _col, value in enumerate(sampled_data):
            if row == 0:  # 最上行
                if value >= y_value:
                    line += "█"
                else:
                    line += " "
            elif row == height - 1:  # 最下行
                if value <= y_value:
                    line += "█"
                else:
                    line += " "
            else:  # 中間行
                prev_y = max_val - ((row - 1) / (height - 1)) * (max_val - min_val)
                next_y = max_val - ((row + 1) / (height - 1)) * (max_val - min_val)

                if prev_y >= value >= next_y:
                    line += "█"
                elif value > prev_y:
                    line += "▀"
                elif value < next_y:
                    line += "▄"
                else:
                    line += " "

        chart_lines.append(f"{y_label}│{line}")

    # X軸
    x_axis = " " * y_label_width + "└" + "─" * chart_width
    chart_lines.append(x_axis)

    return "\n".join(chart_lines)


def create_sparkline(data: List[float], width: int = 20) -> str:
    """
    スパークライン（小さなチャート）を作成

    Args:
        data: データ
        width: 幅

    Returns:
        スパークライン文字列
    """
    if not data:
        return "No data"

    if len(data) == 1:
        return "▄" * width

    # データの正規化
    min_val = min(data)
    max_val = max(data)

    if max_val == min_val:
        return "▄" * width

    # スパークライン文字（下から上へ）
    spark_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    # データを幅に合わせてサンプリング
    if len(data) > width:
        step = len(data) / width
        sampled_data = [data[int(i * step)] for i in range(width)]
    else:
        sampled_data = data[:width]

    # 各値を文字にマッピング
    sparkline = ""
    for value in sampled_data:
        normalized = (value - min_val) / (max_val - min_val)
        char_index = min(int(normalized * len(spark_chars)), len(spark_chars) - 1)
        sparkline += spark_chars[char_index]

    return sparkline


def create_progress_bar_panel(
    current: int, total: int, title: str = "Progress", show_percentage: bool = True
) -> Panel:
    """
    プログレスバーパネルを作成

    Args:
        current: 現在値
        total: 総数
        title: タイトル
        show_percentage: パーセンテージを表示するか

    Returns:
        プログレスバーパネル
    """
    percentage = 0 if total == 0 else current / total * 100

    # プログレスバーの作成
    bar_width = 30
    filled_width = int((current / total) * bar_width) if total > 0 else 0
    bar = "█" * filled_width + "░" * (bar_width - filled_width)

    # テキスト作成
    if show_percentage:
        text = f"{bar} {current}/{total} ({percentage:.1f}%)"
    else:
        text = f"{bar} {current}/{total}"

    return Panel(Text(text, style="cyan"), title=title, border_style="blue")


def create_comparison_table(
    data: Dict[str, Dict[str, Any]], title: str = "比較表"
) -> Table:
    """
    比較テーブルを作成

    Args:
        data: 比較データ（キー: 項目名, 値: データ辞書）
        title: テーブルタイトル

    Returns:
        比較テーブル
    """
    table = Table(title=title, box=box.ROUNDED)

    if not data:
        table.add_column("No Data", style="dim")
        table.add_row("データがありません")
        return table

    # カラムを追加
    table.add_column("項目", style="cyan", no_wrap=True)
    for key in data:
        table.add_column(key, justify="right")

    # 共通キーを取得
    all_keys = set()
    for item_data in data.values():
        all_keys.update(item_data.keys())

    # 行を追加
    for metric in sorted(all_keys):
        row = [metric]
        for _item_key, item_data in data.items():
            value = item_data.get(metric, "N/A")

            # 数値の場合は適切にフォーマット
            if isinstance(value, (int, float)) and value != "N/A":
                if abs(value) >= 1000:
                    formatted_value = format_large_number(value)
                elif metric.lower().endswith(("rate", "ratio", "percent")):
                    formatted_value = format_percentage(value)
                elif metric.lower().endswith(("price", "cost", "value")):
                    formatted_value = format_currency(value)
                else:
                    formatted_value = f"{value:,.2f}"
            else:
                formatted_value = str(value)

            row.append(formatted_value)

        table.add_row(*row)

    return table


def create_heatmap(
    data: List[List[float]],
    labels_x: List[str],
    labels_y: List[str],
    title: str = "Heatmap",
) -> str:
    """
    ASCIIヒートマップを作成

    Args:
        data: 2次元データ配列
        labels_x: X軸ラベル
        labels_y: Y軸ラベル
        title: タイトル

    Returns:
        ASCIIヒートマップ文字列
    """
    if not data or not all(data):
        return f"{title}\n[No data to display]"

    # データの正規化
    flat_data = [val for row in data for val in row if val is not None]
    if not flat_data:
        return f"{title}\n[No valid data]"

    min_val = min(flat_data)
    max_val = max(flat_data)

    if max_val == min_val:
        return f"{title}\n[Data has no variation]"

    # ヒートマップ文字（薄い→濃い）
    heat_chars = [" ", "░", "▒", "▓", "█"]

    lines = [title, "=" * (len(title) + 10)]

    # Y軸ラベルの最大幅
    max_y_label = max(len(label) for label in labels_y) if labels_y else 0

    for i, row in enumerate(data):
        y_label = labels_y[i] if i < len(labels_y) else f"Y{i}"
        y_label = y_label.ljust(max_y_label)

        line = f"{y_label} │"
        for val in row:
            if val is None:
                line += " "
            else:
                normalized = (val - min_val) / (max_val - min_val)
                char_index = min(int(normalized * len(heat_chars)), len(heat_chars) - 1)
                line += heat_chars[char_index]

        lines.append(line)

    # X軸ラベル（回転表示は困難なので省略または短縮）
    if labels_x:
        x_axis = " " * (max_y_label + 2)
        for label in labels_x:
            x_axis += label[0] if label else " "  # 最初の文字のみ
        lines.append(x_axis)

    return "\n".join(lines)


def create_metric_cards(metrics: Dict[str, Any], columns: int = 3) -> Columns:
    """
    メトリクスカードを作成

    Args:
        metrics: メトリクスデータ
        columns: 列数

    Returns:
        カラム表示
    """
    cards = []

    for key, value in metrics.items():
        # 値をフォーマット
        if isinstance(value, (int, float)):
            if key.lower().endswith(("rate", "ratio", "percent")):
                formatted_value = format_percentage(value)
            elif key.lower().endswith(("price", "cost", "value", "amount")):
                formatted_value = format_currency(value)
            elif abs(value) >= 1000:
                formatted_value = format_large_number(value)
            else:
                formatted_value = f"{value:,.2f}"
        else:
            formatted_value = str(value)

        # カードを作成
        card_content = Text()
        card_content.append(formatted_value, style="bold white")

        card = Panel(
            Align.center(card_content), title=key, border_style="blue", width=20
        )

        cards.append(card)

    return Columns(cards, equal=True, expand=True)


def create_trend_indicator(
    current: float, previous: float, label: str = "Trend"
) -> Text:
    """
    トレンド指標を作成

    Args:
        current: 現在値
        previous: 前回値
        label: ラベル

    Returns:
        トレンド表示テキスト
    """
    if previous == 0:
        return Text(f"{label}: N/A", style="dim")

    change = current - previous
    change_percent = (change / previous) * 100

    # トレンド記号
    if change > 0:
        trend_symbol = "↗"
        color = "green"
    elif change < 0:
        trend_symbol = "↘"
        color = "red"
    else:
        trend_symbol = "→"
        color = "yellow"

    text = Text()
    text.append(f"{label}: ", style="white")
    text.append(f"{trend_symbol} ", style=color)
    text.append(f"{format_percentage(change_percent)}", style=color)

    return text


def create_distribution_chart(
    data: List[float], bins: int = 10, title: str = "Distribution"
) -> str:
    """
    分布チャートを作成

    Args:
        data: データ
        bins: ビン数
        title: タイトル

    Returns:
        分布チャート文字列
    """
    if not data:
        return f"{title}\n[No data to display]"

    # ヒストグラムを計算
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return f"{title}\n[All values are the same: {min_val}]"

    bin_width = (max_val - min_val) / bins
    hist = [0] * bins

    for value in data:
        bin_index = min(int((value - min_val) / bin_width), bins - 1)
        hist[bin_index] += 1

    # チャート作成
    max_count = max(hist)
    chart_height = 10

    lines = [title, "=" * len(title)]

    for i in range(chart_height, 0, -1):
        line = f"{i * max_count // chart_height:4d} │"
        for count in hist:
            if count >= (i * max_count // chart_height):
                line += "█"
            else:
                line += " "
        lines.append(line)

    # X軸
    x_axis = "     └" + "─" * bins
    lines.append(x_axis)

    # X軸ラベル
    x_labels = f"     {min_val:.1f}" + " " * (bins - 8) + f"{max_val:.1f}"
    lines.append(x_labels)

    return "\n".join(lines)


def create_status_indicator(status: str, label: str = "Status") -> Text:
    """
    ステータス指標を作成

    Args:
        status: ステータス文字列
        label: ラベル

    Returns:
        ステータス表示テキスト
    """
    status_colors = {
        "success": "green",
        "error": "red",
        "warning": "yellow",
        "info": "blue",
        "active": "green",
        "inactive": "dim",
        "pending": "yellow",
        "completed": "green",
        "failed": "red",
    }

    status_symbols = {
        "success": "✓",
        "error": "✗",
        "warning": "⚠",
        "info": "ℹ",
        "active": "●",
        "inactive": "○",
        "pending": "◐",
        "completed": "✓",
        "failed": "✗",
    }

    color = status_colors.get(status.lower(), "white")
    symbol = status_symbols.get(status.lower(), "•")

    text = Text()
    text.append(f"{label}: ", style="white")
    text.append(f"{symbol} ", style=color)
    text.append(status.title(), style=color)

    return text


# ========== CLI統一フォーマット関数 ==========


def create_cli_header(title: str, subtitle: str = None) -> Panel:
    """
    CLI用統一ヘッダーを作成

    Args:
        title: メインタイトル
        subtitle: サブタイトル（オプション）

    Returns:
        ヘッダーパネル
    """
    content = Text()
    content.append(title, style="bold cyan")

    if subtitle:
        content.append("\n")
        content.append(subtitle, style="dim")

    return Panel(
        Align.center(content),
        border_style="cyan",
        padding=(1, 2)
    )


def create_cli_section(title: str, content: str, style: str = "white") -> Panel:
    """
    CLI用セクションパネルを作成

    Args:
        title: セクションタイトル
        content: セクション内容
        style: テキストスタイル

    Returns:
        セクションパネル
    """
    return Panel(
        Text(content, style=style),
        title=f"[bold]{title}[/bold]",
        border_style="blue",
        padding=(0, 1)
    )


def create_cli_command_help(commands: Dict[str, str]) -> Table:
    """
    CLI用コマンドヘルプテーブルを作成

    Args:
        commands: コマンド辞書（コマンド: 説明）

    Returns:
        コマンドヘルプテーブル
    """
    table = Table(title="利用可能なコマンド", box=box.ROUNDED)
    table.add_column("コマンド", style="cyan", no_wrap=True)
    table.add_column("説明", style="white")

    for command, description in commands.items():
        table.add_row(command, description)

    return table


def create_cli_status_bar(status_items: Dict[str, str]) -> Text:
    """
    CLI用ステータスバーを作成

    Args:
        status_items: ステータス項目（ラベル: 値）

    Returns:
        ステータスバーテキスト
    """
    text = Text()

    for i, (label, value) in enumerate(status_items.items()):
        if i > 0:
            text.append(" | ", style="dim")

        text.append(f"{label}: ", style="yellow")
        text.append(value, style="white")

    return text


def create_cli_list_item(
    index: int,
    title: str,
    description: str = None,
    status: str = None
) -> Text:
    """
    CLI用リストアイテムを作成

    Args:
        index: インデックス番号
        title: アイテムタイトル
        description: アイテム説明（オプション）
        status: ステータス（オプション）

    Returns:
        リストアイテムテキスト
    """
    text = Text()

    # インデックス
    text.append(f"{index:2d}. ", style="dim")

    # タイトル
    text.append(title, style="bold white")

    # ステータス
    if status:
        text.append(f" [{status}]", style="green" if status == "OK" else "red")

    # 説明
    if description:
        text.append(f" - {description}", style="dim")

    return text


def create_cli_confirmation_panel(message: str, default: bool = False) -> Panel:
    """
    CLI用確認パネルを作成

    Args:
        message: 確認メッセージ
        default: デフォルト値

    Returns:
        確認パネル
    """
    default_text = "Y/n" if default else "y/N"
    content = f"{message}\n\n[bold yellow]続行しますか？ [{default_text}][/bold yellow]"

    return Panel(
        Text(content, style="white"),
        title="[bold yellow]確認[/bold yellow]",
        border_style="yellow",
        padding=(1, 2)
    )


def create_cli_loading_indicator(message: str = "処理中...") -> Text:
    """
    CLI用ローディング指標を作成

    Args:
        message: ローディングメッセージ

    Returns:
        ローディング指標テキスト
    """
    text = Text()
    text.append("⏳ ", style="yellow")
    text.append(message, style="white")

    return text


def format_cli_table_data(
    headers: List[str],
    rows: List[List[str]],
    title: str = None
) -> Table:
    """
    CLI用テーブルデータをフォーマット

    Args:
        headers: ヘッダー行
        rows: データ行のリスト
        title: テーブルタイトル（オプション）

    Returns:
        フォーマット済みテーブル
    """
    table = Table(title=title, box=box.ROUNDED)

    # ヘッダーを追加
    for i, header in enumerate(headers):
        style = "cyan" if i == 0 else "white"
        table.add_column(header, style=style)

    # データ行を追加
    for row in rows:
        # 行の長さをヘッダーに合わせる
        padded_row = row[:len(headers)] + [""] * (len(headers) - len(row))
        table.add_row(*padded_row)

    return table
