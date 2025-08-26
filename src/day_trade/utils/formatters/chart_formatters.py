"""
チャート・グラフィック機能
ASCIIチャート、スパークライン、ヒートマップなどの視覚的表示処理
"""

from typing import List


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