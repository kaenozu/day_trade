#!/usr/bin/env python3
"""
テストカバレッジ可視化ツール
Issue 183: テストカバレッジの計測と可視化

このスクリプトはテストカバレッジデータを元に
視覚的なチャートとダッシュボードを生成します。
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# 軽量な可視化ライブラリを使用（外部依存を最小限に）
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib が見つかりません。グラフ生成をスキップします。")

def load_coverage_data(json_file: Path) -> Optional[Dict[str, Any]]:
    """カバレッジJSONファイルを読み込み"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"カバレッジデータ読み込みエラー: {e}")
        return None

def create_coverage_summary_chart(coverage_data: Dict[str, Any], output_dir: Path):
    """カバレッジサマリーチャートを生成"""
    if not HAS_MATPLOTLIB:
        return

    totals = coverage_data.get("totals", {})
    percent_covered = totals.get("percent_covered", 0)

    # 円グラフでカバレッジを表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 左: 全体カバレッジ円グラフ
    covered_pct = percent_covered
    uncovered_pct = 100 - covered_pct

    colors = ['#2ECC71', '#E74C3C']  # 緑（カバー済み）、赤（未カバー）
    ax1.pie([covered_pct, uncovered_pct],
            labels=[f'カバー済み\n{covered_pct:.1f}%', f'未カバー\n{uncovered_pct:.1f}%'],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('全体カバレッジ', fontsize=14, fontweight='bold')

    # 右: カバレッジ目標との比較
    targets = [60, 70, 80, 90]
    target_labels = ['最低限\n(60%)', '良好\n(70%)', '優秀\n(80%)', '理想\n(90%)']
    target_colors = ['#F39C12', '#E67E22', '#27AE60', '#16A085']

    current_color = '#E74C3C'  # デフォルト赤
    for i, target in enumerate(targets):
        if covered_pct >= target:
            current_color = target_colors[i]

    # 横棒グラフ
    y_pos = range(len(targets))
    ax2.barh(y_pos, targets, color=target_colors, alpha=0.3, label='目標')
    ax2.barh(len(targets), covered_pct, color=current_color, alpha=0.8, label='現在')

    ax2.set_yticks(list(y_pos) + [len(targets)])
    ax2.set_yticklabels(target_labels + ['現在'])
    ax2.set_xlabel('カバレッジ (%)')
    ax2.set_title('カバレッジ目標達成度', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)

    # 目標線を追加
    for target in targets:
        ax2.axvline(x=target, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存
    chart_file = output_dir / "coverage_summary.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"サマリーチャートを生成: {chart_file}")

def create_file_coverage_chart(coverage_data: Dict[str, Any], output_dir: Path):
    """ファイル別カバレッジチャートを生成"""
    if not HAS_MATPLOTLIB:
        return

    files_data = coverage_data.get("files", {})
    if not files_data:
        return

    # ファイル別データを抽出・ソート
    file_coverage = []
    for filename, data in files_data.items():
        # パス名を短縮
        short_name = filename.replace("src\\day_trade\\", "").replace("src/day_trade/", "")
        if len(short_name) > 40:
            short_name = "..." + short_name[-37:]

        summary = data.get("summary", {})
        coverage_pct = summary.get("percent_covered", 0)

        file_coverage.append((short_name, coverage_pct))

    # カバレッジでソート（低い順）
    file_coverage.sort(key=lambda x: x[1])

    # 上位20件と下位20件を表示
    top_files = file_coverage[-20:]
    bottom_files = file_coverage[:20]

    # 2つのチャートを作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

    # 上位20件（高カバレッジ）
    if top_files:
        names, values = zip(*top_files)
        colors = ['#27AE60' if v >= 80 else '#F39C12' if v >= 60 else '#E74C3C' for v in values]

        y_pos = range(len(names))
        bars1 = ax1.barh(y_pos, values, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('カバレッジ (%)')
        ax1.set_title('ファイル別カバレッジ（上位20件）', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 100)

        # 値をバーに表示
        for bar, value in zip(bars1, values):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=7)

    # 下位20件（低カバレッジ）
    if bottom_files:
        names, values = zip(*bottom_files)
        colors = ['#E74C3C' if v < 60 else '#F39C12' if v < 80 else '#27AE60' for v in values]

        y_pos = range(len(names))
        bars2 = ax2.barh(y_pos, values, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('カバレッジ (%)')
        ax2.set_title('ファイル別カバレッジ（下位20件）', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)

        # 値をバーに表示
        for bar, value in zip(bars2, values):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}%', ha='left', va='center', fontsize=7)

    plt.tight_layout()

    # 保存
    chart_file = output_dir / "file_coverage.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"ファイル別チャートを生成: {chart_file}")

def create_package_coverage_chart(coverage_data: Dict[str, Any], output_dir: Path):
    """パッケージ別カバレッジチャートを生成"""
    if not HAS_MATPLOTLIB:
        return

    files_data = coverage_data.get("files", {})
    if not files_data:
        return

    # パッケージ別にカバレッジを集計
    package_stats = {}

    for filename, data in files_data.items():
        # パッケージ名を抽出
        path_parts = filename.replace("src\\day_trade\\", "").replace("src/day_trade/", "").split("/")
        if "\\" in filename:
            path_parts = filename.replace("src\\day_trade\\", "").split("\\")

        if len(path_parts) > 1:
            package = path_parts[0]
        else:
            package = "root"

        summary = data.get("summary", {})
        covered_lines = summary.get("covered_lines", 0)
        num_statements = summary.get("num_statements", 0)

        if package not in package_stats:
            package_stats[package] = {"covered": 0, "total": 0}

        package_stats[package]["covered"] += covered_lines
        package_stats[package]["total"] += num_statements

    # パッケージ別カバレッジ率を計算
    package_coverage = []
    for package, stats in package_stats.items():
        if stats["total"] > 0:
            coverage_pct = (stats["covered"] / stats["total"]) * 100
            package_coverage.append((package, coverage_pct, stats["total"]))

    if not package_coverage:
        return

    # ソート（カバレッジ率順）
    package_coverage.sort(key=lambda x: x[1], reverse=True)

    # チャート作成
    fig, ax = plt.subplots(figsize=(12, 8))

    names = [item[0] for item in package_coverage]
    values = [item[1] for item in package_coverage]
    sizes = [item[2] for item in package_coverage]

    # バーの色を設定
    colors = ['#27AE60' if v >= 80 else '#F39C12' if v >= 60 else '#E74C3C' for v in values]

    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('カバレッジ (%)')
    ax.set_title('パッケージ別カバレッジ', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)

    # 値とサイズをバーに表示
    for bar, value, size in zip(bars, values, sizes):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}% ({size} lines)', ha='left', va='center', fontsize=9)

    # 目標線を追加
    for target in [60, 70, 80]:
        ax.axvline(x=target, color='gray', linestyle='--', alpha=0.5)
        ax.text(target, len(names), f'{target}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存
    chart_file = output_dir / "package_coverage.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"パッケージ別チャートを生成: {chart_file}")

def create_coverage_heatmap(coverage_data: Dict[str, Any], output_dir: Path):
    """カバレッジヒートマップを生成（ASCII版）"""
    files_data = coverage_data.get("files", {})
    if not files_data:
        return

    # ファイル別データを抽出
    file_matrix = []
    for filename, data in files_data.items():
        short_name = filename.replace("src\\day_trade\\", "").replace("src/day_trade/", "")
        summary = data.get("summary", {})
        coverage_pct = summary.get("percent_covered", 0)
        num_statements = summary.get("num_statements", 0)

        file_matrix.append({
            "name": short_name,
            "coverage": coverage_pct,
            "size": num_statements
        })

    # サイズでソート
    file_matrix.sort(key=lambda x: x["size"], reverse=True)

    # ASCIIヒートマップを生成
    heatmap_content = "# カバレッジヒートマップ\n\n"
    heatmap_content += "ファイルサイズ（行数）とカバレッジ率の関係:\n\n"
    heatmap_content += "```\n"
    heatmap_content += "Legend: ░ 0-25%  ▒ 25-50%  ▓ 50-75%  █ 75-100%\n"
    heatmap_content += "Size:   ● Large  ◐ Medium  ○ Small\n\n"

    for item in file_matrix[:30]:  # 上位30件
        name = item["name"]
        coverage = item["coverage"]
        size = item["size"]

        # カバレッジ表示文字
        if coverage >= 75:
            cov_char = "█"
        elif coverage >= 50:
            cov_char = "▓"
        elif coverage >= 25:
            cov_char = "▒"
        else:
            cov_char = "░"

        # サイズ表示文字
        if size > 200:
            size_char = "●"
        elif size > 100:
            size_char = "◐"
        else:
            size_char = "○"

        # 名前を調整
        display_name = name[:40].ljust(40)

        heatmap_content += f"{cov_char} {size_char} {display_name} {coverage:6.1f}% ({size:3d} lines)\n"

    heatmap_content += "```\n"

    # ファイルに保存
    heatmap_file = output_dir / "coverage_heatmap.md"
    with open(heatmap_file, 'w', encoding='utf-8') as f:
        f.write(heatmap_content)

    print(f"カバレッジヒートマップを生成: {heatmap_file}")

def create_coverage_dashboard(coverage_files: List[Path], output_dir: Path):
    """HTMLダッシュボードを生成"""
    timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')

    # 最新のカバレッジデータを取得
    latest_file = sorted(coverage_files)[-1] if coverage_files else None
    if not latest_file:
        return

    coverage_data = load_coverage_data(latest_file)
    if not coverage_data:
        return

    totals = coverage_data.get("totals", {})

    # HTML コンテンツ生成
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Day Trade - テストカバレッジダッシュボード</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .coverage-high {{ background-color: #27ae60; }}
        .coverage-medium {{ background-color: #f39c12; }}
        .coverage-low {{ background-color: #e74c3c; }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            text-align: center;
            background-color: #fafafa;
            padding: 20px;
            border-radius: 10px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .recommendations {{
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Day Trade テストカバレッジダッシュボード</h1>
            <p>最終更新: {timestamp}</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{totals.get('percent_covered', 0):.1f}%</div>
                <div class="metric-label">ライン カバレッジ</div>
                <div class="progress-bar">
                    <div class="progress-fill {'coverage-high' if totals.get('percent_covered', 0) >= 80 else 'coverage-medium' if totals.get('percent_covered', 0) >= 60 else 'coverage-low'}"
                         style="width: {totals.get('percent_covered', 0)}%"></div>
                </div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{totals.get('covered_lines', 0):,}</div>
                <div class="metric-label">カバー済み行数</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{totals.get('num_statements', 0):,}</div>
                <div class="metric-label">総行数</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{len(coverage_data.get('files', {}))}</div>
                <div class="metric-label">ファイル数</div>
            </div>
        </div>
"""

    # 推奨事項の追加
    recommendations = []
    coverage_pct = totals.get('percent_covered', 0)

    if coverage_pct < 60:
        recommendations.append("🔴 カバレッジが60%未満です。テストの追加が急務です。")
    elif coverage_pct < 70:
        recommendations.append("🟡 カバレッジは60%以上ですが、70%を目標に改善を続けてください。")
    elif coverage_pct < 80:
        recommendations.append("🟢 良好なカバレッジです。80%達成を目指しましょう。")
    else:
        recommendations.append("🏆 優秀なカバレッジです！この水準を維持してください。")

    # ファイル数が多い場合の推奨
    file_count = len(coverage_data.get('files', {}))
    if file_count > 50:
        recommendations.append("📁 ファイル数が多いため、モジュール別のテスト戦略を検討してください。")

    if recommendations:
        html_content += """
        <div class="recommendations">
            <h3>📋 推奨事項</h3>
            <ul>
"""
        for rec in recommendations:
            html_content += f"                <li>{rec}</li>\n"

        html_content += """
            </ul>
        </div>
"""

    # チャート埋め込み
    chart_files = [
        ("coverage_summary.png", "カバレッジサマリー"),
        ("file_coverage.png", "ファイル別カバレッジ"),
        ("package_coverage.png", "パッケージ別カバレッジ")
    ]

    html_content += """
        <div class="charts">
"""

    for chart_file, chart_title in chart_files:
        chart_path = output_dir / chart_file
        if chart_path.exists():
            html_content += f"""
            <div class="chart-container">
                <h3>{chart_title}</h3>
                <img src="{chart_file}" alt="{chart_title}">
            </div>
"""

    html_content += """
        </div>

        <div class="footer">
            <p>このダッシュボードは scripts/coverage_visualizer.py により自動生成されました。</p>
        </div>
    </div>
</body>
</html>
"""

    # ダッシュボードを保存
    dashboard_file = output_dir / "dashboard.html"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTMLダッシュボードを生成: {dashboard_file}")

def main():
    """メイン実行関数"""
    print("Day Trade カバレッジ可視化ツール")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    coverage_dir = project_root / "reports" / "coverage"

    if not coverage_dir.exists():
        print(f"エラー: カバレッジディレクトリが見つかりません: {coverage_dir}")
        print("まず scripts/coverage_report.py を実行してください。")
        return 1

    # カバレッジJSONファイルを検索
    json_files = list(coverage_dir.glob("coverage_*.json"))
    if not json_files:
        print("エラー: カバレッジJSONファイルが見つかりません。")
        print("まず scripts/coverage_report.py を実行してください。")
        return 1

    # 最新のファイルを使用
    latest_json = sorted(json_files)[-1]
    print(f"カバレッジデータを読み込み: {latest_json}")

    coverage_data = load_coverage_data(latest_json)
    if not coverage_data:
        return 1

    # 可視化出力ディレクトリを作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = coverage_dir / f"visualizations_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    print(f"可視化ファイルを生成中: {output_dir}")

    # 各種チャートを生成
    create_coverage_summary_chart(coverage_data, output_dir)
    create_file_coverage_chart(coverage_data, output_dir)
    create_package_coverage_chart(coverage_data, output_dir)
    create_coverage_heatmap(coverage_data, output_dir)

    # HTMLダッシュボードを生成
    create_coverage_dashboard(json_files, output_dir)

    print("\n" + "=" * 50)
    print("可視化完了!")
    print(f"結果は以下のディレクトリで確認できます:")
    print(f"  {output_dir}")
    print(f"\nHTMLダッシュボード:")
    print(f"  {output_dir / 'dashboard.html'}")

    if not HAS_MATPLOTLIB:
        print("\n注意: matplotlib がインストールされていないため、一部のグラフが生成されませんでした。")
        print("グラフを表示するには以下を実行してください:")
        print("  pip install matplotlib")

    return 0

if __name__ == "__main__":
    sys.exit(main())
