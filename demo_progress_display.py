#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoUpdateOptimizer プログレスバー表示デモ
Issue #881対応：進捗バー・ステータス表示システムのデモンストレーション
"""

import asyncio
import time
import random
import sys
import os
from datetime import datetime
from pathlib import Path

# Windows環境での文字化け対策
os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Rich進捗表示ライブラリ
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        TaskID,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        SpinnerColumn,
        ProgressColumn
    )
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich ライブラリが利用できません。pip install rich でインストールしてください。")

from auto_update_optimizer import AutoUpdateOptimizer

class EnhancedProgressDisplay:
    """拡張プログレス表示システム"""

    def __init__(self, optimizer: AutoUpdateOptimizer):
        self.optimizer = optimizer
        self.console = Console()
        self.start_time = None
        self.is_running = False

    def create_progress_layout(self) -> Layout:
        """プログレス表示レイアウト作成"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=10)
        )

        layout["main"].split_row(
            Layout(name="progress"),
            Layout(name="stats", ratio=1)
        )

        return layout

    def create_header_panel(self) -> Panel:
        """ヘッダーパネル作成"""
        system_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = f"🚀 AutoUpdateOptimizer - 自動更新最適化システム ({system_time})"

        return Panel(
            Text(title, style="bold blue"),
            style="blue"
        )

    def create_progress_panel(self, progress: Progress) -> Panel:
        """プログレスパネル作成"""
        return Panel(
            progress,
            title="📊 処理進捗",
            title_align="left",
            style="green"
        )

    def create_stats_table(self) -> Table:
        """統計テーブル作成"""
        table = Table(title="📈 システム統計", style="cyan")

        table.add_column("項目", style="yellow", no_wrap=True)
        table.add_column("値", style="white")
        table.add_column("詳細", style="dim")

        # システムメトリクス
        metrics = self.optimizer.system_metrics

        table.add_row("💻 CPU使用率", f"{metrics.cpu_usage:.1f}%", "システム全体")
        table.add_row("🧠 メモリ使用量", f"{metrics.memory_usage_mb:.0f}MB", "現在のプロセス")
        table.add_row("⚡ 負荷レベル", metrics.load_level.value, "システム負荷状態")
        table.add_row("🔄 更新頻度", f"{metrics.update_frequency:.0f}秒", "基本更新間隔")

        # 銘柄統計
        if self.optimizer.symbol_manager:
            stats = self.optimizer.get_enhanced_symbol_statistics()
            table.add_row("📊 選択銘柄数", str(stats['current_selected_symbols']), "処理対象")
            table.add_row("🥇 高優先度", str(stats['priority_distribution'].get('high', 0)), "30秒間隔")
            table.add_row("🥈 中優先度", str(stats['priority_distribution'].get('medium', 0)), "60秒間隔")
            table.add_row("🥉 低優先度", str(stats['priority_distribution'].get('low', 0)), "120秒間隔")

        return table

    def create_performance_table(self) -> Table:
        """パフォーマンステーブル作成"""
        table = Table(title="⚡ パフォーマンス指標", style="magenta")

        table.add_column("指標", style="yellow")
        table.add_column("現在値", style="white")
        table.add_column("目標値", style="green")
        table.add_column("状態", style="blue")

        # サンプルパフォーマンスデータ
        table.add_row("平均処理時間", "0.85秒", "< 1.0秒", "✅ 良好")
        table.add_row("成功率", "96.2%", "> 95%", "✅ 良好")
        table.add_row("エラー率", "3.8%", "< 5%", "✅ 良好")
        table.add_row("スループット", "42銘柄/分", "> 30銘柄/分", "✅ 良好")

        return table

    async def run_demo_simulation(self, duration_minutes: int = 2):
        """デモシミュレーション実行"""
        self.start_time = datetime.now()
        self.is_running = True

        # レイアウト作成
        layout = self.create_progress_layout()

        # プログレスバー設定
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False
        )

        # 銘柄リスト
        symbols = self.optimizer.current_symbols[:20]  # デモでは20銘柄に制限
        total_symbols = len(symbols)

        # メインタスク
        main_task = progress.add_task(
            "全体進捗",
            total=total_symbols,
            completed=0
        )

        # 個別処理タスク
        current_task = progress.add_task(
            "現在の処理",
            total=100,
            completed=0
        )

        with Live(layout, refresh_per_second=4, screen=True) as live:

            for i, symbol in enumerate(symbols):
                # レイアウト更新
                layout["header"].update(self.create_header_panel())
                layout["progress"].update(self.create_progress_panel(progress))

                # 統計テーブル（左側）
                stats_layout = Layout()
                stats_layout.split_column(
                    Layout(Panel(self.create_stats_table(), style="cyan"), name="stats"),
                    Layout(Panel(self.create_performance_table(), style="magenta"), name="performance")
                )
                layout["stats"].update(stats_layout)

                # フッター情報
                footer_content = self.create_footer_content(i + 1, total_symbols, symbol)
                layout["footer"].update(Panel(footer_content, title="🔍 詳細情報", style="dim"))

                # 現在の銘柄処理シミュレーション
                progress.update(current_task, description=f"処理中: {symbol}", completed=0)

                # 段階的処理シミュレーション
                stages = [
                    ("データ取得", 25),
                    ("分析実行", 50),
                    ("予測計算", 75),
                    ("結果保存", 100)
                ]

                for stage_name, stage_progress in stages:
                    progress.update(
                        current_task,
                        description=f"{symbol} - {stage_name}",
                        completed=stage_progress
                    )

                    # リアルな処理時間をシミュレート
                    await asyncio.sleep(random.uniform(0.3, 0.8))

                    # システムメトリクス更新
                    self.optimizer.update_system_metrics()

                # メインタスク進捗更新
                progress.update(main_task, completed=i + 1)

                # 短時間休憩
                await asyncio.sleep(0.2)

        self.is_running = False

        # 完了メッセージ
        self.console.print("\n" + "="*80)
        self.console.print("🎉 [bold green]自動更新最適化システム デモ完了![/bold green]")
        self.console.print(f"📊 処理銘柄数: {total_symbols}")
        self.console.print(f"⏱️  実行時間: {(datetime.now() - self.start_time).total_seconds():.1f}秒")
        self.console.print("="*80)

    def create_footer_content(self, current: int, total: int, symbol: str) -> Table:
        """フッター内容作成"""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="white")

        # 実行時間
        elapsed = (datetime.now() - self.start_time).total_seconds()
        table.add_row("🕐 実行時間:", f"{elapsed:.1f}秒")

        # 進捗情報
        progress_pct = (current / total) * 100
        table.add_row("📈 進捗:", f"{progress_pct:.1f}% ({current}/{total})")

        # 現在処理中
        table.add_row("🎯 現在処理:", symbol)

        # 推定残り時間
        if current > 0:
            avg_time = elapsed / current
            remaining_time = avg_time * (total - current)
            table.add_row("⏳ 推定残り時間:", f"{remaining_time:.1f}秒")

        # システム情報
        table.add_row("💾 メモリ使用量:", f"{self.optimizer.system_metrics.memory_usage_mb:.0f}MB")
        table.add_row("⚡ CPU使用率:", f"{self.optimizer.system_metrics.cpu_usage:.1f}%")

        return table

async def main():
    """メイン実行関数"""
    if not RICH_AVAILABLE:
        print("Rich ライブラリが必要です。pip install rich でインストールしてください。")
        return

    console = Console()

    console.print("🚀 [bold blue]AutoUpdateOptimizer プログレスバー表示デモ[/bold blue]")
    console.print("📋 Issue #881対応：進捗バー・ステータス表示システム")
    console.print("")

    try:
        # システム初期化
        console.print("⚙️  システム初期化中...")
        optimizer = AutoUpdateOptimizer()
        await optimizer.initialize()

        console.print(f"✅ 初期化完了: {len(optimizer.current_symbols)}銘柄を管理")
        console.print("")

        # プログレス表示デモ実行
        display = EnhancedProgressDisplay(optimizer)

        console.print("🎬 デモシミュレーション開始（2分間）...")
        console.print("💡 ESCキーでいつでも終了できます")
        console.print("")

        await display.run_demo_simulation(duration_minutes=2)

    except KeyboardInterrupt:
        console.print("\n⚠️  ユーザーによって中断されました")
    except Exception as e:
        console.print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())