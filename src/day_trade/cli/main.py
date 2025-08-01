"""
メインCLIエントリーポイント
"""
import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from ..data.stock_fetcher import StockFetcher
from ..models import init_db

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """デイトレード支援ツール"""
    pass


@cli.command()
def init():
    """データベースを初期化"""
    try:
        init_db()
        console.print("[green]データベースを初期化しました。[/green]")
    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")


@cli.command()
@click.argument("code")
@click.option("--details", "-d", is_flag=True, help="詳細情報を表示")
def stock(code: str, details: bool):
    """個別銘柄の情報を表示"""
    fetcher = StockFetcher()
    
    # 現在価格を取得
    with console.status(f"[bold green]{code}の情報を取得中..."):
        current = fetcher.get_current_price(code)
    
    if not current:
        console.print(f"[red]銘柄コード {code} の情報を取得できませんでした。[/red]")
        return
    
    # 基本情報テーブル
    table = Table(title=f"銘柄情報: {current['symbol']}")
    table.add_column("項目", style="cyan")
    table.add_column("値", style="white")
    
    # 色分け
    change_color = "green" if current['change'] >= 0 else "red"
    
    table.add_row("現在値", f"¥{current['current_price']:,.0f}")
    table.add_row("前日終値", f"¥{current['previous_close']:,.0f}")
    table.add_row(
        "前日比",
        f"[{change_color}]¥{current['change']:+,.0f} ({current['change_percent']:+.2f}%)[/{change_color}]"
    )
    table.add_row("出来高", f"{current['volume']:,}")
    
    console.print(table)
    
    # 詳細情報
    if details:
        with console.status("企業情報を取得中..."):
            info = fetcher.get_company_info(code)
        
        if info:
            detail_table = Table(title="企業情報")
            detail_table.add_column("項目", style="cyan")
            detail_table.add_column("値", style="white")
            
            detail_table.add_row("企業名", info.get('name', 'N/A'))
            detail_table.add_row("セクター", info.get('sector', 'N/A'))
            detail_table.add_row("業種", info.get('industry', 'N/A'))
            if info.get('market_cap'):
                detail_table.add_row("時価総額", f"¥{info['market_cap']:,.0f}")
            
            console.print("\n")
            console.print(detail_table)


@cli.command()
@click.argument("code")
@click.option("--period", "-p", default="5d", help="期間 (1d,5d,1mo,3mo,6mo,1y)")
@click.option("--interval", "-i", default="1d", help="間隔 (1m,5m,15m,30m,60m,1d)")
def history(code: str, period: str, interval: str):
    """ヒストリカルデータを表示"""
    fetcher = StockFetcher()
    
    with console.status(f"[bold green]{code}のヒストリカルデータを取得中..."):
        df = fetcher.get_historical_data(code, period=period, interval=interval)
    
    if df is None or df.empty:
        console.print(f"[red]データを取得できませんでした。[/red]")
        return
    
    # テーブル作成
    table = Table(title=f"{code} - 過去 {period} ({interval})")
    table.add_column("日時", style="cyan")
    table.add_column("始値", justify="right")
    table.add_column("高値", justify="right")
    table.add_column("安値", justify="right")
    table.add_column("終値", justify="right")
    table.add_column("出来高", justify="right")
    
    # 最新10件を表示
    for idx, row in df.tail(10).iterrows():
        # 前日比で色分け
        if len(df) > 1 and idx > df.index[0]:
            prev_close = df.loc[df.index[df.index.get_loc(idx) - 1], 'Close']
            color = "green" if row['Close'] >= prev_close else "red"
        else:
            color = "white"
        
        table.add_row(
            str(idx.strftime('%Y-%m-%d %H:%M')),
            f"¥{row['Open']:,.0f}",
            f"¥{row['High']:,.0f}",
            f"¥{row['Low']:,.0f}",
            f"[{color}]¥{row['Close']:,.0f}[/{color}]",
            f"{int(row['Volume']):,}"
        )
    
    console.print(table)
    
    # サマリー
    console.print(f"\n[bold]サマリー:[/bold]")
    console.print(f"期間高値: ¥{df['High'].max():,.0f}")
    console.print(f"期間安値: ¥{df['Low'].min():,.0f}")
    console.print(f"平均出来高: {int(df['Volume'].mean()):,}")


@cli.command()
@click.argument("codes", nargs=-1, required=True)
def watch(codes):
    """複数銘柄の現在価格を一覧表示"""
    fetcher = StockFetcher()
    
    table = Table(title="ウォッチリスト")
    table.add_column("コード", style="cyan")
    table.add_column("現在値", justify="right")
    table.add_column("前日比", justify="right")
    table.add_column("前日比率", justify="right")
    table.add_column("出来高", justify="right")
    
    with console.status("[bold green]価格情報を取得中..."):
        results = fetcher.get_realtime_data(list(codes))
    
    for code, data in results.items():
        if data:
            change_color = "green" if data['change'] >= 0 else "red"
            table.add_row(
                code,
                f"¥{data['current_price']:,.0f}",
                f"[{change_color}]¥{data['change']:+,.0f}[/{change_color}]",
                f"[{change_color}]{data['change_percent']:+.2f}%[/{change_color}]",
                f"{data['volume']:,}"
            )
    
    console.print(table)


def main():
    """メインエントリーポイント"""
    cli()


if __name__ == "__main__":
    main()