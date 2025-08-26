"""
メインCLIモジュール
全コマンドを統合し、エントリーポイントを提供
"""

from pathlib import Path

import click

from .commands import history, init, stock, validate_codes, watch
from .config import config
from .screening import backtest_command, enhanced_mode, interactive_mode, screen_stocks
from .watchlist import watchlist


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config", "-c", type=click.Path(), help="設定ファイルのパス")
@click.pass_context
def cli(ctx, config_path):
    """対話型デイトレード支援ツール"""
    ctx.ensure_object(dict)
    if config_path:
        ctx.obj["config_path"] = Path(config_path)
    else:
        ctx.obj["config_path"] = None


# 基本コマンドを追加
cli.add_command(init)
cli.add_command(stock)
cli.add_command(history)
cli.add_command(watch)
cli.add_command(validate_codes)

# グループコマンドを追加
cli.add_command(watchlist)
cli.add_command(config)

# スクリーニング・高度機能を追加
cli.add_command(backtest_command)
cli.add_command(enhanced_mode)
cli.add_command(interactive_mode)
cli.add_command(screen_stocks)


if __name__ == "__main__":
    cli()