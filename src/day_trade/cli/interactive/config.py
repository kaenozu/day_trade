"""
設定管理のCLIコマンド群
アプリケーション設定の表示、変更、リセット機能を提供
"""

import logging

import click
from rich.console import Console
from rich.table import Table

from ...core.config import config_manager
from ...utils.formatters import create_error_panel, create_success_panel

logger = logging.getLogger(__name__)
console = Console()


@click.group()
def config():
    """設定管理"""
    pass


@config.command("show")
def config_show():
    """現在の設定を表示"""
    config_dict = config_manager.config.model_dump()

    table = Table(title="設定情報")
    table.add_column("設定項目", style="cyan")
    table.add_column("値", style="white")

    def add_config_rows(data, prefix=""):
        """
        設定データを再帰的にテーブルに追加
        
        Args:
            data: 設定データ辞書
            prefix: プレフィックス文字列
        """
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                add_config_rows(value, full_key)
            else:
                table.add_row(full_key, str(value))

    add_config_rows(config_dict)
    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """設定値を変更"""
    try:
        # 型推定（簡易版）
        if value.lower() in ("true", "false"):
            typed_value = value.lower() == "true"
        elif value.isdigit():
            typed_value = int(value)
        elif value.replace(".", "").isdigit():
            typed_value = float(value)
        else:
            typed_value = value

        config_manager.set(key, typed_value)
        console.print(
            create_success_panel(f"設定を更新しました: {key} = {typed_value}")
        )
    except Exception as e:
        console.print(
            create_error_panel(
                f"設定項目 '{key}' の更新中にエラーが発生しました。"
                f"入力値が正しいかご確認ください。詳細: {e}",
                title="設定更新エラー",
            )
        )


@config.command("reset")
@click.confirmation_option(prompt="設定をリセットしますか？")
def config_reset():
    """設定をデフォルトに戻す"""
    try:
        config_manager.reset()
        console.print(create_success_panel("設定をデフォルトにリセットしました。"))
    except Exception as e:
        console.print(
            create_error_panel(
                f"設定のリセット中に予期せぬエラーが発生しました。詳細: {e}",
                title="設定リセットエラー",
            )
        )