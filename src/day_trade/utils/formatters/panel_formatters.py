"""
パネル作成機能
エラー、成功、警告、情報パネルの作成処理
"""

from typing import List, Optional

from rich.align import Align
from rich.panel import Panel
from rich.text import Text


def create_error_panel(
    message: str,
    title: str = "エラー",
    solutions: Optional[List[str]] = None,
    show_emoji: bool = True,
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

    return Panel(Align.center(content), border_style="cyan", padding=(1, 2))


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
        padding=(0, 1),
    )


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
        padding=(1, 2),
    )