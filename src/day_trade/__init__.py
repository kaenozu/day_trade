"""Day Trade - A CLI-based day trading support application"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"


# 構造化ロギングを自動設定
def _setup_application_logging():
    """アプリケーション起動時の構造化ロギング設定"""
    try:
        from .utils.logging_config import setup_logging

        setup_logging()
    except ImportError:
        # ロギング設定が利用できない場合は警告なしで継続
        pass


# アプリケーション初期化時にロギングを設定
_setup_application_logging()
