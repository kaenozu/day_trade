#!/usr/bin/env python3
"""
Day Trade Personal - 静寂ログ設定

Issue #915対応: ログノイズを削減し、簡潔な表示を実現
"""

import logging
import sys
from pathlib import Path


class QuietHandler(logging.Handler):
    """静寂ログハンドラー - 重要なメッセージのみ表示"""
    
    def __init__(self):
        super().__init__()
        self.important_messages = set()
        
    def emit(self, record):
        """重要なメッセージのみ出力"""
        # デバッグモードの場合は全て出力
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            self._emit_debug(record)
            return
            
        # 重要なメッセージのみフィルタリング
        if self._is_important_message(record):
            self._emit_important(record)
            
    def _is_important_message(self, record) -> bool:
        """重要なメッセージかどうか判定"""
        message = record.getMessage().lower()
        
        # 重要なキーワード
        important_keywords = [
            'エラー', 'error', 'failed', 'exception',
            '警告', 'warning', 'warn',
            '完了', 'completed', 'finished',
            '開始', 'starting', 'initializing',
            '分析', 'analysis', 'analyzing'
        ]
        
        # 除外するノイズキーワード
        noise_keywords = [
            'database engine initialized',
            'yfinance利用可能',
            '設定ファイルを読み込みました',
            '設定の妥当性チェック',
            'logging initialized',
            'モジュール読み込み',
            'import successful'
        ]
        
        # ノイズキーワードが含まれる場合は除外
        for noise in noise_keywords:
            if noise.lower() in message:
                return False
                
        # 重要キーワードが含まれる場合は表示
        for keyword in important_keywords:
            if keyword in message:
                return True
                
        # ERROR、WARNING レベルは常に表示
        if record.levelno >= logging.WARNING:
            return True
            
        return False
        
    def _emit_debug(self, record):
        """デバッグモード時の出力"""
        try:
            msg = self.format(record)
            stream = sys.stderr if record.levelno >= logging.WARNING else sys.stdout
            stream.write(f"[DEBUG] {msg}\n")
            stream.flush()
        except Exception:
            self.handleError(record)
            
    def _emit_important(self, record):
        """重要なメッセージの出力"""
        try:
            msg = self.format(record)
            
            # レベル別アイコン
            level_icons = {
                logging.ERROR: '❌',
                logging.WARNING: '⚠️',
                logging.INFO: 'ℹ️',
                logging.DEBUG: '🔍'
            }
            
            icon = level_icons.get(record.levelno, 'ℹ️')
            
            # 簡潔な形式で出力
            print(f"{icon} {msg}")
            
        except Exception:
            self.handleError(record)


def setup_quiet_logging(debug: bool = False):
    """静寂ログ設定"""
    
    # 既存のハンドラーをクリア
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # 静寂ハンドラーを設定
    quiet_handler = QuietHandler()
    
    if debug:
        # デバッグモード: 詳細表示
        quiet_handler.setLevel(logging.DEBUG)
        root_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(message)s [%(name)s]'
        )
    else:
        # 通常モード: 静寂モード
        quiet_handler.setLevel(logging.INFO)
        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        
    quiet_handler.setFormatter(formatter)
    root_logger.addHandler(quiet_handler)
    
    # 特定のライブラリのログレベルを制御
    _suppress_noisy_libraries()
    
    
def _suppress_noisy_libraries():
    """ノイズの多いライブラリのログを制御"""
    noisy_loggers = [
        'urllib3',
        'requests',
        'yfinance',
        'matplotlib',
        'PIL',
        'asyncio',
        'websockets',
        'socketio'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        
    # coloramaのエラー出力を制御
    try:
        import colorama
        # colorama関連のログを制御
        logging.getLogger('colorama').setLevel(logging.ERROR)
    except ImportError:
        pass


class SilentStartup:
    """起動時のサイレントモード管理"""
    
    def __init__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.startup_complete = False
        
    def __enter__(self):
        """サイレントモード開始"""
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            # デバッグモードでない場合のみサイレント
            sys.stdout = open('nul' if sys.platform == 'win32' else '/dev/null', 'w')
            # stderrは重要なエラーのために残す
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """サイレントモード終了"""
        if hasattr(sys.stdout, 'close') and sys.stdout != self.original_stdout:
            sys.stdout.close()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.startup_complete = True


def quiet_import(module_name: str):
    """静寂インポート - インポート時のメッセージを制御"""
    try:
        with SilentStartup():
            module = __import__(module_name)
        return module
    except ImportError as e:
        # インポートエラーのみ表示
        logging.warning(f"モジュール {module_name} のインポートに失敗: {e}")
        return None


# 使用例関数
def demonstrate_quiet_logging():
    """静寂ログのデモンストレーション"""
    setup_quiet_logging(debug=False)
    
    # これらのメッセージは表示されない（ノイズ）
    logging.info("Database engine initialized")
    logging.info("yfinance利用可能")
    logging.info("設定ファイルを読み込みました")
    
    # これらのメッセージは表示される（重要）
    logging.info("分析開始")
    logging.warning("データ取得に時間がかかっています")
    logging.error("ネットワークエラーが発生しました")
    logging.info("分析完了")


if __name__ == "__main__":
    demonstrate_quiet_logging()