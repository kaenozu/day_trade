"""
セキュリティ設定管理
カスタム条件の安全な実行とセキュリティポリシー
"""

import ast
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Set

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SecurityConfig:
    """セキュリティ設定管理クラス"""

    # 許可されたモジュール・関数の安全なリスト
    ALLOWED_MODULES = {
        'math', 'datetime', 'decimal', 'statistics',
        'numpy', 'pandas', 'talib'  # データ分析系ライブラリ
    }

    # 禁止されたキーワード・関数
    FORBIDDEN_KEYWORDS = {
        'import', '__import__', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'reload', 'exit', 'quit', 'help',
        'subprocess', 'os', 'sys', 'shutil'
    }

    # 許可されたビルトイン関数
    ALLOWED_BUILTINS = {
        'abs', 'all', 'any', 'bool', 'float', 'int', 'len',
        'max', 'min', 'round', 'str', 'sum', 'type',
        'range', 'enumerate', 'zip', 'map', 'filter',
        'sorted', 'reversed'
    }

    def __init__(self, enable_custom_functions: bool = True):
        """
        初期化

        Args:
            enable_custom_functions: カスタム関数の実行を許可するか
        """
        self.enable_custom_functions = enable_custom_functions
        self.function_whitelist: Set[str] = set()
        self.approved_functions: Dict[str, str] = {}  # 関数名: ハッシュ

    def validate_custom_function(self, func: Callable) -> bool:
        """
        カスタム関数のセキュリティ検証

        Args:
            func: 検証する関数

        Returns:
            bool: 安全な関数かどうか
        """
        if not self.enable_custom_functions:
            logger.warning("カスタム関数の実行が無効化されています")
            return False

        try:
            # 関数のソースコードを取得
            source = inspect.getsource(func)

            # ASTパースによる静的解析
            if not self._analyze_function_ast(source):
                logger.error(f"カスタム関数のAST解析でセキュリティ違反を検出: {func.__name__}")
                return False

            # 関数のハッシュを計算
            func_hash = self._calculate_function_hash(source)

            # ホワイトリストチェック
            if func.__name__ in self.function_whitelist:
                if self.approved_functions.get(func.__name__) == func_hash:
                    return True
                else:
                    logger.warning(f"関数が変更されています: {func.__name__}")
                    return False

            # 新規関数の場合は管理者承認が必要
            logger.info(f"新規カスタム関数検出: {func.__name__} (ハッシュ: {func_hash})")
            return self._request_function_approval(func.__name__, source, func_hash)

        except Exception as e:
            logger.error(f"カスタム関数検証エラー: {e}")
            return False

    def _analyze_function_ast(self, source: str) -> bool:
        """
        ASTを使用した関数の静的解析

        Args:
            source: 関数のソースコード

        Returns:
            bool: 安全な関数かどうか
        """
        try:
            tree = ast.parse(source)

            # 禁止されたノードタイプをチェック
            for node in ast.walk(tree):
                # import文の検査
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom):
                        module_name = node.module
                    else:
                        module_name = node.names[0].name if node.names else None

                    if module_name and module_name not in self.ALLOWED_MODULES:
                        logger.error(f"禁止されたモジュールのimport: {module_name}")
                        return False

                # 関数呼び出しの検査
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.FORBIDDEN_KEYWORDS:
                            logger.error(f"禁止された関数の呼び出し: {func_name}")
                            return False
                        elif func_name not in self.ALLOWED_BUILTINS and not func_name.startswith('_'):
                            # 未知の関数は警告レベル
                            logger.warning(f"未知の関数呼び出し: {func_name}")

                # 属性アクセスの検査
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in self.FORBIDDEN_KEYWORDS:
                            logger.error(f"禁止されたオブジェクトへのアクセス: {node.value.id}.{node.attr}")
                            return False

            return True

        except SyntaxError as e:
            logger.error(f"関数の構文エラー: {e}")
            return False
        except Exception as e:
            logger.error(f"AST解析エラー: {e}")
            return False

    def _calculate_function_hash(self, source: str) -> str:
        """
        関数のハッシュ値を計算

        Args:
            source: 関数のソースコード

        Returns:
            str: SHA256ハッシュ値
        """
        return hashlib.sha256(source.encode('utf-8')).hexdigest()

    def _request_function_approval(self, func_name: str, source: str, func_hash: str) -> bool:
        """
        関数の承認をリクエスト

        Args:
            func_name: 関数名
            source: ソースコード
            func_hash: ハッシュ値

        Returns:
            bool: 承認されたかどうか（現在は自動承認）
        """
        # 実際の実装では管理者承認フローが必要
        # 現在は開発用に自動承認
        logger.info(f"カスタム関数を自動承認: {func_name}")
        self.approve_function(func_name, func_hash)
        return True

    def approve_function(self, func_name: str, func_hash: str):
        """
        関数を承認リストに追加

        Args:
            func_name: 関数名
            func_hash: ハッシュ値
        """
        self.function_whitelist.add(func_name)
        self.approved_functions[func_name] = func_hash
        logger.info(f"関数承認: {func_name}")

    def revoke_function(self, func_name: str):
        """
        関数の承認を取り消し

        Args:
            func_name: 関数名
        """
        self.function_whitelist.discard(func_name)
        self.approved_functions.pop(func_name, None)
        logger.info(f"関数承認取り消し: {func_name}")

    def create_safe_globals(self) -> Dict[str, Any]:
        """
        安全なグローバル環境を作成

        Returns:
            Dict[str, Any]: 制限されたグローバル環境
        """
        safe_globals = {
            '__builtins__': {
                name: getattr(__builtins__, name)
                for name in self.ALLOWED_BUILTINS
                if hasattr(__builtins__, name)
            }
        }

        # 許可されたモジュールのみ追加
        try:
            import math
            safe_globals['math'] = math
        except ImportError:
            pass

        try:
            import datetime
            safe_globals['datetime'] = datetime
        except ImportError:
            pass

        try:
            from decimal import Decimal
            safe_globals['Decimal'] = Decimal
        except ImportError:
            pass

        return safe_globals

    def execute_safe_function(
        self,
        func: Callable,
        *args,
        timeout: float = 5.0,
        **kwargs
    ) -> Any:
        """
        安全な環境でカスタム関数を実行

        Args:
            func: 実行する関数
            *args: 関数の引数
            timeout: タイムアウト時間（秒）
            **kwargs: 関数のキーワード引数

        Returns:
            Any: 関数の実行結果
        """
        if not self.validate_custom_function(func):
            raise SecurityError("カスタム関数のセキュリティ検証に失敗しました")

        try:
            # タイムアウト付きで実行
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("カスタム関数の実行がタイムアウトしました")

            # Windowsではsignalが制限されているため、単純に実行
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
                result = func(*args, **kwargs)
                signal.alarm(0)  # タイマーを無効化
                return result
            except AttributeError:
                # Windows環境ではsignal.SIGALRMが利用できない
                return func(*args, **kwargs)

        except Exception as e:
            logger.error(f"カスタム関数実行エラー: {e}")
            raise

    def get_security_report(self) -> Dict[str, Any]:
        """
        セキュリティ状況のレポートを生成

        Returns:
            Dict[str, Any]: セキュリティレポート
        """
        return {
            'custom_functions_enabled': self.enable_custom_functions,
            'approved_functions_count': len(self.approved_functions),
            'approved_functions': list(self.function_whitelist),
            'allowed_modules': list(self.ALLOWED_MODULES),
            'forbidden_keywords_count': len(self.FORBIDDEN_KEYWORDS),
            'allowed_builtins_count': len(self.ALLOWED_BUILTINS)
        }


class SecurityError(Exception):
    """セキュリティ関連のエラー"""
    pass


# グローバルセキュリティ設定インスタンス
security_config = SecurityConfig()
