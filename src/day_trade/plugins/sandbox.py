"""
セキュリティサンドボックス

プラグインセキュリティ・権限制限・安全な実行環境
"""

import ast
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class SecuritySandbox:
    """プラグインセキュリティサンドボックス"""

    def __init__(self):
        """初期化"""
        # 危険なモジュール・関数
        self.dangerous_modules = {
            'os', 'subprocess', 'sys', 'shutil', 'socket',
            'urllib', 'requests', 'ftplib', 'smtplib',
            'pickle', 'marshal', 'exec', 'eval', 'compile',
            '__import__', 'open', 'file', 'input', 'raw_input'
        }

        # 危険なキーワード
        self.dangerous_keywords = {
            '__import__', 'exec', 'eval', 'compile', 'globals',
            'locals', 'vars', 'dir', 'getattr', 'setattr',
            'delattr', 'hasattr', '__getattribute__', '__setattr__'
        }

        # 許可されたモジュール
        self.allowed_modules = {
            'datetime', 'math', 're', 'json', 'typing',
            'dataclasses', 'enum', 'abc', 'collections',
            'itertools', 'functools', 'warnings',
            'numpy', 'pandas', 'sklearn', 'scipy'
        }

        # プラグイン実行制限
        self.max_memory_mb = 512
        self.max_execution_time = 30
        self.max_file_size_kb = 100

        logger.info("セキュリティサンドボックス初期化完了")

    def validate_plugin_file(self, plugin_file: Path) -> bool:
        """
        プラグインファイルセキュリティ検証

        Args:
            plugin_file: プラグインファイルパス

        Returns:
            検証合格フラグ
        """
        try:
            # ファイルサイズチェック
            if not self._check_file_size(plugin_file):
                return False

            # ファイル内容読み込み
            with open(plugin_file, 'r', encoding='utf-8') as f:
                code = f.read()

            # 構文解析
            if not self._validate_syntax(code):
                return False

            # 危険コード検出
            if not self._check_dangerous_code(code):
                return False

            # モジュールインポートチェック
            if not self._check_imports(code):
                return False

            logger.info(f"プラグインセキュリティ検証合格: {plugin_file.name}")
            return True

        except Exception as e:
            logger.error(f"セキュリティ検証エラー {plugin_file.name}: {e}")
            return False

    def _check_file_size(self, plugin_file: Path) -> bool:
        """ファイルサイズチェック"""
        try:
            file_size_kb = plugin_file.stat().st_size / 1024

            if file_size_kb > self.max_file_size_kb:
                logger.warning(f"ファイルサイズ超過: {file_size_kb:.1f}KB")
                return False

            return True

        except Exception as e:
            logger.error(f"ファイルサイズチェックエラー: {e}")
            return False

    def _validate_syntax(self, code: str) -> bool:
        """構文検証"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"構文エラー: {e}")
            return False

    def _check_dangerous_code(self, code: str) -> bool:
        """危険コード検出"""
        try:
            # AST解析による検出
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # 危険な関数呼び出し
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_keywords:
                            logger.warning(f"危険な関数呼び出し: {node.func.id}")
                            return False

                # 危険な属性アクセス
                if isinstance(node, ast.Attribute):
                    if node.attr.startswith('__'):
                        logger.warning(f"危険な属性アクセス: {node.attr}")
                        return False

            # 文字列パターンチェック
            dangerous_patterns = [
                r'__[a-zA-Z_]+__',  # マジックメソッド
                r'exec\s*\(',       # exec呼び出し
                r'eval\s*\(',       # eval呼び出し
                r'import\s+os',     # osモジュール
                r'from\s+os',       # osモジュール
                r'subprocess',      # subprocess
                r'system\(',        # system呼び出し
                r'popen\(',         # popen呼び出し
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    logger.warning(f"危険パターン検出: {pattern}")
                    return False

            return True

        except Exception as e:
            logger.error(f"危険コード検出エラー: {e}")
            return False

    def _check_imports(self, code: str) -> bool:
        """インポート検証"""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if not self._is_module_allowed(module_name):
                            logger.warning(f"危険なモジュール: {module_name}")
                            return False

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if not self._is_module_allowed(module_name):
                            logger.warning(f"危険なFromインポート: {module_name}")
                            return False

            return True

        except Exception as e:
            logger.error(f"インポートチェックエラー: {e}")
            return False

    def _is_module_allowed(self, module_name: str) -> bool:
        """モジュール許可チェック"""
        # 危険モジュール
        if module_name in self.dangerous_modules:
            return False

        # 許可モジュール
        if module_name in self.allowed_modules:
            return True

        # プロジェクト内モジュール
        if module_name.startswith('day_trade') or module_name.startswith('..'):
            return True

        # その他標準ライブラリ（一部許可）
        safe_stdlib_modules = {
            'collections', 'itertools', 'functools', 'operator',
            'copy', 'weakref', 'types', 'inspect', 'importlib',
            'statistics', 'random', 'decimal', 'fractions',
            'logging', 'traceback', 'time', 'calendar'
        }

        if module_name in safe_stdlib_modules:
            return True

        logger.warning(f"未許可モジュール: {module_name}")
        return False

    def create_execution_environment(self, plugin_name: str) -> Dict[str, Any]:
        """
        安全な実行環境作成

        Args:
            plugin_name: プラグイン名

        Returns:
            制限された実行環境
        """
        try:
            # 基本的な組み込み関数のみ許可
            safe_builtins = {
                'abs', 'all', 'any', 'bool', 'dict', 'enumerate',
                'filter', 'float', 'int', 'isinstance', 'len',
                'list', 'map', 'max', 'min', 'range', 'round',
                'set', 'sorted', 'str', 'sum', 'tuple', 'type',
                'zip', 'ValueError', 'TypeError', 'KeyError',
                'IndexError', 'AttributeError'
            }

            # 制限された環境作成
            restricted_env = {
                '__builtins__': {name: getattr(__builtins__, name)
                                for name in safe_builtins if hasattr(__builtins__, name)},
                '__name__': f'plugin_{plugin_name}',
                '__file__': f'<plugin_{plugin_name}>',
            }

            # プラグイン専用ディレクトリ作成
            plugin_dir = Path(f"temp/plugins/{plugin_name}")
            plugin_dir.mkdir(parents=True, exist_ok=True)

            restricted_env['__plugin_dir__'] = str(plugin_dir)

            return restricted_env

        except Exception as e:
            logger.error(f"実行環境作成エラー: {e}")
            return {}

    def monitor_plugin_execution(self, plugin_name: str) -> 'PluginMonitor':
        """
        プラグイン実行監視開始

        Args:
            plugin_name: プラグイン名

        Returns:
            監視オブジェクト
        """
        return PluginMonitor(
            plugin_name=plugin_name,
            max_memory_mb=self.max_memory_mb,
            max_execution_time=self.max_execution_time
        )

    def calculate_file_hash(self, plugin_file: Path) -> str:
        """
        プラグインファイルハッシュ計算

        Args:
            plugin_file: プラグインファイル

        Returns:
            SHA256ハッシュ
        """
        try:
            with open(plugin_file, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"ハッシュ計算エラー: {e}")
            return ""

    def validate_plugin_signature(
        self,
        plugin_file: Path,
        expected_hash: Optional[str] = None
    ) -> bool:
        """
        プラグイン署名検証

        Args:
            plugin_file: プラグインファイル
            expected_hash: 期待ハッシュ

        Returns:
            検証合格フラグ
        """
        try:
            if not expected_hash:
                logger.info("ハッシュ未指定 - 検証スキップ")
                return True

            actual_hash = self.calculate_file_hash(plugin_file)

            if actual_hash == expected_hash:
                logger.info("プラグイン署名検証成功")
                return True
            else:
                logger.error("プラグイン署名検証失敗")
                return False

        except Exception as e:
            logger.error(f"署名検証エラー: {e}")
            return False


class PluginMonitor:
    """プラグイン実行監視"""

    def __init__(
        self,
        plugin_name: str,
        max_memory_mb: int = 512,
        max_execution_time: int = 30
    ):
        """
        初期化

        Args:
            plugin_name: プラグイン名
            max_memory_mb: 最大メモリ使用量(MB)
            max_execution_time: 最大実行時間(秒)
        """
        self.plugin_name = plugin_name
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time

        self.start_time = None
        self.peak_memory_mb = 0
        self.violations = []

    def __enter__(self):
        """監視開始"""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """監視終了"""
        import time

        if self.start_time:
            execution_time = time.time() - self.start_time

            # 実行時間チェック
            if execution_time > self.max_execution_time:
                self.violations.append(f"実行時間超過: {execution_time:.1f}秒")

            # 結果ログ出力
            if self.violations:
                logger.warning(f"プラグイン違反 {self.plugin_name}: {self.violations}")
            else:
                logger.debug(f"プラグイン実行完了 {self.plugin_name}: {execution_time:.2f}秒")

    def check_memory_usage(self):
        """メモリ使用量チェック"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

            if memory_mb > self.max_memory_mb:
                self.violations.append(f"メモリ使用量超過: {memory_mb:.1f}MB")
                return False

            return True

        except ImportError:
            # psutil未インストール時はスキップ
            return True
        except Exception as e:
            logger.error(f"メモリチェックエラー: {e}")
            return True

    def get_violations(self) -> List[str]:
        """違反リスト取得"""
        return self.violations.copy()
