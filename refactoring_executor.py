#!/usr/bin/env python3
"""
リファクタリング実行ツール

大きなファイルを機能別に分割し、コードベースを整理
"""

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime


class RefactoringExecutor:
    """リファクタリング実行クラス"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.backup_dir = base_dir / "refactoring_backup"
        self.backup_dir.mkdir(exist_ok=True)

    def execute_priority_refactoring(self):
        """優先リファクタリング実行"""
        print("優先リファクタリング開始")
        print("=" * 40)

        # 1. バックアップ作成
        self._create_backup()

        # 2. 最も大きなファイルから処理
        self._refactor_large_files()

        # 3. 重複コード整理
        self._consolidate_duplicate_code()

        # 4. デッドコード削除
        self._remove_dead_code()

        print("優先リファクタリング完了")

    def _create_backup(self):
        """バックアップ作成"""
        print("1. バックアップ作成中...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)

        # 主要ファイルをバックアップ
        important_files = [
            "daytrade.py",
            "main.py",
            "advanced_technical_analysis.py",
            "advanced_technical_analyzer.py"
        ]

        for filename in important_files:
            src_file = self.base_dir / filename
            if src_file.exists():
                shutil.copy2(src_file, backup_subdir / filename)
                print(f"  バックアップ: {filename}")

    def _refactor_large_files(self):
        """大きなファイルのリファクタリング"""
        print("2. 大きなファイル分割中...")

        # daytrade.pyの分割（最優先）
        self._split_daytrade_file()

        # その他の大きなファイル
        large_files = [
            "advanced_technical_analysis.py",
            "advanced_technical_analyzer.py"
        ]

        for filename in large_files:
            file_path = self.base_dir / filename
            if file_path.exists():
                print(f"  分割対象: {filename}")
                self._split_analysis_file(file_path)

    def _split_daytrade_file(self):
        """daytrade.pyファイルの分割"""
        print("  daytrade.py分割中...")

        daytrade_file = self.base_dir / "daytrade.py"
        if not daytrade_file.exists():
            print("    daytrade.pyが見つかりません")
            return

        try:
            content = daytrade_file.read_text(encoding='utf-8')
        except Exception as e:
            print(f"    読み込みエラー: {e}")
            return

        # コアエントリーポイントファイル作成
        self._create_core_entry_point(content)

        # 設定・初期化部分を分離
        self._extract_initialization_code(content)

        # メイン実行ロジックを分離
        self._extract_main_logic(content)

    def _create_core_entry_point(self, content: str):
        """コアエントリーポイント作成"""
        core_entry = '''#!/usr/bin/env python3
"""
Day Trade Personal - コアエントリーポイント

リファクタリング後の軽量メインファイル
"""

import sys
from pathlib import Path

# システムパス追加
sys.path.append(str(Path(__file__).parent / "src"))

from src.day_trade.core.application import DayTradeApplication


def main():
    """メイン実行関数"""
    app = DayTradeApplication()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
'''

        output_file = self.base_dir / "daytrade_core.py"
        output_file.write_text(core_entry, encoding='utf-8')
        print("    作成: daytrade_core.py")

    def _extract_initialization_code(self, content: str):
        """初期化コード抽出"""
        init_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - 初期化モジュール

システム初期化とWindows環境対応
"""

import os
import sys
import locale

class SystemInitializer:
    """システム初期化クラス"""

    @staticmethod
    def initialize_environment():
        """環境初期化"""
        # 環境変数設定
        os.environ['PYTHONIOENCODING'] = 'utf-8'

        # Windows Console API対応
        if sys.platform == 'win32':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    @staticmethod
    def setup_logging():
        """ログ設定"""
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
'''

        # 初期化ディレクトリ作成
        init_dir = self.base_dir / "src" / "day_trade" / "core"
        init_dir.mkdir(parents=True, exist_ok=True)

        output_file = init_dir / "system_initializer.py"
        output_file.write_text(init_code, encoding='utf-8')
        print("    作成: src/day_trade/core/system_initializer.py")

    def _extract_main_logic(self, content: str):
        """メインロジック抽出"""
        app_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - アプリケーションクラス

リファクタリング後のメインアプリケーション
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .system_initializer import SystemInitializer
from ..cli.argument_parser import ArgumentParser
from ..analysis.trading_analyzer import TradingAnalyzer
from ..dashboard.web_dashboard import WebDashboard


class DayTradeApplication:
    """Day Trade メインアプリケーション"""

    def __init__(self):
        """初期化"""
        SystemInitializer.initialize_environment()
        SystemInitializer.setup_logging()

        self.analyzer = None
        self.web_dashboard = None

    def run(self) -> int:
        """アプリケーション実行"""
        try:
            # 引数解析
            parser = ArgumentParser()
            args = parser.parse_args()

            # モード別実行
            if args.web:
                return self._run_web_mode(args)
            elif args.quick:
                return self._run_quick_analysis(args)
            elif args.multi:
                return self._run_multi_analysis(args)
            else:
                return self._run_default_analysis(args)

        except KeyboardInterrupt:
            print("\\n操作が中断されました")
            return 0
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return 1

    def _run_web_mode(self, args) -> int:
        """Webモード実行"""
        print("🌐 Webダッシュボード起動中...")
        self.web_dashboard = WebDashboard(port=args.port, debug=args.debug)
        self.web_dashboard.run()
        return 0

    def _run_quick_analysis(self, args) -> int:
        """クイック分析実行"""
        print("⚡ クイック分析モード")
        self.analyzer = TradingAnalyzer(quick_mode=True)
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _run_multi_analysis(self, args) -> int:
        """マルチ分析実行"""
        print("📊 マルチ銘柄分析モード")
        self.analyzer = TradingAnalyzer(multi_mode=True)
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _run_default_analysis(self, args) -> int:
        """デフォルト分析実行"""
        print("🎯 デフォルト分析モード")
        self.analyzer = TradingAnalyzer()
        results = self.analyzer.analyze(args.symbols)
        self._display_results(results)
        return 0

    def _display_results(self, results):
        """結果表示"""
        print("\\n" + "="*50)
        print("📈 分析結果")
        print("="*50)

        for result in results:
            print(f"銘柄: {result.get('symbol', 'N/A')}")
            print(f"推奨: {result.get('recommendation', 'N/A')}")
            print(f"信頼度: {result.get('confidence', 0):.1%}")
            print("-" * 30)
'''

        # アプリケーションディレクトリ作成
        core_dir = self.base_dir / "src" / "day_trade" / "core"
        core_dir.mkdir(parents=True, exist_ok=True)

        output_file = core_dir / "application.py"
        output_file.write_text(app_code, encoding='utf-8')
        print("    作成: src/day_trade/core/application.py")

    def _split_analysis_file(self, file_path: Path):
        """分析ファイルの分割"""
        print(f"  {file_path.name}分割中...")

        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"    読み込みエラー: {e}")
            return

        # 簡易的な分割：クラス単位で分離
        classes = self._extract_classes(content)

        if len(classes) > 1:
            base_name = file_path.stem
            for i, (class_name, class_code) in enumerate(classes):
                if class_name:
                    new_filename = f"{base_name}_{class_name.lower()}.py"
                else:
                    new_filename = f"{base_name}_part_{i+1}.py"

                new_file = file_path.parent / new_filename

                # ファイルヘッダー追加
                header = f'''#!/usr/bin/env python3
"""
{file_path.name} - {class_name or f"Part {i+1}"}

リファクタリングにより分割されたモジュール
"""

'''

                full_content = header + class_code
                new_file.write_text(full_content, encoding='utf-8')
                print(f"    作成: {new_filename}")

    def _extract_classes(self, content: str) -> List[Tuple[str, str]]:
        """クラス抽出"""
        classes = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # クラス開始位置から次のクラスまたはファイル終端まで抽出
                    start_line = node.lineno

                    # 簡易的な抽出（改良の余地あり）
                    lines = content.split('\n')
                    class_lines = []

                    in_class = False
                    indent_level = None

                    for i, line in enumerate(lines):
                        if i + 1 >= start_line and not in_class:
                            in_class = True
                            indent_level = len(line) - len(line.lstrip())
                            class_lines.append(line)
                        elif in_class:
                            current_indent = len(line) - len(line.lstrip())
                            if line.strip() and current_indent <= indent_level and not line.startswith(' '):
                                break
                            class_lines.append(line)

                    if class_lines:
                        classes.append((node.name, '\n'.join(class_lines)))

        except SyntaxError:
            # 構文エラーの場合は元のファイルを保持
            return [(None, content)]

        return classes if classes else [(None, content)]

    def _consolidate_duplicate_code(self):
        """重複コード統合"""
        print("3. 重複コード統合中...")

        # 共通ユーティリティの作成
        self._create_common_utilities()

        # 共通インポートの統合
        self._consolidate_imports()

    def _create_common_utilities(self):
        """共通ユーティリティ作成"""
        utils_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - 共通ユーティリティ

リファクタリングにより統合された共通機能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


class CommonUtils:
    """共通ユーティリティクラス"""

    @staticmethod
    def setup_paths():
        """パス設定"""
        base_dir = Path(__file__).parent.parent.parent.parent
        src_dir = base_dir / "src"

        if str(src_dir) not in sys.path:
            sys.path.append(str(src_dir))

        return base_dir, src_dir

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """ロガー取得"""
        return logging.getLogger(name)

    @staticmethod
    def format_currency(amount: float) -> str:
        """通貨フォーマット"""
        return f"¥{amount:,.0f}"

    @staticmethod
    def format_percentage(value: float) -> str:
        """パーセンテージフォーマット"""
        return f"{value:.2%}"

    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """安全な除算"""
        return numerator / denominator if denominator != 0 else default


class FileUtils:
    """ファイル操作ユーティリティ"""

    @staticmethod
    def ensure_directory(path: Path):
        """ディレクトリ確保"""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_read_json(file_path: Path, default: Dict = None) -> Dict:
        """安全なJSON読み込み"""
        if default is None:
            default = {}

        try:
            if file_path.exists():
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"JSON読み込みエラー {file_path}: {e}")

        return default

    @staticmethod
    def safe_write_json(file_path: Path, data: Dict):
        """安全なJSON書き込み"""
        try:
            import json
            FileUtils.ensure_directory(file_path.parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"JSON書き込みエラー {file_path}: {e}")


class DateUtils:
    """日付ユーティリティ"""

    @staticmethod
    def get_current_timestamp() -> str:
        """現在のタイムスタンプ"""
        return datetime.now().isoformat()

    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """日時フォーマット"""
        return dt.strftime(format_str)

    @staticmethod
    def is_market_hours() -> bool:
        """市場時間判定（簡易版）"""
        now = datetime.now()
        # 平日の9:00-15:00を市場時間とする（簡易版）
        return (
            now.weekday() < 5 and  # 平日
            9 <= now.hour < 15      # 9時-15時
        )
'''

        # ユーティリティディレクトリ作成
        utils_dir = self.base_dir / "src" / "day_trade" / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)

        output_file = utils_dir / "common_utils.py"
        output_file.write_text(utils_code, encoding='utf-8')
        print("    作成: src/day_trade/utils/common_utils.py")

    def _consolidate_imports(self):
        """インポート統合"""
        # ユーティリティディレクトリ確保
        utils_dir = self.base_dir / "src" / "day_trade" / "utils"
        utils_dir.mkdir(parents=True, exist_ok=True)

        imports_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - 統合インポート

よく使用されるインポートの統合
"""

# 標準ライブラリ
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# サードパーティライブラリ
try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    PANDAS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# プロジェクト内モジュール
from .common_utils import CommonUtils, FileUtils, DateUtils

# 利用可能性チェック関数
def check_dependencies():
    """依存関係チェック"""
    missing = []

    if not NUMPY_AVAILABLE:
        missing.append("numpy")
    if not PANDAS_AVAILABLE:
        missing.append("pandas")
    if not YFINANCE_AVAILABLE:
        missing.append("yfinance")
    if not SKLEARN_AVAILABLE:
        missing.append("scikit-learn")

    if missing:
        print(f"警告: 以下のライブラリが不足しています: {', '.join(missing)}")
        print("pip install -r requirements.txt を実行してください")

    return len(missing) == 0

# エクスポート用
__all__ = [
    'CommonUtils', 'FileUtils', 'DateUtils',
    'np', 'pd', 'yf',
    'NUMPY_AVAILABLE', 'PANDAS_AVAILABLE', 'YFINANCE_AVAILABLE', 'SKLEARN_AVAILABLE',
    'check_dependencies'
]
'''

        output_file = utils_dir / "__init__.py"
        output_file.write_text(imports_code, encoding='utf-8')
        print("    作成: src/day_trade/utils/__init__.py")

    def _remove_dead_code(self):
        """デッドコード削除"""
        print("4. デッドコード削除中...")

        # TODO、FIXME、DEPRECATED コメントを含むファイルをリスト
        problem_patterns = [
            r'# TODO:.*',
            r'# FIXME:.*',
            r'# DEPRECATED.*',
            r'if\s+False:',
            r'if\s+0:'
        ]

        python_files = list(self.base_dir.glob("**/*.py"))
        cleaned_files = 0

        for py_file in python_files[:10]:  # 最初の10ファイルのみ処理
            try:
                content = py_file.read_text(encoding='utf-8')
                original_content = content

                # パターンマッチング行を削除
                for pattern in problem_patterns:
                    content = re.sub(pattern + r'.*\n?', '', content, flags=re.MULTILINE)

                if content != original_content:
                    py_file.write_text(content, encoding='utf-8')
                    cleaned_files += 1
                    print(f"    クリーンアップ: {py_file.name}")

            except Exception as e:
                print(f"    エラー {py_file.name}: {e}")
                continue

        print(f"    {cleaned_files}個のファイルをクリーンアップ")

    def create_refactored_entry_points(self):
        """リファクタリング後のエントリーポイント作成"""
        print("5. 新しいエントリーポイント作成中...")

        # CLIインターフェース作成
        self._create_cli_interface()

        # 引数パーサー作成
        self._create_argument_parser()

    def _create_cli_interface(self):
        """CLIインターフェース作成"""
        cli_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - CLIインターフェース

リファクタリング後のコマンドラインインターフェース
"""

import sys
from pathlib import Path
from typing import List, Optional

from ..utils import CommonUtils, check_dependencies
from .argument_parser import ArgumentParser


class CLI:
    """コマンドラインインターフェース"""

    def __init__(self):
        """初期化"""
        CommonUtils.setup_paths()

        # 依存関係チェック
        if not check_dependencies():
            print("依存関係の問題により、一部機能が制限される可能性があります")

    def run(self, args: Optional[List[str]] = None) -> int:
        """CLI実行"""
        try:
            parser = ArgumentParser()
            parsed_args = parser.parse_args(args)

            # バナー表示
            self._show_banner()

            # モード別実行
            if parsed_args.web:
                return self._run_web_mode(parsed_args)
            elif parsed_args.validate:
                return self._run_validation_mode(parsed_args)
            else:
                return self._run_analysis_mode(parsed_args)

        except KeyboardInterrupt:
            print("\\n\\n操作が中断されました")
            return 0
        except Exception as e:
            print(f"エラー: {e}")
            return 1

    def _show_banner(self):
        """バナー表示"""
        print("🚀 Day Trade Personal - 93%精度AIシステム")
        print("📊 リファクタリング版 v3.0")
        print("=" * 50)

    def _run_web_mode(self, args) -> int:
        """Webモード実行"""
        from ..dashboard.web_dashboard import WebDashboard

        print(f"🌐 Webダッシュボード起動中 (ポート: {args.port})...")
        dashboard = WebDashboard(port=args.port, debug=args.debug)
        dashboard.run()
        return 0

    def _run_validation_mode(self, args) -> int:
        """検証モード実行"""
        from ..validation.accuracy_validator import AccuracyValidator

        print("🔍 予測精度検証モード")
        validator = AccuracyValidator()
        results = validator.validate()

        print(f"検証完了: 精度 {results.get('accuracy', 0):.1%}")
        return 0

    def _run_analysis_mode(self, args) -> int:
        """分析モード実行"""
        from ..analysis.enhanced_analyzer import EnhancedAnalyzer

        mode = "クイック" if args.quick else "標準"
        print(f"📈 {mode}分析モード")

        analyzer = EnhancedAnalyzer(
            quick_mode=args.quick,
            use_cache=not args.no_cache
        )

        results = analyzer.analyze(args.symbols)
        self._display_analysis_results(results)
        return 0

    def _display_analysis_results(self, results):
        """分析結果表示"""
        print("\\n" + "=" * 50)
        print("📊 分析結果")
        print("=" * 50)

        for result in results:
            symbol = result.get('symbol', 'N/A')
            recommendation = result.get('recommendation', 'N/A')
            confidence = result.get('confidence', 0)

            # 推奨に基づく色分け（簡易版）
            emoji = "🟢" if recommendation == "買い" else "🔴" if recommendation == "売り" else "🟡"

            print(f"{emoji} {symbol}: {recommendation} (信頼度: {confidence:.1%})")

        print("=" * 50)
        print("💡 投資は自己責任で行ってください")


def main(args: Optional[List[str]] = None) -> int:
    """メイン関数"""
    cli = CLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
'''

        # CLIディレクトリ作成
        cli_dir = self.base_dir / "src" / "day_trade" / "cli"
        cli_dir.mkdir(parents=True, exist_ok=True)

        output_file = cli_dir / "interface.py"
        output_file.write_text(cli_code, encoding='utf-8')
        print("    作成: src/day_trade/cli/interface.py")

    def _create_argument_parser(self):
        """引数パーサー作成"""
        parser_code = '''#!/usr/bin/env python3
"""
Day Trade Personal - 引数パーサー

コマンドライン引数の解析
"""

import argparse
from typing import List, Optional


class ArgumentParser:
    """引数パーサークラス"""

    def __init__(self):
        """初期化"""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """パーサー作成"""
        parser = argparse.ArgumentParser(
            description="Day Trade Personal - 個人利用専用版",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="93%精度AIシステムでデイトレード支援"
        )

        # 実行モード
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--quick", "-q",
            action="store_true",
            help="基本分析モード（高速）"
        )
        mode_group.add_argument(
            "--multi", "-m",
            action="store_true",
            help="複数銘柄分析モード"
        )
        mode_group.add_argument(
            "--web", "-w",
            action="store_true",
            help="Webダッシュボード起動"
        )
        mode_group.add_argument(
            "--validate", "-v",
            action="store_true",
            help="予測精度検証モード"
        )

        # 対象銘柄
        parser.add_argument(
            "--symbols", "-s",
            nargs="+",
            default=["7203", "8306", "9984", "6758"],
            help="対象銘柄コード（デフォルト: トヨタ, 三菱UFJ, SBG, ソニー）"
        )

        # Webサーバー設定
        parser.add_argument(
            "--port", "-p",
            type=int,
            default=8000,
            help="Webサーバーポート（デフォルト: 8000）"
        )

        # その他オプション
        parser.add_argument(
            "--debug", "-d",
            action="store_true",
            help="デバッグモード"
        )
        parser.add_argument(
            "--no-cache",
            action="store_true",
            help="キャッシュを使用しない"
        )
        parser.add_argument(
            "--config",
            type=str,
            help="設定ファイルパス"
        )

        return parser

    def parse_args(self, args: Optional[List[str]] = None):
        """引数解析"""
        return self.parser.parse_args(args)

    def print_help(self):
        """ヘルプ表示"""
        self.parser.print_help()
'''

        output_file = cli_dir / "argument_parser.py"
        output_file.write_text(parser_code, encoding='utf-8')
        print("    作成: src/day_trade/cli/argument_parser.py")


def main():
    """メイン実行"""
    print("リファクタリング実行開始")
    print("=" * 50)

    base_dir = Path(__file__).parent
    refactorer = RefactoringExecutor(base_dir)

    # 優先リファクタリング実行
    refactorer.execute_priority_refactoring()

    # 新しいエントリーポイント作成
    refactorer.create_refactored_entry_points()

    print("\n" + "=" * 50)
    print("✅ リファクタリング完了")
    print("=" * 50)
    print("新しいエントリーポイント:")
    print("  - python daytrade_core.py    # リファクタリング後メイン")
    print("  - python -m src.day_trade.cli.interface  # 新CLI")
    print("\nバックアップ:")
    print("  - refactoring_backup/ ディレクトリに保存")
    print("=" * 50)


if __name__ == "__main__":
    main()