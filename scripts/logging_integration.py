#!/usr/bin/env python3
"""
ロギング統合スクリプト

print文をstructlogベースの構造化ロギングに置き換えるツール。
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List


class PrintToLogConverter:
    """print文をロギングに変換するクラス"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src" / "day_trade"
        self.reports_dir = project_root / "reports" / "logging"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def find_print_statements(self, file_path: Path) -> List[Dict[str, any]]:
        """print文を検出"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return []

        print_statements = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # print文のパターンマッチング
            patterns = [
                r"print\s*\(",  # print(
                r"print\s+",  # print
            ]

            for pattern in patterns:
                if re.search(pattern, line):
                    # コメント行は除外
                    if line.strip().startswith("#"):
                        continue

                    print_statements.append(
                        {
                            "line_number": line_num,
                            "line_content": line.strip(),
                            "file_path": str(file_path),
                            "severity": self._determine_log_level(line),
                        }
                    )

        return print_statements

    def _determine_log_level(self, line: str) -> str:
        """print文の内容からログレベルを推定"""
        line_lower = line.lower()

        if any(word in line_lower for word in ["error", "エラー", "exception", "例外"]):
            return "error"
        elif any(word in line_lower for word in ["warning", "警告", "warn"]):
            return "warning"
        elif any(word in line_lower for word in ["debug", "デバッグ", "trace"]):
            return "debug"
        elif any(
            word in line_lower for word in ["info", "情報", "status", "ステータス"]
        ):
            return "info"
        else:
            return "info"  # デフォルト

    def convert_print_to_log(
        self, file_path: Path, dry_run: bool = True
    ) -> Dict[str, any]:
        """print文をロギングに変換"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return {"success": False, "error": "ファイル読み込みエラー"}

        lines = content.split("\n")
        modified_lines = []
        changes_made = 0
        logging_imported = False

        # ロギングのインポートをチェック
        for line in lines:
            if (
                "from day_trade.utils.logging_config import" in line
                or "import logging" in line
            ):
                logging_imported = True
                break

        for _line_num, line in enumerate(lines):
            # print文を検出して変換
            if re.search(r"print\s*\(", line) and not line.strip().startswith("#"):
                # ログレベルを決定
                log_level = self._determine_log_level(line)

                # print文をロギングに変換
                converted_line = self._convert_line_to_logging(line, log_level)
                modified_lines.append(converted_line)
                changes_made += 1
            else:
                modified_lines.append(line)

        # ロギングのインポートが必要な場合は追加
        if changes_made > 0 and not logging_imported:
            # インポート文を挿入する位置を特定
            import_index = 0
            for i, line in enumerate(modified_lines):
                if line.strip().startswith("import ") or line.strip().startswith(
                    "from "
                ):
                    import_index = i + 1
                elif line.strip() == "" and import_index > 0:
                    break

            # ロギングインポートを追加
            logging_import = "from day_trade.utils.logging_config import get_logger"
            logger_init = "\nlogger = get_logger(__name__)"

            modified_lines.insert(import_index, logging_import)
            modified_lines.insert(import_index + 1, logger_init)

        if not dry_run and changes_made > 0:
            # ファイルに書き戻し
            new_content = "\n".join(modified_lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return {
            "success": True,
            "changes_made": changes_made,
            "logging_imported": logging_imported,
            "file_path": str(file_path),
        }

    def _convert_line_to_logging(self, line: str, log_level: str) -> str:
        """print文をロギング文に変換"""
        # インデントを保持
        indent = re.match(r"^(\s*)", line).group(1)

        # print文の内容を抽出
        print_match = re.search(r"print\s*\((.*)\)", line)
        if not print_match:
            return line

        print_content = print_match.group(1).strip()

        # f-stringやformatメソッドを検出
        if print_content.startswith('f"') or print_content.startswith("f'"):
            # f-string の場合
            converted = f"{indent}logger.{log_level}({print_content})"
        elif ".format(" in print_content:
            # .format() の場合
            converted = f"{indent}logger.{log_level}({print_content})"
        elif "%" in print_content:
            # % formatting の場合
            converted = f"{indent}logger.{log_level}({print_content})"
        else:
            # 通常の文字列の場合
            converted = f"{indent}logger.{log_level}({print_content})"

        return converted

    def analyze_project(self) -> Dict[str, any]:
        """プロジェクト全体のprint文を分析"""
        total_prints = 0
        files_with_prints = 0
        files_analyzed = 0
        file_reports = []

        # Pythonファイルを検索
        python_files = list(self.src_dir.rglob("*.py"))

        for file_path in python_files:
            if file_path.name == "__pycache__":
                continue

            files_analyzed += 1
            print_statements = self.find_print_statements(file_path)

            if print_statements:
                files_with_prints += 1
                total_prints += len(print_statements)

                file_reports.append(
                    {
                        "file_path": str(file_path.relative_to(self.project_root)),
                        "print_count": len(print_statements),
                        "statements": print_statements,
                    }
                )

        return {
            "total_prints": total_prints,
            "files_with_prints": files_with_prints,
            "files_analyzed": files_analyzed,
            "file_reports": file_reports,
        }

    def generate_report(self, analysis_result: Dict) -> str:
        """分析レポートを生成"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"logging_analysis_{timestamp}.md"

        report_content = f"""# 構造化ロギング統合レポート

**生成日時**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}

## 📊 概要

- **分析ファイル数**: {analysis_result["files_analyzed"]}
- **print文を含むファイル数**: {analysis_result["files_with_prints"]}
- **総print文数**: {analysis_result["total_prints"]}

## 📁 ファイル別詳細

"""

        for file_report in analysis_result["file_reports"]:
            report_content += f"""### {file_report["file_path"]}

**print文数**: {file_report["print_count"]}

"""

            for stmt in file_report["statements"]:
                report_content += f"- **行 {stmt['line_number']}** (推奨レベル: {stmt['severity']}): `{stmt['line_content']}`\n"

            report_content += "\n"

        report_content += """
## 🛠️ 推奨アクション

### 1. 自動変換の実行

```bash
# ドライラン（変更は行わず、プレビューのみ）
python scripts/logging_integration.py convert --dry-run

# 実際の変換実行
python scripts/logging_integration.py convert
```

### 2. 手動での対応が必要なケース

- 複雑なprint文
- デバッグ用の一時的なprint文
- ライブラリ固有の出力形式

### 3. ロギング設定の確認

```python
from day_trade.utils.logging_config import setup_logging
setup_logging()
```

### 4. 環境変数の設定

```bash
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export ENVIRONMENT=production
```

## 📝 ベストプラクティス

1. **適切なログレベルの使用**
   - `debug`: 開発時のデバッグ情報
   - `info`: 一般的な情報
   - `warning`: 警告レベルの問題
   - `error`: エラー情報

2. **構造化データの活用**
   ```python
   logger.info("取引実行", symbol="AAPL", quantity=100, price=150.25)
   ```

3. **コンテキスト情報の追加**
   ```python
   logger = get_context_logger(user_id="user123", session_id="sess456")
   logger.info("ユーザーアクション", action="buy_stock")
   ```

---
*このレポートは logging_integration.py により自動生成されました。*
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return str(report_path)

    def batch_convert(self, dry_run: bool = True) -> Dict[str, any]:
        """プロジェクト全体のprint文を一括変換"""
        python_files = list(self.src_dir.rglob("*.py"))

        conversion_results = []
        total_changes = 0

        for file_path in python_files:
            if file_path.name == "__pycache__":
                continue

            result = self.convert_print_to_log(file_path, dry_run)
            if result["success"]:
                conversion_results.append(result)
                total_changes += result["changes_made"]

        return {
            "files_processed": len(conversion_results),
            "total_changes": total_changes,
            "results": conversion_results,
            "dry_run": dry_run,
        }


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="構造化ロギング統合ツール")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="プロジェクトルートディレクトリ",
    )

    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")

    # analyzeコマンド
    subparsers.add_parser("analyze", help="print文を分析")

    # convertコマンド
    convert_parser = subparsers.add_parser("convert", help="print文をロギングに変換")
    convert_parser.add_argument(
        "--dry-run", action="store_true", help="実際には変更せず、変更予定を表示"
    )
    convert_parser.add_argument("--file", type=Path, help="特定のファイルのみ変換")

    # reportコマンド
    subparsers.add_parser("report", help="分析レポートを生成")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    converter = PrintToLogConverter(args.project_root)

    if args.command == "analyze":
        analysis = converter.analyze_project()
        print("分析結果:")
        print(f"  ファイル数: {analysis['files_analyzed']}")
        print(f"  print文を含むファイル数: {analysis['files_with_prints']}")
        print(f"  総print文数: {analysis['total_prints']}")

    elif args.command == "convert":
        if args.file:
            # 特定ファイルの変換
            result = converter.convert_print_to_log(args.file, args.dry_run)
            if result["success"]:
                action = "変更予定" if args.dry_run else "変更完了"
                print(f"{action}: {result['changes_made']}箇所")
            else:
                print(f"エラー: {result.get('error', '不明なエラー')}")
        else:
            # 一括変換
            results = converter.batch_convert(args.dry_run)
            action = "変更予定" if args.dry_run else "変更完了"
            print(f"{action}:")
            print(f"  処理ファイル数: {results['files_processed']}")
            print(f"  総変更箇所: {results['total_changes']}")

    elif args.command == "report":
        analysis = converter.analyze_project()
        report_path = converter.generate_report(analysis)
        print(f"レポートを生成しました: {report_path}")


if __name__ == "__main__":
    main()
