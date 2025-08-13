#!/usr/bin/env python3
"""
テストカバレッジ計測・レポート生成スクリプト
Issue 183: テストカバレッジの計測と可視化

このスクリプトは包括的なテストカバレッジ分析を実行し、
詳細なレポートを生成します。
"""

import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_command(command: List[str], cwd: Optional[str] = None) -> tuple[int, str, str]:
    """コマンドを実行して結果を返す"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5分のタイムアウト
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def parse_coverage_xml(xml_file: Path) -> Dict[str, Any]:
    """coverage.xml ファイルを解析してカバレッジデータを抽出"""
    try:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        coverage_data = {
            "timestamp": datetime.now().isoformat(),
            "overall": {},
            "packages": {},
            "classes": {},
        }

        # 全体のカバレッジ
        overall = root.attrib
        coverage_data["overall"] = {
            "line_rate": float(overall.get("line-rate", 0)),
            "branch_rate": float(overall.get("branch-rate", 0)),
            "lines_covered": int(overall.get("lines-covered", 0)),
            "lines_valid": int(overall.get("lines-valid", 0)),
            "branches_covered": int(overall.get("branches-covered", 0)),
            "branches_valid": int(overall.get("branches-valid", 0)),
        }

        # パッケージ別カバレッジ
        for package in root.findall(".//package"):
            package_name = package.get("name", "unknown")
            coverage_data["packages"][package_name] = {
                "line_rate": float(package.get("line-rate", 0)),
                "branch_rate": float(package.get("branch-rate", 0)),
                "classes": {},
            }

            # クラス別カバレッジ
            for cls in package.findall(".//class"):
                class_name = cls.get("name", "unknown")
                filename = cls.get("filename", "")

                class_data = {
                    "filename": filename,
                    "line_rate": float(cls.get("line-rate", 0)),
                    "branch_rate": float(cls.get("branch-rate", 0)),
                    "lines": [],
                }

                # 行別カバレッジ
                for line in cls.findall(".//line"):
                    line_data = {
                        "number": int(line.get("number", 0)),
                        "hits": int(line.get("hits", 0)),
                        "branch": line.get("branch") == "true",
                        "condition_coverage": line.get("condition-coverage", ""),
                    }
                    class_data["lines"].append(line_data)

                coverage_data["packages"][package_name]["classes"][
                    class_name
                ] = class_data
                coverage_data["classes"][class_name] = class_data

        return coverage_data

    except ElementTree.ParseError as e:
        print(f"XMLパースエラー: {e}")
        return {}
    except Exception as e:
        print(f"カバレッジデータ解析エラー: {e}")
        return {}


def generate_coverage_report() -> Dict[str, Any]:
    """テストカバレッジを実行してレポートを生成"""
    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports" / "coverage"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("テストカバレッジ測定を開始...")

    # カバレッジ測定の実行
    coverage_commands = [
        # 既存の .coverage ファイルを削除
        ["coverage", "erase"],
        # テスト実行（カバレッジ測定付き）
        [
            "coverage",
            "run",
            "--source",
            "src/day_trade",
            "--omit",
            "*/test_*,*/tests/*,*/__pycache__/*",
            "-m",
            "pytest",
            "tests/",
            "--tb=short",
            "-q",
        ],
        # テキストレポート生成
        ["coverage", "report", "--show-missing"],
        # HTMLレポート生成
        ["coverage", "html", "-d", str(reports_dir / f"html_{timestamp}")],
        # XMLレポート生成
        ["coverage", "xml", "-o", str(reports_dir / f"coverage_{timestamp}.xml")],
        # JSON レポート生成
        ["coverage", "json", "-o", str(reports_dir / f"coverage_{timestamp}.json")],
    ]

    results = {
        "timestamp": timestamp,
        "commands": [],
        "files_generated": [],
        "coverage_data": None,
        "summary": {},
    }

    for i, command in enumerate(coverage_commands):
        print(f"実行中 ({i + 1}/{len(coverage_commands)}): {' '.join(command)}")

        returncode, stdout, stderr = run_command(command, cwd=str(project_root))

        command_result = {
            "command": " ".join(command),
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        results["commands"].append(command_result)

        if returncode != 0:
            print(f"警告: コマンドが失敗しました: {' '.join(command)}")
            print(f"標準エラー: {stderr}")
        else:
            print("完了")

    # 生成されたファイルの確認
    generated_files = [
        reports_dir / f"html_{timestamp}" / "index.html",
        reports_dir / f"coverage_{timestamp}.xml",
        reports_dir / f"coverage_{timestamp}.json",
    ]

    for file_path in generated_files:
        if file_path.exists():
            results["files_generated"].append(str(file_path))
            print(f"生成されたファイル: {file_path}")

    # XMLファイルからカバレッジデータを解析
    xml_file = reports_dir / f"coverage_{timestamp}.xml"
    if xml_file.exists():
        coverage_data = parse_coverage_xml(xml_file)
        if coverage_data:
            results["coverage_data"] = coverage_data

            # サマリー情報の生成
            overall = coverage_data.get("overall", {})
            results["summary"] = {
                "line_coverage": f"{overall.get('line_rate', 0) * 100:.1f}%",
                "branch_coverage": f"{overall.get('branch_rate', 0) * 100:.1f}%",
                "lines_covered": overall.get("lines_covered", 0),
                "lines_total": overall.get("lines_valid", 0),
                "packages_count": len(coverage_data.get("packages", {})),
                "classes_count": len(coverage_data.get("classes", {})),
            }

    # JSONファイルからより詳細なデータを取得
    json_file = reports_dir / f"coverage_{timestamp}.json"
    if json_file.exists():
        try:
            with open(json_file) as f:
                json_data = json.load(f)

            # ファイル別カバレッジデータを追加
            files_data = {}
            for filename, file_data in json_data.get("files", {}).items():
                summary = file_data.get("summary", {})
                files_data[filename] = {
                    "covered_lines": summary.get("covered_lines", 0),
                    "missing_lines": summary.get("missing_lines", 0),
                    "num_statements": summary.get("num_statements", 0),
                    "percent_covered": summary.get("percent_covered", 0),
                    "excluded_lines": summary.get("excluded_lines", 0),
                }

            results["files_coverage"] = files_data

        except Exception as e:
            print(f"JSONデータの読み込みエラー: {e}")

    return results


def analyze_coverage_trends(reports_dir: Path) -> Dict[str, Any]:
    """過去のカバレッジデータを分析してトレンドを算出"""
    trends = {"history": [], "trend_analysis": {}, "recommendations": []}

    # 過去のJSONファイルを検索
    json_files = list(reports_dir.glob("coverage_*.json"))
    json_files.sort()  # 日時順にソート

    if len(json_files) < 2:
        return trends

    # 最新の5つのレポートを分析
    recent_files = json_files[-5:]

    for json_file in recent_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            totals = data.get("totals", {})
            history_point = {
                "timestamp": json_file.stem.replace("coverage_", ""),
                "line_coverage": totals.get("percent_covered", 0),
                "num_statements": totals.get("num_statements", 0),
                "covered_lines": totals.get("covered_lines", 0),
                "missing_lines": totals.get("missing_lines", 0),
            }
            trends["history"].append(history_point)

        except Exception as e:
            print(f"履歴データ読み込みエラー ({json_file}): {e}")

    # トレンド分析
    if len(trends["history"]) >= 2:
        latest = trends["history"][-1]
        previous = trends["history"][-2]

        coverage_change = latest["line_coverage"] - previous["line_coverage"]
        statements_change = latest["num_statements"] - previous["num_statements"]

        trends["trend_analysis"] = {
            "coverage_change": coverage_change,
            "coverage_direction": (
                "improving"
                if coverage_change > 0
                else "declining" if coverage_change < 0 else "stable"
            ),
            "statements_change": statements_change,
            "code_growth": statements_change > 0,
        }

        # 推奨事項の生成
        if coverage_change < -2:
            trends["recommendations"].append(
                "カバレッジが2%以上低下しています。新しいテストの追加を検討してください。"
            )
        elif coverage_change > 5:
            trends["recommendations"].append(
                "カバレッジが大幅に改善されました。良い傾向です！"
            )

        if statements_change > 100:
            trends["recommendations"].append(
                "大量のコードが追加されました。対応するテストの追加を確認してください。"
            )

    return trends


def generate_detailed_report(
    coverage_results: Dict[str, Any], trends: Dict[str, Any]
) -> str:
    """詳細なマークダウンレポートを生成"""
    timestamp = coverage_results.get("timestamp", "unknown")
    summary = coverage_results.get("summary", {})

    report = f"""# テストカバレッジレポート

## 基本情報
- **生成日時**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}
- **レポートID**: {timestamp}

## 全体サマリー

| 項目 | 値 |
|------|-----|
| ライン カバレッジ | {summary.get("line_coverage", "N/A")} |
| ブランチ カバレッジ | {summary.get("branch_coverage", "N/A")} |
| カバー済み行数 | {summary.get("lines_covered", "N/A")} |
| 総行数 | {summary.get("lines_total", "N/A")} |
| パッケージ数 | {summary.get("packages_count", "N/A")} |
| クラス数 | {summary.get("classes_count", "N/A")} |

"""

    # カバレッジ目標との比較
    line_coverage = float(summary.get("line_coverage", "0").rstrip("%"))

    report += "## カバレッジ評価\n\n"

    if line_coverage >= 80:
        report += "🟢 **優秀**: カバレッジが80%以上です。\n"
    elif line_coverage >= 70:
        report += "🟡 **良好**: カバレッジが70%以上です。\n"
    elif line_coverage >= 60:
        report += (
            "🟠 **改善要**: カバレッジが60%以上ですが、さらなる改善が推奨されます。\n"
        )
    else:
        report += "🔴 **要改善**: カバレッジが60%未満です。テストの追加が必要です。\n"

    # トレンド分析
    if trends.get("trend_analysis"):
        trend = trends["trend_analysis"]
        report += "\n## トレンド分析\n\n"

        direction_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}

        emoji = direction_emoji.get(trend.get("coverage_direction", "stable"), "➡️")
        change = trend.get("coverage_change", 0)

        report += f"{emoji} **カバレッジ変化**: {change:+.2f}%\n"
        report += f"📊 **コード変化**: {trend.get('statements_change', 0):+d} 行\n\n"

        if trend.get("coverage_direction") == "improving":
            report += "✅ カバレッジが改善されています。\n"
        elif trend.get("coverage_direction") == "declining":
            report += "⚠️ カバレッジが低下しています。\n"

    # 推奨事項
    recommendations = trends.get("recommendations", [])
    if recommendations:
        report += "\n## 推奨事項\n\n"
        for rec in recommendations:
            report += f"- {rec}\n"

    # ファイル別カバレッジ（上位10件と下位10件）
    files_coverage = coverage_results.get("files_coverage", {})
    if files_coverage:
        # カバレッジ率でソート
        sorted_files = sorted(
            files_coverage.items(),
            key=lambda x: x[1].get("percent_covered", 0),
            reverse=True,
        )

        report += "\n## ファイル別カバレッジ（上位10件）\n\n"
        report += "| ファイル | カバレッジ | カバー済み/総行数 |\n"
        report += "|----------|------------|-------------------|\n"

        for filename, data in sorted_files[:10]:
            short_filename = filename.replace("src\\day_trade\\", "").replace(
                "src/day_trade/", ""
            )
            coverage_pct = data.get("percent_covered", 0)
            covered = data.get("covered_lines", 0)
            total = data.get("num_statements", 0)

            report += (
                f"| {short_filename} | {coverage_pct:.1f}% | {covered}/{total} |\n"
            )

        report += "\n## ファイル別カバレッジ（下位10件）\n\n"
        report += "| ファイル | カバレッジ | カバー済み/総行数 |\n"
        report += "|----------|------------|-------------------|\n"

        for filename, data in sorted_files[-10:]:
            short_filename = filename.replace("src\\day_trade\\", "").replace(
                "src/day_trade/", ""
            )
            coverage_pct = data.get("percent_covered", 0)
            covered = data.get("covered_lines", 0)
            total = data.get("num_statements", 0)

            report += (
                f"| {short_filename} | {coverage_pct:.1f}% | {covered}/{total} |\n"
            )

    # 生成されたファイル
    files_generated = coverage_results.get("files_generated", [])
    if files_generated:
        report += "\n## 生成されたレポートファイル\n\n"
        for file_path in files_generated:
            rel_path = Path(file_path).relative_to(Path.cwd())
            report += f"- [{rel_path}]({rel_path})\n"

    # フッター
    report += "\n---\n*このレポートは scripts/coverage_report.py により自動生成されました。*\n"

    return report


def main():
    """メイン実行関数"""
    print("Day Trade テストカバレッジ分析")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    reports_dir = project_root / "reports" / "coverage"

    # カバレッジ測定とレポート生成
    print("\n1. カバレッジ測定実行中...")
    coverage_results = generate_coverage_report()

    # トレンド分析
    print("\n2. トレンド分析実行中...")
    trends = analyze_coverage_trends(reports_dir)

    # 詳細レポート生成
    print("\n3. 詳細レポート生成中...")
    detailed_report = generate_detailed_report(coverage_results, trends)

    # レポートファイルの保存
    timestamp = coverage_results.get(
        "timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    report_file = reports_dir / f"coverage_report_{timestamp}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(detailed_report)

    print(f"\n詳細レポートが保存されました: {report_file}")

    # 結果サマリーの表示
    summary = coverage_results.get("summary", {})
    print("\n" + "=" * 50)
    print("カバレッジサマリー:")
    print(f"ライン カバレッジ: {summary.get('line_coverage', 'N/A')}")
    print(f"総行数: {summary.get('lines_total', 'N/A')}")
    print(f"カバー済み: {summary.get('lines_covered', 'N/A')}")

    # トレンド情報
    if trends.get("trend_analysis"):
        trend = trends["trend_analysis"]
        change = trend.get("coverage_change", 0)
        direction = trend.get("coverage_direction", "stable")

        print(f"前回比: {change:+.2f}% ({direction})")

    print("\n詳細は以下のファイルで確認できます:")
    for file_path in coverage_results.get("files_generated", []):
        print(f"  - {file_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
