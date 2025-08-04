#!/usr/bin/env python3
"""
テストカバレッジ品質チェッカー
Issue 183: テストカバレッジの計測と可視化

設定された目標と閾値に基づいてカバレッジ品質をチェックし、
CI/CDパイプラインでの品質ゲートとして使用できます。
"""

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_coverage_config(config_file: Path) -> Dict[str, Any]:
    """カバレッジ設定ファイルを読み込み"""
    try:
        with open(config_file, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"設定ファイル読み込みエラー: {e}")
        return {}

def load_coverage_data(json_file: Path) -> Optional[Dict[str, Any]]:
    """カバレッジJSONファイルを読み込み"""
    try:
        with open(json_file, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"カバレッジデータ読み込みエラー: {e}")
        return None

def get_package_name(filename: str) -> str:
    """ファイルパスからパッケージ名を抽出"""
    # パスを正規化
    normalized = filename.replace("src\\day_trade\\", "").replace("src/day_trade/", "")

    # パッケージ名を抽出
    if "/" in normalized:
        return normalized.split("/")[0]
    elif "\\" in normalized:
        return normalized.split("\\")[0]
    else:
        return "root"

def match_patterns(filename: str, patterns: List[str]) -> bool:
    """ファイル名がパターンにマッチするかチェック"""
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)

def evaluate_overall_coverage(coverage_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """全体カバレッジの評価"""
    totals = coverage_data.get("totals", {})
    coverage_pct = totals.get("percent_covered", 0)

    targets = config.get("coverage_targets", {}).get("overall", {})
    quality_gates = config.get("quality_gates", {})

    # レベル判定
    if coverage_pct >= targets.get("ideal", 90):
        level = "ideal"
        status = "excellent"
    elif coverage_pct >= targets.get("excellent", 80):
        level = "excellent"
        status = "excellent"
    elif coverage_pct >= targets.get("good", 70):
        level = "good"
        status = "good"
    elif coverage_pct >= targets.get("minimum", 60):
        level = "minimum"
        status = "acceptable"
    else:
        level = "below_minimum"
        status = "poor"

    # 品質ゲートチェック
    passes_quality_gate = coverage_pct >= quality_gates.get("fail_under", 60)
    needs_warning = coverage_pct < quality_gates.get("warn_under", 70)
    blocks_pr = coverage_pct < quality_gates.get("block_pr_under", 50)

    return {
        "coverage_percent": coverage_pct,
        "level": level,
        "status": status,
        "passes_quality_gate": passes_quality_gate,
        "needs_warning": needs_warning,
        "blocks_pr": blocks_pr,
        "target_minimum": targets.get("minimum", 60),
        "target_good": targets.get("good", 70),
        "target_excellent": targets.get("excellent", 80),
        "lines_covered": totals.get("covered_lines", 0),
        "lines_total": totals.get("num_statements", 0)
    }

def evaluate_package_coverage(coverage_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """パッケージ別カバレッジの評価"""
    files_data = coverage_data.get("files", {})
    package_targets = config.get("coverage_targets", {}).get("by_package", {})

    # パッケージ別統計を集計
    package_stats = {}

    for filename, data in files_data.items():
        package = get_package_name(filename)
        summary = data.get("summary", {})

        if package not in package_stats:
            package_stats[package] = {
                "covered_lines": 0,
                "total_lines": 0,
                "file_count": 0
            }

        package_stats[package]["covered_lines"] += summary.get("covered_lines", 0)
        package_stats[package]["total_lines"] += summary.get("num_statements", 0)
        package_stats[package]["file_count"] += 1

    # パッケージ別評価
    package_evaluations = {}

    for package, stats in package_stats.items():
        if stats["total_lines"] == 0:
            continue

        coverage_pct = (stats["covered_lines"] / stats["total_lines"]) * 100
        targets = package_targets.get(package, package_targets.get("default", {
            "minimum": 60, "good": 70, "excellent": 80, "ideal": 90
        }))

        # レベル判定
        if coverage_pct >= targets.get("ideal", 90):
            level = "ideal"
            status = "excellent"
        elif coverage_pct >= targets.get("excellent", 80):
            level = "excellent"
            status = "excellent"
        elif coverage_pct >= targets.get("good", 70):
            level = "good"
            status = "good"
        elif coverage_pct >= targets.get("minimum", 60):
            level = "minimum"
            status = "acceptable"
        else:
            level = "below_minimum"
            status = "poor"

        package_evaluations[package] = {
            "coverage_percent": coverage_pct,
            "level": level,
            "status": status,
            "covered_lines": stats["covered_lines"],
            "total_lines": stats["total_lines"],
            "file_count": stats["file_count"],
            "targets": targets,
            "meets_minimum": coverage_pct >= targets.get("minimum", 60)
        }

    return package_evaluations

def evaluate_file_type_coverage(coverage_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """ファイルタイプ別カバレッジの評価"""
    files_data = coverage_data.get("files", {})
    file_type_targets = config.get("coverage_targets", {}).get("by_file_type", {})

    file_type_stats = {}

    # ファイルタイプ別に分類
    for filename, data in files_data.items():
        summary = data.get("summary", {})

        # 各ファイルタイプをチェック
        for file_type, type_config in file_type_targets.items():
            patterns = type_config.get("patterns", [])

            if match_patterns(filename, patterns):
                if file_type not in file_type_stats:
                    file_type_stats[file_type] = {
                        "covered_lines": 0,
                        "total_lines": 0,
                        "file_count": 0,
                        "files": []
                    }

                file_type_stats[file_type]["covered_lines"] += summary.get("covered_lines", 0)
                file_type_stats[file_type]["total_lines"] += summary.get("num_statements", 0)
                file_type_stats[file_type]["file_count"] += 1
                file_type_stats[file_type]["files"].append(filename)
                break

    # ファイルタイプ別評価
    file_type_evaluations = {}

    for file_type, stats in file_type_stats.items():
        if stats["total_lines"] == 0:
            continue

        coverage_pct = (stats["covered_lines"] / stats["total_lines"]) * 100
        targets = file_type_targets.get(file_type, {})

        # レベル判定
        if coverage_pct >= targets.get("ideal", 90):
            level = "ideal"
            status = "excellent"
        elif coverage_pct >= targets.get("excellent", 80):
            level = "excellent"
            status = "excellent"
        elif coverage_pct >= targets.get("good", 70):
            level = "good"
            status = "good"
        elif coverage_pct >= targets.get("minimum", 60):
            level = "minimum"
            status = "acceptable"
        else:
            level = "below_minimum"
            status = "poor"

        file_type_evaluations[file_type] = {
            "coverage_percent": coverage_pct,
            "level": level,
            "status": status,
            "covered_lines": stats["covered_lines"],
            "total_lines": stats["total_lines"],
            "file_count": stats["file_count"],
            "targets": targets,
            "meets_minimum": coverage_pct >= targets.get("minimum", 60)
        }

    return file_type_evaluations

def generate_quality_report(overall: Dict[str, Any], packages: Dict[str, Any],
                          file_types: Dict[str, Any], config: Dict[str, Any]) -> str:
    """品質チェックレポートを生成"""
    report = "# テストカバレッジ品質チェックレポート\n\n"

    from datetime import datetime
    report += f"**生成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"

    # 全体評価
    report += "## 🎯 全体評価\n\n"

    status_emoji = {
        "excellent": "🟢",
        "good": "🟡",
        "acceptable": "🟠",
        "poor": "🔴"
    }

    emoji = status_emoji.get(overall["status"], "⚪")
    report += f"{emoji} **ステータス**: {overall['status'].upper()}\n"
    report += f"📊 **カバレッジ**: {overall['coverage_percent']:.1f}%\n"
    report += f"📈 **目標**: {overall['target_minimum']:.0f}% (最低) / {overall['target_excellent']:.0f}% (優秀)\n"
    report += f"📝 **カバー済み**: {overall['lines_covered']:,} / {overall['lines_total']:,} 行\n\n"

    # 品質ゲート結果
    report += "## 🚦 品質ゲート\n\n"

    if overall["passes_quality_gate"]:
        report += "✅ **品質ゲート**: 通過\n"
    else:
        report += "❌ **品質ゲート**: 失敗\n"

    if overall["blocks_pr"]:
        report += "🚫 **プルリクエスト**: ブロック推奨\n"
    elif overall["needs_warning"]:
        report += "⚠️ **プルリクエスト**: 警告\n"
    else:
        report += "✅ **プルリクエスト**: 承認可能\n"

    report += "\n"

    # パッケージ別評価
    if packages:
        report += "## 📦 パッケージ別評価\n\n"
        report += "| パッケージ | カバレッジ | ステータス | ファイル数 | 目標達成 |\n"
        report += "|-----------|-----------|----------|----------|----------|\n"

        # ステータス順でソート
        sorted_packages = sorted(
            packages.items(),
            key=lambda x: (x[1]["status"], -x[1]["coverage_percent"])
        )

        for package, eval_data in sorted_packages:
            emoji = status_emoji.get(eval_data["status"], "⚪")
            target_check = "✅" if eval_data["meets_minimum"] else "❌"

            report += f"| {package} | {eval_data['coverage_percent']:.1f}% | {emoji} {eval_data['status']} | {eval_data['file_count']} | {target_check} |\n"

        report += "\n"

    # ファイルタイプ別評価
    if file_types:
        report += "## 📄 ファイルタイプ別評価\n\n"
        report += "| タイプ | カバレッジ | ステータス | ファイル数 | 目標達成 |\n"
        report += "|--------|-----------|----------|----------|----------|\n"

        # ステータス順でソート
        sorted_types = sorted(
            file_types.items(),
            key=lambda x: (x[1]["status"], -x[1]["coverage_percent"])
        )

        for file_type, eval_data in sorted_types:
            emoji = status_emoji.get(eval_data["status"], "⚪")
            target_check = "✅" if eval_data["meets_minimum"] else "❌"

            report += f"| {file_type} | {eval_data['coverage_percent']:.1f}% | {emoji} {eval_data['status']} | {eval_data['file_count']} | {target_check} |\n"

        report += "\n"

    # 改善推奨事項
    report += "## 💡 改善推奨事項\n\n"

    recommendations = []

    # 全体的な推奨
    if overall["coverage_percent"] < 60:
        recommendations.append("🚨 **緊急**: 全体カバレッジが60%未満です。テストの大幅な追加が必要です。")
    elif overall["coverage_percent"] < 70:
        recommendations.append("⚠️ **注意**: 全体カバレッジが70%未満です。追加テストを検討してください。")

    # パッケージ別推奨
    poor_packages = [pkg for pkg, data in packages.items() if not data["meets_minimum"]]
    if poor_packages:
        recommendations.append(f"📦 **パッケージ改善**: {', '.join(poor_packages)} のテストを強化してください。")

    # ファイルタイプ別推奨
    poor_types = [ft for ft, data in file_types.items() if not data["meets_minimum"]]
    if poor_types:
        recommendations.append(f"📄 **ファイルタイプ改善**: {', '.join(poor_types)} のテストを強化してください。")

    # 目標到達の推奨
    if overall["coverage_percent"] >= 70 and overall["coverage_percent"] < 80:
        recommendations.append("🎯 **次の目標**: 80%達成まであと少しです。重要なモジュールを重点的にテストしてください。")

    if not recommendations:
        recommendations.append("🎉 **素晴らしい**: 現在のカバレッジレベルは良好です。この水準を維持してください。")

    for rec in recommendations:
        report += f"- {rec}\n"

    report += "\n"

    # 設定情報
    report += "## ⚙️ 設定情報\n\n"
    quality_gates = config.get("quality_gates", {})
    report += f"- **失敗閾値**: {quality_gates.get('fail_under', 60)}%\n"
    report += f"- **警告閾値**: {quality_gates.get('warn_under', 70)}%\n"
    report += f"- **PR ブロック閾値**: {quality_gates.get('block_pr_under', 50)}%\n"
    report += f"- **新規コード最低要件**: {quality_gates.get('new_code_minimum', 80)}%\n"

    return report

def main():
    """メイン実行関数"""
    print("Day Trade カバレッジ品質チェッカー")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    config_file = project_root / "config" / "coverage_config.json"
    coverage_dir = project_root / "reports" / "coverage"

    # 設定ファイル読み込み
    if not config_file.exists():
        print(f"エラー: 設定ファイルが見つかりません: {config_file}")
        return 1

    config = load_coverage_config(config_file)
    if not config:
        return 1

    print(f"設定ファイル読み込み完了: {config_file}")

    # 最新のカバレッジデータを検索
    if not coverage_dir.exists():
        print(f"エラー: カバレッジディレクトリが見つかりません: {coverage_dir}")
        return 1

    json_files = list(coverage_dir.glob("coverage_*.json"))
    if not json_files:
        print("エラー: カバレッジJSONファイルが見つかりません。")
        return 1

    latest_json = sorted(json_files)[-1]
    print(f"カバレッジデータ読み込み: {latest_json}")

    coverage_data = load_coverage_data(latest_json)
    if not coverage_data:
        return 1

    # 評価実行
    print("カバレッジ評価実行中...")

    overall_eval = evaluate_overall_coverage(coverage_data, config)
    package_eval = evaluate_package_coverage(coverage_data, config)
    file_type_eval = evaluate_file_type_coverage(coverage_data, config)

    # レポート生成
    print("品質レポート生成中...")

    quality_report = generate_quality_report(overall_eval, package_eval, file_type_eval, config)

    # レポート保存
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = coverage_dir / f"quality_report_{timestamp}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(quality_report)

    print(f"品質レポート保存完了: {report_file}")

    # 結果表示
    print("\n" + "=" * 50)
    print("品質チェック結果:")
    print(f"全体カバレッジ: {overall_eval['coverage_percent']:.1f}%")
    print(f"ステータス: {overall_eval['status'].upper()}")
    print(f"品質ゲート: {'通過' if overall_eval['passes_quality_gate'] else '失敗'}")

    if overall_eval["blocks_pr"]:
        print("🚫 プルリクエストのブロックを推奨")
        exit_code = 2
    elif not overall_eval["passes_quality_gate"]:
        print("❌ 品質ゲート失敗")
        exit_code = 1
    elif overall_eval["needs_warning"]:
        print("⚠️ 警告レベル")
        exit_code = 0
    else:
        print("✅ 品質基準達成")
        exit_code = 0

    # パッケージ別サマリー
    poor_packages = [pkg for pkg, data in package_eval.items() if not data["meets_minimum"]]
    if poor_packages:
        print(f"改善が必要なパッケージ: {', '.join(poor_packages)}")

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
