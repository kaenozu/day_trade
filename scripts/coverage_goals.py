#!/usr/bin/env python3
"""
カバレッジ目標管理スクリプト
Issue #183: テストカバレッジの計測と可視化

段階的なカバレッジ改善目標を設定し、進捗を追跡します。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


class CoverageGoals:
    """カバレッジ目標管理クラス"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.reports_dir = project_root / "reports" / "coverage"

        # 段階的な目標設定（6ヶ月計画）
        self.goals = {
            "current": 35.0,      # 現在の最低基準
            "short_term": 45.0,   # 1-2ヶ月目標
            "medium_term": 60.0,  # 3-4ヶ月目標
            "long_term": 75.0,    # 5-6ヶ月目標
            "ideal": 85.0         # 理想的な目標
        }

        # パッケージ別優先度
        self.package_priorities = {
            "core": {"priority": "high", "target": 80.0},
            "models": {"priority": "high", "target": 85.0},
            "utils": {"priority": "medium", "target": 70.0},
            "analysis": {"priority": "medium", "target": 65.0},
            "data": {"priority": "medium", "target": 70.0},
            "config": {"priority": "low", "target": 60.0},
            "cli": {"priority": "low", "target": 50.0},
            "automation": {"priority": "low", "target": 40.0}
        }

    def get_latest_coverage_data(self) -> Optional[Dict[str, Any]]:
        """最新のカバレッジデータを取得"""
        json_files = list(self.reports_dir.glob("coverage_*.json"))
        if not json_files:
            return None

        latest_file = sorted(json_files)[-1]
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"カバレッジデータ読み込みエラー: {e}")
            return None

    def analyze_current_status(self) -> Dict[str, Any]:
        """現在のカバレッジ状況を分析"""
        coverage_data = self.get_latest_coverage_data()
        if not coverage_data:
            return {"error": "カバレッジデータが見つかりません"}

        totals = coverage_data.get("totals", {})
        current_coverage = totals.get("percent_covered", 0)

        # 目標達成状況の評価
        status = "needs_improvement"
        if current_coverage >= self.goals["ideal"]:
            status = "excellent"
        elif current_coverage >= self.goals["long_term"]:
            status = "good"
        elif current_coverage >= self.goals["medium_term"]:
            status = "fair"
        elif current_coverage >= self.goals["short_term"]:
            status = "improving"

        # 次の目標までの差分
        next_goal = None
        for goal_name, goal_value in self.goals.items():
            if current_coverage < goal_value:
                next_goal = {"name": goal_name, "value": goal_value}
                break

        return {
            "current_coverage": current_coverage,
            "status": status,
            "goals": self.goals,
            "next_goal": next_goal,
            "gap_to_next": next_goal["value"] - current_coverage if next_goal else 0,
            "totals": totals
        }

    def analyze_package_coverage(self) -> Dict[str, Any]:
        """パッケージ別カバレッジを分析"""
        coverage_data = self.get_latest_coverage_data()
        if not coverage_data:
            return {}

        files = coverage_data.get("files", {})
        package_stats = {}

        for file_path, file_data in files.items():
            # パッケージ名を抽出
            if "src/day_trade/" in file_path or "src\\day_trade\\" in file_path:
                parts = file_path.replace("src/day_trade/", "").replace("src\\day_trade\\", "").split("/")
                if len(parts) > 1:
                    package = parts[0]
                else:
                    package = "root"
            else:
                continue

            if package not in package_stats:
                package_stats[package] = {
                    "files": [],
                    "total_statements": 0,
                    "covered_lines": 0,
                    "missing_lines": 0
                }

            summary = file_data.get("summary", {})
            package_stats[package]["files"].append(file_path)
            package_stats[package]["total_statements"] += summary.get("num_statements", 0)
            package_stats[package]["covered_lines"] += summary.get("covered_lines", 0)
            package_stats[package]["missing_lines"] += summary.get("missing_lines", 0)

        # パッケージ別カバレッジ率を計算
        for package, stats in package_stats.items():
            if stats["total_statements"] > 0:
                coverage_pct = (stats["covered_lines"] / stats["total_statements"]) * 100
                stats["coverage_percent"] = coverage_pct

                # 目標との比較
                target = self.package_priorities.get(package, {}).get("target", 50.0)
                priority = self.package_priorities.get(package, {}).get("priority", "medium")

                stats["target"] = target
                stats["priority"] = priority
                stats["gap"] = target - coverage_pct
                stats["status"] = "above_target" if coverage_pct >= target else "below_target"
            else:
                stats["coverage_percent"] = 0

        return package_stats

    def generate_improvement_recommendations(self) -> List[str]:
        """改善提案を生成"""
        status = self.analyze_current_status()
        package_stats = self.analyze_package_coverage()

        recommendations = []

        # 全体的な改善提案
        current = status.get("current_coverage", 0)
        next_goal = status.get("next_goal")

        if next_goal:
            gap = status.get("gap_to_next", 0)
            recommendations.append(
                f"現在のカバレッジ{current:.1f}%から{next_goal['name']}目標{next_goal['value']:.1f}%まで"
                f"あと{gap:.1f}%の改善が必要です。"
            )

        # パッケージ別の改善提案
        high_priority_packages = [
            (pkg, stats) for pkg, stats in package_stats.items()
            if stats.get("priority") == "high" and stats.get("status") == "below_target"
        ]

        if high_priority_packages:
            recommendations.append("高優先度パッケージのカバレッジ改善が必要:")
            for pkg, stats in sorted(high_priority_packages, key=lambda x: x[1].get("gap", 0), reverse=True):
                gap = stats.get("gap", 0)
                recommendations.append(
                    f"  - {pkg}: 現在{stats.get('coverage_percent', 0):.1f}% "
                    f"→ 目標{stats.get('target', 0):.1f}% (不足{gap:.1f}%)"
                )

        # 具体的な行動提案
        recommendations.extend([
            "カバレッジ向上のための具体的アクション:",
            "  1. 0%カバレッジのファイルから優先的にテストを追加",
            "  2. coreおよびmodelsパッケージのテスト強化",
            "  3. エッジケースとエラーハンドリングのテスト追加",
            "  4. 統合テストの充実化"
        ])

        return recommendations

    def generate_progress_report(self) -> str:
        """進捗レポートを生成"""
        status = self.analyze_current_status()
        package_stats = self.analyze_package_coverage()
        recommendations = self.generate_improvement_recommendations()

        timestamp = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        report = f"""# カバレッジ目標達成進捗レポート

## 基本情報
- **生成日時**: {timestamp}
- **現在のカバレッジ**: {status.get('current_coverage', 0):.1f}%
- **ステータス**: {status.get('status', 'unknown')}

## 目標設定と進捗

| 目標レベル | 目標値 | 達成状況 | 備考 |
|-----------|-------|---------|------|
| 現在の基準 | {self.goals['current']:.1f}% | {'OK' if status.get('current_coverage', 0) >= self.goals['current'] else 'NG'} | 最低基準 |
| 短期目標 | {self.goals['short_term']:.1f}% | {'OK' if status.get('current_coverage', 0) >= self.goals['short_term'] else 'NG'} | 1-2ヶ月 |
| 中期目標 | {self.goals['medium_term']:.1f}% | {'OK' if status.get('current_coverage', 0) >= self.goals['medium_term'] else 'NG'} | 3-4ヶ月 |
| 長期目標 | {self.goals['long_term']:.1f}% | {'OK' if status.get('current_coverage', 0) >= self.goals['long_term'] else 'NG'} | 5-6ヶ月 |
| 理想目標 | {self.goals['ideal']:.1f}% | {'OK' if status.get('current_coverage', 0) >= self.goals['ideal'] else 'NG'} | 最終目標 |

"""

        if status.get('next_goal'):
            next_goal = status['next_goal']
            gap = status.get('gap_to_next', 0)
            report += f"### 次の目標\n"
            report += f"**{next_goal['name']}**: {next_goal['value']:.1f}% (あと{gap:.1f}%)\n\n"

        # パッケージ別詳細
        report += "## パッケージ別カバレッジ状況\n\n"
        report += "| パッケージ | 現在値 | 目標値 | 優先度 | 状況 | 不足 |\n"
        report += "|-----------|-------|-------|-------|------|------|\n"

        for pkg in sorted(package_stats.keys()):
            stats = package_stats[pkg]
            coverage = stats.get('coverage_percent', 0)
            target = stats.get('target', 0)
            priority = stats.get('priority', 'medium')
            status_icon = 'OK' if stats.get('status') == 'above_target' else 'NG'
            gap = max(0, stats.get('gap', 0))

            priority_prefix = {'high': 'H', 'medium': 'M', 'low': 'L'}.get(priority, 'U')

            report += f"| {pkg} | {coverage:.1f}% | {target:.1f}% | {priority_prefix} {priority} | {status_icon} | {gap:.1f}% |\n"

        # 改善提案
        report += "\n## 改善提案\n\n"
        for rec in recommendations:
            report += f"{rec}\n"

        # 統計情報
        totals = status.get('totals', {})
        report += f"""
## 統計情報

- **総ステートメント数**: {totals.get('num_statements', 0):,}
- **カバー済み行数**: {totals.get('covered_lines', 0):,}
- **未カバー行数**: {totals.get('missing_lines', 0):,}
- **カバー済みパッケージ数**: {len([p for p in package_stats.values() if p.get('coverage_percent', 0) > 0])}
- **目標達成パッケージ数**: {len([p for p in package_stats.values() if p.get('status') == 'above_target'])}

---
*このレポートは scripts/coverage_goals.py により自動生成されました。*
"""

        return report

    def save_progress_report(self) -> Path:
        """進捗レポートをファイルに保存"""
        report = self.generate_progress_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.reports_dir / f"coverage_goals_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return report_file


def main():
    """メイン実行関数"""
    print("カバレッジ目標管理システム")
    print("=" * 50)

    project_root = Path(__file__).parent.parent
    goals_manager = CoverageGoals(project_root)

    # 現在の状況を分析
    print("\n1. 現在のカバレッジ状況を分析中...")
    status = goals_manager.analyze_current_status()

    if "error" in status:
        print(f"エラー: {status['error']}")
        return 1

    print(f"現在のカバレッジ: {status['current_coverage']:.1f}%")
    print(f"ステータス: {status['status']}")

    if status.get('next_goal'):
        next_goal = status['next_goal']
        gap = status.get('gap_to_next', 0)
        print(f"次の目標: {next_goal['name']} ({next_goal['value']:.1f}%) - あと{gap:.1f}%")

    # パッケージ別分析
    print("\n2. パッケージ別カバレッジを分析中...")
    package_stats = goals_manager.analyze_package_coverage()

    high_priority_below_target = [
        (pkg, stats) for pkg, stats in package_stats.items()
        if stats.get('priority') == 'high' and stats.get('status') == 'below_target'
    ]

    if high_priority_below_target:
        print("警告: 高優先度パッケージで目標未達成:")
        for pkg, stats in high_priority_below_target:
            print(f"  - {pkg}: {stats.get('coverage_percent', 0):.1f}% / {stats.get('target', 0):.1f}%")
    else:
        print("OK: 高優先度パッケージは目標を達成しています")

    # 改善提案
    print("\n3. 改善提案を生成中...")
    recommendations = goals_manager.generate_improvement_recommendations()

    print("主な改善提案:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")

    # レポート生成
    print("\n4. 詳細レポートを生成中...")
    report_file = goals_manager.save_progress_report()

    print(f"\n詳細レポートが保存されました: {report_file}")

    # サマリー表示
    print("\n" + "=" * 50)
    print("カバレッジ目標管理サマリー:")
    print(f"現在のカバレッジ: {status['current_coverage']:.1f}%")

    goals = goals_manager.goals
    for goal_name, goal_value in goals.items():
        achieved = "OK" if status['current_coverage'] >= goal_value else "NG"
        print(f"{goal_name}: {goal_value:.1f}% {achieved}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
