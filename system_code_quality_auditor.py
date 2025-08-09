#!/usr/bin/env python3
"""
システム全体コード品質監査ツール
現在システムの詳細改善・完成度向上フェーズ

コード品質、設計パターン、最適化機会の包括的分析
"""

import os
import re
import ast
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum


class QualityLevel(Enum):
    """品質レベル"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class IssueType(Enum):
    """問題タイプ"""
    CODE_SMELL = "code_smell"
    BUG_RISK = "bug_risk"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


@dataclass
class CodeIssue:
    """コード問題"""
    file_path: str
    line_number: int
    issue_type: IssueType
    severity: QualityLevel
    title: str
    description: str
    suggestion: str
    code_snippet: str


@dataclass
class FileAnalysis:
    """ファイル分析結果"""
    file_path: str
    file_size: int
    line_count: int
    function_count: int
    class_count: int
    complexity_score: float
    quality_level: QualityLevel
    issues: List[CodeIssue]
    docstring_coverage: float
    test_coverage: float


@dataclass
class QualityReport:
    """品質レポート"""
    timestamp: datetime
    total_files: int
    total_lines: int
    overall_quality: QualityLevel
    file_analyses: List[FileAnalysis]
    quality_metrics: Dict[str, Any]
    improvement_recommendations: List[str]


class PythonASTAnalyzer:
    """Python AST分析器"""

    def __init__(self):
        self.complexity_weights = {
            'if': 1, 'elif': 1, 'else': 0,
            'for': 2, 'while': 2,
            'try': 1, 'except': 1,
            'with': 1, 'lambda': 1,
            'and': 1, 'or': 1
        }

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """ファイル分析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return None

            # AST解析
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                return FileAnalysis(
                    file_path=str(file_path),
                    file_size=len(content),
                    line_count=len(content.splitlines()),
                    function_count=0,
                    class_count=0,
                    complexity_score=0,
                    quality_level=QualityLevel.CRITICAL,
                    issues=[CodeIssue(
                        file_path=str(file_path),
                        line_number=e.lineno or 1,
                        issue_type=IssueType.BUG_RISK,
                        severity=QualityLevel.CRITICAL,
                        title="構文エラー",
                        description=f"Python構文エラー: {e.msg}",
                        suggestion="構文を修正してください",
                        code_snippet=""
                    )],
                    docstring_coverage=0,
                    test_coverage=0
                )

            # 基本メトリクス
            lines = content.splitlines()
            file_size = len(content)
            line_count = len(lines)

            # AST ウォーカー
            walker = ASTWalker()
            walker.visit(tree)

            # 問題検出
            issues = self._detect_issues(file_path, content, lines, walker)

            # 品質レベル計算
            quality_level = self._calculate_quality_level(walker, issues)

            # ドキュメント化レベル
            docstring_coverage = self._calculate_docstring_coverage(walker)

            return FileAnalysis(
                file_path=str(file_path),
                file_size=file_size,
                line_count=line_count,
                function_count=walker.function_count,
                class_count=walker.class_count,
                complexity_score=walker.complexity_score,
                quality_level=quality_level,
                issues=issues,
                docstring_coverage=docstring_coverage,
                test_coverage=0  # TODO: テストカバレッジ統合
            )

        except Exception as e:
            print(f"ファイル分析エラー {file_path}: {e}")
            return None

    def _detect_issues(self, file_path: Path, content: str, lines: List[str], walker: 'ASTWalker') -> List[CodeIssue]:
        """問題検出"""
        issues = []

        # 長い関数検出
        for func_name, func_lines in walker.function_lengths.items():
            if func_lines > 50:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=walker.function_line_numbers.get(func_name, 1),
                    issue_type=IssueType.MAINTAINABILITY,
                    severity=QualityLevel.FAIR if func_lines < 100 else QualityLevel.POOR,
                    title="長すぎる関数",
                    description=f"関数 '{func_name}' が {func_lines} 行と長すぎます",
                    suggestion="関数を複数の小さな関数に分割することを検討してください",
                    code_snippet=""
                ))

        # 複雑度チェック
        if walker.complexity_score > 20:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                issue_type=IssueType.MAINTAINABILITY,
                severity=QualityLevel.POOR if walker.complexity_score < 30 else QualityLevel.CRITICAL,
                title="高すぎる循環的複雑度",
                description=f"ファイルの循環的複雑度が {walker.complexity_score:.1f} と高すぎます",
                suggestion="条件分岐やループを単純化し、関数を分割してください",
                code_snippet=""
            ))

        # 重複コード検出
        line_hashes = defaultdict(list)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith('#'):
                line_hashes[hash(stripped)].append(i + 1)

        for line_hash, line_numbers in line_hashes.items():
            if len(line_numbers) > 2:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_numbers[0],
                    issue_type=IssueType.CODE_SMELL,
                    severity=QualityLevel.FAIR,
                    title="重複コード",
                    description=f"{len(line_numbers)} 箇所で同じコードが重複しています",
                    suggestion="共通処理を関数として抽出してください",
                    code_snippet=""
                ))

        # TODO/FIXME/HACK コメント検出
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(marker in line_lower for marker in ['todo', 'fixme', 'hack', 'xxx']):
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i + 1,
                    issue_type=IssueType.MAINTAINABILITY,
                    severity=QualityLevel.FAIR,
                    title="未完了タスク",
                    description="TODO/FIXMEコメントが残っています",
                    suggestion="タスクを完了するか、適切にドキュメント化してください",
                    code_snippet=line.strip()
                ))

        # 長い行検出
        for i, line in enumerate(lines):
            if len(line) > 120:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i + 1,
                    issue_type=IssueType.CODE_SMELL,
                    severity=QualityLevel.FAIR,
                    title="長すぎる行",
                    description=f"行が {len(line)} 文字と長すぎます（推奨: 120文字以内）",
                    suggestion="行を分割してください",
                    code_snippet=line.strip()[:50] + "..."
                ))

        # インポートチェック
        import_lines = [line for line in lines if line.strip().startswith(('import ', 'from '))]
        if len(import_lines) > 30:
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=1,
                issue_type=IssueType.MAINTAINABILITY,
                severity=QualityLevel.FAIR,
                title="過多なインポート",
                description=f"{len(import_lines)} のインポート文があります",
                suggestion="必要最小限のインポートに整理してください",
                code_snippet=""
            ))

        # ハードコードされた値検出
        for i, line in enumerate(lines):
            # 数値リテラル検出（設定値っぽいもの）
            if re.search(r'=\s*\d{4,}', line):  # 4桁以上の数値
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=i + 1,
                    issue_type=IssueType.MAINTAINABILITY,
                    severity=QualityLevel.FAIR,
                    title="ハードコードされた値",
                    description="設定値がハードコードされている可能性があります",
                    suggestion="定数として定義するか設定ファイルに移動してください",
                    code_snippet=line.strip()
                ))

        return issues

    def _calculate_quality_level(self, walker: 'ASTWalker', issues: List[CodeIssue]) -> QualityLevel:
        """品質レベル計算"""
        # 重大問題があるかチェック
        if any(issue.severity == QualityLevel.CRITICAL for issue in issues):
            return QualityLevel.CRITICAL

        # 問題数による判定
        high_issues = len([i for i in issues if i.severity == QualityLevel.POOR])
        medium_issues = len([i for i in issues if i.severity == QualityLevel.FAIR])

        if high_issues > 3:
            return QualityLevel.POOR
        elif high_issues > 0 or medium_issues > 8:
            return QualityLevel.FAIR
        elif medium_issues > 3:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT

    def _calculate_docstring_coverage(self, walker: 'ASTWalker') -> float:
        """ドキュメント化率計算"""
        total_items = walker.function_count + walker.class_count
        documented_items = walker.documented_functions + walker.documented_classes

        if total_items == 0:
            return 100.0

        return (documented_items / total_items) * 100


class ASTWalker(ast.NodeVisitor):
    """AST ノードウォーカー"""

    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.complexity_score = 1.0  # 基本複雑度
        self.function_lengths = {}
        self.function_line_numbers = {}
        self.documented_functions = 0
        self.documented_classes = 0
        self.current_function = None

    def visit_FunctionDef(self, node):
        """関数定義訪問"""
        self.function_count += 1
        self.current_function = node.name

        # 関数の行数計算
        func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1
        self.function_lengths[node.name] = func_lines
        self.function_line_numbers[node.name] = node.lineno

        # ドキュメント化チェック
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.documented_functions += 1

        self.generic_visit(node)
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        """非同期関数定義訪問"""
        self.visit_FunctionDef(node)  # 同じ処理

    def visit_ClassDef(self, node):
        """クラス定義訪問"""
        self.class_count += 1

        # ドキュメント化チェック
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.documented_classes += 1

        self.generic_visit(node)

    def visit_If(self, node):
        """if文訪問"""
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """for文訪問"""
        self.complexity_score += 2
        self.generic_visit(node)

    def visit_While(self, node):
        """while文訪問"""
        self.complexity_score += 2
        self.generic_visit(node)

    def visit_Try(self, node):
        """try文訪問"""
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_With(self, node):
        """with文訪問"""
        self.complexity_score += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        """lambda式訪問"""
        self.complexity_score += 1
        self.generic_visit(node)


class SystemCodeQualityAuditor:
    """システムコード品質監査システム"""

    def __init__(self):
        self.analyzer = PythonASTAnalyzer()
        self.file_patterns = ['**/*.py']
        self.exclude_patterns = [
            '**/venv/**', '**/env/**', '**/__pycache__/**',
            '**/node_modules/**', '**/build/**', '**/dist/**',
            '**/.*/**'
        ]

        print("=" * 80)
        print("[AUDIT] システム全体コード品質監査")
        print("現在システムの詳細改善・完成度向上フェーズ")
        print("=" * 80)

    def scan_source_files(self, root_dir: Path) -> List[Path]:
        """ソースファイルスキャン"""
        all_files = []

        # パターンマッチングでファイル収集
        for pattern in self.file_patterns:
            files = list(root_dir.rglob(pattern))
            all_files.extend(files)

        # 除外パターンフィルタリング
        filtered_files = []
        for file_path in all_files:
            should_exclude = False
            for exclude_pattern in self.exclude_patterns:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break

            if not should_exclude and file_path.is_file():
                filtered_files.append(file_path)

        return sorted(filtered_files, key=lambda f: f.stat().st_size, reverse=True)

    def audit_codebase(self, root_dir: str = '.') -> QualityReport:
        """コードベース監査"""
        start_time = time.time()
        root_path = Path(root_dir)

        print(f"[SCAN] コードベース監査開始: {root_path.absolute()}")

        # ソースファイル収集
        source_files = self.scan_source_files(root_path)
        print(f"[INFO] 分析対象ファイル: {len(source_files)}個")

        if not source_files:
            print("[WARNING] 分析対象ファイルが見つかりません")
            return QualityReport(
                timestamp=datetime.now(),
                total_files=0,
                total_lines=0,
                overall_quality=QualityLevel.FAIR,
                file_analyses=[],
                quality_metrics={},
                improvement_recommendations=["分析対象ファイルが見つかりませんでした"]
            )

        # ファイル分析
        file_analyses = []
        total_lines = 0

        for i, file_path in enumerate(source_files, 1):
            print(f"[ANALYZE] ({i}/{len(source_files)}) {file_path.name}")

            analysis = self.analyzer.analyze_file(file_path)
            if analysis:
                file_analyses.append(analysis)
                total_lines += analysis.line_count

        # 全体品質計算
        overall_quality = self._calculate_overall_quality(file_analyses)

        # 品質メトリクス
        quality_metrics = self._calculate_quality_metrics(file_analyses)

        # 改善推奨事項
        recommendations = self._generate_improvement_recommendations(file_analyses)

        execution_time = time.time() - start_time

        report = QualityReport(
            timestamp=datetime.now(),
            total_files=len(file_analyses),
            total_lines=total_lines,
            overall_quality=overall_quality,
            file_analyses=file_analyses,
            quality_metrics=quality_metrics,
            improvement_recommendations=recommendations
        )

        print(f"\n[COMPLETE] 品質監査完了 ({execution_time:.2f}秒)")
        print(f"総ファイル数: {report.total_files}")
        print(f"総行数: {report.total_lines:,}")
        print(f"全体品質レベル: {overall_quality.value.upper()}")

        return report

    def _calculate_overall_quality(self, analyses: List[FileAnalysis]) -> QualityLevel:
        """全体品質レベル計算"""
        if not analyses:
            return QualityLevel.FAIR

        quality_counts = Counter(analysis.quality_level for analysis in analyses)
        total_files = len(analyses)

        # 重大問題があるファイルが10%以上
        if quality_counts[QualityLevel.CRITICAL] / total_files > 0.1:
            return QualityLevel.CRITICAL

        # 低品質ファイルが30%以上
        poor_ratio = (quality_counts[QualityLevel.CRITICAL] + quality_counts[QualityLevel.POOR]) / total_files
        if poor_ratio > 0.3:
            return QualityLevel.POOR

        # 中品質以下が60%以上
        fair_ratio = poor_ratio + quality_counts[QualityLevel.FAIR] / total_files
        if fair_ratio > 0.6:
            return QualityLevel.FAIR

        # 良品質以上が80%以上
        good_ratio = fair_ratio + quality_counts[QualityLevel.GOOD] / total_files
        if good_ratio > 0.8:
            return QualityLevel.GOOD

        return QualityLevel.EXCELLENT

    def _calculate_quality_metrics(self, analyses: List[FileAnalysis]) -> Dict[str, Any]:
        """品質メトリクス計算"""
        if not analyses:
            return {}

        # 基本統計
        total_functions = sum(a.function_count for a in analyses)
        total_classes = sum(a.class_count for a in analyses)
        total_issues = sum(len(a.issues) for a in analyses)

        # 複雑度統計
        complexities = [a.complexity_score for a in analyses if a.complexity_score > 0]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0

        # ドキュメント化統計
        doc_coverages = [a.docstring_coverage for a in analyses]
        avg_doc_coverage = sum(doc_coverages) / len(doc_coverages) if doc_coverages else 0

        # 問題タイプ別統計
        issue_type_counts = Counter()
        for analysis in analyses:
            for issue in analysis.issues:
                issue_type_counts[issue.issue_type.value] += 1

        # 品質レベル分布
        quality_distribution = Counter(a.quality_level.value for a in analyses)

        return {
            'total_functions': total_functions,
            'total_classes': total_classes,
            'total_issues': total_issues,
            'average_complexity': round(avg_complexity, 2),
            'average_docstring_coverage': round(avg_doc_coverage, 2),
            'issue_type_distribution': dict(issue_type_counts),
            'quality_level_distribution': dict(quality_distribution),
            'files_needing_attention': len([a for a in analyses if a.quality_level in [QualityLevel.POOR, QualityLevel.CRITICAL]])
        }

    def _generate_improvement_recommendations(self, analyses: List[FileAnalysis]) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []

        # 品質レベル別推奨事項
        critical_files = [a for a in analyses if a.quality_level == QualityLevel.CRITICAL]
        if critical_files:
            recommendations.append(f"重要: {len(critical_files)}個のファイルに重大な問題があります。優先的に修正してください")

        poor_files = [a for a in analyses if a.quality_level == QualityLevel.POOR]
        if poor_files:
            recommendations.append(f"{len(poor_files)}個のファイルの品質が低いです。リファクタリングを検討してください")

        # 共通問題の推奨事項
        all_issues = []
        for analysis in analyses:
            all_issues.extend(analysis.issues)

        issue_counts = Counter(issue.issue_type for issue in all_issues)

        if issue_counts[IssueType.MAINTAINABILITY] > 20:
            recommendations.append("保守性の問題が多数検出されています。関数の分割や複雑度の削減を検討してください")

        if issue_counts[IssueType.CODE_SMELL] > 15:
            recommendations.append("コードの臭いが検出されています。リファクタリングで改善してください")

        if issue_counts[IssueType.DOCUMENTATION] > 10:
            recommendations.append("ドキュメント不足が検出されています。コメントやdocstringを追加してください")

        # ドキュメント化推奨
        low_doc_files = [a for a in analyses if a.docstring_coverage < 50]
        if len(low_doc_files) > len(analyses) * 0.3:
            recommendations.append("多くのファイルでドキュメント化率が低いです。docstringの追加を推奨します")

        # 複雑度推奨
        high_complexity_files = [a for a in analyses if a.complexity_score > 15]
        if high_complexity_files:
            recommendations.append(f"{len(high_complexity_files)}個のファイルの複雑度が高いです。機能分割を検討してください")

        # 一般的推奨事項
        recommendations.extend([
            "定期的なコードレビューの実施",
            "自動フォーマッター（black, flake8）の導入",
            "静的解析ツールの継続使用",
            "単体テストの充実",
            "CI/CDでの品質チェック統合"
        ])

        return recommendations[:15]  # 最大15個の推奨事項

    def generate_quality_report(self, report: QualityReport, format_type: str = 'summary') -> str:
        """品質レポート生成"""
        if format_type == 'summary':
            return self._format_summary_report(report)
        elif format_type == 'detailed':
            return self._format_detailed_report(report)
        else:
            return json.dumps(asdict(report), indent=2, ensure_ascii=False, default=str)

    def _format_summary_report(self, report: QualityReport) -> str:
        """サマリーレポート形式"""
        metrics = report.quality_metrics

        return f"""
=== コード品質監査サマリーレポート ===
監査時刻: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
総ファイル数: {report.total_files}個
総行数: {report.total_lines:,}行
全体品質レベル: {report.overall_quality.value.upper()}

=== 品質メトリクス ===
総関数数: {metrics.get('total_functions', 0)}個
総クラス数: {metrics.get('total_classes', 0)}個
総問題数: {metrics.get('total_issues', 0)}個
平均複雑度: {metrics.get('average_complexity', 0)}
ドキュメント化率: {metrics.get('average_docstring_coverage', 0):.1f}%

=== 品質レベル分布 ===
{self._format_quality_distribution(metrics.get('quality_level_distribution', {}))}

=== 改善推奨事項（上位5項目）===
{chr(10).join(f"- {rec}" for rec in report.improvement_recommendations[:5])}
"""

    def _format_detailed_report(self, report: QualityReport) -> str:
        """詳細レポート形式"""
        summary = self._format_summary_report(report)

        # 問題のあるファイルの詳細
        problem_files = [a for a in report.file_analyses
                        if a.quality_level in [QualityLevel.CRITICAL, QualityLevel.POOR]]

        detailed = summary + "\n\n=== 要注意ファイル ==="

        for analysis in problem_files[:10]:  # 上位10ファイル
            detailed += f"""
ファイル: {analysis.file_path}
品質レベル: {analysis.quality_level.value.upper()}
行数: {analysis.line_count}行
複雑度: {analysis.complexity_score:.1f}
問題数: {len(analysis.issues)}個
主な問題:"""

            for issue in analysis.issues[:3]:  # 主要問題3個
                detailed += f"\n  - {issue.title} (行{issue.line_number}): {issue.description}"

        return detailed

    def _format_quality_distribution(self, distribution: Dict[str, int]) -> str:
        """品質分布フォーマット"""
        total = sum(distribution.values())
        if total == 0:
            return "データなし"

        result = []
        for level in ['excellent', 'good', 'fair', 'poor', 'critical']:
            count = distribution.get(level, 0)
            percent = (count / total) * 100
            result.append(f"{level.capitalize()}: {count}個 ({percent:.1f}%)")

        return '\n'.join(result)

    def save_quality_report(self, report: QualityReport, filename: str = None) -> str:
        """品質レポート保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"code_quality_report_{timestamp}.json"

        # JSON用にシリアライズ
        report_dict = asdict(report)
        report_dict['timestamp'] = report.timestamp.isoformat()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        return f"品質レポート保存完了: {filename}"


def main():
    """メイン実行"""
    auditor = SystemCodeQualityAuditor()

    try:
        # コード品質監査実行
        report = auditor.audit_codebase()

        # サマリー表示
        try:
            print("\n" + "=" * 80)
            summary_report = auditor.generate_quality_report(report, 'summary')
            print(summary_report.encode('ascii', 'replace').decode('ascii'))
            print("=" * 80)
        except UnicodeEncodeError:
            print("\n[SUMMARY] レポート出力完了（エンコーディング問題のため詳細は省略）")
            print(f"総ファイル数: {report.total_files}")
            print(f"総行数: {report.total_lines:,}")
            print(f"品質レベル: {report.overall_quality.value.upper()}")
            print("=" * 80)

        # レポート保存
        saved_file = auditor.save_quality_report(report)
        print(f"\n[REPORT] {saved_file}")

        # 詳細レポートも表示（問題ファイルがある場合）
        if report.quality_metrics.get('files_needing_attention', 0) > 0:
            print("\n" + "=" * 80)
            print("[DETAILED] 問題のあるファイルの詳細")
            print("=" * 80)
            print(auditor.generate_quality_report(report, 'detailed'))

    except KeyboardInterrupt:
        print("\n[STOP] コード品質監査中断")
    except Exception as e:
        print(f"\n[ERROR] 監査エラー: {e}")


if __name__ == "__main__":
    main()
