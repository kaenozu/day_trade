#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マージコンフリクト詳細分析スクリプト
Issue #883拡張: 高度なコンフリクト分析とレポート生成
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
from enum import Enum


class ConflictType(Enum):
    """コンフリクトタイプ"""
    IMPORT_CONFLICT = "import_conflict"
    FUNCTION_CONFLICT = "function_conflict"
    CLASS_CONFLICT = "class_conflict"
    CONFIG_CONFLICT = "config_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    DOCUMENTATION_CONFLICT = "documentation_conflict"
    DATA_CONFLICT = "data_conflict"
    WHITESPACE_CONFLICT = "whitespace_conflict"
    COMPLEX_CONFLICT = "complex_conflict"


class ConflictSeverity(Enum):
    """コンフリクト重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConflictSection:
    """コンフリクトセクション"""
    file_path: str
    start_line: int
    end_line: int
    base_content: List[str]
    head_content: List[str]
    conflict_type: ConflictType
    severity: ConflictSeverity
    auto_resolvable: bool
    resolution_suggestion: str


@dataclass
class ConflictAnalysisResult:
    """コンフリクト分析結果"""
    total_conflicts: int
    conflicts_by_type: Dict[str, int]
    conflicts_by_severity: Dict[str, int]
    auto_resolvable_count: int
    file_conflicts: List[ConflictSection]
    overall_complexity: str
    resolution_time_estimate: str
    recommended_approach: str


class ConflictAnalyzer:
    """コンフリクト分析エンジン"""

    def __init__(self):
        self.conflict_patterns = {
            ConflictType.IMPORT_CONFLICT: [
                r'^(import\s+\w+|from\s+\w+\s+import)',
                r'__all__\s*=',
            ],
            ConflictType.FUNCTION_CONFLICT: [
                r'def\s+\w+\s*\(',
                r'async\s+def\s+\w+\s*\(',
            ],
            ConflictType.CLASS_CONFLICT: [
                r'class\s+\w+\s*[\(:]',
            ],
            ConflictType.CONFIG_CONFLICT: [
                r'^\s*\w+\s*[:=]',  # YAML/JSON設定
                r'^\[.*\]$',        # INI設定
            ],
            ConflictType.DEPENDENCY_CONFLICT: [
                r'^\w+[>=<~!]',     # requirements.txt
                r'dependencies\s*=',
                r'install_requires',
            ],
            ConflictType.DOCUMENTATION_CONFLICT: [
                r'^#+\s+',          # Markdown headers
                r'^\*\*.*\*\*',     # Markdown bold
                r'^\-\s+',          # Markdown lists
            ],
            ConflictType.WHITESPACE_CONFLICT: [
                r'^\s*$',           # 空行のみ
            ],
        }

    def analyze_file_conflicts(self, file_path: str) -> List[ConflictSection]:
        """ファイル内のコンフリクトを分析"""
        conflicts = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, FileNotFoundError):
            return conflicts

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('<<<<<<<'):
                # コンフリクト開始
                conflict = self._parse_conflict_section(lines, i, file_path)
                if conflict:
                    conflicts.append(conflict)
                    i = conflict.end_line
                else:
                    i += 1
            else:
                i += 1

        return conflicts

    def _parse_conflict_section(self, lines: List[str], start_idx: int, file_path: str) -> ConflictSection:
        """コンフリクトセクションを解析"""
        base_content = []
        head_content = []
        separator_idx = -1
        end_idx = -1

        i = start_idx + 1

        # ベース部分を読み取り
        while i < len(lines) and not lines[i].startswith('======='):
            base_content.append(lines[i].rstrip())
            i += 1

        separator_idx = i
        i += 1

        # ヘッド部分を読み取り
        while i < len(lines) and not lines[i].startswith('>>>>>>>'):
            head_content.append(lines[i].rstrip())
            i += 1

        end_idx = i

        if separator_idx == -1 or end_idx == -1:
            return None

        # コンフリクトタイプを判定
        conflict_type = self._determine_conflict_type(base_content + head_content, file_path)

        # 重要度を判定
        severity = self._determine_severity(conflict_type, base_content, head_content)

        # 自動解決可能性を判定
        auto_resolvable = self._is_auto_resolvable(conflict_type, base_content, head_content)

        # 解決提案を生成
        resolution_suggestion = self._generate_resolution_suggestion(
            conflict_type, base_content, head_content, file_path
        )

        return ConflictSection(
            file_path=file_path,
            start_line=start_idx,
            end_line=end_idx,
            base_content=base_content,
            head_content=head_content,
            conflict_type=conflict_type,
            severity=severity,
            auto_resolvable=auto_resolvable,
            resolution_suggestion=resolution_suggestion
        )

    def _determine_conflict_type(self, content_lines: List[str], file_path: str) -> ConflictType:
        """コンフリクトタイプを判定"""
        file_ext = Path(file_path).suffix.lower()
        content_text = '\n'.join(content_lines)

        # ファイル拡張子による判定
        if file_ext in ['.txt'] and 'requirements' in file_path:
            return ConflictType.DEPENDENCY_CONFLICT
        elif file_ext in ['.md', '.rst']:
            return ConflictType.DOCUMENTATION_CONFLICT
        elif file_ext in ['.yaml', '.yml', '.json', '.toml', '.ini']:
            return ConflictType.CONFIG_CONFLICT

        # 内容による判定（優先度順）
        for conflict_type, patterns in self.conflict_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_text, re.MULTILINE | re.IGNORECASE):
                    return conflict_type

        # 空行のみの場合
        if all(line.strip() == '' for line in content_lines):
            return ConflictType.WHITESPACE_CONFLICT

        # デフォルトは複雑なコンフリクト
        return ConflictType.COMPLEX_CONFLICT

    def _determine_severity(self, conflict_type: ConflictType, base_content: List[str], head_content: List[str]) -> ConflictSeverity:
        """重要度を判定"""
        content_lines = len(base_content) + len(head_content)

        if conflict_type == ConflictType.WHITESPACE_CONFLICT:
            return ConflictSeverity.LOW
        elif conflict_type in [ConflictType.DOCUMENTATION_CONFLICT, ConflictType.IMPORT_CONFLICT]:
            return ConflictSeverity.MEDIUM
        elif conflict_type in [ConflictType.DEPENDENCY_CONFLICT, ConflictType.CONFIG_CONFLICT]:
            return ConflictSeverity.HIGH
        elif conflict_type in [ConflictType.FUNCTION_CONFLICT, ConflictType.CLASS_CONFLICT]:
            return ConflictSeverity.HIGH if content_lines > 10 else ConflictSeverity.MEDIUM
        else:  # COMPLEX_CONFLICT
            return ConflictSeverity.CRITICAL

    def _is_auto_resolvable(self, conflict_type: ConflictType, base_content: List[str], head_content: List[str]) -> bool:
        """自動解決可能性を判定"""
        if conflict_type in [ConflictType.WHITESPACE_CONFLICT, ConflictType.IMPORT_CONFLICT]:
            return True
        elif conflict_type == ConflictType.DEPENDENCY_CONFLICT:
            # 単純な依存関係追加の場合は自動解決可能
            return len(base_content) < 3 and len(head_content) < 3
        elif conflict_type == ConflictType.DOCUMENTATION_CONFLICT:
            # ドキュメントは通常自動解決可能（マージ）
            return True
        elif conflict_type == ConflictType.CONFIG_CONFLICT:
            # 設定の追加の場合は自動解決可能
            return len(base_content) < 5 and len(head_content) < 5
        else:
            return False

    def _generate_resolution_suggestion(self, conflict_type: ConflictType, base_content: List[str], head_content: List[str], file_path: str) -> str:
        """解決提案を生成"""
        suggestions = {
            ConflictType.WHITESPACE_CONFLICT: "空行の違いです。どちらを採用しても問題ありません。",
            ConflictType.IMPORT_CONFLICT: "インポート文の重複です。両方を統合して重複を除去してください。",
            ConflictType.FUNCTION_CONFLICT: "関数定義の競合です。機能を確認して適切にマージしてください。",
            ConflictType.CLASS_CONFLICT: "クラス定義の競合です。設計を見直して統合してください。",
            ConflictType.CONFIG_CONFLICT: "設定の競合です。両方の設定値を確認して適切な値を選択してください。",
            ConflictType.DEPENDENCY_CONFLICT: "依存関係の競合です。バージョン互換性を確認して統合してください。",
            ConflictType.DOCUMENTATION_CONFLICT: "ドキュメントの競合です。両方の内容をマージしてください。",
            ConflictType.DATA_CONFLICT: "データの競合です。データの整合性を確認してマージしてください。",
            ConflictType.COMPLEX_CONFLICT: "複雑な競合です。慎重にレビューして手動で解決してください。"
        }

        base_suggestion = suggestions.get(conflict_type, "手動での解決が必要です。")

        # より具体的な提案を追加
        if conflict_type == ConflictType.IMPORT_CONFLICT:
            base_lines = [line.strip() for line in base_content if line.strip()]
            head_lines = [line.strip() for line in head_content if line.strip()]
            duplicates = set(base_lines) & set(head_lines)
            if duplicates:
                base_suggestion += f" 重複: {', '.join(duplicates)}"

        return base_suggestion

    def generate_comprehensive_report(self, conflict_files: List[str]) -> ConflictAnalysisResult:
        """包括的なコンフリクト分析レポートを生成"""
        all_conflicts = []
        conflicts_by_type = {}
        conflicts_by_severity = {}

        for file_path in conflict_files:
            file_conflicts = self.analyze_file_conflicts(file_path)
            all_conflicts.extend(file_conflicts)

        # 統計集計
        for conflict in all_conflicts:
            # タイプ別集計
            type_name = conflict.conflict_type.value
            conflicts_by_type[type_name] = conflicts_by_type.get(type_name, 0) + 1

            # 重要度別集計
            severity_name = conflict.severity.value
            conflicts_by_severity[severity_name] = conflicts_by_severity.get(severity_name, 0) + 1

        # 自動解決可能数
        auto_resolvable_count = sum(1 for c in all_conflicts if c.auto_resolvable)

        # 全体的な複雑さを判定
        overall_complexity = self._determine_overall_complexity(all_conflicts)

        # 解決時間推定
        resolution_time_estimate = self._estimate_resolution_time(all_conflicts)

        # 推奨アプローチ
        recommended_approach = self._recommend_approach(all_conflicts)

        return ConflictAnalysisResult(
            total_conflicts=len(all_conflicts),
            conflicts_by_type=conflicts_by_type,
            conflicts_by_severity=conflicts_by_severity,
            auto_resolvable_count=auto_resolvable_count,
            file_conflicts=all_conflicts,
            overall_complexity=overall_complexity,
            resolution_time_estimate=resolution_time_estimate,
            recommended_approach=recommended_approach
        )

    def _determine_overall_complexity(self, conflicts: List[ConflictSection]) -> str:
        """全体的な複雑さを判定"""
        if not conflicts:
            return "なし"

        critical_count = sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)
        high_count = sum(1 for c in conflicts if c.severity == ConflictSeverity.HIGH)

        if critical_count > 0:
            return "非常に高い"
        elif high_count > 3:
            return "高い"
        elif high_count > 0:
            return "中程度"
        else:
            return "低い"

    def _estimate_resolution_time(self, conflicts: List[ConflictSection]) -> str:
        """解決時間を推定"""
        if not conflicts:
            return "0分"

        time_estimates = {
            ConflictSeverity.LOW: 2,      # 2分
            ConflictSeverity.MEDIUM: 10,  # 10分
            ConflictSeverity.HIGH: 30,    # 30分
            ConflictSeverity.CRITICAL: 60 # 60分
        }

        total_minutes = sum(time_estimates.get(c.severity, 30) for c in conflicts)

        if total_minutes < 60:
            return f"{total_minutes}分"
        else:
            hours = total_minutes // 60
            minutes = total_minutes % 60
            return f"{hours}時間{minutes}分"

    def _recommend_approach(self, conflicts: List[ConflictSection]) -> str:
        """推奨アプローチ"""
        if not conflicts:
            return "コンフリクトはありません"

        auto_resolvable_count = sum(1 for c in conflicts if c.auto_resolvable)
        total_count = len(conflicts)
        critical_count = sum(1 for c in conflicts if c.severity == ConflictSeverity.CRITICAL)

        if auto_resolvable_count == total_count:
            return "自動解決ツールの使用を推奨"
        elif critical_count > 0:
            return "慎重な手動解決が必要（レビューあり）"
        elif auto_resolvable_count > total_count // 2:
            return "自動解決と手動解決の組み合わせ"
        else:
            return "手動での段階的解決を推奨"


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='マージコンフリクト分析ツール')
    parser.add_argument('files', nargs='+', help='分析対象のコンフリクトファイル')
    parser.add_argument('--output', '-o', help='結果出力ファイル (JSON)')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json', help='出力形式')

    args = parser.parse_args()

    analyzer = ConflictAnalyzer()
    result = analyzer.generate_comprehensive_report(args.files)

    if args.format == 'json':
        # Enumを文字列に変換してからシリアライズ
        result_dict = asdict(result)
        # ConflictSectionのenumフィールドを文字列に変換
        for file_conflict in result_dict['file_conflicts']:
            file_conflict['conflict_type'] = file_conflict['conflict_type'].value if hasattr(file_conflict['conflict_type'], 'value') else str(file_conflict['conflict_type'])
            file_conflict['severity'] = file_conflict['severity'].value if hasattr(file_conflict['severity'], 'value') else str(file_conflict['severity'])

        output = json.dumps(result_dict, indent=2, ensure_ascii=False)
    else:  # markdown
        output = generate_markdown_report(result)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
    else:
        print(output)


def generate_markdown_report(result: ConflictAnalysisResult) -> str:
    """Markdownレポートを生成"""
    percentage = result.auto_resolvable_count/result.total_conflicts*100 if result.total_conflicts > 0 else 0
    md = f"""# コンフリクト分析レポート

## 概要

- **総コンフリクト数**: {result.total_conflicts}
- **自動解決可能**: {result.auto_resolvable_count}/{result.total_conflicts} ({percentage:.1f}%)
- **全体的な複雑さ**: {result.overall_complexity}
- **推定解決時間**: {result.resolution_time_estimate}
- **推奨アプローチ**: {result.recommended_approach}

## タイプ別分布

"""

    for conflict_type, count in result.conflicts_by_type.items():
        md += f"- **{conflict_type}**: {count}件\n"

    md += "\n## 重要度別分布\n\n"

    for severity, count in result.conflicts_by_severity.items():
        md += f"- **{severity}**: {count}件\n"

    md += "\n## ファイル別詳細\n\n"

    for conflict in result.file_conflicts:
        md += f"""### {conflict.file_path}

- **行**: {conflict.start_line}-{conflict.end_line}
- **タイプ**: {conflict.conflict_type.value}
- **重要度**: {conflict.severity.value}
- **自動解決**: {'可能' if conflict.auto_resolvable else '不可能'}
- **提案**: {conflict.resolution_suggestion}

"""

    return md


if __name__ == '__main__':
    main()