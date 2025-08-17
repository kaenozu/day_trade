#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Conflict Analyzer
マージコンフリクト分析ツール

Issue #883対応: CIでコンフリクト検知するための詳細分析スクリプト
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import difflib
import re


@dataclass
class ConflictAnalysis:
    """コンフリクト分析結果"""
    file_path: str
    conflict_type: str
    line_count: int
    conflict_sections: List[Dict[str, Any]]
    resolution_difficulty: str  # easy, medium, hard
    suggested_resolution: str


@dataclass
class MergeAnalysisResult:
    """マージ分析結果"""
    has_conflicts: bool
    target_branch: str
    source_branch: str
    total_conflicts: int
    conflict_files: List[str]
    analysis_timestamp: str
    conflict_analyses: List[ConflictAnalysis]
    overall_difficulty: str
    estimated_resolution_time: str
    recommendations: List[str]


class ConflictAnalyzer:
    """マージコンフリクト分析ツール"""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self.git_available = self._check_git_availability()

    def _check_git_availability(self) -> bool:
        """Gitの利用可能性をチェック"""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_git_command(self, command: List[str]) -> Tuple[bool, str, str]:
        """Gitコマンドを実行"""
        try:
            result = subprocess.run(
                ['git'] + command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""

    def analyze_merge_conflicts(
        self,
        target_branch: str = 'main',
        source_branch: Optional[str] = None
    ) -> MergeAnalysisResult:
        """マージコンフリクトの詳細分析"""

        if not self.git_available:
            raise RuntimeError("Git is not available")

        # 現在のブランチを取得
        if source_branch is None:
            success, current_branch, _ = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
            if not success:
                raise RuntimeError("Failed to get current branch")
            source_branch = current_branch

        print(f"[INFO] マージコンフリクト分析開始:")
        print(f"   ソース: {source_branch}")
        print(f"   ターゲット: {target_branch}")

        # ブランチの存在確認
        success, _, _ = self._run_git_command(['show-ref', '--verify', f'refs/heads/{target_branch}'])
        if not success:
            # リモートブランチの確認
            success, _, _ = self._run_git_command(['show-ref', '--verify', f'refs/remotes/origin/{target_branch}'])
            if not success:
                raise RuntimeError(f"Target branch '{target_branch}' not found")
            target_branch = f'origin/{target_branch}'

        # 現在の状態を保存
        success, original_ref, _ = self._run_git_command(['rev-parse', 'HEAD'])
        if not success:
            raise RuntimeError("Failed to get current HEAD")

        conflict_analyses = []
        has_conflicts = False
        conflict_files = []

        try:
            # テストマージを実行
            print("[INFO] テストマージ実行中...")

            # ターゲットブランチにチェックアウト（detached HEAD）
            success, _, stderr = self._run_git_command(['checkout', target_branch])
            if not success:
                raise RuntimeError(f"Failed to checkout target branch: {stderr}")

            # マージを試行
            success, stdout, stderr = self._run_git_command([
                'merge', '--no-commit', '--no-ff', original_ref
            ])

            if not success:
                has_conflicts = True
                print("[ERROR] マージコンフリクトが発生しました")

                # コンフリクトファイルの取得
                success, files_output, _ = self._run_git_command([
                    'diff', '--name-only', '--diff-filter=U'
                ])

                if success and files_output:
                    conflict_files = files_output.split('\n')
                    print(f"[INFO] コンフリクトファイル数: {len(conflict_files)}")

                    # 各ファイルの詳細分析
                    for file_path in conflict_files:
                        if file_path.strip():
                            analysis = self._analyze_conflict_file(file_path.strip())
                            if analysis:
                                conflict_analyses.append(analysis)

                # マージを中止
                self._run_git_command(['merge', '--abort'])
            else:
                print("[SUCCESS] マージコンフリクトなし")
                # テストマージをリセット
                self._run_git_command(['reset', '--hard', 'HEAD'])

        finally:
            # 元の状態に戻る
            self._run_git_command(['checkout', original_ref])

        # 分析結果をまとめる
        result = MergeAnalysisResult(
            has_conflicts=has_conflicts,
            target_branch=target_branch,
            source_branch=source_branch,
            total_conflicts=len(conflict_files),
            conflict_files=conflict_files,
            analysis_timestamp=datetime.now().isoformat(),
            conflict_analyses=conflict_analyses,
            overall_difficulty=self._assess_overall_difficulty(conflict_analyses),
            estimated_resolution_time=self._estimate_resolution_time(conflict_analyses),
            recommendations=self._generate_recommendations(conflict_analyses)
        )

        return result

    def _analyze_conflict_file(self, file_path: str) -> Optional[ConflictAnalysis]:
        """個別ファイルのコンフリクト分析"""
        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return None

            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # コンフリクトマーカーの検索
            conflict_pattern = r'<<<<<<< .*?\n(.*?)\n=======\n(.*?)\n>>>>>>> .*?\n'
            conflicts = re.findall(conflict_pattern, content, re.DOTALL)

            if not conflicts:
                return None

            # コンフリクトセクションの詳細分析
            conflict_sections = []
            for i, (our_version, their_version) in enumerate(conflicts):
                section = {
                    'section_id': i + 1,
                    'our_lines': len(our_version.split('\n')),
                    'their_lines': len(their_version.split('\n')),
                    'similarity': self._calculate_similarity(our_version, their_version),
                    'our_preview': our_version[:100] + '...' if len(our_version) > 100 else our_version,
                    'their_preview': their_version[:100] + '...' if len(their_version) > 100 else their_version
                }
                conflict_sections.append(section)

            # ファイルタイプベースの分析
            conflict_type = self._determine_conflict_type(file_path, conflicts)
            difficulty = self._assess_file_difficulty(file_path, conflicts, conflict_sections)
            suggested_resolution = self._suggest_resolution(file_path, conflict_type, conflict_sections)

            return ConflictAnalysis(
                file_path=file_path,
                conflict_type=conflict_type,
                line_count=len(content.split('\n')),
                conflict_sections=conflict_sections,
                resolution_difficulty=difficulty,
                suggested_resolution=suggested_resolution
            )

        except Exception as e:
            print(f"[WARNING] ファイル分析エラー {file_path}: {e}")
            return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """2つのテキストの類似度を計算"""
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return round(similarity, 3)

    def _determine_conflict_type(self, file_path: str, conflicts: List[Tuple[str, str]]) -> str:
        """コンフリクトタイプの判定"""
        ext = Path(file_path).suffix.lower()

        if ext == '.py':
            # Pythonファイルの場合、インポート文やクラス定義をチェック
            for our_version, their_version in conflicts:
                if 'import ' in our_version or 'import ' in their_version:
                    return 'python_import'
                elif 'def ' in our_version or 'def ' in their_version:
                    return 'python_function'
                elif 'class ' in our_version or 'class ' in their_version:
                    return 'python_class'
            return 'python_general'

        elif ext in ['.yaml', '.yml']:
            return 'yaml_config'
        elif ext == '.json':
            return 'json_config'
        elif ext == '.md':
            return 'documentation'
        elif ext in ['.txt', '.rst']:
            return 'text'
        elif 'requirements' in file_path.lower():
            return 'dependencies'
        else:
            return 'other'

    def _assess_file_difficulty(
        self,
        file_path: str,
        conflicts: List[Tuple[str, str]],
        conflict_sections: List[Dict[str, Any]]
    ) -> str:
        """ファイルの解決難易度評価"""

        total_conflicts = len(conflicts)
        avg_similarity = sum(section['similarity'] for section in conflict_sections) / len(conflict_sections) if conflict_sections else 0

        # 基本的な難易度判定
        if total_conflicts == 1 and avg_similarity > 0.8:
            return 'easy'
        elif total_conflicts <= 3 and avg_similarity > 0.5:
            return 'medium'
        else:
            base_difficulty = 'hard'

        # ファイルタイプ別の調整
        ext = Path(file_path).suffix.lower()
        if ext in ['.md', '.txt', '.rst']:
            # ドキュメントファイルは通常解決しやすい
            return 'easy' if base_difficulty != 'hard' else 'medium'
        elif ext == '.py':
            # Pythonファイルは構文チェックが必要
            return 'medium' if base_difficulty == 'easy' else 'hard'
        elif ext in ['.yaml', '.yml', '.json']:
            # 設定ファイルは慎重な解決が必要
            return 'medium' if base_difficulty == 'easy' else 'hard'

        return base_difficulty

    def _suggest_resolution(
        self,
        file_path: str,
        conflict_type: str,
        conflict_sections: List[Dict[str, Any]]
    ) -> str:
        """解決方法の提案"""

        suggestions = {
            'python_import': '1. インポート順序を確認 2. 不要なインポートを削除 3. 統合的なインポートブロックを作成',
            'python_function': '1. 関数の目的を確認 2. 機能の重複チェック 3. 必要に応じて関数名変更や統合',
            'python_class': '1. クラス設計の確認 2. メソッドの競合解決 3. 継承関係の整理',
            'python_general': '1. 構文エラーの確認 2. インデントの統一 3. テスト実行で動作確認',
            'yaml_config': '1. YAML構文の確認 2. 設定値の優先度決定 3. 統合設定の検証',
            'json_config': '1. JSON構文の確認 2. 設定項目のマージ 3. スキーマ検証',
            'documentation': '1. 内容の整合性確認 2. 重複セクションの統合 3. 形式の統一',
            'dependencies': '1. バージョン競合の解決 2. 依存関係の整理 3. 互換性の確認',
            'other': '1. ファイル内容の確認 2. 手動マージ 3. 動作テスト'
        }

        base_suggestion = suggestions.get(conflict_type, suggestions['other'])

        # コンフリクトセクション数に基づく追加提案
        if len(conflict_sections) > 3:
            base_suggestion += " 4. 段階的な解決（セクションごと）"

        # 類似度に基づく追加提案
        avg_similarity = sum(section['similarity'] for section in conflict_sections) / len(conflict_sections) if conflict_sections else 0
        if avg_similarity > 0.8:
            base_suggestion += " 注意: 類似度が高いため、変更内容を慎重に比較"

        return base_suggestion

    def _assess_overall_difficulty(self, analyses: List[ConflictAnalysis]) -> str:
        """全体的な解決難易度の評価"""
        if not analyses:
            return 'none'

        difficulty_scores = {'easy': 1, 'medium': 2, 'hard': 3}
        avg_score = sum(difficulty_scores[analysis.resolution_difficulty] for analysis in analyses) / len(analyses)

        if avg_score <= 1.3:
            return 'easy'
        elif avg_score <= 2.3:
            return 'medium'
        else:
            return 'hard'

    def _estimate_resolution_time(self, analyses: List[ConflictAnalysis]) -> str:
        """解決時間の見積もり"""
        if not analyses:
            return '0分'

        time_estimates = {'easy': 5, 'medium': 15, 'hard': 30}  # minutes
        total_time = sum(time_estimates[analysis.resolution_difficulty] for analysis in analyses)

        if total_time <= 10:
            return '5-10分'
        elif total_time <= 30:
            return '15-30分'
        elif total_time <= 60:
            return '30-60分'
        else:
            return '1時間以上'

    def _generate_recommendations(self, analyses: List[ConflictAnalysis]) -> List[str]:
        """推奨事項の生成"""
        recommendations = []

        if not analyses:
            return ['マージコンフリクトなし - 自動マージ可能']

        # ファイルタイプ別の推奨
        python_files = sum(1 for a in analyses if a.conflict_type.startswith('python_'))
        config_files = sum(1 for a in analyses if a.conflict_type.endswith('_config'))

        if python_files > 0:
            recommendations.append(f'{python_files}個のPythonファイルでコンフリクト - 解決後に構文チェックとテスト実行を推奨')

        if config_files > 0:
            recommendations.append(f'{config_files}個の設定ファイルでコンフリクト - 設定値の整合性確認を推奨')

        # 難易度別の推奨
        hard_conflicts = sum(1 for a in analyses if a.resolution_difficulty == 'hard')
        if hard_conflicts > 0:
            recommendations.append(f'{hard_conflicts}個の困難なコンフリクト - 段階的解決とレビューを推奨')

        # 全体的な推奨
        if len(analyses) > 5:
            recommendations.append('多数のコンフリクト - ブランチの最新化とリベースを検討')

        recommendations.append('解決後は必ずテストを実行してください')

        return recommendations

    def export_analysis(self, result: MergeAnalysisResult, output_path: Optional[Path] = None) -> Path:
        """分析結果をJSONファイルにエクスポート"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.repo_path / f'conflict_analysis_{timestamp}.json'

        # データクラスを辞書に変換
        result_dict = asdict(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"[INFO] 分析結果を保存: {output_path}")
        return output_path

    def generate_report(self, result: MergeAnalysisResult) -> str:
        """分析結果のレポート生成"""
        report = f"""
# マージコンフリクト分析レポート

**分析日時**: {result.analysis_timestamp}
**ソースブランチ**: {result.source_branch}
**ターゲットブランチ**: {result.target_branch}

## 概要

- **コンフリクト発生**: {'はい' if result.has_conflicts else 'いいえ'}
- **コンフリクトファイル数**: {result.total_conflicts}
- **全体的な難易度**: {result.overall_difficulty}
- **推定解決時間**: {result.estimated_resolution_time}

## コンフリクトファイル

"""

        if result.conflict_analyses:
            for analysis in result.conflict_analyses:
                report += f"""
### {analysis.file_path}
- **コンフリクトタイプ**: {analysis.conflict_type}
- **解決難易度**: {analysis.resolution_difficulty}
- **コンフリクトセクション数**: {len(analysis.conflict_sections)}
- **推奨解決方法**: {analysis.suggested_resolution}

"""
        else:
            report += "コンフリクトなし\n"

        report += "\n## 推奨事項\n\n"
        for rec in result.recommendations:
            report += f"- {rec}\n"

        return report


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='マージコンフリクト分析ツール')
    parser.add_argument('--target', '-t', default='main', help='ターゲットブランチ (default: main)')
    parser.add_argument('--source', '-s', help='ソースブランチ (default: current)')
    parser.add_argument('--output', '-o', help='出力ファイルパス')
    parser.add_argument('--format', '-f', choices=['json', 'markdown'], default='markdown', help='出力形式')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細出力')

    args = parser.parse_args()

    try:
        analyzer = ConflictAnalyzer()
        result = analyzer.analyze_merge_conflicts(args.target, args.source)

        if args.format == 'json':
            output_path = Path(args.output) if args.output else None
            analyzer.export_analysis(result, output_path)
        else:
            # Markdown形式
            report = analyzer.generate_report(result)
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"[INFO] レポートを保存: {args.output}")
            else:
                print(report)

        # 終了コードの設定
        sys.exit(1 if result.has_conflicts else 0)

    except Exception as e:
        print(f"[ERROR] エラー: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()