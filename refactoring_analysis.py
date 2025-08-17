#!/usr/bin/env python3
"""
リファクタリング分析ツール

現在のコードベースを分析し、リファクタリング対象を特定
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import json
from datetime import datetime


class RefactoringAnalyzer:
    """リファクタリング分析クラス"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.python_files = list(base_dir.glob("**/*.py"))
        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(self.python_files),
            'file_sizes': {},
            'duplicate_patterns': {},
            'dead_code_candidates': [],
            'complex_functions': [],
            'import_analysis': {},
            'architecture_issues': [],
            'performance_issues': [],
            'refactoring_priorities': []
        }

    def run_comprehensive_analysis(self) -> Dict:
        """包括的分析実行"""
        print("コードベース分析開始...")

        # 1. ファイルサイズ分析
        self._analyze_file_sizes()

        # 2. 重複コード検出
        self._detect_duplicate_code()

        # 3. デッドコード検出
        self._detect_dead_code()

        # 4. 複雑度分析
        self._analyze_complexity()

        # 5. インポート分析
        self._analyze_imports()

        # 6. アーキテクチャ問題検出
        self._detect_architecture_issues()

        # 7. パフォーマンス問題検出
        self._detect_performance_issues()

        # 8. リファクタリング優先度設定
        self._set_refactoring_priorities()

        return self.analysis_results

    def _analyze_file_sizes(self):
        """ファイルサイズ分析"""
        print("  ファイルサイズ分析中...")

        large_files = []
        total_lines = 0

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    total_lines += line_count

                    rel_path = py_file.relative_to(self.base_dir)
                    self.analysis_results['file_sizes'][str(rel_path)] = line_count

                    if line_count > 300:  # 300行以上の大きなファイル
                        large_files.append({
                            'file': str(rel_path),
                            'lines': line_count,
                            'priority': 'high' if line_count > 500 else 'medium'
                        })
            except Exception:
                continue

        self.analysis_results['large_files'] = large_files
        self.analysis_results['average_file_size'] = total_lines / len(self.python_files) if self.python_files else 0

    def _detect_duplicate_code(self):
        """重複コード検出"""
        print("  重複コード検出中...")

        function_signatures = defaultdict(list)
        class_names = defaultdict(list)
        import_patterns = defaultdict(list)

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                rel_path = py_file.relative_to(self.base_dir)

                # 関数定義の重複チェック
                func_matches = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
                for func in func_matches:
                    function_signatures[func].append(str(rel_path))

                # クラス名の重複チェック
                class_matches = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', content)
                for cls in class_matches:
                    class_names[cls].append(str(rel_path))

                # インポートパターンの重複チェック
                import_matches = re.findall(r'from\s+[\w.]+\s+import\s+[\w, ]+|import\s+[\w.]+', content)
                for imp in import_matches:
                    import_patterns[imp].append(str(rel_path))

            except Exception:
                continue

        # 重複検出
        duplicate_functions = {k: v for k, v in function_signatures.items() if len(v) > 1}
        duplicate_classes = {k: v for k, v in class_names.items() if len(v) > 1}

        self.analysis_results['duplicate_patterns'] = {
            'functions': duplicate_functions,
            'classes': duplicate_classes,
            'common_imports': dict(list(import_patterns.items())[:10])  # 上位10個
        }

    def _detect_dead_code(self):
        """デッドコード検出候補"""
        print("  デッドコード検出中...")

        # 使用されていない可能性があるファイルを検出
        dead_code_patterns = [
            r'# TODO:.*',
            r'# FIXME:.*',
            r'# DEPRECATED.*',
            r'if\s+False:',
            r'if\s+0:',
            r'def\s+\w+.*:\s*pass\s*$',
            r'class\s+\w+.*:\s*pass\s*$'
        ]

        candidates = []

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                rel_path = py_file.relative_to(self.base_dir)
                issues = []

                for pattern in dead_code_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    if matches:
                        issues.extend(matches)

                if issues:
                    candidates.append({
                        'file': str(rel_path),
                        'issues': issues[:5],  # 最初の5個のみ
                        'total_issues': len(issues)
                    })

            except Exception:
                continue

        self.analysis_results['dead_code_candidates'] = candidates

    def _analyze_complexity(self):
        """複雑度分析"""
        print("  複雑度分析中...")

        complex_functions = []

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                try:
                    tree = ast.parse(content)
                    rel_path = py_file.relative_to(self.base_dir)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # 簡易的な複雑度計算（ネストレベル + 分岐数）
                            complexity = self._calculate_complexity(node)

                            if complexity > 10:  # 複雑度が10を超える関数
                                complex_functions.append({
                                    'file': str(rel_path),
                                    'function': node.name,
                                    'complexity': complexity,
                                    'line': node.lineno
                                })
                except SyntaxError:
                    continue

            except Exception:
                continue

        # 複雑度順にソート
        complex_functions.sort(key=lambda x: x['complexity'], reverse=True)
        self.analysis_results['complex_functions'] = complex_functions[:20]  # 上位20個

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """簡易的な複雑度計算"""
        complexity = 1  # ベース複雑度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _analyze_imports(self):
        """インポート分析"""
        print("  インポート分析中...")

        import_stats = defaultdict(int)
        circular_imports = []

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # インポート文の抽出
                imports = re.findall(r'(?:from\s+[\w.]+\s+)?import\s+[\w., ]+', content)

                for imp in imports:
                    import_stats[imp] += 1

            except Exception:
                continue

        # 最も使用されているインポート（上位10個）
        top_imports = dict(sorted(import_stats.items(), key=lambda x: x[1], reverse=True)[:10])

        self.analysis_results['import_analysis'] = {
            'top_imports': top_imports,
            'total_unique_imports': len(import_stats),
            'potential_consolidation': len([k for k, v in import_stats.items() if v > 5])
        }

    def _detect_architecture_issues(self):
        """アーキテクチャ問題検出"""
        print("  アーキテクチャ問題検出中...")

        issues = []

        # 1. 循環インポートの可能性
        src_files = [f for f in self.python_files if 'src' in str(f)]
        if len(src_files) != len(self.python_files):
            issues.append({
                'type': 'mixed_structure',
                'description': 'srcディレクトリ外にPythonファイルが散在',
                'severity': 'medium'
            })

        # 2. 大きすぎるモジュール
        large_files = self.analysis_results.get('large_files', [])
        if len(large_files) > 5:
            issues.append({
                'type': 'large_modules',
                'description': f'{len(large_files)}個の大きなファイル（300行以上）が存在',
                'severity': 'high'
            })

        # 3. 重複クラス/関数
        duplicates = self.analysis_results.get('duplicate_patterns', {})
        if duplicates.get('functions') or duplicates.get('classes'):
            issues.append({
                'type': 'code_duplication',
                'description': '重複するクラス・関数が検出されました',
                'severity': 'medium'
            })

        self.analysis_results['architecture_issues'] = issues

    def _detect_performance_issues(self):
        """パフォーマンス問題検出"""
        print("  パフォーマンス問題検出中...")

        issues = []

        performance_patterns = [
            (r'\.glob\(.*\*\*.*\)', 'recursive_glob', '再帰的globの使用'),
            (r'for.*in.*\.keys\(\)', 'dict_keys_iteration', '辞書キーの非効率な反復'),
            (r'\.append\(.*\).*in.*for', 'list_comprehension_candidate', 'リスト内包表記の候補'),
            (r'time\.sleep\(', 'blocking_sleep', 'ブロッキングsleep'),
            (r'open\(.*\)(?!\s*with)', 'file_handle_leak', 'ファイルハンドルリークの可能性')
        ]

        for py_file in self.python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                rel_path = py_file.relative_to(self.base_dir)
                file_issues = []

                for pattern, issue_type, description in performance_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        file_issues.append({
                            'type': issue_type,
                            'description': description,
                            'count': len(matches)
                        })

                if file_issues:
                    issues.append({
                        'file': str(rel_path),
                        'issues': file_issues
                    })

            except Exception:
                continue

        self.analysis_results['performance_issues'] = issues

    def _set_refactoring_priorities(self):
        """リファクタリング優先度設定"""
        print("  リファクタリング優先度設定中...")

        priorities = []

        # 1. 大きなファイルの分割
        large_files = self.analysis_results.get('large_files', [])
        for file_info in large_files:
            if file_info['priority'] == 'high':
                priorities.append({
                    'task': f"{file_info['file']} の分割",
                    'priority': 'high',
                    'reason': f"{file_info['lines']}行の大きなファイル",
                    'estimated_effort': 'high'
                })

        # 2. 複雑な関数の簡素化
        complex_functions = self.analysis_results.get('complex_functions', [])
        for func in complex_functions[:5]:  # 上位5個
            priorities.append({
                'task': f"{func['file']}:{func['function']} の簡素化",
                'priority': 'medium',
                'reason': f"複雑度{func['complexity']}の関数",
                'estimated_effort': 'medium'
            })

        # 3. 重複コードの統合
        duplicates = self.analysis_results.get('duplicate_patterns', {})
        if duplicates.get('functions'):
            priorities.append({
                'task': '重複関数の統合',
                'priority': 'medium',
                'reason': f"{len(duplicates['functions'])}個の重複関数",
                'estimated_effort': 'medium'
            })

        # 4. パフォーマンス問題の解決
        perf_issues = self.analysis_results.get('performance_issues', [])
        if len(perf_issues) > 5:
            priorities.append({
                'task': 'パフォーマンス問題の修正',
                'priority': 'low',
                'reason': f"{len(perf_issues)}個のファイルでパフォーマンス問題",
                'estimated_effort': 'medium'
            })

        # 5. デッドコードの削除
        dead_code = self.analysis_results.get('dead_code_candidates', [])
        if len(dead_code) > 3:
            priorities.append({
                'task': 'デッドコードの削除',
                'priority': 'low',
                'reason': f"{len(dead_code)}個のファイルでデッドコード候補",
                'estimated_effort': 'low'
            })

        # 優先度順にソート
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        priorities.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)

        self.analysis_results['refactoring_priorities'] = priorities

    def save_results(self, output_file: Path):
        """結果保存"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)

        print(f"分析結果を保存: {output_file}")

    def generate_report(self) -> str:
        """レポート生成"""
        report = f"""# リファクタリング分析レポート

生成日時: {self.analysis_results['timestamp']}
対象ファイル数: {self.analysis_results['total_files']}個

## 📊 概要

### ファイルサイズ統計
- 平均ファイルサイズ: {self.analysis_results.get('average_file_size', 0):.1f}行
- 大きなファイル（300行以上）: {len(self.analysis_results.get('large_files', []))}個

### 重複コード
- 重複関数: {len(self.analysis_results.get('duplicate_patterns', {}).get('functions', {}))}個
- 重複クラス: {len(self.analysis_results.get('duplicate_patterns', {}).get('classes', {}))}個

### 複雑度
- 高複雑度関数: {len(self.analysis_results.get('complex_functions', []))}個

## 🎯 リファクタリング優先順位

"""

        for i, priority in enumerate(self.analysis_results.get('refactoring_priorities', []), 1):
            report += f"{i}. **{priority['task']}** ({priority['priority']})\n"
            report += f"   - 理由: {priority['reason']}\n"
            report += f"   - 推定工数: {priority['estimated_effort']}\n\n"

        report += f"""
## 📋 詳細分析結果

### 大きなファイル
"""

        for file_info in self.analysis_results.get('large_files', [])[:5]:
            report += f"- {file_info['file']}: {file_info['lines']}行 ({file_info['priority']})\n"

        report += f"""
### 高複雑度関数
"""

        for func in self.analysis_results.get('complex_functions', [])[:5]:
            report += f"- {func['file']}:{func['function']} (複雑度: {func['complexity']})\n"

        return report


def main():
    """メイン実行"""
    print("リファクタリング分析開始")
    print("=" * 40)

    base_dir = Path(__file__).parent
    analyzer = RefactoringAnalyzer(base_dir)

    # 分析実行
    results = analyzer.run_comprehensive_analysis()

    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = base_dir / f"refactoring_analysis_{timestamp}.json"
    analyzer.save_results(output_file)

    # レポート生成
    report = analyzer.generate_report()
    report_file = base_dir / f"refactoring_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nレポート生成完了: {report_file}")

    # 要約表示
    print("\n" + "=" * 40)
    print("リファクタリング分析完了")
    print("=" * 40)
    print(f"対象ファイル: {results['total_files']}個")
    print(f"大きなファイル: {len(results.get('large_files', []))}個")
    print(f"複雑な関数: {len(results.get('complex_functions', []))}個")
    print(f"優先タスク: {len(results.get('refactoring_priorities', []))}個")
    print("=" * 40)


if __name__ == "__main__":
    main()