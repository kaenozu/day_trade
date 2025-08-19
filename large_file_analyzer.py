#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大規模ファイル分析ツール - Issue #959対応
大規模Pythonファイルの構造分析と分割計画生成
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import json
from dataclasses import dataclass, asdict

@dataclass
class FunctionInfo:
    """関数情報"""
    name: str
    line_start: int
    line_end: int
    line_count: int
    args: List[str]
    decorators: List[str]
    complexity: int = 0

@dataclass
class ClassInfo:
    """クラス情報"""
    name: str
    line_start: int
    line_end: int
    line_count: int
    methods: List[FunctionInfo]
    base_classes: List[str]
    decorators: List[str]

@dataclass
class FileAnalysis:
    """ファイル分析結果"""
    filepath: str
    total_lines: int
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    imports: List[str]
    complexity_score: int
    suggested_splits: List[Dict[str, Any]]

class LargeFileAnalyzer:
    """大規模ファイル分析クラス"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.large_files: List[Path] = []
        self.analysis_results: Dict[str, FileAnalysis] = {}
        
    def find_large_files(self, threshold: int = 500) -> List[Path]:
        """大規模ファイルの検出"""
        large_files = []
        
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith('.') or 'test_' in py_file.name:
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    line_count = len(f.readlines())
                    
                if line_count > threshold:
                    large_files.append(py_file)
                    print(f"Large file found: {py_file} ({line_count} lines)")
                    
            except Exception as e:
                print(f"Error reading {py_file}: {e}")
                
        self.large_files = sorted(large_files, key=lambda x: self._get_line_count(x), reverse=True)
        return self.large_files
    
    def _get_line_count(self, filepath: Path) -> int:
        """ファイルの行数取得"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def analyze_file(self, filepath: Path) -> Optional[FileAnalysis]:
        """ファイルの詳細分析"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    classes.append(class_info)
                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    func_info = self._analyze_function(node, content)
                    functions.append(func_info)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.extend(self._extract_imports(node))
            
            total_lines = len(content.splitlines())
            complexity_score = self._calculate_complexity(tree)
            suggested_splits = self._suggest_splits(classes, functions, total_lines)
            
            analysis = FileAnalysis(
                filepath=str(filepath),
                total_lines=total_lines,
                classes=classes,
                functions=functions,
                imports=list(set(imports)),
                complexity_score=complexity_score,
                suggested_splits=suggested_splits
            )
            
            self.analysis_results[str(filepath)] = analysis
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return None
    
    def _analyze_class(self, node: ast.ClassDef, content: str) -> ClassInfo:
        """クラス分析"""
        lines = content.splitlines()
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, content)
                methods.append(method_info)
        
        base_classes = [self._get_node_name(base) for base in node.bases]
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        
        return ClassInfo(
            name=node.name,
            line_start=line_start,
            line_end=line_end,
            line_count=line_end - line_start + 1,
            methods=methods,
            base_classes=base_classes,
            decorators=decorators
        )
    
    def _analyze_function(self, node: ast.FunctionDef, content: str) -> FunctionInfo:
        """関数分析"""
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        
        args = [arg.arg for arg in node.args.args]
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        complexity = self._calculate_function_complexity(node)
        
        return FunctionInfo(
            name=node.name,
            line_start=line_start,
            line_end=line_end,
            line_count=line_end - line_start + 1,
            args=args,
            decorators=decorators,
            complexity=complexity
        )
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """メソッドかどうかの判定"""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _extract_imports(self, node) -> List[str]:
        """import文の抽出"""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _get_node_name(self, node) -> str:
        """ノード名の取得"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _get_decorator_name(self, node) -> str:
        """デコレータ名の取得"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """複雑度計算（簡易版）"""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        return complexity
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """関数の複雑度計算"""
        complexity = 1  # 基本複雑度
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        return complexity
    
    def _suggest_splits(self, classes: List[ClassInfo], functions: List[FunctionInfo], total_lines: int) -> List[Dict[str, Any]]:
        """分割提案の生成"""
        suggestions = []
        
        # クラス別分割提案
        for cls in classes:
            if cls.line_count > 200:
                suggestions.append({
                    "type": "class_split",
                    "target": cls.name,
                    "reason": f"Large class ({cls.line_count} lines)",
                    "suggested_file": f"{cls.name.lower()}.py",
                    "line_range": [cls.line_start, cls.line_end]
                })
        
        # 機能別分割提案
        if len(classes) > 5:
            suggestions.append({
                "type": "module_split",
                "target": "multiple_classes",
                "reason": f"Too many classes ({len(classes)} classes)",
                "suggested_structure": self._suggest_module_structure(classes, functions)
            })
        
        # 複雑度による分割提案
        high_complexity_functions = [f for f in functions if f.complexity > 10]
        if high_complexity_functions:
            suggestions.append({
                "type": "complexity_split",
                "target": [f.name for f in high_complexity_functions],
                "reason": "High complexity functions",
                "suggested_file": "complex_operations.py"
            })
        
        return suggestions
    
    def _suggest_module_structure(self, classes: List[ClassInfo], functions: List[FunctionInfo]) -> Dict[str, List[str]]:
        """モジュール構造提案"""
        structure = defaultdict(list)
        
        # クラス分類（名前パターンベース）
        for cls in classes:
            if 'Manager' in cls.name or 'Handler' in cls.name:
                structure['managers'].append(cls.name)
            elif 'Service' in cls.name or 'API' in cls.name:
                structure['services'].append(cls.name)
            elif 'Model' in cls.name or 'Data' in cls.name:
                structure['models'].append(cls.name)
            elif 'View' in cls.name or 'Template' in cls.name:
                structure['views'].append(cls.name)
            else:
                structure['core'].append(cls.name)
        
        # 関数分類
        utility_functions = [f.name for f in functions if len(f.args) <= 2 and f.complexity <= 5]
        if utility_functions:
            structure['utils'] = utility_functions
        
        return dict(structure)
    
    def generate_refactoring_plan(self, filepath: Path) -> Dict[str, Any]:
        """リファクタリング計画生成"""
        analysis = self.analyze_file(filepath)
        if not analysis:
            return {}
        
        plan = {
            "original_file": str(filepath),
            "total_lines": analysis.total_lines,
            "complexity_score": analysis.complexity_score,
            "refactoring_priority": self._calculate_priority(analysis),
            "suggested_splits": analysis.suggested_splits,
            "new_structure": {},
            "migration_steps": []
        }
        
        # 新しい構造の提案
        if analysis.suggested_splits:
            for suggestion in analysis.suggested_splits:
                if suggestion["type"] == "module_split":
                    plan["new_structure"] = suggestion.get("suggested_structure", {})
                    break
        
        # 移行ステップの生成
        plan["migration_steps"] = self._generate_migration_steps(analysis)
        
        return plan
    
    def _calculate_priority(self, analysis: FileAnalysis) -> str:
        """リファクタリング優先度計算"""
        if analysis.total_lines > 2000:
            return "HIGH"
        elif analysis.total_lines > 1000:
            return "MEDIUM"
        elif analysis.complexity_score > 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_migration_steps(self, analysis: FileAnalysis) -> List[Dict[str, str]]:
        """移行ステップ生成"""
        steps = []
        
        steps.append({
            "step": 1,
            "action": "backup_original",
            "description": "元ファイルのバックアップ作成"
        })
        
        steps.append({
            "step": 2,
            "action": "create_tests",
            "description": "既存機能のテスト作成"
        })
        
        steps.append({
            "step": 3,
            "action": "extract_modules",
            "description": "独立性の高いモジュールから抽出"
        })
        
        steps.append({
            "step": 4,
            "action": "update_imports",
            "description": "import文の更新"
        })
        
        steps.append({
            "step": 5,
            "action": "run_tests",
            "description": "テスト実行・検証"
        })
        
        return steps
    
    def export_analysis(self, output_file: str = "large_file_analysis.json"):
        """分析結果のエクスポート"""
        export_data = {}
        
        for filepath, analysis in self.analysis_results.items():
            export_data[filepath] = {
                "total_lines": analysis.total_lines,
                "classes_count": len(analysis.classes),
                "functions_count": len(analysis.functions),
                "complexity_score": analysis.complexity_score,
                "imports_count": len(analysis.imports),
                "suggested_splits": analysis.suggested_splits,
                "classes": [asdict(cls) for cls in analysis.classes],
                "functions": [asdict(func) for func in analysis.functions]
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis exported to {output_file}")
    
    def print_summary(self):
        """分析結果サマリー表示"""
        print("\n" + "="*60)
        print("LARGE FILE ANALYSIS SUMMARY")
        print("="*60)
        
        for filepath, analysis in self.analysis_results.items():
            print(f"\n{Path(filepath).name}")
            print(f"   Lines: {analysis.total_lines}")
            print(f"   Classes: {len(analysis.classes)}")
            print(f"   Functions: {len(analysis.functions)}")
            print(f"   Complexity: {analysis.complexity_score}")
            print(f"   Splits suggested: {len(analysis.suggested_splits)}")
            
            if analysis.suggested_splits:
                print("   Suggestions:")
                for i, suggestion in enumerate(analysis.suggested_splits, 1):
                    print(f"      {i}. {suggestion['type']}: {suggestion['reason']}")

def main():
    """メイン実行"""
    print("Large File Analyzer - Issue #959")
    print("="*50)
    
    analyzer = LargeFileAnalyzer()
    
    # 大規模ファイル検出
    large_files = analyzer.find_large_files(threshold=500)
    print(f"\nFound {len(large_files)} large files")
    
    # 詳細分析
    for filepath in large_files[:5]:  # 上位5ファイルを分析
        print(f"\nAnalyzing {filepath.name}...")
        analysis = analyzer.analyze_file(filepath)
        
        if analysis:
            # リファクタリング計画生成
            plan = analyzer.generate_refactoring_plan(filepath)
            if plan:
                plan_file = f"refactoring_plan_{filepath.stem}.json"
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(plan, f, indent=2, ensure_ascii=False)
                print(f"   Refactoring plan saved to {plan_file}")
    
    # サマリー表示
    analyzer.print_summary()
    
    # 結果エクスポート
    analyzer.export_analysis()

if __name__ == "__main__":
    main()