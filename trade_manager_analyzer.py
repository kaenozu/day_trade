#!/usr/bin/env python3
"""
Trade Manager 詳細分析ツール
trade_manager.py (2,586行) の構造分析とリファクタリング戦略立案

Issue #959: 大型ファイルリファクタリング - Phase 2
最高優先ファイル: trade_manager.py の分析
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class FunctionInfo:
    """関数情報"""
    name: str
    start_line: int
    end_line: int
    line_count: int
    complexity: int
    dependencies: Set[str]
    is_method: bool
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_annotation: Optional[str] = None


@dataclass
class ClassInfo:
    """クラス情報"""
    name: str
    start_line: int
    end_line: int
    line_count: int
    methods: List[FunctionInfo]
    dependencies: Set[str]
    base_classes: List[str]
    docstring: Optional[str] = None


@dataclass
class ModuleInfo:
    """モジュール情報"""
    imports: List[str]
    global_functions: List[FunctionInfo]
    classes: List[ClassInfo]
    constants: List[str]
    global_variables: List[str]
    total_lines: int


class TradeManagerAnalyzer:
    """Trade Manager 分析器"""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.content = self.file_path.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')
        self.tree = ast.parse(self.content)

    def analyze(self) -> ModuleInfo:
        """完全分析実行"""
        imports = self._extract_imports()
        constants = self._extract_constants()
        global_vars = self._extract_global_variables()
        classes = self._analyze_classes()
        global_functions = self._analyze_global_functions()

        return ModuleInfo(
            imports=imports,
            global_functions=global_functions,
            classes=classes,
            constants=constants,
            global_variables=global_vars,
            total_lines=len(self.lines)
        )

    def _extract_imports(self) -> List[str]:
        """インポート文抽出"""
        imports = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        return imports

    def _extract_constants(self) -> List[str]:
        """定数抽出（大文字のみの変数）"""
        constants = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)
        return constants

    def _extract_global_variables(self) -> List[str]:
        """グローバル変数抽出"""
        variables = []
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
        return variables

    def _analyze_classes(self) -> List[ClassInfo]:
        """クラス分析"""
        classes = []
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                classes.append(class_info)
        return classes

    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """個別クラス分析"""
        methods = []
        dependencies = set()

        # メソッド分析
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_info = self._analyze_function(child, is_method=True, class_name=node.name)
                methods.append(method_info)
                dependencies.update(method_info.dependencies)

        # 基底クラス
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)

        # docstring抽出
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value

        return ClassInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
            methods=methods,
            dependencies=dependencies,
            base_classes=base_classes,
            docstring=docstring
        )

    def _analyze_global_functions(self) -> List[FunctionInfo]:
        """グローバル関数分析"""
        functions = []
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node, is_method=False)
                functions.append(func_info)
        return functions

    def _analyze_function(self, node: ast.FunctionDef, is_method: bool = False,
                         class_name: Optional[str] = None) -> FunctionInfo:
        """個別関数分析"""
        # 依存関係抽出
        dependencies = self._extract_function_dependencies(node)

        # 複雑度計算（簡易版）
        complexity = self._calculate_complexity(node)

        # パラメータ抽出
        parameters = [arg.arg for arg in node.args.args]

        # 戻り値アノテーション
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        # docstring抽出
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value

        return FunctionInfo(
            name=node.name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            line_count=(node.end_lineno or node.lineno) - node.lineno + 1,
            complexity=complexity,
            dependencies=dependencies,
            is_method=is_method,
            class_name=class_name,
            docstring=docstring,
            parameters=parameters,
            return_annotation=return_annotation
        )

    def _extract_function_dependencies(self, node: ast.FunctionDef) -> Set[str]:
        """関数の依存関係抽出"""
        dependencies = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.add(f"{child.value.id}.{child.attr}")
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        dependencies.add(f"{child.func.value.id}.{child.func.attr}")

        return dependencies

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """サイクロマティック複雑度計算（簡易版）"""
        complexity = 1  # 基本パス

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def generate_refactoring_plan(self, analysis: ModuleInfo) -> Dict[str, any]:
        """リファクタリング計画生成"""
        plan = {
            "overview": {
                "total_lines": analysis.total_lines,
                "total_classes": len(analysis.classes),
                "total_functions": len(analysis.global_functions),
                "total_methods": sum(len(cls.methods) for cls in analysis.classes)
            },
            "target_modules": {},
            "extraction_strategy": {},
            "dependencies": {},
            "priority_order": []
        }

        # 1. TradeManager クラス分析
        trade_manager_class = None
        for cls in analysis.classes:
            if cls.name == "TradeManager":
                trade_manager_class = cls
                break

        if trade_manager_class:
            plan["target_modules"]["trade_manager_core"] = self._plan_core_module(trade_manager_class)
            plan["target_modules"]["trade_manager_execution"] = self._plan_execution_module(trade_manager_class)
            plan["target_modules"]["trade_manager_validation"] = self._plan_validation_module(trade_manager_class)
            plan["target_modules"]["trade_manager_monitoring"] = self._plan_monitoring_module(trade_manager_class)

        # 2. モデルクラス分離計画
        model_classes = [cls for cls in analysis.classes if cls.name in ["Trade", "Position", "BuyLot", "RealizedPnL", "TradeStatus"]]
        if model_classes:
            plan["target_modules"]["trade_models"] = self._plan_models_module(model_classes)

        # 3. ユーティリティ関数分離計画
        utility_functions = analysis.global_functions
        if utility_functions:
            plan["target_modules"]["trade_utils"] = self._plan_utils_module(utility_functions)

        # 4. 優先順位設定
        plan["priority_order"] = [
            "trade_models",      # 1. モデル分離（依存関係最小）
            "trade_utils",       # 2. ユーティリティ分離
            "trade_manager_core", # 3. コア機能
            "trade_manager_validation", # 4. バリデーション
            "trade_manager_execution",  # 5. 実行機能
            "trade_manager_monitoring"  # 6. 監視機能
        ]

        return plan

    def _plan_core_module(self, cls: ClassInfo) -> Dict[str, any]:
        """コアモジュール計画"""
        core_methods = []
        for method in cls.methods:
            if any(keyword in method.name.lower() for keyword in [
                "__init__", "reset", "clear", "save", "load", "export", "import",
                "get_position", "get_trades", "get_summary", "calculate"
            ]):
                core_methods.append({
                    "name": method.name,
                    "lines": method.line_count,
                    "complexity": method.complexity
                })

        return {
            "description": "取引管理の基本機能とデータアクセス",
            "estimated_lines": sum(m["lines"] for m in core_methods),
            "methods": core_methods,
            "dependencies": ["trade_models", "trade_utils"]
        }

    def _plan_execution_module(self, cls: ClassInfo) -> Dict[str, any]:
        """実行モジュール計画"""
        execution_methods = []
        for method in cls.methods:
            if any(keyword in method.name.lower() for keyword in [
                "buy", "sell", "add_trade", "update_position", "process"
            ]):
                execution_methods.append({
                    "name": method.name,
                    "lines": method.line_count,
                    "complexity": method.complexity
                })

        return {
            "description": "取引実行とポジション更新",
            "estimated_lines": sum(m["lines"] for m in execution_methods),
            "methods": execution_methods,
            "dependencies": ["trade_models", "trade_utils", "trade_manager_core"]
        }

    def _plan_validation_module(self, cls: ClassInfo) -> Dict[str, any]:
        """バリデーションモジュール計画"""
        validation_methods = []
        for method in cls.methods:
            if any(keyword in method.name.lower() for keyword in [
                "validate", "check", "verify", "_is_valid", "_ensure"
            ]):
                validation_methods.append({
                    "name": method.name,
                    "lines": method.line_count,
                    "complexity": method.complexity
                })

        return {
            "description": "入力検証とビジネスルール検証",
            "estimated_lines": sum(m["lines"] for m in validation_methods),
            "methods": validation_methods,
            "dependencies": ["trade_models", "trade_utils"]
        }

    def _plan_monitoring_module(self, cls: ClassInfo) -> Dict[str, any]:
        """監視モジュール計画"""
        monitoring_methods = []
        for method in cls.methods:
            if any(keyword in method.name.lower() for keyword in [
                "log", "monitor", "track", "audit", "report"
            ]):
                monitoring_methods.append({
                    "name": method.name,
                    "lines": method.line_count,
                    "complexity": method.complexity
                })

        return {
            "description": "ログ出力とパフォーマンス監視",
            "estimated_lines": sum(m["lines"] for m in monitoring_methods),
            "methods": monitoring_methods,
            "dependencies": ["trade_models", "trade_utils"]
        }

    def _plan_models_module(self, classes: List[ClassInfo]) -> Dict[str, any]:
        """モデルモジュール計画"""
        model_info = []
        total_lines = 0

        for cls in classes:
            total_lines += cls.line_count
            model_info.append({
                "name": cls.name,
                "lines": cls.line_count,
                "methods": len(cls.methods)
            })

        return {
            "description": "取引関連のデータモデル",
            "estimated_lines": total_lines,
            "classes": model_info,
            "dependencies": []
        }

    def _plan_utils_module(self, functions: List[FunctionInfo]) -> Dict[str, any]:
        """ユーティリティモジュール計画"""
        util_functions = []
        total_lines = 0

        for func in functions:
            total_lines += func.line_count
            util_functions.append({
                "name": func.name,
                "lines": func.line_count,
                "complexity": func.complexity
            })

        return {
            "description": "共通ユーティリティ関数",
            "estimated_lines": total_lines,
            "functions": util_functions,
            "dependencies": []
        }


def main():
    """メイン実行"""
    file_path = "C:/gemini-thinkpad/day_trade/src/day_trade/core/trade_manager.py"

    print("Trade Manager 分析開始...")
    analyzer = TradeManagerAnalyzer(file_path)

    # 分析実行
    analysis = analyzer.analyze()

    # リファクタリング計画生成
    plan = analyzer.generate_refactoring_plan(analysis)

    # 結果出力
    print(f"\n分析結果:")
    print(f"  総行数: {analysis.total_lines:,} 行")
    print(f"  クラス数: {len(analysis.classes)} 個")
    print(f"  グローバル関数数: {len(analysis.global_functions)} 個")
    print(f"  総メソッド数: {sum(len(cls.methods) for cls in analysis.classes)} 個")

    print(f"\nクラス一覧:")
    for cls in analysis.classes:
        print(f"  - {cls.name}: {cls.line_count} 行, {len(cls.methods)} メソッド")

    print(f"\nリファクタリング計画:")
    for module_name in plan["priority_order"]:
        if module_name in plan["target_modules"]:
            module_plan = plan["target_modules"][module_name]
            print(f"  {len(plan['priority_order']) - plan['priority_order'].index(module_name)}. {module_name}")
            print(f"     説明: {module_plan['description']}")
            print(f"     予想行数: {module_plan['estimated_lines']} 行")
            print(f"     依存関係: {', '.join(module_plan['dependencies']) if module_plan['dependencies'] else 'なし'}")

    # 詳細計画をJSONで保存
    plan_file = "trade_manager_refactoring_plan.json"
    with open(plan_file, 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n詳細計画を {plan_file} に保存しました")
    print("分析完了")


if __name__ == "__main__":
    main()