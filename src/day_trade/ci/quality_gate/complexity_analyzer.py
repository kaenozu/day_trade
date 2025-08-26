#!/usr/bin/env python3
"""
品質ゲートシステム - コード複雑度解析

コード品質評価におけるMcCabe複雑度とハルステッド複雑度の計算、
保守性指標の算出を提供するモジュール。
"""

import ast
import math
from typing import Any, Dict, Set


class ComplexityVisitor(ast.NodeVisitor):
    """McCabe複雑度計算Visitor
    
    ASTを走査してMcCabe循環的複雑度を計算するVisitor。
    条件分岐、ループ、例外処理などの制御フロー構造を
    カウントして複雑度を算出する。
    """

    def __init__(self):
        """初期化
        
        複雑度の基本値を1に設定し、
        関数とクラスの情報を格納するリストを初期化。
        """
        self.complexity = 1  # 基本複雑度
        self.functions = []
        self.classes = []
        self.current_function = None
        self.current_class = None

    def visit_FunctionDef(self, node):
        """関数定義の複雑度を計算
        
        関数内の制御フロー構造をカウントして
        関数レベルの複雑度を算出する。
        
        Args:
            node: 関数定義のASTノード
        """
        self.current_function = node.name
        function_complexity = 1

        for child in ast.walk(node):
            if isinstance(
                child,
                (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith),
            ):
                function_complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                function_complexity += 1
            elif isinstance(child, ast.BoolOp):
                function_complexity += len(child.values) - 1

        self.functions.append(
            {
                "name": node.name,
                "complexity": function_complexity,
                "line_number": node.lineno,
            }
        )

        self.complexity += function_complexity - 1
        self.generic_visit(node)
        self.current_function = None

    def visit_ClassDef(self, node):
        """クラス定義を記録
        
        クラス情報を記録し、内部のメソッドを処理する。
        
        Args:
            node: クラス定義のASTノード
        """
        self.current_class = node.name
        self.classes.append(
            {"name": node.name, "line_number": node.lineno, "methods": []}
        )
        self.generic_visit(node)
        self.current_class = None

    def visit_If(self, node):
        """If文の複雑度をカウント"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """While文の複雑度をカウント"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """For文の複雑度をカウント"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """例外ハンドラの複雑度をカウント"""
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        """ブール演算子の複雑度をカウント"""
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class HalsteadVisitor(ast.NodeVisitor):
    """ハルステッドメトリクス計算Visitor
    
    ASTを走査してハルステッドソフトウェア科学の
    メトリクスを計算するVisitor。オペレータと
    オペランドの出現頻度を記録する。
    """

    def __init__(self):
        """初期化
        
        オペレータとオペランドの集合、
        および出現回数のカウンタを初期化。
        """
        self.operators = set()
        self.operands = set()
        self.operator_count = 0
        self.operand_count = 0

    def visit_BinOp(self, node):
        """二項演算子を記録"""
        self.operators.add(type(node.op).__name__)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        """単項演算子を記録"""
        self.operators.add(type(node.op).__name__)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        """比較演算子を記録"""
        for op in node.ops:
            self.operators.add(type(op).__name__)
            self.operator_count += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        """変数名を記録"""
        self.operands.add(node.id)
        self.operand_count += 1
        self.generic_visit(node)

    def visit_Constant(self, node):
        """定数を記録"""
        self.operands.add(repr(node.value))
        self.operand_count += 1
        self.generic_visit(node)


class CodeComplexityAnalyzer:
    """コード複雑度解析システム
    
    Pythonソースコードの複雑度を多角的に分析するクラス。
    McCabe複雑度、ハルステッド複雑度、保守性指標を算出し、
    コード品質の定量的な評価を提供する。
    """

    def __init__(self):
        """初期化
        
        複雑度計算結果のキャッシュを初期化。
        """
        self.complexity_cache = {}

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """ファイルの複雑度を解析
        
        指定されたPythonファイルを解析し、
        各種複雑度メトリクスを算出する。
        
        Args:
            file_path: 解析対象のファイルパス
            
        Returns:
            複雑度解析結果を含む辞書。エラーが発生した場合は
            エラー情報を含む辞書を返す。
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            analyzer = ComplexityVisitor()
            analyzer.visit(tree)

            # McCabe複雑度計算
            mccabe_complexity = analyzer.complexity

            # ハルステッド複雑度計算
            halstead_metrics = self._calculate_halstead_metrics(content)

            # 保守性指標計算
            maintainability_index = self._calculate_maintainability_index(
                mccabe_complexity, halstead_metrics, len(content.splitlines())
            )

            return {
                "file_path": file_path,
                "lines_of_code": len(
                    [
                        line
                        for line in content.splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                ),
                "mccabe_complexity": mccabe_complexity,
                "halstead_metrics": halstead_metrics,
                "maintainability_index": maintainability_index,
                "functions": analyzer.functions,
                "classes": analyzer.classes,
            }

        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "lines_of_code": 0,
                "mccabe_complexity": 0,
                "maintainability_index": 0,
            }

    def _calculate_halstead_metrics(self, content: str) -> Dict[str, float]:
        """ハルステッド複雑度メトリクスを計算
        
        ソースコードからハルステッドメトリクスの各指標を算出。
        プログラムの語彙、長さ、ボリューム、難易度、労力を計算する。
        
        Args:
            content: ソースコード文字列
            
        Returns:
            ハルステッドメトリクスの各指標を含む辞書
        """
        try:
            tree = ast.parse(content)
            visitor = HalsteadVisitor()
            visitor.visit(tree)

            # ハルステッド指標
            n1 = len(visitor.operators)  # ユニークオペレータ数
            n2 = len(visitor.operands)  # ユニークオペランド数
            N1 = visitor.operator_count  # 総オペレータ数
            N2 = visitor.operand_count  # 総オペランド数

            if n1 == 0 or n2 == 0:
                return {
                    "vocabulary": 0,
                    "length": 0,
                    "volume": 0,
                    "difficulty": 0,
                    "effort": 0,
                }

            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            return {
                "vocabulary": vocabulary,
                "length": length,
                "volume": volume,
                "difficulty": difficulty,
                "effort": effort,
            }

        except Exception:
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
            }

    def _calculate_maintainability_index(
        self, complexity: float, halstead: Dict[str, float], loc: int
    ) -> float:
        """保守性指標を計算
        
        McCabe複雑度、ハルステッドボリューム、コード行数から
        保守性指標（Maintainability Index）を算出。
        
        Args:
            complexity: McCabe複雑度
            halstead: ハルステッドメトリクス
            loc: コード行数
            
        Returns:
            0-100の保守性指標値
        """
        try:
            # 標準的な保守性指標計算
            # MI = 171 - 5.2 * ln(HalsteadVolume) - 0.23 * CyclomaticComplexity - 16.2 * ln(LinesOfCode)

            volume = max(halstead.get("volume", 1), 1)
            complexity = max(complexity, 1)
            loc = max(loc, 1)

            mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)

            # 0-100にスケール調整
            return max(0, min(100, mi))

        except Exception:
            return 50.0  # デフォルト値