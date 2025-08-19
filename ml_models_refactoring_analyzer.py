#!/usr/bin/env python3
"""
ML Prediction Models リファクタリング分析ツール

ml_prediction_models_improved.py (2,303行) の構造分析と
リファクタリング計画の策定
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ClassInfo:
    name: str
    start_line: int
    end_line: int
    methods: List[str]
    line_count: int
    inheritance: List[str]


@dataclass
class ModuleAnalysis:
    total_lines: int
    classes: List[ClassInfo]
    functions: List[str]
    imports: List[str]
    complexity_score: int


def analyze_ml_models_file() -> ModuleAnalysis:
    """ML予測モデルファイルの構造分析"""
    file_path = Path("src/day_trade/ml/ml_prediction_models_improved.py")
    
    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"構文エラー: {e}")
        return ModuleAnalysis(len(lines), [], [], [], 0)
    
    classes = []
    functions = []
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # クラス情報収集
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(item.name)
            
            # 継承情報
            inheritance = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    inheritance.append(base.id)
                elif isinstance(base, ast.Attribute):
                    inheritance.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else str(base.attr))
            
            # 行数計算
            start_line = node.lineno
            end_line = max([getattr(item, 'end_lineno', item.lineno) for item in node.body] + [node.lineno])
            line_count = end_line - start_line + 1
            
            classes.append(ClassInfo(
                name=node.name,
                start_line=start_line,
                end_line=end_line,
                methods=methods,
                line_count=line_count,
                inheritance=inheritance
            ))
        
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            # トップレベル関数
            functions.append(node.name)
        
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # インポート文
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            else:
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
    
    # 複雑度スコア計算
    complexity_score = len(classes) * 10 + len(functions) * 3 + len(imports)
    
    return ModuleAnalysis(
        total_lines=len(lines),
        classes=classes,
        functions=functions,
        imports=imports,
        complexity_score=complexity_score
    )


def generate_refactoring_plan(analysis: ModuleAnalysis) -> Dict[str, Any]:
    """リファクタリング計画策定"""
    
    # クラスをカテゴリ別に分類
    model_classes = []
    data_classes = []
    config_classes = []
    utility_classes = []
    exception_classes = []
    
    for cls in analysis.classes:
        cls_name_lower = cls.name.lower()
        
        if any(keyword in cls_name_lower for keyword in ['error', 'exception']):
            exception_classes.append(cls)
        elif any(keyword in cls_name_lower for keyword in ['config', 'settings', 'param']):
            config_classes.append(cls)
        elif any(keyword in cls_name_lower for keyword in ['model', 'predictor', 'classifier', 'regressor']):
            model_classes.append(cls)
        elif any(keyword in cls_name_lower for keyword in ['data', 'feature', 'preprocessing', 'engineer']):
            data_classes.append(cls)
        else:
            utility_classes.append(cls)
    
    # モジュール分離計画
    modules = {
        "ml_exceptions.py": {
            "description": "ML予測システムの例外クラス",
            "classes": exception_classes,
            "estimated_lines": sum(cls.line_count for cls in exception_classes)
        },
        "ml_config.py": {
            "description": "ML設定とパラメータ管理",
            "classes": config_classes,
            "estimated_lines": sum(cls.line_count for cls in config_classes)
        },
        "ml_data_processing.py": {
            "description": "データ処理と特徴量エンジニアリング",
            "classes": data_classes,
            "estimated_lines": sum(cls.line_count for cls in data_classes)
        },
        "ml_model_base.py": {
            "description": "ベースモデルクラスと共通機能",
            "classes": [cls for cls in model_classes if any(keyword in cls.name.lower() for keyword in ['base', 'abstract', 'interface'])],
            "estimated_lines": sum(cls.line_count for cls in model_classes if any(keyword in cls.name.lower() for keyword in ['base', 'abstract', 'interface']))
        },
        "ml_prediction_models.py": {
            "description": "具体的な予測モデル実装",
            "classes": [cls for cls in model_classes if not any(keyword in cls.name.lower() for keyword in ['base', 'abstract', 'interface'])],
            "estimated_lines": sum(cls.line_count for cls in model_classes if not any(keyword in cls.name.lower() for keyword in ['base', 'abstract', 'interface']))
        },
        "ml_utilities.py": {
            "description": "ユーティリティクラスとヘルパー機能",
            "classes": utility_classes,
            "estimated_lines": sum(cls.line_count for cls in utility_classes)
        }
    }
    
    # 残り行数計算（関数、インポート、その他）
    total_class_lines = sum(cls.line_count for cls in analysis.classes)
    remaining_lines = analysis.total_lines - total_class_lines
    
    return {
        "original_file": {
            "total_lines": analysis.total_lines,
            "total_classes": len(analysis.classes),
            "total_functions": len(analysis.functions),
            "complexity_score": analysis.complexity_score
        },
        "categorization": {
            "model_classes": len(model_classes),
            "data_classes": len(data_classes),
            "config_classes": len(config_classes),
            "utility_classes": len(utility_classes),
            "exception_classes": len(exception_classes)
        },
        "modules": modules,
        "remaining_lines": remaining_lines,
        "phase_plan": {
            "phase_1": "ml_exceptions.py と ml_config.py の抽出",
            "phase_2": "ml_data_processing.py の分離",
            "phase_3": "ml_model_base.py の作成",
            "phase_4": "ml_prediction_models.py と ml_utilities.py の分離",
            "phase_5": "統合テストと検証"
        }
    }


def print_analysis_results(analysis: ModuleAnalysis, plan: Dict[str, Any]):
    """分析結果とリファクタリング計画の出力"""
    
    print("ML Prediction Models リファクタリング分析")
    print("=" * 60)
    print(f"ファイル: src/day_trade/ml/ml_prediction_models_improved.py")
    print(f"総行数: {analysis.total_lines:,}")
    print(f"クラス数: {len(analysis.classes)}")
    print(f"関数数: {len(analysis.functions)}")
    print(f"複雑度: {analysis.complexity_score}")
    print()
    
    print("クラス詳細:")
    print("-" * 40)
    for cls in sorted(analysis.classes, key=lambda x: x.line_count, reverse=True):
        inheritance_info = f" (継承: {', '.join(cls.inheritance)})" if cls.inheritance else ""
        print(f"  {cls.name}: {cls.line_count}行, {len(cls.methods)}メソッド{inheritance_info}")
    print()
    
    print("カテゴリ分類:")
    print("-" * 40)
    cat = plan["categorization"]
    print(f"  モデルクラス: {cat['model_classes']}個")
    print(f"  データクラス: {cat['data_classes']}個")
    print(f"  設定クラス: {cat['config_classes']}個")
    print(f"  ユーティリティクラス: {cat['utility_classes']}個")
    print(f"  例外クラス: {cat['exception_classes']}個")
    print()
    
    print("提案されるモジュール分離:")
    print("-" * 40)
    for module_name, module_info in plan["modules"].items():
        if module_info["classes"]:
            print(f"  {module_name} ({module_info['estimated_lines']}行)")
            print(f"    - {module_info['description']}")
            for cls in module_info["classes"]:
                print(f"      * {cls.name} ({cls.line_count}行)")
            print()
    
    print("リファクタリングフェーズ:")
    print("-" * 40)
    for phase, description in plan["phase_plan"].items():
        print(f"  {phase}: {description}")
    print()
    
    # 難易度評価
    total_classes = len(analysis.classes)
    if total_classes > 15:
        difficulty = "非常に高い"
    elif total_classes > 10:
        difficulty = "高い"
    elif total_classes > 5:
        difficulty = "中程度"
    else:
        difficulty = "低い"
    
    print(f"リファクタリング難易度: {difficulty}")
    print(f"推定作業時間: {total_classes * 15}分 - {total_classes * 25}分")


def main():
    """メイン実行関数"""
    try:
        print("ML Prediction Models 構造分析開始...")
        print()
        
        analysis = analyze_ml_models_file()
        plan = generate_refactoring_plan(analysis)
        
        print_analysis_results(analysis, plan)
        
    except Exception as e:
        print(f"分析エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()