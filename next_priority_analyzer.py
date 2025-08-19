#!/usr/bin/env python3
"""
次の優先度ファイル分析器

大型ファイルを特定し、リファクタリング優先度を評価
"""

import os
import ast
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import subprocess


@dataclass
class FileAnalysis:
    path: str
    lines: int
    functions: int
    classes: int
    complexity_score: int
    priority_score: int
    refactor_difficulty: str


def count_lines(filepath: Path) -> int:
    """ファイルの行数をカウント"""
    try:
        return len(filepath.read_text(encoding='utf-8').splitlines())
    except:
        try:
            return len(filepath.read_text(encoding='cp932').splitlines())
        except:
            return 0


def analyze_python_file(filepath: Path) -> Tuple[int, int, int]:
    """Pythonファイルの構造を解析"""
    try:
        content = filepath.read_text(encoding='utf-8')
        tree = ast.parse(content)
        
        functions = 0
        classes = 0
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
                # 関数の複雑度を簡易計算
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                        complexity += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
                
        return functions, classes, complexity
        
    except:
        return 0, 0, 1


def calculate_priority_score(analysis: FileAnalysis) -> int:
    """優先度スコアを計算"""
    score = 0
    
    # ファイルサイズによるスコア
    if analysis.lines > 2000:
        score += 100
    elif analysis.lines > 1000:
        score += 50
    elif analysis.lines > 500:
        score += 25
    
    # 関数数によるスコア
    if analysis.functions > 30:
        score += 30
    elif analysis.functions > 20:
        score += 20
    elif analysis.functions > 10:
        score += 10
    
    # クラス数によるスコア
    if analysis.classes > 5:
        score += 20
    elif analysis.classes > 3:
        score += 10
    
    # 複雑度によるスコア
    if analysis.complexity_score > 50:
        score += 25
    elif analysis.complexity_score > 30:
        score += 15
    
    # 重要なファイル名パターンのボーナス
    filename = Path(analysis.path).name.lower()
    if any(keyword in filename for keyword in ['main', 'core', 'manager', 'engine', 'system']):
        score += 20
    
    return score


def determine_difficulty(analysis: FileAnalysis) -> str:
    """リファクタリング難易度を判定"""
    total_elements = analysis.functions + analysis.classes
    
    if analysis.lines > 2000 and total_elements > 40:
        return "CRITICAL"
    elif analysis.lines > 1000 and total_elements > 25:
        return "HIGH"
    elif analysis.lines > 500 and total_elements > 15:
        return "MEDIUM"
    else:
        return "LOW"


def main():
    """メイン分析実行"""
    print("次の優先度ファイル分析開始...")
    
    # 対象ディレクトリ
    base_dir = Path("C:/gemini-thinkpad/day_trade")
    
    # 除外するディレクトリとファイル
    exclude_dirs = {
        '.git', '__pycache__', 'node_modules', 'venv', '.venv',
        'refactoring_backup', 'htmlcov', '.pytest_cache', 'dist', 'build'
    }
    
    exclude_files = {
        'trade_manager.py'  # 既にリファクタリング済み
    }
    
    analyses = []
    
    # Pythonファイルを再帰的に検索
    for py_file in base_dir.rglob("*.py"):
        # 除外条件チェック
        if any(exclude_dir in py_file.parts for exclude_dir in exclude_dirs):
            continue
        if py_file.name in exclude_files:
            continue
        if py_file.name.startswith('test_') and py_file.stat().st_size < 10000:
            continue  # 小さなテストファイルをスキップ
            
        # ファイル分析
        lines = count_lines(py_file)
        if lines < 200:  # 200行未満は除外
            continue
            
        functions, classes, complexity = analyze_python_file(py_file)
        
        analysis = FileAnalysis(
            path=str(py_file.relative_to(base_dir)),
            lines=lines,
            functions=functions,
            classes=classes,
            complexity_score=complexity,
            priority_score=0,
            refactor_difficulty=""
        )
        
        analysis.priority_score = calculate_priority_score(analysis)
        analysis.refactor_difficulty = determine_difficulty(analysis)
        
        analyses.append(analysis)
    
    # 優先度順にソート
    analyses.sort(key=lambda x: x.priority_score, reverse=True)
    
    print(f"\n分析結果: {len(analyses)} ファイル")
    print("=" * 80)
    
    print("\nTOP 10 優先度ファイル:")
    for i, analysis in enumerate(analyses[:10], 1):
        print(f"{i:2d}. {analysis.path}")
        print(f"    行数: {analysis.lines:,} | 関数: {analysis.functions} | クラス: {analysis.classes}")
        print(f"    複雑度: {analysis.complexity_score} | 優先度: {analysis.priority_score} | 難易度: {analysis.refactor_difficulty}")
        print()
    
    # CRITICAL と HIGH 難易度のファイルを特に強調
    critical_files = [a for a in analyses if a.refactor_difficulty in ["CRITICAL", "HIGH"]]
    if critical_files:
        print(f"\n緊急対応が必要なファイル ({len(critical_files)} 件):")
        print("=" * 50)
        for analysis in critical_files[:5]:
            print(f"⚠️  {analysis.path}")
            print(f"    {analysis.lines:,} 行 | 難易度: {analysis.refactor_difficulty}")
            print(f"    推定工数: {analysis.lines // 100} 日")
            print()
    
    # 次の推奨作業
    if analyses:
        next_target = analyses[0]
        print("推奨次期リファクタリング対象:")
        print("=" * 40)
        print(f"ファイル: {next_target.path}")
        print(f"理由: {next_target.lines:,}行、{next_target.functions}関数、優先度{next_target.priority_score}")
        print(f"推定工数: {next_target.lines // 150} 日")
        print(f"難易度: {next_target.refactor_difficulty}")


if __name__ == "__main__":
    main()