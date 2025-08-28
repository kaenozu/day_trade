#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trade システム カバレッジ分析
"""

import os
import sys
from pathlib import Path
import ast
from collections import defaultdict

def count_lines_of_code(file_path):
    """コード行数をカウント"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # コメント行と空行を除く
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                code_lines += 1
        
        return len(lines), code_lines
    except:
        return 0, 0

def analyze_coverage():
    """カバレッジを分析"""
    project_root = Path('.')
    
    # ソースファイル統計
    src_files = list(project_root.glob('src/**/*.py'))
    web_files = list(project_root.glob('web/**/*.py')) 
    backend_files = list(project_root.glob('backend/**/*.py'))
    
    # テストファイル統計  
    test_files = list(project_root.glob('tests/**/*.py'))
    
    # モジュール別統計
    module_stats = defaultdict(lambda: {'src_files': 0, 'test_files': 0, 'src_lines': 0, 'test_lines': 0})
    
    print("Day Trade システム カバレッジ分析")
    print("=" * 50)
    
    # ソースファイル分析
    total_src_files = len(src_files) + len(web_files) + len(backend_files)
    total_src_lines = 0
    total_src_code_lines = 0
    
    for file_path in src_files + web_files + backend_files:
        total_lines, code_lines = count_lines_of_code(file_path)
        total_src_lines += total_lines
        total_src_code_lines += code_lines
        
        # モジュール判定
        path_parts = file_path.parts
        if 'src' in path_parts:
            module = path_parts[path_parts.index('src') + 2] if len(path_parts) > 3 else 'core'
        elif 'web' in path_parts:
            module = 'web'
        elif 'backend' in path_parts:
            module = 'backend'
        else:
            module = 'other'
            
        module_stats[module]['src_files'] += 1
        module_stats[module]['src_lines'] += code_lines
    
    # テストファイル分析  
    total_test_files = len(test_files)
    total_test_lines = 0
    total_test_code_lines = 0
    
    for file_path in test_files:
        total_lines, code_lines = count_lines_of_code(file_path)
        total_test_lines += total_lines
        total_test_code_lines += code_lines
        
        # テスト対象モジュール推定
        name = file_path.stem.replace('test_', '')
        
        # モジュールマッピング
        module_mapping = {
            'trade_manager': 'trading',
            'stock_fetcher': 'data',
            'database': 'models',
            'ml_': 'ml',
            'ensemble': 'ml',
            'hyperparameter': 'ml',
            'performance': 'performance',
            'integration': 'integration',
            'backtest': 'backtest',
            'alerts': 'alerts'
        }
        
        module = 'other'
        for key, mod in module_mapping.items():
            if key in name:
                module = mod
                break
        
        module_stats[module]['test_files'] += 1
        module_stats[module]['test_lines'] += code_lines
    
    # 統計出力
    print(f"\n全体統計:")
    print(f"   ソースファイル数: {total_src_files:,}")
    print(f"   ソースコード行数: {total_src_code_lines:,}")
    print(f"   テストファイル数: {total_test_files:,}")  
    print(f"   テストコード行数: {total_test_code_lines:,}")
    
    # カバレッジ比率
    test_to_src_ratio = total_test_files / total_src_files if total_src_files > 0 else 0
    test_lines_ratio = total_test_code_lines / total_src_code_lines if total_src_code_lines > 0 else 0
    
    print(f"\nカバレッジ指標:")
    print(f"   テスト/ソース ファイル比: {test_to_src_ratio:.2f} ({test_to_src_ratio*100:.1f}%)")
    print(f"   テスト/ソース 行数比: {test_lines_ratio:.2f} ({test_lines_ratio*100:.1f}%)")
    
    # モジュール別詳細
    print(f"\nモジュール別カバレッジ:")
    print(f"{'モジュール':<15} {'ソース':<8} {'テスト':<8} {'カバレッジ':<10} {'品質':<10}")
    print("-" * 60)
    
    module_coverage = {}
    for module, stats in sorted(module_stats.items()):
        src_files = stats['src_files'] 
        test_files = stats['test_files']
        coverage = test_files / src_files if src_files > 0 else 0
        
        # 品質評価
        if coverage >= 0.8:
            quality = "優秀"
        elif coverage >= 0.5:
            quality = "良好"
        elif coverage >= 0.3:
            quality = "普通"
        elif coverage > 0:
            quality = "要改善"
        else:
            quality = "未テスト"
        
        module_coverage[module] = coverage
        print(f"{module:<15} {src_files:<8} {test_files:<8} {coverage*100:>7.1f}% {quality:<10}")
    
    # 最適化システムのカバレッジ
    print(f"\n最適化システム分析:")
    optimization_files = list(project_root.glob('src/day_trade/optimization/*.py'))
    optimization_tests = list(project_root.glob('tests/*optimization*.py')) + \
                        list(project_root.glob('tests/**/*optimization*.py'))
    
    print(f"   最適化モジュール数: {len(optimization_files)}")
    print(f"   最適化テスト数: {len(optimization_tests)}")
    
    if optimization_files:
        opt_coverage = len(optimization_tests) / len(optimization_files)
        print(f"   最適化カバレッジ: {opt_coverage*100:.1f}%")
    
    # 推奨改善点
    print(f"\n改善推奨:")
    
    low_coverage_modules = [mod for mod, cov in module_coverage.items() 
                           if 0 < cov < 0.5 and module_stats[mod]['src_files'] > 5]
    
    if low_coverage_modules:
        print(f"   テストカバレッジ向上が必要: {', '.join(low_coverage_modules)}")
    
    untested_modules = [mod for mod, cov in module_coverage.items() 
                       if cov == 0 and module_stats[mod]['src_files'] > 3]
    
    if untested_modules:
        print(f"   テスト未実装モジュール: {', '.join(untested_modules)}")
    
    if test_to_src_ratio < 0.3:
        print(f"   全体的なテストファイル数が不足しています")
    
    if test_lines_ratio < 0.4:  
        print(f"   テストコードのボリュームが不足しています")
    
    # 品質スコア算出
    coverage_score = min(100, test_to_src_ratio * 100 + test_lines_ratio * 50)
    quality_modules = sum(1 for cov in module_coverage.values() if cov >= 0.5)
    module_score = (quality_modules / len(module_coverage)) * 100 if module_coverage else 0
    
    overall_score = (coverage_score + module_score) / 2
    
    print(f"\n総合品質スコア: {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("   評価: 優秀 - 高品質なテスト体制です")
    elif overall_score >= 60:
        print("   評価: 良好 - 基本的なテスト体制が整っています")  
    elif overall_score >= 40:
        print("   評価: 普通 - テスト体制の改善が必要です")
    else:
        print("   評価: 要改善 - テスト体制の大幅な強化が必要です")
        
    return {
        'total_src_files': total_src_files,
        'total_test_files': total_test_files, 
        'coverage_ratio': test_to_src_ratio,
        'quality_score': overall_score,
        'module_coverage': module_coverage
    }

if __name__ == "__main__":
    analyze_coverage()