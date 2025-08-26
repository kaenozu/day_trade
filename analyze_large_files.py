import os
import subprocess
import sys

# 既に完全に分割済みのディレクトリのファイル
split_files_with_directories = [
    'src/day_trade/analysis/signals.py',  # signals/ directory exists
    'src/day_trade/analysis/ensemble.py',  # ensemble/ directory exists
    'src/day_trade/data/stock_master.py',  # stock_master/ directory exists
    'src/day_trade/data/data_version_manager.py',  # version_manager/ directory exists
    'src/day_trade/automation/orchestrator.py',  # orchestrator/ directory exists
    'src/day_trade/core/trade_manager/manager.py',  # already in trade_manager/
    'src/day_trade/models/database.py',  # database/ directory exists
    'src/day_trade/api/external_api_client.py',  # external_api/ directory exists
    'src/day_trade/data/data_freshness_monitor.py',  # freshness_monitor/ directory exists
    'src/day_trade/data/enterprise_master_data_management.py',  # enterprise_master/ directory exists
    'src/day_trade/data/master_data_manager.py',  # master_data/ directory exists
    'src/day_trade/ml/feature_store_backup.py',  # backup file
    'src/day_trade/ml/base_models/base_model_interface.py',  # already in base_models/
    'src/day_trade/cache/persistent_cache_system.py',  # already in cache/
    'src/day_trade/ensemble/performance_analyzer.py',  # already in ensemble/
    'src/day_trade/ensemble/ensemble_optimizer.py',  # already in ensemble/
    'src/day_trade/analysis/backtest/backtest_engine.py',  # already in backtest/
]

# 後方互換性ファイル（分割済みで統合ファイルが残されているもの）
compatibility_files = [
    'src/day_trade/ml/gpu_accelerated_inference.py'  # gpu_inference/ directory exists
]

def analyze_file_content(filepath):
    """ファイルの内容を分析して分割の必要性を判断"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 後方互換性ファイルかチェック
        compatibility_markers = [
            '後方互換性', 'backward compatibility', 'compatibility', '統合インポート',
            'from .', 'from ..', '新しいモジュール構造', 'モジュラー構造'
        ]
        
        is_compatibility = any(marker in content for marker in compatibility_markers)
        
        # クラス数をカウント
        class_count = content.count('class ')
        
        # 関数数をカウント（メソッドを除く）
        lines = content.split('\n')
        function_count = 0
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                # インデントレベルをチェックしてトップレベル関数のみカウント
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces == 0:
                    function_count += 1
        
        return {
            'is_compatibility': is_compatibility,
            'class_count': class_count,
            'function_count': function_count,
            'needs_splitting': class_count > 3 or function_count > 10
        }
    except Exception as e:
        return {'error': str(e)}

# 入力から1000行以上のファイルを処理
results = []

for line in sys.stdin:
    line = line.strip()
    if not line or 'total' in line:
        continue
        
    parts = line.split()
    if len(parts) >= 2:
        lines = int(parts[0])
        filepath = parts[1]
        
        if lines > 1000:
            # 分割済みファイルをスキップ
            if filepath in split_files_with_directories:
                continue
                
            # ファイル分析
            analysis = analyze_file_content(filepath)
            
            result = {
                'filepath': filepath,
                'lines': lines,
                'analysis': analysis
            }
            
            # 後方互換性ファイルは別途表示
            if filepath in compatibility_files:
                result['category'] = 'compatibility'
            elif analysis.get('is_compatibility', False):
                result['category'] = 'compatibility'
            elif analysis.get('needs_splitting', False):
                result['category'] = 'needs_splitting'
            else:
                result['category'] = 'review_needed'
                
            results.append(result)

# 結果を分類して表示
print("=== 分割が必要な大規模ファイル ===")
needs_splitting = [r for r in results if r['category'] == 'needs_splitting']
needs_splitting.sort(key=lambda x: x['lines'], reverse=True)

for result in needs_splitting[:15]:  # 上位15ファイル
    analysis = result['analysis']
    print(f"{result['lines']:>6} {result['filepath']}")
    print(f"       Classes: {analysis.get('class_count', 0)}, Functions: {analysis.get('function_count', 0)}")

print(f"\n=== 後方互換性ファイル ({len([r for r in results if r['category'] == 'compatibility'])}件) ===")
compatibility = [r for r in results if r['category'] == 'compatibility']
for result in compatibility[:5]:
    print(f"{result['lines']:>6} {result['filepath']}")

print(f"\n=== レビューが必要なファイル ===")
review_needed = [r for r in results if r['category'] == 'review_needed']
review_needed.sort(key=lambda x: x['lines'], reverse=True)
for result in review_needed[:10]:
    analysis = result['analysis']
    print(f"{result['lines']:>6} {result['filepath']}")
    print(f"       Classes: {analysis.get('class_count', 0)}, Functions: {analysis.get('function_count', 0)}")

print(f"\n総計: 分割必要 {len(needs_splitting)}件, 後方互換性 {len(compatibility)}件, レビュー必要 {len(review_needed)}件")

