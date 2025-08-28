import os
import sys

# 既に分割済みのモジュラーディレクトリ
modular_dirs = [
    'database/', 'external_api/', 'gpu_inference/', 'prediction_models/', 
    'enhanced_dashboard/', 'web_dashboard/', 'interactive/', 'watchlist/', 
    'analysis_server/', 'backtest/', 'ensemble/', 'signals/', 'screening/',
    'orchestrator/', 'trade_manager/', 'stock_master/', 'version_manager/',
    'freshness_monitor/', 'enterprise_master/', 'master_data/',
    'stock_fetcher/', 'fetchers/', 'ml_engine/', 'cache/', 
    'feature_store/', 'base_models/', 'deep_learning/', 'gpu/', 
    'quantization/', 'prediction/', 'weighting/'
]

# バックアップ・レガシーファイルのパターン
exclude_patterns = [
    '_backup', 'legacy_', 'backup.py', 'original.py', 'old.py',
    'test_', '_test.py', 'bak', '.bak'
]

def should_exclude_file(filepath):
    # モジュラーディレクトリ内のファイルをチェック
    for modular_dir in modular_dirs:
        if f'/{modular_dir}' in filepath:
            return True
    
    # バックアップ・レガシーファイルをチェック
    for pattern in exclude_patterns:
        if pattern in filepath:
            return True
    
    return False

# 入力から1000行以上のファイルをフィルタリング
for line in sys.stdin:
    line = line.strip()
    if not line or 'total' in line:
        continue
        
    parts = line.split()
    if len(parts) >= 2:
        lines = int(parts[0])
        filepath = parts[1]
        
        if lines > 1000 and not should_exclude_file(filepath):
            print(f"{lines:>6} {filepath}")

