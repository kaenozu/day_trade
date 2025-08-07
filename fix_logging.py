import os
import re

# 単一キー=値の修正パターン
pattern = r"(logger\.(info|warning|error|debug)\([^,)]+),\s*(\w+)=([^,)]+)(\))"

# 処理対象ファイル
files_to_fix = [
    "src/day_trade/utils/performance_analyzer.py",
    "src/day_trade/utils/transaction_best_practices.py",
    "src/day_trade/models/optimized_database_operations.py",
    "src/day_trade/models/optimized_database.py",
    "src/day_trade/automation/orchestrator.py",
    "src/day_trade/utils/enhanced_error_handler.py",
    "src/day_trade/automation/optimized_orchestrator.py",
    "src/day_trade/models/database_backup.py",
    "src/day_trade/utils/api_resilience.py",
    "src/day_trade/models/database.py",
    "src/day_trade/core/unified_error_handler.py",
    "src/day_trade/analysis/indicators.py",
    "src/day_trade/core/alerts.py",
    "src/day_trade/core/trade_operations.py",
    "src/day_trade/core/trade_manager.py",
]

fixed_count = 0

for file_path in files_to_fix:
    if os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # 単一キー=値の修正
            content = re.sub(
                r"(logger\.(info|warning|error|debug)\([^,)]+),\s*(\w+)=([^,)]+)(\))",
                r'\1, extra={"\3": \4}\5',
                content,
            )

            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed: {file_path}")
                fixed_count += 1
            else:
                print(f"No changes needed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

print(f"\nFixed {fixed_count} files")
