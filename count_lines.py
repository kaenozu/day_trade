import sys
import os

# analyze_large_files.py でスキップされるディレクトリやファイル
EXCLUDED_DIRS = ['__pycache__', '.git', '.vscode', '.pytest_cache', 'build', 'dist', 'data']


def count_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.readlines())
    except Exception as e:
        # print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return 0

def main():
    for root, dirs, files in os.walk('.'):
        # 除外ディレクトリを探索対象から外す
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                lines = count_lines(filepath)
                if lines > 0:
                    # analyze_large_files.py が期待するフォーマットで出力
                    print(f"{lines} {filepath.replace(os.sep, '/')}")

if __name__ == "__main__":
    main()
