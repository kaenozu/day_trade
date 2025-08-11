import re
import sys
from collections import defaultdict

# ruff のエラーログを標準入力から読み込む
error_log = sys.stdin.read()

# F821 (Undefined name) エラーを抽出
f821_errors = re.findall(r"(.+?):(\d+):(\d+): F821 Undefined name `(.+?)`", error_log)

# E402 (Module level import not at top of file) エラーを抽出
e402_errors = re.findall(
    r"(.+?):(\d+):(\d+): E402 Module level import not at top of file", error_log
)

# ファイルごとにエラーを集計
f821_file_errors = defaultdict(list)
for file, line, col, name in f821_errors:
    f821_file_errors[file].append((int(line), name))

e402_file_errors = defaultdict(list)
for file, line, col in e402_errors:
    e402_file_errors[file].append(int(line))

# 結果を表示
print("=== F821 Errors (Undefined Name) ===")
for file, errors in f821_file_errors.items():
    print(f"{file}:")
    for line, name in sorted(errors):
        print(f"  Line {line}: Undefined name `{name}`")
    print()

print("\n=== E402 Errors (Import Not At Top) ===")
for file, lines in e402_file_errors.items():
    print(f"{file}: Lines {sorted(lines)}")
