# セキュリティ強化: 行数制限
line_count = csv_text.count("\n") + 1
max_lines = 100000  # 10万行制限
if line_count > max_lines:
    logger.error(f"CSV行数が多すぎます: {line_count}行")
    raise ValueError("CSV行数が制限を超過しています")

# セキュリティ強化: 危険なCSVパターンチェック（拡張版）
dangerous_csv_patterns = [
    "=cmd|",  # Excelコマンド実行
    "=system(",  # システムコマンド
    "@SUM(",  # Excel関数インジェクション
    "=HYPERLINK(",  # ハイパーリンクインジェクション
    "javascript:",  # JavaScriptスキーム
    "data:text/html",  # HTMLデータスキーム
    "=WEBSERVICE(",  # Web サービス関数
    "=IMPORTXML(",  # XML インポート関数
    "=IMPORTDATA(",  # データインポート関数
    "<script",  # スクリプトタグ
    "</script>",  # スクリプトタグ終了
    "vbscript:",  # VBScript
    "file://",  # ファイルプロトコル
]

csv_lower = csv_text.lower()
for pattern in dangerous_csv_patterns:
    if pattern.lower() in csv_lower:
        logger.error(f"危険なCSVパターンを検出: {pattern}")
        raise ValueError("CSVデータに危険なパターンが含まれています")