def _sanitize_error_message(self, error_message: str, error_type: str) -> str:
    """エラーメッセージの機密情報サニタイゼーション (セキュリティ強化版)"""
    # 公開用の安全なエラーメッセージ生成
    safe_messages = {
        "ClientError": "外部APIとの通信でエラーが発生しました",
        "TimeoutError": "外部APIからの応答がタイムアウトしました",
        "ConnectionError": "外部APIサーバーとの接続に失敗しました",
        "JSONDecodeError": "外部APIからの応答形式が不正です",
        "ValueError": "リクエストパラメータまたはレスポンスが不正です",
        "KeyError": "APIレスポンスの形式が予期しないものです",
        "ParserError": "データ解析処理でエラーが発生しました",
        "SecurityError": "セキュリティ制約によりリクエストが拒否されました",
        "default": "外部API処理でエラーが発生しました",
    }

    # エラータイプに応じた安全なメッセージを返す
    safe_message = safe_messages.get(error_type, safe_messages["default"])

    # 追加の機密情報チェック（拡張版）
    sensitive_patterns = [
        r"/[a-zA-Z]:/[^/\s]+",  # Windowsファイルパス
        r"/[^/\s]+/[^/\s]+/[^/\s]+",  # Unixファイルパス (深い階層)
        r"[a-zA-Z0-9]{32,}",  # 長いAPIキー・トークン様文字列
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IPアドレス
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # メールアドレス
        r"[a-zA-Z]+://[^\s]+",  # URL
        r"(?:password|token|key|secret|auth|bearer)[:=]\s*[^\s]+",  # 認証情報
        r"(?:username|user|login)[:=]\s*[^\s]+",  # ユーザー情報
        r"(?:host|hostname|server)[:=]\s*[^\s]+",  # サーバー情報
        r"(?:database|db|schema)[:=]\s*[^\s]+",  # データベース情報
        r"(?:port)[:=]\s*\d+",  # ポート番号
        r"[A-Z]{2,}_[A-Z_]+",  # 環境変数名のパターン
    ]

    for pattern in sensitive_patterns:
        if re.search(pattern, error_message, re.IGNORECASE):
            logger.warning(f"エラーメッセージに機密情報が含まれる可能性を検出: {error_type}")
            # より汎用的なメッセージにさらに変更
            return f"{safe_message} (詳細はシステムログを確認してください)"

    # 一般的なエラーメッセージはそのまま返す
    return safe_message
