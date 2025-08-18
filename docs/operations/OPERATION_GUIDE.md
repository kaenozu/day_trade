# Day Trade Personal - 運用ガイド

## システム起動

### 統合ダッシュボード起動
```bash
python -c "import sys; sys.path.append('src'); from src.day_trade.dashboard.integrated_dashboard_system import main; import asyncio; asyncio.run(main())"
```

### Webダッシュボード起動
```bash
python src/day_trade/dashboard/web_dashboard.py
```

## システム監視

### ヘルスチェック
- 統合ダッシュボード: http://localhost:5000
- システムステータス: /api/status
- システムメトリクス: /api/history/system

### ログ確認
- アプリケーションログ: logs/
- アラートログ: alerts_*.log
- システムログ: stability_data/system.log

## メンテナンス

### データベース管理
- バックアップ: stability_data/backups/
- データベース最適化: 定期実行推奨
- 古いログファイルの削除: 月次実行

### セキュリティ
- セキュリティログ確認: security_data/security_events.log
- アクセス制御設定: security_data/access_control_config.json
- 暗号化キー管理: security/keys/

## トラブルシューティング

### 一般的な問題
1. データベース接続エラー: データベースファイルの権限確認
2. メモリ不足: キャッシュクリアの実行
3. ポート競合: 設定ファイルでポート変更

### 緊急時対応
1. システム停止: Ctrl+C
2. データバックアップ: backup/ディレクトリ確認
3. ログ分析: verification_results/で詳細確認

---
生成日時: 2025-08-17 16:39:46
