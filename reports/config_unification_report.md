# 設定管理統一化レポート

## 実行日時
2025-08-17 17:07:27

## 統一化結果
- 発見設定ファイル数: 5
- 統一設定ファイル数: 1  
- 環境別設定数: 3

## 作成された設定ファイル
- config/config.json - メイン設定
- config/config_development.json - 開発環境
- config/config_staging.json - ステージング環境
- config/config_production.json - 本番環境

## 管理ツール
- tools/config_loader.py - 設定読み込み
- tools/config_validator.py - 設定検証

## 使用方法

### 設定読み込み
```python
from tools.config_loader import ConfigLoader

loader = ConfigLoader()
config = loader.load_config("production")
db_url = loader.get_value("database.url")
```

### 設定検証
```bash
python tools/config_validator.py
```

### 環境変数
```bash
export ENVIRONMENT=production
```

---
生成日時: 2025-08-17T17:07:27.300946
