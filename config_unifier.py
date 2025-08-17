#!/usr/bin/env python3
"""
設定管理統一化ツール

分散した設定ファイルの統一化と管理の改善
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class ConfigUnifier:
    """設定管理統一化クラス"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results = {
            'files_found': 0,
            'files_unified': 0,
            'environments': 0
        }

    def unify_configuration(self):
        """設定管理統一化実行"""
        print("設定管理統一化開始")
        print("=" * 40)

        # 1. 既存設定分析
        self._analyze_configs()

        # 2. 統一設定作成
        self._create_unified_configs()

        # 3. 管理ツール作成
        self._create_management_tools()

        print("設定管理統一化完了")

    def _analyze_configs(self):
        """既存設定分析"""
        print("1. 既存設定ファイル分析中...")

        config_files = list(self.base_dir.glob("**/*.json"))
        config_files.extend(list(self.base_dir.glob("config/*")))

        for config_file in config_files[:5]:
            if config_file.is_file() and config_file.suffix == '.json':
                self.results['files_found'] += 1
                print(f"  発見: {config_file.name}")

        print(f"  {self.results['files_found']}個の設定ファイルを発見")

    def _create_unified_configs(self):
        """統一設定作成"""
        print("2. 統一設定ファイル作成中...")

        # 設定ディレクトリ作成
        config_dir = self.base_dir / "config"
        config_dir.mkdir(exist_ok=True)

        # メイン設定
        main_config = {
            "application": {
                "name": "Day Trade Personal",
                "version": "3.0.0",
                "environment": "development"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/application.log"
            },
            "database": {
                "url": "sqlite:///data/trading.db",
                "timeout": 30,
                "backup": {
                    "enabled": True,
                    "interval_hours": 24
                }
            },
            "trading": {
                "enabled": True,
                "market_hours": {
                    "start": "09:00",
                    "end": "15:00"
                },
                "risk_management": {
                    "max_position_size": 10.0,
                    "stop_loss_percentage": 3.0
                }
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl": 1800,
                "max_workers": 4
            }
        }

        # メイン設定保存
        main_config_file = config_dir / "config.json"
        with open(main_config_file, 'w', encoding='utf-8') as f:
            json.dump(main_config, f, indent=2, ensure_ascii=False)

        print(f"  作成: {main_config_file.name}")
        self.results['files_unified'] += 1

        # 環境別設定作成
        environments = ['development', 'staging', 'production']

        for env in environments:
            env_config = main_config.copy()
            env_config['application']['environment'] = env

            if env == 'development':
                env_config['logging']['level'] = 'DEBUG'
                env_config['trading']['enabled'] = False

            elif env == 'production':
                env_config['logging']['level'] = 'WARNING'
                env_config['performance']['max_workers'] = 8

            env_file = config_dir / f"config_{env}.json"
            with open(env_file, 'w', encoding='utf-8') as f:
                json.dump(env_config, f, indent=2, ensure_ascii=False)

            print(f"  作成: {env_file.name}")
            self.results['environments'] += 1

    def _create_management_tools(self):
        """設定管理ツール作成"""
        print("3. 設定管理ツール作成中...")

        # ツールディレクトリ作成
        tools_dir = self.base_dir / "tools"
        tools_dir.mkdir(exist_ok=True)

        # 設定ローダー作成
        loader_code = '''#!/usr/bin/env python3
"""設定ローダー"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"

    def load_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        if not environment:
            environment = os.getenv("ENVIRONMENT", "development")

        config_file = self.config_dir / f"config_{environment}.json"

        if not config_file.exists():
            config_file = self.config_dir / "config.json"

        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_value(self, key_path: str, default: Any = None) -> Any:
        config = self.load_config()
        keys = key_path.split('.')
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
'''

        loader_file = tools_dir / "config_loader.py"
        loader_file.write_text(loader_code, encoding='utf-8')
        print(f"  作成: {loader_file.name}")

        # 設定検証ツール作成
        validator_code = '''#!/usr/bin/env python3
"""設定検証ツール"""

import json
from pathlib import Path
from typing import Dict, List


def validate_config(config_file: Path) -> Dict:
    result = {"valid": False, "errors": []}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 必須項目チェック
        required_keys = ["application", "logging", "database"]
        for key in required_keys:
            if key not in config:
                result["errors"].append(f"必須項目が不足: {key}")

        if not result["errors"]:
            result["valid"] = True

    except Exception as e:
        result["errors"].append(f"ファイルエラー: {e}")

    return result


def main():
    config_dir = Path(__file__).parent.parent / "config"
    config_files = list(config_dir.glob("*.json"))

    print("設定ファイル検証結果")
    print("=" * 30)

    for config_file in config_files:
        result = validate_config(config_file)
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {config_file.name}")

        for error in result["errors"]:
            print(f"  エラー: {error}")


if __name__ == "__main__":
    main()
'''

        validator_file = tools_dir / "config_validator.py"
        validator_file.write_text(validator_code, encoding='utf-8')
        print(f"  作成: {validator_file.name}")

        # レポート作成
        self._create_report()

    def _create_report(self):
        """レポート作成"""
        report_content = f"""# 設定管理統一化レポート

## 実行日時
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 統一化結果
- 発見設定ファイル数: {self.results['files_found']}
- 統一設定ファイル数: {self.results['files_unified']}
- 環境別設定数: {self.results['environments']}

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
生成日時: {datetime.now().isoformat()}
"""

        reports_dir = self.base_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        report_file = reports_dir / "config_unification_report.md"
        report_file.write_text(report_content, encoding='utf-8')
        print(f"  レポート作成: {report_file.name}")


def main():
    """メイン実行"""
    print("設定管理統一化ツール実行開始")
    print("=" * 50)

    base_dir = Path(__file__).parent
    unifier = ConfigUnifier(base_dir)

    unifier.unify_configuration()

    print("\n" + "=" * 50)
    print("✅ 設定管理統一化完了")
    print("=" * 50)
    print("作成内容:")
    print("  - config/          : 統一設定ファイル")
    print("  - tools/           : 設定管理ツール")
    print("  - 環境別設定       : development/staging/production")
    print("\n設定管理:")
    print("  python tools/config_validator.py      # 検証")
    print("=" * 50)


if __name__ == "__main__":
    main()