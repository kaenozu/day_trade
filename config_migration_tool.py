#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定ファイル移行ツール - Issue #960対応
散在する設定ファイルを統合設定システムに移行
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
import shutil
from datetime import datetime
from unified_config_manager import UnifiedConfigManager
from config_analyzer import ConfigAnalyzer
import hashlib

class ConfigMigrationTool:
    """設定ファイル移行ツール"""
    
    def __init__(self, project_root: str = ".", unified_config_root: str = "config_unified"):
        self.project_root = Path(project_root)
        self.unified_config_root = Path(unified_config_root)
        self.config_manager = UnifiedConfigManager(unified_config_root)
        self.analyzer = ConfigAnalyzer(project_root)
        self.logger = self._setup_logging()
        
        # 移行統計
        self.migration_stats = {
            "files_processed": 0,
            "files_migrated": 0,
            "files_skipped": 0,
            "duplicates_removed": 0,
            "errors": 0
        }
        
        # 除外するファイル（移行しない）
        self.exclude_files = {
            "package.json",
            "package-lock.json",
            "poetry.lock",
            "requirements.txt",
            "coverage.json",
            "test_results.json",
            ".gitignore",
            "README.md"
        }
        
        # バックアップディレクトリの作成
        self.backup_dir = self.project_root / "config_migration_backup"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('config_migration')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_migration(self) -> Dict[str, Any]:
        """移行実行"""
        self.logger.info("Config Migration Tool - Issue #960")
        self.logger.info("=" * 50)
        
        # 設定ファイル分析
        config_files = self.analyzer.find_config_files()
        self.logger.info(f"Found {len(config_files)} config files to process")
        
        # 重複ファイルの特定
        duplicate_groups = self.analyzer.find_duplicates()
        self.logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        
        # Phase 1: 重要な設定ファイルの移行
        self._migrate_core_configs(config_files)
        
        # Phase 2: 重複ファイルの除去
        self._remove_duplicates(duplicate_groups)
        
        # Phase 3: 機能別設定の整理
        self._migrate_feature_configs(config_files)
        
        # Phase 4: バックアップファイルの整理
        self._cleanup_backup_files()
        
        # 移行レポート生成
        report = self._generate_migration_report()
        
        self.logger.info("Migration completed successfully!")
        return report
    
    def _migrate_core_configs(self, config_files: List) -> None:
        """コア設定の移行"""
        self.logger.info("Phase 1: Migrating core configurations...")
        
        core_categories = ["application", "database", "security"]
        
        for config_file in config_files:
            if config_file.category in core_categories:
                try:
                    self._migrate_single_config(config_file)
                    self.migration_stats["files_migrated"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to migrate {config_file.path}: {e}")
                    self.migration_stats["errors"] += 1
                
                self.migration_stats["files_processed"] += 1
    
    def _migrate_single_config(self, config_file) -> None:
        """単一設定ファイルの移行"""
        source_path = Path(config_file.path)
        
        # ファイル名が除外リストにある場合はスキップ
        if source_path.name in self.exclude_files:
            self.migration_stats["files_skipped"] += 1
            return
        
        # 元ファイルの読み込み
        with open(source_path, 'r', encoding='utf-8') as f:
            if config_file.type == 'json':
                content = json.load(f)
            else:
                content = yaml.safe_load(f)
        
        if not isinstance(content, dict):
            self.migration_stats["files_skipped"] += 1
            return
        
        # カテゴリに基づく移行先決定
        target_config_type = self._determine_target_config_type(config_file)
        
        if target_config_type:
            # 既存設定の取得とマージ
            try:
                existing_config = self.config_manager.get_config(target_config_type)
            except:
                existing_config = {}
            
            # 設定のマージ
            merged_config = self._merge_configs(existing_config, content, config_file)
            
            # 統合設定への保存
            self.config_manager.set_config(target_config_type, merged_config, "base")
            
            # 元ファイルのバックアップ
            self._backup_original_file(source_path)
            
            self.logger.info(f"Migrated: {source_path} -> {target_config_type}")
        else:
            self.migration_stats["files_skipped"] += 1
    
    def _determine_target_config_type(self, config_file) -> Optional[str]:
        """移行先設定タイプの決定"""
        category_mapping = {
            "application": "application",
            "database": "database", 
            "security": "security",
            "ml_ai": "ml_models",
            "monitoring": "monitoring",
            "deployment": "deployment"
        }
        
        return category_mapping.get(config_file.category)
    
    def _merge_configs(self, existing: Dict[str, Any], new: Dict[str, Any], config_file) -> Dict[str, Any]:
        """設定のマージ"""
        merged = existing.copy()
        
        # ファイル名に基づくキー生成
        file_key = self._generate_file_key(Path(config_file.path))
        
        # 新しい設定を適切なキーの下に配置
        if file_key and file_key not in merged:
            merged[file_key] = new
        else:
            # 直接マージ
            for key, value in new.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
        
        return merged
    
    def _generate_file_key(self, file_path: Path) -> Optional[str]:
        """ファイルパスからキー生成"""
        # パスの一部をキーとして使用
        parts = file_path.parts
        
        # 特定のパターンに基づくキー生成
        if "config" in str(file_path).lower():
            return file_path.stem
        elif len(parts) > 1:
            return f"{parts[-2]}_{file_path.stem}"
        else:
            return file_path.stem
    
    def _remove_duplicates(self, duplicate_groups: List[List]) -> None:
        """重複ファイルの除去"""
        self.logger.info("Phase 2: Removing duplicate files...")
        
        for group in duplicate_groups:
            if len(group) <= 1:
                continue
            
            # 最も新しいファイルを保持
            group_paths = [Path(file_info.path if hasattr(file_info, 'path') else file_info) for file_info in group]
            
            # 修正時刻でソート
            try:
                group_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                keep_file = group_paths[0]
                remove_files = group_paths[1:]
                
                for remove_file in remove_files:
                    if remove_file.exists():
                        # バックアップしてから削除
                        self._backup_original_file(remove_file)
                        remove_file.unlink()
                        self.migration_stats["duplicates_removed"] += 1
                        self.logger.info(f"Removed duplicate: {remove_file}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to process duplicate group: {e}")
    
    def _migrate_feature_configs(self, config_files: List) -> None:
        """機能別設定の移行"""
        self.logger.info("Phase 3: Migrating feature configurations...")
        
        feature_categories = ["ml_ai", "monitoring", "deployment", "general"]
        
        for config_file in config_files:
            if config_file.category in feature_categories and not self._is_already_processed(config_file):
                try:
                    # 機能別ディレクトリに移行
                    self._migrate_to_features_dir(config_file)
                    self.migration_stats["files_migrated"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to migrate feature config {config_file.path}: {e}")
                    self.migration_stats["errors"] += 1
                
                self.migration_stats["files_processed"] += 1
    
    def _migrate_to_features_dir(self, config_file) -> None:
        """機能別ディレクトリへの移行"""
        source_path = Path(config_file.path)
        
        if source_path.name in self.exclude_files:
            return
        
        # 移行先パス決定
        features_dir = self.unified_config_root / "features"
        features_dir.mkdir(exist_ok=True)
        
        # カテゴリ別サブディレクトリ
        category_dir = features_dir / config_file.category
        category_dir.mkdir(exist_ok=True)
        
        target_path = category_dir / source_path.name
        
        # ファイルコピー
        if source_path.exists() and not target_path.exists():
            shutil.copy2(source_path, target_path)
            self._backup_original_file(source_path)
            self.logger.info(f"Migrated feature config: {source_path} -> {target_path}")
    
    def _is_already_processed(self, config_file) -> bool:
        """既に処理済みかチェック"""
        # バックアップディレクトリに同名ファイルがあるかチェック
        backup_path = self.backup_dir / Path(config_file.path).name
        return backup_path.exists()
    
    def _cleanup_backup_files(self) -> None:
        """バックアップファイルの整理"""
        self.logger.info("Phase 4: Cleaning up backup files...")
        
        # backupsディレクトリの重複ファイル除去
        backup_dirs = ["backups", "exports", "cache_data"]
        
        for dir_name in backup_dirs:
            backup_path = self.project_root / dir_name
            if backup_path.exists():
                self._cleanup_directory_duplicates(backup_path)
    
    def _cleanup_directory_duplicates(self, directory: Path) -> None:
        """ディレクトリ内の重複ファイル除去"""
        if not directory.exists():
            return
        
        # ファイルのハッシュ値でグループ化
        hash_groups = {}
        
        for file_path in directory.glob("*"):
            if file_path.is_file():
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash not in hash_groups:
                        hash_groups[file_hash] = []
                    hash_groups[file_hash].append(file_path)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to hash {file_path}: {e}")
        
        # 重複ファイルの除去
        for file_hash, files in hash_groups.items():
            if len(files) > 1:
                # 最新ファイルを保持
                files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                keep_file = files[0]
                
                for remove_file in files[1:]:
                    try:
                        remove_file.unlink()
                        self.migration_stats["duplicates_removed"] += 1
                        self.logger.info(f"Removed backup duplicate: {remove_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {remove_file}: {e}")
    
    def _backup_original_file(self, file_path: Path) -> None:
        """元ファイルのバックアップ"""
        if not file_path.exists():
            return
        
        backup_path = self.backup_dir / file_path.name
        
        # 同名ファイルがある場合はタイムスタンプ付きで保存
        if backup_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
        
        try:
            shutil.copy2(file_path, backup_path)
        except Exception as e:
            self.logger.warning(f"Failed to backup {file_path}: {e}")
    
    def _generate_migration_report(self) -> Dict[str, Any]:
        """移行レポート生成"""
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "statistics": self.migration_stats,
            "unified_config_structure": {
                "core_configs": self.config_manager.list_config_types(),
                "environment_configs": self.config_manager._get_available_environments(),
                "features_directory": str(self.unified_config_root / "features")
            },
            "recommendations": [
                "統合設定システムを使用して設定を管理してください",
                "環境別設定は environments/ ディレクトリで管理してください", 
                "機能別設定は features/ ディレクトリで管理してください",
                "設定変更時は検証スキーマを更新してください"
            ]
        }
        
        # レポート保存
        report_file = self.project_root / "config_migration_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Migration report saved: {report_file}")
        return report

def main():
    """メイン実行"""
    migration_tool = ConfigMigrationTool()
    report = migration_tool.run_migration()
    
    print("\nMIGRATION SUMMARY")
    print("=" * 50)
    print(f"Files processed: {report['statistics']['files_processed']}")
    print(f"Files migrated: {report['statistics']['files_migrated']}")
    print(f"Files skipped: {report['statistics']['files_skipped']}")
    print(f"Duplicates removed: {report['statistics']['duplicates_removed']}")
    print(f"Errors: {report['statistics']['errors']}")
    print(f"\nUnified configs available: {len(report['unified_config_structure']['core_configs'])}")
    print(f"Environment configs: {len(report['unified_config_structure']['environment_configs'])}")

if __name__ == "__main__":
    main()