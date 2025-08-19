#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定ファイル分析・統合ツール - Issue #960対応
散在する設定ファイルの分析と統合計画生成
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import hashlib
import re

@dataclass
class ConfigFile:
    """設定ファイル情報"""
    path: str
    type: str  # json, yaml, yml
    size: int
    keys: Set[str]
    nested_keys: Set[str]
    content_hash: str
    category: str
    is_duplicate: bool = False
    similarity_score: float = 0.0

@dataclass
class ConfigAnalysisResult:
    """設定分析結果"""
    total_files: int
    json_files: int
    yaml_files: int
    total_size: int
    duplicate_groups: List[List[str]]
    common_keys: Set[str]
    category_distribution: Dict[str, int]
    consolidation_plan: Dict[str, List[str]]

class ConfigAnalyzer:
    """設定ファイル分析クラス"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config_files: List[ConfigFile] = []
        self.exclude_patterns = [
            "**/node_modules/**",
            "**/venv/**",
            "**/build/**",
            "**/dist/**",
            "**/.git/**",
            "**/backups/**",
            "**/logs/**",
            "**/cache/**",
            "**/temp/**",
            "**/test_**",
            "**/*_test.json",
            "**/*_results.json",
            "**/coverage.json",
            "**/package-lock.json",
            "**/poetry.lock"
        ]
        
    def find_config_files(self) -> List[ConfigFile]:
        """設定ファイルの検出"""
        config_files = []
        
        # JSON, YAML, YMLファイルを検索
        patterns = ["**/*.json", "**/*.yaml", "**/*.yml"]
        
        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if self._should_exclude(file_path):
                    continue
                    
                try:
                    config_file = self._analyze_config_file(file_path)
                    if config_file:
                        config_files.append(config_file)
                        
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        
        self.config_files = config_files
        return config_files
    
    def _should_exclude(self, file_path: Path) -> bool:
        """ファイルを除外すべきかチェック"""
        file_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if file_path.match(pattern) or pattern.replace("**/", "") in file_str:
                return True
                
        return False
    
    def _analyze_config_file(self, file_path: Path) -> Optional[ConfigFile]:
        """設定ファイルの分析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # ファイルタイプ判定
            file_type = file_path.suffix.lower().lstrip('.')
            
            # 内容解析
            try:
                if file_type == 'json':
                    data = json.loads(content)
                elif file_type in ['yaml', 'yml']:
                    data = yaml.safe_load(content)
                else:
                    return None
            except (json.JSONDecodeError, yaml.YAMLError):
                return None
            
            if not isinstance(data, dict):
                return None
            
            # キー抽出
            keys = set(data.keys())
            nested_keys = self._extract_nested_keys(data)
            
            # ハッシュ値計算
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # カテゴリ分類
            category = self._categorize_config(file_path, keys)
            
            return ConfigFile(
                path=str(file_path),
                type=file_type,
                size=len(content),
                keys=keys,
                nested_keys=nested_keys,
                content_hash=content_hash,
                category=category
            )
            
        except Exception as e:
            print(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _extract_nested_keys(self, data: Dict, prefix: str = "") -> Set[str]:
        """ネストしたキーの抽出"""
        nested_keys = set()
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            nested_keys.add(full_key)
            
            if isinstance(value, dict):
                nested_keys.update(self._extract_nested_keys(value, full_key))
                
        return nested_keys
    
    def _categorize_config(self, file_path: Path, keys: Set[str]) -> str:
        """設定ファイルのカテゴリ分類"""
        path_str = str(file_path).lower()
        
        # パス基準の分類
        if "database" in path_str or "db" in path_str:
            return "database"
        elif "security" in path_str or "auth" in path_str:
            return "security"
        elif "monitoring" in path_str or "prometheus" in path_str or "grafana" in path_str:
            return "monitoring"
        elif "deployment" in path_str or "docker" in path_str or "k8s" in path_str:
            return "deployment"
        elif "ml" in path_str or "model" in path_str or "ai" in path_str:
            return "ml_ai"
        elif "test" in path_str:
            return "testing"
        elif "config" in path_str and "environment" in path_str:
            return "environment"
        
        # キー基準の分類
        if any(key in ["host", "port", "database", "db_url"] for key in keys):
            return "database"
        elif any(key in ["secret", "token", "key", "password"] for key in keys):
            return "security"
        elif any(key in ["prometheus", "grafana", "monitoring"] for key in keys):
            return "monitoring"
        elif any(key in ["model", "ml", "ai", "training"] for key in keys):
            return "ml_ai"
        elif any(key in ["app", "application", "server"] for key in keys):
            return "application"
        else:
            return "general"
    
    def find_duplicates(self) -> List[List[ConfigFile]]:
        """重複設定ファイルの検出"""
        duplicate_groups = []
        content_hash_groups = defaultdict(list)
        
        # ハッシュ値で重複チェック
        for config_file in self.config_files:
            content_hash_groups[config_file.content_hash].append(config_file)
        
        # 完全重複の検出
        for hash_value, files in content_hash_groups.items():
            if len(files) > 1:
                for file in files:
                    file.is_duplicate = True
                duplicate_groups.append(files)
        
        # 類似度チェック
        similarity_groups = self._find_similar_configs()
        duplicate_groups.extend(similarity_groups)
        
        return duplicate_groups
    
    def _find_similar_configs(self, similarity_threshold: float = 0.8) -> List[List[ConfigFile]]:
        """類似設定ファイルの検出"""
        similar_groups = []
        processed = set()
        
        for i, config1 in enumerate(self.config_files):
            if config1.path in processed:
                continue
                
            similar_files = [config1]
            
            for j, config2 in enumerate(self.config_files[i+1:], i+1):
                if config2.path in processed:
                    continue
                
                similarity = self._calculate_similarity(config1, config2)
                if similarity >= similarity_threshold:
                    config2.similarity_score = similarity
                    similar_files.append(config2)
                    processed.add(config2.path)
            
            if len(similar_files) > 1:
                similar_groups.append(similar_files)
                for file in similar_files:
                    processed.add(file.path)
        
        return similar_groups
    
    def _calculate_similarity(self, config1: ConfigFile, config2: ConfigFile) -> float:
        """設定ファイル間の類似度計算"""
        # キーの重複度
        keys1, keys2 = config1.keys, config2.keys
        common_keys = len(keys1.intersection(keys2))
        total_keys = len(keys1.union(keys2))
        
        if total_keys == 0:
            return 0.0
        
        key_similarity = common_keys / total_keys
        
        # ネストキーの重複度
        nested1, nested2 = config1.nested_keys, config2.nested_keys
        common_nested = len(nested1.intersection(nested2))
        total_nested = len(nested1.union(nested2))
        
        nested_similarity = common_nested / total_nested if total_nested > 0 else 0.0
        
        # カテゴリの一致
        category_bonus = 0.2 if config1.category == config2.category else 0.0
        
        return (key_similarity * 0.5 + nested_similarity * 0.3 + category_bonus)
    
    def analyze_key_usage(self) -> Dict[str, int]:
        """キー使用頻度の分析"""
        key_usage = Counter()
        
        for config_file in self.config_files:
            for key in config_file.nested_keys:
                key_usage[key] += 1
        
        return dict(key_usage)
    
    def generate_consolidation_plan(self) -> Dict[str, Any]:
        """統合計画の生成"""
        categories = defaultdict(list)
        
        # カテゴリ別グループ化
        for config_file in self.config_files:
            categories[config_file.category].append(config_file)
        
        consolidation_plan = {
            "core": {
                "target_file": "config/core/application.yaml",
                "sources": [],
                "description": "アプリケーション基本設定"
            },
            "database": {
                "target_file": "config/core/database.yaml", 
                "sources": [],
                "description": "データベース関連設定"
            },
            "security": {
                "target_file": "config/core/security.yaml",
                "sources": [],
                "description": "セキュリティ関連設定"
            },
            "environments": {
                "target_file": "config/environments/",
                "sources": [],
                "description": "環境別設定"
            },
            "features": {
                "target_file": "config/features/",
                "sources": [],
                "description": "機能別設定"
            }
        }
        
        # ソースファイルの割り当て
        for category, files in categories.items():
            if category == "application":
                consolidation_plan["core"]["sources"].extend([f.path for f in files])
            elif category == "database":
                consolidation_plan["database"]["sources"].extend([f.path for f in files])
            elif category == "security":
                consolidation_plan["security"]["sources"].extend([f.path for f in files])
            elif category == "environment":
                consolidation_plan["environments"]["sources"].extend([f.path for f in files])
            else:
                consolidation_plan["features"]["sources"].extend([f.path for f in files])
        
        return consolidation_plan
    
    def create_unified_config_structure(self) -> Dict[str, Any]:
        """統合設定構造の作成"""
        # 基本設定構造
        unified_config = {
            "core": {
                "application": {
                    "name": "day-trade-personal",
                    "version": "2.1.0",
                    "debug": False,
                    "environment": "production"
                },
                "database": {
                    "url": "sqlite:///data/trading.db",
                    "pool_size": 10,
                    "echo": False
                },
                "security": {
                    "secret_key": "${SECRET_KEY}",
                    "encryption_key": "${ENCRYPTION_KEY}",
                    "session_timeout": 3600
                }
            },
            "features": {
                "ml_models": {
                    "enabled": True,
                    "model_path": "models/",
                    "batch_size": 32
                },
                "monitoring": {
                    "enabled": True,
                    "prometheus_port": 9090,
                    "grafana_port": 3000
                },
                "api": {
                    "rate_limit": 1000,
                    "timeout": 30,
                    "max_connections": 100
                }
            },
            "environments": {
                "development": {
                    "debug": True,
                    "database": {
                        "echo": True
                    }
                },
                "staging": {
                    "debug": False,
                    "database": {
                        "pool_size": 5
                    }
                },
                "production": {
                    "debug": False,
                    "security": {
                        "strict_mode": True
                    }
                }
            }
        }
        
        return unified_config
    
    def run_analysis(self) -> ConfigAnalysisResult:
        """分析実行"""
        print("Config File Analysis - Issue #960")
        print("=" * 50)
        
        # 設定ファイル検出
        config_files = self.find_config_files()
        print(f"Found {len(config_files)} config files")
        
        # 重複検出
        duplicate_groups = self.find_duplicates()
        print(f"Found {len(duplicate_groups)} duplicate groups")
        
        # キー使用頻度分析
        key_usage = self.analyze_key_usage()
        common_keys = {k for k, v in key_usage.items() if v > 1}
        print(f"Found {len(common_keys)} common keys")
        
        # カテゴリ分布
        category_dist = Counter(cf.category for cf in config_files)
        
        # 統計計算
        json_files = len([cf for cf in config_files if cf.type == 'json'])
        yaml_files = len(config_files) - json_files
        total_size = sum(cf.size for cf in config_files)
        
        # 統合計画生成
        consolidation_plan = self.generate_consolidation_plan()
        
        result = ConfigAnalysisResult(
            total_files=len(config_files),
            json_files=json_files,
            yaml_files=yaml_files,
            total_size=total_size,
            duplicate_groups=[[cf.path for cf in group] for group in duplicate_groups],
            common_keys=common_keys,
            category_distribution=dict(category_dist),
            consolidation_plan=consolidation_plan
        )
        
        return result
    
    def export_analysis(self, result: ConfigAnalysisResult, output_file: str = "config_analysis_report.json"):
        """分析結果のエクスポート"""
        export_data = {
            "analysis_summary": asdict(result),
            "config_files": [
                {
                    "path": cf.path,
                    "type": cf.type,
                    "size": cf.size,
                    "category": cf.category,
                    "keys_count": len(cf.keys),
                    "is_duplicate": cf.is_duplicate,
                    "similarity_score": cf.similarity_score
                }
                for cf in self.config_files
            ],
            "unified_config_structure": self.create_unified_config_structure(),
            "migration_recommendations": self._generate_migration_recommendations()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Analysis exported to {output_file}")
    
    def _generate_migration_recommendations(self) -> List[Dict[str, Any]]:
        """移行推奨事項の生成"""
        recommendations = [
            {
                "priority": "HIGH",
                "action": "consolidate_core_configs",
                "description": "基本設定の統合",
                "files_affected": len([cf for cf in self.config_files if cf.category in ["application", "database", "security"]]),
                "expected_reduction": "60%"
            },
            {
                "priority": "MEDIUM",
                "action": "standardize_environment_configs",
                "description": "環境別設定の標準化",
                "files_affected": len([cf for cf in self.config_files if "environment" in cf.path]),
                "expected_reduction": "70%"
            },
            {
                "priority": "MEDIUM",
                "action": "remove_duplicates",
                "description": "重複設定の除去",
                "files_affected": len([cf for cf in self.config_files if cf.is_duplicate]),
                "expected_reduction": "90%"
            },
            {
                "priority": "LOW", 
                "action": "organize_feature_configs",
                "description": "機能別設定の整理",
                "files_affected": len([cf for cf in self.config_files if cf.category in ["ml_ai", "monitoring"]]),
                "expected_reduction": "40%"
            }
        ]
        
        return recommendations
    
    def print_summary(self, result: ConfigAnalysisResult):
        """分析結果サマリー表示"""
        print("\n" + "=" * 60)
        print("CONFIG FILE ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"Total files: {result.total_files}")
        print(f"JSON files: {result.json_files}")
        print(f"YAML files: {result.yaml_files}")
        print(f"Total size: {result.total_size:,} bytes")
        print(f"Duplicate groups: {len(result.duplicate_groups)}")
        print(f"Common keys: {len(result.common_keys)}")
        
        print("\nCategory distribution:")
        for category, count in result.category_distribution.items():
            print(f"  {category}: {count} files")
        
        print("\nConsolidation targets:")
        for target, info in result.consolidation_plan.items():
            if isinstance(info, dict) and "sources" in info:
                print(f"  {target}: {len(info['sources'])} files -> {info.get('target_file', 'N/A')}")

def main():
    """メイン実行"""
    analyzer = ConfigAnalyzer()
    result = analyzer.run_analysis()
    analyzer.print_summary(result)
    analyzer.export_analysis(result)

if __name__ == "__main__":
    main()