#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重要ファイル優先リファクタリング分析ツール
最も影響度の高い大型ファイルを特定し、優先順位付けしてリファクタリング計画を作成
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class FileImpactAnalysis:
    """ファイル影響度分析結果"""
    path: str
    lines: int
    complexity_score: float
    import_count: int
    external_dependencies: int
    usage_frequency: int
    critical_functions: List[str]
    risk_level: str
    refactoring_priority: int
    estimated_effort_days: float

@dataclass
class RefactoringPlan:
    """リファクタリング計画"""
    file_path: str
    target_modules: List[str]
    extraction_strategy: str
    estimated_modules: int
    dependencies: List[str]
    risks: List[str]
    benefits: List[str]

class CriticalRefactoringAnalyzer:
    """重要ファイル優先リファクタリング分析クラス"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
        # 重要度判定基準
        self.critical_patterns = [
            r"trade_manager",
            r"core",
            r"main",
            r"application",
            r"api",
            r"security",
            r"database",
            r"ml_.*prediction",
            r"ensemble",
            r"realtime"
        ]
        
        # ファイル使用頻度の推定（インポート解析基準）
        self.import_analysis = defaultdict(int)
        
    def _setup_logging(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger('critical_refactoring')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def analyze_critical_files(self) -> Dict[str, Any]:
        """重要ファイルの分析実行"""
        self.logger.info("Critical File Refactoring Analysis")
        self.logger.info("=" * 50)
        
        # Step 1: 大型ファイルの検出
        large_files = self._find_large_files()
        
        # Step 2: インポート関係の分析
        self._analyze_import_relationships()
        
        # Step 3: 影響度分析
        impact_analyses = self._analyze_file_impacts(large_files)
        
        # Step 4: 優先順位付け
        prioritized_files = self._prioritize_files(impact_analyses)
        
        # Step 5: リファクタリング計画生成
        refactoring_plans = self._generate_refactoring_plans(prioritized_files[:10])
        
        # 結果まとめ
        result = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_large_files": len(large_files),
            "critical_files_count": len([f for f in impact_analyses if f.risk_level == "CRITICAL"]),
            "high_priority_files": prioritized_files[:5],
            "refactoring_plans": refactoring_plans,
            "estimated_total_effort_days": sum(f.estimated_effort_days for f in prioritized_files[:10]),
            "immediate_action_files": [f.path for f in prioritized_files[:3]]
        }
        
        self.logger.info(f"Analysis completed: {len(prioritized_files)} files prioritized")
        return result
    
    def _find_large_files(self) -> List[Tuple[str, int]]:
        """大型ファイル（500行以上）の検出"""
        large_files = []
        
        for py_file in self.project_root.glob("**/*.py"):
            # 除外パターン
            if any(exclude in str(py_file) for exclude in [
                "__pycache__", ".git", "venv", ".venv",
                "build", "dist", ".eggs", "node_modules",
                "config_migration_backup"
            ]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                
                if line_count >= 500:
                    large_files.append((str(py_file), line_count))
                    
            except Exception as e:
                self.logger.debug(f"Failed to read {py_file}: {e}")
        
        return sorted(large_files, key=lambda x: x[1], reverse=True)
    
    def _analyze_import_relationships(self) -> None:
        """インポート関係の分析"""
        for py_file in self.project_root.glob("**/*.py"):
            if any(exclude in str(py_file) for exclude in [
                "__pycache__", ".git", "venv", "build", "dist"
            ]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # インポート文の抽出
                import_pattern = r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import|import\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
                imports = re.findall(import_pattern, content)
                
                for from_module, import_module in imports:
                    module = from_module or import_module
                    if module and 'day_trade' in module:
                        self.import_analysis[module] += 1
                        
            except Exception as e:
                self.logger.debug(f"Failed to analyze imports in {py_file}: {e}")
    
    def _analyze_file_impacts(self, large_files: List[Tuple[str, int]]) -> List[FileImpactAnalysis]:
        """ファイル影響度の分析"""
        impact_analyses = []
        
        for file_path, line_count in large_files:
            try:
                analysis = self._analyze_single_file_impact(file_path, line_count)
                if analysis:
                    impact_analyses.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return impact_analyses
    
    def _analyze_single_file_impact(self, file_path: str, line_count: int) -> FileImpactAnalysis:
        """単一ファイルの影響度分析"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # 複雑度計算
            complexity_score = self._calculate_complexity(tree)
            
            # インポート数
            import_count = len([node for node in ast.walk(tree) 
                              if isinstance(node, (ast.Import, ast.ImportFrom))])
            
            # 外部依存関係数
            external_deps = self._count_external_dependencies(content)
            
            # 使用頻度（インポート解析基準）
            module_name = self._extract_module_name(file_path)
            usage_frequency = self.import_analysis.get(module_name, 0)
            
            # 重要な関数の特定
            critical_functions = self._identify_critical_functions(tree)
            
            # リスクレベル決定
            risk_level = self._determine_risk_level(file_path, line_count, complexity_score)
            
            # 優先度計算
            priority = self._calculate_priority(
                line_count, complexity_score, usage_frequency, risk_level
            )
            
            # 作業工数推定
            effort_days = self._estimate_effort(line_count, complexity_score, len(critical_functions))
            
            return FileImpactAnalysis(
                path=file_path,
                lines=line_count,
                complexity_score=complexity_score,
                import_count=import_count,
                external_dependencies=external_deps,
                usage_frequency=usage_frequency,
                critical_functions=critical_functions,
                risk_level=risk_level,
                refactoring_priority=priority,
                estimated_effort_days=effort_days
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """コード複雑度の計算"""
        complexity = 0
        
        for node in ast.walk(tree):
            # サイクロマティック複雑度の簡易計算
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
        
        return complexity
    
    def _count_external_dependencies(self, content: str) -> int:
        """外部依存関係数のカウント"""
        external_imports = set()
        
        # 標準ライブラリ以外のインポートを検出
        import_pattern = r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)|from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import'
        imports = re.findall(import_pattern, content)
        
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'logging', 'typing',
            'collections', 'itertools', 'functools', 'operator', 're', 'math',
            'random', 'time', 'threading', 'multiprocessing', 'asyncio'
        }
        
        for imp_module, from_module in imports:
            module = (imp_module or from_module).split('.')[0]
            if module not in stdlib_modules and not module.startswith('day_trade'):
                external_imports.add(module)
        
        return len(external_imports)
    
    def _extract_module_name(self, file_path: str) -> str:
        """ファイルパスからモジュール名を抽出"""
        path = Path(file_path)
        parts = path.parts
        
        # src/day_trade/ 以下のパスを抽出
        try:
            src_index = parts.index('src')
            module_parts = parts[src_index+1:]
            module_name = '.'.join(module_parts).replace('.py', '')
            return module_name
        except (ValueError, IndexError):
            return path.stem
    
    def _identify_critical_functions(self, tree: ast.AST) -> List[str]:
        """重要な関数の特定"""
        critical_functions = []
        
        critical_patterns = [
            r'.*trade.*', r'.*predict.*', r'.*analyze.*', r'.*execute.*',
            r'.*process.*', r'.*manage.*', r'.*optimize.*', r'.*calculate.*',
            r'main', r'run', r'start', r'init', r'setup'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for pattern in critical_patterns:
                    if re.match(pattern, func_name, re.IGNORECASE):
                        critical_functions.append(func_name)
                        break
        
        return critical_functions
    
    def _determine_risk_level(self, file_path: str, line_count: int, complexity: float) -> str:
        """リスクレベルの決定"""
        file_path_lower = file_path.lower()
        
        # クリティカルパスのチェック
        is_critical_path = any(pattern in file_path_lower for pattern in [
            'trade_manager', 'core', 'main', 'api', 'security', 'database'
        ])
        
        if is_critical_path and line_count > 1500:
            return "CRITICAL"
        elif line_count > 2000 or complexity > 100:
            return "HIGH"
        elif line_count > 1000 or complexity > 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_priority(self, line_count: int, complexity: float, 
                          usage_frequency: int, risk_level: str) -> int:
        """優先度の計算（高いほど優先）"""
        priority = 0
        
        # 行数による重み
        priority += min(line_count // 100, 50)
        
        # 複雑度による重み
        priority += min(int(complexity), 30)
        
        # 使用頻度による重み
        priority += min(usage_frequency * 2, 20)
        
        # リスクレベルによる重み
        risk_weights = {"CRITICAL": 50, "HIGH": 30, "MEDIUM": 15, "LOW": 5}
        priority += risk_weights.get(risk_level, 0)
        
        return priority
    
    def _estimate_effort(self, line_count: int, complexity: float, 
                        critical_functions: int) -> float:
        """作業工数の推定（日数）"""
        # 基本工数：行数ベース
        base_effort = line_count / 500  # 500行 = 1日
        
        # 複雑度による追加工数
        complexity_effort = complexity / 20  # 複雑度20 = 1日
        
        # 重要関数による追加工数
        function_effort = critical_functions * 0.5  # 重要関数1つ = 0.5日
        
        total_effort = base_effort + complexity_effort + function_effort
        
        # 最小1日、最大20日に制限
        return max(1.0, min(total_effort, 20.0))
    
    def _prioritize_files(self, impact_analyses: List[FileImpactAnalysis]) -> List[FileImpactAnalysis]:
        """ファイルの優先順位付け"""
        return sorted(impact_analyses, key=lambda x: x.refactoring_priority, reverse=True)
    
    def _generate_refactoring_plans(self, prioritized_files: List[FileImpactAnalysis]) -> List[RefactoringPlan]:
        """リファクタリング計画の生成"""
        plans = []
        
        for file_analysis in prioritized_files:
            plan = self._create_refactoring_plan(file_analysis)
            plans.append(plan)
        
        return plans
    
    def _create_refactoring_plan(self, analysis: FileImpactAnalysis) -> RefactoringPlan:
        """個別リファクタリング計画の作成"""
        file_path = analysis.path
        file_name = Path(file_path).stem
        
        # 抽出すべきモジュール推定
        estimated_modules = max(2, min(analysis.lines // 300, 8))
        
        # モジュール分割戦略
        if 'trade_manager' in file_path:
            target_modules = [
                f"{file_name}_core", f"{file_name}_execution", 
                f"{file_name}_validation", f"{file_name}_monitoring"
            ]
            strategy = "機能別分割（実行・検証・監視）"
        elif 'api' in file_path or 'web' in file_path:
            target_modules = [
                f"{file_name}_routes", f"{file_name}_services", 
                f"{file_name}_models", f"{file_name}_utils"
            ]
            strategy = "レイヤー別分割（MVC）"
        elif 'ml' in file_path or 'prediction' in file_path:
            target_modules = [
                f"{file_name}_models", f"{file_name}_features", 
                f"{file_name}_pipeline", f"{file_name}_evaluation"
            ]
            strategy = "ML工程別分割"
        else:
            target_modules = [
                f"{file_name}_core", f"{file_name}_utils", 
                f"{file_name}_config", f"{file_name}_tests"
            ]
            strategy = "機能・責任別分割"
        
        # 依存関係の特定
        dependencies = self._identify_dependencies(analysis)
        
        # リスクの特定
        risks = self._identify_refactoring_risks(analysis)
        
        # メリットの特定
        benefits = self._identify_refactoring_benefits(analysis)
        
        return RefactoringPlan(
            file_path=file_path,
            target_modules=target_modules[:estimated_modules],
            extraction_strategy=strategy,
            estimated_modules=estimated_modules,
            dependencies=dependencies,
            risks=risks,
            benefits=benefits
        )
    
    def _identify_dependencies(self, analysis: FileImpactAnalysis) -> List[str]:
        """依存関係の特定"""
        # ファイルパスから推測される依存関係
        deps = []
        
        if 'core' in analysis.path:
            deps.extend(['database', 'security', 'config'])
        if 'api' in analysis.path:
            deps.extend(['authentication', 'validation', 'serialization'])
        if 'ml' in analysis.path:
            deps.extend(['data_processing', 'feature_engineering', 'model_storage'])
        
        return deps
    
    def _identify_refactoring_risks(self, analysis: FileImpactAnalysis) -> List[str]:
        """リファクタリングリスクの特定"""
        risks = []
        
        if analysis.usage_frequency > 10:
            risks.append("高使用頻度による影響範囲拡大")
        
        if analysis.risk_level == "CRITICAL":
            risks.append("システム停止リスク")
        
        if analysis.external_dependencies > 10:
            risks.append("外部依存関係による複雑性")
        
        if analysis.lines > 2000:
            risks.append("大規模リファクタリングによる品質低下リスク")
        
        return risks
    
    def _identify_refactoring_benefits(self, analysis: FileImpactAnalysis) -> List[str]:
        """リファクタリングメリットの特定"""
        benefits = []
        
        benefits.append(f"保守性向上：{analysis.lines}行の分割による可読性改善")
        benefits.append("テスタビリティ向上：モジュール別テスト可能")
        benefits.append("再利用性向上：独立したモジュール設計")
        
        if analysis.complexity_score > 50:
            benefits.append("複雑度削減による品質向上")
        
        if analysis.lines > 1500:
            benefits.append("開発効率向上：並行開発可能")
        
        return benefits
    
    def export_analysis(self, result: Dict[str, Any], output_file: str = "critical_refactoring_analysis.json") -> None:
        """分析結果のエクスポート"""
        # JSON シリアライゼーション用の変換
        export_data = {}
        
        for key, value in result.items():
            if isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                export_data[key] = [asdict(item) for item in value]
            else:
                export_data[key] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Critical refactoring analysis exported to {output_file}")

def main():
    """メイン実行"""
    analyzer = CriticalRefactoringAnalyzer()
    result = analyzer.analyze_critical_files()
    
    print("\nCRITICAL REFACTORING ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total large files analyzed: {result['total_large_files']}")
    print(f"Critical risk files: {result['critical_files_count']}")
    print(f"Estimated total effort: {result['estimated_total_effort_days']:.1f} days")
    
    print(f"\nIMMEDIATE ACTION REQUIRED:")
    for i, file_path in enumerate(result['immediate_action_files'], 1):
        print(f"{i}. {Path(file_path).name}")
    
    print(f"\nTOP 5 PRIORITY FILES:")
    for i, file_analysis in enumerate(result['high_priority_files'], 1):
        print(f"{i}. {Path(file_analysis['path']).name}")
        print(f"   Lines: {file_analysis['lines']}, Priority: {file_analysis['refactoring_priority']}")
        print(f"   Risk: {file_analysis['risk_level']}, Effort: {file_analysis['estimated_effort_days']:.1f} days")
    
    # 分析結果エクスポート
    analyzer.export_analysis(result)

if __name__ == "__main__":
    main()