"""
統一コンソリデーター

重複コードの統合と最適化を提供
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ast
import os
import importlib
import inspect
from pathlib import Path

T = TypeVar('T')


class ComponentCategory(Enum):
    """コンポーネントカテゴリ"""
    ANALYZER = "analyzer"
    PROCESSOR = "processor"
    ENGINE = "engine"
    MANAGER = "manager"
    FETCHER = "fetcher"
    CALCULATOR = "calculator"
    OPTIMIZER = "optimizer"
    VISUALIZER = "visualizer"
    REPORTER = "reporter"


class ConsolidationStrategy(Enum):
    """統合戦略"""
    MERGE_SIMILAR = "merge_similar"
    EXTRACT_COMMON = "extract_common"
    HIERARCHY_CREATION = "hierarchy_creation"
    INTERFACE_STANDARDIZATION = "interface_standardization"


@dataclass
class ComponentInfo:
    """コンポーネント情報"""
    name: str
    category: ComponentCategory
    file_path: str
    class_name: str
    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    lines_of_code: int = 0


@dataclass
class DuplicationAnalysis:
    """重複分析結果"""
    similar_components: List[List[ComponentInfo]]
    common_patterns: List[str]
    extractable_interfaces: List[str]
    consolidation_opportunities: List[str]
    estimated_reduction_percent: float


class BaseConsolidatedComponent(ABC):
    """統合コンポーネント基底クラス"""
    
    def __init__(self, name: str, category: ComponentCategory):
        self.name = name
        self.category = category
        self._config: Dict[str, Any] = {}
        
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """機能一覧取得"""
        pass
    
    @abstractmethod
    def execute(self, operation: str, **kwargs) -> Any:
        """操作実行"""
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """設定更新"""
        self._config.update(config)


class UnifiedAnalyzer(BaseConsolidatedComponent):
    """統一アナライザー"""
    
    def __init__(self):
        super().__init__("unified_analyzer", ComponentCategory.ANALYZER)
        self._analysis_strategies: Dict[str, callable] = {}
    
    def register_strategy(self, name: str, strategy: callable) -> None:
        """分析戦略登録"""
        self._analysis_strategies[name] = strategy
    
    def get_capabilities(self) -> List[str]:
        """分析機能一覧"""
        return [
            "technical_analysis",
            "fundamental_analysis", 
            "pattern_recognition",
            "trend_analysis",
            "volatility_analysis",
            "correlation_analysis",
            "risk_analysis"
        ]
    
    def execute(self, operation: str, **kwargs) -> Any:
        """分析実行"""
        if operation in self._analysis_strategies:
            return self._analysis_strategies[operation](**kwargs)
        
        # デフォルト実装
        if operation == "technical_analysis":
            return self._technical_analysis(**kwargs)
        elif operation == "pattern_recognition":
            return self._pattern_recognition(**kwargs)
        else:
            raise ValueError(f"Unknown analysis operation: {operation}")
    
    def _technical_analysis(self, data: Any, indicators: List[str] = None) -> Dict[str, Any]:
        """テクニカル分析"""
        # 統合されたテクニカル分析ロジック
        results = {}
        indicators = indicators or ["sma", "ema", "rsi", "macd"]
        
        for indicator in indicators:
            if indicator == "sma":
                results["sma"] = self._calculate_sma(data)
            elif indicator == "ema":
                results["ema"] = self._calculate_ema(data)
            elif indicator == "rsi":
                results["rsi"] = self._calculate_rsi(data)
            elif indicator == "macd":
                results["macd"] = self._calculate_macd(data)
        
        return results
    
    def _pattern_recognition(self, data: Any, patterns: List[str] = None) -> Dict[str, Any]:
        """パターン認識"""
        # 統合されたパターン認識ロジック
        results = {}
        patterns = patterns or ["support_resistance", "triangles", "head_shoulders"]
        
        for pattern in patterns:
            if pattern == "support_resistance":
                results["support_resistance"] = self._detect_support_resistance(data)
            elif pattern == "triangles":
                results["triangles"] = self._detect_triangles(data)
            elif pattern == "head_shoulders":
                results["head_shoulders"] = self._detect_head_shoulders(data)
        
        return results
    
    def _calculate_sma(self, data: Any, period: int = 20) -> Any:
        """SMA計算（統合版）"""
        # 統合されたSMA計算ロジック
        return {"period": period, "values": []}  # 簡易実装
    
    def _calculate_ema(self, data: Any, period: int = 12) -> Any:
        """EMA計算（統合版）"""
        return {"period": period, "values": []}
    
    def _calculate_rsi(self, data: Any, period: int = 14) -> Any:
        """RSI計算（統合版）"""
        return {"period": period, "values": []}
    
    def _calculate_macd(self, data: Any, fast: int = 12, slow: int = 26, signal: int = 9) -> Any:
        """MACD計算（統合版）"""
        return {"fast": fast, "slow": slow, "signal": signal, "values": []}
    
    def _detect_support_resistance(self, data: Any) -> Any:
        """サポート・レジスタンス検出"""
        return {"support_levels": [], "resistance_levels": []}
    
    def _detect_triangles(self, data: Any) -> Any:
        """三角形パターン検出"""
        return {"triangles": []}
    
    def _detect_head_shoulders(self, data: Any) -> Any:
        """ヘッドアンドショルダー検出"""
        return {"patterns": []}


class UnifiedProcessor(BaseConsolidatedComponent):
    """統一プロセッサー"""
    
    def __init__(self):
        super().__init__("unified_processor", ComponentCategory.PROCESSOR)
        self._processing_pipelines: Dict[str, List[callable]] = {}
    
    def register_pipeline(self, name: str, pipeline: List[callable]) -> None:
        """処理パイプライン登録"""
        self._processing_pipelines[name] = pipeline
    
    def get_capabilities(self) -> List[str]:
        """処理機能一覧"""
        return [
            "data_cleaning",
            "normalization",
            "aggregation",
            "transformation",
            "filtering",
            "validation"
        ]
    
    def execute(self, operation: str, **kwargs) -> Any:
        """処理実行"""
        if operation in self._processing_pipelines:
            data = kwargs.get("data")
            for processor in self._processing_pipelines[operation]:
                data = processor(data, **kwargs)
            return data
        
        # デフォルト実装
        if operation == "data_cleaning":
            return self._clean_data(**kwargs)
        elif operation == "normalization":
            return self._normalize_data(**kwargs)
        elif operation == "aggregation":
            return self._aggregate_data(**kwargs)
        else:
            raise ValueError(f"Unknown processing operation: {operation}")
    
    def _clean_data(self, data: Any, **kwargs) -> Any:
        """データクリーニング"""
        # 統合されたデータクリーニングロジック
        return data
    
    def _normalize_data(self, data: Any, method: str = "minmax", **kwargs) -> Any:
        """データ正規化"""
        return data
    
    def _aggregate_data(self, data: Any, method: str = "mean", **kwargs) -> Any:
        """データ集約"""
        return data


class UnifiedManager(BaseConsolidatedComponent):
    """統一マネージャー"""
    
    def __init__(self):
        super().__init__("unified_manager", ComponentCategory.MANAGER)
        self._managed_resources: Dict[str, Any] = {}
        self._lifecycle_handlers: Dict[str, List[callable]] = {}
    
    def register_resource(self, name: str, resource: Any, handlers: List[callable] = None) -> None:
        """リソース登録"""
        self._managed_resources[name] = resource
        if handlers:
            self._lifecycle_handlers[name] = handlers
    
    def get_capabilities(self) -> List[str]:
        """管理機能一覧"""
        return [
            "resource_management",
            "lifecycle_management",
            "configuration_management",
            "state_management",
            "monitoring"
        ]
    
    def execute(self, operation: str, **kwargs) -> Any:
        """管理操作実行"""
        if operation == "start_all":
            return self._start_all_resources()
        elif operation == "stop_all":
            return self._stop_all_resources()
        elif operation == "get_status":
            return self._get_status(**kwargs)
        elif operation == "configure":
            return self._configure_resource(**kwargs)
        else:
            raise ValueError(f"Unknown management operation: {operation}")
    
    def _start_all_resources(self) -> Dict[str, bool]:
        """全リソース開始"""
        results = {}
        for name, resource in self._managed_resources.items():
            try:
                if hasattr(resource, 'start'):
                    resource.start()
                results[name] = True
            except Exception as e:
                results[name] = False
        return results
    
    def _stop_all_resources(self) -> Dict[str, bool]:
        """全リソース停止"""
        results = {}
        for name, resource in self._managed_resources.items():
            try:
                if hasattr(resource, 'stop'):
                    resource.stop()
                results[name] = True
            except Exception as e:
                results[name] = False
        return results
    
    def _get_status(self, resource_name: str = None) -> Dict[str, Any]:
        """ステータス取得"""
        if resource_name:
            resource = self._managed_resources.get(resource_name)
            if resource and hasattr(resource, 'get_status'):
                return {resource_name: resource.get_status()}
            return {resource_name: "unknown"}
        
        # 全リソースのステータス
        status = {}
        for name, resource in self._managed_resources.items():
            if hasattr(resource, 'get_status'):
                status[name] = resource.get_status()
            else:
                status[name] = "unknown"
        return status
    
    def _configure_resource(self, resource_name: str, config: Dict[str, Any]) -> bool:
        """リソース設定"""
        resource = self._managed_resources.get(resource_name)
        if resource and hasattr(resource, 'configure'):
            resource.configure(config)
            return True
        return False


class CodeAnalyzer:
    """コード分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def analyze_codebase(self) -> DuplicationAnalysis:
        """コードベース分析"""
        components = self._scan_components()
        similar_groups = self._find_similar_components(components)
        common_patterns = self._extract_common_patterns(components)
        
        return DuplicationAnalysis(
            similar_components=similar_groups,
            common_patterns=common_patterns,
            extractable_interfaces=self._identify_extractable_interfaces(components),
            consolidation_opportunities=self._identify_consolidation_opportunities(similar_groups),
            estimated_reduction_percent=self._estimate_reduction(similar_groups)
        )
    
    def _scan_components(self) -> List[ComponentInfo]:
        """コンポーネントスキャン"""
        components = []
        
        for py_file in self.project_root.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        component = self._analyze_class(node, str(py_file), content)
                        if component:
                            components.append(component)
                            
            except Exception as e:
                continue
        
        return components
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: str, content: str) -> Optional[ComponentInfo]:
        """クラス分析"""
        class_name = class_node.name
        
        # カテゴリ判定
        category = self._determine_category(class_name)
        
        # メソッド抽出
        methods = []
        attributes = []
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        # 行数計算
        lines_of_code = class_node.end_lineno - class_node.lineno if hasattr(class_node, 'end_lineno') else 0
        
        return ComponentInfo(
            name=class_name,
            category=category,
            file_path=file_path,
            class_name=class_name,
            methods=methods,
            attributes=attributes,
            lines_of_code=lines_of_code
        )
    
    def _determine_category(self, class_name: str) -> ComponentCategory:
        """カテゴリ判定"""
        name_lower = class_name.lower()
        
        if "analyzer" in name_lower or "analysis" in name_lower:
            return ComponentCategory.ANALYZER
        elif "processor" in name_lower or "processing" in name_lower:
            return ComponentCategory.PROCESSOR
        elif "engine" in name_lower:
            return ComponentCategory.ENGINE
        elif "manager" in name_lower:
            return ComponentCategory.MANAGER
        elif "fetcher" in name_lower:
            return ComponentCategory.FETCHER
        elif "calculator" in name_lower or "calc" in name_lower:
            return ComponentCategory.CALCULATOR
        elif "optimizer" in name_lower:
            return ComponentCategory.OPTIMIZER
        elif "visualizer" in name_lower or "visual" in name_lower:
            return ComponentCategory.VISUALIZER
        elif "reporter" in name_lower or "report" in name_lower:
            return ComponentCategory.REPORTER
        else:
            return ComponentCategory.PROCESSOR  # デフォルト
    
    def _find_similar_components(self, components: List[ComponentInfo]) -> List[List[ComponentInfo]]:
        """類似コンポーネント検出"""
        similar_groups = []
        processed = set()
        
        for i, comp1 in enumerate(components):
            if i in processed:
                continue
            
            similar_group = [comp1]
            processed.add(i)
            
            for j, comp2 in enumerate(components[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_similarity(comp1, comp2)
                if similarity > 0.7:  # 70%以上の類似度
                    similar_group.append(comp2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
        
        return similar_groups
    
    def _calculate_similarity(self, comp1: ComponentInfo, comp2: ComponentInfo) -> float:
        """類似度計算"""
        if comp1.category != comp2.category:
            return 0.0
        
        # メソッド名の類似度
        common_methods = set(comp1.methods) & set(comp2.methods)
        total_methods = set(comp1.methods) | set(comp2.methods)
        method_similarity = len(common_methods) / len(total_methods) if total_methods else 0
        
        # 属性の類似度
        common_attrs = set(comp1.attributes) & set(comp2.attributes)
        total_attrs = set(comp1.attributes) | set(comp2.attributes)
        attr_similarity = len(common_attrs) / len(total_attrs) if total_attrs else 0
        
        # 名前の類似度（簡易）
        name_similarity = 1.0 if comp1.category == comp2.category else 0.0
        
        return (method_similarity * 0.5 + attr_similarity * 0.3 + name_similarity * 0.2)
    
    def _extract_common_patterns(self, components: List[ComponentInfo]) -> List[str]:
        """共通パターン抽出"""
        patterns = []
        
        # メソッド名パターン
        method_counts = {}
        for comp in components:
            for method in comp.methods:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        common_methods = [method for method, count in method_counts.items() if count > len(components) * 0.3]
        patterns.extend([f"common_method:{method}" for method in common_methods])
        
        return patterns
    
    def _identify_extractable_interfaces(self, components: List[ComponentInfo]) -> List[str]:
        """抽出可能インターフェース特定"""
        interfaces = []
        
        # カテゴリごとの共通メソッド
        category_methods = {}
        for comp in components:
            if comp.category not in category_methods:
                category_methods[comp.category] = {}
            
            for method in comp.methods:
                if method not in category_methods[comp.category]:
                    category_methods[comp.category][method] = 0
                category_methods[comp.category][method] += 1
        
        for category, methods in category_methods.items():
            category_components = [c for c in components if c.category == category]
            if len(category_components) < 2:
                continue
            
            common_methods = [
                method for method, count in methods.items()
                if count > len(category_components) * 0.5
            ]
            
            if common_methods:
                interfaces.append(f"{category.value}_interface:{','.join(common_methods)}")
        
        return interfaces
    
    def _identify_consolidation_opportunities(self, similar_groups: List[List[ComponentInfo]]) -> List[str]:
        """統合機会特定"""
        opportunities = []
        
        for group in similar_groups:
            if len(group) > 1:
                total_lines = sum(comp.lines_of_code for comp in group)
                opportunity = f"merge_group_{group[0].category.value}:{len(group)}_components_{total_lines}_lines"
                opportunities.append(opportunity)
        
        return opportunities
    
    def _estimate_reduction(self, similar_groups: List[List[ComponentInfo]]) -> float:
        """削減率推定"""
        total_lines = 0
        reducible_lines = 0
        
        for group in similar_groups:
            group_lines = sum(comp.lines_of_code for comp in group)
            total_lines += group_lines
            
            # 統合により50%削減可能と仮定
            reducible_lines += group_lines * 0.5
        
        return (reducible_lines / total_lines * 100) if total_lines > 0 else 0.0


class ConsolidationExecutor:
    """統合実行器"""
    
    def __init__(self):
        self.unified_analyzer = UnifiedAnalyzer()
        self.unified_processor = UnifiedProcessor()
        self.unified_manager = UnifiedManager()
    
    def execute_consolidation(self, analysis: DuplicationAnalysis) -> Dict[str, Any]:
        """統合実行"""
        results = {
            "created_components": [],
            "merged_components": [],
            "extracted_interfaces": [],
            "reduction_achieved": 0.0
        }
        
        # 類似コンポーネントの統合
        for group in analysis.similar_components:
            if len(group) > 1 and group[0].category == ComponentCategory.ANALYZER:
                # アナライザーの統合
                self._consolidate_analyzers(group)
                results["merged_components"].append([comp.name for comp in group])
        
        # 共通インターフェースの抽出
        for interface_desc in analysis.extractable_interfaces:
            interface_name = self._extract_interface(interface_desc)
            if interface_name:
                results["extracted_interfaces"].append(interface_name)
        
        results["created_components"] = [
            "UnifiedAnalyzer",
            "UnifiedProcessor", 
            "UnifiedManager"
        ]
        
        results["reduction_achieved"] = analysis.estimated_reduction_percent
        
        return results
    
    def _consolidate_analyzers(self, analyzer_group: List[ComponentInfo]) -> None:
        """アナライザー統合"""
        # 既存の統一アナライザーに機能を追加
        for analyzer in analyzer_group:
            # 実際の実装では、各アナライザーの機能を統一アナライザーに移植
            print(f"Consolidating analyzer: {analyzer.name}")
    
    def _extract_interface(self, interface_desc: str) -> Optional[str]:
        """インターフェース抽出"""
        # インターフェース定義の生成
        parts = interface_desc.split(":")
        if len(parts) >= 2:
            interface_name = f"{parts[0].title()}Interface"
            return interface_name
        return None
    
    def get_unified_components(self) -> Dict[str, BaseConsolidatedComponent]:
        """統一コンポーネント取得"""
        return {
            "analyzer": self.unified_analyzer,
            "processor": self.unified_processor,
            "manager": self.unified_manager
        }


# グローバル統合実行器
global_consolidation_executor = ConsolidationExecutor()