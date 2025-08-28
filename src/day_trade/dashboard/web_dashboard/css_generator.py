#!/usr/bin/env python3
"""
Webダッシュボード CSS生成モジュール

CSSスタイルファイル生成機能
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class CSSGenerator:
    """CSS生成クラス"""

    def create_css_file(self, static_dir: Path, security_manager):
        """CSSファイル作成"""
        css_content = """
body {
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.card-header {
    background-color: #fff;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
}

#status-report {
    font-family: 'Courier New', monospace;
    font-size: 11px;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 0.75rem;
    max-height: 400px;
    overflow-y: auto;
}

.chart-container {
    position: relative;
    height: 300px;
}

.chart-container img {
    max-width: 100%;
    height: auto;
}

#connection-status.disconnected {
    background-color: #dc3545 !important;
}

.alert-custom {
    border-radius: 0.5rem;
    border-left: 4px solid;
}

.alert-custom.alert-danger {
    border-left-color: #dc3545;
    background-color: #f8d7da;
}

.alert-custom.alert-warning {
    border-left-color: #ffc107;
    background-color: #fff3cd;
}

.spinner-border {
    width: 3rem;
    height: 3rem;
}

/* 分析ダッシュボード用スタイル */
.analysis-result {
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #fff;
}

.analysis-result.positive {
    border-left: 4px solid #28a745;
}

.analysis-result.negative {
    border-left: 4px solid #dc3545;
}

.analysis-result.neutral {
    border-left: 4px solid #ffc107;
}

.symbol-tag {
    font-family: 'Courier New', monospace;
    font-weight: bold;
    background-color: #e9ecef;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    margin-right: 0.5rem;
}

.recommendation-badge {
    font-size: 0.875rem;
    padding: 0.375rem 0.75rem;
}

#symbol-select {
    min-height: 150px;
}

.tier-buttons {
    margin-top: 0.5rem;
}

.tier-buttons .btn {
    margin-right: 0.25rem;
    margin-bottom: 0.25rem;
}

/* レスポンシブ対応 */
@media (max-width: 768px) {
    .card-body {
        padding: 0.75rem;
    }
    
    .chart-container {
        height: 250px;
    }
    
    #status-report {
        font-size: 10px;
        max-height: 300px;
    }
    
    .symbol-tag {
        font-size: 0.75rem;
        padding: 0.125rem 0.375rem;
    }
}

/* ダークモード対応 */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .card {
        background-color: #2d3748;
        border-color: #4a5568;
    }
    
    .card-header {
        background-color: #2d3748;
        border-color: #4a5568;
    }
    
    #status-report {
        background-color: #1a202c;
        border-color: #4a5568;
        color: #e2e8f0;
    }
    
    .analysis-result {
        background-color: #2d3748;
        border-color: #4a5568;
    }
    
    .symbol-tag {
        background-color: #4a5568;
        color: #e2e8f0;
    }
}

/* アニメーション効果 */
.card {
    transition: box-shadow 0.15s ease-in-out;
}

.card:hover {
    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
}

.btn {
    transition: all 0.15s ease-in-out;
}

.spinner-border {
    animation: spinner-border 0.75s linear infinite;
}

@keyframes spinner-border {
    to {
        transform: rotate(360deg);
    }
}

/* カスタムスクロールバー */
#status-report::-webkit-scrollbar {
    width: 6px;
}

#status-report::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

#status-report::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

#status-report::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* アクセシビリティ改善 */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
}

/* フォーカス表示の改善 */
button:focus,
.btn:focus {
    outline: 2px solid #0d6efd;
    outline-offset: 2px;
}

select:focus {
    outline: 2px solid #0d6efd;
    outline-offset: 2px;
}
        """

        # セキュアなファイル作成
        security_manager.create_secure_file(
            static_dir / "dashboard.css", css_content, 0o644
        )