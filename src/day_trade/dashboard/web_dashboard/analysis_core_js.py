#!/usr/bin/env python3
"""
Webダッシュボード 分析コアJavaScript生成モジュール

分析ダッシュボードのコア機能JavaScript生成
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AnalysisCoreJSGenerator:
    """分析コアJavaScript生成クラス"""

    def generate_core_js(self):
        """分析コア機能JavaScript生成"""
        return """
// WebSocket接続
const socket = io();

// グローバル変数
let symbolsData = {};
let isAnalyzing = false;
let analysisHistory = [];

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeAnalysis();
    loadSymbols();
    loadAnalysisHistory();
});

function initializeAnalysis() {
    // WebSocketイベントリスナー
    socket.on('connect', function() {
        updateConnectionStatus(true);
        console.log('分析ダッシュボード: WebSocket接続');
    });

    socket.on('disconnect', function() {
        updateConnectionStatus(false);
        console.log('分析ダッシュボード: WebSocket切断');
    });

    socket.on('analysis_progress', function(data) {
        updateProgress(data);
    });

    socket.on('analysis_complete', function(data) {
        displayResults(data.results);
        saveAnalysisToHistory(data.results);
        hideProgress();
        isAnalyzing = false;
        document.getElementById('analyze-btn').disabled = false;
        updateStatistics();
    });

    socket.on('analysis_error', function(data) {
        showError('分析エラー: ' + data.message);
        hideProgress();
        isAnalyzing = false;
        document.getElementById('analyze-btn').disabled = false;
    });

    // キーボードショートカット
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey || event.metaKey) {
            switch(event.key) {
                case 'Enter':
                    event.preventDefault();
                    if (!isAnalyzing) {
                        runAnalysis();
                    }
                    break;
                case 'Escape':
                    event.preventDefault();
                    clearSelection();
                    break;
            }
        }
    });
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.className = 'badge bg-light text-dark me-3';
        statusElement.innerHTML = '<i class="fas fa-circle text-success"></i> 接続中';
    } else {
        statusElement.className = 'badge bg-danger me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 切断';
    }
}

function loadSymbols() {
    fetch('/api/symbols')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                symbolsData = data;
                populateSymbolSelect(data.symbols);
            } else {
                showError('銘柄データの取得に失敗しました: ' + data.error);
            }
        })
        .catch(error => {
            console.error('銘柄取得エラー:', error);
            showError('銘柄データの取得中にエラーが発生しました');
        });
}

function populateSymbolSelect(symbols) {
    const select = document.getElementById('symbol-select');
    select.innerHTML = '';

    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol.code;
        option.textContent = `${symbol.code} - ${symbol.name}`;
        select.appendChild(option);
    });
}

function selectTierSymbols(tier) {
    fetch(`/api/tier-symbols/${tier}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const select = document.getElementById('symbol-select');
                clearSelection();
                
                Array.from(select.options).forEach(option => {
                    if (data.symbols.includes(option.value)) {
                        option.selected = true;
                    }
                });
                showAlert(`Tier${tier}の${data.symbols.length}銘柄を選択しました`, 'info');
            }
        })
        .catch(error => {
            console.error(`Tier${tier}銘柄取得エラー:`, error);
            showError(`Tier${tier}銘柄の取得に失敗しました`);
        });
}

function clearSelection() {
    const select = document.getElementById('symbol-select');
    Array.from(select.options).forEach(option => {
        option.selected = false;
    });
}

function runAnalysis() {
    if (isAnalyzing) {
        return;
    }

    const select = document.getElementById('symbol-select');
    const selectedSymbols = Array.from(select.selectedOptions).map(option => option.value);

    if (selectedSymbols.length === 0) {
        showError('分析する銘柄を選択してください');
        return;
    }

    if (selectedSymbols.length > 50) {
        showError('一度に分析できる銘柄は50個までです');
        return;
    }

    isAnalyzing = true;
    document.getElementById('analyze-btn').disabled = true;
    showProgress();
    clearResults();

    // 分析実行要求
    fetch('/api/analysis', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            symbols: selectedSymbols,
            mode: 'web'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            showError('分析開始に失敗しました: ' + data.error);
            hideProgress();
            isAnalyzing = false;
            document.getElementById('analyze-btn').disabled = false;
        } else {
            showAlert(`${data.symbol_count}銘柄の分析を開始しました`, 'success');
        }
    })
    .catch(error => {
        console.error('分析実行エラー:', error);
        showError('分析実行中にエラーが発生しました');
        hideProgress();
        isAnalyzing = false;
        document.getElementById('analyze-btn').disabled = false;
    });
}
        """