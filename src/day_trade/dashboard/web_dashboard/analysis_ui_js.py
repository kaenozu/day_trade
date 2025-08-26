#!/usr/bin/env python3
"""
Webダッシュボード 分析UIJavaScript生成モジュール

分析ダッシュボードのUI機能JavaScript生成
"""

from pathlib import Path

from ...utils.logging_config import get_context_logger

logger = get_context_logger(__name__)


class AnalysisUIJSGenerator:
    """分析UIJavaScript生成クラス"""

    def generate_ui_js(self):
        """分析UI機能JavaScript生成"""
        return """
function showProgress() {
    document.getElementById('progress-container').style.display = 'block';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '準備中...';
}

function updateProgress(data) {
    const percent = (data.current / data.total * 100).toFixed(1);
    document.getElementById('progress-bar').style.width = percent + '%';
    document.getElementById('progress-bar').setAttribute('aria-valuenow', percent);
    document.getElementById('progress-text').textContent = 
        `${data.current}/${data.total} - 現在: ${data.currentSymbol || '準備中'}`;
}

function hideProgress() {
    document.getElementById('progress-container').style.display = 'none';
}

function clearResults() {
    document.getElementById('results-content').innerHTML = '';
}

function displayResults(results) {
    const container = document.getElementById('results-content');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<div class="text-muted">分析結果がありません</div>';
        return;
    }

    let html = '<div class="row">';
    
    results.forEach((result, index) => {
        const recommendation = result.recommendation || 'NEUTRAL';
        const confidenceScore = (result.confidence_score || 0) * 100;
        
        let cardClass = 'neutral';
        let badgeClass = 'bg-secondary';
        
        if (recommendation === 'BUY' || recommendation === 'STRONG_BUY') {
            cardClass = 'positive';
            badgeClass = 'bg-success';
        } else if (recommendation === 'SELL' || recommendation === 'STRONG_SELL') {
            cardClass = 'negative';
            badgeClass = 'bg-danger';
        }

        html += `
            <div class="col-md-6 mb-3">
                <div class="analysis-result ${cardClass}">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <div>
                            <span class="symbol-tag">${result.symbol}</span>
                            <span class="badge recommendation-badge ${badgeClass}">${recommendation}</span>
                        </div>
                        <small class="text-muted">信頼度: ${confidenceScore.toFixed(1)}%</small>
                    </div>
                    <div class="analysis-summary">
                        <small>${result.summary || '分析結果の要約がありません'}</small>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    container.innerHTML = html;
}

function saveAnalysisToHistory(results) {
    const historyItem = {
        timestamp: new Date().toISOString(),
        results: results,
        count: results.length
    };
    analysisHistory.unshift(historyItem);
    
    // 最新10件まで保持
    if (analysisHistory.length > 10) {
        analysisHistory = analysisHistory.slice(0, 10);
    }
    
    updateHistoryDisplay();
}

function loadAnalysisHistory() {
    // ローカルストレージから履歴を読み込み
    const saved = localStorage.getItem('analysisHistory');
    if (saved) {
        try {
            analysisHistory = JSON.parse(saved);
            updateHistoryDisplay();
        } catch (e) {
            console.error('履歴データの読み込みエラー:', e);
        }
    }
}

function updateHistoryDisplay() {
    const container = document.getElementById('history-content');
    
    if (analysisHistory.length === 0) {
        container.innerHTML = '<div class="text-muted">分析履歴はありません</div>';
        return;
    }
    
    let html = '';
    analysisHistory.forEach((item, index) => {
        const date = new Date(item.timestamp);
        const timeStr = date.toLocaleString('ja-JP');
        
        html += `
            <div class="history-item mb-2 p-2 border rounded">
                <div class="d-flex justify-content-between">
                    <span>${timeStr}</span>
                    <span class="badge bg-info">${item.count}銘柄</span>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
    
    // ローカルストレージに保存
    localStorage.setItem('analysisHistory', JSON.stringify(analysisHistory));
}

function updateStatistics() {
    let todayCount = 0;
    let buyCount = 0;
    let sellCount = 0;
    
    const today = new Date().toDateString();
    
    analysisHistory.forEach(item => {
        const itemDate = new Date(item.timestamp).toDateString();
        if (itemDate === today) {
            todayCount += item.count;
            
            item.results.forEach(result => {
                const rec = result.recommendation;
                if (rec === 'BUY' || rec === 'STRONG_BUY') {
                    buyCount++;
                } else if (rec === 'SELL' || rec === 'STRONG_SELL') {
                    sellCount++;
                }
            });
        }
    });
    
    document.getElementById('today-analysis-count').textContent = todayCount;
    document.getElementById('buy-recommendations').textContent = buyCount;
    document.getElementById('sell-recommendations').textContent = sellCount;
}

function showError(message) {
    const container = document.getElementById('results-content');
    container.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-triangle me-2"></i>${message}</div>`;
}

function showAlert(message, type = 'info') {
    // 簡易アラート表示
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 5秒後に自動削除
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

// 初期統計更新
document.addEventListener('DOMContentLoaded', function() {
    updateStatistics();
});
        """