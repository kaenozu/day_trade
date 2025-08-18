// デイトレード分析ダッシュボード JavaScript

// WebSocket接続
const socket = io();

// グローバル状態
let isConnected = false;
let currentResults = [];
let tierSymbols = {
    1: [], // 主要銘柄
    2: [], // 拡張セット
    3: [], // 包括セット
    4: []  // 全東証銘柄
};

// 初期化
document.addEventListener('DOMContentLoaded', function() {
    initializeAnalysisDashboard();
    loadSymbolData();
});

function initializeAnalysisDashboard() {
    // WebSocketイベントリスナー
    socket.on('connect', function() {
        isConnected = true;
        updateConnectionStatus(true);
        console.log('WebSocketに接続されました');
    });

    socket.on('disconnect', function() {
        isConnected = false;
        updateConnectionStatus(false);
        console.log('WebSocketから切断されました');
    });

    socket.on('analysis_progress', function(data) {
        updateProgressBar(data.current, data.total, data.currentSymbol);
    });

    socket.on('analysis_complete', function(data) {
        hideProgressBar();
        displayAnalysisResults(data.results);
        showAlert('分析が完了しました！', 'success');
    });

    socket.on('analysis_error', function(data) {
        hideProgressBar();
        showAlert('分析中にエラーが発生しました: ' + data.message, 'danger');
        enableAnalyzeButton();
    });

    socket.on('error', function(data) {
        showAlert('エラー: ' + data.message, 'danger');
    });

    // 分析ボタンイベント
    document.getElementById('analyze-btn').addEventListener('click', function() {
        startAnalysis();
    });

    // 手動更新ボタン
    document.getElementById('refresh-btn').addEventListener('click', function() {
        if (currentResults.length > 0) {
            showAlert('分析結果を更新中...', 'info');
            startAnalysis();
        }
    });

    // 銘柄選択ダブルクリックで分析実行
    document.getElementById('symbol-select').addEventListener('dblclick', function() {
        startAnalysis();
    });
}

function loadSymbolData() {
    // 銘柄マスタデータ読み込み
    fetch('/api/symbols')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateSymbolSelect(data.symbols);
                tierSymbols = data.tier_symbols || tierSymbols;
            }
        })
        .catch(error => {
            console.error('銘柄データ読み込みエラー:', error);
            // フォールバック: デフォルト銘柄を使用
            const defaultSymbols = [
                {code: '7203', name: 'トヨタ自動車'},
                {code: '8306', name: '三菱UFJ銀行'},
                {code: '9984', name: 'ソフトバンクグループ'},
                {code: '6758', name: 'ソニー'},
                {code: '7974', name: '任天堂'},
                {code: '4689', name: 'ヤフー'},
                {code: '9434', name: 'ソフトバンク'},
                {code: '6861', name: 'キーエンス'}
            ];
            updateSymbolSelect(defaultSymbols);
        });
}

function updateSymbolSelect(symbols) {
    const selectElement = document.getElementById('symbol-select');
    selectElement.innerHTML = '';
    
    symbols.forEach(symbol => {
        const option = document.createElement('option');
        option.value = symbol.code;
        option.textContent = `${symbol.code} - ${symbol.name}`;
        selectElement.appendChild(option);
    });
}

function selectTier(tier) {
    const selectElement = document.getElementById('symbol-select');
    
    // 全選択解除
    for (let option of selectElement.options) {
        option.selected = false;
    }
    
    // ティア別銘柄選択（API経由で取得）
    fetch(`/api/tier-symbols/${tier}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const symbols = data.symbols;
                
                // 該当する銘柄を選択
                for (let option of selectElement.options) {
                    if (symbols.includes(option.value)) {
                        option.selected = true;
                    }
                }
                
                // フィードバック表示
                const tierNames = {
                    1: '主要銘柄',
                    2: '拡張セット',
                    3: '包括セット',
                    4: '全東証銘柄'
                };
                showAlert(`${tierNames[tier]} (${symbols.length}銘柄) を選択しました`, 'info');
            }
        })
        .catch(error => {
            console.error('ティア別銘柄取得エラー:', error);
            showAlert('銘柄選択でエラーが発生しました', 'warning');
        });
}

function startAnalysis() {
    const selectElement = document.getElementById('symbol-select');
    const selectedSymbols = Array.from(selectElement.selectedOptions).map(option => option.value);
    
    if (selectedSymbols.length === 0) {
        showAlert('分析する銘柄を選択してください', 'warning');
        return;
    }
    
    // 大量銘柄の警告
    if (selectedSymbols.length > 100) {
        if (!confirm(`${selectedSymbols.length}銘柄の分析には時間がかかります。続行しますか？`)) {
            return;
        }
    }
    
    // UI状態更新
    disableAnalyzeButton();
    showProgressBar();
    hideResults();
    
    // 分析API呼び出し
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
        if (data.success) {
            // WebSocket経由で進捗を受信
            console.log('分析開始:', data.message);
        } else {
            hideProgressBar();
            enableAnalyzeButton();
            showAlert('分析開始に失敗しました: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('分析API呼び出しエラー:', error);
        hideProgressBar();
        enableAnalyzeButton();
        showAlert('分析開始でエラーが発生しました', 'danger');
    });
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.className = 'badge bg-success me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 接続中';
    } else {
        statusElement.className = 'badge bg-danger me-3';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> 切断';
    }
}

function showProgressBar() {
    document.getElementById('analysis-progress').style.display = 'block';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '分析を開始しています...';
}

function updateProgressBar(current, total, currentSymbol = '') {
    const percentage = Math.round((current / total) * 100);
    document.getElementById('progress-bar').style.width = percentage + '%';
    
    let text = `${current}/${total} 銘柄完了 (${percentage}%)`;
    if (currentSymbol) {
        text += ` - 現在処理中: ${currentSymbol}`;
    }
    document.getElementById('progress-text').textContent = text;
}

function hideProgressBar() {
    document.getElementById('analysis-progress').style.display = 'none';
    enableAnalyzeButton();
}

function showResults() {
    document.getElementById('results-container').style.display = 'block';
    document.getElementById('results-container').classList.add('fade-in-up');
}

function hideResults() {
    document.getElementById('results-container').style.display = 'none';
}

function disableAnalyzeButton() {
    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>分析中...';
}

function enableAnalyzeButton() {
    const btn = document.getElementById('analyze-btn');
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-play me-2"></i>分析開始';
}

function displayAnalysisResults(results) {
    currentResults = results;
    
    // サマリー更新
    updateSummaryCards(results);
    
    // 推奨別カード更新
    updateRecommendationCards(results);
    
    // 詳細テーブル更新
    updateResultsTable(results);
    
    // 結果表示
    showResults();
}

function updateSummaryCards(results) {
    const summary = {
        buy: 0,
        sell: 0,
        hold: 0,
        totalConfidence: 0,
        validResults: 0
    };
    
    results.forEach(result => {
        const rec = result.recommendation;
        if (rec !== 'SKIP') {
            summary.validResults++;
            summary.totalConfidence += result.confidence || 0;
            
            if (rec === 'BUY') summary.buy++;
            else if (rec === 'SELL') summary.sell++;
            else if (rec === 'HOLD') summary.hold++;
        }
    });
    
    document.getElementById('buy-count').textContent = summary.buy;
    document.getElementById('sell-count').textContent = summary.sell;
    document.getElementById('hold-count').textContent = summary.hold;
    
    const avgConfidence = summary.validResults > 0 
        ? Math.round((summary.totalConfidence / summary.validResults) * 100)
        : 0;
    document.getElementById('avg-confidence').textContent = avgConfidence + '%';
}

function updateRecommendationCards(results) {
    const buyContainer = document.getElementById('buy-recommendations');
    const sellContainer = document.getElementById('sell-recommendations');
    const holdContainer = document.getElementById('hold-recommendations');
    
    // カード内容クリア
    buyContainer.innerHTML = '';
    sellContainer.innerHTML = '';
    holdContainer.innerHTML = '';
    
    const buyResults = results.filter(r => r.recommendation === 'BUY');
    const sellResults = results.filter(r => r.recommendation === 'SELL');
    const holdResults = results.filter(r => r.recommendation === 'HOLD');
    
    // BUY推奨
    if (buyResults.length === 0) {
        buyContainer.innerHTML = '<p class="text-muted">BUY推奨の銘柄はありません</p>';
    } else {
        buyResults.forEach(result => {
            buyContainer.appendChild(createRecommendationCard(result));
        });
    }
    
    // SELL推奨
    if (sellResults.length === 0) {
        sellContainer.innerHTML = '<p class="text-muted">SELL推奨の銘柄はありません</p>';
    } else {
        sellResults.forEach(result => {
            sellContainer.appendChild(createRecommendationCard(result));
        });
    }
    
    // HOLD推奨
    if (holdResults.length === 0) {
        holdContainer.innerHTML = '<p class="text-muted">HOLD推奨の銘柄はありません</p>';
    } else {
        holdResults.forEach(result => {
            holdContainer.appendChild(createRecommendationCard(result));
        });
    }
}

function createRecommendationCard(result) {
    const card = document.createElement('div');
    card.className = 'recommendation-item slide-in-right';
    
    const confidence = Math.round((result.confidence || 0) * 100);
    const confidenceClass = confidence >= 80 ? 'high' : confidence >= 60 ? 'medium' : 'low';
    
    card.innerHTML = `
        <div class="d-flex justify-content-between align-items-start">
            <div>
                <div class="symbol-code">${result.symbol}</div>
                <div class="company-name">${getCompanyName(result.symbol)}</div>
            </div>
            <div class="text-end">
                <div class="badge bg-primary">${confidence}%</div>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
        </div>
        ${result.reason ? `<small class="text-muted mt-1 d-block">${result.reason}</small>` : ''}
    `;
    
    return card;
}

function updateResultsTable(results) {
    const tableBody = document.getElementById('results-table-body');
    tableBody.innerHTML = '';
    
    if (results.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center text-muted">
                    分析結果がありません
                </td>
            </tr>
        `;
        return;
    }
    
    results.forEach(result => {
        const row = document.createElement('tr');
        const rec = result.recommendation;
        const confidence = Math.round((result.confidence || 0) * 100);
        
        // 推奨バッジのクラス
        let badgeClass = 'recommendation-skip';
        if (rec === 'BUY') badgeClass = 'recommendation-buy';
        else if (rec === 'SELL') badgeClass = 'recommendation-sell';
        else if (rec === 'HOLD') badgeClass = 'recommendation-hold';
        
        row.innerHTML = `
            <td><span class="symbol-code">${result.symbol}</span></td>
            <td>${getCompanyName(result.symbol)}</td>
            <td><span class="recommendation-badge ${badgeClass}">${rec}</span></td>
            <td>${confidence}%</td>
            <td>${result.current_price ? '¥' + Math.round(result.current_price).toLocaleString() : '-'}</td>
            <td>${result.current_rsi ? result.current_rsi.toFixed(1) : '-'}</td>
            <td>${result.current_macd ? result.current_macd.toFixed(3) : '-'}</td>
            <td>${result.sma_20 ? '¥' + Math.round(result.sma_20).toLocaleString() : '-'}</td>
            <td><small>${result.reason || result.error || '-'}</small></td>
        `;
        
        tableBody.appendChild(row);
    });
}

function getCompanyName(symbol) {
    // 簡易的な会社名マッピング（実際は設定ファイルから取得）
    const companyMap = {
        '7203': 'トヨタ自動車',
        '8306': '三菱UFJ銀行',
        '9984': 'ソフトバンクグループ',
        '6758': 'ソニー',
        '7974': '任天堂',
        '4689': 'ヤフー',
        '9434': 'ソフトバンク',
        '6861': 'キーエンス'
    };
    return companyMap[symbol] || symbol;
}

function exportResults() {
    if (currentResults.length === 0) {
        showAlert('エクスポートする結果がありません', 'warning');
        return;
    }
    
    // CSV形式でエクスポート
    const headers = ['銘柄コード', '企業名', '推奨', '信頼度', '現在価格', 'RSI', 'MACD', 'SMA20', '判断理由'];
    const csvContent = [
        headers.join(','),
        ...currentResults.map(result => [
            result.symbol,
            getCompanyName(result.symbol),
            result.recommendation,
            Math.round((result.confidence || 0) * 100) + '%',
            result.current_price ? Math.round(result.current_price) : '',
            result.current_rsi ? result.current_rsi.toFixed(1) : '',
            result.current_macd ? result.current_macd.toFixed(3) : '',
            result.sma_20 ? Math.round(result.sma_20) : '',
            result.reason || result.error || ''
        ].map(field => `"${field}"`).join(','))
    ].join('\n');
    
    // ダウンロード
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `分析結果_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showAlert('分析結果をCSVファイルでダウンロードしました', 'success');
}

function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    const alertId = 'alert-' + Date.now();
    
    const alertHTML = `
        <div id="${alertId}" class="alert alert-${type} alert-custom alert-dismissible fade show" role="alert">
            <strong><i class="fas fa-info-circle me-2"></i></strong>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;
    
    alertsContainer.insertAdjacentHTML('beforeend', alertHTML);
    
    // 5秒後に自動削除
    setTimeout(function() {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}