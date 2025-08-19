async function runAnalysis() {
    const resultDiv = document.getElementById('analysisResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '分析中...';
    
    try {
        const response = await fetch('/api/analysis/7203');
        const data = await response.json();
        
        resultDiv.innerHTML = `
            <strong>トヨタ自動車 (${data.symbol})</strong><br>
            推奨: ${data.recommendation}<br>
            信頼度: ${(data.confidence * 100).toFixed(1)}%<br>
            価格: ¥${data.price}<br>
            変動: ${data.change > 0 ? '+' : ''}${data.change}%
        `;
    } catch (error) {
        resultDiv.innerHTML = 'エラーが発生しました: ' + error.message;
    }
}

async function loadRecommendations() {
    const container = document.getElementById('recommendationsContainer');
    const summaryDiv = document.getElementById('summaryStats');
    const listDiv = document.getElementById('recommendationsList');
    
    container.style.display = 'block';
    listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">推奨銘柄を読み込み中...</div>';
    
    // 更新状態の表示
    updateStatusDisplay('更新中...', false);
    
    try {
        const response = await fetch('/api/recommendations');
        const data = await response.json();
        
        summaryDiv.innerHTML = `
            <div class="stat-item">
                <div class="stat-number">${data.total_count}</div>
                <div class="stat-label">総銘柄数</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.high_confidence_count}</div>
                <div class="stat-label">高信頼度</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.buy_count}</div>
                <div class="stat-label">買い推奨</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.sell_count}</div>
                <div class="stat-label">売り推奨</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.hold_count}</div>
                <div class="stat-label">様子見</div>
            </div>
        `;
        
        let recommendationsHtml = '';
        data.recommendations.forEach(stock => {
            const recClass = `rec-${stock.recommendation.toLowerCase()}`;
            const confidenceClass = getConfidenceClass(stock.confidence);
            const badgeClass = getBadgeClass(stock.recommendation);
            const changeColor = stock.change >= 0 ? '#48bb78' : '#f56565';
            const changePrefix = stock.change >= 0 ? '+' : '';
            
            recommendationsHtml += `
                <div class="recommendation-card ${recClass} ${confidenceClass}">
                    <div class="stock-header">
                        <div>
                            <div class="stock-name">
                                ${stock.name}
                                <span class="stock-category">${stock.category}</span>
                            </div>
                            <div class="stock-symbol">${stock.symbol} | ${stock.sector}</div>
                            <div class="star-rating">${stock.star_rating}</div>
                        </div>
                        <div class="rec-badge ${badgeClass}">${stock.recommendation_friendly || stock.recommendation}</div>
                    </div>
                    <div class="price-info">
                        <div>
                            <strong>¥${stock.price.toLocaleString()}</strong>
                            <span style="color: ${changeColor}; margin-left: 8px;">
                                ${changePrefix}${stock.change}%
                            </span>
                        </div>
                        <div>
                            <span style="font-size: 0.9rem; color: #4a5568; font-weight: bold;">
                                ${stock.confidence_friendly || 'おすすめ度: ' + (stock.confidence * 100).toFixed(1) + '%'}
                            </span>
                        </div>
                    </div>
                    <div class="stock-details">
                        <div class="detail-row">
                            <span class="detail-label">安全度:</span>
                            <span class="detail-value">${stock.risk_friendly || stock.risk_level}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">安定性:</span>
                            <span class="detail-value">${stock.stability}</span>
                        </div>
                        <div class="who-suitable">${stock.who_suitable}</div>
                    </div>
                    <div class="reason-friendly">
                        ${stock.friendly_reason || stock.reason}
                    </div>
                </div>
            `;
        });
        
        listDiv.innerHTML = recommendationsHtml;
        
        // 更新完了状態の表示
        updateStatusDisplay('更新完了', true);
        
    } catch (error) {
        console.error('推奨銘柄読み込みエラー:', error);
        listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #f56565;">エラーが発生しました: ' + error.message + '</div>';
        updateStatusDisplay('エラー', false, true);
    }
}

function getConfidenceClass(confidence) {
    if (confidence > 0.85) return 'confidence-high';
    if (confidence > 0.70) return 'confidence-medium';
    return 'confidence-low';
}

function getBadgeClass(recommendation) {
    switch (recommendation) {
        case 'BUY': return 'buy-badge';
        case 'SELL': return 'sell-badge';
        case 'HOLD': return 'hold-badge';
        default: return 'hold-badge';
    }
}

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        console.log('システム状態:', data.status);
    } catch (error) {
        console.error('状態更新エラー:', error);
    }
}

// 画面読み込み完了時の初回自動更新
document.addEventListener('DOMContentLoaded', function() {
    console.log('画面読み込み完了 - 初回データ更新を開始');
    
    // 初回推奨銘柄の自動読み込み
    loadRecommendations();
    
    // システム状態の更新
    updateStatus();
    
    // 定期的な状態更新（10秒間隔）
    setInterval(updateStatus, 10000);
    
    console.log('自動更新システム開始完了');
});

// 手動更新ボタン用の関数（既存の機能を維持）
function manualRefresh() {
    console.log('手動更新が要求されました');
    loadRecommendations();
}

// リアルタイム価格更新機能
async function updateRealTimeData() {
    try {
        const response = await fetch('/api/realtime/snapshot');
        const data = await response.json();
        
        if (data.prices && Object.keys(data.prices).length > 0) {
            console.log('リアルタイム価格データを更新:', Object.keys(data.prices).length + '銘柄');
            updatePriceDisplay(data.prices);
        }
    } catch (error) {
        console.error('リアルタイムデータ更新エラー:', error);
    }
}

// 価格表示の更新
function updatePriceDisplay(prices) {
    // 推奨銘柄カードの価格を更新
    document.querySelectorAll('.recommendation-card').forEach(card => {
        const symbolElement = card.querySelector('.stock-symbol');
        if (symbolElement) {
            const symbol = symbolElement.textContent.split(' ')[0];
            if (prices[symbol]) {
                const priceInfo = card.querySelector('.price-info strong');
                if (priceInfo) {
                    priceInfo.textContent = `¥${prices[symbol].current_price.toLocaleString()}`;
                    
                    // 価格変動の色を更新
                    const changeElement = priceInfo.nextElementSibling;
                    if (changeElement) {
                        const change = prices[symbol].price_change_pct;
                        const changeColor = change >= 0 ? '#48bb78' : '#f56565';
                        const changePrefix = change >= 0 ? '+' : '';
                        changeElement.style.color = changeColor;
                        changeElement.textContent = `${changePrefix}${change}%`;
                    }
                }
            }
        }
    });
}

// 拡張された自動更新システム
function startEnhancedAutoUpdate() {
    // 推奨銘柄の定期更新（5分間隔）
    setInterval(() => {
        console.log('定期的な推奨銘柄更新を実行');
        loadRecommendations();
    }, 300000); // 5分

    // リアルタイム価格の定期更新（30秒間隔）
    setInterval(() => {
        updateRealTimeData();
    }, 30000); // 30秒
    
    console.log('拡張自動更新システムが開始されました');
}

// 拡張自動更新の開始
setTimeout(startEnhancedAutoUpdate, 2000); // 2秒後に開始

// 状態表示の更新
function updateStatusDisplay(message, isSuccess = true, isError = false) {
    const statusText = document.getElementById('statusText');
    const lastUpdate = document.getElementById('lastUpdate');
    
    if (statusText) {
        statusText.textContent = message;
        statusText.style.color = isError ? '#f56565' : isSuccess ? '#48bb78' : '#ed8936';
    }
    
    if (lastUpdate && isSuccess) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('ja-JP', { 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
        lastUpdate.textContent = `(最終更新: ${timeString})`;
    }
}

// ページの可視性変更時の処理
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        console.log('ページが再表示されました - データを更新');
        loadRecommendations();
    }
});

// ネットワーク接続状態の監視
window.addEventListener('online', function() {
    console.log('インターネット接続が復帰しました');
    updateStatusDisplay('接続復帰 - 更新中...', false);
    loadRecommendations();
});

window.addEventListener('offline', function() {
    console.log('インターネット接続が切断されました');
    updateStatusDisplay('オフライン', false, true);
});
