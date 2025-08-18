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
        
    } catch (error) {
        console.error('推奨銘柄読み込みエラー:', error);
        listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #f56565;">エラーが発生しました: ' + error.message + '</div>';
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

setInterval(updateStatus, 10000);
updateStatus();
