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
        
        // システム状態を定期更新
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('システム状態:', data.status);
            } catch (error) {
                console.error('状態更新エラー:', error);
            }
        }
        
        // 10秒ごとに状態更新
        setInterval(updateStatus, 10000);
        updateStatus();