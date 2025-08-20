#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template Service - Issue #959 リファクタリング対応
HTMLテンプレート管理サービス
"""

class TemplateService:
    """HTMLテンプレート管理サービス"""

    @staticmethod
    def get_dashboard_template() -> str:
        """メインダッシュボードテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Day Trade Personal</h1>
            <p>プロダクション対応 - 個人投資家専用版</p>
        </div>

        <div class="dashboard">
            <!-- 機能メニューカード -->
            <div class="card">
                <h3>🔗 投資管理メニュー</h3>
                <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                    <button class="nav-btn" onclick="location.href='/portfolio'" style="background: #3b82f6; color: white; border: none; padding: 15px 25px; border-radius: 8px; font-weight: bold; cursor: pointer;">💼 ポートフォリオ追跡</button>
                </div>
            </div>
        </div>

        <!-- 拡張推奨銘柄セクション -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄分析</h2>

            <div id="progressIndicator" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; display: none; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div id="loadingSpinner" style="width: 24px; height: 24px; border: 3px solid #e5e7eb; border-top: 3px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <div style="flex: 1;">
                        <div id="progressMessage" style="font-weight: 600; color: #1f2937; margin-bottom: 8px;">データを読み込み中...</div>
                        <div style="width: 100%; background: #e5e7eb; border-radius: 6px; height: 6px; overflow: hidden;">
                            <div id="progressBarInline" style="width: 0%; height: 100%; background: linear-gradient(90deg, #3b82f6, #1d4ed8); transition: width 0.3s ease;"></div>
                        </div>
                        <div id="progressPercentage" style="margin-top: 5px; font-size: 0.8rem; color: #6b7280;">0% 完了</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>🎯 分析機能</h3>
                <p>主要銘柄の即座分析が可能です</p>
                <button class="btn" onclick="runAnalysis()">単一分析実行</button>
                <button class="btn" onclick="initializeData()" style="margin-left: 10px;">推奨銘柄表示</button>
                <div id="analysisResult" style="margin-top: 15px; padding: 10px; background: #f7fafc; border-radius: 6px; display: none;"></div>
            </div>
        </div>

        <!-- 拡張推奨銘柄セクション -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (35銘柄)</h2>
            <div id="filterButtons" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 15px;">
                <!-- フィルタリングボタンがJavaScriptで動的生成される -->
            </div>
            <div id="quickStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 8px; padding: 12px; background: #f9fafb; border-radius: 6px;">
                <!-- 統計情報がJavaScriptで動的生成される -->
            </div>
            <div id="recommendationsContainer">
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>

        <div class="footer">
            <p>© 2025 Day Trade Personal - プロダクション対応版</p>
        </div>
    </div>

    <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script>
        async function initializeData() {
            const recommendationsList = document.getElementById('recommendationsList');
            const progressIndicator = document.getElementById('progressIndicator');
            progressIndicator.style.display = 'block';
            progressIndicator.innerHTML = '<div style="display: flex; align-items: center; gap: 15px;"><div style="width: 24px; height: 24px; border: 3px solid #e5e7eb; border-top: 3px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite;"></div><div>推奨銘柄データを取得中...</div></div>';

            try {
                const response = await fetch('/api/recommendations');
                const data = await response.json();
                const recommendations = data.recommendations || [];
                displayRecommendationsList(recommendations);
            } catch (error) {
                console.error('データ取得エラー:', error);
                recommendationsList.innerHTML = '<p style="color: red; text-align: center;">データの読み込みに失敗しました。</p>';
            } finally {
                progressIndicator.style.display = 'none';
            }
        }

        function displayRecommendationsList(recommendations) {
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = recommendations.map(rec => `
                <div class="recommendation-card" style="border-left: 5px solid ${rec.recommendation === 'BUY' ? '#10b981' : rec.recommendation === 'SELL' ? '#ef4444' : '#f59e0b'}; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h3 style="margin: 0; font-size: 1.4rem; color: #1f2937;">${rec.symbol} (${rec.name})</h3>
                        <span style="padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; background: ${rec.recommendation === 'BUY' ? '#dcfdf7; color: #047857' : rec.recommendation === 'SELL' ? '#fee2e2; color: #dc2626' : '#fef3c7; color: #d97706'};">${rec.recommendation}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px; border-top: 1px solid #eee; padding-top: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">信頼度</div>
                            <div style="font-weight: 600; font-size: 1.1rem;">${rec.confidence ? (rec.confidence * 100).toFixed(0) : '-'}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">現在価格</div>
                            <div style="font-weight: 600; font-size: 1.1rem;">¥${rec.price ? rec.price.toLocaleString() : '-'}</div>
                        </div>
                    </div>
                    <div style="background: #f8fafc; padding: 15px; border-radius: 8px;">
                        <div style="margin-bottom: 10px;">
                            <strong style="color: #10b981;">購入タイミング:</strong>
                            <p style="margin: 5px 0 0 0; white-space: pre-wrap;">${rec.buy_timing || '分析中'}</p>
                        </div>
                        <div>
                            <strong style="color: #ef4444;">売却タイミング:</strong>
                            <p style="margin: 5px 0 0 0; white-space: pre-wrap;">${rec.sell_timing || '分析中'}</p>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        document.addEventListener('DOMContentLoaded', initializeData);
    </script>
</body>
</html>
        """

    @staticmethod
    def get_recommendations_template() -> str:
        """推奨銘柄フィルタリングテンプレート"""
        return """"""

    @staticmethod
    def get_portfolio_template() -> str:
        """ポートフォリオ追跡テンプレート（改善版）"""
        return """"""

    @staticmethod
    def get_scheduler_template() -> str:
        """スケジューラー管理テンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: white; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; color: #38bdf8; }
        .nav { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }
        .nav-btn { padding: 10px 20px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; color: white; cursor: pointer; transition: all 0.2s; }
        .nav-btn:hover { background: #334155; }

        .scheduler-status { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; }
        .status-number { font-size: 2rem; font-weight: bold; margin-bottom: 5px; color: #38bdf8; }
        .status-label { color: #94a3b8; }

        .tasks-grid { display: grid; gap: 20px; }
        .task-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; transition: transform 0.2s; }
        .task-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
        .task-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .task-name { font-size: 1.2rem; font-weight: bold; color: #f1f5f9; }
        .task-status { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
        .status-ready { background: #065f46; color: #34d399; }
        .status-running { background: #7c2d12; color: #fb923c; }
        .status-paused { background: #374151; color: #9ca3af; }

        .task-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 15px; }
        .task-detail { }
        .detail-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 2px; }
        .detail-value { color: #f1f5f9; font-weight: 500; }

        .task-actions { display: flex; gap: 10px; justify-content: center; }
        .action-btn { padding: 6px 12px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 500; transition: all 0.2s; }
        .btn-start { background: #059669; color: white; }
        .btn-start:hover { background: #047857; }
        .btn-pause { background: #d97706; color: white; }
        .btn-pause:hover { background: #b45309; }
        .btn-stop { background: #dc2626; color: white; }
        .btn-stop:hover { background: #b91c1c; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚙️ スケジューラー管理</h1>
            <p>自動化タスクの監視・管理システム</p>
        </div>

        <div class="nav">
            <button class="nav-btn" onclick="location.href=\'/\' ">ホーム</button>
            <button class="nav-btn" onclick="location.href=\'/swing-trade\' ">スイング</button>
        </div>

        <div class="scheduler-status" id="statusContainer">
            <!-- ステータス情報がJavaScriptで動的に表示される -->
        </div>

        <div class="tasks-grid" id="tasksContainer">
            <!-- タスク情報がJavaScriptで動的に表示される -->
        </div>
    </div>

    <script>
        async function loadSchedulerData() {
            try {
                const response = await fetch('/api/scheduler/tasks');
                const data = await response.json();

                displayStatus(data);
                displayTasks(data.tasks);
            } catch (error) {
                console.error('データ取得エラー:', error);
            }
        }

        function displayStatus(data) {
            const statusHtml = `
                <div class="status-card">
                    <div class="status-number">${data.total_tasks}</div>
                    <div class="status-label">総タスク数</div>
                </div>
                <div class="status-card">
                    <div class="status-number">${data.active_tasks}</div>
                    <div class="status-label">アクティブ</div>
                </div>
                <div class="status-card">
                    <div class="status-number">${data.avg_success_rate}%</div>
                    <div class="status-label">平均成功率</div>
                </div>
                <div class="status-card">
                    <div class="status-number">${data.scheduler_status === 'running' ? '稼働中' : '停止中'}</div>
                    <div class="status-label">スケジューラー状態</div>
                </div>
            `;
            document.getElementById('statusContainer').innerHTML = statusHtml;
        }

        function displayTasks(tasks) {
            const tasksHtml = tasks.map(task => `
                <div class="task-card">
                    <div class="task-header">
                        <div class="task-name">${task.name}</div>
                        <div class="task-status status-${task.status}">${task.status.toUpperCase()}</div>
                    </div>

                    <div class="task-details">
                        <div class="task-detail">
                            <div class="detail-label">スケジュール</div>
                            <div class="detail-value">${task.schedule_type}</div>
                        </div>
                        <div class="task-detail">
                            <div class="detail-label">実行時間</div>
                            <div class="detail-value">${task.schedule_time || task.interval_minutes ? task.interval_minutes + '分間隔' : '連続'}</div>
                        </div>
                        <div class="task-detail">
                            <div class="detail-label">成功率</div>
                            <div class="detail-value">${task.success_rate}%</div>
                        </div>
                        <div class="task-detail">
                            <div class="detail-label">最終実行</div>
                            <div class="detail-value">${new Date(task.last_execution).toLocaleString('ja-JP')}</div>
                        </div>
                        <div class="task-detail">
                            <div class="detail-label">次回実行</div>
                            <div class="detail-value">${new Date(task.next_execution).toLocaleString('ja-JP')}</div>
                        </div>
                    </div>

                    <div class="task-actions">
                        <button class="action-btn btn-start" onclick="startTask('${task.task_id}')">手動実行</button>
                        <button class="action-btn btn-pause">一時停止</button>
                        <button class="action-btn btn-stop">停止</button>
                    </div>
                </div>
            `).join('');
            document.getElementById('tasksContainer').innerHTML = tasksHtml;
        }

        async function startTask(taskId) {
            try {
                const response = await fetch(`/api/scheduler/start/${taskId}`, { method: 'POST' });
                const result = await response.json();

                if (response.ok) {
                    alert(`タスク「${result.task_name}」を手動実行しました`);
                    loadSchedulerData(); // データを再読み込み
                } else {
                    alert(`エラー: ${result.error}`);
                }
            } catch (error) {
                console.error('タスク実行エラー:', error);
                alert('タスク実行でエラーが発生しました');
            }
        }

        // 初期表示
        document.addEventListener('DOMContentLoaded', loadSchedulerData);

        // 30秒ごとに自動更新
        setInterval(loadSchedulerData, 30000);
    </script>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
        """