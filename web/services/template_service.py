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
            
            <!-- インライン進捗インジケーター -->
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
                <button class="btn" onclick="loadRecommendations()" style="margin-left: 10px;">推奨銘柄表示</button>
                <div id="analysisResult" style="margin-top: 15px; padding: 10px; background: #f7fafc; border-radius: 6px; display: none;"></div>
            </div>
        </div>

        <!-- 拡張推奨銘柄セクション -->
        <div class="recommendations-section" style="margin-top: 30px;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">📈 推奨銘柄一覧 (35銘柄)</h2>
            <div id="recommendationsContainer" style="display: none;">
                <div class="recommendations-summary" style="background: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
                    <div id="summaryStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px;"></div>
                </div>
                <div id="filterButtons" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 15px;">
                    <!-- フィルタリングボタンがJavaScriptで動的生成される -->
                </div>
                <div id="quickStats" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 8px; padding: 12px; background: #f9fafb; border-radius: 6px;">
                    <!-- 統計情報がJavaScriptで動的生成される -->
                </div>
            </div>

            <div id="recommendationsContainer" style="display: none;">
                <div id="recommendationsList" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;"></div>
            </div>
        </div>

        <div class="footer">
            <p>© 2025 Day Trade Personal - プロダクション対応版</p>
        </div>
    </div>

    <style>
        .clickable-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .clickable-stat {
            text-align: center;
            padding: 20px;
            background: #ffffff;
            color: #1f2937;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .clickable-stat:hover {
            border-color: #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
            transform: translateY(-2px);
        }
        .clickable-stat.active {
            background: #3b82f6;
            color: #ffffff;
            border-color: #3b82f6;
        }
        .clickable-stat.active .stat-number {
            color: #ffffff;
        }
        .clickable-stat.active .stat-label {
            color: #ffffff;
            opacity: 0.9;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9rem;
            font-weight: 500;
            opacity: 0.8;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script>
        let currentFilter = '';
        let allRecommendations = [];
        
        // フィルタリング関数（クライアントサイド）
        function filterRecommendations(filter) {
            currentFilter = filter;
            
            // フィルターボタンのアクティブ状態更新
            setActiveFilter(filter);
            
            // クライアントサイドフィルタリング実行
            applyClientSideFilter();
        }
        
        // クライアントサイドフィルタリング実行
        function applyClientSideFilter() {
            if (!allRecommendations.length) {
                loadAllRecommendations();
                return;
            }
            
            let filteredRecommendations = [...allRecommendations];
            
            // フィルタ条件適用
            if (currentFilter === 'high') {
                filteredRecommendations = allRecommendations.filter(r => r.confidence >= 0.8);
            } else if (currentFilter === 'buy') {
                filteredRecommendations = allRecommendations.filter(r => r.recommendation === 'BUY');
            } else if (currentFilter === 'growth') {
                filteredRecommendations = allRecommendations.filter(r => r.category === '成長株');
            }
            
            // 統計計算
            const stats = {
                total_count: allRecommendations.length,
                filtered_count: filteredRecommendations.length,
                high_confidence_count: allRecommendations.filter(r => r.confidence >= 0.8).length,
                buy_count: allRecommendations.filter(r => r.recommendation === 'BUY').length,
                growth_count: allRecommendations.filter(r => r.category === '成長株').length,
                recommendations: filteredRecommendations
            };
            
            displayFilteredRecommendations(stats);
            updateStatNumbers(stats);
        }
        
        // 全推奨銘柄を取得（初回のみ）
        async function loadAllRecommendations() {
            try {
                document.getElementById('summaryStats').innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">データを取得中...</div>';
                document.getElementById('recommendationsList').innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">銘柄情報を読み込み中...</div>';
                
                const response = await fetch('/api/recommendations');
                const data = await response.json();
                
                allRecommendations = data.recommendations || [];
                applyClientSideFilter();
            } catch (error) {
                console.error('データ取得エラー:', error);
                document.getElementById('summaryStats').innerHTML = '<div style="text-align: center; padding: 40px; color: #dc2626;">データ取得エラーが発生しました</div>';
            }
        }
        
        // フィルタリング結果表示（新構造対応版）
        function displayFilteredRecommendations(data) {
            // 新しい固定フィルタリングエリアのフィルターボタンと統計情報を更新
            updateFilterButtonsAndStats(data);
            
            // 推奨銘柄一覧のみを表示
            const recommendationsHtml = data.recommendations.map(rec => `
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <span style="font-weight: bold; font-size: 1.3rem; color: #1f2937;">${rec.symbol}</span>
                            <span style="color: #6b7280; margin-left: 10px; font-size: 1rem;">${rec.name}</span>
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span style="padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; 
                                background: ${rec.recommendation === 'BUY' ? '#dcfdf7; color: #047857' : 
                                rec.recommendation === 'HOLD' ? '#fef3c7; color: #d97706' : '#fee2e2; color: #dc2626'};">
                                ${rec.recommendation}
                            </span>
                            <div style="text-align: center;">
                                <div style="font-size: 0.8rem; color: #6b7280;">売却予測</div>
                                <div style="font-weight: 600; color: #059669;">${rec.sell_prediction ? rec.sell_prediction.predicted_days + '日後' : '-'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">株価</div>
                            <div style="font-weight: 600;">¥${rec.price ? rec.price.toLocaleString() : '-'}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">信頼度</div>
                            <div style="font-weight: 600;">${Math.round((rec.confidence || 0) * 100)}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">カテゴリ</div>
                            <div style="font-weight: 600; color: #059669;">${rec.category || '-'}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">戦略</div>
                            <div style="font-weight: 600; color: #3b82f6;">${rec.sell_prediction ? rec.sell_prediction.strategy : '-'}</div>
                        </div>
                    </div>
                    
                    <div style="padding: 15px; background: #f8fafc; border-radius: 8px; margin-bottom: 15px;">
                        <h4 style="margin: 0 0 10px 0; color: #1f2937; font-size: 1rem;">💡 売買タイミング予測</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 10px;">
                            <div style="background: #ecfdf5; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981;">
                                <div style="font-size: 0.8rem; color: #065f46; font-weight: 600;">📈 購入タイミング</div>
                                <div style="font-size: 0.9rem; color: #047857; font-weight: bold; margin-top: 5px;">${rec.buy_timing_safe || rec.buy_timing || '分析中'}</div>
                            </div>
                            <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                                <div style="font-size: 0.8rem; color: #92400e; font-weight: 600;">📉 売却タイミング</div>
                                <div style="font-size: 0.9rem; color: #d97706; font-weight: bold; margin-top: 5px;">${rec.sell_timing_safe || rec.sell_timing || '分析中'}</div>
                            </div>
                        </div>
                        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <div style="font-size: 0.85rem; color: #1e40af;"><strong>投資戦略:</strong> ${rec.action || '情報なし'}</div>
                        </div>
                        
                        <!-- ポートフォリオ追加ボタンエリア -->
                        <div style="margin-top: 15px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 15px;">
                            <button class="add-to-portfolio-btn" 
                                    data-symbol="${rec.symbol}" 
                                    data-name="${rec.name}" 
                                    data-price="${rec.price || 0}"
                                    data-buy-timing="${(rec.buy_timing || '').replace(/\"/g, '&quot;')}"
                                    data-sell-timing="${(rec.sell_timing || '').replace(/\"/g, '&quot;')}" 
                                    style="background: #10b981; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);"
                                    onmouseover="this.style.background='#059669'; this.style.transform='translateY(-1px)'"
                                    onmouseout="this.style.background='#10b981'; this.style.transform='translateY(0px)'">
                                📝 ポートフォリオに追加
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
            
            // 推奨銘柄一覧のみを更新
            const recommendationsList = document.getElementById('recommendationsList');
            if (recommendationsList) {
                recommendationsList.innerHTML = recommendationsHtml;
            }
        }

        // フィルターボタンと統計情報を更新
        function updateFilterButtonsAndStats(data) {
            const resultCount = document.getElementById('resultCount');
            const filterButtons = document.getElementById('filterButtons');
            const quickStats = document.getElementById('quickStats');
            
            if (resultCount) {
                resultCount.textContent = `${data.filtered_count}件表示中`;
            }
            
            if (filterButtons) {
                filterButtons.innerHTML = `
                    <button onclick="filterRecommendations('')" data-filter="all" class="filter-btn" 
                            style="padding: 12px; border: 2px solid #667eea; background: #667eea; color: white; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        📈 全て<br><span style="font-size: 1.1rem; margin-top: 4px;">${data.total_count}</span>
                    </button>
                    <button onclick="filterRecommendations('high')" data-filter="high-confidence" class="filter-btn"
                            style="padding: 12px; border: 2px solid #059669; background: transparent; color: #059669; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        ⭐ 高信頼度<br><span style="font-size: 1.1rem; margin-top: 4px;">${data.high_confidence_count}</span>
                    </button>
                    <button onclick="filterRecommendations('buy')" data-filter="buy-recommendation" class="filter-btn"
                            style="padding: 12px; border: 2px solid #dc2626; background: transparent; color: #dc2626; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        🚀 買い推奨<br><span style="font-size: 1.1rem; margin-top: 4px;">${data.buy_count}</span>
                    </button>
                    <button onclick="filterRecommendations('growth')" data-filter="growth-stocks" class="filter-btn"
                            style="padding: 12px; border: 2px solid #3b82f6; background: transparent; color: #3b82f6; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        📈 成長株<br><span style="font-size: 1.1rem; margin-top: 4px;">${data.growth_count}</span>
                    </button>
                `;
            }
            
            if (quickStats) {
                quickStats.innerHTML = `
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #059669;">${data.buy_count}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">買い</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #d97706;">${data.hold_count || 0}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">保留</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #dc2626;">${data.sell_count || 0}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">売り</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">${data.high_confidence_count}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">高信頼</div>
                    </div>
                `;
            }
        }
        
        // 推奨銘柄一覧表示（フィルタリング固定版）
        function displayRecommendationsList(recommendations) {
            if (!recommendations || recommendations.length === 0) {
                console.log('表示する推奨銘柄がありません');
                return;
            }
            
            console.log('推奨銘柄一覧を表示中:', recommendations.length, '件');
            
            // 統計情報計算
            const highConfidenceCount = recommendations.filter(r => r.confidence >= 0.8).length;
            const buyCount = recommendations.filter(r => r.recommendation === 'BUY').length;
            const sellCount = recommendations.filter(r => r.recommendation === 'SELL').length;
            const holdCount = recommendations.filter(r => r.recommendation === 'HOLD').length;
            const growthCount = recommendations.filter(r => r.category === '成長株').length;
            
            // 固定フィルタリングエリアを表示・更新
            const filterButtonsArea = document.getElementById('filterButtonsArea');
            const resultCountSpan = document.getElementById('resultCount');
            const filterButtons = document.getElementById('filterButtons');
            const quickStats = document.getElementById('quickStats');
            
            if (filterButtonsArea && resultCountSpan && filterButtons && quickStats) {
                filterButtonsArea.style.display = 'block';
                resultCountSpan.textContent = `${recommendations.length}件表示中`;
                
                // フィルタリングボタンを固定エリアに配置
                filterButtons.innerHTML = `
                    <button onclick="filterRecommendations('')" data-filter="all" class="filter-btn" 
                            style="padding: 12px; border: 2px solid #667eea; background: #667eea; color: white; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        📈 全て<br><span style="font-size: 1.1rem; margin-top: 4px;">${recommendations.length}</span>
                    </button>
                    <button onclick="filterRecommendations('high')" data-filter="high-confidence" class="filter-btn"
                            style="padding: 12px; border: 2px solid #059669; background: transparent; color: #059669; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        ⭐ 高信頼度<br><span style="font-size: 1.1rem; margin-top: 4px;">${highConfidenceCount}</span>
                    </button>
                    <button onclick="filterRecommendations('buy')" data-filter="buy-recommendation" class="filter-btn"
                            style="padding: 12px; border: 2px solid #dc2626; background: transparent; color: #dc2626; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        🚀 買い推奨<br><span style="font-size: 1.1rem; margin-top: 4px;">${buyCount}</span>
                    </button>
                    <button onclick="filterRecommendations('growth')" data-filter="growth-stocks" class="filter-btn"
                            style="padding: 12px; border: 2px solid #3b82f6; background: transparent; color: #3b82f6; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.2s;">
                        📈 成長株<br><span style="font-size: 1.1rem; margin-top: 4px;">${growthCount}</span>
                    </button>
                `;
                
                // 統計情報を固定エリアに配置
                quickStats.innerHTML = `
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #059669;">${buyCount}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">買い</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #d97706;">${holdCount}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">保留</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #dc2626;">${sellCount}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">売り</div>
                    </div>
                    <div style="text-align: center; padding: 8px;">
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">${highConfidenceCount}</div>
                        <div style="font-size: 0.7rem; color: #6b7280;">高信頼</div>
                    </div>
                `;
            }
            
            const recommendationsHtml = recommendations.map(rec => `
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <div>
                            <span style="font-weight: bold; font-size: 1.3rem; color: #1f2937;">${rec.symbol}</span>
                            <span style="color: #6b7280; margin-left: 10px; font-size: 1rem;">${rec.name}</span>
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span style="padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; 
                                background: ${rec.recommendation === 'BUY' ? '#dcfdf7; color: #047857' : 
                                rec.recommendation === 'HOLD' ? '#fef3c7; color: #d97706' : '#fee2e2; color: #dc2626'};">
                                ${rec.recommendation}
                            </span>
                            <div style="text-align: center;">
                                <div style="font-size: 0.8rem; color: #6b7280;">売却予測</div>
                                <div style="font-weight: 600; color: #059669;">${rec.sell_prediction ? rec.sell_prediction.predicted_days + '日後' : '-'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">株価</div>
                            <div style="font-weight: 600;">¥${rec.price ? rec.price.toLocaleString() : '-'}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">信頼度</div>
                            <div style="font-weight: 600;">${Math.round((rec.confidence || 0) * 100)}%</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">カテゴリ</div>
                            <div style="font-weight: 600; color: #059669;">${rec.category || '-'}</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 0.8rem; color: #6b7280;">戦略</div>
                            <div style="font-weight: 600; color: #3b82f6;">${rec.sell_prediction ? rec.sell_prediction.strategy : '-'}</div>
                        </div>
                    </div>
                    
                    <div style="padding: 15px; background: #f8fafc; border-radius: 8px; margin-bottom: 15px;">
                        <h4 style="margin: 0 0 10px 0; color: #1f2937; font-size: 1rem;">💡 売買タイミング予測</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 10px;">
                            <div style="background: #ecfdf5; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981;">
                                <div style="font-size: 0.8rem; color: #065f46; font-weight: 600;">📈 購入タイミング</div>
                                <div style="font-size: 0.9rem; color: #047857; font-weight: bold; margin-top: 5px;">${rec.buy_timing_safe || rec.buy_timing || '分析中'}</div>
                            </div>
                            <div style="background: #fef3c7; padding: 12px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                                <div style="font-size: 0.8rem; color: #92400e; font-weight: 600;">📉 売却タイミング</div>
                                <div style="font-size: 0.9rem; color: #d97706; font-weight: bold; margin-top: 5px;">${rec.sell_timing_safe || rec.sell_timing || '分析中'}</div>
                            </div>
                        </div>
                        <div style="background: #eff6ff; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            <div style="font-size: 0.85rem; color: #1e40af;"><strong>投資戦略:</strong> ${rec.action || '情報なし'}</div>
                        </div>
                        
                        <!-- ポートフォリオ追加ボタンエリア -->
                        <div style="margin-top: 15px; text-align: center; border-top: 1px solid #e5e7eb; padding-top: 15px;">
                            <button class="add-to-portfolio-btn" 
                                    data-symbol="${rec.symbol}" 
                                    data-name="${rec.name}" 
                                    data-price="${rec.price || 0}"
                                    data-buy-timing="${(rec.buy_timing || '').replace(/\"/g, '&quot;')}"
                                    data-sell-timing="${(rec.sell_timing || '').replace(/\"/g, '&quot;')}" 
                                    style="background: #10b981; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: all 0.2s; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);"
                                    onmouseover="this.style.background='#059669'; this.style.transform='translateY(-1px)'"
                                    onmouseout="this.style.background='#10b981'; this.style.transform='translateY(0px)'">
                                📝 ポートフォリオに追加
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
            
            // 推奨銘柄一覧のみを表示（フィルタリングボタンは固定エリアに分離済み）
            document.getElementById('recommendationsList').innerHTML = recommendationsHtml;
            document.getElementById('recommendationsContainer').style.display = 'block';
            console.log('フィルタリング固定版推奨銘柄一覧HTMLを設定完了');
        }
        
        // 統計数字を動的更新
        function updateStatNumbers(stats) {
            if (!stats) return;
            
            document.getElementById('totalStocks').textContent = stats.total_count;
            document.getElementById('highConfidenceStocks').textContent = stats.high_confidence_count;
            document.getElementById('buyStocks').textContent = stats.buy_count;
            document.getElementById('growthStocks').textContent = stats.growth_count;
        }
        
        // フィルターボタンのアクティブ状態切り替え
        function setActiveFilter(activeFilter) {
            const buttons = document.querySelectorAll('.filter-btn');
            buttons.forEach(btn => {
                const filter = btn.getAttribute('onclick').match(/'([^']*)'/)[1];
                if (filter === activeFilter) {
                    // アクティブ状態
                    const borderColor = btn.style.borderColor;
                    btn.style.background = borderColor;
                    btn.style.color = 'white';
                } else {
                    // 非アクティブ状態
                    const borderColor = btn.style.borderColor;
                    btn.style.background = 'transparent';
                    btn.style.color = borderColor;
                }
            });
        }
        
        // インライン進捗インジケーター表示
        function showLoadingProgress(message = 'データを読み込み中...', progress = 0) {
            const progressIndicator = document.getElementById('progressIndicator');
            const progressMessage = document.getElementById('progressMessage');
            const progressBarInline = document.getElementById('progressBarInline');
            const progressPercentage = document.getElementById('progressPercentage');
            
            if (progressIndicator && progressMessage && progressBarInline && progressPercentage) {
                progressIndicator.style.display = 'block';
                progressMessage.textContent = message;
                progressBarInline.style.width = `${progress}%`;
                progressPercentage.textContent = `${progress}% 完了`;
                
                console.log(`進捗表示: ${message} (${progress}%)`);
            }
        }
        
        // インライン進捗更新
        function updateProgress(message, progress) {
            const progressMessage = document.getElementById('progressMessage');
            const progressBarInline = document.getElementById('progressBarInline');
            const progressPercentage = document.getElementById('progressPercentage');
            
            if (progressMessage) progressMessage.textContent = message;
            if (progressBarInline) progressBarInline.style.width = progress + '%';
            if (progressPercentage) progressPercentage.textContent = progress + '% 完了';
            
            console.log(`進捗更新: ${message} (${progress}%)`);
        }
        
        // インライン進捗非表示
        function hideLoading() {
            const progressIndicator = document.getElementById('progressIndicator');
            if (progressIndicator) {
                progressIndicator.style.opacity = '0';
                progressIndicator.style.transition = 'opacity 0.3s ease';
                setTimeout(() => {
                    progressIndicator.style.display = 'none';
                    progressIndicator.style.opacity = '1';
                }, 300);
                console.log('進捗インジケーターを非表示にしました');
            }
        }

        // 初回データ取得（プログレス表示対応）
        async function initializeData() {
            try {
                showLoadingProgress('実際の株価データを取得中...', 10);
                console.log('=== 初期データ取得開始 ===');
                
                updateProgress('Yahoo Finance APIに接続中...', 25);
                await new Promise(resolve => setTimeout(resolve, 500)); // UI表示待ち
                
                const response = await fetch('/api/recommendations');
                updateProgress('株価データを解析中...', 50);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                updateProgress('推奨銘柄を計算中...', 75);
                console.log('API応答データ:', data);
                
                allRecommendations = data.recommendations || [];
                console.log('取得した銘柄数:', allRecommendations.length);
                
                updateProgress('画面を構築中...', 90);
                await new Promise(resolve => setTimeout(resolve, 300));
                
                // 推奨銘柄一覧を表示（統計込み）
                displayRecommendationsList(allRecommendations);
                
                // 推奨銘柄セクションを表示
                const recommendationsContainer = document.getElementById('recommendationsContainer');
                if (recommendationsContainer) {
                    recommendationsContainer.style.display = 'block';
                }
                
                updateProgress('完了しました！', 100);
                await new Promise(resolve => setTimeout(resolve, 500));
                hideLoading();
                
                console.log('=== 推奨銘柄一覧表示完了 ===');
                
            } catch (error) {
                console.error('初期データ取得エラー:', error);
                updateProgress('エラーが発生しました', 0);
                setTimeout(hideLoading, 2000);
                
                // エラー時の表示
                document.getElementById('recommendationsList').innerHTML = `
                    <div style="background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0;">
                        <h3 style="color: #dc2626; margin: 0 0 10px 0;">⚠️ データ取得エラー</h3>
                        <p style="color: #7f1d1d; margin: 0;">サーバーに接続できませんでした。しばらく待ってから再度お試しください。</p>
                        <button onclick="window.location.reload()" style="margin-top: 15px; padding: 10px 20px; background: #dc2626; color: white; border: none; border-radius: 6px; cursor: pointer;">
                            🔄 再読み込み
                        </button>
                    </div>
                `;
            }
        }

        // ポートフォリオに銘柄を追加する関数
        async function addToPortfolio(symbol, name, price, buyTiming, sellTiming) {
            try {
                showLoadingProgress('ポートフォリオに追加中...', 30);
                
                const portfolioData = {
                    symbol: symbol,
                    name: name,
                    price: price,
                    buy_timing: buyTiming,
                    sell_timing: sellTiming,
                    added_date: new Date().toISOString()
                };
                
                updateProgress('サーバーに送信中...', 70);
                
                const response = await fetch('/api/portfolio/add', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(portfolioData)
                });
                
                updateProgress('追加完了', 100);
                
                if (response.ok) {
                    const result = await response.json();
                    
                    // 成功通知
                    showNotification(\`✅ \${name} (\${symbol}) をポートフォリオに追加しました！\`, 'success');
                    
                    // ボタンを一時的に無効化
                    const button = event.target;
                    const originalText = button.textContent;
                    button.textContent = '✅ 追加済み';
                    button.style.background = '#059669';
                    button.disabled = true;
                    
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.style.background = '#10b981';
                        button.disabled = false;
                    }, 2000);
                } else {
                    throw new Error('追加に失敗しました');
                }
                
                hideLoading();
                
            } catch (error) {
                console.error('ポートフォリオ追加エラー:', error);
                showNotification('❌ ポートフォリオへの追加に失敗しました', 'error');
                hideLoading();
            }
        }

        // 通知表示関数
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.style.cssText = \`
                position: fixed; top: 20px; right: 20px; z-index: 10000;
                padding: 15px 20px; border-radius: 8px; font-weight: 600;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                background: \${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
                color: white;
            \`;
            notification.textContent = message;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }

        // 初期化処理（強化版）
        document.addEventListener('DOMContentLoaded', function() {
            console.log('=== DOM準備完了 ===');
            
            // デフォルトで「総銘柄数」をアクティブに
            const allBtn = document.querySelector('[data-filter="all"]');
            if (allBtn) {
                allBtn.classList.add('active');
                console.log('「総銘柄数」ボタンをアクティブに設定');
            }
            
            // 少し遅らせて初期データを取得（DOM完全準備待ち）
            setTimeout(() => {
                console.log('=== 初期データ取得実行 ===');
                initializeData();
            }, 100);
        });

        // バックアップ初期化（ページ読み込み完了後）
        window.addEventListener('load', function() {
            console.log('=== ページ読み込み完了 ===');
            
            // まだデータが読み込まれていない場合の保険
            setTimeout(() => {
                const totalElement = document.getElementById('totalStocks');
                if (totalElement && totalElement.textContent === '35') {
                    console.log('保険として再度データ取得実行');
                    initializeData();
                }
            }, 500);
        });

        // ポートフォリオボタンのイベントハンドラ（安全な実装）
        document.addEventListener('click', function(event) {
            if (event.target.classList.contains('add-to-portfolio-btn')) {
                const button = event.target;
                const symbol = button.dataset.symbol;
                const name = button.dataset.name;
                const price = parseFloat(button.dataset.price);
                const buyTiming = button.dataset.buyTiming;
                const sellTiming = button.dataset.sellTiming;
                
                addToPortfolio(symbol, name, price, buyTiming, sellTiming);
            }
        });
    </script>
</body>
</html>
        """

    @staticmethod
    def get_recommendations_template() -> str:
        """推奨銘柄フィルタリングテンプレート"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #f8fafc; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #1f2937; font-size: 2.5rem; margin-bottom: 10px; }
        .nav { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }
        .nav-btn { padding: 10px 20px; background: #e5e7eb; border: none; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
        .nav-btn:hover { background: #3b82f6; color: white; }
        
        .filters { display: flex; gap: 15px; margin-bottom: 30px; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .filter-group { display: flex; flex-direction: column; }
        .filter-group label { font-weight: 600; margin-bottom: 5px; color: #374151; }
        .filter-btn { padding: 8px 16px; border: 2px solid #e5e7eb; background: white; border-radius: 6px; cursor: pointer; transition: all 0.2s; }
        .filter-btn:hover { border-color: #3b82f6; }
        .filter-btn.active { background: #3b82f6; color: white; border-color: #3b82f6; }
        
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; }
        .stat-number { font-size: 2rem; font-weight: bold; margin-bottom: 5px; }
        .stat-label { font-size: 0.9rem; opacity: 0.9; }
        
        .recommendations { display: grid; gap: 15px; }
        .rec-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .rec-card:hover { transform: translateY(-2px); }
        .rec-header { display: flex; justify-content: between; align-items: center; margin-bottom: 15px; }
        .rec-symbol { font-weight: bold; font-size: 1.2rem; color: #1f2937; }
        .rec-name { color: #6b7280; margin-left: 10px; }
        .rec-recommendation { padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; }
        .rec-recommendation.buy { background: #dcfdf7; color: #047857; }
        .rec-recommendation.hold { background: #fef3c7; color: #d97706; }
        .rec-recommendation.sell { background: #fee2e2; color: #dc2626; }
        .rec-details { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
        .rec-detail { text-align: center; }
        .rec-detail-label { font-size: 0.8rem; color: #6b7280; margin-bottom: 2px; }
        .rec-detail-value { font-weight: 600; color: #1f2937; }
        .confidence-high { color: #059669; }
        .confidence-medium { color: #d97706; }
        .confidence-low { color: #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 推奨銘柄フィルタリング</h1>
            <p>35銘柄から最適な投資判断をサポート</p>
        </div>
        
        <div class="nav">
            <button class="nav-btn" onclick="location.href='/'">ホーム</button>
            <button class="nav-btn" onclick="location.href='/swing-trade'">スイング</button>
            <button class="nav-btn" onclick="location.href='/scheduler'">スケジューラー</button>
        </div>
        
        <div class="filters">
            <div class="filter-group">
                <label>カテゴリフィルタ</label>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <button class="filter-btn" data-filter="category" data-value="">全て</button>
                    <button class="filter-btn" data-filter="category" data-value="大型株">大型株</button>
                    <button class="filter-btn" data-filter="category" data-value="中型株">中型株</button>
                    <button class="filter-btn" data-filter="category" data-value="高配当株">高配当株</button>
                    <button class="filter-btn" data-filter="category" data-value="成長株">成長株</button>
                    <button class="filter-btn" data-filter="category" data-value="小型株">小型株</button>
                </div>
            </div>
            
            <div class="filter-group">
                <label>信頼度フィルタ</label>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <button class="filter-btn" data-filter="confidence" data-value="">全て</button>
                    <button class="filter-btn" data-filter="confidence" data-value="high">高信頼度</button>
                    <button class="filter-btn" data-filter="confidence" data-value="medium">中信頼度</button>
                    <button class="filter-btn" data-filter="confidence" data-value="low">低信頼度</button>
                </div>
            </div>
            
            <div class="filter-group">
                <label>推奨フィルタ</label>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <button class="filter-btn" data-filter="recommendation" data-value="">全て</button>
                    <button class="filter-btn" data-filter="recommendation" data-value="buy">買い推奨</button>
                    <button class="filter-btn" data-filter="recommendation" data-value="hold">保有</button>
                    <button class="filter-btn" data-filter="recommendation" data-value="sell">売却</button>
                </div>
            </div>
        </div>
        
        <div class="stats" id="statsContainer">
            <!-- 統計情報がJavaScriptで動的に表示される -->
        </div>
        
        <div class="recommendations" id="recommendationsContainer">
            <!-- 推奨銘柄がJavaScriptで動的に表示される -->
        </div>
    </div>
    
    <script>
        let currentFilters = {
            category: '',
            confidence: '',
            recommendation: ''
        };
        
        // フィルターボタンクリック処理
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const filterType = this.dataset.filter;
                const filterValue = this.dataset.value;
                
                // 同グループの他のボタンを非アクティブに
                document.querySelectorAll(`[data-filter="${filterType}"]`).forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                // フィルターを更新
                currentFilters[filterType] = filterValue;
                loadRecommendations();
            });
        });
        
        // 推奨銘柄を取得・表示
        async function loadRecommendations() {
            try {
                const params = new URLSearchParams();
                if (currentFilters.category) params.append('category', currentFilters.category);
                if (currentFilters.confidence) params.append('confidence', currentFilters.confidence);
                if (currentFilters.recommendation) params.append('recommendation', currentFilters.recommendation);
                
                const response = await fetch(`/api/recommendations?${params.toString()}`);
                const data = await response.json();
                
                displayStats(data);
                displayRecommendations(data.recommendations);
            } catch (error) {
                console.error('データ取得エラー:', error);
            }
        }
        
        // 統計情報表示
        function displayStats(data) {
            const statsHtml = `
                <div class="stat-card">
                    <div class="stat-number">${data.total_count}</div>
                    <div class="stat-label">総銘柄数</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.filtered_count}</div>
                    <div class="stat-label">フィルタ後</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.high_confidence_count}</div>
                    <div class="stat-label">高信頼度</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">${data.buy_count}</div>
                    <div class="stat-label">買い推奨</div>
                </div>
            `;
            document.getElementById('statsContainer').innerHTML = statsHtml;
        }
        
        // 推奨銘柄表示
        function displayRecommendations(recommendations) {
            const recsHtml = recommendations.map(rec => `
                <div class="rec-card">
                    <div class="rec-header">
                        <div>
                            <span class="rec-symbol">${rec.symbol}</span>
                            <span class="rec-name">${rec.name}</span>
                        </div>
                        <span class="rec-recommendation ${rec.recommendation.toLowerCase()}">${rec.recommendation}</span>
                    </div>
                    <div class="rec-details">
                        <div class="rec-detail">
                            <div class="rec-detail-label">信頼度</div>
                            <div class="rec-detail-value confidence-${rec.confidence > 0.8 ? 'high' : rec.confidence > 0.6 ? 'medium' : 'low'}">${(rec.confidence * 100).toFixed(1)}%</div>
                        </div>
                        <div class="rec-detail">
                            <div class="rec-detail-label">価格</div>
                            <div class="rec-detail-value">${rec.price}円</div>
                        </div>
                        <div class="rec-detail">
                            <div class="rec-detail-label">変動</div>
                            <div class="rec-detail-value" style="color: ${rec.change > 0 ? '#059669' : '#dc2626'}">${rec.change > 0 ? '+' : ''}${rec.change}%</div>
                        </div>
                        <div class="rec-detail">
                            <div class="rec-detail-label">カテゴリ</div>
                            <div class="rec-detail-value">${rec.category}</div>
                        </div>
                    </div>
                </div>
            `).join('');
            document.getElementById('recommendationsContainer').innerHTML = recsHtml;
        }
        
        // 初期表示
        document.addEventListener('DOMContentLoaded', function() {
            // デフォルトで全てフィルタをアクティブに
            document.querySelectorAll('[data-value=""]').forEach(btn => btn.classList.add('active'));
            loadRecommendations();
        });
    </script>
</body>
</html>
        """

    @staticmethod
    def get_portfolio_template() -> str:
        """ポートフォリオ追跡テンプレート（改善版）"""
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #f8fafc; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; color: #1f2937; }
        .nav { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; }
        .nav-btn { padding: 10px 20px; background: #3b82f6; border: none; border-radius: 8px; color: white; cursor: pointer; transition: all 0.2s; }
        .nav-btn:hover { background: #2563eb; }
        
        .portfolio-grid { display: grid; gap: 20px; margin-bottom: 30px; }
        .portfolio-card { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); transition: transform 0.2s; }
        .portfolio-card:hover { transform: translateY(-2px); }
        .portfolio-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .portfolio-symbol { font-size: 1.3rem; font-weight: bold; color: #1f2937; }
        .portfolio-name { color: #6b7280; margin-left: 10px; }
        .sell-date { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
        .sell-today { background: #fee2e2; color: #dc2626; }
        .sell-soon { background: #fef3c7; color: #d97706; }
        .sell-later { background: #dcfdf7; color: #047857; }
        
        .portfolio-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; margin-bottom: 15px; }
        .portfolio-metric { text-align: center; }
        .metric-label { font-size: 0.8rem; color: #6b7280; margin-bottom: 3px; }
        .metric-value { font-weight: bold; color: #1f2937; }
        .profit-positive { color: #059669; }
        .profit-negative { color: #dc2626; }
        
        .sell-recommendation { background: #f8fafc; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 15px; }
        .recommendation-title { font-weight: 600; color: #1f2937; margin-bottom: 5px; }
        .recommendation-text { color: #6b7280; font-size: 0.9rem; }
        
        .action-buttons { display: flex; gap: 10px; justify-content: flex-end; }
        .edit-btn { background: #f59e0b; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }
        .edit-btn:hover { background: #d97706; }
        .delete-btn { background: #dc2626; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }
        .delete-btn:hover { background: #b91c1c; }
        
        .calendar-section { background: white; border-radius: 12px; padding: 25px; margin-top: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .calendar-title { font-size: 1.5rem; margin-bottom: 20px; text-align: center; color: #1f2937; }
        .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; }
        .calendar-day { padding: 10px; text-align: center; border-radius: 8px; border: 1px solid #e5e7eb; position: relative; cursor: pointer; transition: all 0.2s; }
        .calendar-day.has-sell { background: #fee2e2; border-color: #dc2626; }
        .calendar-day.has-buy { background: #dcfdf7; border-color: #059669; }
        .calendar-day:hover { transform: scale(1.05); }
        
        .stock-label { position: absolute; bottom: 2px; left: 2px; right: 2px; font-size: 0.7rem; background: rgba(0,0,0,0.7); color: white; padding: 2px; border-radius: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .tooltip { position: absolute; background: #1f2937; color: white; padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; white-space: nowrap; z-index: 1000; opacity: 0; pointer-events: none; transition: opacity 0.2s; }
        
        .empty-state { text-align: center; padding: 40px 20px; color: #6b7280; }
        .empty-state h3 { margin-bottom: 10px; }
        .empty-state p { margin-bottom: 20px; }
        .add-stock-btn { background: #10b981; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 1rem; transition: all 0.2s; }
        .add-stock-btn:hover { background: #059669; }
        
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }
        .modal-content { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 12px; padding: 25px; max-width: 500px; width: 90%; }
        .modal-title { font-size: 1.3rem; margin-bottom: 20px; color: #1f2937; }
        .form-group { margin-bottom: 15px; }
        .form-label { display: block; margin-bottom: 5px; color: #374151; font-weight: 500; }
        .form-input { width: 100%; padding: 10px; border: 2px solid #e5e7eb; border-radius: 6px; font-size: 1rem; }
        .form-input:focus { border-color: #3b82f6; outline: none; }
        .modal-buttons { display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px; }
        .modal-btn-primary { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
        .modal-btn-secondary { background: #6b7280; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💼 ポートフォリオ追跡</h1>
            <p>購入した銘柄の売却タイミング管理</p>
        </div>
        
        <div class="nav">
            <button class="nav-btn" onclick="location.href='/'">🏠 ホーム</button>
            <button class="nav-btn" onclick="location.reload()">🔄 データ更新</button>
        </div>
        
        <div id="portfolioContainer" class="portfolio-grid">
            <!-- ポートフォリオ銘柄がJavaScriptで動的に表示される -->
        </div>
        
        <!-- 空の状態 -->
        <div id="emptyState" class="empty-state" style="display: none;">
            <h3>📦 ポートフォリオが空です</h3>
            <p>推奨銘柄から「📝 ポートフォリオに追加」ボタンで銘柄を追加してください。</p>
            <button class="add-stock-btn" onclick="location.href='/'">推奨銘柄を見る</button>
        </div>
        
        <div class="calendar-section">
            <div class="calendar-title">📅 売却予定カレンダー</div>
            <div id="calendarContainer">
                <!-- カレンダーがJavaScriptで動的に表示される -->
            </div>
        </div>
    </div>
    
    <!-- 売却タイミング編集モーダル -->
    <div id="editModal" class="modal">
        <div class="modal-content">
            <div class="modal-title">🔧 売却タイミング編集</div>
            <div class="form-group">
                <label class="form-label">銘柄</label>
                <input type="text" id="editSymbol" class="form-input" readonly>
            </div>
            <div class="form-group">
                <label class="form-label">新しい売却タイミング</label>
                <input type="text" id="editSellTiming" class="form-input" placeholder="例：12/25 14:00">
            </div>
            <div class="form-group">
                <label class="form-label">売却理由（任意）</label>
                <input type="text" id="editReason" class="form-input" placeholder="例：市況変化により早期売却">
            </div>
            <div class="modal-buttons">
                <button class="modal-btn-secondary" onclick="closeEditModal()">キャンセル</button>
                <button class="modal-btn-primary" onclick="saveEditChanges()">変更を保存</button>
            </div>
        </div>
    </div>
    
    <script>
        let portfolioStocks = [];
        let currentEditingId = null;
        
        // APIからポートフォリオデータを読み込み（実際のAPI連携）
        async function loadPortfolioData() {
            try {
                const response = await fetch('/api/portfolio');
                if (response.ok) {
                    const data = await response.json();
                    portfolioStocks = data.portfolio || [];
                    console.log('ポートフォリオデータ取得成功:', portfolioStocks);
                } else {
                    console.error('ポートフォリオデータ取得エラー:', response.status);
                    portfolioStocks = [];
                }
            } catch (error) {
                console.error('API通信エラー:', error);
                portfolioStocks = [];
            }
        }
        
        // ポートフォリオ表示（改善版）
        function displayPortfolio() {
            const portfolioContainer = document.getElementById('portfolioContainer');
            const emptyState = document.getElementById('emptyState');
            
            if (portfolioStocks.length === 0) {
                portfolioContainer.innerHTML = '';
                emptyState.style.display = 'block';
                return;
            }
            
            emptyState.style.display = 'none';
            
            const portfolioHtml = portfolioStocks.map(stock => {
                const currentPrice = stock.price || stock.currentPrice || 0;
                const purchasePrice = stock.purchasePrice || currentPrice;
                const profit = purchasePrice > 0 ? ((currentPrice - purchasePrice) / purchasePrice * 100).toFixed(2) : 0;
                const profitClass = profit >= 0 ? 'profit-positive' : 'profit-negative';
                
                // 売却タイミングから日数を計算
                const sellTiming = stock.sell_timing || stock.sellDate || '';
                let daysToSell = 0;
                let sellTimingDisplay = '未設定';
                
                if (sellTiming) {
                    const sellDate = new Date(sellTiming.split(' ')[0].replace(/(\d{2})\/(\d{2})/, '2024-$1-$2'));
                    if (!isNaN(sellDate)) {
                        daysToSell = Math.ceil((sellDate - new Date()) / (1000 * 60 * 60 * 24));
                        sellTimingDisplay = sellTiming;
                    }
                }
                
                let sellDateClass = 'sell-later';
                if (daysToSell <= 0) sellDateClass = 'sell-today';
                else if (daysToSell <= 3) sellDateClass = 'sell-soon';
                
                const sellDateText = daysToSell <= 0 ? '今日売却' : daysToSell > 0 ? `${daysToSell}日後売却` : '期限不明';
                
                return `
                    <div class="portfolio-card">
                        <div class="portfolio-header">
                            <div>
                                <span class="portfolio-symbol">${stock.symbol}</span>
                                <span class="portfolio-name">${stock.name}</span>
                            </div>
                            <span class="sell-date ${sellDateClass}">
                                ${sellDateText}
                            </span>
                        </div>
                        
                        <div class="portfolio-metrics">
                            <div class="portfolio-metric">
                                <div class="metric-label">追加価格</div>
                                <div class="metric-value">¥${purchasePrice.toLocaleString()}</div>
                            </div>
                            <div class="portfolio-metric">
                                <div class="metric-label">現在価格</div>
                                <div class="metric-value">¥${currentPrice.toLocaleString()}</div>
                            </div>
                            <div class="portfolio-metric">
                                <div class="metric-label">予想損益</div>
                                <div class="metric-value ${profitClass}">${profit >= 0 ? '+' : ''}${profit}%</div>
                            </div>
                            <div class="portfolio-metric">
                                <div class="metric-label">ステータス</div>
                                <div class="metric-value">${stock.status === 'watching' ? '監視中' : stock.status}</div>
                            </div>
                        </div>
                        
                        <div class="sell-recommendation">
                            <div class="recommendation-title">📈 売却タイミング</div>
                            <div class="recommendation-text">${sellTimingDisplay}</div>
                        </div>
                        
                        <div class="action-buttons">
                            <button class="edit-btn" onclick="openEditModal(${stock.id})">🔧 編集</button>
                            <button class="delete-btn" onclick="deleteStock(${stock.id})">🗑️ 削除</button>
                        </div>
                    </div>
                `;
            }).join('');
            
            portfolioContainer.innerHTML = portfolioHtml;
        }
        
        // カレンダー表示（改善版：銘柄名表示付き）
        function displayCalendar() {
            const today = new Date();
            const currentMonth = today.getMonth();
            const currentYear = today.getFullYear();
            const firstDay = new Date(currentYear, currentMonth, 1).getDay();
            const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
            
            let calendarHtml = '<div class="calendar-grid">';
            
            // 曜日ヘッダー
            const dayNames = ['日', '月', '火', '水', '木', '金', '土'];
            dayNames.forEach(day => {
                calendarHtml += `<div style="font-weight: bold; text-align: center; padding: 10px; background: #f3f4f6; border-radius: 8px;">${day}</div>`;
            });
            
            // 空白セル
            for (let i = 0; i < firstDay; i++) {
                calendarHtml += '<div class="calendar-day"></div>';
            }
            
            // 日付セル
            for (let day = 1; day <= daysInMonth; day++) {
                const dateStr = `${(currentMonth + 1).toString().padStart(2, '0')}/${day.toString().padStart(2, '0')}`;
                
                // この日に売却予定の銘柄を検索
                const sellStocks = portfolioStocks.filter(stock => {
                    const sellTiming = stock.sell_timing || stock.sellDate || '';
                    return sellTiming.includes(dateStr);
                });
                
                let cellClass = 'calendar-day';
                let stockLabels = '';
                let tooltipText = '';
                
                if (sellStocks.length > 0) {
                    cellClass = 'calendar-day has-sell';
                    
                    // 最初の銘柄のみラベル表示（スペース節約）
                    const firstStock = sellStocks[0];
                    stockLabels = `<div class="stock-label">${firstStock.symbol}</div>`;
                    
                    // ツールチップ用のテキスト生成
                    tooltipText = sellStocks.map(s => `${s.symbol} ${s.name}`).join('\\n');
                }
                
                calendarHtml += `
                    <div class="${cellClass}" 
                         onclick="showDayDetails('${dateStr}')"
                         onmouseenter="showTooltip(this, '${tooltipText}')"
                         onmouseleave="hideTooltip()">
                        ${day}
                        ${stockLabels}
                    </div>
                `;
            }
            
            calendarHtml += '</div>';
            document.getElementById('calendarContainer').innerHTML = calendarHtml;
        }
        
        // カレンダーの日付詳細表示
        function showDayDetails(dateStr) {
            const sellStocks = portfolioStocks.filter(stock => {
                const sellTiming = stock.sell_timing || stock.sellDate || '';
                return sellTiming.includes(dateStr);
            });
            
            if (sellStocks.length === 0) return;
            
            const stockList = sellStocks.map(s => `• ${s.symbol} ${s.name}`).join('\\n');
            alert(`📅 ${dateStr} 売却予定\\n\\n${stockList}`);
        }
        
        // ツールチップ表示
        function showTooltip(element, text) {
            if (!text) return;
            
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = text;
            tooltip.style.opacity = '1';
            
            document.body.appendChild(tooltip);
            
            const rect = element.getBoundingClientRect();
            tooltip.style.left = rect.left + 'px';
            tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
        }
        
        // ツールチップ非表示
        function hideTooltip() {
            const tooltip = document.querySelector('.tooltip');
            if (tooltip) {
                tooltip.remove();
            }
        }
        
        // 編集モーダル表示
        function openEditModal(stockId) {
            const stock = portfolioStocks.find(s => s.id === stockId);
            if (!stock) return;
            
            currentEditingId = stockId;
            document.getElementById('editSymbol').value = `${stock.symbol} - ${stock.name}`;
            document.getElementById('editSellTiming').value = stock.sell_timing || '';
            document.getElementById('editReason').value = stock.notes || '';
            document.getElementById('editModal').style.display = 'block';
        }
        
        // 編集モーダル閉じる
        function closeEditModal() {
            document.getElementById('editModal').style.display = 'none';
            currentEditingId = null;
        }
        
        // 編集内容保存
        async function saveEditChanges() {
            if (!currentEditingId) return;
            
            const sellTiming = document.getElementById('editSellTiming').value;
            const reason = document.getElementById('editReason').value;
            
            try {
                const response = await fetch(`/api/portfolio/${currentEditingId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        sell_timing: sellTiming,
                        notes: reason
                    })
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('編集成功:', result.message);
                    
                    // ローカルデータも更新
                    const stock = portfolioStocks.find(s => s.id === currentEditingId);
                    if (stock) {
                        stock.sell_timing = sellTiming;
                        stock.notes = reason;
                    }
                    
                    displayPortfolio();
                    displayCalendar();
                    closeEditModal();
                    
                    // 成功通知
                    showNotification('売却タイミングを更新しました！', 'success');
                } else {
                    const error = await response.json();
                    console.error('編集エラー:', error);
                    showNotification('更新に失敗しました: ' + error.error, 'error');
                }
            } catch (error) {
                console.error('API通信エラー:', error);
                showNotification('通信エラーが発生しました', 'error');
            }
        }
        
        // 銘柄削除
        async function deleteStock(stockId) {
            const stock = portfolioStocks.find(s => s.id === stockId);
            if (!stock) return;
            
            if (!confirm(`${stock.symbol} ${stock.name} をポートフォリオから削除しますか？`)) {
                return;
            }
            
            try {
                const response = await fetch(`/api/portfolio/${stockId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('削除成功:', result.message);
                    
                    // ローカルデータからも削除
                    portfolioStocks = portfolioStocks.filter(s => s.id !== stockId);
                    
                    displayPortfolio();
                    displayCalendar();
                    
                    // 成功通知
                    showNotification('銘柄を削除しました', 'success');
                } else {
                    const error = await response.json();
                    console.error('削除エラー:', error);
                    showNotification('削除に失敗しました: ' + error.error, 'error');
                }
            } catch (error) {
                console.error('API通信エラー:', error);
                showNotification('通信エラーが発生しました', 'error');
            }
        }
        
        // 通知表示
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 8px;
                color: white;
                font-weight: 600;
                z-index: 10000;
                transition: all 0.3s ease;
                background: ${type === 'success' ? '#10b981' : type === 'error' ? '#dc2626' : '#3b82f6'};
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.opacity = '0';
                notification.style.transform = 'translateX(100%)';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
        
        // 初期表示（API連携版）
        document.addEventListener('DOMContentLoaded', async function() {
            await loadPortfolioData();
            displayPortfolio();
            displayCalendar();
        });
    </script>
</body>
</html>
        """

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
            <button class="nav-btn" onclick="location.href='/'">ホーム</button>
            <button class="nav-btn" onclick="location.href='/swing-trade'">スイング</button>
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
