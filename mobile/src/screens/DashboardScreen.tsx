import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  StyleSheet,
  RefreshControl,
  ScrollView,
  Dimensions,
  Alert,
} from 'react-native';
import {
  Surface,
  Text,
  Card,
  Button,
  FAB,
  Portal,
  Modal,
  ActivityIndicator,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useQuery } from 'react-query';
import { useFocusEffect } from '@react-navigation/native';
import { LineChart } from 'react-native-chart-kit';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  withTiming,
  interpolate,
} from 'react-native-reanimated';

// Components
import { KPICard } from '../components/dashboard/KPICard';
import { AnalysisCard } from '../components/analysis/AnalysisCard';
import { AlertCard } from '../components/alerts/AlertCard';
import { MarketOverview } from '../components/market/MarketOverview';
import { ConnectionStatus } from '../components/ui/ConnectionStatus';
import { ErrorMessage } from '../components/ui/ErrorMessage';

// Hooks
import { useWebSocket } from '../hooks/useWebSocket';
import { useNotifications } from '../hooks/useNotifications';
import { useBiometric } from '../hooks/useBiometric';
import { useTheme } from '../hooks/useTheme';

// Services
import { dashboardService } from '../services/dashboardService';

// Types
import type { DashboardData, RealtimeUpdate } from '../types/dashboard';

// Utils
import { formatCurrency, formatPercentage } from '../utils/format';

const { width: screenWidth } = Dimensions.get('window');

export const DashboardScreen: React.FC = () => {
  const { colors } = useTheme();
  const { showNotification } = useNotifications();
  const { isBiometricAvailable, authenticateWithBiometric } = useBiometric();

  // State
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [quickActionVisible, setQuickActionVisible] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');

  // Animations
  const fadeAnim = useSharedValue(0);
  const slideAnim = useSharedValue(100);

  // WebSocket接続
  const {
    data: realtimeData,
    isConnected,
    error: wsError,
    reconnect,
  } = useWebSocket('/ws/dashboard');

  // ダッシュボードデータ取得
  const {
    data: dashboardData,
    isLoading,
    error,
    refetch,
  } = useQuery<DashboardData>(
    ['dashboard', selectedTimeRange],
    () => dashboardService.getDashboardData({ timeRange: selectedTimeRange }),
    {
      refetchInterval: isConnected ? undefined : 30000, // WebSocket接続時は自動リフレッシュ無効
      onSuccess: () => {
        // 成功時のアニメーション
        fadeAnim.value = withTiming(1, { duration: 500 });
        slideAnim.value = withSpring(0);
      },
      onError: (err) => {
        console.error('Dashboard data fetch error:', err);
        showNotification({
          type: 'error',
          title: 'データ取得エラー',
          message: 'ダッシュボードデータの取得に失敗しました',
        });
      },
    }
  );

  // リアルタイムデータ処理
  useEffect(() => {
    if (realtimeData) {
      // リアルタイム更新通知
      showNotification({
        type: 'info',
        title: 'リアルタイム更新',
        message: 'データが更新されました',
        duration: 2000,
      });
    }
  }, [realtimeData, showNotification]);

  // 画面フォーカス時の処理
  useFocusEffect(
    useCallback(() => {
      // 生体認証チェック
      const checkBiometric = async () => {
        if (isBiometricAvailable) {
          try {
            const result = await authenticateWithBiometric();
            if (!result.success) {
              Alert.alert('認証失敗', '生体認証に失敗しました');
            }
          } catch (err) {
            console.warn('Biometric authentication error:', err);
          }
        }
      };

      checkBiometric();

      // データ更新
      refetch();
    }, [isBiometricAvailable, authenticateWithBiometric, refetch])
  );

  // Pull to refresh
  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      await refetch();
    } finally {
      setIsRefreshing(false);
    }
  }, [refetch]);


  // アニメーションスタイル
  const fadeStyle = useAnimatedStyle(() => {
    return {
      opacity: fadeAnim.value,
    };
  });

  const slideStyle = useAnimatedStyle(() => {
    return {
      transform: [
        {
          translateY: interpolate(
            slideAnim.value,
            [0, 100],
            [0, 100]
          ),
        },
      ],
    };
  });

  // チャートデータ準備
  const chartData = dashboardData?.charts?.roiTrend ? {
    labels: dashboardData.charts.roiTrend.labels.slice(-7), // 最新7件
    datasets: [
      {
        data: dashboardData.charts.roiTrend.datasets[0]?.data.slice(-7) || [],
        color: (opacity = 1) => `rgba(37, 99, 235, ${opacity})`,
        strokeWidth: 2,
      },
    ],
  } : null;

  if (isLoading && !dashboardData) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Text style={styles.loadingText}>ダッシュボードを読み込み中...</Text>
        </View>
      </SafeAreaView>
    );
  }

  if (error && !dashboardData) {
    return (
      <SafeAreaView style={styles.container}>
        <ErrorMessage
          message="ダッシュボードデータの取得に失敗しました"
          onRetry={refetch}
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* 接続状態 */}
      <ConnectionStatus
        isConnected={isConnected}
        error={wsError}
        onReconnect={reconnect}
      />

      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={onRefresh}
            colors={[colors.primary]}
            tintColor={colors.primary}
          />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* ヘッダー */}
        <Animated.View style={[styles.header, fadeStyle]}>
          <Text variant="headlineMedium" style={styles.title}>
            ダッシュボード
          </Text>
          <Text variant="bodyMedium" style={styles.subtitle}>
            リアルタイム取引監視
          </Text>
        </Animated.View>

        {/* KPI Cards */}
        <Animated.View style={[styles.kpiContainer, slideStyle]}>
          <View style={styles.kpiGrid}>
            <KPICard
              title="Today's ROI"
              value={dashboardData?.kpis?.roi || 0}
              unit="%"
              format="percentage"
              trend="up"
              realTimeValue={realtimeData?.roi}
              style={styles.kpiCard}
            />
            <KPICard
              title="分析実行数"
              value={dashboardData?.kpis?.trades || 0}
              unit="件"
              format="number"
              trend="up"
              realTimeValue={realtimeData?.trades}
              style={styles.kpiCard}
            />
            <KPICard
              title="ML精度"
              value={dashboardData?.kpis?.accuracy || 0}
              unit="%"
              format="percentage"
              trend="up"
              realTimeValue={realtimeData?.accuracy}
              style={styles.kpiCard}
            />
            <KPICard
              title="ポートフォリオ"
              value={dashboardData?.kpis?.portfolioValue || 0}
              unit=""
              format="currency"
              trend="up"
              realTimeValue={realtimeData?.portfolioValue}
              style={styles.kpiCard}
            />
          </View>
        </Animated.View>

        {/* ROI チャート */}
        {chartData && (
          <Card style={styles.chartCard}>
            <Card.Title title="ROI トレンド" subtitle="過去7日間" />
            <Card.Content>
              <LineChart
                data={chartData}
                width={screenWidth - 60}
                height={220}
                chartConfig={{
                  backgroundColor: colors.surface,
                  backgroundGradientFrom: colors.surface,
                  backgroundGradientTo: colors.surface,
                  decimalPlaces: 1,
                  color: (opacity = 1) => `rgba(37, 99, 235, ${opacity})`,
                  labelColor: (opacity = 1) => `rgba(107, 114, 128, ${opacity})`,
                  style: {
                    borderRadius: 16,
                  },
                  propsForDots: {
                    r: '4',
                    strokeWidth: '2',
                    stroke: colors.primary,
                  },
                }}
                bezier
                style={styles.chart}
              />
            </Card.Content>
          </Card>
        )}

        {/* 市場概要 */}
        <MarketOverview
          data={dashboardData?.market}
          realTimeData={realtimeData?.market}
          style={styles.marketCard}
        />

        {/* 最近の分析 */}
        <Card style={styles.analysesCard}>
          <Card.Title title="最近の分析" subtitle="最新5件" />
          <Card.Content>
            {dashboardData?.recentAnalyses?.slice(0, 5).map((analysis) => (
              <AnalysisCard
                key={analysis.id}
                analysis={analysis}
                style={styles.analysisItem}
              />
            ))}
          </Card.Content>
        </Card>

        {/* アラート */}
        {dashboardData?.alerts && dashboardData.alerts.length > 0 && (
          <Card style={styles.alertsCard}>
            <Card.Title title="アラート" subtitle={`${dashboardData.alerts.length}件`} />
            <Card.Content>
              {dashboardData.alerts.slice(0, 3).map((alert) => (
                <AlertCard
                  key={alert.id}
                  alert={alert}
                  style={styles.alertItem}
                />
              ))}
            </Card.Content>
          </Card>
        )}

        {/* 最下部スペース */}
        <View style={styles.bottomSpace} />
      </ScrollView>

      {/* クイックアクション FAB */}
      <Portal>
        <FAB.Group
          open={quickActionVisible}
          visible
          icon={quickActionVisible ? 'close' : 'plus'}
          actions={[
            {
              icon: 'refresh',
              label: '更新',
              onPress: () => refetch(),
            },
            {
              icon: 'chart-line',
              label: '分析',
              onPress: () => {
                // 分析画面へナビゲーション
              },
            },
          ]}
          onStateChange={({ open }) => setQuickActionVisible(open)}
          fabStyle={{
            backgroundColor: colors.primary,
          }}
        />
      </Portal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 16,
    textAlign: 'center',
  },
  scrollView: {
    flex: 1,
  },
  header: {
    padding: 20,
    paddingBottom: 10,
  },
  title: {
    fontWeight: 'bold',
    color: '#111827',
  },
  subtitle: {
    color: '#6b7280',
    marginTop: 4,
  },
  kpiContainer: {
    paddingHorizontal: 20,
    marginBottom: 20,
  },
  kpiGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  kpiCard: {
    width: '48%',
    marginBottom: 12,
  },
  chartCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 12,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 16,
  },
  marketCard: {
    marginHorizontal: 20,
    marginBottom: 20,
  },
  analysesCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 12,
  },
  analysisItem: {
    marginBottom: 8,
  },
  alertsCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    borderRadius: 12,
  },
  alertItem: {
    marginBottom: 8,
  },
  bottomSpace: {
    height: 100, // FABのためのスペース
  },
});