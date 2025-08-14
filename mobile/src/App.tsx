import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import {
  Provider as PaperProvider,
  DefaultTheme,
  DarkTheme,
  configureFonts,
} from 'react-native-paper';
import { StatusBar } from 'expo-status-bar';
import { useColorScheme } from 'react-native';
import { QueryClient, QueryClientProvider } from 'react-query';
import Toast from 'react-native-toast-message';
import * as Notifications from 'expo-notifications';
import * as SecureStore from 'expo-secure-store';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

// Screens
import { DashboardScreen } from './screens/DashboardScreen';
import { TradingScreen } from './screens/TradingScreen';
import { AnalyticsScreen } from './screens/AnalyticsScreen';
import { SettingsScreen } from './screens/SettingsScreen';
import { LoginScreen } from './screens/auth/LoginScreen';
import { SplashScreen } from './screens/SplashScreen';

// Components
import { TabBarIcon } from './components/navigation/TabBarIcon';
import { ErrorBoundary } from './components/error/ErrorBoundary';

// Contexts
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Services
import { notificationService } from './services/notificationService';
import { authService } from './services/authService';

// Utils
import { navigationRef } from './utils/navigation';

// Types
import type { RootStackParamList, TabParamList } from './types/navigation';

const Tab = createBottomTabNavigator<TabParamList>();
const Stack = createStackNavigator<RootStackParamList>();

// React Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

// 通知設定
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
  }),
});

// フォント設定
const fontConfig = {
  default: {
    regular: {
      fontFamily: 'Inter-Regular',
      fontWeight: '400' as const,
    },
    medium: {
      fontFamily: 'Inter-Medium',
      fontWeight: '500' as const,
    },
    light: {
      fontFamily: 'Inter-Light',
      fontWeight: '300' as const,
    },
    thin: {
      fontFamily: 'Inter-Thin',
      fontWeight: '100' as const,
    },
  },
};

// カスタムテーマ
const lightTheme = {
  ...DefaultTheme,
  fonts: configureFonts(fontConfig),
  colors: {
    ...DefaultTheme.colors,
    primary: '#2563eb',
    accent: '#10b981',
    surface: '#ffffff',
    background: '#f9fafb',
    error: '#ef4444',
    warning: '#f59e0b',
    success: '#10b981',
    info: '#06b6d4',
  },
};

const darkTheme = {
  ...DarkTheme,
  fonts: configureFonts(fontConfig),
  colors: {
    ...DarkTheme.colors,
    primary: '#3b82f6',
    accent: '#34d399',
    surface: '#1f2937',
    background: '#111827',
    error: '#f87171',
    warning: '#fbbf24',
    success: '#34d399',
    info: '#38bdf8',
  },
};

// メインタブナビゲーター
function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => (
          <TabBarIcon
            name={getTabIconName(route.name)}
            focused={focused}
            color={color}
            size={size}
          />
        ),
        tabBarActiveTintColor: '#2563eb',
        tabBarInactiveTintColor: '#6b7280',
        tabBarStyle: {
          backgroundColor: '#ffffff',
          borderTopColor: '#e5e7eb',
          paddingBottom: 5,
          height: 60,
        },
        headerShown: false,
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '500',
        },
      })}
    >
      <Tab.Screen
        name="Dashboard"
        component={DashboardScreen}
        options={{
          tabBarLabel: 'ダッシュボード',
        }}
      />
      <Tab.Screen
        name="Trading"
        component={TradingScreen}
        options={{
          tabBarLabel: '取引',
        }}
      />
      <Tab.Screen
        name="Analytics"
        component={AnalyticsScreen}
        options={{
          tabBarLabel: '分析',
        }}
      />
      <Tab.Screen
        name="Settings"
        component={SettingsScreen}
        options={{
          tabBarLabel: '設定',
        }}
      />
    </Tab.Navigator>
  );
}

// 認証フロー管理
function AuthenticatedApp() {
  const { user, isLoading } = useAuth();

  if (isLoading) {
    return <SplashScreen />;
  }

  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      {user ? (
        <Stack.Screen name="Main" component={TabNavigator} />
      ) : (
        <Stack.Screen name="Login" component={LoginScreen} />
      )}
    </Stack.Navigator>
  );
}

// メインアプリコンポーネント
export default function App() {
  const colorScheme = useColorScheme();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    // アプリ初期化
    const initializeApp = async () => {
      try {
        // 通知権限要求
        await notificationService.requestPermissions();

        // セキュアストレージからトークン取得
        const token = await SecureStore.getItemAsync('auth_token');
        if (token) {
          await authService.validateToken(token);
        }

        // 通知リスナー設定
        const notificationListener = Notifications.addNotificationReceivedListener(
          (notification) => {
            console.log('Notification received:', notification);
          }
        );

        const responseListener = Notifications.addNotificationResponseReceivedListener(
          (response) => {
            console.log('Notification response:', response);
            // 通知タップ時のナビゲーション処理
            handleNotificationResponse(response);
          }
        );

        setIsReady(true);

        return () => {
          Notifications.removeNotificationSubscription(notificationListener);
          Notifications.removeNotificationSubscription(responseListener);
        };
      } catch (error) {
        console.error('App initialization error:', error);
        setIsReady(true); // エラーでも続行
      }
    };

    initializeApp();
  }, []);

  const handleNotificationResponse = (response: Notifications.NotificationResponse) => {
    const data = response.notification.request.content.data;

    // 通知データに基づいてナビゲーション
    if (data?.screen) {
      navigationRef.current?.navigate(data.screen as never);
    }
  };

  if (!isReady) {
    return <SplashScreen />;
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ErrorBoundary>
        <QueryClientProvider client={queryClient}>
          <PaperProvider theme={colorScheme === 'dark' ? darkTheme : lightTheme}>
            <ThemeProvider>
              <AuthProvider>
                <WebSocketProvider>
                  <NotificationProvider>
                    <NavigationContainer
                      ref={navigationRef}
                      theme={colorScheme === 'dark' ? darkTheme : lightTheme}
                    >
                      <StatusBar
                        style={colorScheme === 'dark' ? 'light' : 'dark'}
                        backgroundColor={
                          colorScheme === 'dark' ? '#111827' : '#ffffff'
                        }
                      />
                      <AuthenticatedApp />
                    </NavigationContainer>

                    {/* Toast Messages */}
                    <Toast />
                  </NotificationProvider>
                </WebSocketProvider>
              </AuthProvider>
            </ThemeProvider>
          </PaperProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </GestureHandlerRootView>
  );
}

// タブアイコン名取得ヘルパー
function getTabIconName(routeName: string): string {
  switch (routeName) {
    case 'Dashboard':
      return 'chart-line';
    case 'Trading':
      return 'trending-up';
    case 'Analytics':
      return 'bar-chart-2';
    case 'Settings':
      return 'settings';
    default:
      return 'home';
  }
}