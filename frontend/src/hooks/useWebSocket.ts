import { useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { toast } from 'react-hot-toast';

// Types
interface WebSocketConfig {
  autoReconnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  timeout?: number;
}

interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  reconnectCount: number;
}

interface UseWebSocketReturn<T> {
  data: T | null;
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  reconnectCount: number;
  send: (event: string, data?: any) => void;
  disconnect: () => void;
  reconnect: () => void;
}

const DEFAULT_CONFIG: Required<WebSocketConfig> = {
  autoReconnect: true,
  reconnectInterval: 5000,
  maxReconnectAttempts: 10,
  timeout: 10000,
};

export function useWebSocket<T = any>(
  endpoint: string,
  config: WebSocketConfig = {}
): UseWebSocketReturn<T> {
  const configRef = useRef({ ...DEFAULT_CONFIG, ...config });
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isManualDisconnectRef = useRef(false);

  const [data, setData] = useState<T | null>(null);
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    reconnectCount: 0,
  });

  // エラーハンドリング
  const handleError = useCallback((error: string) => {
    setState(prev => ({ ...prev, error, isConnecting: false }));
    console.error('WebSocket error:', error);

    // ユーザーに通知（重要なエラーの場合）
    if (error.includes('authentication') || error.includes('unauthorized')) {
      toast.error('認証エラーが発生しました。再ログインしてください。');
    }
  }, []);

  // 再接続処理
  const scheduleReconnect = useCallback(() => {
    const { autoReconnect, reconnectInterval, maxReconnectAttempts } = configRef.current;

    if (!autoReconnect || isManualDisconnectRef.current) {
      return;
    }

    setState(prev => {
      if (prev.reconnectCount >= maxReconnectAttempts) {
        handleError(`最大再接続試行回数(${maxReconnectAttempts})に達しました`);
        return prev;
      }

      // 再接続スケジュール
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log(`WebSocket reconnecting... (attempt ${prev.reconnectCount + 1}/${maxReconnectAttempts})`);
        connect();
      }, reconnectInterval);

      return {
        ...prev,
        reconnectCount: prev.reconnectCount + 1,
        isConnecting: true,
      };
    });
  }, [handleError]);

  // 接続処理
  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    setState(prev => ({ ...prev, isConnecting: true, error: null }));
    isManualDisconnectRef.current = false;

    try {
      // Socket.IO接続設定
      const socket = io(endpoint, {
        timeout: configRef.current.timeout,
        transports: ['websocket', 'polling'],
        upgrade: true,
        rememberUpgrade: true,
        auth: {
          // 認証トークンを追加
          token: localStorage.getItem('auth_token'),
        },
      });

      // 接続成功
      socket.on('connect', () => {
        console.log('WebSocket connected to:', endpoint);
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null,
          reconnectCount: 0,
        }));

        // 再接続成功時の通知
        if (state.reconnectCount > 0) {
          toast.success('リアルタイム接続が復旧しました');
        }
      });

      // データ受信
      socket.on('data', (receivedData: T) => {
        setData(receivedData);
      });

      // 特定イベントリスナー
      socket.on('trade_update', (tradeData: any) => {
        setData(prev => ({ ...prev, latestTrade: tradeData }));
      });

      socket.on('kpi_update', (kpiData: any) => {
        setData(prev => ({ ...prev, ...kpiData }));
      });

      socket.on('alert', (alertData: any) => {
        setData(prev => ({ ...prev, alerts: alertData }));

        // 重要なアラートはToast表示
        if (alertData.severity === 'critical') {
          toast.error(alertData.message);
        } else if (alertData.severity === 'warning') {
          toast.warning(alertData.message);
        }
      });

      // 切断
      socket.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setState(prev => ({ ...prev, isConnected: false, isConnecting: false }));

        // 自動再接続が必要な切断の場合
        if (reason === 'io server disconnect' || reason === 'transport close') {
          scheduleReconnect();
        }
      });

      // エラー
      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        handleError(`接続エラー: ${error.message}`);
        scheduleReconnect();
      });

      // 認証エラー
      socket.on('unauthorized', (error) => {
        handleError('認証に失敗しました');
        isManualDisconnectRef.current = true; // 自動再接続を停止
      });

      socketRef.current = socket;

    } catch (error) {
      handleError(`WebSocket初期化エラー: ${error}`);
    }
  }, [endpoint, handleError, scheduleReconnect, state.reconnectCount]);

  // 手動切断
  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }

    setState({
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectCount: 0,
    });

    setData(null);
  }, []);

  // 手動再接続
  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(() => {
      setState(prev => ({ ...prev, reconnectCount: 0 }));
      connect();
    }, 1000);
  }, [disconnect, connect]);

  // メッセージ送信
  const send = useCallback((event: string, data?: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    } else {
      console.warn('WebSocket not connected. Cannot send message:', event, data);
    }
  }, []);

  // 初期接続
  useEffect(() => {
    connect();

    // クリーンアップ
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // ページフォーカス時の再接続
  useEffect(() => {
    const handleFocus = () => {
      if (!state.isConnected && !state.isConnecting && !isManualDisconnectRef.current) {
        console.log('Page focused, attempting to reconnect WebSocket');
        reconnect();
      }
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        handleFocus();
      }
    };

    window.addEventListener('focus', handleFocus);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      window.removeEventListener('focus', handleFocus);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [state.isConnected, state.isConnecting, reconnect]);

  // オンライン/オフライン状態の監視
  useEffect(() => {
    const handleOnline = () => {
      if (!state.isConnected && !isManualDisconnectRef.current) {
        console.log('Network back online, attempting to reconnect WebSocket');
        reconnect();
      }
    };

    const handleOffline = () => {
      console.log('Network offline, WebSocket will reconnect when back online');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [state.isConnected, reconnect]);

  return {
    data,
    isConnected: state.isConnected,
    isConnecting: state.isConnecting,
    error: state.error,
    reconnectCount: state.reconnectCount,
    send,
    disconnect,
    reconnect,
  };
}