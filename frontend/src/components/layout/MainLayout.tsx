import { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bars3Icon,
  XMarkIcon,
  BellIcon,
  UserCircleIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';

// Components
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { NotificationPanel } from '@/components/notifications/NotificationPanel';
import { UserMenu } from '@/components/user/UserMenu';
import { QuickActions } from '@/components/layout/QuickActions';
import { StatusBar } from '@/components/layout/StatusBar';

// Hooks
import { useAuth } from '@/hooks/useAuth';
import { useNotifications } from '@/hooks/useNotifications';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';

// Utils
import { cn } from '@/utils/cn';

interface MainLayoutProps {
  className?: string;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ className }) => {
  const location = useLocation();
  const { user } = useAuth();
  const { notifications, unreadCount } = useNotifications();

  // UI State
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notificationPanelOpen, setNotificationPanelOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  // キーボードショートカット
  useKeyboardShortcuts({
    'cmd+k': () => {
      // コマンドパレットを開く
      console.log('Opening command palette...');
    },
    'cmd+/': () => {
      // ヘルプを開く
      console.log('Opening help...');
    },
    'cmd+shift+n': () => {
      // 通知パネルを開く
      setNotificationPanelOpen(!notificationPanelOpen);
    },
    'esc': () => {
      // 全てのパネルを閉じる
      setSidebarOpen(false);
      setNotificationPanelOpen(false);
      setUserMenuOpen(false);
    },
  });

  const pageVariants = {
    initial: { opacity: 0, x: 20 },
    in: { opacity: 1, x: 0 },
    out: { opacity: 0, x: -20 },
  };

  const pageTransition = {
    type: 'tween',
    ease: 'anticipate',
    duration: 0.3,
  };

  return (
    <div className={cn('flex h-screen bg-gray-50 dark:bg-gray-900', className)}>
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        className="hidden lg:flex"
      />

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {sidebarOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
              onClick={() => setSidebarOpen(false)}
            />
            <motion.div
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'tween', duration: 0.3 }}
              className="fixed inset-y-0 left-0 z-50 w-64 lg:hidden"
            >
              <Sidebar
                open={true}
                onClose={() => setSidebarOpen(false)}
                mobile
              />
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <Header className="border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between px-4 py-3">
            {/* Left Section */}
            <div className="flex items-center space-x-4">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 lg:hidden"
                aria-label="Open sidebar"
              >
                <Bars3Icon className="h-6 w-6" />
              </button>

              {/* Page Title */}
              <div className="hidden sm:block">
                <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
                  {getPageTitle(location.pathname)}
                </h1>
              </div>
            </div>

            {/* Right Section */}
            <div className="flex items-center space-x-4">
              {/* Quick Actions */}
              <QuickActions />

              {/* Notifications */}
              <div className="relative">
                <button
                  onClick={() => setNotificationPanelOpen(!notificationPanelOpen)}
                  className="p-2 text-gray-400 hover:text-gray-500 hover:bg-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  aria-label={`Notifications ${unreadCount > 0 ? `(${unreadCount} unread)` : ''}`}
                >
                  <BellIcon className="h-6 w-6" />
                  {unreadCount > 0 && (
                    <span className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {unreadCount > 99 ? '99+' : unreadCount}
                    </span>
                  )}
                </button>
              </div>

              {/* Settings */}
              <button
                onClick={() => {/* Navigate to settings */}}
                className="p-2 text-gray-400 hover:text-gray-500 hover:bg-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-label="Settings"
              >
                <Cog6ToothIcon className="h-6 w-6" />
              </button>

              {/* User Menu */}
              <div className="relative">
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center space-x-2 p-2 text-gray-400 hover:text-gray-500 hover:bg-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  aria-label="User menu"
                >
                  {user?.avatar ? (
                    <img
                      className="h-8 w-8 rounded-full"
                      src={user.avatar}
                      alt={user.name}
                    />
                  ) : (
                    <UserCircleIcon className="h-8 w-8" />
                  )}
                  <span className="hidden sm:block text-sm font-medium text-gray-700 dark:text-gray-300">
                    {user?.name}
                  </span>
                </button>

                {/* User Menu Dropdown */}
                <AnimatePresence>
                  {userMenuOpen && (
                    <UserMenu
                      onClose={() => setUserMenuOpen(false)}
                      className="absolute right-0 mt-2 w-48"
                    />
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </Header>

        {/* Main Content */}
        <main className="flex-1 overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
              className="h-full"
            >
              <div className="h-full overflow-auto p-4 lg:p-6">
                <Outlet />
              </div>
            </motion.div>
          </AnimatePresence>
        </main>

        {/* Status Bar */}
        <StatusBar className="border-t border-gray-200 dark:border-gray-700" />
      </div>

      {/* Notification Panel */}
      <AnimatePresence>
        {notificationPanelOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black bg-opacity-50"
              onClick={() => setNotificationPanelOpen(false)}
            />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'tween', duration: 0.3 }}
              className="fixed inset-y-0 right-0 z-50 w-80"
            >
              <NotificationPanel
                notifications={notifications}
                onClose={() => setNotificationPanelOpen(false)}
              />
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

// ページタイトルを取得するヘルパー関数
function getPageTitle(pathname: string): string {
  const routes: Record<string, string> = {
    '/': 'ダッシュボード',
    '/dashboard': 'ダッシュボード',
    '/trading': 'トレーディング',
    '/analytics': 'アナリティクス',
    '/monitoring': 'システム監視',
    '/reports': 'レポート',
    '/settings': '設定',
  };

  return routes[pathname] || 'Day Trade ML';
}