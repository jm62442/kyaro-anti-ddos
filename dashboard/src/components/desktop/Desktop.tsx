
import React, { useState, useRef, useEffect } from 'react';
import { 
  LayoutDashboard, Activity, Shield, Loader, ServerCrash, 
  FileText, Settings, Terminal, Users, Database, Wifi, Cpu, Brain, 
  Code, Layers, Globe, Lock
} from 'lucide-react';
import Taskbar from './Taskbar';
import StartMenu from './StartMenu';
import Window from './Window';
import DesktopIcon from './DesktopIcon';
import Dashboard from '../dashboard/Dashboard';
import ThreatAnalytics from '../dashboard/ThreatAnalytics';
import TrafficMonitor from '../dashboard/TrafficMonitor';
import KyaroLogo from './KyaroLogo';
import LayerControls from '../dashboard/LayerControls';
import MLSettings from '../dashboard/MLSettings';
import KyaroBadge from './KyaroBadge';
import { toast } from "sonner";

const Desktop: React.FC = () => {
  const [openStartMenu, setOpenStartMenu] = useState(false);
  const [windows, setWindows] = useState<Array<{
    id: string;
    title: string;
    icon: React.ReactNode;
    content: React.ReactNode;
    position: { x: number; y: number };
    size: { width: number | string; height: number | string };
    isActive: boolean;
    isMinimized: boolean;
  }>>([]);
  
  const desktopRef = useRef<HTMLDivElement>(null);
  const [iconsPerRow, setIconsPerRow] = useState(2);

  // Monitor desktop width and adjust icons per row
  useEffect(() => {
    const updateLayout = () => {
      const width = desktopRef.current?.clientWidth || 0;
      setIconsPerRow(width < 640 ? 1 : 2);
    };

    updateLayout();
    window.addEventListener('resize', updateLayout);
    return () => window.removeEventListener('resize', updateLayout);
  }, []);

  useEffect(() => {
    // Close start menu when clicking outside
    const handleClickOutside = (event: MouseEvent) => {
      const targetElement = event.target as HTMLElement;
      const isStartMenu = targetElement.closest('.start-menu');
      const isTaskbarButton = targetElement.closest('.taskbar button[aria-label="Menu chính"]');
      
      if (!isStartMenu && !isTaskbarButton && openStartMenu) {
        setOpenStartMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [openStartMenu]);

  const toggleStartMenu = () => {
    setOpenStartMenu(prev => !prev);
  };

  const closeStartMenu = () => {
    setOpenStartMenu(false);
  };

  const openApp = (appId: string) => {
    let newWindow;
    const randomOffset = Math.floor(Math.random() * 50);
    
    const windowIcons: Record<string, React.ReactNode> = {
      dashboard: <LayoutDashboard size={16} className="text-white" />,
      threats: <Activity size={16} className="text-white" />,
      protection: <Shield size={16} className="text-white" />,
      traffic: <Loader size={16} className="text-white" />,
      attacks: <ServerCrash size={16} className="text-white" />,
      clients: <Users size={16} className="text-white" />,
      logs: <FileText size={16} className="text-white" />,
      console: <Terminal size={16} className="text-white" />,
      settings: <Settings size={16} className="text-white" />,
      layer7: <Database size={16} className="text-white" />,
      layer4: <Wifi size={16} className="text-white" />,
      layer3: <Cpu size={16} className="text-white" />,
      mlsettings: <Brain size={16} className="text-white" />,
      api: <Code size={16} className="text-white" />,
      integration: <Layers size={16} className="text-white" />,
      global: <Globe size={16} className="text-white" />,
      badge: <Lock size={16} className="text-white" />,
    };
    
    const windowTitles: Record<string, string> = {
      dashboard: 'Bảng Điều Khiển Kyaro Anti-DDoS',
      threats: 'Phân Tích Mối Đe Dọa',
      protection: 'Trạng Thái Bảo Vệ',
      traffic: 'Giám Sát Lưu Lượng',
      attacks: 'Lịch Sử Tấn Công',
      clients: 'Quản Lý Khách Hàng',
      logs: 'Nhật Ký Hệ Thống',
      console: 'Bảng Điều Khiển',
      settings: 'Cài Đặt',
      layer7: 'Điều Khiển Layer 7',
      layer4: 'Điều Khiển Layer 4',
      layer3: 'Điều Khiển Layer 3',
      mlsettings: 'Cài Đặt ML/DL',
      api: 'Tài Liệu API',
      integration: 'Hướng Dẫn Tích Hợp',
      global: 'Mạng Toàn Cầu',
      badge: 'Huy Hiệu Bảo Mật Kyaro',
    };
    
    const windowContents: Record<string, React.ReactNode> = {
      dashboard: <Dashboard />,
      threats: <ThreatAnalytics />,
      traffic: <TrafficMonitor />,
      protection: <div className="p-4 text-white">Nội Dung Trạng Thái Bảo Vệ</div>,
      attacks: <div className="p-4 text-white">Nội Dung Lịch Sử Tấn Công</div>,
      clients: <div className="p-4 text-white">Nội Dung Quản Lý Khách Hàng</div>,
      logs: <div className="p-4 text-white">Nội Dung Nhật Ký Hệ Thống</div>,
      layer7: <LayerControls />,
      layer4: <LayerControls />,
      layer3: <LayerControls />,
      mlsettings: <MLSettings />,
      console: <div className="p-4 text-white bg-black/40 font-mono h-full rounded">
        <div className="mb-2">{"> Kyaro Anti-DDoS Console v1.0"}</div>
        <div className="mb-2">{"> Gõ 'help' để xem danh sách lệnh"}</div>
        <div className="mb-2 text-green-400">{"> Các mô-đun bảo mật đã được tải thành công"}</div>
        <div className="mb-2 text-yellow-400">{"> Đã phát hiện và ngăn chặn 2 mối đe dọa"}</div>
        <div className="flex items-center">
          <span className="mr-1">{">"}</span>
          <div className="w-2 h-4 bg-white animate-pulse"></div>
        </div>
      </div>,
      settings: <div className="p-4 text-white">Nội Dung Cài Đặt</div>,
      api: <div className="p-4 text-white">Nội Dung Tài Liệu API</div>,
      integration: <div className="p-4 text-white">Nội Dung Hướng Dẫn Tích Hợp</div>,
      global: <div className="p-4 text-white">Nội Dung Mạng Toàn Cầu</div>,
      badge: <div className="p-4 text-white flex flex-col items-center justify-center space-y-6">
        <h2 className="text-xl font-semibold">Huy Hiệu Bảo Mật Kyaro</h2>
        <p className="text-center max-w-md text-white/70">Hiển thị huy hiệu Kyaro Anti-DDoS trên trang web của bạn để cho khách truy cập biết rằng trang web của bạn được bảo vệ khỏi các cuộc tấn công DDoS.</p>
        <div className="my-6">
          <KyaroBadge size="lg" />
        </div>
        <div className="text-center max-w-md text-white/70">
          <p>Sao chép mã ở trên để nhúng huy hiệu vào trang web của bạn. Huy hiệu sẽ liên kết đến Kyaro.com.</p>
        </div>
      </div>,
    };
    
    newWindow = {
      id: appId + Date.now(),
      title: windowTitles[appId],
      icon: windowIcons[appId],
      content: windowContents[appId],
      position: { x: 100 + randomOffset, y: 100 + randomOffset },
      size: { width: 900, height: 600 },
      isActive: true,
      isMinimized: false,
    };
    
    setWindows(prev => {
      // Make all windows inactive
      const updatedWindows = prev.map(w => ({ ...w, isActive: false }));
      // Add new window
      return [...updatedWindows, newWindow];
    });
    
    // Close the start menu after opening an app
    closeStartMenu();
  };

  const closeWindow = (id: string) => {
    setWindows(prev => prev.filter(w => w.id !== id));
  };

  const setActiveWindow = (id: string) => {
    setWindows(prev => 
      prev.map(w => ({
        ...w,
        isActive: w.id === id,
      }))
    );
  };

  const minimizeWindow = (id: string) => {
    setWindows(prev => 
      prev.map(w => ({
        ...w,
        isMinimized: w.id === id ? true : w.isMinimized,
        isActive: w.id === id ? false : w.isActive
      }))
    );
  };

  const restoreWindow = (id: string) => {
    setWindows(prev => 
      prev.map(w => ({
        ...w,
        isMinimized: w.id === id ? false : w.isMinimized,
        isActive: w.id === id ? true : false
      }))
    );
  };

  const renderDesktopIcons = () => {
    const icons = [
      { id: 'dashboard', icon: LayoutDashboard, label: 'Bảng Điều Khiển' },
      { id: 'threats', icon: Activity, label: 'Mối Đe Dọa' },
      { id: 'layer7', icon: Database, label: 'Layer 7' },
      { id: 'layer4', icon: Wifi, label: 'Layer 4' },
      { id: 'layer3', icon: Cpu, label: 'Layer 3' },
      { id: 'mlsettings', icon: Brain, label: 'ML/DL' },
      { id: 'badge', icon: Lock, label: 'Huy Hiệu' },
      { id: 'settings', icon: Settings, label: 'Cài Đặt' },
    ];

    return (
      <div className={`absolute top-4 left-4 grid grid-cols-${iconsPerRow} gap-2`}>
        {icons.map(icon => (
          <DesktopIcon 
            key={icon.id}
            icon={icon.icon}
            label={icon.label}
            onClick={() => openApp(icon.id)}
          />
        ))}
      </div>
    );
  };

  return (
    <div 
      className="desktop-container w-full h-screen bg-[hsl(var(--desktop-bg))] overflow-hidden relative"
      ref={desktopRef}
    >
      {/* Desktop background effects */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0f172a] to-[#1e293b] opacity-80"></div>
      <div className="absolute inset-0 opacity-20">
        <svg width="100%" height="100%">
          <defs>
            <pattern id="smallGrid" width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M 8 0 L 0 0 0 8" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
            </pattern>
            <pattern id="grid" width="80" height="80" patternUnits="userSpaceOnUse">
              <rect width="80" height="80" fill="url(#smallGrid)" />
              <path d="M 80 0 L 0 0 0 80" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>
      
      {/* Kyaro Logo watermark */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-5">
        <KyaroLogo size={500} />
      </div>

      {/* Desktop icons */}
      {renderDesktopIcons()}

      {/* Windows */}
      {windows.map(window => (
        !window.isMinimized && (
          <Window
            key={window.id}
            id={window.id}
            title={window.title}
            icon={window.icon}
            isActive={window.isActive}
            defaultPosition={window.position}
            defaultSize={window.size}
            onClose={() => closeWindow(window.id)}
            onFocus={() => setActiveWindow(window.id)}
            onMinimize={() => minimizeWindow(window.id)}
          >
            {window.content}
          </Window>
        )
      ))}

      {/* Start menu */}
      <StartMenu 
        isOpen={openStartMenu} 
        onClose={closeStartMenu}
        openApp={openApp}
      />

      {/* Taskbar */}
      <Taskbar 
        openStartMenu={openStartMenu}
        toggleStartMenu={toggleStartMenu}
        openWindows={windows.map(w => ({ 
          id: w.id, 
          title: w.title, 
          icon: w.icon,
          isMinimized: w.isMinimized
        }))}
        setActiveWindow={setActiveWindow}
        restoreWindow={restoreWindow}
      />
    </div>
  );
};

export default Desktop;
