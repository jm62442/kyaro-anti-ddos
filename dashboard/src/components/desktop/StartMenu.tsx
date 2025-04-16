
import React from "react";
import { 
  Search, Power, Settings, User, Shield, FileText, 
  LayoutDashboard, Activity, Loader, ServerCrash, Users, Terminal,
  Database, Cpu, Layers, Brain, Code, Globe, Lock, Wifi
} from "lucide-react";
import KyaroLogo from "./KyaroLogo";
import { Button } from "@/components/ui/button";

interface StartMenuProps {
  isOpen: boolean;
  onClose: () => void;
  openApp: (appId: string) => void;
}

const StartMenu: React.FC<StartMenuProps> = ({ isOpen, onClose, openApp }) => {
  const apps = [
    { id: "dashboard", name: "Bảng Điều Khiển", icon: <LayoutDashboard size={24} /> },
    { id: "threats", name: "Phân Tích Mối Đe Dọa", icon: <Activity size={24} /> },
    { id: "protection", name: "Trạng Thái Bảo Vệ", icon: <Shield size={24} /> },
    { id: "traffic", name: "Giám Sát Lưu Lượng", icon: <Loader size={24} /> },
    { id: "attacks", name: "Lịch Sử Tấn Công", icon: <ServerCrash size={24} /> },
    { id: "clients", name: "Quản Lý Khách Hàng", icon: <Users size={24} /> },
    { id: "logs", name: "Nhật Ký Hệ Thống", icon: <FileText size={24} /> },
    { id: "console", name: "Bảng Điều Khiển", icon: <Terminal size={24} /> },
  ];

  const layerControls = [
    { id: "layer7", name: "Điều Khiển Layer 7", icon: <Database size={24} /> },
    { id: "layer4", name: "Điều Khiển Layer 4", icon: <Wifi size={24} /> },
    { id: "layer3", name: "Điều Khiển Layer 3", icon: <Cpu size={24} /> },
    { id: "mlsettings", name: "Cài Đặt ML/DL", icon: <Brain size={24} /> },
  ];

  const developerTools = [
    { id: "api", name: "Tài Liệu API", icon: <Code size={24} /> },
    { id: "integration", name: "Hướng Dẫn Tích Hợp", icon: <Layers size={24} /> },
    { id: "global", name: "Mạng Toàn Cầu", icon: <Globe size={24} /> },
    { id: "badge", name: "Huy Hiệu Bảo Mật", icon: <Lock size={24} /> },
  ];

  if (!isOpen) return null;

  const handleAppClick = (appId: string) => {
    openApp(appId);
    onClose();
  };

  return (
    <div 
      className="start-menu animate-slide-up z-50 absolute bottom-12 left-0 w-80 sm:w-96 bg-black/90 backdrop-blur-lg p-4 rounded-lg border border-white/10 shadow-2xl"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <KyaroLogo size={32} />
          <div>
            <h2 className="text-lg font-semibold text-white">Kyaro</h2>
            <p className="text-xs text-white/70">Nền Tảng Chống DDoS</p>
          </div>
        </div>
        <User className="bg-blue-600 text-white p-1 rounded-full" size={32} />
      </div>
      
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-white/60" size={18} />
        <input 
          type="text" 
          className="w-full bg-white/10 rounded-lg py-2 pl-10 pr-4 text-white placeholder-white/60 outline-none focus:ring-2 focus:ring-blue-500/50"
          placeholder="Tìm kiếm ứng dụng..."
        />
      </div>
      
      <h3 className="text-sm font-medium text-white/70 mb-2">Công Cụ Chống DDoS</h3>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {apps.map(app => (
          <button 
            key={app.id}
            className="flex items-center space-x-2 p-2 rounded-lg hover:bg-white/10 transition-colors text-left"
            onClick={() => handleAppClick(app.id)}
          >
            <div className="p-2 rounded-lg bg-[hsl(var(--primary))]">
              {app.icon}
            </div>
            <span className="text-sm text-white">{app.name}</span>
          </button>
        ))}
      </div>
      
      <h3 className="text-sm font-medium text-white/70 mb-2">Lớp Bảo Vệ</h3>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {layerControls.map(app => (
          <button 
            key={app.id}
            className="flex items-center space-x-2 p-2 rounded-lg hover:bg-white/10 transition-colors text-left"
            onClick={() => handleAppClick(app.id)}
          >
            <div className="p-2 rounded-lg bg-purple-600">
              {app.icon}
            </div>
            <span className="text-sm text-white">{app.name}</span>
          </button>
        ))}
      </div>
      
      <h3 className="text-sm font-medium text-white/70 mb-2">Tài Nguyên Cho Nhà Phát Triển</h3>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {developerTools.map(app => (
          <button 
            key={app.id}
            className="flex items-center space-x-2 p-2 rounded-lg hover:bg-white/10 transition-colors text-left"
            onClick={() => handleAppClick(app.id)}
          >
            <div className="p-2 rounded-lg bg-blue-600">
              {app.icon}
            </div>
            <span className="text-sm text-white">{app.name}</span>
          </button>
        ))}
      </div>
      
      <div className="flex items-center justify-between mt-2">
        <button 
          className="flex items-center space-x-2 p-2 rounded-lg hover:bg-white/10 transition-colors"
          onClick={() => handleAppClick('settings')}
        >
          <Settings size={20} className="text-white/70" />
          <span className="text-sm text-white">Cài Đặt</span>
        </button>
        <button className="flex items-center space-x-2 p-2 rounded-lg hover:bg-white/10 transition-colors">
          <Power size={20} className="text-white/70" />
          <span className="text-sm text-white">Tắt Máy</span>
        </button>
      </div>
      
      <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-center">
        <Button variant="ghost" size="sm" className="text-xs text-white/60">
          Kyaro Anti-DDoS v1.0.0
        </Button>
      </div>
    </div>
  );
};

export default StartMenu;
