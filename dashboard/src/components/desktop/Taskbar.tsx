
import React, { useState, useEffect } from "react";
import { ChevronUp, Menu, Settings, Shield, Bell, Wifi, Volume2, Clock } from "lucide-react";
import KyaroLogo from "./KyaroLogo";
import { showNotification } from "./NotificationToast";
import KyaroBadge from "./KyaroBadge";
import { Button } from "@/components/ui/button";

interface TaskbarProps {
  openStartMenu: boolean;
  toggleStartMenu: () => void;
  openWindows: { id: string; title: string; icon: React.ReactNode; isMinimized: boolean }[];
  setActiveWindow: (id: string) => void;
  restoreWindow: (id: string) => void;
}

const Taskbar: React.FC<TaskbarProps> = ({ 
  openStartMenu, 
  toggleStartMenu, 
  openWindows,
  setActiveWindow,
  restoreWindow
}) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [showBadge, setShowBadge] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const handleSystemButtonClick = (type: string) => {
    showNotification({ message: "Chức năng này chưa cần nên không có sẵn đâu :)" });
  };

  const toggleBadge = () => {
    setShowBadge(!showBadge);
  };
  
  const handleWindowClick = (id: string, isMinimized: boolean) => {
    if (isMinimized) {
      restoreWindow(id);
    } else {
      setActiveWindow(id);
    }
  };

  return (
    <div className="taskbar w-full fixed bottom-0 left-0 h-12 flex items-center justify-between px-4 z-50 bg-black/80 backdrop-blur-md">
      <div className="flex items-center space-x-2">
        <button 
          className={`p-2 rounded-lg transition-colors duration-200 ${openStartMenu ? 'bg-white/20' : 'hover:bg-white/10'}`}
          onClick={toggleStartMenu}
          aria-label="Menu chính"
        >
          <KyaroLogo size={26} />
        </button>
        
        <div className="h-6 w-px bg-white/20 mx-2" />
        
        {openWindows.map(window => (
          <button 
            key={window.id}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg ${window.isMinimized ? 'bg-white/5' : 'bg-white/10'} hover:bg-white/20 transition-colors duration-200`}
            onClick={() => handleWindowClick(window.id, window.isMinimized)}
          >
            {window.icon}
            <span className="text-xs font-medium text-white">{window.title}</span>
          </button>
        ))}
      </div>
      
      <div className="flex items-center space-x-4 text-white">
        <div className="flex items-center space-x-1">
          <button 
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-200"
            onClick={() => handleSystemButtonClick('wifi')}
          >
            <Wifi size={18} />
          </button>
          <button 
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-200"
            onClick={() => handleSystemButtonClick('volume')}
          >
            <Volume2 size={18} />
          </button>
          <button 
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-200"
            onClick={() => handleSystemButtonClick('notifications')}
          >
            <Bell size={18} />
          </button>
          <Button 
            variant="ghost" 
            size="sm" 
            className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-200"
            onClick={toggleBadge}
          >
            <Shield size={18} />
          </Button>
        </div>
        
        {showBadge && (
          <div className="mr-2">
            <KyaroBadge size="sm" />
          </div>
        )}
        
        <div className="flex flex-col items-end bg-white/5 px-3 py-1 rounded-lg text-xs">
          <span>{formatTime(currentTime)}</span>
          <span className="text-white/70">{formatDate(currentTime)}</span>
        </div>
        
        <button 
          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-200"
          onClick={() => handleSystemButtonClick('up')}
        >
          <ChevronUp size={18} />
        </button>
      </div>
    </div>
  );
};

export default Taskbar;
