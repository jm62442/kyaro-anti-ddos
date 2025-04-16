
import React from "react";
import { LucideIcon } from "lucide-react";

interface DesktopIconProps {
  icon: LucideIcon;
  label: string;
  onClick: () => void;
}

const DesktopIcon: React.FC<DesktopIconProps> = ({ icon: Icon, label, onClick }) => {
  return (
    <div 
      className="desktop-icon text-white cursor-pointer w-24 h-24 flex flex-col items-center justify-center hover:bg-white/5 rounded-lg transition-colors"
      onClick={onClick}
    >
      <div className="p-2 rounded-lg bg-white/10 backdrop-blur-sm mb-1 hover:bg-white/20 transition-colors">
        <Icon size={32} />
      </div>
      <span className="text-xs text-center font-medium tracking-tight">{label}</span>
    </div>
  );
};

export default DesktopIcon;
