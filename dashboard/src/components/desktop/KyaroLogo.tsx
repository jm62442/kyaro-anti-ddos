
import React from "react";

interface KyaroLogoProps {
  size?: number;
  className?: string;
}

const KyaroLogo: React.FC<KyaroLogoProps> = ({ size = 40, className = "" }) => {
  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <svg 
        viewBox="0 0 100 100" 
        fill="none" 
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        <circle cx="50" cy="50" r="45" fill="url(#kyaroGradient)" />
        <path 
          d="M30 30L50 50M50 50L70 30M50 50L30 70M50 50L70 70" 
          stroke="white" 
          strokeWidth="6" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
        />
        <circle cx="50" cy="50" r="35" stroke="rgba(255,255,255,0.3)" strokeWidth="2" strokeDasharray="4 4" />
        <defs>
          <linearGradient id="kyaroGradient" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#3B82F6" />
            <stop offset="100%" stopColor="#8B5CF6" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 bg-white/10 rounded-full blur-sm"></div>
    </div>
  );
};

export default KyaroLogo;
