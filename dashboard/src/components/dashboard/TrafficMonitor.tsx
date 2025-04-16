
import React from 'react';
import { Activity, ArrowDown, ArrowUp, Zap, Clock, RefreshCw } from 'lucide-react';

const TrafficMonitor: React.FC = () => {
  return (
    <div className="text-white">
      <div className="glass-dark p-4 rounded-lg mb-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium flex items-center">
            <Activity className="mr-2" size={20} />
            Live Traffic Monitor
          </h3>
          <div className="flex items-center space-x-2">
            <div className="flex items-center bg-green-500/20 text-green-400 text-xs py-1 px-2 rounded-full">
              <div className="h-2 w-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
              <span>Live</span>
            </div>
            <button className="p-1 rounded-lg hover:bg-white/10 transition-colors">
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
        
        <div className="mb-6">
          <div className="grid grid-cols-4 gap-4 mb-4">
            <NetworkStat 
              label="Incoming" 
              value="345 Mbps" 
              icon={<ArrowDown className="text-blue-400" />} 
              percentage={62}
              color="bg-blue-500"
            />
            <NetworkStat 
              label="Outgoing" 
              value="127 Mbps" 
              icon={<ArrowUp className="text-purple-400" />} 
              percentage={24}
              color="bg-purple-500"
            />
            <NetworkStat 
              label="Latency" 
              value="18 ms" 
              icon={<Clock className="text-yellow-400" />} 
              percentage={12}
              color="bg-yellow-500"
            />
            <NetworkStat 
              label="Filtered" 
              value="87 Mbps" 
              icon={<Zap className="text-red-400" />} 
              percentage={32}
              color="bg-red-500"
            />
          </div>
        </div>
        
        <div className="relative h-60 bg-[#1a1f2c]/50 rounded-lg p-4 overflow-hidden">
          <div className="absolute inset-0 flex flex-col justify-end">
            {/* Graph lines */}
            <svg className="w-full h-full" viewBox="0 0 100 40" preserveAspectRatio="none">
              {/* Grid lines */}
              <line x1="0" y1="10" x2="100" y2="10" stroke="rgba(255,255,255,0.1)" strokeWidth="0.1" />
              <line x1="0" y1="20" x2="100" y2="20" stroke="rgba(255,255,255,0.1)" strokeWidth="0.1" />
              <line x1="0" y1="30" x2="100" y2="30" stroke="rgba(255,255,255,0.1)" strokeWidth="0.1" />
              
              {/* Incoming traffic */}
              <path
                d="M0,35 C10,32 15,25 20,20 S30,10 40,15 S50,25 60,20 S70,5 80,10 S90,15 100,10"
                fill="none"
                stroke="rgba(59, 130, 246, 0.8)"
                strokeWidth="0.5"
              />
              <path
                d="M0,35 C10,32 15,25 20,20 S30,10 40,15 S50,25 60,20 S70,5 80,10 S90,15 100,10"
                fill="url(#blueGradient)"
                fillOpacity="0.2"
                stroke="none"
              />
              
              {/* Outgoing traffic */}
              <path
                d="M0,38 C5,36 15,35 25,32 S35,25 45,28 S55,32 65,30 S75,25 85,28 S95,30 100,28"
                fill="none"
                stroke="rgba(139, 92, 246, 0.8)"
                strokeWidth="0.5"
              />
              <path
                d="M0,38 C5,36 15,35 25,32 S35,25 45,28 S55,32 65,30 S75,25 85,28 S95,30 100,28"
                fill="url(#purpleGradient)"
                fillOpacity="0.2"
                stroke="none"
              />
              
              {/* Filtered traffic */}
              <path
                d="M0,39 C10,38 20,37 30,38 S40,36 50,37 S60,38 70,36 S80,34 90,37 S95,39 100,37"
                fill="none"
                stroke="rgba(239, 68, 68, 0.8)"
                strokeWidth="0.5"
              />
              <path
                d="M0,39 C10,38 20,37 30,38 S40,36 50,37 S60,38 70,36 S80,34 90,37 S95,39 100,37"
                fill="url(#redGradient)"
                fillOpacity="0.2"
                stroke="none"
              />
              
              <defs>
                <linearGradient id="blueGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(59, 130, 246, 0.5)" />
                  <stop offset="100%" stopColor="rgba(59, 130, 246, 0)" />
                </linearGradient>
                <linearGradient id="purpleGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(139, 92, 246, 0.5)" />
                  <stop offset="100%" stopColor="rgba(139, 92, 246, 0)" />
                </linearGradient>
                <linearGradient id="redGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(239, 68, 68, 0.5)" />
                  <stop offset="100%" stopColor="rgba(239, 68, 68, 0)" />
                </linearGradient>
              </defs>
            </svg>
            
            {/* Time labels */}
            <div className="flex justify-between text-xs text-white/50 px-2">
              <span>1m ago</span>
              <span>45s</span>
              <span>30s</span>
              <span>15s</span>
              <span>now</span>
            </div>
          </div>
          
          {/* Graph legend */}
          <div className="absolute top-4 right-4 bg-black/30 p-2 rounded-lg">
            <div className="flex items-center text-xs mb-1">
              <div className="h-2 w-2 bg-blue-500 rounded-full mr-2"></div>
              <span>Incoming</span>
            </div>
            <div className="flex items-center text-xs mb-1">
              <div className="h-2 w-2 bg-purple-500 rounded-full mr-2"></div>
              <span>Outgoing</span>
            </div>
            <div className="flex items-center text-xs">
              <div className="h-2 w-2 bg-red-500 rounded-full mr-2"></div>
              <span>Filtered</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3">Top Protocols</h3>
          <div className="space-y-3">
            {[
              { protocol: "HTTPS", percentage: 64, color: "bg-green-500" },
              { protocol: "HTTP", percentage: 18, color: "bg-blue-500" },
              { protocol: "DNS", percentage: 10, color: "bg-yellow-500" },
              { protocol: "SMTP", percentage: 5, color: "bg-purple-500" },
              { protocol: "Other", percentage: 3, color: "bg-gray-500" }
            ].map((item, index) => (
              <div key={index}>
                <div className="flex justify-between mb-1">
                  <span className="text-xs">{item.protocol}</span>
                  <span className="text-xs font-medium">{item.percentage}%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div 
                    className={`${item.color} h-2 rounded-full`} 
                    style={{ width: `${item.percentage}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3">Network Status</h3>
          <div className="space-y-3">
            {[
              { node: "Main Gateway", status: "Online", statusColor: "text-green-400" },
              { node: "Backup Gateway", status: "Standby", statusColor: "text-blue-400" },
              { node: "Load Balancer", status: "Online", statusColor: "text-green-400" },
              { node: "CDN Edge", status: "Online", statusColor: "text-green-400" },
              { node: "Firewall", status: "Active", statusColor: "text-green-400" }
            ].map((item, index) => (
              <div key={index} className="flex justify-between items-center bg-white/5 p-2 rounded-lg">
                <span className="text-sm">{item.node}</span>
                <span className={`text-xs font-medium ${item.statusColor}`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

interface NetworkStatProps {
  label: string;
  value: string;
  icon: React.ReactNode;
  percentage: number;
  color: string;
}

const NetworkStat: React.FC<NetworkStatProps> = ({ label, value, icon, percentage, color }) => {
  return (
    <div className="bg-white/5 p-3 rounded-lg">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-white/70">{label}</span>
        {icon}
      </div>
      <p className="text-xl font-bold mb-2">{value}</p>
      <div className="w-full bg-white/10 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full`} style={{ width: `${percentage}%` }}></div>
      </div>
    </div>
  );
};

export default TrafficMonitor;
