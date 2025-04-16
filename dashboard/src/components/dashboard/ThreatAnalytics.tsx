
import React from 'react';
import { BarChart3, Activity, Map, AlertTriangle, Shield, ArrowUpRight, ArrowDownRight } from 'lucide-react';

const ThreatAnalytics: React.FC = () => {
  return (
    <div className="text-white">
      <div className="grid grid-cols-3 gap-4 mb-6">
        <StatCard 
          title="DDoS Attempts" 
          value="847" 
          change="+23%" 
          trend="up" 
          icon={<Activity size={20} />} 
        />
        <StatCard 
          title="Blocked Attacks" 
          value="823" 
          change="+18%" 
          trend="up" 
          icon={<Shield size={20} />} 
        />
        <StatCard 
          title="Success Rate" 
          value="97.2%" 
          change="-0.5%" 
          trend="down" 
          icon={<AlertTriangle size={20} />} 
        />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3 flex items-center">
            <BarChart3 className="mr-2" size={20} />
            Attack Types
          </h3>
          <div className="space-y-4">
            {[
              { type: 'TCP/SYN Flood', percentage: 42, color: 'bg-blue-500' },
              { type: 'UDP Flood', percentage: 28, color: 'bg-purple-500' },
              { type: 'HTTP Flood', percentage: 17, color: 'bg-green-500' },
              { type: 'Other', percentage: 13, color: 'bg-yellow-500' }
            ].map((item, index) => (
              <div key={index}>
                <div className="flex justify-between mb-1">
                  <span className="text-xs">{item.type}</span>
                  <span className="text-xs font-bold">{item.percentage}%</span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-2">
                  <div className={`${item.color} h-2 rounded-full`} style={{ width: `${item.percentage}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3 flex items-center">
            <Map className="mr-2" size={20} />
            Attack Origins
          </h3>
          <div className="h-[220px] relative bg-[#1a1f2c] rounded-lg overflow-hidden">
            {/* World map visualization */}
            <div className="absolute inset-0 opacity-20">
              <svg viewBox="0 0 800 450" className="w-full h-full">
                <path 
                  d="M 100 100 L 200 150 L 300 120 L 400 180 L 500 150 L 600 200 L 700 180" 
                  stroke="white" 
                  strokeWidth="1" 
                  fill="none" 
                />
                <path 
                  d="M 150 180 L 250 220 L 350 200 L 450 250 L 550 220 L 650 280" 
                  stroke="white" 
                  strokeWidth="1" 
                  fill="none" 
                />
                <path 
                  d="M 120 250 L 220 280 L 320 260 L 420 300 L 520 280 L 620 320" 
                  stroke="white" 
                  strokeWidth="1" 
                  fill="none" 
                />
              </svg>
            </div>
            
            {/* Attack hotspots */}
            <div className="absolute left-[20%] top-[30%]">
              <div className="h-4 w-4 bg-red-500 rounded-full animate-ping opacity-75"></div>
            </div>
            <div className="absolute left-[35%] top-[60%]">
              <div className="h-5 w-5 bg-orange-500 rounded-full animate-ping opacity-75"></div>
            </div>
            <div className="absolute left-[65%] top-[25%]">
              <div className="h-6 w-6 bg-yellow-500 rounded-full animate-ping opacity-75"></div>
            </div>
            <div className="absolute left-[80%] top-[45%]">
              <div className="h-4 w-4 bg-red-500 rounded-full animate-ping opacity-75"></div>
            </div>
            
            {/* Legend */}
            <div className="absolute bottom-2 left-2 bg-black/50 p-2 rounded text-xs">
              <div className="flex items-center">
                <div className="h-2 w-2 bg-red-500 rounded-full mr-1"></div>
                <span>High</span>
              </div>
              <div className="flex items-center">
                <div className="h-2 w-2 bg-orange-500 rounded-full mr-1"></div>
                <span>Medium</span>
              </div>
              <div className="flex items-center">
                <div className="h-2 w-2 bg-yellow-500 rounded-full mr-1"></div>
                <span>Low</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-dark p-4 rounded-lg">
        <h3 className="text-lg font-medium mb-3">Top Attack Sources</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-white/70 text-xs border-b border-white/10">
                <th className="pb-2 font-medium">IP Address</th>
                <th className="pb-2 font-medium">Country</th>
                <th className="pb-2 font-medium">Attack Type</th>
                <th className="pb-2 font-medium">Requests</th>
                <th className="pb-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {[
                { ip: "103.24.xx.xx", country: "China", type: "SYN Flood", requests: "23,421", status: "Blocked" },
                { ip: "185.12.xx.xx", country: "Russia", type: "UDP Flood", requests: "18,743", status: "Blocked" },
                { ip: "72.53.xx.xx", country: "United States", type: "HTTP Flood", requests: "12,382", status: "Blocked" },
                { ip: "91.134.xx.xx", country: "France", type: "TCP Connect", requests: "9,834", status: "Blocked" },
                { ip: "45.76.xx.xx", country: "Germany", type: "ICMP Flood", requests: "7,245", status: "Blocked" }
              ].map((item, index) => (
                <tr key={index} className="border-b border-white/5">
                  <td className="py-3">{item.ip}</td>
                  <td className="py-3">{item.country}</td>
                  <td className="py-3">{item.type}</td>
                  <td className="py-3">{item.requests}</td>
                  <td className="py-3">
                    <span className="bg-green-500/20 text-green-400 text-xs py-1 px-2 rounded-full">
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

interface StatCardProps {
  title: string;
  value: string;
  change: string;
  trend: 'up' | 'down';
  icon: React.ReactNode;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, change, trend, icon }) => {
  return (
    <div className="glass-dark p-4 rounded-lg">
      <div className="flex justify-between mb-2">
        <div className="p-2 rounded-lg bg-white/10">
          {icon}
        </div>
        <div className={`flex items-center ${trend === 'up' ? 'text-green-400' : 'text-red-400'}`}>
          <span className="text-xs font-medium">{change}</span>
          {trend === 'up' ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
        </div>
      </div>
      <p className="text-sm text-white/70">{title}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
};

export default ThreatAnalytics;
