
import React from 'react';
import { Activity, Shield, AlertTriangle, Clock, Server, Globe, Cpu, Eye } from 'lucide-react';

interface DashboardProps {
  className?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ className = '' }) => {
  return (
    <div className={`text-white ${className}`}>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <StatusCard 
          title="Protection Status" 
          status="Active" 
          icon={<Shield className="text-green-400" />} 
          statusColor="text-green-400"
        />
        <StatusCard 
          title="Threats Detected" 
          status="12" 
          icon={<AlertTriangle className="text-yellow-400" />} 
          statusColor="text-yellow-400"
        />
      </div>

      <div className="glass-dark p-4 rounded-lg mb-6">
        <h3 className="text-lg font-medium mb-3 flex items-center">
          <Activity className="mr-2" size={20} />
          Attack Statistics
        </h3>
        <div className="grid grid-cols-4 gap-4">
          <StatCard label="Today" value="28" />
          <StatCard label="This Week" value="143" />
          <StatCard label="This Month" value="1,247" />
          <StatCard label="Total Blocked" value="24,892" />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3 flex items-center">
            <Globe className="mr-2" size={20} />
            Traffic Overview
          </h3>
          <div className="h-40 flex items-center justify-center">
            <div className="w-full h-full bg-gradient-to-t from-blue-500/20 to-transparent relative">
              {/* Simulated traffic graph */}
              <div className="absolute inset-0 overflow-hidden">
                <svg viewBox="0 0 100 40" className="w-full h-full">
                  <path
                    d="M0,30 Q10,5 20,25 T40,15 T60,30 T80,5 T100,20"
                    fill="none"
                    stroke="rgba(59, 130, 246, 0.8)"
                    strokeWidth="2"
                  />
                  <path
                    d="M0,35 Q15,20 30,30 T50,20 T70,25 T100,30"
                    fill="none"
                    stroke="rgba(139, 92, 246, 0.8)"
                    strokeWidth="2"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-dark p-4 rounded-lg">
          <h3 className="text-lg font-medium mb-3 flex items-center">
            <Server className="mr-2" size={20} />
            Server Load
          </h3>
          <div className="mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-xs">CPU Usage</span>
              <span className="text-xs font-bold text-blue-400">42%</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full" style={{ width: '42%' }}></div>
            </div>
          </div>
          <div className="mb-4">
            <div className="flex justify-between mb-1">
              <span className="text-xs">Memory Usage</span>
              <span className="text-xs font-bold text-purple-400">68%</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-2">
              <div className="bg-purple-500 h-2 rounded-full" style={{ width: '68%' }}></div>
            </div>
          </div>
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-xs">Network Load</span>
              <span className="text-xs font-bold text-green-400">23%</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-2">
              <div className="bg-green-500 h-2 rounded-full" style={{ width: '23%' }}></div>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-dark p-4 rounded-lg">
        <h3 className="text-lg font-medium mb-3 flex items-center">
          <Clock className="mr-2" size={20} />
          Recent Activities
        </h3>
        <div className="space-y-3">
          <ActivityItem 
            message="DDoS attack blocked from 103.24.xx.xx" 
            time="2 minutes ago" 
            type="blocked"
          />
          <ActivityItem 
            message="New client connected: Server #4212" 
            time="15 minutes ago"
            type="info"
          />
          <ActivityItem 
            message="System update completed successfully" 
            time="1 hour ago"
            type="success"
          />
          <ActivityItem 
            message="Suspicious activity detected from 185.12.xx.xx" 
            time="3 hours ago"
            type="warning"
          />
        </div>
      </div>
    </div>
  );
};

interface StatusCardProps {
  title: string;
  status: string;
  icon: React.ReactNode;
  statusColor: string;
}

const StatusCard: React.FC<StatusCardProps> = ({ title, status, icon, statusColor }) => {
  return (
    <div className="glass-dark p-4 rounded-lg flex items-center justify-between">
      <div className="flex items-center">
        <div className="p-3 rounded-lg bg-white/10 mr-3">
          {icon}
        </div>
        <div>
          <h3 className="font-medium">{title}</h3>
          <p className={`text-lg font-bold ${statusColor}`}>{status}</p>
        </div>
      </div>
      <div className="h-12 w-12 rounded-full bg-white/5 flex items-center justify-center">
        <div className={`h-3 w-3 rounded-full animate-pulse ${statusColor.replace('text', 'bg')}`}></div>
      </div>
    </div>
  );
};

interface StatCardProps {
  label: string;
  value: string;
}

const StatCard: React.FC<StatCardProps> = ({ label, value }) => {
  return (
    <div className="bg-white/5 p-3 rounded-lg text-center">
      <p className="text-xs text-white/70 mb-1">{label}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
};

interface ActivityItemProps {
  message: string;
  time: string;
  type: 'success' | 'warning' | 'blocked' | 'info';
}

const ActivityItem: React.FC<ActivityItemProps> = ({ message, time, type }) => {
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <Shield className="text-green-400" size={16} />;
      case 'warning':
        return <AlertTriangle className="text-yellow-400" size={16} />;
      case 'blocked':
        return <Eye className="text-red-400" size={16} />;
      case 'info':
        return <Cpu className="text-blue-400" size={16} />;
      default:
        return null;
    }
  };

  return (
    <div className="flex items-center bg-white/5 p-2 rounded-lg">
      <div className="p-2 rounded-full bg-white/10 mr-3">
        {getIcon()}
      </div>
      <div className="flex-1">
        <p className="text-sm">{message}</p>
        <p className="text-xs text-white/60">{time}</p>
      </div>
    </div>
  );
};

export default Dashboard;
