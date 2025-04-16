import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, ShieldAlert, Activity, Server, Clock, Database, Shield } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { toast } from "@/components/ui/use-toast";
import { Link } from "react-router-dom";

// API types
interface NetworkMetrics {
  packets_per_second: number;
  bytes_per_second: number;
  connections_per_second: number;
  request_distribution: Record<string, number>;
}

interface ThreatInfo {
  source_ip: string;
  timestamp: string;
  layer3_attack: string | null;
  layer4_attack: string | null;
  layer7_attack: string | null;
  threat_level: "Low" | "Medium" | "High" | "Critical";
  mitigation_action: string;
  geo_location: string | null;
  request_rate: number | null;
  confidence_score: number;
  is_known_attacker: boolean;
}

interface DashboardStats {
  total_traffic: number;
  blocked_attacks: number;
  active_mitigations: number;
  current_threats: ThreatInfo[];
  network_metrics: NetworkMetrics;
  top_attack_sources: [string, number][];
  top_attack_types: [string, number][];
  attack_trend: [string, number][];
}

interface ApiResponse<T> {
  success: boolean;
  message: string;
  data: T | null;
}

// Fetch dashboard stats from API
const fetchStats = async (): Promise<DashboardStats> => {
  try {
    const response = await fetch("http://localhost:6868/api/stats");
    if (!response.ok) {
      throw new Error("Failed to fetch stats");
    }
    const data: ApiResponse<DashboardStats> = await response.json();
    if (!data.success || !data.data) {
      throw new Error(data.message || "Failed to fetch stats");
    }
    return data.data;
  } catch (error) {
    console.error("Error fetching stats:", error);
    // Return mock data for development
    return {
      total_traffic: 1024 * 1024 * 50, // 50 MB
      blocked_attacks: 157,
      active_mitigations: 23,
      current_threats: [
        {
          source_ip: "192.168.1.1",
          timestamp: new Date().toISOString(),
          layer3_attack: null,
          layer4_attack: "SynFlood",
          layer7_attack: null,
          threat_level: "High",
          mitigation_action: "Block",
          geo_location: "Unknown",
          request_rate: 500,
          confidence_score: 0.92,
          is_known_attacker: true
        }
      ],
      network_metrics: {
        packets_per_second: 5000,
        bytes_per_second: 1024 * 1024 * 2, // 2 MB/s
        connections_per_second: 150,
        request_distribution: {
          "GET": 70,
          "POST": 20,
          "OTHER": 10
        }
      },
      top_attack_sources: [
        ["192.168.1.1", 50],
        ["192.168.1.2", 30],
        ["192.168.1.3", 20]
      ],
      top_attack_types: [
        ["L4: SynFlood", 60],
        ["L7: HttpFlood", 25],
        ["L3: FragmentationAttack", 15]
      ],
      attack_trend: [
        [new Date(Date.now() - 3600000).toISOString(), 20],
        [new Date(Date.now() - 1800000).toISOString(), 50],
        [new Date(Date.now()).toISOString(), 30]
      ]
    };
  }
};

// Format bytes to human-readable format
const formatBytes = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

// Get color based on threat level
const getThreatLevelColor = (level: string): string => {
  switch (level) {
    case "Low":
      return "bg-yellow-500";
    case "Medium":
      return "bg-orange-500";
    case "High":
      return "bg-red-500";
    case "Critical":
      return "bg-purple-700";
    default:
      return "bg-gray-500";
  }
};

const Dashboard = () => {
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds
  
  const { data: stats, error, isLoading, refetch } = useQuery({
    queryKey: ["dashboardStats"],
    queryFn: fetchStats,
    refetchInterval: refreshInterval,
  });
  
  useEffect(() => {
    if (error) {
      toast({
        title: "Error",
        description: "Failed to fetch dashboard data. Using mock data.",
        variant: "destructive",
      });
    }
  }, [error]);
  
  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
          <Shield className="mr-2" /> Kyaro Anti-DDoS Dashboard
        </h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => refetch()}>
            <Clock className="mr-2 h-4 w-4" /> Refresh
          </Button>
          <Link to="/configuration">
            <Button>
              <Server className="mr-2 h-4 w-4" /> Configuration
            </Button>
          </Link>
        </div>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-500">Loading dashboard data...</p>
          </div>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Traffic</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatBytes(stats?.total_traffic || 0)}</div>
                <p className="text-xs text-gray-500 mt-1">
                  {(stats?.network_metrics.bytes_per_second ? formatBytes(stats.network_metrics.bytes_per_second) : '0 B') + '/s'}
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Blocked Attacks</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats?.blocked_attacks || 0}</div>
                <p className="text-xs text-gray-500 mt-1">
                  {stats?.active_mitigations || 0} active mitigations
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Network Load</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats?.network_metrics.packets_per_second || 0} pps</div>
                <p className="text-xs text-gray-500 mt-1">
                  {stats?.network_metrics.connections_per_second || 0} connections/s
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">Current Threats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats?.current_threats.length || 0}</div>
                <div className="flex mt-1">
                  {stats?.current_threats.length ? (
                    <Link to="/threats">
                      <Badge variant="destructive" className="cursor-pointer">
                        View Threats
                      </Badge>
                    </Link>
                  ) : (
                    <Badge variant="outline" className="text-xs text-gray-500">No active threats</Badge>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
          
          {/* Alerts */}
          {stats?.current_threats.length ? (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Active Threats Detected</AlertTitle>
              <AlertDescription>
                There are {stats.current_threats.length} active threats being mitigated. 
                <Link to="/threats" className="ml-2 underline">
                  View details
                </Link>
              </AlertDescription>
            </Alert>
          ) : null}
          
          {/* Tabbed Content */}
          <Tabs defaultValue="overview" className="mb-6">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="threats">Threats</TabsTrigger>
              <TabsTrigger value="traffic">Traffic</TabsTrigger>
            </TabsList>
            
            <TabsContent value="overview">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Top Attack Sources</CardTitle>
                    <CardDescription>IP addresses with the most attack attempts</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {stats?.top_attack_sources.length ? (
                      <div className="space-y-4">
                        {stats.top_attack_sources.slice(0, 5).map(([ip, count], index) => (
                          <div key={index}>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">{ip}</span>
                              <span className="text-sm text-gray-500">{count} attacks</span>
                            </div>
                            <Progress value={(count / stats.top_attack_sources[0][1]) * 100} />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500">No attack data available</div>
                    )}
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Top Attack Types</CardTitle>
                    <CardDescription>Most common attack vectors detected</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {stats?.top_attack_types.length ? (
                      <div className="space-y-4">
                        {stats.top_attack_types.slice(0, 5).map(([type, count], index) => (
                          <div key={index}>
                            <div className="flex justify-between mb-1">
                              <span className="text-sm font-medium">{type}</span>
                              <span className="text-sm text-gray-500">{count} instances</span>
                            </div>
                            <Progress value={(count / stats.top_attack_types[0][1]) * 100} />
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-6 text-gray-500">No attack type data available</div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            <TabsContent value="threats">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Threats</CardTitle>
                  <CardDescription>Latest detected threats and their status</CardDescription>
                </CardHeader>
                <CardContent>
                  {stats?.current_threats.length ? (
                    <div className="space-y-4">
                      {stats.current_threats.slice(0, 5).map((threat, index) => (
                        <div key={index} className="flex items-center p-3 border rounded-lg">
                          <div className={`w-3 h-3 rounded-full mr-3 ${getThreatLevelColor(threat.threat_level)}`}></div>
                          <div className="flex-1">
                            <div className="flex justify-between">
                              <span className="font-medium">{threat.source_ip}</span>
                              <Badge variant={threat.is_known_attacker ? "destructive" : "secondary"}>
                                {threat.is_known_attacker ? "Known Attacker" : "New Threat"}
                              </Badge>
                            </div>
                            <div className="text-sm text-gray-500 mt-1">
                              Attack: {threat.layer3_attack || threat.layer4_attack || threat.layer7_attack || "Unknown"}
                            </div>
                            <div className="flex justify-between mt-1 text-xs">
                              <span>Action: {threat.mitigation_action}</span>
                              <span>Confidence: {(threat.confidence_score * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        </div>
                      ))}
                      {stats.current_threats.length > 5 && (
                        <div className="text-center mt-4">
                          <Link to="/threats">
                            <Button variant="outline" size="sm">View All Threats</Button>
                          </Link>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-12 text-gray-500">
                      <ShieldAlert className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                      <p>No threats currently detected</p>
                      <p className="text-sm mt-2">Your network is secure</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="traffic">
              <Card>
                <CardHeader>
                  <CardTitle>Traffic Analysis</CardTitle>
                  <CardDescription>Current network traffic statistics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-lg font-medium mb-4">Request Distribution</h3>
                      {stats?.network_metrics.request_distribution && Object.keys(stats.network_metrics.request_distribution).length ? (
                        <div className="space-y-4">
                          {Object.entries(stats.network_metrics.request_distribution).map(([method, count], index) => (
                            <div key={index}>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium">{method}</span>
                                <span className="text-sm text-gray-500">{count} requests</span>
                              </div>
                              <Progress 
                                value={(count / Object.values(stats.network_metrics.request_distribution).reduce((a, b) => a + b, 0)) * 100} 
                              />
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-6 text-gray-500">No request data available</div>
                      )}
                    </div>
                    
                    <div>
                      <h3 className="text-lg font-medium mb-4">System Performance</h3>
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Bandwidth Utilization</span>
                            <span className="text-sm text-gray-500">
                              {stats?.network_metrics.bytes_per_second 
                                ? formatBytes(stats.network_metrics.bytes_per_second) + '/s'
                                : 'Unknown'}
                            </span>
                          </div>
                          <Progress value={stats?.network_metrics.bytes_per_second 
                            ? Math.min((stats.network_metrics.bytes_per_second / (1024 * 1024 * 10)) * 100, 100)
                            : 0} 
                          />
                        </div>
                        
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Packet Rate</span>
                            <span className="text-sm text-gray-500">
                              {stats?.network_metrics.packets_per_second || 0} pps
                            </span>
                          </div>
                          <Progress value={stats?.network_metrics.packets_per_second 
                            ? Math.min((stats.network_metrics.packets_per_second / 10000) * 100, 100)
                            : 0} 
                          />
                        </div>
                        
                        <div>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Connection Rate</span>
                            <span className="text-sm text-gray-500">
                              {stats?.network_metrics.connections_per_second || 0} conn/s
                            </span>
                          </div>
                          <Progress value={stats?.network_metrics.connections_per_second 
                            ? Math.min((stats.network_metrics.connections_per_second / 500) * 100, 100) 
                            : 0} 
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default Dashboard;
