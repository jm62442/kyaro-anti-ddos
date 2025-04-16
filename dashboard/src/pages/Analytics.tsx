import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Activity, Clock, FileBarChart, ArrowUpRight, BarChart3, PieChart } from "lucide-react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useToast } from "@/components/ui/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Mock data for charts
const generateMockData = () => {
  const now = new Date();
  const hourlyData = Array.from({ length: 24 }, (_, i) => {
    const hour = new Date(now);
    hour.setHours(now.getHours() - 23 + i);
    
    return {
      timestamp: hour.toISOString(),
      attacks: Math.floor(Math.random() * 50),
      traffic: Math.random() * 100000000, // Random traffic in bytes
    };
  });
  
  return {
    hourlyData,
    attackTypes: [
      { type: "SYN Flood", count: 325 },
      { type: "HTTP Flood", count: 210 },
      { type: "UDP Flood", count: 180 },
      { type: "DNS Amplification", count: 120 },
      { type: "ICMP Flood", count: 95 },
    ],
    sourceCountries: [
      { country: "Unknown", count: 450 },
      { country: "China", count: 210 },
      { country: "Russia", count: 180 },
      { country: "United States", count: 150 },
      { country: "Brazil", count: 95 },
    ],
    mitigationEffectiveness: {
      success: 85,
      partial: 10,
      failed: 5,
    },
    trafficStats: {
      total: 15.7e9, // 15.7 GB
      blocked: 4.3e9, // 4.3 GB
      legitimate: 11.4e9, // 11.4 GB
      peak: 250e6, // 250 MB/s
    },
  };
};

// Fetch analytics data
const fetchAnalytics = async (timeRange: string) => {
  try {
    // In a real implementation, we would fetch from the API with the timeRange parameter
    // const response = await fetch(`http://localhost:6868/api/analytics?timeRange=${timeRange}`);
    // return await response.json();
    
    // For now, just return mock data
    return generateMockData();
  } catch (error) {
    console.error("Error fetching analytics:", error);
    return generateMockData();
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

// Component for a stat card with trend
const StatCard = ({ title, value, trend, icon: Icon, trendDirection = "up" }: { 
  title: string, 
  value: string, 
  trend: string, 
  icon: any,
  trendDirection?: "up" | "down"
}) => (
  <Card>
    <CardHeader className="pb-2">
      <div className="flex justify-between items-start">
        <CardTitle className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </div>
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">{value}</div>
      <div className="mt-1 flex items-center text-xs">
        <ArrowUpRight className={`h-3 w-3 mr-1 ${
          trendDirection === "up" ? "text-green-500" : "text-red-500 transform rotate-180"
        }`} />
        <span className={trendDirection === "up" ? "text-green-500" : "text-red-500"}>{trend}</span>
      </div>
    </CardContent>
  </Card>
);

const Analytics = () => {
  const [timeRange, setTimeRange] = useState("24h");
  const { toast } = useToast();
  
  const { data, error, isLoading, refetch } = useQuery({
    queryKey: ["analytics", timeRange],
    queryFn: () => fetchAnalytics(timeRange),
  });
  
  if (error) {
    toast({
      title: "Error",
      description: "Failed to fetch analytics data. Using mock data.",
      variant: "destructive",
    });
  }
  
  return (
    <div className="container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Link to="/" className="mr-4">
            <Button variant="outline" size="icon">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white flex items-center">
            <Activity className="mr-2" /> Analytics & Reports
          </h1>
        </div>
        <div className="flex items-center gap-4">
          <Select defaultValue="24h" onValueChange={setTimeRange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last Hour</SelectItem>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={() => refetch()}>
            <Clock className="mr-2 h-4 w-4" /> Refresh
          </Button>
          <Button variant="outline">
            <FileBarChart className="mr-2 h-4 w-4" /> Export Report
          </Button>
        </div>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
            <p className="mt-4 text-gray-500">Loading analytics data...</p>
          </div>
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <StatCard 
              title="Total Attacks Blocked" 
              value={data?.hourlyData.reduce((sum, hour) => sum + hour.attacks, 0).toString() || "0"} 
              trend="+12.5% from last period" 
              icon={BarChart3} 
              trendDirection="down"
            />
            <StatCard 
              title="Total Traffic" 
              value={formatBytes(data?.trafficStats.total || 0)} 
              trend="+8.2% from last period" 
              icon={Activity}
              trendDirection="up"
            />
            <StatCard 
              title="Traffic Blocked" 
              value={formatBytes(data?.trafficStats.blocked || 0)} 
              trend="+15.3% from last period" 
              icon={Activity}
              trendDirection="down"
            />
            <StatCard 
              title="Peak Traffic" 
              value={formatBytes(data?.trafficStats.peak || 0) + "/s"} 
              trend="+5.1% from last period" 
              icon={Activity}
              trendDirection="up"
            />
          </div>
          
          <Tabs defaultValue="attacks" className="mb-6">
            <TabsList>
              <TabsTrigger value="attacks">Attack Analytics</TabsTrigger>
              <TabsTrigger value="traffic">Traffic Analytics</TabsTrigger>
              <TabsTrigger value="geo">Geographic Analytics</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
            </TabsList>
            
            {/* Attack Analytics Tab */}
            <TabsContent value="attacks">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Attack Distribution</CardTitle>
                    <CardDescription>Types of attacks detected and blocked</CardDescription>
                  </CardHeader>
                  <CardContent className="min-h-[300px]">
                    <div className="flex flex-col space-y-4">
                      {data?.attackTypes.map((attack, index) => (
                        <div key={index}>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">{attack.type}</span>
                            <span className="text-sm text-gray-500">{attack.count} attacks</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                            <div 
                              className="bg-primary h-2.5 rounded-full" 
                              style={{ width: `${(attack.count / data.attackTypes[0].count) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Attack Timeline</CardTitle>
                    <CardDescription>Number of attacks over time</CardDescription>
                  </CardHeader>
                  <CardContent className="min-h-[300px]">
                    <div className="flex items-end space-x-1 h-60">
                      {data?.hourlyData.map((hour, index) => (
                        <div 
                          key={index} 
                          className="flex-1 bg-primary rounded-t hover:opacity-80 transition-opacity"
                          style={{ 
                            height: `${(hour.attacks / Math.max(...data.hourlyData.map(h => h.attacks))) * 100}%`,
                            minHeight: hour.attacks > 0 ? '4px' : '0',
                          }}
                          title={`${new Date(hour.timestamp).toLocaleTimeString()}: ${hour.attacks} attacks`}
                        />
                      ))}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-gray-500">
                      <span>{new Date(data?.hourlyData[0].timestamp || "").toLocaleTimeString()}</span>
                      <span>{new Date(data?.hourlyData[data.hourlyData.length-1].timestamp || "").toLocaleTimeString()}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Traffic Analytics Tab */}
            <TabsContent value="traffic">
              <div className="grid grid-cols-1 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Traffic Overview</CardTitle>
                    <CardDescription>Network traffic over time</CardDescription>
                  </CardHeader>
                  <CardContent className="min-h-[300px]">
                    <div className="flex items-end space-x-1 h-60">
                      {data?.hourlyData.map((hour, index) => (
                        <div key={index} className="flex-1 flex flex-col items-stretch">
                          <div 
                            className="bg-red-500 rounded-t"
                            style={{ 
                              height: `${(hour.attacks * 100000 / Math.max(...data.hourlyData.map(h => h.traffic))) * 100}%`,
                              minHeight: hour.attacks > 0 ? '4px' : '0',
                            }}
                            title={`Blocked: ${formatBytes(hour.attacks * 100000)}`}
                          />
                          <div 
                            className="bg-green-500 rounded-t mt-px"
                            style={{ 
                              height: `${((hour.traffic - hour.attacks * 100000) / Math.max(...data.hourlyData.map(h => h.traffic))) * 100}%`,
                              minHeight: (hour.traffic - hour.attacks * 100000) > 0 ? '4px' : '0',
                            }}
                            title={`Legitimate: ${formatBytes(hour.traffic - hour.attacks * 100000)}`}
                          />
                        </div>
                      ))}
                    </div>
                    <div className="flex justify-between mt-2 text-xs text-gray-500">
                      <span>{new Date(data?.hourlyData[0].timestamp || "").toLocaleTimeString()}</span>
                      <span>{new Date(data?.hourlyData[data.hourlyData.length-1].timestamp || "").toLocaleTimeString()}</span>
                    </div>
                    <div className="flex items-center justify-center mt-4 space-x-6">
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                        <span className="text-sm">Legitimate Traffic</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                        <span className="text-sm">Blocked Traffic</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Geographic Analytics Tab */}
            <TabsContent value="geo">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Attack Sources by Country</CardTitle>
                    <CardDescription>Geographic distribution of attack sources</CardDescription>
                  </CardHeader>
                  <CardContent className="min-h-[300px]">
                    <div className="flex flex-col space-y-4">
                      {data?.sourceCountries.map((country, index) => (
                        <div key={index}>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">{country.country}</span>
                            <span className="text-sm text-gray-500">{country.count} attacks</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                            <div 
                              className="bg-primary h-2.5 rounded-full" 
                              style={{ width: `${(country.count / data.sourceCountries[0].count) * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>Global Attack Map</CardTitle>
                    <CardDescription>Visualized attack sources on world map</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-center h-[300px] bg-gray-100 dark:bg-gray-800 rounded-md">
                      <div className="text-center text-gray-500">
                        <PieChart className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                        <p>Interactive map visualization would appear here</p>
                        <p className="text-sm mt-2">Showing attack sources globally</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Performance Tab */}
            <TabsContent value="performance">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Mitigation Effectiveness</CardTitle>
                    <CardDescription>Success rate of mitigation strategies</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-center h-[300px]">
                      <div className="w-48 h-48 rounded-full border-8 border-gray-200 dark:border-gray-700 relative">
                        <div 
                          className="absolute top-0 left-0 w-full h-full rounded-full border-8 border-green-500 border-t-transparent border-r-transparent"
                          style={{ transform: `rotate(${data?.mitigationEffectiveness.success * 3.6}deg)` }}
                        ></div>
                        <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
                          <div className="text-center">
                            <div className="text-3xl font-bold">{data?.mitigationEffectiveness.success}%</div>
                            <div className="text-xs text-gray-500">Success Rate</div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-center mt-4 space-x-6">
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                        <span className="text-sm">Successful ({data?.mitigationEffectiveness.success}%)</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full mr-2"></div>
                        <span className="text-sm">Partial ({data?.mitigationEffectiveness.partial}%)</span>
                      </div>
                      <div className="flex items-center">
                        <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                        <span className="text-sm">Failed ({data?.mitigationEffectiveness.failed}%)</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader>
                    <CardTitle>System Resource Usage</CardTitle>
                    <CardDescription>CPU, memory, and bandwidth utilization</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">CPU Usage</span>
                          <span className="text-sm text-gray-500">42%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                          <div className="bg-blue-500 h-2.5 rounded-full" style={{ width: '42%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Memory Usage</span>
                          <span className="text-sm text-gray-500">1.2 GB / 8 GB</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                          <div className="bg-purple-500 h-2.5 rounded-full" style={{ width: '15%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Bandwidth Utilization</span>
                          <span className="text-sm text-gray-500">120 Mbps / 1 Gbps</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                          <div className="bg-yellow-500 h-2.5 rounded-full" style={{ width: '12%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Packet Processing</span>
                          <span className="text-sm text-gray-500">25,000 pps</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                          <div className="bg-green-500 h-2.5 rounded-full" style={{ width: '25%' }}></div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default Analytics;
