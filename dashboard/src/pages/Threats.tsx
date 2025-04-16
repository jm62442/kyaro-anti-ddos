import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useToast } from "@/components/ui/use-toast";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Search, Shield, ShieldAlert, Ban, Clock, Globe, AlertTriangle } from "lucide-react";
import { Link } from "react-router-dom";

// API types
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

interface MitigationRule {
  rule_id: string;
  description: string;
  source_ip: string | null;
  source_network: string | null;
  destination_port: number | null;
  protocol: string | null;
  action: string;
  duration: number | null;
  is_active: boolean;
  created_at: string;
  modified_at: string;
}

interface ApiResponse<T> {
  success: boolean;
  message: string;
  data: T | null;
}

// API function to fetch threats
const fetchThreats = async (): Promise<ThreatInfo[]> => {
  try {
    const response = await fetch("http://localhost:6868/api/threats");
    if (!response.ok) {
      throw new Error("Failed to fetch threats");
    }
    const data: ApiResponse<ThreatInfo[]> = await response.json();
    if (!data.success || !data.data) {
      throw new Error(data.message || "Failed to fetch threats");
    }
    return data.data;
  } catch (error) {
    console.error("Error fetching threats:", error);
    // Return mock data for development
    return [
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
      },
      {
        source_ip: "10.0.0.15",
        timestamp: new Date(Date.now() - 300000).toISOString(),
        layer3_attack: "FragmentationAttack",
        layer4_attack: null,
        layer7_attack: null,
        threat_level: "Medium",
        mitigation_action: "RateLimit",
        geo_location: "Unknown",
        request_rate: 350,
        confidence_score: 0.78,
        is_known_attacker: false
      },
      {
        source_ip: "172.16.33.9",
        timestamp: new Date(Date.now() - 600000).toISOString(),
        layer3_attack: null,
        layer4_attack: null,
        layer7_attack: "HttpFlood",
        threat_level: "Critical",
        mitigation_action: "Block",
        geo_location: "Unknown",
        request_rate: 1200,
        confidence_score: 0.98,
        is_known_attacker: true
      }
    ];
  }
};

// API function to fetch mitigations
const fetchMitigations = async (): Promise<MitigationRule[]> => {
  try {
    const response = await fetch("http://localhost:6868/api/mitigations");
    if (!response.ok) {
      throw new Error("Failed to fetch mitigations");
    }
    const data: ApiResponse<MitigationRule[]> = await response.json();
    if (!data.success || !data.data) {
      throw new Error(data.message || "Failed to fetch mitigations");
    }
    return data.data;
  } catch (error) {
    console.error("Error fetching mitigations:", error);
    // Return mock data for development
    return [
      {
        rule_id: "r1",
        description: "Block known attacker",
        source_ip: "192.168.1.1",
        source_network: null,
        destination_port: null,
        protocol: null,
        action: "Block",
        duration: 3600,
        is_active: true,
        created_at: new Date(Date.now() - 1800000).toISOString(),
        modified_at: new Date(Date.now() - 1800000).toISOString()
      },
      {
        rule_id: "r2",
        description: "Rate limit suspicious traffic",
        source_ip: "10.0.0.15",
        source_network: null,
        destination_port: null,
        protocol: null,
        action: "RateLimit",
        duration: 1800,
        is_active: true,
        created_at: new Date(Date.now() - 300000).toISOString(),
        modified_at: new Date(Date.now() - 300000).toISOString()
      }
    ];
  }
};

// API function to block an IP
const blockIp = async (ip: string): Promise<void> => {
  try {
    const response = await fetch("http://localhost:6868/api/block", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ 
        ip,
        duration_seconds: 3600, // 1 hour
        reason: "Manually blocked via dashboard"
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to block IP");
    }
    const data: ApiResponse<null> = await response.json();
    if (!data.success) {
      throw new Error(data.message || "Failed to block IP");
    }
  } catch (error) {
    console.error("Error blocking IP:", error);
    throw error;
  }
};

// API function to unblock an IP
const unblockIp = async (ip: string): Promise<void> => {
  try {
    const response = await fetch("http://localhost:6868/api/unblock", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ip }),
    });
    if (!response.ok) {
      throw new Error("Failed to unblock IP");
    }
    const data: ApiResponse<null> = await response.json();
    if (!data.success) {
      throw new Error(data.message || "Failed to unblock IP");
    }
  } catch (error) {
    console.error("Error unblocking IP:", error);
    throw error;
  }
};

// Get color based on threat level
const getThreatLevelColor = (level: string): string => {
  switch (level) {
    case "Low":
      return "bg-yellow-500 text-white";
    case "Medium":
      return "bg-orange-500 text-white";
    case "High":
      return "bg-red-500 text-white";
    case "Critical":
      return "bg-purple-700 text-white";
    default:
      return "bg-gray-500 text-white";
  }
};

// Format date to readable format
const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleString();
};

// Calculate time ago
const getTimeAgo = (dateString: string): string => {
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return `${seconds} seconds ago`;
  
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minutes ago`;
  
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hours ago`;
  
  const days = Math.floor(hours / 24);
  return `${days} days ago`;
};

const Threats = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // Query for threats data
  const { 
    data: threats = [], 
    error: threatsError, 
    isLoading: threatsLoading,
    refetch: refetchThreats
  } = useQuery({
    queryKey: ["threats"],
    queryFn: fetchThreats,
  });
  
  // Query for mitigations data
  const { 
    data: mitigations = [], 
    error: mitigationsError, 
    isLoading: mitigationsLoading,
    refetch: refetchMitigations
  } = useQuery({
    queryKey: ["mitigations"],
    queryFn: fetchMitigations,
  });
  
  // Mutation for blocking IP
  const blockMutation = useMutation({
    mutationFn: blockIp,
    onSuccess: () => {
      toast({
        title: "IP Blocked",
        description: "The IP address has been successfully blocked.",
      });
      queryClient.invalidateQueries({ queryKey: ["threats"] });
      queryClient.invalidateQueries({ queryKey: ["mitigations"] });
    },
    onError: (error) => {
      toast({
        title: "Block Failed",
        description: error instanceof Error ? error.message : "Failed to block IP address",
        variant: "destructive",
      });
    },
  });
  
  // Mutation for unblocking IP
  const unblockMutation = useMutation({
    mutationFn: unblockIp,
    onSuccess: () => {
      toast({
        title: "IP Unblocked",
        description: "The IP address has been successfully unblocked.",
      });
      queryClient.invalidateQueries({ queryKey: ["threats"] });
      queryClient.invalidateQueries({ queryKey: ["mitigations"] });
    },
    onError: (error) => {
      toast({
        title: "Unblock Failed",
        description: error instanceof Error ? error.message : "Failed to unblock IP address",
        variant: "destructive",
      });
    },
  });
  
  // Filter threats based on search query
  const filteredThreats = threats.filter((threat) => 
    threat.source_ip.includes(searchQuery) || 
    (threat.layer3_attack && threat.layer3_attack.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (threat.layer4_attack && threat.layer4_attack.toLowerCase().includes(searchQuery.toLowerCase())) ||
    (threat.layer7_attack && threat.layer7_attack.toLowerCase().includes(searchQuery.toLowerCase())) ||
    threat.threat_level.toLowerCase().includes(searchQuery.toLowerCase()) ||
    threat.mitigation_action.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
  // Filter mitigations based on search query
  const filteredMitigations = mitigations.filter((rule) => 
    (rule.source_ip && rule.source_ip.includes(searchQuery)) || 
    (rule.source_network && rule.source_network.includes(searchQuery)) ||
    rule.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    rule.action.toLowerCase().includes(searchQuery.toLowerCase())
  );
  
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
            <ShieldAlert className="mr-2" /> Threat Management
          </h1>
        </div>
        <Button variant="outline" onClick={() => {
          refetchThreats();
          refetchMitigations();
        }}>
          <Clock className="mr-2 h-4 w-4" /> Refresh
        </Button>
      </div>
      
      {/* Search Bar */}
      <div className="relative mb-6">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
        <Input
          className="pl-10"
          placeholder="Search by IP, attack type, or mitigation action..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>
      
      {(threatsError || mitigationsError) && (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            {threatsError instanceof Error 
              ? threatsError.message 
              : mitigationsError instanceof Error 
                ? mitigationsError.message 
                : "Failed to fetch data"}
          </AlertDescription>
        </Alert>
      )}
      
      <Tabs defaultValue="active" className="mb-6">
        <TabsList>
          <TabsTrigger value="active">Active Threats</TabsTrigger>
          <TabsTrigger value="mitigations">Active Mitigations</TabsTrigger>
        </TabsList>
        
        <TabsContent value="active">
          <Card>
            <CardHeader>
              <CardTitle>Active Threats</CardTitle>
              <CardDescription>Currently detected threats and their details</CardDescription>
            </CardHeader>
            <CardContent>
              {threatsLoading ? (
                <div className="flex justify-center items-center h-64">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
                    <p className="mt-4 text-gray-500">Loading threats...</p>
                  </div>
                </div>
              ) : filteredThreats.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <ShieldAlert className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>No threats found</p>
                  {searchQuery && <p className="text-sm mt-2">Try adjusting your search query</p>}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>IP Address</TableHead>
                      <TableHead>Attack Type</TableHead>
                      <TableHead>Threat Level</TableHead>
                      <TableHead>Mitigation</TableHead>
                      <TableHead>Detected</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredThreats.map((threat, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">
                          <div className="flex items-center">
                            {threat.source_ip}
                            {threat.is_known_attacker && (
                              <Badge variant="destructive" className="ml-2">Known Attacker</Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          {threat.layer3_attack 
                            ? `L3: ${threat.layer3_attack}` 
                            : threat.layer4_attack 
                              ? `L4: ${threat.layer4_attack}` 
                              : threat.layer7_attack 
                                ? `L7: ${threat.layer7_attack}` 
                                : "Unknown"}
                        </TableCell>
                        <TableCell>
                          <Badge className={getThreatLevelColor(threat.threat_level)}>
                            {threat.threat_level}
                          </Badge>
                        </TableCell>
                        <TableCell>{threat.mitigation_action}</TableCell>
                        <TableCell title={formatDate(threat.timestamp)}>
                          {getTimeAgo(threat.timestamp)}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Button 
                              variant="destructive" 
                              size="sm"
                              onClick={() => blockMutation.mutate(threat.source_ip)}
                              disabled={blockMutation.isPending || unblockMutation.isPending}
                            >
                              <Ban className="h-4 w-4 mr-1" /> Block
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="mitigations">
          <Card>
            <CardHeader>
              <CardTitle>Active Mitigations</CardTitle>
              <CardDescription>Currently applied mitigation rules</CardDescription>
            </CardHeader>
            <CardContent>
              {mitigationsLoading ? (
                <div className="flex justify-center items-center h-64">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
                    <p className="mt-4 text-gray-500">Loading mitigations...</p>
                  </div>
                </div>
              ) : filteredMitigations.length === 0 ? (
                <div className="text-center py-12 text-gray-500">
                  <Shield className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                  <p>No active mitigations found</p>
                  {searchQuery && <p className="text-sm mt-2">Try adjusting your search query</p>}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Target</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Created</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredMitigations.map((rule) => (
                      <TableRow key={rule.rule_id}>
                        <TableCell className="font-medium">
                          {rule.source_ip || rule.source_network || (
                            rule.destination_port ? `Port: ${rule.destination_port}` : "Global Rule"
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant={rule.action === "Block" ? "destructive" : "secondary"}>
                            {rule.action}
                          </Badge>
                        </TableCell>
                        <TableCell>{rule.description}</TableCell>
                        <TableCell title={formatDate(rule.created_at)}>
                          {getTimeAgo(rule.created_at)}
                        </TableCell>
                        <TableCell>
                          {rule.source_ip && (
                            <Button 
                              variant="outline" 
                              size="sm"
                              onClick={() => unblockMutation.mutate(rule.source_ip!)}
                              disabled={unblockMutation.isPending || blockMutation.isPending}
                            >
                              Remove
                            </Button>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      {/* Manual IP Block Form */}
      <Card>
        <CardHeader>
          <CardTitle>Manual IP Control</CardTitle>
          <CardDescription>Manually block or unblock IP addresses</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row gap-4">
            <Input 
              placeholder="Enter IP address (e.g. 192.168.1.1)" 
              id="manual-ip"
              className="flex-1"
            />
            <div className="flex gap-2">
              <Button 
                variant="destructive"
                onClick={() => {
                  const ip = (document.getElementById("manual-ip") as HTMLInputElement).value;
                  if (ip) {
                    blockMutation.mutate(ip);
                  } else {
                    toast({
                      title: "Empty IP",
                      description: "Please enter an IP address",
                      variant: "destructive",
                    });
                  }
                }}
                disabled={blockMutation.isPending || unblockMutation.isPending}
              >
                <Ban className="mr-2 h-4 w-4" /> Block IP
              </Button>
              <Button 
                variant="outline"
                onClick={() => {
                  const ip = (document.getElementById("manual-ip") as HTMLInputElement).value;
                  if (ip) {
                    unblockMutation.mutate(ip);
                  } else {
                    toast({
                      title: "Empty IP",
                      description: "Please enter an IP address",
                      variant: "destructive",
                    });
                  }
                }}
                disabled={unblockMutation.isPending || blockMutation.isPending}
              >
                Unblock IP
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Threats;
