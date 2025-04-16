
import React, { useState } from "react";
import { Shield, Wifi, Cpu, Database, PlayCircle, PauseCircle, Settings, Layers, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";

const LayerControls: React.FC = () => {
  const [activeLayer3, setActiveLayer3] = useState(true);
  const [activeLayer4, setActiveLayer4] = useState(true);
  const [activeLayer7, setActiveLayer7] = useState(true);
  
  const toggleLayer = (layer: string) => {
    switch(layer) {
      case 'layer3':
        setActiveLayer3(!activeLayer3);
        break;
      case 'layer4':
        setActiveLayer4(!activeLayer4);
        break;
      case 'layer7':
        setActiveLayer7(!activeLayer7);
        break;
    }
  };
  
  return (
    <div className="grid gap-6 h-full">
      <Tabs defaultValue="layer7" className="h-full">
        <TabsList className="grid grid-cols-3 mb-4">
          <TabsTrigger value="layer7" className="flex items-center gap-2">
            <Database size={16} />
            Layer 7
          </TabsTrigger>
          <TabsTrigger value="layer4" className="flex items-center gap-2">
            <Wifi size={16} />
            Layer 4
          </TabsTrigger>
          <TabsTrigger value="layer3" className="flex items-center gap-2">
            <Cpu size={16} />
            Layer 3
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="layer7" className="h-[calc(100%-60px)]">
          <div className="grid gap-4 h-full">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">HTTP/HTTPS Protection</h3>
                <p className="text-sm text-white/70">Application layer DDoS protection</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={activeLayer7 ? "default" : "outline"} className="bg-green-600">
                  {activeLayer7 ? 'Active' : 'Inactive'}
                </Badge>
                <Button size="sm" variant="ghost" onClick={() => toggleLayer('layer7')}>
                  {activeLayer7 ? <PauseCircle size={18} /> : <PlayCircle size={18} />}
                </Button>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Active Rules</h4>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Rate limiting: 100 req/s
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Bot detection
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    WAF rules
                  </li>
                </ul>
              </div>
              
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Statistics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Blocked requests:</span>
                    <span className="text-red-400">23,456</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Average response:</span>
                    <span>78ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Traffic cleansed:</span>
                    <span className="text-green-400">98.7%</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-1 gap-4 mt-2">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">ML/DL Protection</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-purple-600/30">ML</Badge>
                    <span className="text-sm">Anomaly detection</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-blue-600/30">DL</Badge>
                    <span className="text-sm">Pattern recognition</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-purple-600/30">ML</Badge>
                    <span className="text-sm">Rate pattern learning</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="bg-blue-600/30">DL</Badge>
                    <span className="text-sm">Behavior analysis</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="layer4" className="h-[calc(100%-60px)]">
          <div className="grid gap-4 h-full">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">TCP/UDP Protection</h3>
                <p className="text-sm text-white/70">Transport layer DDoS protection</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={activeLayer4 ? "default" : "outline"} className="bg-green-600">
                  {activeLayer4 ? 'Active' : 'Inactive'}
                </Badge>
                <Button size="sm" variant="ghost" onClick={() => toggleLayer('layer4')}>
                  {activeLayer4 ? <PauseCircle size={18} /> : <PlayCircle size={18} />}
                </Button>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">TCP Protection</h4>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    SYN flood protection
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    TCP connection validation
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    TCP state tracking
                  </li>
                </ul>
              </div>
              
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">UDP Protection</h4>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Amplification protection
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Rate limiting
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    UDP flood mitigation
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="grid grid-cols-1 gap-4 mt-2">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Active Mitigation</h4>
                <div className="flex items-center justify-between">
                  <div>
                    <span className="text-sm">Current scrubbing capacity:</span>
                    <span className="ml-2 text-green-400 font-medium">10 Tbps</span>
                  </div>
                  <Button size="sm" variant="outline" className="flex items-center gap-2">
                    <RefreshCw size={14} />
                    Update Capacity
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="layer3" className="h-[calc(100%-60px)]">
          <div className="grid gap-4 h-full">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Network Layer Protection</h3>
                <p className="text-sm text-white/70">IP-based DDoS protection</p>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={activeLayer3 ? "default" : "outline"} className="bg-green-600">
                  {activeLayer3 ? 'Active' : 'Inactive'}
                </Badge>
                <Button size="sm" variant="ghost" onClick={() => toggleLayer('layer3')}>
                  {activeLayer3 ? <PauseCircle size={18} /> : <PlayCircle size={18} />}
                </Button>
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">IP Protection</h4>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    ICMP flood protection
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    IP fragment protection
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Spoofed IP blocking
                  </li>
                </ul>
              </div>
              
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">BGP Protection</h4>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Anycast routing
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    Blackhole routing
                  </li>
                  <li className="flex items-center gap-2">
                    <Shield size={14} className="text-green-400" />
                    FlowSpec rules
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="grid grid-cols-1 gap-4 mt-2">
              <div className="bg-black/20 p-4 rounded-lg">
                <h4 className="font-medium mb-2">Global Network Status</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-white/70">Edge Locations:</span>
                    <span className="ml-2 text-white">250+</span>
                  </div>
                  <div>
                    <span className="text-white/70">Network Capacity:</span>
                    <span className="ml-2 text-white">175 Tbps</span>
                  </div>
                  <div>
                    <span className="text-white/70">Average Latency:</span>
                    <span className="ml-2 text-white">10ms</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default LayerControls;
