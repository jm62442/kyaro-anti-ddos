
import React, { useState } from "react";
import { Brain, BarChart, Share2, Settings, RefreshCw, Database, Save, ChevronRight } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const MLSettings: React.FC = () => {
  const [anomalyDetection, setAnomalyDetection] = useState(true);
  const [behaviorAnalysis, setBehaviorAnalysis] = useState(true);
  const [signatureMatching, setSignatureMatching] = useState(true);
  const [rateLimiting, setRateLimiting] = useState(true);
  
  const [sensitivityLevel, setSensitivityLevel] = useState(75);
  const [trainingFrequency, setTrainingFrequency] = useState(50);
  const [responseAggressiveness, setResponseAggressiveness] = useState(60);
  
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => setIsSaving(false), 1500);
  };

  return (
    <div className="grid gap-6 h-full p-1">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            <Brain className="mr-2" size={20} />
            ML/DL Protection Settings
          </h3>
          <p className="text-sm text-white/70">Configure machine learning and deep learning protection</p>
        </div>
        <Badge className="bg-purple-600">Advanced</Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-black/20 p-4 rounded-lg">
          <h4 className="font-medium mb-4">Protection Modules</h4>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h5 className="font-medium">Anomaly Detection</h5>
                <p className="text-xs text-white/70">Identify unusual traffic patterns</p>
              </div>
              <Switch checked={anomalyDetection} onCheckedChange={setAnomalyDetection} />
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h5 className="font-medium">Behavior Analysis</h5>
                <p className="text-xs text-white/70">Monitor client behavior patterns</p>
              </div>
              <Switch checked={behaviorAnalysis} onCheckedChange={setBehaviorAnalysis} />
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h5 className="font-medium">Signature Matching</h5>
                <p className="text-xs text-white/70">Compare with known attack patterns</p>
              </div>
              <Switch checked={signatureMatching} onCheckedChange={setSignatureMatching} />
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <h5 className="font-medium">Adaptive Rate Limiting</h5>
                <p className="text-xs text-white/70">Automatically adjust thresholds</p>
              </div>
              <Switch checked={rateLimiting} onCheckedChange={setRateLimiting} />
            </div>
          </div>
        </div>
        
        <div className="bg-black/20 p-4 rounded-lg">
          <h4 className="font-medium mb-4">Sensitivity & Training</h4>
          
          <div className="space-y-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Sensitivity Level</span>
                <span className="text-sm font-medium">{sensitivityLevel}%</span>
              </div>
              <Slider 
                value={[sensitivityLevel]} 
                onValueChange={(value) => setSensitivityLevel(value[0])} 
                max={100} 
                step={1}
              />
              <p className="text-xs text-white/70 mt-1">Higher sensitivity may lead to more false positives</p>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Model Training Frequency</span>
                <span className="text-sm font-medium">{trainingFrequency}%</span>
              </div>
              <Slider 
                value={[trainingFrequency]} 
                onValueChange={(value) => setTrainingFrequency(value[0])} 
                max={100} 
                step={1}
              />
              <p className="text-xs text-white/70 mt-1">How often the model retrains with new data</p>
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">Response Aggressiveness</span>
                <span className="text-sm font-medium">{responseAggressiveness}%</span>
              </div>
              <Slider 
                value={[responseAggressiveness]} 
                onValueChange={(value) => setResponseAggressiveness(value[0])} 
                max={100} 
                step={1}
              />
              <p className="text-xs text-white/70 mt-1">How aggressive the system responds to detected threats</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 gap-4">
        <div className="bg-black/20 p-4 rounded-lg">
          <h4 className="font-medium mb-4">ML/DL Engine Status</h4>
          
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-black/30 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-white/70">Algorithm</span>
                <Badge className="bg-blue-600/50 text-xs">Neural Net</Badge>
              </div>
              <p className="text-sm font-medium">Deep Learning v2.3</p>
            </div>
            
            <div className="bg-black/30 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-white/70">Last Training</span>
                <Badge className="bg-green-600/50 text-xs">Recent</Badge>
              </div>
              <p className="text-sm font-medium">2 hours ago</p>
            </div>
            
            <div className="bg-black/30 p-3 rounded-lg">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-white/70">Accuracy</span>
                <Badge className="bg-purple-600/50 text-xs">High</Badge>
              </div>
              <p className="text-sm font-medium">98.7% detection</p>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <Button variant="outline" size="sm" className="flex items-center gap-2">
              <RefreshCw size={14} />
              Retrain Model
            </Button>
            
            <Button variant="outline" size="sm" className="flex items-center gap-2">
              <Database size={14} />
              View Training Data
            </Button>
            
            <Button 
              onClick={handleSave}
              className="flex items-center gap-2 bg-blue-600"
              disabled={isSaving}
            >
              {isSaving ? <RefreshCw className="animate-spin" size={14} /> : <Save size={14} />}
              {isSaving ? "Saving..." : "Save Settings"}
            </Button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gradient-to-br from-purple-600/20 to-blue-600/20 p-4 rounded-lg flex items-center justify-between">
          <div>
            <h4 className="font-medium">Integration Code</h4>
            <p className="text-xs text-white/70">Get code to integrate ML/DL protection</p>
          </div>
          <Button variant="outline" size="sm" className="flex items-center gap-1">
            View <ChevronRight size={14} />
          </Button>
        </div>
        
        <div className="bg-gradient-to-br from-blue-600/20 to-green-600/20 p-4 rounded-lg flex items-center justify-between">
          <div>
            <h4 className="font-medium">API Documentation</h4>
            <p className="text-xs text-white/70">Access ML/DL API documentation</p>
          </div>
          <Button variant="outline" size="sm" className="flex items-center gap-1">
            View <ChevronRight size={14} />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default MLSettings;
