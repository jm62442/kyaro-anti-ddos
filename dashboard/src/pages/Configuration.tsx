import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Save, RotateCw, Shield, Database, Network, Cpu, Brain } from "lucide-react";
import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/components/ui/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrayInput } from "@/components/ArrayInput";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Import config mapper and API functions
import { 
  ConfigGroup, 
  ConfigItem, 
  KyaroConfig,
  mapConfigToGroups, 
  mapGroupsToConfig, 
  generateDefaultConfig 
} from "@/lib/config-mapper";
import { fetchConfig, saveConfig } from "@/lib/api";

// Main Configuration component
const Configuration = () => {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  
  // State
  const [activeTab, setActiveTab] = useState("general");
  const [configGroups, setConfigGroups] = useState<ConfigGroup[]>([]);
  const [originalConfig, setOriginalConfig] = useState<KyaroConfig | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  // Fetch configuration from API
  const { isLoading, isError, error } = useQuery({
    queryKey: ["config"],
    queryFn: fetchConfig,
    onSuccess: (data) => {
      // On successful fetch, map the API data to our UI format
      setOriginalConfig(data);
      const groups = mapConfigToGroups(data);
      setConfigGroups(groups);
    },
    onError: (err) => {
      console.error("Error fetching config:", err);
      // If API fails, try to generate default config
      const defaultConfig = generateDefaultConfig();
      setOriginalConfig(defaultConfig);
      const groups = mapConfigToGroups(defaultConfig);
      setConfigGroups(groups);
    },
    retry: 1, // Only retry once
    refetchOnWindowFocus: false,
  });
  
  // Save configuration mutation
  const { isPending: isSaving, mutate: submitSaveConfig } = useMutation({
    mutationFn: (config: KyaroConfig) => saveConfig(config),
    onSuccess: (data) => {
      // Update state with new config from API
      setOriginalConfig(data);
      const groups = mapConfigToGroups(data);
      setConfigGroups(groups);
      setHasUnsavedChanges(false);
      
      // Show success toast
      toast({
        title: "Configuration Saved",
        description: "Your changes have been applied successfully.",
      });
      
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ["config"] });
    },
    onError: (err) => {
      console.error("Error saving config:", err);
      toast({
        title: "Error Saving Configuration",
        description: "Failed to save changes. Please try again.",
        variant: "destructive",
      });
    },
  });
  
  // Handle configuration item changes
  const handleConfigChange = (groupId: string, itemId: string, value: string | number | boolean | string[] | number[]) => {
    const updatedGroups = configGroups.map(group => {
      if (group.id !== groupId) return group;
      
      return {
        ...group,
        items: group.items.map(item => {
          if (item.id !== itemId) return item;
          return { ...item, value };
        }),
      };
    });
    
    setConfigGroups(updatedGroups);
    setHasUnsavedChanges(true);
  };
  
  // Handle save button click
  const handleSaveConfig = () => {
    if (!originalConfig) return;
    
    const updatedConfig = mapGroupsToConfig(configGroups, originalConfig);
    submitSaveConfig(updatedConfig);
  };
  
  // Handle reset button click
  const handleResetConfig = () => {
    if (!originalConfig) return;
    
    const groups = mapConfigToGroups(originalConfig);
    setConfigGroups(groups);
    setHasUnsavedChanges(false);
    
    toast({
      title: "Configuration Reset",
      description: "All changes have been reverted.",
    });
  };
  
  // Render input element based on config item type
  const renderConfigInput = (group: ConfigGroup, item: ConfigItem) => {
    const { id, type, value, options, min, max, step, unit } = item;
    
    switch (type) {
      case "boolean":
        return (
          <Switch
            checked={value as boolean}
            onCheckedChange={(checked) => handleConfigChange(group.id, id, checked)}
          />
        );
        
      case "select":
        return (
          <Select
            value={value as string}
            onValueChange={(val) => handleConfigChange(group.id, id, val)}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select option" />
            </SelectTrigger>
            <SelectContent>
              {options?.map((option) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
        
      case "number":
        // Use slider for more limited ranges, input for wider ranges
        if (typeof min === "number" && typeof max === "number" && (max - min) <= 10) {
          return (
            <div className="space-y-2">
              <Slider
                value={[value as number]}
                min={min}
                max={max}
                step={step || 1}
                onValueChange={(vals) => handleConfigChange(group.id, id, vals[0])}
              />
              <div className="text-sm text-muted-foreground">
                {value} {unit}
              </div>
            </div>
          );
        } else {
          return (
            <div className="flex items-center space-x-2">
              <Input
                type="number"
                value={value as number}
                onChange={(e) => handleConfigChange(group.id, id, Number(e.target.value))}
                min={min}
                max={max}
                step={step || 1}
              />
              {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
            </div>
          );
        }
        
      case "text":
        return (
          <Input
            type="text"
            value={value as string}
            onChange={(e) => handleConfigChange(group.id, id, e.target.value)}
          />
        );
        
      case "array":
        // Use ArrayInput component for handling arrays
        return (
          <ArrayInput
            value={value as string[] | number[]}
            onChange={(newValue) => handleConfigChange(group.id, id, newValue)}
            isNumeric={id === "blocked_ports"}
            placeholder={id === "blocked_countries" ? "Add country code..." : 
                        id === "blocked_ports" ? "Add port number..." : 
                        id === "blocked_http_methods" ? "Add HTTP method..." : 
                        "Add item..."}
          />
        );
        
      default:
        return <div>Unsupported type: {type}</div>;
    }
  };
  
  // Get icon based on category
  const getCategoryIcon = (categoryId: string) => {
    switch (categoryId) {
      case "general":
        return <Database className="h-4 w-4" />;
      case "layer3":
        return <Network className="h-4 w-4" />;
      case "layer4":
        return <Shield className="h-4 w-4" />;
      case "layer7":
        return <Cpu className="h-4 w-4" />;
      case "ml":
        return <Brain className="h-4 w-4" />;
      default:
        return <Shield className="h-4 w-4" />;
    }
  };
  
  // Render loading state
  if (isLoading) {
    return (
      <div className="container mx-auto py-6 space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold">Configuration</h1>
          <Link to="/dashboard">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
            </Button>
          </Link>
        </div>
        
        <Card>
          <CardHeader>
            <Skeleton className="h-8 w-3/4" />
            <Skeleton className="h-4 w-1/2" />
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {Array(5).fill(0).map((_, i) => (
                <div key={i} className="space-y-2">
                  <Skeleton className="h-5 w-1/3" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }
  
  // Render error state
  if (isError) {
    return (
      <div className="container mx-auto py-6 space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold">Configuration</h1>
          <Link to="/dashboard">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
            </Button>
          </Link>
        </div>
        
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>
            Failed to load configuration: {error instanceof Error ? error.message : 'Unknown error'}. 
            Using default configuration instead.
          </AlertDescription>
        </Alert>
        
        {/* Render with default config */}
      </div>
    );
  }
  
  return (
    <div className="container mx-auto py-6 space-y-6">
      {/* Header with back button and save/reset buttons */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Configuration</h1>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={handleResetConfig}
            disabled={!hasUnsavedChanges || isSaving}
          >
            <RotateCw className="mr-2 h-4 w-4" /> Reset
          </Button>
          <Button
            onClick={handleSaveConfig}
            disabled={!hasUnsavedChanges || isSaving}
          >
            {isSaving ? (
              <>Saving...</>
            ) : (
              <>
                <Save className="mr-2 h-4 w-4" /> Save Changes
              </>
            )}
          </Button>
          <Link to="/dashboard">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back
            </Button>
          </Link>
        </div>
      </div>

      {/* Tabs for different config sections */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-5">
          {configGroups.map((group) => (
            <TabsTrigger key={group.id} value={group.id} className="flex items-center gap-1">
              {getCategoryIcon(group.id)}
              <span>{group.name}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        {configGroups.map((group) => (
          <TabsContent key={group.id} value={group.id}>
            <Card>
              <CardHeader>
                <CardTitle>{group.name}</CardTitle>
                <CardDescription>{group.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6 md:grid-cols-2">
                  {group.items.map((item) => (
                    <div key={item.id} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <Label htmlFor={item.id} className="text-base">
                          {item.name}
                        </Label>
                        {item.type === "boolean" && renderConfigInput(group, item)}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {item.description}
                      </div>
                      {item.type !== "boolean" && (
                        <div className="pt-1">
                          {renderConfigInput(group, item)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default Configuration;
