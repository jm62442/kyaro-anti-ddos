// Utility functions to map between backend API config and frontend UI config

// Backend config types - must match the Rust structures
export interface Layer3Config {
  enable_ip_blacklisting: boolean;
  enable_geo_blocking: boolean;
  enable_rate_limiting: boolean;
  max_packet_size: number;
  min_ttl: number;
  blocked_countries: string[];
  rate_limit_threshold: number;
}

export interface Layer4Config {
  enable_syn_protection: boolean;
  enable_udp_protection: boolean;
  enable_icmp_protection: boolean;
  syn_rate_threshold: number;
  udp_rate_threshold: number;
  icmp_rate_threshold: number;
  max_concurrent_connections: number;
  blocked_ports: number[];
}

export interface Layer7Config {
  enable_http_protection: boolean;
  enable_dns_protection: boolean;
  enable_ssl_protection: boolean;
  http_request_rate_threshold: number;
  blocked_http_methods: string[];
  challenge_suspicious_requests: boolean;
  bot_protection_level: number;
  waf_rules_enabled: boolean;
}

export interface MlConfig {
  enable_ml: boolean;
  training_interval_hours: number;
  detection_threshold: number;
  anomaly_sensitivity: number;
}

export interface KyaroConfig {
  layer3: Layer3Config;
  layer4: Layer4Config;
  layer7: Layer7Config;
  ml: MlConfig;
  log_level: string;
  data_retention_days: number;
  api_port: number;
}

// Frontend config types
export interface ConfigItem {
  id: string;
  name: string;
  value: string | number | boolean | string[] | number[];
  type: "text" | "number" | "boolean" | "select" | "multiselect" | "array";
  description: string;
  options?: string[];
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  category: "layer3" | "layer4" | "layer7" | "ml" | "general";
}

export interface ConfigGroup {
  id: string;
  name: string;
  description: string;
  items: ConfigItem[];
}

// Convert backend config to frontend config groups
export function mapConfigToGroups(config: KyaroConfig): ConfigGroup[] {
  return [
    {
      id: "general",
      name: "General Settings",
      description: "Basic configuration for Kyaro Anti-DDoS",
      items: [
        {
          id: "log_level",
          name: "Logging Level",
          value: config.log_level,
          type: "select",
          options: ["trace", "debug", "info", "warn", "error"],
          description: "Detail level of system logs",
          category: "general"
        },
        {
          id: "api_port",
          name: "API Port",
          value: config.api_port,
          type: "number",
          description: "Port for the Kyaro API server",
          min: 1024,
          max: 65535,
          category: "general"
        },
        {
          id: "data_retention_days",
          name: "Data Retention",
          value: config.data_retention_days,
          type: "number",
          description: "Number of days to retain historical data",
          unit: "days",
          min: 1,
          max: 365,
          category: "general"
        }
      ]
    },
    {
      id: "layer3",
      name: "Layer 3 Protection",
      description: "Network layer protection settings",
      items: [
        {
          id: "enable_ip_blacklisting",
          name: "IP Blacklisting",
          value: config.layer3.enable_ip_blacklisting,
          type: "boolean",
          description: "Automatically block malicious IP addresses",
          category: "layer3"
        },
        {
          id: "enable_geo_blocking",
          name: "Geographic Blocking",
          value: config.layer3.enable_geo_blocking,
          type: "boolean",
          description: "Block traffic from specific countries",
          category: "layer3"
        },
        {
          id: "blocked_countries",
          name: "Blocked Countries",
          value: config.layer3.blocked_countries,
          type: "array",
          description: "List of country codes to block",
          category: "layer3"
        },
        {
          id: "enable_rate_limiting",
          name: "Rate Limiting",
          value: config.layer3.enable_rate_limiting,
          type: "boolean",
          description: "Limit packet rate from individual sources",
          category: "layer3"
        },
        {
          id: "rate_limit_threshold",
          name: "Rate Limit Threshold",
          value: config.layer3.rate_limit_threshold,
          type: "number",
          description: "Maximum packets per second per IP",
          unit: "pps",
          min: 10,
          max: 100000,
          category: "layer3"
        },
        {
          id: "max_packet_size",
          name: "Maximum Packet Size",
          value: config.layer3.max_packet_size,
          type: "number",
          description: "Largest allowed packet size in bytes",
          unit: "bytes",
          min: 512,
          max: 65535,
          category: "layer3"
        },
        {
          id: "min_ttl",
          name: "Minimum TTL",
          value: config.layer3.min_ttl,
          type: "number",
          description: "Minimum allowed Time-To-Live value",
          min: 1,
          max: 255,
          category: "layer3"
        }
      ]
    },
    {
      id: "layer4",
      name: "Layer 4 Protection",
      description: "Transport layer protection settings",
      items: [
        {
          id: "enable_syn_protection",
          name: "SYN Flood Protection",
          value: config.layer4.enable_syn_protection,
          type: "boolean",
          description: "Protect against TCP SYN flood attacks",
          category: "layer4"
        },
        {
          id: "syn_rate_threshold",
          name: "SYN Rate Threshold",
          value: config.layer4.syn_rate_threshold,
          type: "number",
          description: "Maximum SYN packets per second per IP",
          unit: "pps",
          min: 10,
          max: 10000,
          category: "layer4"
        },
        {
          id: "enable_udp_protection",
          name: "UDP Flood Protection",
          value: config.layer4.enable_udp_protection,
          type: "boolean",
          description: "Protect against UDP flood attacks",
          category: "layer4"
        },
        {
          id: "udp_rate_threshold",
          name: "UDP Rate Threshold",
          value: config.layer4.udp_rate_threshold,
          type: "number",
          description: "Maximum UDP packets per second per IP",
          unit: "pps",
          min: 10,
          max: 10000,
          category: "layer4"
        },
        {
          id: "enable_icmp_protection",
          name: "ICMP Flood Protection",
          value: config.layer4.enable_icmp_protection,
          type: "boolean",
          description: "Protect against ICMP flood attacks",
          category: "layer4"
        },
        {
          id: "icmp_rate_threshold",
          name: "ICMP Rate Threshold",
          value: config.layer4.icmp_rate_threshold,
          type: "number",
          description: "Maximum ICMP packets per second per IP",
          unit: "pps",
          min: 5,
          max: 1000,
          category: "layer4"
        },
        {
          id: "max_concurrent_connections",
          name: "Max Concurrent Connections",
          value: config.layer4.max_concurrent_connections,
          type: "number",
          description: "Maximum concurrent connections per IP",
          min: 10,
          max: 10000,
          category: "layer4"
        },
        {
          id: "blocked_ports",
          name: "Blocked Ports",
          value: config.layer4.blocked_ports,
          type: "array",
          description: "List of ports to block",
          category: "layer4"
        }
      ]
    },
    {
      id: "layer7",
      name: "Layer 7 Protection",
      description: "Application layer protection settings",
      items: [
        {
          id: "enable_http_protection",
          name: "HTTP Protection",
          value: config.layer7.enable_http_protection,
          type: "boolean",
          description: "Protect against HTTP-based attacks",
          category: "layer7"
        },
        {
          id: "http_request_rate_threshold",
          name: "HTTP Request Rate Threshold",
          value: config.layer7.http_request_rate_threshold,
          type: "number",
          description: "Maximum HTTP requests per second per IP",
          unit: "req/s",
          min: 1,
          max: 1000,
          category: "layer7"
        },
        {
          id: "blocked_http_methods",
          name: "Blocked HTTP Methods",
          value: config.layer7.blocked_http_methods,
          type: "array",
          description: "HTTP methods to block",
          category: "layer7"
        },
        {
          id: "enable_dns_protection",
          name: "DNS Protection",
          value: config.layer7.enable_dns_protection,
          type: "boolean",
          description: "Protect against DNS-based attacks",
          category: "layer7"
        },
        {
          id: "enable_ssl_protection",
          name: "SSL/TLS Protection",
          value: config.layer7.enable_ssl_protection,
          type: "boolean",
          description: "Protect against SSL/TLS-based attacks",
          category: "layer7"
        },
        {
          id: "challenge_suspicious_requests",
          name: "Challenge Suspicious Requests",
          value: config.layer7.challenge_suspicious_requests,
          type: "boolean",
          description: "Present challenges to suspicious clients",
          category: "layer7"
        },
        {
          id: "bot_protection_level",
          name: "Bot Protection Level",
          value: config.layer7.bot_protection_level,
          type: "number",
          description: "Level of protection against bots (0-3)",
          min: 0,
          max: 3,
          step: 1,
          category: "layer7"
        },
        {
          id: "waf_rules_enabled",
          name: "WAF Rules Enabled",
          value: config.layer7.waf_rules_enabled,
          type: "boolean",
          description: "Enable Web Application Firewall rules",
          category: "layer7"
        }
      ]
    },
    {
      id: "ml",
      name: "Machine Learning",
      description: "ML-based threat detection settings",
      items: [
        {
          id: "enable_ml",
          name: "Enable ML Detection",
          value: config.ml.enable_ml,
          type: "boolean",
          description: "Use machine learning for threat detection",
          category: "ml"
        },
        {
          id: "training_interval_hours",
          name: "Training Interval",
          value: config.ml.training_interval_hours,
          type: "number",
          description: "Hours between model retraining",
          unit: "hours",
          min: 1,
          max: 168,
          category: "ml"
        },
        {
          id: "detection_threshold",
          name: "Detection Threshold",
          value: config.ml.detection_threshold,
          type: "number",
          description: "Threshold for ML-based detection (0-1)",
          min: 0.1,
          max: 1.0,
          step: 0.05,
          category: "ml"
        },
        {
          id: "anomaly_sensitivity",
          name: "Anomaly Sensitivity",
          value: config.ml.anomaly_sensitivity,
          type: "number",
          description: "Sensitivity to anomalies (0-1)",
          min: 0.1,
          max: 1.0,
          step: 0.05,
          category: "ml"
        }
      ]
    }
  ];
}

// Convert frontend config groups back to backend config
export function mapGroupsToConfig(groups: ConfigGroup[], originalConfig: KyaroConfig): KyaroConfig {
  const newConfig: KyaroConfig = JSON.parse(JSON.stringify(originalConfig));
  
  // Process each config item
  for (const group of groups) {
    for (const item of group.items) {
      const { id, value, category } = item;
      
      if (category === "general") {
        // @ts-ignore
        newConfig[id] = value;
      } else {
        // For layer3, layer4, layer7, ml categories
        // @ts-ignore
        newConfig[category][id] = value;
      }
    }
  }
  
  return newConfig;
}

// Generate a default config (as a fallback)
export function generateDefaultConfig(): KyaroConfig {
  return {
    layer3: {
      enable_ip_blacklisting: true,
      enable_geo_blocking: false,
      enable_rate_limiting: true,
      max_packet_size: 65535,
      min_ttl: 64,
      blocked_countries: [],
      rate_limit_threshold: 1000
    },
    layer4: {
      enable_syn_protection: true,
      enable_udp_protection: true,
      enable_icmp_protection: true,
      syn_rate_threshold: 100,
      udp_rate_threshold: 500,
      icmp_rate_threshold: 50,
      max_concurrent_connections: 100,
      blocked_ports: []
    },
    layer7: {
      enable_http_protection: true,
      enable_dns_protection: true,
      enable_ssl_protection: true,
      http_request_rate_threshold: 50,
      blocked_http_methods: ["TRACE", "TRACK"],
      challenge_suspicious_requests: true,
      bot_protection_level: 1,
      waf_rules_enabled: true
    },
    ml: {
      enable_ml: true,
      training_interval_hours: 24,
      detection_threshold: 0.8,
      anomaly_sensitivity: 0.7
    },
    log_level: "info",
    data_retention_days: 30,
    api_port: 6868
  };
} 