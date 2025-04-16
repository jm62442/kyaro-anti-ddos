use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::time::{Duration, SystemTime};

// Common enums and structs for layer 3 protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layer3AttackType {
    IpSpoofing,
    FragmentationAttack,
    AbnormalPacketSize,
    TTLBasedAttack,
    UnusualIPOptions,
    Unknown,
}

// Common enums and structs for layer 4 protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layer4AttackType {
    SynFlood,
    UdpFlood,
    IcmpFlood,
    AbnormalTcpFlags,
    ConnectionFlood,
    SlowLoris,
    TearDrop,
    Unknown,
}

// Common enums and structs for layer 7 protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Layer7AttackType {
    HttpFlood,
    SlowHttpAttack,
    DnsAmplification,
    SslAbuse,
    ApiAbuse,
    WebScraping,
    BotActivity,
    OWASPAttack(String), // Specific OWASP attack type
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MitigationAction {
    Block,
    RateLimit,
    Challenge,
    Monitor,
    Redirect,
    Blackhole,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatInfo {
    pub source_ip: IpAddr,
    pub timestamp: SystemTime,
    pub layer3_attack: Option<Layer3AttackType>,
    pub layer4_attack: Option<Layer4AttackType>,
    pub layer7_attack: Option<Layer7AttackType>,
    pub threat_level: ThreatLevel,
    pub mitigation_action: MitigationAction,
    pub geo_location: Option<String>,
    pub request_rate: Option<f64>,
    pub confidence_score: f32,
    pub is_known_attacker: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub packets_per_second: u64,
    pub bytes_per_second: u64,
    pub connections_per_second: u64,
    pub request_distribution: std::collections::HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationRule {
    pub rule_id: String,
    pub description: String,
    pub source_ip: Option<IpAddr>,
    pub source_network: Option<String>,
    pub destination_port: Option<u16>,
    pub protocol: Option<String>,
    pub action: MitigationAction,
    pub duration: Option<Duration>,
    pub is_active: bool,
    pub created_at: SystemTime,
    pub modified_at: SystemTime,
}

// Config structures for settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer3Config {
    pub enable_ip_blacklisting: bool,
    pub enable_geo_blocking: bool,
    pub enable_rate_limiting: bool,
    pub max_packet_size: usize,
    pub min_ttl: u8,
    pub blocked_countries: Vec<String>,
    pub rate_limit_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer4Config {
    pub enable_syn_protection: bool,
    pub enable_udp_protection: bool,
    pub enable_icmp_protection: bool,
    pub syn_rate_threshold: u32,
    pub udp_rate_threshold: u32,
    pub icmp_rate_threshold: u32,
    pub max_concurrent_connections: u32,
    pub blocked_ports: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer7Config {
    pub enable_http_protection: bool,
    pub enable_dns_protection: bool,
    pub enable_ssl_protection: bool,
    pub http_request_rate_threshold: u32,
    pub blocked_http_methods: Vec<String>,
    pub challenge_suspicious_requests: bool,
    pub bot_protection_level: u8,
    pub waf_rules_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlConfig {
    pub enable_ml: bool,
    pub training_interval_hours: u32,
    pub detection_threshold: f32,
    pub anomaly_sensitivity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyaroConfig {
    pub layer3: Layer3Config,
    pub layer4: Layer4Config,
    pub layer7: Layer7Config,
    pub ml: MlConfig,
    pub log_level: String,
    pub data_retention_days: u32,
    pub api_port: u16,
}

// Communication message between Rust and Python
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlRequest {
    pub traffic_data: Vec<u8>,
    pub model_name: String,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlResponse {
    pub prediction: Vec<f32>,
    pub is_attack: bool,
    pub attack_type: String,
    pub confidence: f32,
}

// Dashboard API structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStats {
    pub total_traffic: u64,
    pub blocked_attacks: u64,
    pub active_mitigations: u32,
    pub current_threats: Vec<ThreatInfo>,
    pub network_metrics: NetworkMetrics,
    pub top_attack_sources: Vec<(IpAddr, u64)>,
    pub top_attack_types: Vec<(String, u64)>,
    pub attack_trend: Vec<(SystemTime, u64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
}
