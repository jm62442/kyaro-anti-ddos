use anyhow::Result;
use std::{
    collections::{HashMap, HashSet},
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::{
    types::{Layer4Config, MitigationAction, PacketInfo, TcpFlags, ThreatInfo, ThreatLevel, Layer4AttackType},
    AppState, IP_BLACKLIST,
};

// Time window for rate limiting (in seconds)
const RATE_LIMIT_WINDOW: u64 = 60;
// Maximum connections to track per IP (to prevent memory exhaustion)
const MAX_CONNECTIONS_PER_IP: usize = 1000;

// TCP Connection states
#[derive(Debug, Clone, PartialEq, Eq)]
enum TcpState {
    SynSent,
    SynReceived,
    Established,
    FinWait,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
    Closed,
}

// Connection tracking for TCP
#[derive(Debug)]
struct Connection {
    src_addr: SocketAddr,
    dst_addr: SocketAddr,
    state: TcpState,
    created_at: Instant,
    last_seen: Instant,
    packets_sent: u32,
    bytes_sent: u64,
}

impl Connection {
    fn new(src_addr: SocketAddr, dst_addr: SocketAddr, state: TcpState) -> Self {
        let now = Instant::now();
        Self {
            src_addr,
            dst_addr,
            state,
            created_at: now,
            last_seen: now,
            packets_sent: 1,
            bytes_sent: 0,
        }
    }
    
    fn update(&mut self, state: Option<TcpState>, bytes: u64) {
        self.last_seen = Instant::now();
        self.packets_sent += 1;
        self.bytes_sent += bytes;
        
        if let Some(new_state) = state {
            self.state = new_state;
        }
    }
    
    fn age(&self) -> Duration {
        self.last_seen.elapsed()
    }
    
    fn total_duration(&self) -> Duration {
        self.last_seen.duration_since(self.created_at)
    }
    
    fn is_active(&self) -> bool {
        self.state != TcpState::Closed && self.age() < Duration::from_secs(300)
    }
}

// Stats for an IP address
struct IpConnectionStats {
    // Connection tracking
    connections: HashMap<(SocketAddr, SocketAddr), Connection>,
    // TCP stats
    syn_count: u32,
    ack_count: u32,
    rst_count: u32,
    fin_count: u32,
    // Connection timing stats
    first_seen: Instant,
    last_seen: Instant,
    // Port stats
    port_distribution: HashMap<u16, u32>,
    // Attack detection
    unusual_flag_combinations: u32,
    half_open_connections: u32,
    // UDP stats
    udp_packets: u32,
    udp_bytes: u64,
    // ICMP stats
    icmp_packets: u32,
    icmp_echo_requests: u32,
}

impl IpConnectionStats {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            connections: HashMap::new(),
            syn_count: 0,
            ack_count: 0,
            rst_count: 0,
            fin_count: 0,
            first_seen: now,
            last_seen: now,
            port_distribution: HashMap::new(),
            unusual_flag_combinations: 0,
            half_open_connections: 0,
            udp_packets: 0,
            udp_bytes: 0,
            icmp_packets: 0,
            icmp_echo_requests: 0,
        }
    }
    
    fn update_tcp(&mut self, packet: &PacketInfo, flags: &TcpFlags) {
        self.last_seen = Instant::now();
        
        // Track port distribution
        if let Some(dst_port) = packet.dst_port {
            *self.port_distribution.entry(dst_port).or_insert(0) += 1;
        }
        
        // Count TCP flags
        if flags.syn {
            self.syn_count += 1;
        }
        if flags.ack {
            self.ack_count += 1;
        }
        if flags.rst {
            self.rst_count += 1;
        }
        if flags.fin {
            self.fin_count += 1;
        }
        
        // Detect unusual flag combinations
        if (flags.syn && flags.fin) || 
           (flags.syn && flags.rst) || 
           (flags.fin && flags.rst) {
            self.unusual_flag_combinations += 1;
        }
        
        // Update connection state
        if let (Some(src_port), Some(dst_port)) = (packet.src_port, packet.dst_port) {
            let src_addr = SocketAddr::new(packet.src_ip, src_port);
            let dst_addr = SocketAddr::new(packet.dst_ip, dst_port);
            let conn_key = (src_addr, dst_addr);
            
            // Update connection state based on TCP flags
            if flags.syn && !flags.ack {
                // SYN packet - new connection attempt
                if !self.connections.contains_key(&conn_key) {
                    // Only insert if we haven't reached the limit
                    if self.connections.len() < MAX_CONNECTIONS_PER_IP {
                        self.connections.insert(
                            conn_key,
                            Connection::new(src_addr, dst_addr, TcpState::SynSent)
                        );
                        self.half_open_connections += 1;
                    }
                }
            } else if flags.syn && flags.ack {
                // SYN-ACK packet - connection response
                if let Some(conn) = self.connections.get_mut(&conn_key) {
                    conn.update(Some(TcpState::SynReceived), packet.size as u64);
                }
            } else if flags.ack && !flags.syn && !flags.fin && !flags.rst {
                // Regular ACK
                if let Some(conn) = self.connections.get_mut(&conn_key) {
                    if conn.state == TcpState::SynReceived {
                        conn.update(Some(TcpState::Established), packet.size as u64);
                        // Connection established, decrement half-open counter
                        if self.half_open_connections > 0 {
                            self.half_open_connections -= 1;
                        }
                    } else {
                        conn.update(None, packet.size as u64);
                    }
                }
            } else if flags.fin {
                // FIN packet - connection close initiated
                if let Some(conn) = self.connections.get_mut(&conn_key) {
                    let new_state = match conn.state {
                        TcpState::Established => TcpState::FinWait,
                        TcpState::CloseWait => TcpState::LastAck,
                        _ => conn.state.clone(),
                    };
                    conn.update(Some(new_state), packet.size as u64);
                }
            } else if flags.rst {
                // RST packet - abruptly terminate connection
                if let Some(conn) = self.connections.get_mut(&conn_key) {
                    conn.update(Some(TcpState::Closed), packet.size as u64);
                    // Connection reset, decrement half-open counter if needed
                    if conn.state == TcpState::SynSent || conn.state == TcpState::SynReceived {
                        if self.half_open_connections > 0 {
                            self.half_open_connections -= 1;
                        }
                    }
                }
            } else {
                // Regular data packet, update existing connection if exists
                if let Some(conn) = self.connections.get_mut(&conn_key) {
                    conn.update(None, packet.size as u64);
                }
            }
        }
    }
    
    fn update_udp(&mut self, packet: &PacketInfo) {
        self.last_seen = Instant::now();
        self.udp_packets += 1;
        self.udp_bytes += packet.size as u64;
        
        // Track port distribution
        if let Some(dst_port) = packet.dst_port {
            *self.port_distribution.entry(dst_port).or_insert(0) += 1;
        }
    }
    
    fn update_icmp(&mut self, packet: &PacketInfo, is_echo_request: bool) {
        self.last_seen = Instant::now();
        self.icmp_packets += 1;
        
        if is_echo_request {
            self.icmp_echo_requests += 1;
        }
    }
    
    fn cleanup_expired_connections(&mut self) {
        // Remove closed connections and those that are idle for too long
        self.connections.retain(|_, conn| {
            let retain = conn.is_active();
            
            // Update half-open connection counter if removing a half-open connection
            if !retain && (conn.state == TcpState::SynSent || conn.state == TcpState::SynReceived) {
                if self.half_open_connections > 0 {
                    self.half_open_connections -= 1;
                }
            }
            
            retain
        });
    }
    
    fn syn_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.syn_count as f64 / elapsed
        } else {
            self.syn_count as f64
        }
    }
    
    fn udp_packet_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.udp_packets as f64 / elapsed
        } else {
            self.udp_packets as f64
        }
    }
    
    fn icmp_packet_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.icmp_packets as f64 / elapsed
        } else {
            self.icmp_packets as f64
        }
    }
    
    fn unusual_flags_rate(&self) -> f64 {
        let total_tcp = self.syn_count + self.ack_count + self.rst_count + self.fin_count;
        if total_tcp > 0 {
            self.unusual_flag_combinations as f64 / total_tcp as f64
        } else {
            0.0
        }
    }
    
    fn active_connections(&self) -> usize {
        self.connections.values().filter(|c| c.is_active()).count()
    }
    
    fn port_scan_score(&self) -> f64 {
        let total_ports = self.port_distribution.len();
        let total_packets = self.port_distribution.values().sum::<u32>();
        
        if total_packets < 5 || total_ports < 3 {
            return 0.0;
        }
        
        // Calculate entropy of port distribution
        // High entropy means more evenly distributed ports (potential port scan)
        let mut entropy = 0.0;
        for &count in self.port_distribution.values() {
            let p = count as f64 / total_packets as f64;
            entropy -= p * p.log2();
        }
        
        // Normalize and scale the entropy score
        let max_entropy = (total_ports as f64).log2();
        if max_entropy > 0.0 {
            (entropy / max_entropy).min(1.0)
        } else {
            0.0
        }
    }
    
    fn is_expired(&self, window: Duration) -> bool {
        self.last_seen.elapsed() > window
    }
}

// Layer 4 protection service
pub struct Layer4Protection {
    stats: RwLock<HashMap<IpAddr, IpConnectionStats>>,
    config: Arc<RwLock<Layer4Config>>,
    ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>,
    last_cleanup: RwLock<Instant>,
}

impl Layer4Protection {
    pub fn new(config: Arc<RwLock<Layer4Config>>, ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>) -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
            config,
            ip_blacklist,
            last_cleanup: RwLock::new(Instant::now()),
        }
    }
    
    // Process a packet for Layer 4 protection
    pub async fn process_packet(&self, packet: &PacketInfo) -> Result<MitigationAction> {
        // Check if the IP is blacklisted
        let is_blacklisted = {
            let blacklist = self.ip_blacklist.read().await;
            blacklist.contains(&packet.src_ip)
        };
        
        if is_blacklisted {
            return Ok(MitigationAction::Drop);
        }
        
        let config = self.config.read().await;
        
        // Check if destination port is blocked
        if let Some(dst_port) = packet.dst_port {
            if config.blocked_ports.contains(&dst_port) {
                debug!("Blocked traffic to restricted port {} from {}", dst_port, packet.src_ip);
                return Ok(MitigationAction::Drop);
            }
        }
        
        // Update stats for this IP
        self.update_stats(packet).await;
        
        // Check for potential threats
        if let Some(threat) = self.detect_threats(packet).await {
            // Apply appropriate mitigation based on threat level
            match threat.threat_level {
                ThreatLevel::Low => {
                    // Just monitor for low-level threats
                    debug!("Low-level Layer 4 threat detected from {}: {:?}", packet.src_ip, threat);
                    Ok(MitigationAction::Monitor)
                },
                ThreatLevel::Medium => {
                    // Rate limit medium-level threats
                    info!("Medium-level Layer 4 threat detected from {}: {:?}", packet.src_ip, threat);
                    Ok(MitigationAction::RateLimit)
                },
                ThreatLevel::High | ThreatLevel::Critical => {
                    // Block high and critical threats
                    warn!("High-level Layer 4 threat detected from {}: {:?}", packet.src_ip, threat);
                    
                    // Add to blacklist
                    let mut blacklist = self.ip_blacklist.write().await;
                    blacklist.insert(packet.src_ip);
                    info!("Added {} to IP blacklist", packet.src_ip);
                    
                    Ok(MitigationAction::Drop)
                }
            }
        } else {
            // No threat detected, allow the packet
            Ok(MitigationAction::Allow)
        }
    }
    
    // Update stats for an IP address
    async fn update_stats(&self, packet: &PacketInfo) -> () {
        let mut stats = self.stats.write().await;
        
        // Create or update stats for this IP
        let ip_stats = stats
            .entry(packet.src_ip)
            .or_insert_with(IpConnectionStats::new);
        
        // Update protocol-specific stats
        match packet.protocol.as_str() {
            "TCP" => {
                if let Some(tcp_flags) = &packet.tcp_flags {
                    ip_stats.update_tcp(packet, tcp_flags);
                }
            },
            "UDP" => {
                ip_stats.update_udp(packet);
            },
            "ICMP" => {
                ip_stats.update_icmp(packet, packet.icmp_type.map_or(false, |t| t == 8));
            },
            _ => {}
        }
        
        // Clean up expired connections within this IP's stats
        ip_stats.cleanup_expired_connections();
        
        // Occasionally cleanup the entire stats collection
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.read().await;
            last_cleanup.elapsed() > Duration::from_secs(300) // Every 5 minutes
        };
        
        if should_cleanup {
            self.cleanup_old_stats().await;
        }
    }
    
    // Clean up old stats entries
    async fn cleanup_old_stats(&self) {
        let mut last_cleanup = self.last_cleanup.write().await;
        *last_cleanup = Instant::now();
        
        let mut stats = self.stats.write().await;
        
        // Remove entries older than the rate limit window
        let window = Duration::from_secs(RATE_LIMIT_WINDOW * 2); // Give some extra time
        stats.retain(|_, s| !s.is_expired(window));
        
        debug!("Cleaned up old Layer 4 stats, remaining entries: {}", stats.len());
    }
    
    // Detect Layer 4 threats from packet information and connection stats
    async fn detect_threats(&self, packet: &PacketInfo) -> Option<ThreatInfo> {
        let config = self.config.read().await;
        let stats = self.stats.read().await;
        
        let ip_stats = match stats.get(&packet.src_ip) {
            Some(s) => s,
            None => return None, // No stats yet for this IP
        };
        
        let mut threat_indicators = Vec::new();
        let mut threat_level = ThreatLevel::Low;
        let mut is_attack = false;
        let mut attack_type: Option<Layer4AttackType> = None;
        
        // TCP-specific checks
        if packet.protocol == "TCP" {
            // SYN flood detection
            if config.enable_syn_protection {
                let syn_rate = ip_stats.syn_rate();
                if syn_rate > config.syn_rate_threshold as f64 {
                    threat_indicators.push(format!("SYN flood detected: {:.2} SYN/s", syn_rate));
                    threat_level = ThreatLevel::Medium;
                    is_attack = true;
                    attack_type = Some(Layer4AttackType::SynFlood);
                    
                    // Upgrade threat level for severe SYN flood
                    if syn_rate > config.syn_rate_threshold as f64 * 5.0 {
                        threat_level = ThreatLevel::High;
                    }
                }
            }
            
            // Check for unusual TCP flag combinations
            let unusual_flags_rate = ip_stats.unusual_flags_rate();
            if unusual_flags_rate > 0.1 { // More than 10% of packets have unusual flags
                threat_indicators.push(format!("Unusual TCP flags detected: {:.2}%", unusual_flags_rate * 100.0));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
                attack_type = Some(Layer4AttackType::AbnormalTcpFlags);
            }
            
            // Check for too many connections
            let active_connections = ip_stats.active_connections();
            if active_connections > config.max_concurrent_connections as usize {
                threat_indicators.push(format!("Excessive connections: {}", active_connections));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
                attack_type = Some(Layer4AttackType::ConnectionFlood);
                
                // Upgrade threat level for connection flooding
                if active_connections > config.max_concurrent_connections as usize * 2 {
                    threat_level = ThreatLevel::High;
                }
            }
            
            // Half-open connection attack detection (SYN flood variant)
            if ip_stats.half_open_connections > config.max_concurrent_connections as u32 / 2 {
                threat_indicators.push(format!("High number of half-open connections: {}", ip_stats.half_open_connections));
                threat_level = threat_level.max(ThreatLevel::High);
                is_attack = true;
                attack_type = Some(Layer4AttackType::SynFlood);
            }
        }
        
        // UDP-specific checks
        if packet.protocol == "UDP" && config.enable_udp_protection {
            let udp_rate = ip_stats.udp_packet_rate();
            if udp_rate > config.udp_rate_threshold as f64 {
                threat_indicators.push(format!("UDP flood detected: {:.2} packets/s", udp_rate));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
                attack_type = Some(Layer4AttackType::UdpFlood);
                
                // Upgrade threat level for severe UDP flood
                if udp_rate > config.udp_rate_threshold as f64 * 5.0 {
                    threat_level = ThreatLevel::High;
                }
            }
        }
        
        // ICMP-specific checks
        if packet.protocol == "ICMP" && config.enable_icmp_protection {
            let icmp_rate = ip_stats.icmp_packet_rate();
            if icmp_rate > config.icmp_rate_threshold as f64 {
                threat_indicators.push(format!("ICMP flood detected: {:.2} packets/s", icmp_rate));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
                attack_type = Some(Layer4AttackType::IcmpFlood);
                
                // Upgrade threat level for severe ICMP flood
                if icmp_rate > config.icmp_rate_threshold as f64 * 5.0 {
                    threat_level = ThreatLevel::High;
                }
            }
        }
        
        // Port scanning detection
        let port_scan_score = ip_stats.port_scan_score();
        if port_scan_score > 0.7 {
            threat_indicators.push(format!("Potential port scan detected (score: {:.2})", port_scan_score));
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
            
            // Don't set attack_type for port scans as they're not a specific DDoS type
            // but we still want to track and potentially mitigate them
        }
        
        // Return threat info if attack indicators were found
        if is_attack {
            let threat_info = ThreatInfo {
                source_ip: packet.src_ip,
                timestamp: SystemTime::now(),
                threat_level,
                layer3_attack: None,
                layer4_attack: attack_type,
                layer7_attack: None,
                mitigation_action: match threat_level {
                    ThreatLevel::Low => MitigationAction::Monitor,
                    ThreatLevel::Medium => MitigationAction::RateLimit,
                    _ => MitigationAction::Drop,
                },
                geo_location: "Unknown".to_string(),
                request_rate: match packet.protocol.as_str() {
                    "TCP" => ip_stats.syn_rate(),
                    "UDP" => ip_stats.udp_packet_rate(),
                    "ICMP" => ip_stats.icmp_packet_rate(),
                    _ => 0.0,
                },
                confidence_score: 0.85,
                details: threat_indicators.join(", "),
                is_known_attacker: false,
            };
            
            Some(threat_info)
        } else {
            None
        }
    }
    
    // Get Layer 4 stats for monitoring and reporting
    pub async fn get_connection_stats(&self) -> HashMap<IpAddr, HashMap<String, serde_json::Value>> {
        let stats = self.stats.read().await;
        let mut result = HashMap::new();
        
        for (ip, ip_stats) in stats.iter() {
            let mut metrics = HashMap::new();
            
            // Add numeric stats
            metrics.insert("active_connections".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from(ip_stats.active_connections())));
            metrics.insert("syn_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_stats.syn_rate()).unwrap_or(serde_json::Number::from(0))));
            metrics.insert("udp_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_stats.udp_packet_rate()).unwrap_or(serde_json::Number::from(0))));
            metrics.insert("icmp_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_stats.icmp_packet_rate()).unwrap_or(serde_json::Number::from(0))));
            metrics.insert("unusual_flags_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_stats.unusual_flags_rate()).unwrap_or(serde_json::Number::from(0))));
            metrics.insert("port_scan_score".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_stats.port_scan_score()).unwrap_or(serde_json::Number::from(0))));
            metrics.insert("half_open_connections".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from(ip_stats.half_open_connections)));
            
            // Add port distribution (top 10 ports)
            let mut port_counts: Vec<(u16, u32)> = ip_stats.port_distribution
                .iter()
                .map(|(&port, &count)| (port, count))
                .collect();
            
            port_counts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count, descending
            
            let top_ports: serde_json::Value = serde_json::to_value(
                port_counts.iter()
                    .take(10)
                    .map(|(port, count)| (port.to_string(), *count))
                    .collect::<HashMap<String, u32>>()
            ).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            
            metrics.insert("top_ports".to_string(), top_ports);
            
            result.insert(*ip, metrics);
        }
        
        result
    }
} 