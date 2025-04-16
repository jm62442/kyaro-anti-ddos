use anyhow::Result;
use std::{
    collections::{HashMap, HashSet},
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use maxminddb::geoip2;

use crate::{
    types::{Layer3Config, MitigationAction, PacketInfo, ThreatInfo, ThreatLevel},
    AppState, IP_BLACKLIST,
};

// Time window for rate limiting (in seconds)
const RATE_LIMIT_WINDOW: u64 = 60;

// Struct to store traffic stats for an IP
struct IpTrafficStats {
    first_seen: Instant,
    last_seen: Instant,
    packet_count: u64,
    byte_count: u64,
    packet_sizes: Vec<u16>,
    ttl_values: Vec<u8>,
    unusual_options_count: u64,
    fragmented_packet_count: u64,
}

impl IpTrafficStats {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            first_seen: now,
            last_seen: now,
            packet_count: 0,
            byte_count: 0,
            packet_sizes: Vec::with_capacity(100),
            ttl_values: Vec::with_capacity(100),
            unusual_options_count: 0,
            fragmented_packet_count: 0,
        }
    }
    
    fn update(&mut self, packet: &PacketInfo) {
        self.last_seen = Instant::now();
        self.packet_count += 1;
        self.byte_count += packet.size as u64;
        
        // Keep only recent packet sizes to prevent memory bloat
        if self.packet_sizes.len() >= 100 {
            self.packet_sizes.remove(0);
        }
        self.packet_sizes.push(packet.size);
        
        // Keep only recent TTL values to prevent memory bloat
        if self.ttl_values.len() >= 100 {
            self.ttl_values.remove(0);
        }
        if let Some(ttl) = packet.ttl {
            self.ttl_values.push(ttl);
        }
        
        if packet.has_unusual_options {
            self.unusual_options_count += 1;
        }
        
        if packet.is_fragmented {
            self.fragmented_packet_count += 1;
        }
    }
    
    fn packet_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.packet_count as f64 / elapsed
        } else {
            self.packet_count as f64 // Avoid division by zero
        }
    }
    
    fn byte_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.byte_count as f64 / elapsed
        } else {
            self.byte_count as f64 // Avoid division by zero
        }
    }
    
    fn packet_size_mean(&self) -> f64 {
        if self.packet_sizes.is_empty() {
            return 0.0;
        }
        self.packet_sizes.iter().map(|&s| s as f64).sum::<f64>() / self.packet_sizes.len() as f64
    }
    
    fn packet_size_std(&self) -> f64 {
        if self.packet_sizes.len() <= 1 {
            return 0.0;
        }
        
        let mean = self.packet_size_mean();
        let variance = self.packet_sizes.iter()
            .map(|&s| {
                let diff = s as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / (self.packet_sizes.len() - 1) as f64;
        
        variance.sqrt()
    }
    
    fn ttl_mean(&self) -> f64 {
        if self.ttl_values.is_empty() {
            return 0.0;
        }
        self.ttl_values.iter().map(|&t| t as f64).sum::<f64>() / self.ttl_values.len() as f64
    }
    
    fn ttl_variance(&self) -> f64 {
        if self.ttl_values.len() <= 1 {
            return 0.0;
        }
        
        let mean = self.ttl_mean();
        let variance = self.ttl_values.iter()
            .map(|&t| {
                let diff = t as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / (self.ttl_values.len() - 1) as f64;
        
        variance
    }
    
    fn unusual_options_rate(&self) -> f64 {
        if self.packet_count == 0 {
            return 0.0;
        }
        self.unusual_options_count as f64 / self.packet_count as f64
    }
    
    fn fragmentation_rate(&self) -> f64 {
        if self.packet_count == 0 {
            return 0.0;
        }
        self.fragmented_packet_count as f64 / self.packet_count as f64
    }
    
    fn is_expired(&self, window: Duration) -> bool {
        self.last_seen.elapsed() > window
    }
}

// Layer 3 protection service
pub struct Layer3Protection {
    stats: RwLock<HashMap<IpAddr, IpTrafficStats>>,
    config: Arc<RwLock<Layer3Config>>,
    ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>,
    // GeoIP database for geolocation (optional)
    geoip_reader: Option<maxminddb::Reader<Vec<u8>>>,
    // Last cleanup time
    last_cleanup: RwLock<Instant>,
}

impl Layer3Protection {
    pub fn new(config: Arc<RwLock<Layer3Config>>, ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>) -> Self {
        // Try to load GeoIP database if available
        let geoip_reader = match maxminddb::Reader::open_readfile("GeoLite2-Country.mmdb") {
            Ok(reader) => {
                info!("Loaded GeoIP database for country lookup");
                Some(reader)
            },
            Err(e) => {
                warn!("Could not load GeoIP database: {}", e);
                None
            }
        };
        
        Self {
            stats: RwLock::new(HashMap::new()),
            config,
            ip_blacklist,
            geoip_reader,
            last_cleanup: RwLock::new(Instant::now()),
        }
    }
    
    // Process a packet for Layer 3 protection
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
        
        // Check if geo-blocking is enabled and the country is blocked
        if config.enable_geo_blocking && !config.blocked_countries.is_empty() {
            if let Some(country) = self.get_country_code(&packet.src_ip) {
                if config.blocked_countries.contains(&country) {
                    debug!("Blocked IP {} from blocked country {}", packet.src_ip, country);
                    return Ok(MitigationAction::Drop);
                }
            }
        }
        
        // Update stats for the IP
        self.update_stats(packet).await;
        
        // Check for potential threats
        if let Some(threat) = self.detect_threats(packet).await {
            // Apply appropriate mitigation based on threat level
            match threat.threat_level {
                ThreatLevel::Low => {
                    // Just monitor for low-level threats
                    debug!("Low-level Layer 3 threat detected from {}: {:?}", packet.src_ip, threat);
                    Ok(MitigationAction::Monitor)
                },
                ThreatLevel::Medium => {
                    // Rate limit medium-level threats
                    info!("Medium-level Layer 3 threat detected from {}: {:?}", packet.src_ip, threat);
                    Ok(MitigationAction::RateLimit)
                },
                ThreatLevel::High | ThreatLevel::Critical => {
                    // Block high and critical threats
                    warn!("High-level Layer 3 threat detected from {}: {:?}", packet.src_ip, threat);
                    
                    // Add to blacklist if IP blacklisting is enabled
                    if config.enable_ip_blacklisting {
                        let mut blacklist = self.ip_blacklist.write().await;
                        blacklist.insert(packet.src_ip);
                        info!("Added {} to IP blacklist", packet.src_ip);
                    }
                    
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
            .or_insert_with(IpTrafficStats::new);
        
        ip_stats.update(packet);
        
        // Clean up old entries occasionally to prevent memory leaks
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
        let window = Duration::from_secs(RATE_LIMIT_WINDOW);
        stats.retain(|_, s| !s.is_expired(window));
        
        debug!("Cleaned up old Layer 3 stats, remaining entries: {}", stats.len());
    }
    
    // Detect Layer 3 threats from packet information and traffic stats
    async fn detect_threats(&self, packet: &PacketInfo) -> Option<ThreatInfo> {
        let config = self.config.read().await;
        let stats = self.stats.read().await;
        
        let ip_stats = match stats.get(&packet.src_ip) {
            Some(s) => s,
            None => return None, // No stats yet for this IP
        };
        
        // Check for various Layer 3 attack indicators
        let mut threat_indicators = Vec::new();
        let mut threat_level = ThreatLevel::Low;
        let mut is_attack = false;
        
        // Rate limiting check
        if config.enable_rate_limiting {
            let packet_rate = ip_stats.packet_rate();
            if packet_rate > config.rate_limit_threshold as f64 {
                threat_indicators.push(format!("High packet rate: {:.2} pps", packet_rate));
                threat_level = ThreatLevel::Medium;
                is_attack = true;
                
                // Upgrade to high if extremely high packet rate
                if packet_rate > config.rate_limit_threshold as f64 * 5.0 {
                    threat_level = ThreatLevel::High;
                }
            }
        }
        
        // Packet size anomaly detection
        if let Some(max_packet_size) = config.max_packet_size {
            if packet.size > max_packet_size {
                threat_indicators.push(format!("Oversized packet: {} bytes", packet.size));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
            }
        }
        
        // Very small packets can be used in some attacks
        if packet.size < 20 {
            threat_indicators.push(format!("Suspiciously small packet: {} bytes", packet.size));
            threat_level = threat_level.max(ThreatLevel::Low);
            is_attack = true;
        }
        
        // TTL-based checks
        if let Some(min_ttl) = config.min_ttl {
            if let Some(packet_ttl) = packet.ttl {
                if packet_ttl < min_ttl {
                    threat_indicators.push(format!("Low TTL value: {}", packet_ttl));
                    threat_level = threat_level.max(ThreatLevel::Low);
                    is_attack = true;
                }
            }
        }
        
        // Check for TTL variation which can indicate IP spoofing
        if ip_stats.ttl_values.len() > 5 && ip_stats.ttl_variance() > 100.0 {
            threat_indicators.push(format!("High TTL variance: {:.2}", ip_stats.ttl_variance()));
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
        }
        
        // Unusual IP options
        if packet.has_unusual_options {
            threat_indicators.push("Packet contains unusual IP options".to_string());
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
        }
        
        // Fragmentation attacks
        if packet.is_fragmented {
            // High fragmentation rate can indicate attacks
            if ip_stats.fragmentation_rate() > 0.5 {
                threat_indicators.push(format!("High fragmentation rate: {:.2}%", ip_stats.fragmentation_rate() * 100.0));
                threat_level = threat_level.max(ThreatLevel::High);
                is_attack = true;
            } else {
                threat_indicators.push("Fragmented packet".to_string());
                // Just a single fragmented packet isn't necessarily an attack
            }
        }
        
        // Return threat info if attack indicators were found
        if is_attack {
            let country_code = self.get_country_code(&packet.src_ip).unwrap_or_else(|| "Unknown".to_string());
            
            let threat_info = ThreatInfo {
                source_ip: packet.src_ip,
                timestamp: SystemTime::now(),
                threat_level,
                layer3_attack: Some(packet.protocol.clone()),
                layer4_attack: None,
                layer7_attack: None,
                mitigation_action: match threat_level {
                    ThreatLevel::Low => MitigationAction::Monitor,
                    ThreatLevel::Medium => MitigationAction::RateLimit,
                    _ => MitigationAction::Drop,
                },
                geo_location: country_code,
                request_rate: ip_stats.packet_rate(),
                confidence_score: 0.85,
                details: threat_indicators.join(", "),
                is_known_attacker: false,
            };
            
            Some(threat_info)
        } else {
            None
        }
    }
    
    // Get the country code for an IP address
    fn get_country_code(&self, ip: &IpAddr) -> Option<String> {
        if let Some(ref reader) = self.geoip_reader {
            match reader.lookup::<geoip2::Country>(*ip) {
                Ok(country) => {
                    country.country
                        .and_then(|c| c.iso_code)
                        .map(|iso| iso.to_string())
                },
                Err(_) => None,
            }
        } else {
            None
        }
    }
    
    // Get traffic stats for monitoring purposes
    pub async fn get_traffic_stats(&self) -> HashMap<IpAddr, HashMap<String, f64>> {
        let stats = self.stats.read().await;
        let mut result = HashMap::new();
        
        for (ip, ip_stats) in stats.iter() {
            let mut metrics = HashMap::new();
            
            metrics.insert("packet_rate".to_string(), ip_stats.packet_rate());
            metrics.insert("byte_rate".to_string(), ip_stats.byte_rate());
            metrics.insert("packet_size_mean".to_string(), ip_stats.packet_size_mean());
            metrics.insert("packet_size_std".to_string(), ip_stats.packet_size_std());
            metrics.insert("ttl_mean".to_string(), ip_stats.ttl_mean());
            metrics.insert("ttl_variance".to_string(), ip_stats.ttl_variance());
            metrics.insert("unusual_options_rate".to_string(), ip_stats.unusual_options_rate());
            metrics.insert("fragmentation_rate".to_string(), ip_stats.fragmentation_rate());
            
            result.insert(*ip, metrics);
        }
        
        result
    }
} 