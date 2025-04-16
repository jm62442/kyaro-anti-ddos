use anyhow::Result;
use pnet::{
    datalink::{self, Channel, NetworkInterface},
    packet::{
        ethernet::{EthernetPacket, EtherTypes},
        ip::IpNextHeaderProtocols,
        ipv4::Ipv4Packet,
        tcp::TcpPacket,
        udp::UdpPacket,
        Packet,
    },
};
use std::{
    collections::HashMap,
    net::IpAddr,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{sync::Mutex, time};
use tracing::{debug, error, info, warn};

use crate::{
    ml_integration::{self, analyze_traffic_patterns},
    types::{Layer3AttackType, Layer4AttackType, Layer7AttackType, MitigationAction, ThreatInfo, ThreatLevel},
    AppState, IP_BLACKLIST, IP_WHITELIST, RATE_LIMITER, ACTIVE_MITIGATIONS,
};

// Start packet analysis for DDoS detection and mitigation
pub async fn start_packet_analysis(state: Arc<AppState>) -> Result<()> {
    // Get network interfaces
    let interfaces = datalink::interfaces();
    let interface_names = interfaces.iter().map(|i| i.name.clone()).collect::<Vec<_>>();
    
    info!("Available network interfaces: {:?}", interface_names);
    
    let mut packet_handlers = Vec::new();
    
    // Set up packet capture for each interface
    for interface in interfaces {
        if !interface.is_up() || interface.is_loopback() {
            continue;
        }
        
        info!("Starting packet analysis on interface: {}", interface.name);
        
        let interface_state = state.clone();
        let interface_clone = interface.clone();
        
        let handler = tokio::spawn(async move {
            if let Err(e) = monitor_interface(interface_clone, interface_state).await {
                error!("Error monitoring interface {}: {}", interface_clone.name, e);
            }
        });
        
        packet_handlers.push(handler);
    }
    
    // Set up a periodic task to update stats
    let stats_state = state.clone();
    let stats_updater = tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            
            // Process rate limiters and update stats
            let mut attack_count = 0;
            let mut threat_map = HashMap::new();
            
            // Check rate limiters for potential attacks
            for entry in RATE_LIMITER.iter() {
                let ip = *entry.key();
                let timestamps = entry.value();
                
                // Skip whitelisted IPs
                if IP_WHITELIST.contains_key(&ip) {
                    continue;
                }
                
                // Calculate request rate
                let now = SystemTime::now();
                let recent_requests = timestamps.iter().filter(|&&ts| {
                    match now.duration_since(ts) {
                        Ok(duration) => duration < Duration::from_secs(60),
                        Err(_) => false,
                    }
                }).count();
                
                // Get thresholds from config
                let config = stats_state.config.lock().await;
                let l3_threshold = config.layer3.rate_limit_threshold;
                let l4_tcp_threshold = config.layer4.syn_rate_threshold;
                let l4_udp_threshold = config.layer4.udp_rate_threshold;
                let l7_http_threshold = config.layer7.http_request_rate_threshold;
                
                // Determine if this is an attack based on rate
                if recent_requests as u32 > l3_threshold {
                    attack_count += 1;
                    
                    // Create threat info
                    let threat = ThreatInfo {
                        source_ip: ip,
                        timestamp: now,
                        layer3_attack: Some(Layer3AttackType::Unknown),
                        layer4_attack: None,
                        layer7_attack: None,
                        threat_level: ThreatLevel::Medium,
                        mitigation_action: MitigationAction::RateLimit,
                        geo_location: None,
                        request_rate: Some(recent_requests as f64 / 60.0),
                        confidence_score: 0.8,
                        is_known_attacker: IP_BLACKLIST.contains_key(&ip),
                    };
                    
                    threat_map.insert(ip, threat);
                    
                    // If not already blacklisted, add to blacklist
                    if !IP_BLACKLIST.contains_key(&ip) {
                        debug!("Rate limit exceeded for IP: {}, adding to blacklist", ip);
                        
                        // Add to blacklist for 10 minutes
                        match now.checked_add(Duration::from_secs(600)) {
                            Some(expiry) => {
                                IP_BLACKLIST.insert(ip, expiry);
                            }
                            None => {
                                error!("Failed to calculate expiry time for blacklist");
                            }
                        }
                    }
                }
            }
            
            // Update stats
            let mut stats = stats_state.stats.lock().await;
            stats.blocked_attacks += attack_count as u64;
            stats.current_threats = threat_map.into_values().collect();
            
            // Update top attack sources
            let mut source_counts = HashMap::new();
            for threat in &stats.current_threats {
                *source_counts.entry(threat.source_ip).or_insert(0) += 1;
            }
            
            stats.top_attack_sources = source_counts
                .into_iter()
                .map(|(ip, count)| (ip, count))
                .collect::<Vec<_>>();
            
            stats.top_attack_sources.sort_by(|a, b| b.1.cmp(&a.1));
            if stats.top_attack_sources.len() > 10 {
                stats.top_attack_sources.truncate(10);
            }
            
            // Update top attack types
            let mut type_counts = HashMap::new();
            for threat in &stats.current_threats {
                let attack_type = if let Some(l3) = &threat.layer3_attack {
                    format!("L3: {:?}", l3)
                } else if let Some(l4) = &threat.layer4_attack {
                    format!("L4: {:?}", l4)
                } else if let Some(l7) = &threat.layer7_attack {
                    format!("L7: {:?}", l7)
                } else {
                    "Unknown".to_string()
                };
                
                *type_counts.entry(attack_type).or_insert(0) += 1;
            }
            
            stats.top_attack_types = type_counts
                .into_iter()
                .map(|(attack_type, count)| (attack_type, count))
                .collect::<Vec<_>>();
            
            stats.top_attack_types.sort_by(|a, b| b.1.cmp(&a.1));
            if stats.top_attack_types.len() > 10 {
                stats.top_attack_types.truncate(10);
            }
        }
    });
    
    for handler in packet_handlers {
        handler.await?;
    }
    
    stats_updater.await?;
    
    Ok(())
}

// Monitor a specific network interface
async fn monitor_interface(interface: NetworkInterface, state: Arc<AppState>) -> Result<()> {
    // Create a channel to receive on
    let (_, mut rx) = match datalink::channel(&interface, Default::default()) {
        Ok(Channel::Ethernet(tx, rx)) => (tx, rx),
        Ok(_) => {
            warn!("Unsupported channel type for interface: {}", interface.name);
            return Ok(());
        }
        Err(e) => {
            error!("Failed to create datalink channel for {}: {}", interface.name, e);
            return Ok(());
        }
    };
    
    info!("Successfully created channel for interface: {}", interface.name);
    
    // Packet processing loop
    loop {
        match rx.next() {
            Ok(packet) => {
                if let Some(ethernet) = EthernetPacket::new(packet) {
                    process_ethernet_packet(ethernet, &state).await?;
                }
            }
            Err(e) => {
                error!("Failed to read packet: {}", e);
            }
        }
    }
}

// Process an ethernet packet
async fn process_ethernet_packet(ethernet: EthernetPacket, state: &Arc<AppState>) -> Result<()> {
    match ethernet.get_ethertype() {
        EtherTypes::Ipv4 => {
            if let Some(ipv4) = Ipv4Packet::new(ethernet.payload()) {
                process_ipv4_packet(ipv4, state).await?;
            }
        }
        EtherTypes::Ipv6 => {
            // IPv6 handling can be added here
        }
        _ => {
            // Ignore other ethernet types
        }
    }
    
    Ok(())
}

// Process an IPv4 packet
async fn process_ipv4_packet(ipv4: Ipv4Packet, state: &Arc<AppState>) -> Result<()> {
    let source_ip = IpAddr::V4(ipv4.get_source());
    
    // Layer 3 checks - Check for blacklisted IPs
    if IP_BLACKLIST.contains_key(&source_ip) {
        // IP is blacklisted, drop the packet
        debug!("Dropping packet from blacklisted IP: {}", source_ip);
        return Ok(());
    }
    
    // Layer 3 checks - Check for abnormal packet size
    let config = state.config.lock().await;
    if ipv4.payload().len() > config.layer3.max_packet_size {
        debug!("Detected abnormally large packet from IP: {}", source_ip);
        
        // Add to stats and potentially blacklist
        record_threat(
            source_ip,
            Some(Layer3AttackType::AbnormalPacketSize),
            None,
            None,
            ThreatLevel::Medium,
            MitigationAction::Block,
            state,
        ).await;
        
        return Ok(());
    }
    
    // Layer 3 checks - Check TTL
    if ipv4.get_ttl() < config.layer3.min_ttl {
        debug!("Detected packet with low TTL from IP: {}", source_ip);
        
        // Add to stats
        record_threat(
            source_ip,
            Some(Layer3AttackType::TTLBasedAttack),
            None,
            None,
            ThreatLevel::Low,
            MitigationAction::Monitor,
            state,
        ).await;
    }
    
    // Add to rate limiter for this IP
    RATE_LIMITER
        .entry(source_ip)
        .or_insert_with(Vec::new)
        .push(SystemTime::now());
    
    // Layer 4 checks
    match ipv4.get_next_level_protocol() {
        IpNextHeaderProtocols::Tcp => {
            if let Some(tcp) = TcpPacket::new(ipv4.payload()) {
                process_tcp_packet(tcp, source_ip, state).await?;
            }
        }
        IpNextHeaderProtocols::Udp => {
            if let Some(udp) = UdpPacket::new(ipv4.payload()) {
                process_udp_packet(udp, source_ip, state).await?;
            }
        }
        IpNextHeaderProtocols::Icmp => {
            process_icmp_packet(ipv4.payload(), source_ip, state).await?;
        }
        _ => {
            // Other protocols
        }
    }
    
    // Update total traffic stats
    let mut stats = state.stats.lock().await;
    stats.total_traffic += ipv4.payload().len() as u64;
    stats.network_metrics.bytes_per_second += ipv4.payload().len() as u64;
    
    Ok(())
}

// Process a TCP packet
async fn process_tcp_packet(tcp: TcpPacket, source_ip: IpAddr, state: &Arc<AppState>) -> Result<()> {
    let config = state.config.lock().await;
    
    // Check if SYN flag is set (potential SYN flood)
    let flags = tcp.get_flags();
    let is_syn = (flags & 0x02) != 0;
    let is_ack = (flags & 0x10) != 0;
    
    if is_syn && !is_ack && config.layer4.enable_syn_protection {
        // Potential SYN flood - count SYN packets for this IP
        let syn_count = RATE_LIMITER
            .entry(source_ip)
            .or_insert_with(Vec::new)
            .len();
        
        // If over threshold, consider it an attack
        if syn_count as u32 > config.layer4.syn_rate_threshold {
            debug!("Detected potential SYN flood from IP: {}", source_ip);
            
            // Add to stats and potentially blacklist
            record_threat(
                source_ip,
                None,
                Some(Layer4AttackType::SynFlood),
                None,
                ThreatLevel::High,
                MitigationAction::Block,
                state,
            ).await;
            
            // Block the IP
            let now = SystemTime::now();
            if let Some(expiry) = now.checked_add(Duration::from_secs(3600)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
            
            return Ok(());
        }
    }
    
    // Check for abnormal TCP flags (potential flag abuse)
    let fin = (flags & 0x01) != 0;
    let rst = (flags & 0x04) != 0;
    let psh = (flags & 0x08) != 0;
    let urg = (flags & 0x20) != 0;
    
    // Detect abnormal flag combinations
    if (fin && syn) || (rst && syn) || (urg && !ack) {
        debug!("Detected abnormal TCP flags from IP: {}", source_ip);
        
        // Add to stats
        record_threat(
            source_ip,
            None,
            Some(Layer4AttackType::AbnormalTcpFlags),
            None,
            ThreatLevel::Medium,
            MitigationAction::Monitor,
            state,
        ).await;
    }
    
    // Update connection stats
    let mut stats = state.stats.lock().await;
    stats.network_metrics.connections_per_second += 1;
    
    // Analyze for Layer 7 if this is a web request (port 80 or 443)
    let dest_port = tcp.get_destination();
    if (dest_port == 80 || dest_port == 443) && config.layer7.enable_http_protection {
        // Extract HTTP request from payload if available
        let payload = tcp.payload();
        if !payload.is_empty() && (payload.starts_with(b"GET ") || payload.starts_with(b"POST ") ||
           payload.starts_with(b"PUT ") || payload.starts_with(b"DELETE ") ||
           payload.starts_with(b"HEAD ") || payload.starts_with(b"OPTIONS ")) {
            
            process_http_request(payload, source_ip, state).await?;
        }
    }
    
    Ok(())
}

// Process a UDP packet
async fn process_udp_packet(udp: UdpPacket, source_ip: IpAddr, state: &Arc<AppState>) -> Result<()> {
    let config = state.config.lock().await;
    
    if config.layer4.enable_udp_protection {
        // Count UDP packets for this IP
        let udp_count = RATE_LIMITER
            .entry(source_ip)
            .or_insert_with(Vec::new)
            .len();
        
        // If over threshold, consider it an attack
        if udp_count as u32 > config.layer4.udp_rate_threshold {
            debug!("Detected potential UDP flood from IP: {}", source_ip);
            
            // Add to stats and potentially blacklist
            record_threat(
                source_ip,
                None,
                Some(Layer4AttackType::UdpFlood),
                None,
                ThreatLevel::High,
                MitigationAction::Block,
                state,
            ).await;
            
            // Block the IP
            let now = SystemTime::now();
            if let Some(expiry) = now.checked_add(Duration::from_secs(1800)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
            
            return Ok(());
        }
    }
    
    // DNS Amplification attack detection (port 53)
    let dest_port = udp.get_destination();
    if dest_port == 53 && config.layer7.enable_dns_protection {
        // Check for DNS query size - large queries could indicate amplification attack
        let payload_size = udp.payload().len();
        if payload_size > 512 {  // Standard DNS query should be small
            debug!("Detected potential DNS amplification from IP: {}", source_ip);
            
            // Add to stats
            record_threat(
                source_ip,
                None,
                None,
                Some(Layer7AttackType::DnsAmplification),
                ThreatLevel::High,
                MitigationAction::Block,
                state,
            ).await;
            
            // Block the IP
            let now = SystemTime::now();
            if let Some(expiry) = now.checked_add(Duration::from_secs(3600)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
        }
    }
    
    Ok(())
}

// Process ICMP packet
async fn process_icmp_packet(icmp_payload: &[u8], source_ip: IpAddr, state: &Arc<AppState>) -> Result<()> {
    let config = state.config.lock().await;
    
    if config.layer4.enable_icmp_protection {
        // Count ICMP packets for this IP
        let icmp_count = RATE_LIMITER
            .entry(source_ip)
            .or_insert_with(Vec::new)
            .len();
        
        // If over threshold, consider it an attack
        if icmp_count as u32 > config.layer4.icmp_rate_threshold {
            debug!("Detected potential ICMP flood from IP: {}", source_ip);
            
            // Add to stats and potentially blacklist
            record_threat(
                source_ip,
                None,
                Some(Layer4AttackType::IcmpFlood),
                None,
                ThreatLevel::Medium,
                MitigationAction::Block,
                state,
            ).await;
            
            // Block the IP
            let now = SystemTime::now();
            if let Some(expiry) = now.checked_add(Duration::from_secs(1800)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
            
            return Ok(());
        }
    }
    
    Ok(())
}

// Process HTTP request (Layer 7)
async fn process_http_request(payload: &[u8], source_ip: IpAddr, state: &Arc<AppState>) -> Result<()> {
    let config = state.config.lock().await;
    
    if !config.layer7.enable_http_protection {
        return Ok(());
    }
    
    // Try to convert payload to string
    if let Ok(request_str) = std::str::from_utf8(payload) {
        // Check for disallowed HTTP methods
        for method in &config.layer7.blocked_http_methods {
            if request_str.starts_with(method) {
                debug!("Blocked HTTP method {} from IP: {}", method, source_ip);
                
                // Add to stats
                record_threat(
                    source_ip,
                    None,
                    None,
                    Some(Layer7AttackType::HttpFlood),
                    ThreatLevel::Medium,
                    MitigationAction::Block,
                    state,
                ).await;
                
                return Ok(());
            }
        }
        
        // Count HTTP requests for this IP
        let http_count = RATE_LIMITER
            .entry(source_ip)
            .or_insert_with(Vec::new)
            .len();
        
        // If over threshold, consider it an attack
        if http_count as u32 > config.layer7.http_request_rate_threshold {
            debug!("Detected HTTP flood from IP: {}", source_ip);
            
            // Add to stats and potentially blacklist
            record_threat(
                source_ip,
                None,
                None,
                Some(Layer7AttackType::HttpFlood),
                ThreatLevel::High,
                MitigationAction::Challenge,
                state,
            ).await;
            
            // Apply rate limiting rather than blocking
            let now = SystemTime::now();
            if let Some(expiry) = now.checked_add(Duration::from_secs(300)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
            
            return Ok(());
        }
        
        // Basic WAF functionality - check for common attack patterns
        if config.layer7.waf_rules_enabled {
            // Check for SQL injection
            if request_str.contains("' OR '1'='1") || 
               request_str.contains("--") || 
               request_str.contains(";--") || 
               request_str.contains("/*") || 
               request_str.contains("*/") {
                debug!("Detected potential SQL injection from IP: {}", source_ip);
                
                record_threat(
                    source_ip,
                    None,
                    None,
                    Some(Layer7AttackType::OWASPAttack("SQL Injection".to_string())),
                    ThreatLevel::High,
                    MitigationAction::Block,
                    state,
                ).await;
                
                return Ok(());
            }
            
            // Check for XSS
            if request_str.contains("<script>") || 
               request_str.contains("javascript:") ||
               request_str.contains("onerror=") ||
               request_str.contains("onload=") {
                debug!("Detected potential XSS from IP: {}", source_ip);
                
                record_threat(
                    source_ip,
                    None,
                    None,
                    Some(Layer7AttackType::OWASPAttack("XSS".to_string())),
                    ThreatLevel::High,
                    MitigationAction::Block,
                    state,
                ).await;
                
                return Ok(());
            }
            
            // Check for path traversal
            if request_str.contains("../") || 
               request_str.contains("..\\") ||
               request_str.contains("..%2f") {
                debug!("Detected potential path traversal from IP: {}", source_ip);
                
                record_threat(
                    source_ip,
                    None,
                    None,
                    Some(Layer7AttackType::OWASPAttack("Path Traversal".to_string())),
                    ThreatLevel::High,
                    MitigationAction::Block,
                    state,
                ).await;
                
                return Ok(());
            }
        }
        
        // Update stats
        let mut stats = state.stats.lock().await;
        let req_type = if request_str.starts_with("GET") {
            "GET"
        } else if request_str.starts_with("POST") {
            "POST"
        } else if request_str.starts_with("PUT") {
            "PUT"
        } else if request_str.starts_with("DELETE") {
            "DELETE"
        } else {
            "OTHER"
        };
        
        *stats.network_metrics.request_distribution.entry(req_type.to_string()).or_insert(0) += 1;
    }
    
    Ok(())
}

// Record a threat for statistics and analysis
async fn record_threat(
    source_ip: IpAddr,
    layer3_attack: Option<Layer3AttackType>,
    layer4_attack: Option<Layer4AttackType>,
    layer7_attack: Option<Layer7AttackType>,
    threat_level: ThreatLevel,
    mitigation_action: MitigationAction,
    state: &Arc<AppState>,
) {
    let now = SystemTime::now();
    
    // Create threat info
    let threat = ThreatInfo {
        source_ip,
        timestamp: now,
        layer3_attack,
        layer4_attack,
        layer7_attack,
        threat_level,
        mitigation_action,
        geo_location: None, // Can be updated with GeoIP lookup
        request_rate: None,
        confidence_score: 0.9,
        is_known_attacker: IP_BLACKLIST.contains_key(&source_ip),
    };
    
    // Add to current threats in stats
    let mut stats = state.stats.lock().await;
    stats.current_threats.push(threat.clone());
    
    // Limit the number of current threats stored
    if stats.current_threats.len() > 1000 {
        stats.current_threats.remove(0);
    }
    
    // Send to ML for analysis if enabled
    let config = state.config.lock().await;
    if config.ml.enable_ml {
        tokio::spawn(async move {
            if let Err(e) = ml_integration::classify_threat(&threat).await {
                error!("Failed to classify threat with ML: {}", e);
            }
        });
    }
    
    // Apply mitigation
    match mitigation_action {
        MitigationAction::Block => {
            // Add to blacklist for 1 hour
            if let Some(expiry) = now.checked_add(Duration::from_secs(3600)) {
                IP_BLACKLIST.insert(source_ip, expiry);
                
                // Create mitigation rule
                let rule_id = uuid::Uuid::new_v4().to_string();
                let rule = crate::types::MitigationRule {
                    rule_id: rule_id.clone(),
                    description: format!("Automatic block for {:?}", threat_level),
                    source_ip: Some(source_ip),
                    source_network: None,
                    destination_port: None,
                    protocol: None,
                    action: mitigation_action,
                    duration: Some(Duration::from_secs(3600)),
                    is_active: true,
                    created_at: now,
                    modified_at: now,
                };
                
                ACTIVE_MITIGATIONS.insert(rule_id, rule);
            }
        }
        MitigationAction::RateLimit => {
            // Add to blacklist for 10 minutes
            if let Some(expiry) = now.checked_add(Duration::from_secs(600)) {
                IP_BLACKLIST.insert(source_ip, expiry);
            }
        }
        _ => {
            // Other mitigation actions can be implemented here
        }
    }
}
