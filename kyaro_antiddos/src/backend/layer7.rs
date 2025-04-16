use anyhow::Result;
use regex::Regex;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use base64::decode;
use lazy_static::lazy_static;

use crate::{
    types::{HttpRequest, Layer7Config, MitigationAction, ThreatInfo, ThreatLevel, Layer7AttackType},
    AppState, IP_BLACKLIST, 
};

// Time window for rate limiting (in seconds)
const RATE_LIMIT_WINDOW: u64 = 60;
// Maximum number of requests to store in history per IP
const MAX_REQUEST_HISTORY: usize = 100;

// Rules for Web Application Firewall (WAF)
lazy_static! {
    // SQL Injection patterns
    static ref SQL_INJECTION_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"(?i:(?:')(?:(?:\s)|(?:/\*.*?\*/)|(?:--[^\r\n]*)|(?:#.*?\n))*(?:(?:OR|AND)\s+[\w]+(?:\s*=|\s*<>|\s*>|\s*<|\s*LIKE|\s*IS\s+NULL|\s+IS\s+NOT\s+NULL)))").unwrap(),
        Regex::new(r"(?i:UNION[\s/\*]+SELECT)").unwrap(),
        Regex::new(r"(?i:(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\s+)").unwrap(),
    ];
    
    // XSS patterns
    static ref XSS_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"(?i:<script[^>]*>[\s\S]*?<\/script>)").unwrap(),
        Regex::new(r"(?i:javascript:)").unwrap(),
        Regex::new(r"(?i:on\w+\s*=)").unwrap(),
        Regex::new(r"(?i:<[^>]*\s+[^>]*\b(?:on\w+)\s*=)").unwrap(),
    ];
    
    // Path traversal
    static ref PATH_TRAVERSAL_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"(?i:(?:\.\./|\.\.\\|%2e%2e[\\/]|\.\.%2f|%2e%2e%2f|\.\.%5c|%2e%2e%5c|\./\./))").unwrap(),
    ];
    
    // Command injection
    static ref COMMAND_INJECTION_PATTERNS: Vec<Regex> = vec![
        Regex::new(r"(?i:;|\$\(|\|\||&&|`|\$\{)").unwrap(),
    ];
}

// HTTP Request history for an IP
struct RequestHistory {
    requests: VecDeque<HttpRequest>,
    by_path: HashMap<String, u32>,
    by_method: HashMap<String, u32>,
    by_status: HashMap<u16, u32>,
    user_agents: HashSet<String>,
    first_seen: Instant,
    last_seen: Instant,
    waf_blocks: u32,
    bot_score: f64,
}

impl RequestHistory {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            requests: VecDeque::with_capacity(MAX_REQUEST_HISTORY),
            by_path: HashMap::new(),
            by_method: HashMap::new(),
            by_status: HashMap::new(),
            user_agents: HashSet::new(),
            first_seen: now,
            last_seen: now,
            waf_blocks: 0,
            bot_score: 0.0,
        }
    }
    
    fn add_request(&mut self, request: HttpRequest) {
        self.last_seen = Instant::now();
        
        // Update path stats
        *self.by_path.entry(request.path.clone()).or_insert(0) += 1;
        
        // Update method stats
        *self.by_method.entry(request.method.clone()).or_insert(0) += 1;
        
        // Update status stats if status is set
        if let Some(status) = request.status {
            *self.by_status.entry(status).or_insert(0) += 1;
        }
        
        // Update user agent set
        if let Some(user_agent) = &request.user_agent {
            self.user_agents.insert(user_agent.clone());
        }
        
        // Add to request history
        if self.requests.len() >= MAX_REQUEST_HISTORY {
            self.requests.pop_front();
        }
        self.requests.push_back(request);
    }
    
    fn request_rate(&self) -> f64 {
        let elapsed = self.last_seen.duration_since(self.first_seen).as_secs_f64();
        if elapsed > 0.0 {
            self.requests.len() as f64 / elapsed
        } else {
            self.requests.len() as f64
        }
    }
    
    fn path_entropy(&self) -> f64 {
        if self.by_path.is_empty() {
            return 0.0;
        }
        
        let total = self.requests.len() as f64;
        let mut entropy = 0.0;
        
        for &count in self.by_path.values() {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
        
        // Normalize entropy to a 0-1 range
        let max_entropy = (self.by_path.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }
    
    fn get_status_ratio(&self, status_range: (u16, u16)) -> f64 {
        let (start, end) = status_range;
        let total = self.requests.len() as f64;
        
        if total == 0.0 {
            return 0.0;
        }
        
        let matching_count = self.by_status
            .iter()
            .filter(|(&status, _)| status >= start && status <= end)
            .map(|(_, &count)| count)
            .sum::<u32>() as f64;
        
        matching_count / total
    }
    
    fn calculate_bot_score(&mut self) -> f64 {
        let mut score = 0.0;
        
        // Consider user agent diversity (bots often use a single user agent)
        if !self.user_agents.is_empty() && self.requests.len() > 10 {
            let agent_ratio = self.user_agents.len() as f64 / self.requests.len().min(20) as f64;
            
            // Lower ratio suggests bot behavior
            if agent_ratio < 0.1 {
                score += 0.3;
            } else if agent_ratio < 0.3 {
                score += 0.15;
            }
        }
        
        // Consider request rate (high rates suggest automated behavior)
        let rate = self.request_rate();
        if rate > 10.0 {
            score += 0.3;
        } else if rate > 5.0 {
            score += 0.15;
        }
        
        // Consider error rates (bots often generate more errors)
        let error_ratio = self.get_status_ratio((400, 499));
        if error_ratio > 0.5 {
            score += 0.2;
        } else if error_ratio > 0.2 {
            score += 0.1;
        }
        
        // Consider path entropy (low entropy suggests automated targeting)
        let entropy = self.path_entropy();
        if entropy < 0.3 && self.requests.len() > 10 {
            score += 0.2;
        }
        
        // Save the score
        self.bot_score = score;
        score
    }
    
    fn is_expired(&self, window: Duration) -> bool {
        self.last_seen.elapsed() > window
    }
}

// Layer 7 protection service
pub struct Layer7Protection {
    request_history: RwLock<HashMap<IpAddr, RequestHistory>>,
    config: Arc<RwLock<Layer7Config>>,
    ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>,
    last_cleanup: RwLock<Instant>,
}

impl Layer7Protection {
    pub fn new(config: Arc<RwLock<Layer7Config>>, ip_blacklist: Arc<RwLock<HashSet<IpAddr>>>) -> Self {
        Self {
            request_history: RwLock::new(HashMap::new()),
            config,
            ip_blacklist,
            last_cleanup: RwLock::new(Instant::now()),
        }
    }
    
    // Process an HTTP request for Layer 7 protection
    pub async fn process_request(&self, request: HttpRequest) -> Result<MitigationAction> {
        // Check if the IP is blacklisted
        let is_blacklisted = {
            let blacklist = self.ip_blacklist.read().await;
            blacklist.contains(&request.client_ip)
        };
        
        if is_blacklisted {
            return Ok(MitigationAction::Drop);
        }
        
        let config = self.config.read().await;
        
        // Check if the HTTP method is blocked
        if config.blocked_http_methods.contains(&request.method) {
            debug!("Blocked request with forbidden HTTP method {} from {}", request.method, request.client_ip);
            return Ok(MitigationAction::Drop);
        }
        
        // Check WAF rules if enabled
        if config.waf_rules_enabled {
            if let Some(attack_type) = self.check_waf_rules(&request).await {
                warn!("WAF blocked request from {}: {:?}", request.client_ip, attack_type);
                
                // Record the WAF block in request history
                let mut history = self.request_history.write().await;
                let ip_history = history
                    .entry(request.client_ip)
                    .or_insert_with(RequestHistory::new);
                
                ip_history.waf_blocks += 1;
                
                return Ok(MitigationAction::Drop);
            }
        }
        
        // Update request history
        self.update_history(request.clone()).await;
        
        // Check for potential threats
        if let Some(threat) = self.detect_threats(&request).await {
            // Apply appropriate mitigation based on threat level
            match threat.threat_level {
                ThreatLevel::Low => {
                    // Just monitor for low-level threats
                    debug!("Low-level Layer 7 threat detected from {}: {:?}", request.client_ip, threat);
                    Ok(MitigationAction::Monitor)
                },
                ThreatLevel::Medium => {
                    // Rate limit medium-level threats or challenge them
                    info!("Medium-level Layer 7 threat detected from {}: {:?}", request.client_ip, threat);
                    
                    if config.challenge_suspicious_requests {
                        Ok(MitigationAction::Challenge)
                    } else {
                        Ok(MitigationAction::RateLimit)
                    }
                },
                ThreatLevel::High | ThreatLevel::Critical => {
                    // Block high and critical threats
                    warn!("High-level Layer 7 threat detected from {}: {:?}", request.client_ip, threat);
                    
                    // Add to blacklist
                    let mut blacklist = self.ip_blacklist.write().await;
                    blacklist.insert(request.client_ip);
                    info!("Added {} to IP blacklist", request.client_ip);
                    
                    Ok(MitigationAction::Drop)
                }
            }
        } else {
            // No threat detected, allow the request
            Ok(MitigationAction::Allow)
        }
    }
    
    // Check Web Application Firewall rules
    async fn check_waf_rules(&self, request: &HttpRequest) -> Option<Layer7AttackType> {
        // Combine URL path and query parameters for checking
        let url = if let Some(ref query) = request.query_string {
            format!("{}?{}", request.path, query)
        } else {
            request.path.clone()
        };
        
        // Check URL patterns
        for pattern in SQL_INJECTION_PATTERNS.iter() {
            if pattern.is_match(&url) {
                return Some(Layer7AttackType::SqlInjection);
            }
        }
        
        for pattern in XSS_PATTERNS.iter() {
            if pattern.is_match(&url) {
                return Some(Layer7AttackType::CrossSiteScripting);
            }
        }
        
        for pattern in PATH_TRAVERSAL_PATTERNS.iter() {
            if pattern.is_match(&url) {
                return Some(Layer7AttackType::PathTraversal);
            }
        }
        
        for pattern in COMMAND_INJECTION_PATTERNS.iter() {
            if pattern.is_match(&url) {
                return Some(Layer7AttackType::CommandInjection);
            }
        }
        
        // Check request body if available
        if let Some(ref body) = request.body {
            // Check if it's base64 encoded
            let body_content = if let Ok(decoded) = decode(body) {
                if let Ok(text) = String::from_utf8(decoded) {
                    text
                } else {
                    body.clone()
                }
            } else {
                body.clone()
            };
            
            // Run pattern checks on body content
            for pattern in SQL_INJECTION_PATTERNS.iter() {
                if pattern.is_match(&body_content) {
                    return Some(Layer7AttackType::SqlInjection);
                }
            }
            
            for pattern in XSS_PATTERNS.iter() {
                if pattern.is_match(&body_content) {
                    return Some(Layer7AttackType::CrossSiteScripting);
                }
            }
            
            for pattern in COMMAND_INJECTION_PATTERNS.iter() {
                if pattern.is_match(&body_content) {
                    return Some(Layer7AttackType::CommandInjection);
                }
            }
        }
        
        // Check headers for suspicious patterns
        for (header_name, header_value) in &request.headers {
            // Special checks for common headers
            if header_name.eq_ignore_ascii_case("Referer") || 
               header_name.eq_ignore_ascii_case("User-Agent") ||
               header_name.eq_ignore_ascii_case("Cookie") {
                
                for pattern in SQL_INJECTION_PATTERNS.iter() {
                    if pattern.is_match(header_value) {
                        return Some(Layer7AttackType::SqlInjection);
                    }
                }
                
                for pattern in XSS_PATTERNS.iter() {
                    if pattern.is_match(header_value) {
                        return Some(Layer7AttackType::CrossSiteScripting);
                    }
                }
            }
        }
        
        None
    }
    
    // Update request history for an IP
    async fn update_history(&self, request: HttpRequest) -> () {
        let mut history = self.request_history.write().await;
        
        // Create or update history for this IP
        let ip_history = history
            .entry(request.client_ip)
            .or_insert_with(RequestHistory::new);
        
        ip_history.add_request(request);
        
        // Recalculate bot score
        ip_history.calculate_bot_score();
        
        // Clean up old entries occasionally
        let should_cleanup = {
            let last_cleanup = self.last_cleanup.read().await;
            last_cleanup.elapsed() > Duration::from_secs(300) // Every 5 minutes
        };
        
        if should_cleanup {
            self.cleanup_old_history().await;
        }
    }
    
    // Clean up old history entries
    async fn cleanup_old_history(&self) {
        let mut last_cleanup = self.last_cleanup.write().await;
        *last_cleanup = Instant::now();
        
        let mut history = self.request_history.write().await;
        
        // Remove entries older than the rate limit window
        let window = Duration::from_secs(RATE_LIMIT_WINDOW * 2); // Give some extra time
        history.retain(|_, h| !h.is_expired(window));
        
        debug!("Cleaned up old Layer 7 history, remaining entries: {}", history.len());
    }
    
    // Detect Layer 7 threats from request patterns
    async fn detect_threats(&self, request: &HttpRequest) -> Option<ThreatInfo> {
        let config = self.config.read().await;
        let history = self.request_history.read().await;
        
        let ip_history = match history.get(&request.client_ip) {
            Some(h) => h,
            None => return None, // No history yet for this IP
        };
        
        let mut threat_indicators = Vec::new();
        let mut threat_level = ThreatLevel::Low;
        let mut is_attack = false;
        let mut attack_type: Option<Layer7AttackType> = None;
        
        // HTTP flood detection
        if config.enable_http_protection {
            let request_rate = ip_history.request_rate();
            if request_rate > config.http_request_rate_threshold as f64 {
                threat_indicators.push(format!("HTTP flood detected: {:.2} req/s", request_rate));
                threat_level = ThreatLevel::Medium;
                is_attack = true;
                attack_type = Some(Layer7AttackType::HttpFlood);
                
                // Upgrade threat level for severe HTTP flood
                if request_rate > config.http_request_rate_threshold as f64 * 5.0 {
                    threat_level = ThreatLevel::High;
                }
            }
        }
        
        // Bot detection
        if config.bot_protection_level > 0 {
            let bot_score = ip_history.bot_score;
            let detection_threshold = match config.bot_protection_level {
                1 => 0.7, // Low - detect only obvious bots
                2 => 0.5, // Medium
                3 => 0.3, // High - more aggressive detection
                _ => 0.5, // Default to medium
            };
            
            if bot_score > detection_threshold {
                threat_indicators.push(format!("Bot activity detected (score: {:.2})", bot_score));
                threat_level = threat_level.max(ThreatLevel::Medium);
                is_attack = true;
                attack_type = Some(Layer7AttackType::BotActivity);
                
                // Upgrade threat level for high confidence bot detection
                if bot_score > 0.7 {
                    threat_level = ThreatLevel::High;
                }
            }
        }
        
        // WAF block history
        if ip_history.waf_blocks > 0 {
            threat_indicators.push(format!("Multiple WAF blocks: {}", ip_history.waf_blocks));
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
            
            // Attack type is already determined by WAF
            if attack_type.is_none() {
                attack_type = Some(Layer7AttackType::WafViolation);
            }
            
            // Escalate threat level for repeated WAF violations
            if ip_history.waf_blocks > 5 {
                threat_level = ThreatLevel::High;
            }
        }
        
        // Check for API abuse (high rate of 4xx errors)
        let error_ratio = ip_history.get_status_ratio((400, 499));
        if error_ratio > 0.5 && ip_history.requests.len() > 10 {
            threat_indicators.push(format!("High error rate: {:.2}%", error_ratio * 100.0));
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
            attack_type = Some(Layer7AttackType::ApiAbuse);
        }
        
        // Check for Slow HTTP attacks (incomplete request bodies, long headers)
        if request.method == "POST" && 
           request.body.as_ref().map_or(false, |b| b.is_empty()) && 
           request.headers.get("Content-Length").map_or(0, |c| c.parse::<usize>().unwrap_or(0)) > 1000 {
            
            threat_indicators.push("Potential Slow HTTP attack".to_string());
            threat_level = threat_level.max(ThreatLevel::Medium);
            is_attack = true;
            attack_type = Some(Layer7AttackType::SlowHttpAttack);
        }
        
        // Return threat info if attack indicators were found
        if is_attack {
            let threat_info = ThreatInfo {
                source_ip: request.client_ip,
                timestamp: SystemTime::now(),
                threat_level,
                layer3_attack: None,
                layer4_attack: None,
                layer7_attack: attack_type,
                mitigation_action: match threat_level {
                    ThreatLevel::Low => MitigationAction::Monitor,
                    ThreatLevel::Medium => {
                        if config.challenge_suspicious_requests {
                            MitigationAction::Challenge
                        } else {
                            MitigationAction::RateLimit
                        }
                    },
                    _ => MitigationAction::Drop,
                },
                geo_location: "Unknown".to_string(),
                request_rate: ip_history.request_rate(),
                confidence_score: 0.9,
                details: threat_indicators.join(", "),
                is_known_attacker: false,
            };
            
            Some(threat_info)
        } else {
            None
        }
    }
    
    // Get Layer 7 stats for monitoring and reporting
    pub async fn get_request_stats(&self) -> HashMap<IpAddr, HashMap<String, serde_json::Value>> {
        let history = self.request_history.read().await;
        let mut result = HashMap::new();
        
        for (ip, ip_history) in history.iter() {
            let mut metrics = HashMap::new();
            
            // Add numeric stats
            metrics.insert("request_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_history.request_rate()).unwrap_or(serde_json::Number::from(0))));
            
            metrics.insert("bot_score".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_history.bot_score).unwrap_or(serde_json::Number::from(0))));
            
            metrics.insert("total_requests".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from(ip_history.requests.len())));
            
            metrics.insert("waf_blocks".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from(ip_history.waf_blocks)));
            
            metrics.insert("error_rate".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_history.get_status_ratio((400, 599))).unwrap_or(serde_json::Number::from(0))));
            
            metrics.insert("path_entropy".to_string(), 
                          serde_json::Value::Number(serde_json::Number::from_f64(ip_history.path_entropy()).unwrap_or(serde_json::Number::from(0))));
            
            // Add HTTP methods distribution
            let methods: serde_json::Value = serde_json::to_value(
                &ip_history.by_method
            ).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            
            metrics.insert("methods".to_string(), methods);
            
            // Add top paths
            let mut path_counts: Vec<(String, u32)> = ip_history.by_path
                .iter()
                .map(|(path, &count)| (path.clone(), count))
                .collect();
            
            path_counts.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count, descending
            
            let top_paths: serde_json::Value = serde_json::to_value(
                path_counts.iter()
                    .take(10)
                    .map(|(path, count)| (path.clone(), *count))
                    .collect::<HashMap<String, u32>>()
            ).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            
            metrics.insert("top_paths".to_string(), top_paths);
            
            // Add status code distribution
            let status_codes: serde_json::Value = serde_json::to_value(
                &ip_history.by_status
            ).unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
            
            metrics.insert("status_codes".to_string(), status_codes);
            
            result.insert(*ip, metrics);
        }
        
        result
    }
} 