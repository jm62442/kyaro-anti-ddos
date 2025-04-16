use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use clap::Parser;
use dashmap::DashMap;
use ipnet::IpNet;
use once_cell::sync::Lazy;
use pnet::datalink;
use serde_json::json;
use std::{collections::HashMap, net::IpAddr, sync::Arc, time::{Duration, SystemTime}};
use tokio::{net::TcpListener, sync::Mutex, time};
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod api;
mod defense;
mod firewall;
mod ml_integration;
mod monitoring;
mod utils;

// Define paths to include our common types
#[path = "../common/types.rs"]
mod types;
use types::*;

// Global state for the application
static IP_BLACKLIST: Lazy<DashMap<IpAddr, SystemTime>> = Lazy::new(DashMap::new);
static IP_WHITELIST: Lazy<DashMap<IpAddr, ()>> = Lazy::new(DashMap::new);
static RATE_LIMITER: Lazy<DashMap<IpAddr, Vec<SystemTime>>> = Lazy::new(DashMap::new);
static ACTIVE_MITIGATIONS: Lazy<DashMap<String, MitigationRule>> = Lazy::new(DashMap::new);

// Command line arguments for the application
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, default_value = "config.json")]
    config: String,

    #[clap(short, long, default_value = "6868")]
    port: u16,

    #[clap(short, long, default_value = "info")]
    log_level: String,
}

// Application state
struct AppState {
    config: Arc<Mutex<KyaroConfig>>,
    stats: Arc<Mutex<DashboardStats>>,
}

// Main function
#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| format!("{}=info", env!("CARGO_PKG_NAME"))),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting Kyaro Anti-DDoS v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = match std::fs::read_to_string(&args.config) {
        Ok(content) => {
            match serde_json::from_str::<KyaroConfig>(&content) {
                Ok(config) => config,
                Err(e) => {
                    warn!("Failed to parse config file: {}, using default config", e);
                    create_default_config()
                }
            }
        }
        Err(e) => {
            warn!("Failed to read config file: {}, using default config", e);
            create_default_config()
        }
    };

    // Initialize application state
    let app_state = Arc::new(AppState {
        config: Arc::new(Mutex::new(config)),
        stats: Arc::new(Mutex::new(DashboardStats {
            total_traffic: 0,
            blocked_attacks: 0,
            active_mitigations: 0,
            current_threats: Vec::new(),
            network_metrics: NetworkMetrics {
                packets_per_second: 0,
                bytes_per_second: 0,
                connections_per_second: 0,
                request_distribution: HashMap::new(),
            },
            top_attack_sources: Vec::new(),
            top_attack_types: Vec::new(),
            attack_trend: Vec::new(),
        })),
    });

    // Initialize network interfaces
    info!("Initializing network interfaces...");
    let interfaces = datalink::interfaces();
    for interface in interfaces.iter() {
        info!("Found interface: {}", interface.name);
    }

    // Start the packet capture and analysis task
    let packet_state = app_state.clone();
    tokio::spawn(async move {
        if let Err(e) = defense::start_packet_analysis(packet_state).await {
            error!("Packet analysis task failed: {}", e);
        }
    });

    // Start the ML integration task
    let ml_state = app_state.clone();
    tokio::spawn(async move {
        if let Err(e) = ml_integration::start_ml_integration(ml_state).await {
            error!("ML integration task failed: {}", e);
        }
    });

    // Start the cleaner task to remove expired entries
    let cleaner_state = app_state.clone();
    tokio::spawn(async move {
        loop {
            clean_expired_entries();
            tokio::time::sleep(Duration::from_secs(60)).await;
            
            // Update stats periodically
            let mut stats = cleaner_state.stats.lock().await;
            stats.active_mitigations = ACTIVE_MITIGATIONS.len() as u32;
            stats.attack_trend.push((SystemTime::now(), stats.blocked_attacks));
            
            // Limit the trend data to last 24 hours
            if stats.attack_trend.len() > 1440 {
                stats.attack_trend.remove(0);
            }
        }
    });

    // Create the API router
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/stats", get(api::get_stats))
        .route("/api/config", get(api::get_config))
        .route("/api/config", post(api::update_config))
        .route("/api/threats", get(api::get_threats))
        .route("/api/block", post(api::block_ip))
        .route("/api/unblock", post(api::unblock_ip))
        .route("/api/mitigations", get(api::get_mitigations))
        .layer(cors)
        .with_state(app_state.clone());

    // Start the API server
    let listener = TcpListener::bind(format!("0.0.0.0:{}", args.port))
        .await
        .with_context(|| format!("Failed to bind to port {}", args.port))?;
    
    info!("API server listening on port {}", args.port);
    axum::serve(listener, app).await?;

    Ok(())
}

// Clean expired entries from blacklists and rate limiters
fn clean_expired_entries() {
    let now = SystemTime::now();
    
    // Clean blacklist
    IP_BLACKLIST.retain(|_, &mut expiry| {
        expiry > now
    });
    
    // Clean rate limiter
    RATE_LIMITER.retain(|_, timestamps| {
        // Remove timestamps older than 1 minute
        timestamps.retain(|&ts| {
            match now.duration_since(ts) {
                Ok(duration) => duration.as_secs() < 60,
                Err(_) => false,
            }
        });
        !timestamps.is_empty()
    });
    
    // Clean active mitigations
    ACTIVE_MITIGATIONS.retain(|_, rule| {
        if let Some(duration) = rule.duration {
            match rule.created_at.checked_add(duration) {
                Some(expiry) => expiry > now,
                None => true,
            }
        } else {
            true // Rules without duration don't expire
        }
    });
}

// Create default configuration
fn create_default_config() -> KyaroConfig {
    KyaroConfig {
        layer3: Layer3Config {
            enable_ip_blacklisting: true,
            enable_geo_blocking: true,
            enable_rate_limiting: true,
            max_packet_size: 65535,
            min_ttl: 5,
            blocked_countries: Vec::new(),
            rate_limit_threshold: 1000,
        },
        layer4: Layer4Config {
            enable_syn_protection: true,
            enable_udp_protection: true,
            enable_icmp_protection: true,
            syn_rate_threshold: 500,
            udp_rate_threshold: 500,
            icmp_rate_threshold: 100,
            max_concurrent_connections: 1000,
            blocked_ports: Vec::new(),
        },
        layer7: Layer7Config {
            enable_http_protection: true,
            enable_dns_protection: true,
            enable_ssl_protection: true,
            http_request_rate_threshold: 100,
            blocked_http_methods: vec!["TRACE".to_string(), "TRACK".to_string()],
            challenge_suspicious_requests: true,
            bot_protection_level: 2,
            waf_rules_enabled: true,
        },
        ml: MlConfig {
            enable_ml: true,
            training_interval_hours: 24,
            detection_threshold: 0.85,
            anomaly_sensitivity: 0.7,
        },
        log_level: "info".to_string(),
        data_retention_days: 30,
        api_port: 6868,
    }
}
