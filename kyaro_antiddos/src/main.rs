mod backend;
mod common;
mod ml;
mod api;
mod types;

use std::{
    collections::HashSet,
    net::IpAddr,
    path::PathBuf,
    sync::Arc,
    time::Duration,
};
use tokio::{
    fs,
    sync::RwLock,
    time,
};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use anyhow::{Context, Result};

use crate::{
    backend::{
        layer3::Layer3Protection,
        layer4::Layer4Protection,
        layer7::Layer7Protection,
        ml_integration,
    },
    api::server::start_api_server,
    types::{Config, PacketInfo, HttpRequest, MitigationAction},
};

// Global blacklist for blocked IPs
pub static IP_BLACKLIST: once_cell::sync::Lazy<Arc<RwLock<HashSet<IpAddr>>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(HashSet::new())));

// Application state
pub struct AppState {
    config: Arc<RwLock<Config>>,
    layer3: Arc<Layer3Protection>,
    layer4: Arc<Layer4Protection>,
    layer7: Arc<Layer7Protection>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    setup_logging()?;
    
    info!("Kyaro Anti-DDoS starting up...");
    
    // Load configuration
    let config_path = PathBuf::from("config.json");
    let config = load_config(&config_path).await?;
    let config = Arc::new(RwLock::new(config));
    
    // Initialize protection layers
    let layer3 = Arc::new(Layer3Protection::new(
        Arc::clone(&config), 
        Arc::clone(&IP_BLACKLIST)
    ));
    
    let layer4 = Arc::new(Layer4Protection::new(
        Arc::clone(&config), 
        Arc::clone(&IP_BLACKLIST)
    ));
    
    let layer7 = Arc::new(Layer7Protection::new(
        Arc::clone(&config), 
        Arc::clone(&IP_BLACKLIST)
    ));
    
    // Create application state
    let state = Arc::new(AppState {
        config: Arc::clone(&config),
        layer3: Arc::clone(&layer3),
        layer4: Arc::clone(&layer4),
        layer7: Arc::clone(&layer7),
    });
    
    // Start ML integration if enabled
    ml_integration::start_ml_integration(Arc::clone(&state)).await?;
    
    // Start packet capture and processing in the background
    tokio::spawn(packet_processing_loop(Arc::clone(&state)));
    
    // Start config reload task in the background
    tokio::spawn(config_reload_task(Arc::clone(&state), config_path));
    
    // Start API server
    let api_port = {
        let config = config.read().await;
        config.api_port
    };
    
    info!("Starting API server on port {}", api_port);
    start_api_server(Arc::clone(&state), api_port).await?;
    
    Ok(())
}

// Setup logging with tracing
fn setup_logging() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .with_file(true)
        .with_line_number(true)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set global default subscriber")?;
    
    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs")
        .context("Failed to create logs directory")?;
    
    Ok(())
}

// Load configuration from JSON file
async fn load_config(path: &PathBuf) -> Result<Config> {
    info!("Loading configuration from {:?}", path);
    
    let config_json = if path.exists() {
        fs::read_to_string(path).await.context("Failed to read config file")?
    } else {
        warn!("Config file not found, using default configuration");
        include_str!("../config.example.json").to_string()
    };
    
    let config: Config = serde_json::from_str(&config_json)
        .context("Failed to parse config JSON")?;
    
    info!("Configuration loaded successfully");
    Ok(config)
}

// Periodically reload config
async fn config_reload_task(state: Arc<AppState>, config_path: PathBuf) {
    let mut interval = time::interval(Duration::from_secs(60)); // Check every minute
    
    loop {
        interval.tick().await;
        
        match load_config(&config_path).await {
            Ok(new_config) => {
                let mut config = state.config.write().await;
                *config = new_config;
                debug!("Configuration reloaded");
            },
            Err(e) => {
                warn!("Failed to reload configuration: {}", e);
            }
        }
    }
}

// Main packet processing loop
async fn packet_processing_loop(state: Arc<AppState>) {
    info!("Starting packet processing loop");
    
    // TODO: Replace with actual packet capture library like libpnet
    // This is a mock implementation for demonstration purposes
    
    let mut interval = time::interval(Duration::from_millis(100));
    
    loop {
        interval.tick().await;
        
        // Simulate packet capture - in a real implementation, this would
        // be replaced with actual packet capture code
        if let Some(packet) = capture_packet().await {
            process_packet(Arc::clone(&state), packet).await;
        }
    }
}

// Mock packet capture function
async fn capture_packet() -> Option<PacketInfo> {
    // In a real implementation, this would capture packets from the network
    // For demo purposes, we'll return None to avoid simulating traffic
    None
}

// Process a captured packet through all protection layers
async fn process_packet(state: Arc<AppState>, packet: PacketInfo) {
    // Process through Layer 3 protection
    match state.layer3.process_packet(&packet).await {
        Ok(MitigationAction::Allow) => {
            // If Layer 3 allows, pass to Layer 4
            match state.layer4.process_packet(&packet).await {
                Ok(MitigationAction::Allow) => {
                    // If Layer 4 allows and this is HTTP traffic, pass to Layer 7
                    if is_http_traffic(&packet) {
                        if let Some(http_request) = extract_http_request(&packet) {
                            if let Err(e) = state.layer7.process_request(http_request).await {
                                error!("Error processing HTTP request: {}", e);
                            }
                        }
                    }
                },
                Ok(action) => {
                    debug!("Layer 4 mitigation applied: {:?} for {}", action, packet.src_ip);
                },
                Err(e) => {
                    error!("Error in Layer 4 processing: {}", e);
                }
            }
        },
        Ok(action) => {
            debug!("Layer 3 mitigation applied: {:?} for {}", action, packet.src_ip);
        },
        Err(e) => {
            error!("Error in Layer 3 processing: {}", e);
        }
    }
}

// Check if packet represents HTTP traffic
fn is_http_traffic(packet: &PacketInfo) -> bool {
    if packet.protocol != "TCP" {
        return false;
    }
    
    // Check for common HTTP ports
    let is_http_port = packet.dst_port.map_or(false, |p| p == 80 || p == 443 || p == 8080 || p == 8443);
    
    // In a real implementation, we would look at the actual packet payload
    // to determine if it's HTTP traffic
    is_http_port
}

// Extract HTTP request from packet
fn extract_http_request(packet: &PacketInfo) -> Option<HttpRequest> {
    // In a real implementation, this would parse the packet payload
    // to extract HTTP request details
    
    // This is a mock implementation - in a real scenario,
    // we would return None for non-HTTP packets
    None
} 