use axum::{
    extract::{Json, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::{net::IpAddr, sync::Arc, time::{Duration, SystemTime}};
use tracing::{error, info};
use uuid::Uuid;

use crate::{
    types::{ApiResponse, DashboardStats, KyaroConfig, MitigationRule, ThreatInfo},
    ACTIVE_MITIGATIONS, IP_BLACKLIST, IP_WHITELIST, AppState,
};

// Get current stats for the dashboard
pub async fn get_stats(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<DashboardStats>>) {
    let stats = state.stats.lock().await.clone();
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Stats retrieved successfully".to_string(),
            data: Some(stats),
        }),
    )
}

// Get current configuration
pub async fn get_config(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<KyaroConfig>>) {
    let config = state.config.lock().await.clone();
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Config retrieved successfully".to_string(),
            data: Some(config),
        }),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigUpdateRequest {
    pub config: KyaroConfig,
}

// Update configuration
pub async fn update_config(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ConfigUpdateRequest>,
) -> (StatusCode, Json<ApiResponse<KyaroConfig>>) {
    let mut config = state.config.lock().await;
    *config = request.config;
    
    info!("Configuration updated");
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Config updated successfully".to_string(),
            data: Some(config.clone()),
        }),
    )
}

// Get current threats
pub async fn get_threats(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<Vec<ThreatInfo>>>) {
    let stats = state.stats.lock().await;
    let threats = stats.current_threats.clone();
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Threats retrieved successfully".to_string(),
            data: Some(threats),
        }),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockIpRequest {
    pub ip: IpAddr,
    pub duration_seconds: Option<u64>,
    pub reason: Option<String>,
}

// Manually block an IP
pub async fn block_ip(
    State(state): State<Arc<AppState>>,
    Json(request): Json<BlockIpRequest>,
) -> (StatusCode, Json<ApiResponse<()>>) {
    let ip = request.ip;
    let now = SystemTime::now();
    
    // Set expiry time if duration is provided
    let expiry = if let Some(duration_secs) = request.duration_seconds {
        match now.checked_add(Duration::from_secs(duration_secs)) {
            Some(time) => time,
            None => {
                error!("Failed to add duration to current time");
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        success: false,
                        message: "Invalid duration".to_string(),
                        data: None,
                    }),
                );
            }
        }
    } else {
        // Default to 24 hours
        match now.checked_add(Duration::from_secs(86400)) {
            Some(time) => time,
            None => {
                error!("Failed to add default duration to current time");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse {
                        success: false,
                        message: "Internal server error".to_string(),
                        data: None,
                    }),
                );
            }
        }
    };
    
    // Add to blacklist
    IP_BLACKLIST.insert(ip, expiry);
    
    // Create mitigation rule
    let rule_id = Uuid::new_v4().to_string();
    let rule = MitigationRule {
        rule_id: rule_id.clone(),
        description: request.reason.unwrap_or_else(|| "Manually blocked".to_string()),
        source_ip: Some(ip),
        source_network: None,
        destination_port: None,
        protocol: None,
        action: crate::types::MitigationAction::Block,
        duration: Some(Duration::from_secs(request.duration_seconds.unwrap_or(86400))),
        is_active: true,
        created_at: now,
        modified_at: now,
    };
    
    ACTIVE_MITIGATIONS.insert(rule_id, rule);
    
    info!("IP {} manually blocked", ip);
    
    // Update stats
    let mut stats = state.stats.lock().await;
    stats.active_mitigations = ACTIVE_MITIGATIONS.len() as u32;
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: format!("IP {} blocked successfully", ip),
            data: None,
        }),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UnblockIpRequest {
    pub ip: IpAddr,
}

// Unblock an IP
pub async fn unblock_ip(
    State(state): State<Arc<AppState>>,
    Json(request): Json<UnblockIpRequest>,
) -> (StatusCode, Json<ApiResponse<()>>) {
    let ip = request.ip;
    
    // Remove from blacklist
    IP_BLACKLIST.remove(&ip);
    
    // Add to whitelist temporarily
    IP_WHITELIST.insert(ip, ());
    
    // Remove mitigation rules for this IP
    ACTIVE_MITIGATIONS.retain(|_, rule| {
        if let Some(rule_ip) = rule.source_ip {
            rule_ip != ip
        } else {
            true
        }
    });
    
    info!("IP {} unblocked", ip);
    
    // Update stats
    let mut stats = state.stats.lock().await;
    stats.active_mitigations = ACTIVE_MITIGATIONS.len() as u32;
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: format!("IP {} unblocked successfully", ip),
            data: None,
        }),
    )
}

// Get active mitigations
pub async fn get_mitigations(
    State(_state): State<Arc<AppState>>,
) -> (StatusCode, Json<ApiResponse<Vec<MitigationRule>>>) {
    let mut mitigations = Vec::new();
    
    for entry in ACTIVE_MITIGATIONS.iter() {
        mitigations.push(entry.value().clone());
    }
    
    (
        StatusCode::OK,
        Json(ApiResponse {
            success: true,
            message: "Mitigations retrieved successfully".to_string(),
            data: Some(mitigations),
        }),
    )
}
