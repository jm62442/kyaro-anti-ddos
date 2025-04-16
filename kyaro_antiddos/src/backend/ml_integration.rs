use anyhow::{Context, Result};
use pyo3::{prelude::*, types::PyDict};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{fs, process::Command as TokioCommand, sync::Mutex, time};
use tracing::{debug, error, info, warn};

use crate::{
    types::{MlRequest, MlResponse, ThreatInfo},
    AppState, IP_BLACKLIST,
};

// Path to the Python module
const ML_MODULE_PATH: &str = "src/ml";
const ML_SCRIPT: &str = "ml_engine.py";

// Initialize and start ML integration
pub async fn start_ml_integration(state: Arc<AppState>) -> Result<()> {
    info!("Starting ML integration...");
    
    // Check if Python is available
    if !is_python_available().await {
        warn!("Python is not available, ML features will be limited");
        return Ok(());
    }
    
    // Create initial ML module if needed
    if let Err(e) = create_ml_module().await {
        warn!("Failed to create ML module: {}", e);
        return Ok(());
    }
    
    // Get configuration
    let config = state.config.lock().await;
    let ml_config = config.ml.clone();
    
    // Start ML processing loop
    if ml_config.enable_ml {
        tokio::spawn(async move {
            let training_interval = Duration::from_secs(ml_config.training_interval_hours as u64 * 3600);
            
            // Initial training
            if let Err(e) = train_model(state.clone()).await {
                error!("Initial ML model training failed: {}", e);
            }
            
            // Periodic training
            let mut interval = time::interval(training_interval);
            loop {
                interval.tick().await;
                
                match train_model(state.clone()).await {
                    Ok(_) => info!("ML model training completed successfully"),
                    Err(e) => error!("ML model training failed: {}", e),
                }
            }
        });
    }
    
    Ok(())
}

// Check if Python is available
async fn is_python_available() -> bool {
    let python_check = tokio::process::Command::new("python")
        .args(["--version"])
        .output()
        .await;
    
    match python_check {
        Ok(output) => output.status.success(),
        Err(_) => {
            // Try python3 if python command fails
            let python3_check = tokio::process::Command::new("python3")
                .args(["--version"])
                .output()
                .await;
            
            match python3_check {
                Ok(output) => output.status.success(),
                Err(_) => false,
            }
        }
    }
}

// Create the ML module if it doesn't exist
async fn create_ml_module() -> Result<()> {
    let ml_path = PathBuf::from(ML_MODULE_PATH);
    let ml_script_path = ml_path.join(ML_SCRIPT);
    
    if !ml_script_path.exists() {
        warn!("ML script not found at {:?}, creating example", ml_script_path);
        
        // Ensure directory exists
        tokio::fs::create_dir_all(&ml_path).await
            .context("Failed to create ML directory")?;
        
        // For now we assume the file has been created separately
        // Return success even if file doesn't exist - a warning has been logged
    }
    
    // Check requirements.txt
    let requirements_path = ml_path.join("requirements.txt");
    if !requirements_path.exists() {
        warn!("ML requirements file not found, please install required packages manually");
    } else {
        // Try to install requirements
        let install_result = tokio::process::Command::new("pip")
            .args(["install", "-r", &requirements_path.to_string_lossy()])
            .output()
            .await;
        
        match install_result {
            Ok(output) => {
                if !output.status.success() {
                    warn!("Failed to install ML requirements: {}", String::from_utf8_lossy(&output.stderr));
                    
                    // Try with pip3 if pip fails
                    let pip3_result = tokio::process::Command::new("pip3")
                        .args(["install", "-r", &requirements_path.to_string_lossy()])
                        .output()
                        .await;
                    
                    if let Ok(output) = pip3_result {
                        if !output.status.success() {
                            warn!("Failed to install ML requirements with pip3: {}", String::from_utf8_lossy(&output.stderr));
                        }
                    }
                }
            },
            Err(e) => warn!("Failed to run pip: {}", e),
        }
    }
    
    Ok(())
}

// Train the ML model
async fn train_model(state: Arc<AppState>) -> Result<()> {
    info!("Training ML model...");
    
    // Get configuration
    let config = state.config.lock().await;
    
    // Direct Python process call to train the model
    let python_cmd = if cfg!(windows) { "python" } else { "python3" };
    
    let training_result = tokio::process::Command::new(python_cmd)
        .args([ML_MODULE_PATH.to_string() + "/" + ML_SCRIPT, "train"])
        .output()
        .await?;
    
    if !training_result.status.success() {
        let error_msg = String::from_utf8_lossy(&training_result.stderr);
        error!("ML model training failed: {}", error_msg);
        return Err(anyhow::anyhow!("ML model training failed: {}", error_msg));
    }
    
    info!("ML model training completed");
    Ok(())
}

// Analyze traffic patterns with ML
pub async fn analyze_traffic_patterns(traffic_data: &[u8]) -> Result<Value> {
    Python::with_gil(|py| {
        // Add the ML module directory to Python path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (ML_MODULE_PATH,))?;
        
        // Import the ML engine module
        let ml_engine = PyModule::import(py, "ml_engine")?;
        
        // Create an instance of the KyaroMLEngine class
        let engine = ml_engine.getattr("KyaroMLEngine")?.call0()?;
        
        // Parse the traffic data
        let json_str = std::str::from_utf8(traffic_data)?;
        let os_json = py.import("json")?;
        let traffic_dict = os_json.call_method1("loads", (json_str,))?;
        
        // Call the analyze_traffic method
        let result = engine.call_method1("analyze_traffic", (traffic_dict,))?;
        
        // Convert result to JSON string
        let result_str = os_json.call_method1("dumps", (result,))?.extract::<String>()?;
        
        // Parse the JSON string to serde_json::Value
        let value: Value = serde_json::from_str(&result_str)?;
        
        Ok(value)
    }).map_err(|e| anyhow::anyhow!("Python error: {}", e))
}

// Classify a threat with ML
pub async fn classify_threat(threat: &ThreatInfo) -> Result<()> {
    // Convert the threat to a JSON Value
    let threat_value = extract_features_from_threat(threat);
    
    Python::with_gil(|py| {
        // Add the ML module directory to Python path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (ML_MODULE_PATH,))?;
        
        // Import the ML engine module
        let ml_engine = PyModule::import(py, "ml_engine")?;
        
        // Create an instance of the KyaroMLEngine class
        let engine = ml_engine.getattr("KyaroMLEngine")?.call0()?;
        
        // Create a Python dictionary from the threat data
        let os_json = py.import("json")?;
        let threat_json = serde_json::to_string(&threat_value)?;
        let threat_dict = os_json.call_method1("loads", (threat_json,))?;
        
        // Call the classify_threat method
        let result = engine.call_method1("classify_threat", (threat_dict,))?;
        
        // Log the result
        let result_str = os_json.call_method1("dumps", (result,))?.extract::<String>()?;
        info!("ML threat classification result: {}", result_str);
        
        Ok(())
    }).map_err(|e| anyhow::anyhow!("Python error: {}", e))
}

// Extract features from a threat for ML processing
fn extract_features_from_threat(threat: &ThreatInfo) -> serde_json::Value {
    let mut features = json!({
        "source_ip": threat.source_ip.to_string(),
        "timestamp": format!("{:?}", threat.timestamp),
        "threat_level": format!("{:?}", threat.threat_level),
        "mitigation_action": format!("{:?}", threat.mitigation_action),
        "geo_location": threat.geo_location,
        "request_rate": threat.request_rate,
        "confidence_score": threat.confidence_score,
        "is_known_attacker": threat.is_known_attacker,
    });
    
    // Add layer-specific attack types
    if let Some(attack) = &threat.layer3_attack {
        features["layer3_attack"] = json!(format!("{:?}", attack));
        features["attack_layer"] = json!("Layer3");
    }
    
    if let Some(attack) = &threat.layer4_attack {
        features["layer4_attack"] = json!(format!("{:?}", attack));
        features["attack_layer"] = json!("Layer4");
    }
    
    if let Some(attack) = &threat.layer7_attack {
        features["layer7_attack"] = json!(format!("{:?}", attack));
        features["attack_layer"] = json!("Layer7");
    }
    
    features
}
