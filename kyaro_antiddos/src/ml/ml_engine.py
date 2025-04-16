#!/usr/bin/env python3
# Kyaro Anti-DDoS Machine Learning Engine
# This module handles the AI/ML aspects of DDoS detection and analysis

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'ml_engine.log'))
    ]
)
logger = logging.getLogger("kyaro_ml")

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Model file paths
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.joblib')
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, 'classifier_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Attack types
LAYER3_ATTACKS = ['IpSpoofing', 'FragmentationAttack', 'AbnormalPacketSize', 'TTLBasedAttack', 'UnusualIPOptions']
LAYER4_ATTACKS = ['SynFlood', 'UdpFlood', 'IcmpFlood', 'AbnormalTcpFlags', 'ConnectionFlood', 'SlowLoris', 'TearDrop']
LAYER7_ATTACKS = ['HttpFlood', 'SlowHttpAttack', 'DnsAmplification', 'SslAbuse', 'ApiAbuse', 'WebScraping', 'BotActivity']

class KyaroMLEngine:
    """
    Main class for Kyaro Anti-DDoS Machine Learning Engine.
    Handles model training, prediction, and anomaly detection.
    """
    
    def __init__(self, training_interval_hours=24, detection_threshold=0.85, anomaly_sensitivity=0.7):
        """
        Initialize the ML engine with configuration parameters.
        
        Args:
            training_interval_hours: Hours between model retraining
            detection_threshold: Threshold for attack classification
            anomaly_sensitivity: Sensitivity for anomaly detection
        """
        self.training_interval_hours = training_interval_hours
        self.detection_threshold = detection_threshold
        self.anomaly_sensitivity = anomaly_sensitivity
        
        self.anomaly_model = None
        self.classifier_model = None
        self.scaler = None
        
        # Load existing models if available
        self._load_models()
        
        # Start periodic training thread if models don't exist
        if self.classifier_model is None or self.anomaly_model is None:
            self._initial_training()
        
        # Start background training thread
        self.training_thread = threading.Thread(target=self._periodic_training, daemon=True)
        self.training_thread.start()
        
        logger.info("Kyaro ML Engine initialized")
    
    def _load_models(self) -> None:
        """Load trained models from disk if they exist."""
        try:
            if os.path.exists(ANOMALY_MODEL_PATH):
                self.anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
                logger.info("Loaded anomaly model from disk")
            
            if os.path.exists(CLASSIFIER_MODEL_PATH):
                self.classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
                logger.info("Loaded classifier model from disk")
            
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Loaded scaler from disk")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.anomaly_model = None
            self.classifier_model = None
            self.scaler = None
    
    def _save_models(self) -> None:
        """Save trained models to disk."""
        try:
            if self.anomaly_model:
                joblib.dump(self.anomaly_model, ANOMALY_MODEL_PATH)
            
            if self.classifier_model:
                joblib.dump(self.classifier_model, CLASSIFIER_MODEL_PATH)
            
            if self.scaler:
                joblib.dump(self.scaler, SCALER_PATH)
                
            logger.info("Saved models to disk")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _initial_training(self) -> None:
        """Perform initial training if no models exist."""
        logger.info("Performing initial model training")
        
        # Generate synthetic data for initial training if no real data exists
        if not self._has_training_data():
            self._generate_synthetic_data()
        
        self._train_models()
    
    def _periodic_training(self) -> None:
        """Periodically retrain models based on the configured interval."""
        while True:
            # Sleep for the configured interval
            time.sleep(self.training_interval_hours * 3600)
            
            logger.info("Starting periodic model retraining")
            self._train_models()
    
    def _has_training_data(self) -> bool:
        """Check if we have sufficient training data."""
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        return len(data_files) > 0
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data for initial training."""
        logger.info("Generating synthetic training data")
        
        # Number of samples to generate
        n_samples = 10000
        
        # Generate normal traffic data (70%)
        normal_samples = int(n_samples * 0.7)
        
        # Features for normal traffic
        normal_data = {
            'packet_rate': np.random.normal(500, 100, normal_samples),
            'byte_rate': np.random.normal(50000, 10000, normal_samples),
            'packet_size_mean': np.random.normal(800, 200, normal_samples),
            'packet_size_std': np.random.normal(100, 30, normal_samples),
            'new_conn_rate': np.random.normal(20, 5, normal_samples),
            'active_conn': np.random.normal(100, 30, normal_samples),
            'syn_rate': np.random.normal(15, 5, normal_samples),
            'fin_rate': np.random.normal(10, 3, normal_samples),
            'rst_rate': np.random.normal(5, 2, normal_samples),
            'ack_rate': np.random.normal(450, 100, normal_samples),
            'http_req_rate': np.random.normal(50, 15, normal_samples),
            'dns_req_rate': np.random.normal(20, 10, normal_samples),
            'unique_src_ips': np.random.normal(50, 15, normal_samples),
            'ttl_mean': np.random.normal(64, 10, normal_samples),
            'attack_type': ['Normal'] * normal_samples,
            'is_attack': [0] * normal_samples
        }
        
        # Generate attack data (30%)
        attack_samples = n_samples - normal_samples
        
        # All attack types 
        all_attack_types = LAYER3_ATTACKS + LAYER4_ATTACKS + LAYER7_ATTACKS
        
        # Features for attack traffic - higher rates, more variance
        attack_data = {
            'packet_rate': np.random.normal(5000, 2000, attack_samples),
            'byte_rate': np.random.normal(500000, 200000, attack_samples),
            'packet_size_mean': np.random.normal(600, 300, attack_samples),
            'packet_size_std': np.random.normal(400, 100, attack_samples),
            'new_conn_rate': np.random.normal(200, 100, attack_samples),
            'active_conn': np.random.normal(1000, 500, attack_samples),
            'syn_rate': np.random.normal(500, 300, attack_samples),
            'fin_rate': np.random.normal(50, 30, attack_samples),
            'rst_rate': np.random.normal(100, 50, attack_samples),
            'ack_rate': np.random.normal(4000, 2000, attack_samples),
            'http_req_rate': np.random.normal(500, 300, attack_samples),
            'dns_req_rate': np.random.normal(200, 150, attack_samples),
            'unique_src_ips': np.random.normal(10, 5, attack_samples),  # Often fewer source IPs in attacks
            'ttl_mean': np.random.normal(40, 20, attack_samples),
            'attack_type': np.random.choice(all_attack_types, attack_samples),
            'is_attack': [1] * attack_samples
        }
        
        # Combine normal and attack data
        data = {}
        for key in normal_data.keys():
            data[key] = np.concatenate([normal_data[key], attack_data[key]])
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(data)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        df.to_csv(os.path.join(DATA_DIR, f'synthetic_data_{timestamp}.csv'), index=False)
        logger.info(f"Generated synthetic data with {n_samples} samples")
    
    def _train_models(self) -> None:
        """Train anomaly detection and classification models."""
        try:
            # Load training data
            X, y, attack_types = self._load_training_data()
            
            if X.shape[0] == 0:
                logger.warning("No training data available")
                return
            
            logger.info(f"Training with {X.shape[0]} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train anomaly detection model
            self.anomaly_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.anomaly_model.fit(X_train_scaled)
            
            # Train classification model
            self.classifier_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.classifier_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            y_pred = self.classifier_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def _load_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load training data from CSV files in the data directory.
        
        Returns:
            Tuple containing:
                - X: Feature matrix
                - y: Binary labels (attack or not)
                - attack_types: List of attack type labels
        """
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        
        if not data_files:
            return np.array([]), np.array([]), []
        
        dfs = []
        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            return np.array([]), np.array([]), []
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Extract features, binary labels and attack types
        feature_cols = [col for col in combined_df.columns if col not in ['is_attack', 'attack_type']]
        X = combined_df[feature_cols].values
        y = combined_df['is_attack'].values
        attack_types = combined_df['attack_type'].tolist()
        
        return X, y, attack_types
    
    def save_traffic_data(self, traffic_data: Dict[str, Any]) -> None:
        """
        Save traffic data for future training.
        
        Args:
            traffic_data: Dictionary of traffic metrics
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([traffic_data])
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f'traffic_data_{timestamp}.csv'
            
            # Save to CSV, append if file exists
            filepath = os.path.join(DATA_DIR, filename)
            df.to_csv(filepath, index=False, mode='a', header=not os.path.exists(filepath))
            
            logger.debug(f"Saved traffic data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving traffic data: {e}")
    
    def analyze_traffic(self, traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze traffic data for anomalies and classify attacks.
        
        Args:
            traffic_data: Dictionary of traffic metrics
            
        Returns:
            Dictionary with analysis results
        """
        if self.anomaly_model is None or self.classifier_model is None or self.scaler is None:
            logger.warning("Models not yet trained, unable to analyze traffic")
            return {
                "is_attack": False,
                "confidence": 0.0,
                "attack_type": "Unknown",
                "analysis": "Models not yet trained"
            }
        
        try:
            # Extract features in the correct order
            feature_cols = [
                'packet_rate', 'byte_rate', 'packet_size_mean', 'packet_size_std',
                'new_conn_rate', 'active_conn', 'syn_rate', 'fin_rate', 
                'rst_rate', 'ack_rate', 'http_req_rate', 'dns_req_rate',
                'unique_src_ips', 'ttl_mean'
            ]
            
            # Ensure all features are present
            for col in feature_cols:
                if col not in traffic_data:
                    traffic_data[col] = 0.0
            
            # Create feature vector
            X = np.array([[traffic_data[col] for col in feature_cols]])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Anomaly detection
            anomaly_score = self.anomaly_model.score_samples(X_scaled)[0]
            is_anomaly = anomaly_score < -self.anomaly_sensitivity
            
            # Attack classification
            class_probs = self.classifier_model.predict_proba(X_scaled)[0]
            prediction = int(np.argmax(class_probs))
            confidence = float(np.max(class_probs))
            
            is_attack = is_anomaly and confidence > self.detection_threshold
            
            # Get attack type
            if is_attack:
                attack_classes = ['Normal'] + LAYER3_ATTACKS + LAYER4_ATTACKS + LAYER7_ATTACKS
                attack_type = attack_classes[prediction] if prediction < len(attack_classes) else "Unknown"
            else:
                attack_type = "Normal"
            
            # Save data for future training
            traffic_data['is_attack'] = 1 if is_attack else 0
            traffic_data['attack_type'] = attack_type
            self.save_traffic_data(traffic_data)
            
            return {
                "is_attack": is_attack,
                "confidence": confidence,
                "attack_type": attack_type,
                "anomaly_score": float(anomaly_score),
                "analysis": {
                    "packet_rate_anomaly": traffic_data['packet_rate'] > 1000,
                    "connection_rate_anomaly": traffic_data['new_conn_rate'] > 100,
                    "syn_rate_anomaly": traffic_data['syn_rate'] > 200,
                    "http_req_anomaly": traffic_data['http_req_rate'] > 200
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing traffic: {e}")
            return {
                "is_attack": False,
                "confidence": 0.0,
                "attack_type": "Unknown",
                "analysis": f"Error during analysis: {str(e)}"
            }
    
    def classify_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a specific threat.
        
        Args:
            threat_data: Dictionary with threat information
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Extract features for classification
            features = self._extract_features_from_threat(threat_data)
            
            if self.classifier_model is None or self.scaler is None:
                logger.warning("Models not yet trained, unable to classify threat")
                return {
                    "is_attack": True,  # Assume it's an attack since it was flagged as a threat
                    "confidence": 0.8,  # Reasonable default
                    "attack_type": "Unknown",
                    "analysis": "Models not yet trained"
                }
            
            # Scale features
            features_scaled = self.scaler.transform(np.array([features]))
            
            # Get classification and confidence
            class_probs = self.classifier_model.predict_proba(features_scaled)[0]
            prediction = int(np.argmax(class_probs))
            confidence = float(np.max(class_probs))
            
            # Get attack type
            attack_classes = ['Normal'] + LAYER3_ATTACKS + LAYER4_ATTACKS + LAYER7_ATTACKS
            attack_type = attack_classes[prediction] if prediction < len(attack_classes) else "Unknown"
            
            # Determine attack layer
            if attack_type in LAYER3_ATTACKS:
                layer = "Layer3"
            elif attack_type in LAYER4_ATTACKS:
                layer = "Layer4"
            elif attack_type in LAYER7_ATTACKS:
                layer = "Layer7"
            else:
                layer = "Unknown"
            
            return {
                "is_attack": True,
                "confidence": confidence,
                "attack_type": attack_type,
                "attack_layer": layer,
                "severity": self._calculate_severity(threat_data, confidence),
                "recommendation": self._generate_recommendation(attack_type, confidence)
            }
            
        except Exception as e:
            logger.error(f"Error classifying threat: {e}")
            return {
                "is_attack": True,
                "confidence": 0.7,
                "attack_type": "Unknown",
                "analysis": f"Error during classification: {str(e)}"
            }
    
    def _extract_features_from_threat(self, threat_data: Dict[str, Any]) -> List[float]:
        """Extract features from threat data for classification."""
        # Basic feature set - can be expanded
        features = [
            threat_data.get('packet_rate', 0),
            threat_data.get('byte_rate', 0),
            threat_data.get('packet_size_mean', 0),
            threat_data.get('packet_size_std', 0),
            threat_data.get('new_conn_rate', 0),
            threat_data.get('active_conn', 0),
            threat_data.get('syn_rate', 0),
            threat_data.get('fin_rate', 0),
            threat_data.get('rst_rate', 0),
            threat_data.get('ack_rate', 0),
            threat_data.get('http_req_rate', 0),
            threat_data.get('dns_req_rate', 0),
            threat_data.get('unique_src_ips', 0),
            threat_data.get('ttl_mean', 0)
        ]
        return features
    
    def _calculate_severity(self, threat_data: Dict[str, Any], confidence: float) -> str:
        """Calculate the severity of a threat."""
        # Simple severity calculation - can be made more sophisticated
        packet_rate = threat_data.get('packet_rate', 0)
        byte_rate = threat_data.get('byte_rate', 0)
        
        # Calculate a severity score from 0 to 1
        severity_score = 0.4 * confidence + 0.3 * min(1.0, packet_rate / 10000) + 0.3 * min(1.0, byte_rate / 1000000)
        
        if severity_score > 0.8:
            return "Critical"
        elif severity_score > 0.6:
            return "High"
        elif severity_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendation(self, attack_type: str, confidence: float) -> str:
        """Generate a mitigation recommendation based on attack type."""
        if attack_type == "SynFlood":
            return "Implement SYN cookies and rate limit new connections"
        elif attack_type == "HttpFlood":
            return "Deploy challenge-response tests and implement HTTP rate limiting"
        elif attack_type == "UdpFlood":
            return "Rate limit UDP traffic and implement UDP filtering"
        elif attack_type == "DnsAmplification":
            return "Block suspicious DNS traffic and implement response rate limiting"
        elif attack_type in LAYER3_ATTACKS:
            return "Apply IP-based filtering and packet validation"
        elif attack_type in LAYER4_ATTACKS:
            return "Implement connection rate limiting and stateful inspection"
        elif attack_type in LAYER7_ATTACKS:
            return "Deploy application-specific protections and bot detection"
        else:
            return "Monitor traffic patterns and implement general rate limiting"

# Command line interface
if __name__ == "__main__":
    print("Kyaro Anti-DDoS ML Engine")
    print("-------------------------")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training models...")
            engine = KyaroMLEngine()
            engine._train_models()
            print("Training complete.")
        elif sys.argv[1] == "generate":
            print("Generating synthetic data...")
            engine = KyaroMLEngine()
            engine._generate_synthetic_data()
            print("Data generation complete.")
        elif sys.argv[1] == "analyze" and len(sys.argv) > 2:
            print(f"Analyzing data from {sys.argv[2]}...")
            engine = KyaroMLEngine()
            try:
                with open(sys.argv[2], 'r') as f:
                    data = json.load(f)
                result = engine.analyze_traffic(data)
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Unknown command")
            print("Usage: ml_engine.py [train|generate|analyze <data_file>]")
    else:
        print("Starting ML Engine in standalone mode...")
        engine = KyaroMLEngine()
        print("ML Engine is running. Use Ctrl+C to stop.")
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("ML Engine stopped.")
