#!/usr/bin/env python3
# Test script for the advanced DDoS detection model

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_advanced")

def generate_test_traffic(num_samples=20, include_attack=True):
    """Generate synthetic traffic data for testing."""
    
    # Base time for the traffic sequence
    base_time = datetime.now()
    
    # Normal traffic parameters
    normal_params = {
        'packet_rate': (500, 100),  # mean, std
        'byte_rate': (50000, 10000),
        'packet_size_mean': (800, 200),
        'packet_size_std': (100, 30),
        'new_conn_rate': (20, 5),
        'active_conn': (100, 30),
        'syn_rate': (15, 5),
        'fin_rate': (10, 3),
        'rst_rate': (5, 2),
        'ack_rate': (450, 100),
        'http_req_rate': (50, 15),
        'dns_req_rate': (20, 10),
        'unique_src_ips': (50, 15),
        'ttl_mean': (64, 10)
    }
    
    # Attack traffic parameters
    attack_params = {
        'packet_rate': (5000, 2000),
        'byte_rate': (500000, 200000),
        'packet_size_mean': (600, 300),
        'packet_size_std': (400, 100),
        'new_conn_rate': (200, 100),
        'active_conn': (1000, 500),
        'syn_rate': (500, 300),
        'fin_rate': (50, 30),
        'rst_rate': (100, 50),
        'ack_rate': (4000, 2000),
        'http_req_rate': (500, 300),
        'dns_req_rate': (200, 150),
        'unique_src_ips': (10, 5),
        'ttl_mean': (40, 20)
    }
    
    traffic_data = []
    
    # Generate normal traffic for the first part
    normal_samples = num_samples if not include_attack else num_samples // 2
    
    for i in range(normal_samples):
        sample = {}
        timestamp = base_time + timedelta(seconds=i)
        sample['timestamp'] = timestamp.isoformat()
        
        for feature, (mean, std) in normal_params.items():
            sample[feature] = float(np.random.normal(mean, std))
        
        sample['is_attack'] = 0
        sample['attack_type'] = 'Normal'
        traffic_data.append(sample)
    
    # Generate attack traffic for the second part if requested
    if include_attack:
        attack_samples = num_samples - normal_samples
        
        for i in range(attack_samples):
            sample = {}
            timestamp = base_time + timedelta(seconds=normal_samples + i)
            sample['timestamp'] = timestamp.isoformat()
            
            for feature, (mean, std) in attack_params.items():
                sample[feature] = float(np.random.normal(mean, std))
            
            sample['is_attack'] = 1
            sample['attack_type'] = np.random.choice(['SynFlood', 'UdpFlood', 'HttpFlood'])
            traffic_data.append(sample)
    
    return traffic_data

def main():
    try:
        # Import the advanced model
        from advanced_dl_model import KyaroAdvancedDL
        
        # Create the advanced DL model
        print("Initializing advanced DL model...")
        adv_dl = KyaroAdvancedDL(sequence_length=10, use_explainer=True)
        
        # Check if models are loaded
        model_info = adv_dl.get_model_info()
        print("\nModel Information:")
        print(json.dumps(model_info, indent=2))
        
        if not model_info['models_loaded']['ensemble']:
            print("\nModels not found. Training new models with synthetic data...")
            
            # Generate synthetic data for training
            from ml_engine import KyaroMLEngine
            ml_engine = KyaroMLEngine()
            ml_engine._generate_synthetic_data()
            
            # Train the models
            adv_dl.train_models()
            print("Models trained successfully.")
        
        # Generate test traffic
        print("\nGenerating test traffic data...")
        traffic_data = generate_test_traffic(num_samples=20, include_attack=True)
        
        # Prepare sequence data
        print("Preparing sequence data...")
        sequence = adv_dl.prepare_sequence_from_traffic(traffic_data)
        
        # Make prediction
        print("Making prediction...")
        start_time = time.time()
        result = adv_dl.predict(sequence)
        prediction_time = time.time() - start_time
        
        # Print results
        print("\nPrediction Results:")
        print(json.dumps(result, indent=2))
        print(f"\nPrediction Time: {prediction_time*1000:.2f} ms")
        
        # Plot traffic data
        plt.figure(figsize=(12, 8))
        
        # Extract timestamps and features
        timestamps = [pd.to_datetime(sample['timestamp']) for sample in traffic_data]
        packet_rates = [sample['packet_rate'] for sample in traffic_data]
        byte_rates = [sample['byte_rate'] for sample in traffic_data]
        is_attack = [sample['is_attack'] for sample in traffic_data]
        
        # Plot packet rate
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, packet_rates, 'b-', label='Packet Rate')
        
        # Highlight attack periods
        attack_periods = []
        current_period = None
        
        for i, (ts, attack) in enumerate(zip(timestamps, is_attack)):
            if attack and current_period is None:
                current_period = (i, None)
            elif not attack and current_period is not None:
                current_period = (current_period[0], i-1)
                attack_periods.append(current_period)
                current_period = None
        
        if current_period is not None:
            current_period = (current_period[0], len(timestamps)-1)
            attack_periods.append(current_period)
        
        for start, end in attack_periods:
            plt.axvspan(timestamps[start], timestamps[end], alpha=0.3, color='red')
        
        plt.title('Network Traffic Analysis')
        plt.ylabel('Packets per Second')
        plt.legend()
        
        # Plot byte rate
        plt.subplot(2, 1, 2)
        plt.plot(timestamps, byte_rates, 'g-', label='Byte Rate')
        
        for start, end in attack_periods:
            plt.axvspan(timestamps[start], timestamps[end], alpha=0.3, color='red')
        
        plt.ylabel('Bytes per Second')
        plt.xlabel('Time')
        plt.legend()
        
        # Add prediction result as text
        plt.figtext(0.5, 0.01, 
                   f"Model Prediction: {'Attack' if result['is_attack'] else 'Normal'} "
                   f"(Confidence: {result['confidence']:.2f})", 
                   ha='center', fontsize=12, 
                   bbox={'facecolor':'orange', 'alpha':0.5, 'pad':5})
        
        plt.tight_layout()
        
        # Save the plot
        plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plot_path = os.path.join(plots_dir, f'test_prediction_{timestamp}.png')
        plt.savefig(plot_path)
        print(f"\nPlot saved to: {plot_path}")
        
        return 0
        
    except Exception as e:
        logger.error("Error during testing: %s", e, exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())