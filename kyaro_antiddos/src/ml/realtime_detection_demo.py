#!/usr/bin/env python3
# Real-time DDoS detection demo script

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime_demo")

def generate_traffic_sample(is_attack=False, attack_type=None):
    """Generate a single traffic sample."""
    
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
        'SynFlood': {
            'packet_rate': (5000, 2000),
            'byte_rate': (300000, 100000),
            'packet_size_mean': (60, 10),
            'packet_size_std': (5, 2),
            'new_conn_rate': (2000, 500),
            'active_conn': (5000, 1000),
            'syn_rate': (4500, 1000),
            'fin_rate': (10, 5),
            'rst_rate': (5, 2),
            'ack_rate': (100, 50),
            'http_req_rate': (0, 0),
            'dns_req_rate': (0, 0),
            'unique_src_ips': (10, 5),
            'ttl_mean': (40, 10)
        },
        'UdpFlood': {
            'packet_rate': (8000, 3000),
            'byte_rate': (1000000, 300000),
            'packet_size_mean': (500, 200),
            'packet_size_std': (300, 100),
            'new_conn_rate': (0, 0),
            'active_conn': (0, 0),
            'syn_rate': (0, 0),
            'fin_rate': (0, 0),
            'rst_rate': (0, 0),
            'ack_rate': (0, 0),
            'http_req_rate': (0, 0),
            'dns_req_rate': (0, 0),
            'unique_src_ips': (5, 2),
            'ttl_mean': (35, 5)
        },
        'HttpFlood': {
            'packet_rate': (3000, 1000),
            'byte_rate': (800000, 200000),
            'packet_size_mean': (1200, 300),
            'packet_size_std': (400, 100),
            'new_conn_rate': (100, 30),
            'active_conn': (500, 100),
            'syn_rate': (100, 30),
            'fin_rate': (100, 30),
            'rst_rate': (10, 5),
            'ack_rate': (2500, 800),
            'http_req_rate': (2000, 500),
            'dns_req_rate': (0, 0),
            'unique_src_ips': (20, 10),
            'ttl_mean': (50, 10)
        }
    }
    
    sample = {}
    timestamp = datetime.now()
    sample['timestamp'] = timestamp.isoformat()
    
    if is_attack:
        if attack_type is None:
            attack_type = np.random.choice(['SynFlood', 'UdpFlood', 'HttpFlood'])
        
        params = attack_params[attack_type]
        sample['is_attack'] = 1
        sample['attack_type'] = attack_type
    else:
        params = normal_params
        sample['is_attack'] = 0
        sample['attack_type'] = 'Normal'
    
    for feature, (mean, std) in params.items():
        sample[feature] = float(np.random.normal(mean, std))
    
    return sample

def simulate_traffic(duration_seconds=60, attack_start=None, attack_end=None, attack_type=None, interval=1.0):
    """
    Simulate network traffic for a specified duration.
    
    Args:
        duration_seconds: Total simulation time in seconds
        attack_start: When to start the attack (seconds from start)
        attack_end: When to end the attack (seconds from start)
        attack_type: Type of attack to simulate
        interval: Time between samples in seconds
    
    Returns:
        List of traffic samples
    """
    traffic_data = []
    start_time = time.time()
    current_time = start_time
    end_time = start_time + duration_seconds
    
    if attack_start is None:
        attack_start = duration_seconds * 0.4  # Start attack at 40% of the way through
    
    if attack_end is None:
        attack_end = duration_seconds * 0.7  # End attack at 70% of the way through
    
    attack_start_time = start_time + attack_start
    attack_end_time = start_time + attack_end
    
    print(f"Simulation started. Duration: {duration_seconds}s")
    print(f"Attack window: {attack_start}s - {attack_end}s")
    
    try:
        while current_time < end_time:
            is_attack = attack_start_time <= current_time <= attack_end_time
            
            sample = generate_traffic_sample(is_attack, attack_type if is_attack else None)
            traffic_data.append(sample)
            
            # Print status
            elapsed = current_time - start_time
            status = "ATTACK" if is_attack else "NORMAL"
            print(f"\rTime: {elapsed:.1f}s / {duration_seconds}s | Status: {status} | Samples: {len(traffic_data)}", end="")
            
            # Sleep until next interval
            time.sleep(interval)
            current_time = time.time()
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    print("\nSimulation completed")
    return traffic_data

def main():
    parser = argparse.ArgumentParser(description="Real-time DDoS detection demo")
    parser.add_argument("--duration", type=int, default=60, help="Simulation duration in seconds")
    parser.add_argument("--attack-start", type=int, help="When to start attack (seconds from start)")
    parser.add_argument("--attack-end", type=int, help="When to end attack (seconds from start)")
    parser.add_argument("--attack-type", choices=["SynFlood", "UdpFlood", "HttpFlood"], help="Type of attack to simulate")
    parser.add_argument("--interval", type=float, default=1.0, help="Time between samples in seconds")
    parser.add_argument("--window-size", type=int, default=10, help="Detection window size")
    parser.add_argument("--output", help="Save results to file")
    
    args = parser.parse_args()
    
    try:
        # Import the advanced model
        from advanced_dl_model import KyaroAdvancedDL
        
        # Create the advanced DL model
        print("Initializing advanced DL model...")
        adv_dl = KyaroAdvancedDL(sequence_length=args.window_size, use_explainer=True)
        
        # Check if models are loaded
        model_info = adv_dl.get_model_info()
        print("\nModel Information:")
        print(json.dumps(model_info, indent=2))
        
        if not model_info['models_loaded']['cnn']:
            print("\nCNN model not found. Please train the model first.")
            return 1
        
        # Simulate traffic
        print("\nStarting traffic simulation...")
        traffic_data = simulate_traffic(
            duration_seconds=args.duration,
            attack_start=args.attack_start,
            attack_end=args.attack_end,
            attack_type=args.attack_type,
            interval=args.interval
        )
        
        # Prepare for real-time detection
        detection_results = []
        window = []
        
        print("\nAnalyzing traffic patterns...")
        for i, sample in enumerate(traffic_data):
            # Add sample to window
            window.append(sample)
            
            # Keep window at the right size
            if len(window) > args.window_size:
                window.pop(0)
            
            # Only start detection when we have enough samples
            if len(window) == args.window_size:
                # Prepare sequence data
                sequence = adv_dl.prepare_sequence_from_traffic(window)
                
                # Make prediction
                start_time = time.time()
                result = adv_dl.predict(sequence)
                prediction_time = time.time() - start_time
                
                # Add result
                result['sample_index'] = i
                result['actual_label'] = sample['is_attack']
                result['actual_type'] = sample['attack_type']
                result['detection_time_ms'] = prediction_time * 1000
                detection_results.append(result)
                
                # Print result
                status = "ATTACK DETECTED" if result['is_attack'] else "NORMAL TRAFFIC"
                confidence = result['confidence'] * 100
                print(f"\rSample {i+1}/{len(traffic_data)} | {status} | Confidence: {confidence:.2f}% | Detection time: {prediction_time*1000:.2f}ms", end="")
        
        print("\n\nDetection completed")
        
        # Calculate accuracy
        correct = sum(1 for r in detection_results if r['is_attack'] == r['actual_label'])
        accuracy = correct / len(detection_results) if detection_results else 0
        
        # Calculate average detection time
        avg_detection_time = sum(r['detection_time_ms'] for r in detection_results) / len(detection_results) if detection_results else 0
        
        print(f"\nResults:")
        print(f"  Total samples analyzed: {len(detection_results)}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Average detection time: {avg_detection_time:.2f}ms")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        sample_indices = [r['sample_index'] for r in detection_results]
        actual_labels = [r['actual_label'] for r in detection_results]
        predicted_labels = [r['is_attack'] for r in detection_results]
        confidences = [r['confidence'] for r in detection_results]
        
        # Plot actual vs predicted
        plt.subplot(2, 1, 1)
        plt.plot(sample_indices, actual_labels, 'b-', label='Actual (1=Attack, 0=Normal)')
        plt.plot(sample_indices, predicted_labels, 'r--', label='Predicted')
        plt.title('DDoS Detection Results')
        plt.ylabel('Attack Status')
        plt.legend()
        
        # Plot confidence
        plt.subplot(2, 1, 2)
        plt.plot(sample_indices, confidences, 'g-')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.title('Detection Confidence')
        plt.xlabel('Sample Index')
        plt.ylabel('Confidence')
        
        plt.tight_layout()
        
        # Save or show plot
        if args.output:
            plt.savefig(args.output)
            print(f"Results saved to {args.output}")
        else:
            plt.savefig('detection_results.png')
            print("Results saved to detection_results.png")
        
        return 0
        
    except Exception as e:
        logger.error("Error during demo: %s", e, exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())