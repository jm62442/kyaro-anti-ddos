#!/usr/bin/env python3
# Script to train the advanced deep learning model for DDoS detection

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'training.log'))
    ]
)
logger = logging.getLogger("train_advanced")

def main():
    parser = argparse.ArgumentParser(description="Train advanced DDoS detection models")
    parser.add_argument("--sequence-length", type=int, default=10, help="Sequence length for models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--no-explainer", action="store_true", help="Disable SHAP explainer")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data if no real data exists")
    args = parser.parse_args()
    
    try:
        # Import the advanced model
        from advanced_dl_model import KyaroAdvancedDL
        
        # Check if we need to generate synthetic data
        if args.generate_data:
            logger.info("Checking if synthetic data generation is needed")
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            if not data_files:
                logger.info("No data files found, generating synthetic data")
                from ml_engine import KyaroMLEngine
                
                # Create ML engine instance and generate synthetic data
                ml_engine = KyaroMLEngine()
                ml_engine._generate_synthetic_data()
                logger.info("Synthetic data generated")
        
        # Create the advanced DL model
        logger.info("Creating advanced DL model with sequence_length=%d, batch_size=%d, epochs=%d",
                   args.sequence_length, args.batch_size, args.epochs)
        
        adv_dl = KyaroAdvancedDL(
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_explainer=not args.no_explainer
        )
        
        # Train the models
        logger.info("Starting model training")
        metrics = adv_dl.train_models()
        
        # Save metrics to file
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        metrics_file = os.path.join(os.path.dirname(__file__), f'training_metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training completed successfully. Metrics saved to %s", metrics_file)
        
        # Print summary
        print("\n=== Training Summary ===")
        print(f"CNN Accuracy: {metrics['cnn']['accuracy']:.4f}")
        print(f"BiGRU Accuracy: {metrics['bigru']['accuracy']:.4f}")
        print(f"Ensemble Accuracy: {metrics['ensemble']['accuracy']:.4f}")
        print(f"Training Time: {metrics['training_time_seconds']:.2f} seconds")
        print("========================\n")
        
        return 0
        
    except Exception as e:
        logger.error("Error during training: %s", e, exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())