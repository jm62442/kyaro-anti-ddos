#!/usr/bin/env python3
# Kyaro Anti-DDoS Deep Learning Module
# This module implements deep learning models for advanced DDoS detection

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger("kyaro_dl")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'deep_learning.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model file paths
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'dl_cnn_model.h5')
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'dl_lstm_model.h5')
DL_SCALER_PATH = os.path.join(MODEL_DIR, 'dl_scaler.joblib')

# Feature columns used for deep learning
FEATURE_COLUMNS = [
    'packet_rate', 'byte_rate', 'packet_size_mean', 'packet_size_std',
    'new_conn_rate', 'active_conn', 'syn_rate', 'fin_rate', 
    'rst_rate', 'ack_rate', 'http_req_rate', 'dns_req_rate',
    'unique_src_ips', 'ttl_mean'
]

class KyaroDeepLearning:
    """Deep learning model implementation for DDoS detection in Kyaro Anti-DDoS."""
    
    def __init__(self, sequence_length: int = 10, validation_split: float = 0.2,
                 batch_size: int = 32, epochs: int = 100, patience: int = 10):
        """
        Initialize the deep learning module.
        
        Args:
            sequence_length: Number of time steps for sequence models
            validation_split: Fraction of data to use for validation
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
        """
        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.scaler = None
        self.cnn_model = None
        self.lstm_model = None
        
        # Try to load existing models
        self._load_models()
        
        logger.info("Deep learning module initialized")
    
    def _load_models(self) -> None:
        """Load existing models if available."""
        try:
            if os.path.exists(CNN_MODEL_PATH):
                self.cnn_model = load_model(CNN_MODEL_PATH)
                logger.info("Loaded CNN model from %s", CNN_MODEL_PATH)
            
            if os.path.exists(LSTM_MODEL_PATH):
                self.lstm_model = load_model(LSTM_MODEL_PATH)
                logger.info("Loaded LSTM model from %s", LSTM_MODEL_PATH)
            
            if os.path.exists(DL_SCALER_PATH):
                self.scaler = joblib.load(DL_SCALER_PATH)
                logger.info("Loaded scaler from %s", DL_SCALER_PATH)
        
        except Exception as e:
            logger.error("Error loading models: %s", e)
            self.cnn_model = None
            self.lstm_model = None
            self.scaler = None
    
    def _save_models(self) -> None:
        """Save trained models and scaler."""
        try:
            if self.cnn_model:
                self.cnn_model.save(CNN_MODEL_PATH)
                logger.info("Saved CNN model to %s", CNN_MODEL_PATH)
            
            if self.lstm_model:
                self.lstm_model.save(LSTM_MODEL_PATH)
                logger.info("Saved LSTM model to %s", LSTM_MODEL_PATH)
            
            if self.scaler:
                joblib.dump(self.scaler, DL_SCALER_PATH)
                logger.info("Saved scaler to %s", DL_SCALER_PATH)
        
        except Exception as e:
            logger.error("Error saving models: %s", e)
    
    def _build_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build a 1D CNN model for time series classification.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, 
                  kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Flatten(),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        logger.info("Built CNN model: %s", model.summary())
        return model
    
    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build an LSTM model for time series classification.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True, 
                 kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, kernel_regularizer=l2(0.001), recurrent_dropout=0.2),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        logger.info("Built LSTM model: %s", model.summary())
        return model
    
    def _prepare_sequence_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for sequence models by creating sliding windows.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            X_seq: Sequence data with shape (n_samples, sequence_length, n_features)
            y_seq: Target values corresponding to the sequences
        """
        n_samples, n_features = X.shape
        X_seq = []
        y_seq = []
        
        for i in range(n_samples - self.sequence_length + 1):
            X_seq.append(X[i:i+self.sequence_length])
            # Use the label of the last timestep in the sequence
            y_seq.append(y[i+self.sequence_length-1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data and prepare for deep learning models.
        
        Returns:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
        """
        # Load data
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        if not data_files:
            raise ValueError("No training data available")
        
        dfs = []
        for file in data_files:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                dfs.append(df)
            except Exception as e:
                logger.error("Error loading %s: %s", file, e)
        
        if not dfs:
            raise ValueError("Failed to load any data files")
        
        # Combine all data
        df = pd.concat(dfs, ignore_index=True)
        
        # Extract features and target
        X = df[FEATURE_COLUMNS].values
        y = df['is_attack'].values
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train CNN and LSTM models on the available data.
        
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info("Starting deep learning model training")
            
            # Load and prepare data
            X_train, X_test, y_train, y_test = self._load_and_prepare_data()
            
            # Prepare sequence data for models
            X_train_seq, y_train_seq = self._prepare_sequence_data(X_train, y_train)
            X_test_seq, y_test_seq = self._prepare_sequence_data(X_test, y_test)
            
            logger.info("Data prepared with shapes: X_train_seq %s, y_train_seq %s", 
                       X_train_seq.shape, y_train_seq.shape)
            
            # CNN model
            logger.info("Training CNN model")
            self.cnn_model = self._build_cnn_model((self.sequence_length, X_train.shape[1]))
            
            cnn_checkpoint = ModelCheckpoint(
                CNN_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'
            )
            
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.patience, restore_best_weights=True
            )
            
            cnn_history = self.cnn_model.fit(
                X_train_seq, y_train_seq,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=[early_stopping, cnn_checkpoint],
                verbose=2
            )
            
            # LSTM model
            logger.info("Training LSTM model")
            self.lstm_model = self._build_lstm_model((self.sequence_length, X_train.shape[1]))
            
            lstm_checkpoint = ModelCheckpoint(
                LSTM_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'
            )
            
            lstm_history = self.lstm_model.fit(
                X_train_seq, y_train_seq,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=[early_stopping, lstm_checkpoint],
                verbose=2
            )
            
            # Evaluate models
            cnn_eval = self.cnn_model.evaluate(X_test_seq, y_test_seq)
            lstm_eval = self.lstm_model.evaluate(X_test_seq, y_test_seq)
            
            metrics = {
                "cnn": {
                    "loss": float(cnn_eval[0]),
                    "accuracy": float(cnn_eval[1]),
                    "precision": float(cnn_eval[2]),
                    "recall": float(cnn_eval[3]),
                    "epochs_trained": len(cnn_history.history['loss'])
                },
                "lstm": {
                    "loss": float(lstm_eval[0]),
                    "accuracy": float(lstm_eval[1]),
                    "precision": float(lstm_eval[2]),
                    "recall": float(lstm_eval[3]),
                    "epochs_trained": len(lstm_history.history['loss'])
                }
            }
            
            # Plot training history
            self._plot_training_history(cnn_history, 'CNN')
            self._plot_training_history(lstm_history, 'LSTM')
            
            # Save models
            self._save_models()
            
            logger.info("Deep learning models trained successfully: %s", metrics)
            return metrics
            
        except Exception as e:
            logger.error("Error training deep learning models: %s", e)
            raise
    
    def _plot_training_history(self, history, model_name: str) -> None:
        """
        Plot and save training history.
        
        Args:
            history: Keras history object
            model_name: Name of the model for plot title
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{model_name} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{model_name} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot precision
        plt.subplot(2, 2, 3)
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title(f'{model_name} Model Precision')
        plt.ylabel('Precision')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot recall
        plt.subplot(2, 2, 4)
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title(f'{model_name} Model Recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_{timestamp}.png')
        plt.savefig(plot_path)
        logger.info("Saved training history plot to %s", plot_path)
    
    def prepare_sequence_from_traffic(self, traffic_data: List[Dict[str, float]]) -> np.ndarray:
        """
        Prepare a sequence from recent traffic data for prediction.
        
        Args:
            traffic_data: List of traffic data dictionaries
            
        Returns:
            Prepared sequence data for model input
        """
        if len(traffic_data) < self.sequence_length:
            # Pad with copies of the first item if we don't have enough data
            padding = [traffic_data[0]] * (self.sequence_length - len(traffic_data))
            traffic_data = padding + traffic_data
        
        # Use the most recent data up to sequence_length
        recent_data = traffic_data[-self.sequence_length:]
        
        # Extract features in correct order
        X = np.array([[item.get(col, 0.0) for col in FEATURE_COLUMNS] for item in recent_data])
        
        # Scale the features
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Reshape for model input (1, sequence_length, n_features)
        X = X.reshape(1, self.sequence_length, len(FEATURE_COLUMNS))
        
        return X
    
    def predict(self, traffic_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using both CNN and LSTM models.
        
        Args:
            traffic_sequence: Prepared sequence data
            
        Returns:
            Dictionary with prediction results
        """
        if self.cnn_model is None or self.lstm_model is None:
            logger.warning("Models not loaded, cannot make predictions")
            return {
                "is_attack": False,
                "confidence": 0.0,
                "model_status": "not_loaded"
            }
        
        try:
            # Get predictions from both models
            cnn_pred = self.cnn_model.predict(traffic_sequence)[0][0]
            lstm_pred = self.lstm_model.predict(traffic_sequence)[0][0]
            
            # Ensemble the predictions (weighted average)
            ensemble_pred = 0.4 * cnn_pred + 0.6 * lstm_pred
            is_attack = ensemble_pred > 0.5
            
            logger.info("Prediction: CNN=%.4f, LSTM=%.4f, Ensemble=%.4f, Attack=%s", 
                       cnn_pred, lstm_pred, ensemble_pred, is_attack)
            
            return {
                "is_attack": bool(is_attack),
                "confidence": float(ensemble_pred),
                "cnn_confidence": float(cnn_pred),
                "lstm_confidence": float(lstm_pred),
                "model_status": "active"
            }
            
        except Exception as e:
            logger.error("Error making prediction: %s", e)
            return {
                "is_attack": False,
                "confidence": 0.0,
                "error": str(e),
                "model_status": "error"
            }

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Kyaro Anti-DDoS Deep Learning Module')
    parser.add_argument('command', choices=['train', 'info'], 
                        help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Training deep learning models...")
        dl = KyaroDeepLearning()
        metrics = dl.train_models()
        print("Training complete. Results:")
        print(json.dumps(metrics, indent=2))
        
    elif args.command == 'info':
        dl = KyaroDeepLearning()
        models_status = {
            "cnn_model": "loaded" if dl.cnn_model else "not_loaded",
            "lstm_model": "loaded" if dl.lstm_model else "not_loaded",
            "scaler": "loaded" if dl.scaler else "not_loaded",
            "sequence_length": dl.sequence_length,
            "model_paths": {
                "cnn": CNN_MODEL_PATH,
                "lstm": LSTM_MODEL_PATH,
                "scaler": DL_SCALER_PATH
            }
        }
        print(json.dumps(models_status, indent=2)) 