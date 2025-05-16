#!/usr/bin/env python3
# Kyaro Anti-DDoS Advanced Deep Learning Module
# This module implements advanced deep learning models for real-time DDoS detection

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Input, BatchNormalization, Conv1D, 
    MaxPooling1D, Flatten, Concatenate, GlobalAveragePooling1D,
    Bidirectional, GRU, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import logging
import shap
import time
from typing import Tuple, List, Dict, Any, Optional, Union
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger("kyaro_advanced_dl")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'advanced_dl.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
EXPLAINER_DIR = os.path.join(os.path.dirname(__file__), 'explainers')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(EXPLAINER_DIR, exist_ok=True)

# Model file paths
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.h5')
BIGRU_MODEL_PATH = os.path.join(MODEL_DIR, 'bigru_model.h5')
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.h5')
ADV_SCALER_PATH = os.path.join(MODEL_DIR, 'advanced_scaler.joblib')
SHAP_EXPLAINER_PATH = os.path.join(EXPLAINER_DIR, 'shap_explainer.joblib')

# Feature columns used for deep learning
FEATURE_COLUMNS = [
    'packet_rate', 'byte_rate', 'packet_size_mean', 'packet_size_std',
    'new_conn_rate', 'active_conn', 'syn_rate', 'fin_rate', 
    'rst_rate', 'ack_rate', 'http_req_rate', 'dns_req_rate',
    'unique_src_ips', 'ttl_mean',
    # Additional features that could be added if available
    # 'tcp_window_size_mean', 'tcp_window_size_std', 'packet_inter_arrival_time_mean',
    # 'packet_inter_arrival_time_std', 'flow_duration_mean', 'flow_duration_std',
    # 'packet_size_entropy', 'src_port_entropy', 'dst_port_entropy'
]

class KyaroAdvancedDL:
    """Advanced deep learning model implementation for real-time DDoS detection in Kyaro Anti-DDoS."""
    
    def __init__(self, 
                 sequence_length: int = 10, 
                 validation_split: float = 0.2,
                 batch_size: int = 32, 
                 epochs: int = 100, 
                 patience: int = 15,
                 learning_rate: float = 0.001,
                 use_explainer: bool = True):
        """
        Initialize the advanced deep learning module.
        
        Args:
            sequence_length: Number of time steps for sequence models
            validation_split: Fraction of data to use for validation
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            patience: Patience for early stopping
            learning_rate: Initial learning rate for optimizer
            use_explainer: Whether to use SHAP explainer for model interpretability
        """
        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.use_explainer = use_explainer
        
        self.scaler = None
        self.cnn_model = None
        self.bigru_model = None
        self.ensemble_model = None
        self.shap_explainer = None
        
        # Try to load existing models
        self._load_models()
        
        logger.info("Advanced deep learning module initialized")
    
    def _load_models(self) -> None:
        """Load existing models if available."""
        try:
            if os.path.exists(CNN_MODEL_PATH):
                self.cnn_model = load_model(CNN_MODEL_PATH)
                logger.info("Loaded CNN model from %s", CNN_MODEL_PATH)
            
            if os.path.exists(BIGRU_MODEL_PATH):
                self.bigru_model = load_model(BIGRU_MODEL_PATH)
                logger.info("Loaded BiGRU model from %s", BIGRU_MODEL_PATH)
            
            if os.path.exists(ENSEMBLE_MODEL_PATH):
                self.ensemble_model = load_model(ENSEMBLE_MODEL_PATH)
                logger.info("Loaded Ensemble model from %s", ENSEMBLE_MODEL_PATH)
            
            if os.path.exists(ADV_SCALER_PATH):
                self.scaler = joblib.load(ADV_SCALER_PATH)
                logger.info("Loaded scaler from %s", ADV_SCALER_PATH)
                
            if self.use_explainer and os.path.exists(SHAP_EXPLAINER_PATH):
                self.shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
                logger.info("Loaded SHAP explainer from %s", SHAP_EXPLAINER_PATH)
        
        except Exception as e:
            logger.error("Error loading models: %s", e)
            self.cnn_model = None
            self.bigru_model = None
            self.ensemble_model = None
            self.scaler = None
            self.shap_explainer = None
    
    def _save_models(self) -> None:
        """Save trained models and scaler."""
        try:
            if self.cnn_model:
                self.cnn_model.save(CNN_MODEL_PATH)
                logger.info("Saved CNN model to %s", CNN_MODEL_PATH)
            
            if self.bigru_model:
                self.bigru_model.save(BIGRU_MODEL_PATH)
                logger.info("Saved BiGRU model to %s", BIGRU_MODEL_PATH)
            
            if self.ensemble_model:
                self.ensemble_model.save(ENSEMBLE_MODEL_PATH)
                logger.info("Saved Ensemble model to %s", ENSEMBLE_MODEL_PATH)
            
            if self.scaler:
                joblib.dump(self.scaler, ADV_SCALER_PATH)
                logger.info("Saved scaler to %s", ADV_SCALER_PATH)
                
            if self.use_explainer and self.shap_explainer:
                joblib.dump(self.shap_explainer, SHAP_EXPLAINER_PATH)
                logger.info("Saved SHAP explainer to %s", SHAP_EXPLAINER_PATH)
        
        except Exception as e:
            logger.error("Error saving models: %s", e)
    
    def _build_cnn_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build a CNN model for time series classification.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        
        # Initial Conv1D layer
        x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                  kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        
        # Second Conv1D block
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Dropout(0.2)(x)
        
        # Third Conv1D block
        x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu',
                  kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Dropout(0.2)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Final dense layers
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        logger.info("Built CNN model: %s", model.summary())
        return model
    
    def _build_bigru_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Build a Bidirectional GRU model for time series classification.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=input_shape)
        
        # Bidirectional GRU layers
        x = Bidirectional(GRU(64, return_sequences=True, 
                             kernel_regularizer=l2(0.001), 
                             recurrent_dropout=0.1))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(GRU(128, return_sequences=True, 
                             kernel_regularizer=l2(0.001), 
                             recurrent_dropout=0.1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(GRU(64, kernel_regularizer=l2(0.001), 
                             recurrent_dropout=0.1))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Final dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        logger.info("Built BiGRU model: %s", model.summary())
        return model
    
    def _build_ensemble_model(self, cnn_model: Model, bigru_model: Model) -> Model:
        """
        Build an ensemble model that combines CNN and BiGRU models.
        
        Args:
            cnn_model: Trained CNN model
            bigru_model: Trained BiGRU model
            
        Returns:
            Compiled ensemble model
        """
        # Get the input shape from one of the models
        input_shape = cnn_model.input_shape[1:]
        
        # Create a new input layer
        inputs = Input(shape=input_shape)
        
        # Get predictions from both models
        cnn_outputs = cnn_model(inputs)
        bigru_outputs = bigru_model(inputs)
        
        # Concatenate the outputs
        concatenated = Concatenate()([cnn_outputs, bigru_outputs])
        
        # Add a dense layer to learn the best combination
        x = Dense(16, activation='relu')(concatenated)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create and compile the ensemble model
        ensemble_model = Model(inputs=inputs, outputs=outputs)
        ensemble_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),  # Lower learning rate for fine-tuning
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        logger.info("Built ensemble model: %s", ensemble_model.summary())
        return ensemble_model
    
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
        Train transformer and BiGRU models, then create an ensemble.
        
        Returns:
            Dictionary with training metrics
        """
        try:
            start_time = time.time()
            logger.info("Starting advanced deep learning model training")
            
            # Load and prepare data
            X_train, X_test, y_train, y_test = self._load_and_prepare_data()
            
            # Prepare sequence data for models
            X_train_seq, y_train_seq = self._prepare_sequence_data(X_train, y_train)
            X_test_seq, y_test_seq = self._prepare_sequence_data(X_test, y_test)
            
            logger.info("Data prepared with shapes: X_train_seq %s, y_train_seq %s", 
                       X_train_seq.shape, y_train_seq.shape)
            
            # Callbacks for training
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.patience, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
            )
            
            # Train CNN model
            logger.info("Training CNN model")
            self.cnn_model = self._build_cnn_model((self.sequence_length, X_train.shape[1]))
            
            cnn_checkpoint = ModelCheckpoint(
                CNN_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'
            )
            
            cnn_history = self.cnn_model.fit(
                X_train_seq, y_train_seq,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=[early_stopping, reduce_lr, cnn_checkpoint],
                verbose=2
            )
            
            # Train BiGRU model
            logger.info("Training BiGRU model")
            self.bigru_model = self._build_bigru_model((self.sequence_length, X_train.shape[1]))
            
            bigru_checkpoint = ModelCheckpoint(
                BIGRU_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'
            )
            
            bigru_history = self.bigru_model.fit(
                X_train_seq, y_train_seq,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_split,
                callbacks=[early_stopping, reduce_lr, bigru_checkpoint],
                verbose=2
            )
            
            # Train ensemble model
            logger.info("Training ensemble model")
            self.ensemble_model = self._build_ensemble_model(self.cnn_model, self.bigru_model)
            
            ensemble_checkpoint = ModelCheckpoint(
                ENSEMBLE_MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'
            )
            
            # Freeze the base models during ensemble training
            self.cnn_model.trainable = False
            self.bigru_model.trainable = False
            
            ensemble_history = self.ensemble_model.fit(
                X_train_seq, y_train_seq,
                batch_size=self.batch_size,
                epochs=min(30, self.epochs // 2),  # Fewer epochs for ensemble
                validation_split=self.validation_split,
                callbacks=[early_stopping, ensemble_checkpoint],
                verbose=2
            )
            
            # Evaluate models
            cnn_eval = self.cnn_model.evaluate(X_test_seq, y_test_seq)
            bigru_eval = self.bigru_model.evaluate(X_test_seq, y_test_seq)
            ensemble_eval = self.ensemble_model.evaluate(X_test_seq, y_test_seq)
            
            # Create SHAP explainer for the ensemble model
            if self.use_explainer:
                logger.info("Creating SHAP explainer")
                # Use a small subset of training data for the explainer
                explainer_data = X_train_seq[:100]
                
                def ensemble_predict(x):
                    return self.ensemble_model.predict(x)
                
                self.shap_explainer = shap.KernelExplainer(ensemble_predict, explainer_data)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            metrics = {
                "cnn": {
                    "loss": float(cnn_eval[0]),
                    "accuracy": float(cnn_eval[1]),
                    "precision": float(cnn_eval[2]),
                    "recall": float(cnn_eval[3]),
                    "auc": float(cnn_eval[4]),
                    "epochs_trained": len(cnn_history.history['loss'])
                },
                "bigru": {
                    "loss": float(bigru_eval[0]),
                    "accuracy": float(bigru_eval[1]),
                    "precision": float(bigru_eval[2]),
                    "recall": float(bigru_eval[3]),
                    "auc": float(bigru_eval[4]),
                    "epochs_trained": len(bigru_history.history['loss'])
                },
                "ensemble": {
                    "loss": float(ensemble_eval[0]),
                    "accuracy": float(ensemble_eval[1]),
                    "precision": float(ensemble_eval[2]),
                    "recall": float(ensemble_eval[3]),
                    "auc": float(ensemble_eval[4]),
                    "epochs_trained": len(ensemble_history.history['loss'])
                },
                "training_time_seconds": training_time
            }
            
            # Plot training history
            self._plot_training_history(cnn_history, 'CNN')
            self._plot_training_history(bigru_history, 'BiGRU')
            self._plot_training_history(ensemble_history, 'Ensemble')
            
            # Plot ROC curves
            self._plot_roc_curves(X_test_seq, y_test_seq)
            
            # Save models
            self._save_models()
            
            logger.info("Advanced deep learning models trained successfully: %s", metrics)
            return metrics
            
        except Exception as e:
            logger.error("Error training advanced deep learning models: %s", e)
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
    
    def _plot_roc_curves(self, X_test_seq: np.ndarray, y_test_seq: np.ndarray) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            X_test_seq: Test sequence data
            y_test_seq: Test labels
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.figure(figsize=(10, 8))
        
        # Get predictions
        cnn_preds = self.cnn_model.predict(X_test_seq).ravel()
        bigru_preds = self.bigru_model.predict(X_test_seq).ravel()
        ensemble_preds = self.ensemble_model.predict(X_test_seq).ravel()
        
        # Calculate ROC curves
        cnn_fpr, cnn_tpr, _ = roc_curve(y_test_seq, cnn_preds)
        bigru_fpr, bigru_tpr, _ = roc_curve(y_test_seq, bigru_preds)
        ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test_seq, ensemble_preds)
        
        # Calculate AUC
        cnn_auc = auc(cnn_fpr, cnn_tpr)
        bigru_auc = auc(bigru_fpr, bigru_tpr)
        ensemble_auc = auc(ensemble_fpr, ensemble_tpr)
        
        # Plot ROC curves
        plt.plot(cnn_fpr, cnn_tpr, label=f'CNN (AUC = {cnn_auc:.4f})')
        plt.plot(bigru_fpr, bigru_tpr, label=f'BiGRU (AUC = {bigru_auc:.4f})')
        plt.plot(ensemble_fpr, ensemble_tpr, label=f'Ensemble (AUC = {ensemble_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, f'roc_curves_{timestamp}.png')
        plt.savefig(plot_path)
        logger.info("Saved ROC curves plot to %s", plot_path)
    
    def prepare_sequence_from_traffic(self, traffic_data: List[Dict[str, float]]) -> np.ndarray:
        """
        Prepare a sequence from recent traffic data for prediction.
        
        Args:
            traffic_data: List of traffic data dictionaries
            
        Returns:
            Prepared sequence data for model input
        """
        if len(traffic_data) < self.sequence_length:
            # Pad with zeros if not enough data points
            padding_needed = self.sequence_length - len(traffic_data)
            padding = [dict.fromkeys(FEATURE_COLUMNS, 0.0) for _ in range(padding_needed)]
            traffic_data = padding + traffic_data
        
        # Take the most recent sequence_length data points
        recent_data = traffic_data[-self.sequence_length:]
        
        # Extract features in the correct order
        feature_matrix = np.array([[data.get(col, 0.0) for col in FEATURE_COLUMNS] for data in recent_data])
        
        # Scale the features
        if self.scaler:
            feature_matrix = self.scaler.transform(feature_matrix)
        
        # Reshape for model input (add batch dimension)
        return np.expand_dims(feature_matrix, axis=0)
    
    def predict(self, traffic_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using the available models.
        
        Args:
            traffic_sequence: Prepared sequence data
            
        Returns:
            Dictionary with prediction results and explanations
        """
        try:
            start_time = time.time()
            
            # Check which models are available
            if self.cnn_model is None:
                logger.warning("No models loaded, cannot make predictions")
                return {
                    "is_attack": False,
                    "confidence": 0.0,
                    "model_status": "not_loaded"
                }
            
            # Get predictions from CNN model
            cnn_pred = float(self.cnn_model.predict(traffic_sequence)[0][0])
            
            # Initialize other predictions
            bigru_pred = 0.0
            ensemble_pred = 0.0
            
            # If BiGRU model is available, get its prediction
            if self.bigru_model is not None:
                bigru_pred = float(self.bigru_model.predict(traffic_sequence)[0][0])
            
            # If ensemble model is available, use it for final prediction
            if self.ensemble_model is not None:
                ensemble_pred = float(self.ensemble_model.predict(traffic_sequence)[0][0])
                is_attack = ensemble_pred > 0.5
                confidence = ensemble_pred
            elif self.bigru_model is not None:
                # If only CNN and BiGRU are available, use weighted average
                ensemble_pred = 0.4 * cnn_pred + 0.6 * bigru_pred
                is_attack = ensemble_pred > 0.5
                confidence = ensemble_pred
            else:
                # If only CNN is available, use it directly
                is_attack = cnn_pred > 0.5
                confidence = cnn_pred
                ensemble_pred = cnn_pred
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            result = {
                "is_attack": bool(is_attack),
                "confidence": float(confidence),
                "cnn_confidence": cnn_pred,
                "bigru_confidence": bigru_pred if self.bigru_model is not None else None,
                "ensemble_confidence": ensemble_pred if self.ensemble_model is not None else None,
                "model_status": "active",
                "prediction_time_ms": prediction_time * 1000
            }
            
            # Add SHAP explanations if available
            if self.use_explainer and self.shap_explainer and is_attack:
                try:
                    # Get SHAP values for this prediction
                    shap_values = self.shap_explainer.shap_values(traffic_sequence)
                    
                    # Calculate feature importance
                    feature_importance = {}
                    for i, feature in enumerate(FEATURE_COLUMNS):
                        # Average SHAP values across the sequence for each feature
                        importance = float(np.mean(np.abs(shap_values[0, :, i])))
                        feature_importance[feature] = importance
                    
                    # Sort features by importance
                    sorted_importance = sorted(
                        feature_importance.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    # Add top 5 important features to result
                    result["top_features"] = [
                        {"feature": feature, "importance": importance}
                        for feature, importance in sorted_importance[:5]
                    ]
                    
                    # Generate explanation text
                    explanation = "Attack detected due to "
                    if sorted_importance:
                        top_feature, _ = sorted_importance[0]
                        explanation += f"abnormal {top_feature}"
                        if len(sorted_importance) > 1:
                            second_feature, _ = sorted_importance[1]
                            explanation += f" and {second_feature}"
                    explanation += "."
                    result["explanation"] = explanation
                    
                except Exception as e:
                    logger.error("Error generating SHAP explanations: %s", e)
            
            logger.info("Prediction: CNN=%.4f, BiGRU=%.4f, Ensemble=%.4f, Attack=%s, Time=%.2fms",
                       cnn_pred, bigru_pred, ensemble_pred, is_attack, prediction_time * 1000)
            
            return result
        
        except Exception as e:
            logger.error("Error making prediction: %s", e)
            return {
                "is_attack": False,
                "confidence": 0.0,
                "error": str(e),
                "model_status": "error"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the models.
        
        Returns:
            Dictionary with model information
        """
        models_loaded = {
            "cnn": self.cnn_model is not None,
            "bigru": self.bigru_model is not None,
            "ensemble": self.ensemble_model is not None,
            "scaler": self.scaler is not None,
            "explainer": self.shap_explainer is not None if self.use_explainer else False
        }
        
        model_info = {
            "models_loaded": models_loaded,
            "sequence_length": self.sequence_length,
            "features": FEATURE_COLUMNS,
            "use_explainer": self.use_explainer
        }
        
        # Add model parameters if available
        if self.cnn_model:
            model_info["cnn_params"] = {
                "trainable_params": self.cnn_model.count_params(),
                "layers": len(self.cnn_model.layers)
            }
        
        if self.bigru_model:
            model_info["bigru_params"] = {
                "trainable_params": self.bigru_model.count_params(),
                "layers": len(self.bigru_model.layers)
            }
        
        if self.ensemble_model:
            model_info["ensemble_params"] = {
                "trainable_params": self.ensemble_model.count_params(),
                "layers": len(self.ensemble_model.layers)
            }
        
        return model_info

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kyaro Advanced Deep Learning Module")
    parser.add_argument("action", choices=["train", "info"], help="Action to perform")
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--no_explainer", action="store_true", help="Disable SHAP explainer")
    
    args = parser.parse_args()
    
    # Create the advanced DL module
    adv_dl = KyaroAdvancedDL(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_explainer=not args.no_explainer
    )
    
    if args.action == "train":
        try:
            metrics = adv_dl.train_models()
            print(json.dumps(metrics, indent=2))
        except Exception as e:
            print(f"Error training models: {e}")
            exit(1)
    
    elif args.action == "info":
        info = adv_dl.get_model_info()
        print(json.dumps(info, indent=2))