"""
LSTM Models for Mining Stability Prediction

This module implements Task 8: Design LSTM Architecture with:
- Basic LSTM model structure with TensorFlow/Keras
- Multi-step sequence prediction capabilities  
- Attention mechanisms for important event weighting
- Multi-output architecture for different prediction horizons
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import List, Dict, Optional, Any, Tuple

from .lstm_config import LSTMConfig, AttentionConfig, MultiStepConfig, ModelArchitectureConfig
from .attention_layers import AttentionLayer, MultiHeadAttention, DomainSpecificAttention


class BasicLSTMModel(Model):
    """
    Basic LSTM model for mining stability prediction
    
    Implements core LSTM architecture with:
    - Stacked LSTM layers
    - Dense classification layers
    - Dropout for regularization
    """
    
    def __init__(self, config: LSTMConfig, **kwargs):
        super(BasicLSTMModel, self).__init__(**kwargs)
        self.config = config
        
        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(config.lstm_units):
            return_sequences = i < len(config.lstm_units) - 1  # All but last return sequences
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=config.dropout_rate,
                    recurrent_dropout=config.recurrent_dropout if hasattr(config, 'recurrent_dropout') else 0.0,
                    name=f'lstm_{i+1}'
                )
            )
        
        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(config.dense_units):
            self.dense_layers.append(
                layers.Dense(
                    units=units,
                    activation=getattr(config, 'activation', 'relu'),
                    name=f'dense_{i+1}'
                )
            )
            self.dense_layers.append(
                layers.Dropout(config.dropout_rate, name=f'dropout_dense_{i+1}')
            )
        
        # Output layer
        self.output_layer = layers.Dense(
            units=config.num_classes,
            activation=getattr(config, 'output_activation', 'softmax'),
            name='output'
        )
        
        # Batch normalization (optional)
        self.use_batch_norm = getattr(config, 'use_batch_norm', False)
        if self.use_batch_norm:
            self.batch_norm_layers = [
                layers.BatchNormalization(name=f'batch_norm_{i+1}') 
                for i in range(len(config.dense_units))
            ]
    
    def call(self, inputs, training=None):
        """Forward pass through the model"""
        x = inputs
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Pass through dense layers
        for i, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x, training=training)
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and i % 2 == 0:  # Apply to Dense layers only
                batch_norm_idx = i // 2
                if batch_norm_idx < len(self.batch_norm_layers):
                    x = self.batch_norm_layers[batch_norm_idx](x, training=training)
        
        # Output layer
        output = self.output_layer(x, training=training)
        
        return output
    
    def get_config(self):
        return {
            'config': self.config,
            'name': self.name
        }


class MultiStepLSTMModel(Model):
    """
    Multi-step LSTM model for different prediction horizons
    
    Predicts stability at multiple future time points:
    - 24h, 48h, 72h, 1 week ahead
    - Uses shared LSTM backbone with multiple output heads
    """
    
    def __init__(self, lstm_config: LSTMConfig, multi_step_config: MultiStepConfig, **kwargs):
        super(MultiStepLSTMModel, self).__init__(**kwargs)
        self.lstm_config = lstm_config
        self.multi_step_config = multi_step_config
        
        # Shared LSTM backbone
        self.lstm_layers = []
        for i, units in enumerate(lstm_config.lstm_units):
            return_sequences = i < len(lstm_config.lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=lstm_config.dropout_rate,
                    name=f'shared_lstm_{i+1}'
                )
            )
        
        # Shared dense layers
        self.shared_dense = []
        for i, units in enumerate(lstm_config.dense_units[:-1]):  # All but last
            self.shared_dense.append(
                layers.Dense(units, activation='relu', name=f'shared_dense_{i+1}')
            )
            self.shared_dense.append(
                layers.Dropout(lstm_config.dropout_rate, name=f'shared_dropout_{i+1}')
            )
        
        # Multi-step output heads
        self.output_heads = {}
        for horizon_name in multi_step_config.horizon_names:
            self.output_heads[horizon_name] = [
                layers.Dense(lstm_config.dense_units[-1], activation='relu', name=f'{horizon_name}_dense'),
                layers.Dropout(lstm_config.dropout_rate, name=f'{horizon_name}_dropout'),
                layers.Dense(lstm_config.num_classes, activation='softmax', name=f'{horizon_name}_output')
            ]
    
    def call(self, inputs, training=None):
        """Forward pass with multi-step outputs"""
        x = inputs
        
        # Shared LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Shared dense layers
        for dense_layer in self.shared_dense:
            x = dense_layer(x, training=training)
        
        # Multi-step outputs
        outputs = {}
        for horizon_name, head_layers in self.output_heads.items():
            head_x = x
            for layer in head_layers:
                head_x = layer(head_x, training=training)
            outputs[horizon_name] = head_x
        
        return outputs
    
    def predict_recursive(self, inputs, steps: int = 5):
        """
        Recursive prediction for extended horizons
        
        Args:
            inputs: Initial input sequence
            steps: Number of recursive steps
            
        Returns:
            predictions: Extended predictions
        """
        current_input = inputs
        predictions = []
        
        for step in range(steps):
            # Get prediction for next step
            step_pred = self(current_input, training=False)
            
            # Use the first horizon prediction as next input
            first_horizon = list(step_pred.keys())[0]
            next_pred = step_pred[first_horizon]
            predictions.append(next_pred)
            
            # For recursive prediction, we need to convert predictions back to feature space
            # This is a simplified approach - in practice, you'd need domain-specific
            # feature reconstruction from predictions
            
            # Option 1: Use a simple sliding window approach (shift existing features)
            # This maintains the original feature dimensions
            batch_size = tf.shape(current_input)[0]
            seq_len = tf.shape(current_input)[1]
            feature_dim = tf.shape(current_input)[2]
            
            # Create a pseudo-feature vector by repeating the last timestep
            # In a real implementation, this would involve feature reconstruction
            last_timestep = current_input[:, -1:, :]  # Shape: [batch, 1, features]
            
            # Shift sequence and append the last timestep (maintaining feature consistency)
            current_input = tf.concat([current_input[:, 1:, :], last_timestep], axis=1)
        
        return tf.stack(predictions, axis=1)


class AttentionLSTMModel(Model):
    """
    LSTM model with attention mechanisms
    
    Implements attention-enhanced LSTM for:
    - Important event weighting
    - Temporal pattern focus
    - Domain-specific attention for mining events
    """
    
    def __init__(self, lstm_config: LSTMConfig, attention_config: AttentionConfig, **kwargs):
        super(AttentionLSTMModel, self).__init__(**kwargs)
        self.lstm_config = lstm_config
        self.attention_config = attention_config
        
        # LSTM layers (return sequences for attention)
        self.lstm_layers = []
        for i, units in enumerate(lstm_config.lstm_units):
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=True,  # Always return sequences for attention
                    dropout=lstm_config.dropout_rate,
                    name=f'attention_lstm_{i+1}'
                )
            )
        
        # Attention mechanisms
        if attention_config.attention_type == 'basic':
            self.attention = AttentionLayer(units=attention_config.key_dim)
        elif attention_config.attention_type == 'multi_head':
            self.attention = MultiHeadAttention(
                num_heads=attention_config.num_heads,
                key_dim=attention_config.key_dim,
                value_dim=getattr(attention_config, 'value_dim', None),
                dropout_rate=attention_config.dropout_rate
            )
        
        # Domain-specific attention for mining events
        if getattr(attention_config, 'use_domain_attention', False):
            self.domain_attention = DomainSpecificAttention(
                event_types=getattr(attention_config, 'event_types', []),
                embedding_dim=getattr(attention_config, 'event_embedding_dim', 16),
                attention_units=attention_config.key_dim
            )
        else:
            self.domain_attention = None
        
        # Dense layers
        self.dense_layers = []
        for i, units in enumerate(lstm_config.dense_units):
            self.dense_layers.append(
                layers.Dense(units, activation='relu', name=f'attention_dense_{i+1}')
            )
            self.dense_layers.append(
                layers.Dropout(lstm_config.dropout_rate, name=f'attention_dropout_{i+1}')
            )
        
        # Output layer
        self.output_layer = layers.Dense(
            lstm_config.num_classes, 
            activation='softmax', 
            name='attention_output'
        )
    
    def call(self, inputs, event_types=None, training=None):
        """Forward pass with attention mechanisms"""
        x = inputs
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Apply attention mechanism
        if self.attention_config.attention_type == 'basic':
            attended_features, attention_weights = self.attention(x)
        elif self.attention_config.attention_type == 'multi_head':
            attended_features = self.attention(query=x, value=x, key=x, training=training)
            # For multi-head attention, we need to pool the sequence
            attended_features = tf.reduce_mean(attended_features, axis=1)
        
        # Apply domain-specific attention if available
        if self.domain_attention is not None and event_types is not None:
            domain_features, domain_weights = self.domain_attention(x, event_types)
            # Combine attention outputs
            attended_features = (attended_features + domain_features) / 2.0
        
        # Pass through dense layers
        x = attended_features
        for dense_layer in self.dense_layers:
            x = dense_layer(x, training=training)
        
        # Output layer
        output = self.output_layer(x, training=training)
        
        return output


class MultiOutputLSTMModel(Model):
    """
    Multi-output LSTM model for comprehensive risk assessment
    
    Implements multiple output heads for:
    - Different prediction horizons
    - Different risk assessment types
    - Comprehensive stability evaluation
    """
    
    def __init__(self, lstm_config: LSTMConfig, multi_step_config: MultiStepConfig, **kwargs):
        super(MultiOutputLSTMModel, self).__init__(**kwargs)
        self.lstm_config = lstm_config
        self.multi_step_config = multi_step_config
        
        # Shared LSTM backbone
        self.lstm_layers = []
        for i, units in enumerate(lstm_config.lstm_units):
            return_sequences = i < len(lstm_config.lstm_units) - 1
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=lstm_config.dropout_rate,
                    name=f'multi_lstm_{i+1}'
                )
            )
        
        # Shared feature extraction
        self.shared_features = []
        if hasattr(lstm_config, 'dense_units') and lstm_config.dense_units:
            for i, units in enumerate(lstm_config.dense_units):
                self.shared_features.append(
                    layers.Dense(units, activation='relu', name=f'shared_features_{i+1}')
                )
                self.shared_features.append(
                    layers.Dropout(lstm_config.dropout_rate, name=f'shared_dropout_{i+1}')
                )
        
        # Multiple output heads
        self.output_heads = {}
        for horizon_name in multi_step_config.horizon_names:
            self.output_heads[horizon_name] = [
                layers.Dense(64, activation='relu', name=f'{horizon_name}_head_dense'),
                layers.Dropout(lstm_config.dropout_rate, name=f'{horizon_name}_head_dropout'),
                layers.Dense(lstm_config.num_classes, activation='softmax', name=f'{horizon_name}_head_output')
            ]
    
    def call(self, inputs, training=None):
        """Forward pass with multiple outputs"""
        x = inputs
        
        # Shared LSTM backbone
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Shared feature extraction
        for feature_layer in self.shared_features:
            x = feature_layer(x, training=training)
        
        # Multiple output heads
        outputs = {}
        for horizon_name, head_layers in self.output_heads.items():
            head_x = x
            for layer in head_layers:
                head_x = layer(head_x, training=training)
            outputs[horizon_name] = head_x
        
        return outputs
    
    def compile_multi_output(self, optimizer='adam', **kwargs):
        """Compile model with multi-output loss configuration"""
        # Create loss dictionary for each output
        losses = {}
        loss_weights = {}
        metrics = {}
        
        for horizon_name in self.multi_step_config.horizon_names:
            losses[f'{horizon_name}_head_output'] = 'sparse_categorical_crossentropy'
            loss_weights[f'{horizon_name}_head_output'] = 1.0
            metrics[f'{horizon_name}_head_output'] = ['accuracy']
        
        self.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
            **kwargs
        )


class CompleteLSTMPredictor(Model):
    """
    Complete LSTM predictor combining all features
    
    Integrates:
    - Multi-step prediction
    - Attention mechanisms
    - Multi-output architecture
    - Domain-specific features
    """
    
    def __init__(self, 
                 lstm_config: LSTMConfig, 
                 attention_config: AttentionConfig,
                 multi_step_config: MultiStepConfig,
                 **kwargs):
        super(CompleteLSTMPredictor, self).__init__(**kwargs)
        self.lstm_config = lstm_config
        self.attention_config = attention_config
        self.multi_step_config = multi_step_config
        
        # LSTM backbone with attention
        self.lstm_layers = []
        for i, units in enumerate(lstm_config.lstm_units):
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=True,  # Always return sequences for attention
                    dropout=lstm_config.dropout_rate,
                    name=f'complete_lstm_{i+1}'
                )
            )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=attention_config.num_heads,
            key_dim=attention_config.key_dim,
            dropout_rate=attention_config.dropout_rate
        )
        
        # Domain-specific attention
        if getattr(attention_config, 'use_domain_attention', False):
            self.domain_attention = DomainSpecificAttention(
                event_types=getattr(attention_config, 'event_types', []),
                embedding_dim=getattr(attention_config, 'event_embedding_dim', 16)
            )
        else:
            self.domain_attention = None
        
        # Feature fusion
        self.feature_fusion = layers.Dense(
            lstm_config.lstm_units[-1], 
            activation='relu', 
            name='feature_fusion'
        )
        self.fusion_dropout = layers.Dropout(lstm_config.dropout_rate)
        
        # Multi-output heads
        self.output_heads = {}
        for horizon_name in multi_step_config.horizon_names:
            self.output_heads[horizon_name] = [
                layers.Dense(64, activation='relu', name=f'complete_{horizon_name}_dense'),
                layers.Dropout(lstm_config.dropout_rate, name=f'complete_{horizon_name}_dropout'),
                layers.Dense(lstm_config.num_classes, activation='softmax', name=f'complete_{horizon_name}_output')
            ]
    
    def call(self, inputs, event_types=None, training=None):
        """Forward pass through complete model"""
        x = inputs
        
        # LSTM processing
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Multi-head attention
        attended_x = self.attention(query=x, value=x, key=x, training=training)
        
        # Domain-specific attention if available
        if self.domain_attention is not None and event_types is not None:
            domain_x, _ = self.domain_attention(x, event_types)
            # Combine attention outputs
            pooled_attended = tf.reduce_mean(attended_x, axis=1)
            fused_features = tf.concat([pooled_attended, domain_x], axis=-1)
        else:
            fused_features = tf.reduce_mean(attended_x, axis=1)
        
        # Feature fusion
        fused_features = self.feature_fusion(fused_features, training=training)
        fused_features = self.fusion_dropout(fused_features, training=training)
        
        # Multi-output predictions
        outputs = {}
        for horizon_name, head_layers in self.output_heads.items():
            head_x = fused_features
            for layer in head_layers:
                head_x = layer(head_x, training=training)
            outputs[horizon_name] = head_x
        
        return outputs
    
    def get_attention_weights(self, inputs, event_types=None):
        """Extract attention weights for analysis"""
        x = inputs
        
        # LSTM processing
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=False)
        
        # Get attention weights
        _, attention_weights = self.attention(
            query=x, value=x, key=x, training=False, return_attention_scores=True
        )
        
        attention_weights_list = [attention_weights]
        
        # Get domain attention weights if available
        if self.domain_attention is not None and event_types is not None:
            _, domain_weights = self.domain_attention(x, event_types)
            attention_weights_list.append(domain_weights)
        
        return attention_weights_list
