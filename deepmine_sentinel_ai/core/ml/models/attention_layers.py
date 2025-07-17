"""
Attention Mechanisms for LSTM Models

This module implements attention layers for Task 8:
- Basic attention layer for sequence weighting
- Multi-head attention for complex pattern recognition
- Domain-specific attention for mining event types
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Optional, List, Dict, Any


class AttentionLayer(layers.Layer):
    """
    Basic attention layer for sequence weighting
    
    Computes attention weights to focus on important timesteps
    in mining stability prediction sequences
    """
    
    def __init__(self, units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units) 
        self.V = layers.Dense(1)
        
    def call(self, inputs):
        """
        Apply attention mechanism to input sequences
        
        Args:
            inputs: Input sequences [batch_size, seq_len, features]
            
        Returns:
            context_vector: Weighted context vector
            attention_weights: Attention weights for each timestep
        """
        # Calculate attention scores
        score = self.V(tf.nn.tanh(self.W1(inputs)))  # [batch_size, seq_len, 1]
        
        # Calculate attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # [batch_size, seq_len, 1]
        
        # Calculate context vector
        context_vector = attention_weights * inputs  # [batch_size, seq_len, features]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_size, features]
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class MultiHeadAttention(layers.Layer):
    """
    Multi-head attention layer for complex pattern recognition
    
    Implements scaled dot-product attention with multiple heads
    for capturing different types of relationships in mining data
    """
    
    def __init__(self, 
                 num_heads: int = 8,
                 key_dim: int = 64,
                 value_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim or key_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        
        # Multi-head attention layer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=self.value_dim,
            dropout=dropout_rate,
            use_bias=use_bias
        )
        
        # Layer normalization
        self.layernorm = layers.LayerNormalization()
        
        # Dropout for regularization
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, query, value, key=None, training=None, return_attention_scores=False):
        """
        Apply multi-head attention
        
        Args:
            query: Query sequences [batch_size, seq_len, features]
            value: Value sequences [batch_size, seq_len, features]
            key: Key sequences (optional, defaults to value)
            training: Training mode flag
            return_attention_scores: Whether to return attention scores
            
        Returns:
            output: Attention output
            attention_scores: Attention scores (if requested)
        """
        if key is None:
            key = value
            
        # Apply multi-head attention
        attention_output, attention_scores = self.mha(
            query=query,
            value=value, 
            key=key,
            training=training,
            return_attention_scores=True
        )
        
        # Apply dropout
        attention_output = self.dropout(attention_output, training=training)
        
        # Add & norm (residual connection)
        output = self.layernorm(query + attention_output)
        
        if return_attention_scores:
            return output, attention_scores
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'value_dim': self.value_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias
        })
        return config


class DomainSpecificAttention(layers.Layer):
    """
    Domain-specific attention for mining event types
    
    Applies different attention patterns based on mining event types
    to focus on relevant patterns for each event category
    """
    
    def __init__(self,
                 event_types: List[str],
                 embedding_dim: int = 16,
                 attention_units: int = 64,
                 **kwargs):
        super(DomainSpecificAttention, self).__init__(**kwargs)
        self.event_types = event_types
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units
        self.num_event_types = len(event_types)
        
        # Event type embedding
        self.event_embedding = layers.Embedding(
            input_dim=self.num_event_types,
            output_dim=embedding_dim
        )
        
        # Event-specific attention weights
        self.event_attention = layers.Dense(attention_units, activation='tanh')
        self.event_weights = layers.Dense(1)
        
        # Combined attention
        self.combine_attention = layers.Dense(attention_units, activation='tanh')
        self.final_weights = layers.Dense(1)
    
    def call(self, inputs, event_types=None):
        """
        Apply domain-specific attention
        
        Args:
            inputs: Input sequences [batch_size, seq_len, features]
            event_types: Event type indices [batch_size, seq_len]
            
        Returns:
            context_vector: Weighted context vector
            attention_weights: Combined attention weights
        """
        batch_size, seq_len, features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        if event_types is not None:
            # Get event embeddings
            event_emb = self.event_embedding(event_types)  # [batch_size, seq_len, embedding_dim]
            
            # Calculate event-specific attention
            event_scores = self.event_weights(
                self.event_attention(event_emb)
            )  # [batch_size, seq_len, 1]
            
            # Combine with input features for attention calculation
            combined_features = tf.concat([inputs, event_emb], axis=-1)
            attention_scores = self.final_weights(
                self.combine_attention(combined_features)
            )  # [batch_size, seq_len, 1]
            
            # Combine event and feature attention
            combined_scores = attention_scores + event_scores
            
        else:
            # Default attention without event information
            attention_scores = self.final_weights(
                self.combine_attention(inputs)
            )  # [batch_size, seq_len, 1]
            combined_scores = attention_scores
        
        # Calculate final attention weights
        attention_weights = tf.nn.softmax(combined_scores, axis=1)  # [batch_size, seq_len, 1]
        
        # Calculate context vector
        context_vector = attention_weights * inputs  # [batch_size, seq_len, features]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_size, features]
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'event_types': self.event_types,
            'embedding_dim': self.embedding_dim,
            'attention_units': self.attention_units
        })
        return config


class TemporalAttention(layers.Layer):
    """
    Temporal attention for time-aware pattern recognition
    
    Applies attention based on temporal patterns and importance
    of different time periods in mining stability prediction
    """
    
    def __init__(self, 
                 time_units: int = 32,
                 attention_units: int = 64,
                 **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.time_units = time_units
        self.attention_units = attention_units
        
        # Time encoding layers
        self.time_encoder = layers.Dense(time_units, activation='relu')
        
        # Temporal attention weights
        self.temporal_attention = layers.Dense(attention_units, activation='tanh')
        self.temporal_weights = layers.Dense(1)
        
        # Position encoding for time-awareness
        self.positional_encoding = self._create_positional_encoding
    
    def _create_positional_encoding(self, seq_len: int, d_model: int):
        """Create positional encoding for temporal awareness"""
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                         -(np.log(10000.0) / d_model))
        
        pos_encoding = tf.zeros((seq_len, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
            tf.sin(position * div_term)
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
            tf.cos(position * div_term)
        )
        
        return pos_encoding[tf.newaxis, ...]
    
    def call(self, inputs):
        """
        Apply temporal attention
        
        Args:
            inputs: Input sequences [batch_size, seq_len, features]
            
        Returns:
            context_vector: Temporally weighted context vector
            attention_weights: Temporal attention weights
        """
        batch_size, seq_len, features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(seq_len, self.time_units)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])
        
        # Encode temporal information
        time_features = self.time_encoder(inputs)  # [batch_size, seq_len, time_units]
        time_features = time_features + pos_encoding
        
        # Calculate temporal attention weights
        attention_scores = self.temporal_weights(
            self.temporal_attention(time_features)
        )  # [batch_size, seq_len, 1]
        
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch_size, seq_len, 1]
        
        # Calculate context vector
        context_vector = attention_weights * inputs  # [batch_size, seq_len, features]
        context_vector = tf.reduce_sum(context_vector, axis=1)  # [batch_size, features]
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'time_units': self.time_units,
            'attention_units': self.attention_units
        })
        return config
