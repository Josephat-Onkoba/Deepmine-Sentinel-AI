"""
Feature Engineering for Operational Event Patterns

Advanced feature extraction and engineering for LSTM training:
- Temporal pattern features
- Operational event pattern recognition
- Rolling statistics and trend analysis
- Interaction features between static and dynamic data

Based on PROJECT_FINAL_REPORT.md specifications for feature engineering
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from core.models import Stope, OperationalEvent
from core.data.preprocessor import SequenceData

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering operations"""
    rolling_windows: List[int] = None  # [6, 12, 24, 48] hours
    trend_periods: List[int] = None    # [12, 24, 48] hours for trend analysis
    event_clustering: bool = True      # Enable event pattern clustering
    pca_components: int = 10           # PCA components for dimensionality reduction
    interaction_features: bool = True  # Generate interaction features
    spectral_features: bool = True     # FFT-based frequency features
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [6, 12, 24, 48]  # 6h, 12h, 1d, 2d
        if self.trend_periods is None:
            self.trend_periods = [12, 24, 48]


class FeatureEngineer:
    """
    Advanced feature engineering for operational event patterns
    
    Generates sophisticated features to capture:
    - Temporal patterns and seasonality
    - Event clustering and pattern recognition
    - Rolling statistics and trend indicators
    - Frequency domain characteristics
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        """Initialize feature engineer with configuration"""
        self.config = config or FeatureEngineeringConfig()
        
        # Initialize clustering models for pattern recognition
        self.event_clusterer = KMeans(n_clusters=5, random_state=42) if self.config.event_clustering else None
        self.impact_clusterer = KMeans(n_clusters=3, random_state=42)
        
        # PCA for dimensionality reduction
        self.pca_transformer = PCA(n_components=self.config.pca_components) if self.config.pca_components > 0 else None
        
        logger.info("Feature Engineer initialized")
        logger.info(f"Rolling windows: {self.config.rolling_windows}")
        logger.info(f"Trend periods: {self.config.trend_periods}")
    
    def extract_temporal_features(self, sequences: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Extract temporal pattern features from sequences
        
        Features include:
        - Hour of day patterns
        - Day of week patterns  
        - Seasonal trends
        - Time since last significant event
        """
        # Handle empty input gracefully
        if sequences.size == 0:
            logger.warning("Empty sequences provided to temporal feature extraction")
            return np.array([])
        
        batch_size, sequence_length, feature_dim = sequences.shape
        temporal_features = []
        
        # Handle invalid or empty timestamps
        if timestamps.size == 0 or len(timestamps) < batch_size:
            logger.warning("Invalid timestamps provided, using default timestamps")
            # Create default timestamps starting from a reference time
            base_time = pd.Timestamp.now()
            timestamps = np.array([base_time + pd.Timedelta(hours=i) for i in range(batch_size)])
        
        for i, sequence in enumerate(sequences):
            # Convert timestamps to datetime objects for this sequence
            try:
                # Handle case where timestamps[i] might be empty or invalid
                if i < len(timestamps) and timestamps[i] is not None:
                    try:
                        seq_start = pd.to_datetime(timestamps[i])
                    except (ValueError, TypeError):
                        seq_start = pd.Timestamp.now() + pd.Timedelta(hours=i)
                else:
                    seq_start = pd.Timestamp.now() + pd.Timedelta(hours=i)
                
                seq_times = pd.date_range(start=seq_start, periods=sequence_length, freq='1h')
                
                # Hour of day features (cyclical encoding)
                hours = seq_times.hour
                hour_sin = np.sin(2 * np.pi * hours / 24)
                hour_cos = np.cos(2 * np.pi * hours / 24)
                
                # Day of week features (cyclical encoding)
                days = seq_times.dayofweek
                day_sin = np.sin(2 * np.pi * days / 7)
                day_cos = np.cos(2 * np.pi * days / 7)
                
                # Time-based statistics
                seq_features = np.column_stack([
                    hour_sin, hour_cos,
                    day_sin, day_cos
                ])
                
                temporal_features.append(seq_features)
                
            except Exception as e:
                logger.warning(f"Error processing timestamp {i}: {e}")
                # Create zero features as fallback
                seq_features = np.zeros((sequence_length, 4))
                temporal_features.append(seq_features)
        
        if not temporal_features:
            logger.warning("No temporal features generated")
            return np.array([])
        
        temporal_array = np.array(temporal_features, dtype=np.float32)
        logger.debug(f"Generated temporal features: shape {temporal_array.shape}")
        return temporal_array
    
    def extract_rolling_statistics(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract rolling statistical features from impact sequences
        
        For each rolling window:
        - Mean, std, min, max
        - Slope and trend indicators
        - Variance and volatility measures
        """
        if sequences.size == 0:
            logger.warning("Empty sequences provided to rolling statistics extraction")
            return np.array([])
            
        batch_size, sequence_length, feature_dim = sequences.shape
        rolling_features = []
        
        for sequence in sequences:
            seq_rolling_features = []
            
            # Extract impact scores (assume first feature is primary impact)
            if feature_dim > 0:
                impact_scores = sequence[:, 0]  # Primary impact score
            else:
                logger.warning("No features available for rolling statistics")
                # Create dummy feature array
                rolling_features.append(np.zeros((sequence_length, len(self.config.rolling_windows) * 8)))
                continue
            
            for window_size in self.config.rolling_windows:
                if window_size >= sequence_length:
                    # If window is larger than sequence, create zero features
                    window_rolling_stats = np.zeros((sequence_length, 8))
                    seq_rolling_features.extend(window_rolling_stats.T)
                    continue
                
                # Calculate rolling statistics
                rolling_stats = []
                
                for i in range(window_size, sequence_length):
                    window_data = impact_scores[i-window_size:i]
                    
                    # Basic statistics
                    window_mean = np.mean(window_data)
                    window_std = np.std(window_data)
                    window_min = np.min(window_data)
                    window_max = np.max(window_data)
                    
                    # Trend indicators
                    if len(window_data) > 1:
                        try:
                            slope, _, r_value, _, _ = stats.linregress(range(len(window_data)), window_data)
                            trend_strength = abs(r_value)
                        except Exception as e:
                            logger.warning(f"Error in trend calculation: {e}")
                            slope = 0.0
                            trend_strength = 0.0
                    else:
                        slope = 0.0
                        trend_strength = 0.0
                    
                    # Volatility measures
                    volatility = window_std / (window_mean + 1e-6)  # Coefficient of variation
                    
                    # Recent vs historical comparison
                    recent_mean = np.mean(window_data[-window_size//4:]) if window_size >= 4 else window_mean
                    historical_mean = np.mean(window_data[:-window_size//4]) if window_size >= 4 else window_mean
                    relative_change = (recent_mean - historical_mean) / (historical_mean + 1e-6)
                    
                    window_stats = [
                        window_mean, window_std, window_min, window_max,
                        slope, trend_strength, volatility, relative_change
                    ]
                    rolling_stats.append(window_stats)
                
                # Pad beginning with zeros where window is not available
                padding = [[0.0] * 8] * window_size
                window_rolling_stats = padding + rolling_stats
                
                # Ensure we have the correct sequence length
                if len(window_rolling_stats) != sequence_length:
                    logger.warning(f"Rolling stats length mismatch: {len(window_rolling_stats)} vs {sequence_length}")
                    # Pad or truncate to match sequence length
                    if len(window_rolling_stats) < sequence_length:
                        padding_needed = sequence_length - len(window_rolling_stats)
                        window_rolling_stats.extend([[0.0] * 8] * padding_needed)
                    else:
                        window_rolling_stats = window_rolling_stats[:sequence_length]
                
                seq_rolling_features.extend(np.array(window_rolling_stats).T)  # Transpose for feature-wise organization
            
            if seq_rolling_features:
                try:
                    feature_matrix = np.column_stack(seq_rolling_features)
                    rolling_features.append(feature_matrix)
                except Exception as e:
                    logger.warning(f"Error creating feature matrix: {e}")
                    # Create zero matrix as fallback
                    rolling_features.append(np.zeros((sequence_length, len(self.config.rolling_windows) * 8)))
            else:
                rolling_features.append(np.zeros((sequence_length, len(self.config.rolling_windows) * 8)))
        
        if not rolling_features:
            logger.warning("No rolling features generated")
            return np.array([])
            
        rolling_array = np.array(rolling_features, dtype=np.float32)
        logger.debug(f"Generated rolling statistics features: shape {rolling_array.shape}")
        return rolling_array
    
    def extract_event_pattern_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract operational event pattern features
        
        Features include:
        - Event clustering patterns
        - Event frequency and intensity patterns
        - Event type interaction patterns
        - Event timing patterns
        """
        batch_size, sequence_length, feature_dim = sequences.shape
        
        # Assuming event features start from index 3 (after impact, previous, change)
        event_start_idx = 3
        event_features = sequences[:, :, event_start_idx:]  # Extract event columns
        
        pattern_features = []
        
        # Fit clustering model on all event patterns if enabled
        if self.config.event_clustering and self.event_clusterer is not None:
            # Reshape for clustering: (batch * sequence, event_features)
            event_flat = event_features.reshape(-1, event_features.shape[-1])
            event_clusters = self.event_clusterer.fit_predict(event_flat)
            event_clusters = event_clusters.reshape(batch_size, sequence_length)
        else:
            event_clusters = np.zeros((batch_size, sequence_length))
        
        for i, sequence in enumerate(sequences):
            seq_event_features = event_features[i]  # (sequence_length, event_feature_dim)
            seq_patterns = []
            
            for t in range(sequence_length):
                # Current event pattern
                current_events = seq_event_features[t]
                
                # Event intensity features
                total_intensity = np.sum(current_events)
                max_intensity = np.max(current_events) if len(current_events) > 0 else 0.0
                event_diversity = np.count_nonzero(current_events) / len(current_events) if len(current_events) > 0 else 0.0
                
                # Event clustering pattern
                cluster_id = event_clusters[i, t] if self.config.event_clustering else 0
                
                # Event type dominance (which type is most active)
                dominant_event_type = np.argmax(current_events) if len(current_events) > 0 else 0
                
                # Recent event history patterns (look back 12 hours)
                lookback_window = min(12, t + 1)
                if t >= lookback_window - 1:
                    recent_events = seq_event_features[t-lookback_window+1:t+1]
                    
                    # Event frequency in recent history
                    event_frequency = np.mean(np.sum(recent_events, axis=1))
                    
                    # Event trend (increasing or decreasing activity)
                    recent_intensities = np.sum(recent_events, axis=1)
                    if len(recent_intensities) > 1:
                        event_trend, _, _, _, _ = stats.linregress(range(len(recent_intensities)), recent_intensities)
                    else:
                        event_trend = 0.0
                        
                    # Event pattern stability (how consistent are event patterns)
                    pattern_stability = 1.0 / (1.0 + np.std(recent_intensities))
                else:
                    event_frequency = total_intensity
                    event_trend = 0.0
                    pattern_stability = 1.0
                
                pattern_features_t = [
                    total_intensity,
                    max_intensity,
                    event_diversity,
                    cluster_id,
                    dominant_event_type,
                    event_frequency,
                    event_trend,
                    pattern_stability
                ]
                
                seq_patterns.append(pattern_features_t)
            
            pattern_features.append(seq_patterns)
        
        pattern_array = np.array(pattern_features, dtype=np.float32)
        logger.debug(f"Generated event pattern features: shape {pattern_array.shape}")
        return pattern_array
    
    def extract_spectral_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        Extract frequency domain features using FFT
        
        Captures cyclical patterns in impact scores and operational events
        """
        if not self.config.spectral_features:
            return np.array([])
            
        batch_size, sequence_length, feature_dim = sequences.shape
        spectral_features = []
        
        for sequence in sequences:
            # Extract impact time series (first column)
            impact_series = sequence[:, 0]
            
            # Apply FFT to extract frequency components
            fft_result = np.fft.fft(impact_series)
            fft_magnitude = np.abs(fft_result)
            
            # Extract key frequency features
            # Take first half of spectrum (positive frequencies)
            half_spectrum = fft_magnitude[:sequence_length//2]
            
            # Extract dominant frequencies
            dominant_freq_idx = np.argmax(half_spectrum[1:]) + 1  # Skip DC component
            dominant_freq_magnitude = half_spectrum[dominant_freq_idx]
            
            # Spectral energy in different bands
            low_freq_energy = np.sum(half_spectrum[1:sequence_length//8])    # Low frequency patterns
            mid_freq_energy = np.sum(half_spectrum[sequence_length//8:sequence_length//4])  # Mid frequency
            high_freq_energy = np.sum(half_spectrum[sequence_length//4:])    # High frequency (noise)
            
            # Spectral centroid (center of mass of spectrum)
            freqs = np.arange(len(half_spectrum))
            spectral_centroid = np.sum(freqs * half_spectrum) / (np.sum(half_spectrum) + 1e-6)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumulative_energy = np.cumsum(half_spectrum)
            total_energy = cumulative_energy[-1]
            rolloff_idx = np.argmax(cumulative_energy >= 0.85 * total_energy)
            spectral_rolloff = rolloff_idx / len(half_spectrum)
            
            seq_spectral_features = [
                dominant_freq_idx / len(half_spectrum),  # Normalized dominant frequency
                dominant_freq_magnitude,
                low_freq_energy,
                mid_freq_energy,
                high_freq_energy,
                spectral_centroid / len(half_spectrum),  # Normalized centroid
                spectral_rolloff
            ]
            
            spectral_features.append(seq_spectral_features)
        
        spectral_array = np.array(spectral_features, dtype=np.float32)
        logger.debug(f"Generated spectral features: shape {spectral_array.shape}")
        return spectral_array
    
    def extract_interaction_features(self, static_features: np.ndarray, 
                                   dynamic_sequences: np.ndarray) -> np.ndarray:
        """
        Extract interaction features between static and dynamic characteristics
        
        Captures how geological conditions interact with operational patterns
        """
        if not self.config.interaction_features:
            return np.array([])
            
        batch_size, sequence_length, dynamic_dim = dynamic_sequences.shape
        static_dim = static_features.shape[1]
        
        interaction_features = []
        
        for i in range(batch_size):
            static_props = static_features[i]  # Static properties for this stope
            dynamic_seq = dynamic_sequences[i]  # Dynamic sequence for this stope
            
            seq_interactions = []
            
            for t in range(sequence_length):
                dynamic_t = dynamic_seq[t]
                
                # Key interaction features:
                
                # 1. Depth-Impact interaction (deeper stopes may have different impact responses)
                depth_normalized = static_props[1]  # Assuming depth is 2nd static feature
                depth_impact_interaction = depth_normalized * dynamic_t[0]  # depth * current impact
                
                # 2. Rock quality (RQD) interaction with impact accumulation
                rqd_normalized = static_props[0]  # Assuming RQD is 1st static feature
                rqd_impact_interaction = rqd_normalized * dynamic_t[0]
                
                # 3. Rock type influence on event response
                rock_type_encoded = static_props[3]  # Assuming encoded rock type is 4th feature
                rock_event_interaction = rock_type_encoded * np.sum(dynamic_t[3:])  # rock type * event activity
                
                # 4. Mining method interaction with operational patterns
                mining_method_encoded = static_props[4]  # Assuming encoded mining method is 5th feature
                method_pattern_interaction = mining_method_encoded * np.mean(dynamic_t[3:])
                
                # 5. Support density interaction with impact development
                support_density = static_props[6]  # Assuming support density is 7th feature
                support_impact_interaction = support_density * dynamic_t[0]
                
                # 6. Geometric factors (HR ratio) interaction with stability
                hr_ratio = static_props[7]  # Assuming HR is 8th feature
                geometry_stability_interaction = hr_ratio * dynamic_t[0]
                
                interaction_t = [
                    depth_impact_interaction,
                    rqd_impact_interaction,
                    rock_event_interaction,
                    method_pattern_interaction,
                    support_impact_interaction,
                    geometry_stability_interaction
                ]
                
                seq_interactions.append(interaction_t)
            
            interaction_features.append(seq_interactions)
        
        interaction_array = np.array(interaction_features, dtype=np.float32)
        logger.debug(f"Generated interaction features: shape {interaction_array.shape}")
        return interaction_array
    
    def engineer_features(self, sequence_data: SequenceData) -> SequenceData:
        """
        Apply complete feature engineering pipeline to sequence data
        
        Returns enhanced SequenceData with additional engineered features
        """
        logger.info("Starting feature engineering pipeline")
        
        original_dynamic = sequence_data.dynamic_features
        static_features = sequence_data.static_features
        timestamps = sequence_data.timestamps
        
        # Extract all feature types
        engineered_features = [original_dynamic]
        
        # 1. Temporal features
        temporal_features = self.extract_temporal_features(original_dynamic, timestamps)
        if temporal_features.size > 0:
            engineered_features.append(temporal_features)
        
        # 2. Rolling statistics features
        rolling_features = self.extract_rolling_statistics(original_dynamic)
        if rolling_features.size > 0:
            engineered_features.append(rolling_features)
        
        # 3. Event pattern features
        pattern_features = self.extract_event_pattern_features(original_dynamic)
        if pattern_features.size > 0:
            engineered_features.append(pattern_features)
        
        # 4. Interaction features
        interaction_features = self.extract_interaction_features(static_features, original_dynamic)
        if interaction_features.size > 0:
            # Interaction features are per-sequence, need to expand to sequence dimension
            interaction_expanded = np.expand_dims(interaction_features, axis=1)
            interaction_tiled = np.tile(interaction_expanded, (1, original_dynamic.shape[1], 1))
            engineered_features.append(interaction_tiled)
        
        # 5. Spectral features (per-sequence features)
        spectral_features = self.extract_spectral_features(original_dynamic)
        if spectral_features.size > 0:
            # Expand spectral features to sequence dimension
            spectral_expanded = np.expand_dims(spectral_features, axis=1)
            spectral_tiled = np.tile(spectral_expanded, (1, original_dynamic.shape[1], 1))
            engineered_features.append(spectral_tiled)
        
        # Concatenate all features along the feature dimension
        # First check dimensions and remove incompatible arrays
        valid_features = []
        target_shape = original_dynamic.shape[:2]  # (batch_size, sequence_length)
        
        for i, feature_array in enumerate(engineered_features):
            if feature_array.size == 0:
                logger.debug(f"Skipping empty feature array at index {i}")
                continue
                
            # Check if array has correct dimensions
            if len(feature_array.shape) != 3:
                logger.warning(f"Feature array at index {i} has {len(feature_array.shape)} dimensions, expected 3")
                continue
                
            # Check if batch and sequence dimensions match
            if feature_array.shape[:2] != target_shape:
                logger.warning(f"Feature array at index {i} has shape {feature_array.shape[:2]}, expected {target_shape}")
                continue
                
            valid_features.append(feature_array)
            logger.debug(f"Added feature array {i} with shape {feature_array.shape}")
        
        if not valid_features:
            logger.warning("No valid features to concatenate, using original features")
            enhanced_dynamic_features = original_dynamic
        else:
            enhanced_dynamic_features = np.concatenate(valid_features, axis=2)
        
        # Update metadata
        enhanced_metadata = sequence_data.metadata.copy()
        enhanced_metadata.update({
            'original_feature_dim': original_dynamic.shape[2],
            'enhanced_feature_dim': enhanced_dynamic_features.shape[2],
            'feature_engineering_applied': True,
            'temporal_features': temporal_features.shape[2] if temporal_features.size > 0 else 0,
            'rolling_features': rolling_features.shape[2] if rolling_features.size > 0 else 0,
            'pattern_features': pattern_features.shape[2] if pattern_features.size > 0 else 0,
            'interaction_features': interaction_features.shape[1] if interaction_features.size > 0 else 0,
            'spectral_features': spectral_features.shape[1] if spectral_features.size > 0 else 0
        })
        
        logger.info(f"Feature engineering completed:")
        logger.info(f"  Original features: {original_dynamic.shape[2]}")
        logger.info(f"  Enhanced features: {enhanced_dynamic_features.shape[2]}")
        logger.info(f"  Feature breakdown: {enhanced_metadata}")
        
        return SequenceData(
            static_features=sequence_data.static_features,
            dynamic_features=enhanced_dynamic_features,
            risk_labels=sequence_data.risk_labels,
            timestamps=sequence_data.timestamps,
            stope_ids=sequence_data.stope_ids,
            metadata=enhanced_metadata
        )
    
    def apply_dimensionality_reduction(self, sequence_data: SequenceData) -> SequenceData:
        """
        Apply PCA dimensionality reduction to dynamic features if configured
        
        Helps reduce feature dimensionality while preserving most important patterns
        """
        if not self.pca_transformer or self.config.pca_components <= 0:
            return sequence_data
        
        original_shape = sequence_data.dynamic_features.shape
        batch_size, sequence_length, feature_dim = original_shape
        
        # Reshape to (batch * sequence, features) for PCA
        features_reshaped = sequence_data.dynamic_features.reshape(-1, feature_dim)
        
        # Apply PCA
        features_reduced = self.pca_transformer.fit_transform(features_reshaped)
        
        # Reshape back to sequence format
        reduced_shape = (batch_size, sequence_length, self.config.pca_components)
        features_final = features_reduced.reshape(reduced_shape)
        
        # Update metadata
        reduced_metadata = sequence_data.metadata.copy()
        reduced_metadata.update({
            'pca_applied': True,
            'pca_components': self.config.pca_components,
            'pca_explained_variance_ratio': self.pca_transformer.explained_variance_ratio_.tolist(),
            'pca_total_variance_explained': np.sum(self.pca_transformer.explained_variance_ratio_),
            'original_feature_dim_before_pca': feature_dim,
            'reduced_feature_dim': self.config.pca_components
        })
        
        logger.info(f"Applied PCA dimensionality reduction:")
        logger.info(f"  Original features: {feature_dim}")
        logger.info(f"  Reduced features: {self.config.pca_components}")
        logger.info(f"  Variance explained: {reduced_metadata['pca_total_variance_explained']:.3f}")
        
        return SequenceData(
            static_features=sequence_data.static_features,
            dynamic_features=features_final,
            risk_labels=sequence_data.risk_labels,
            timestamps=sequence_data.timestamps,
            stope_ids=sequence_data.stope_ids,
            metadata=reduced_metadata
        )
