"""
Data Preprocessing Pipeline for LSTM Training

Transforms raw operational data into sequences suitable for LSTM training:
- Time series sequence generation (168-hour w                'timestamp': record.timestamp,
                'impact_score': record.new_score,
                'previous_score': record.previous_score,ows)
- Feature standardization and normalization
- Static feature encoding
- Data validation and quality checks

Based on PROJECT_FINAL_REPORT.md specifications:
- Sequence length: 168 hours (7 days)
- Prediction horizon: 24 hours
- Static features: RQD, depth, dip, rock type, mining method
- Dynamic features: Impact scores, operational events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from django.utils import timezone
from django.db.models import Q, Avg, Count, Max, Min
from core.models import Stope, OperationalEvent, ImpactScore, ImpactHistory
from core.impact.impact_service import ImpactCalculationService

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing pipeline"""
    sequence_length: int = 168  # 7 days in hours
    prediction_horizon: int = 24  # 24 hours ahead
    sampling_interval: int = 1  # 1 hour intervals
    min_events_per_sequence: int = 1  # Minimum events to include sequence
    validation_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    

@dataclass
class SequenceData:
    """Container for processed sequence data"""
    static_features: np.ndarray  # Static stope characteristics
    dynamic_features: np.ndarray  # Time series impact sequences
    risk_labels: np.ndarray  # Risk level classifications
    timestamps: np.ndarray  # Sequence timestamps
    stope_ids: np.ndarray  # Stope identifiers
    metadata: Dict  # Additional information
    
    @property
    def num_samples(self) -> int:
        """Number of samples in the dataset"""
        return len(self.static_features) if self.static_features is not None else 0
    
    @property 
    def feature_dim(self) -> int:
        """Number of dynamic features per timestep"""
        return self.dynamic_features.shape[-1] if self.dynamic_features is not None else 0


class DataPreprocessor:
    """
    Main preprocessing pipeline for LSTM training data
    
    Handles:
    - Time series sequence extraction
    - Feature standardization
    - Static feature encoding
    - Data validation and cleaning
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """Initialize preprocessor with configuration"""
        self.config = config or PreprocessingConfig()
        self.impact_service = ImpactCalculationService()
        
        # Scalers for different feature types
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.impact_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Encoders for categorical features
        self.rock_type_encoder = LabelEncoder()
        self.mining_method_encoder = LabelEncoder()
        
        # Feature statistics for validation
        self.feature_stats = {}
        
        logger.info("Data Preprocessor initialized")
        logger.info(f"Sequence length: {self.config.sequence_length} hours")
        logger.info(f"Prediction horizon: {self.config.prediction_horizon} hours")
    
    def prepare_sequences(self, stopes: List[Stope] = None) -> Optional[SequenceData]:
        """
        Prepare LSTM training sequences from all available data
        
        Simplified wrapper for validation commands
        """
        try:
            return self.process_all_stopes(stopes)
        except Exception as e:
            logger.error(f"Failed to prepare sequences: {str(e)}")
            return None
    
    def extract_static_features(self, stopes: List[Stope]) -> np.ndarray:
        """
        Extract and encode static stope characteristics
        
        Features based on PROJECT_FINAL_REPORT.md:
        - RQD (Rock Quality Designation)
        - Depth
        - Dip angle
        - Rock type (encoded)
        - Mining method (encoded)
        - Undercut width
        - Support density
        - Height/Width ratio
        """
        static_data = []
        
        # Extract categorical values for encoding
        rock_types = [stope.rock_type for stope in stopes]
        mining_methods = [stope.mining_method for stope in stopes]
        
        # Fit encoders
        self.rock_type_encoder.fit(rock_types)
        self.mining_method_encoder.fit(mining_methods)
        
        for stope in stopes:
            features = [
                stope.rqd,  # Rock Quality Designation
                stope.depth,  # Depth in meters
                stope.dip,  # Dip angle in degrees
                self.rock_type_encoder.transform([stope.rock_type])[0],  # Encoded rock type
                self.mining_method_encoder.transform([stope.mining_method])[0],  # Encoded mining method
                stope.undercut_width,  # Undercut width
                stope.support_density,  # Support density
                stope.hr,  # Height/Width ratio
            ]
            static_data.append(features)
        
        static_features = np.array(static_data, dtype=np.float32)
        
        # Standardize numerical features
        static_features = self.static_scaler.fit_transform(static_features)
        
        logger.info(f"Extracted static features: shape {static_features.shape}")
        return static_features
    
    def generate_time_series_sequences(self, stope: Stope, 
                                     start_time: datetime, 
                                     end_time: datetime) -> List[Dict]:
        """
        Generate time series sequences for a single stope
        
        Creates sliding windows of impact scores and operational events
        """
        sequences = []
        
        # Get all impact history for the stope in time range
        impact_history = ImpactHistory.objects.filter(
            stope=stope,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')
        
        if impact_history.count() < self.config.sequence_length:
            logger.debug(f"Insufficient impact history for {stope.stope_name}")
            return sequences
        
        # Convert to pandas DataFrame for easier manipulation
        impact_data = []
        for record in impact_history:
            impact_data.append({
                'timestamp': record.timestamp,
                'impact_score': record.new_score,
                'previous_score': record.previous_score
            })
        
        impact_df = pd.DataFrame(impact_data)
        impact_df['timestamp'] = pd.to_datetime(impact_df['timestamp'])
        impact_df = impact_df.set_index('timestamp')
        
        # Resample to hourly intervals and forward fill missing values
        hourly_impact = impact_df.resample('1h').last().ffill()
        
        # Generate sliding windows
        window_size = self.config.sequence_length
        prediction_offset = self.config.prediction_horizon
        
        for i in range(len(hourly_impact) - window_size - prediction_offset + 1):
            # Extract sequence window
            sequence_start = i
            sequence_end = i + window_size
            prediction_time = i + window_size + prediction_offset - 1
            
            if prediction_time >= len(hourly_impact):
                break
            
            # Extract features for this sequence
            sequence_impact = hourly_impact.iloc[sequence_start:sequence_end]
            prediction_impact = hourly_impact.iloc[prediction_time]
            
            # Calculate additional dynamic features
            sequence_features = []
            for _, row in sequence_impact.iterrows():
                # Basic impact features
                features = [
                    row['impact_score'],
                    row['previous_score'] if pd.notna(row['previous_score']) else 0.0,
                    row['impact_score'] - (row['previous_score'] if pd.notna(row['previous_score']) else 0.0)  # Impact change
                ]
                sequence_features.append(features)
            
            # Get operational events in sequence window
            sequence_start_time = hourly_impact.index[sequence_start]
            sequence_end_time = hourly_impact.index[sequence_end - 1]
            
            events_in_window = OperationalEvent.objects.filter(
                stope=stope,
                timestamp__gte=sequence_start_time,
                timestamp__lte=sequence_end_time
            )
            
            # Add event-based features to sequence
            event_features = self._extract_event_features(events_in_window, 
                                                        sequence_start_time, 
                                                        sequence_end_time)
            
            # Combine impact and event features
            for i, features in enumerate(sequence_features):
                hour_events = event_features[i] if i < len(event_features) else [0.0] * 6
                features.extend(hour_events)
            
            # Determine risk level for prediction target
            prediction_score = prediction_impact['impact_score']
            risk_level = self._score_to_risk_level(prediction_score)
            
            sequence_data = {
                'stope_id': stope.id,
                'stope_name': stope.stope_name,
                'sequence_features': np.array(sequence_features, dtype=np.float32),
                'risk_label': risk_level,
                'prediction_score': prediction_score,
                'sequence_start': sequence_start_time,
                'sequence_end': sequence_end_time,
                'prediction_time': hourly_impact.index[prediction_time],
                'event_count': events_in_window.count()
            }
            
            sequences.append(sequence_data)
        
        logger.debug(f"Generated {len(sequences)} sequences for {stope.stope_name}")
        return sequences
    
    def _extract_event_features(self, events, start_time: datetime, end_time: datetime) -> List[List[float]]:
        """
        Extract event-based features for each hour in sequence window
        
        Features per hour:
        - Blasting intensity sum
        - Equipment operation count
        - Water exposure duration
        - Drilling count
        - Mucking count  
        - Support installation count
        """
        # Create hourly bins
        time_range = pd.date_range(start=start_time, end=end_time, freq='1H')
        hourly_features = []
        
        # Event type mappings
        event_type_map = {
            'blasting': 0,
            'heavy_equipment': 1,
            'water_exposure': 2,
            'drilling': 3,
            'mucking': 4,
            'support_installation': 5
        }
        
        for hour_start in time_range[:-1]:  # Exclude last hour (end time)
            hour_end = hour_start + timedelta(hours=1)
            
            # Initialize feature vector for this hour
            hour_features = [0.0] * 6  # One for each event type
            
            # Find events in this hour
            hour_events = events.filter(
                timestamp__gte=hour_start,
                timestamp__lt=hour_end
            )
            
            for event in hour_events:
                event_idx = event_type_map.get(event.event_type, -1)
                if event_idx >= 0:
                    if event.event_type == 'blasting':
                        hour_features[event_idx] += event.intensity  # Sum intensity for blasting
                    elif event.event_type == 'water_exposure':
                        hour_features[event_idx] += event.duration  # Sum duration for water
                    else:
                        hour_features[event_idx] += 1.0  # Count for other events
            
            hourly_features.append(hour_features)
        
        return hourly_features
    
    def _score_to_risk_level(self, impact_score: float) -> int:
        """
        Convert impact score to risk level classification
        
        Risk levels based on established thresholds:
        0: Stable (0.0 - 2.0)
        1: Elevated (2.0 - 5.0)
        2: High Risk (5.0 - 8.0)
        3: Critical (8.0+)
        """
        if impact_score < 2.0:
            return 0  # Stable
        elif impact_score < 5.0:
            return 1  # Elevated
        elif impact_score < 8.0:
            return 2  # High Risk
        else:
            return 3  # Critical
    
    def process_all_stopes(self, stopes: List[Stope] = None) -> SequenceData:
        """
        Process all stopes to generate complete training dataset
        
        Returns processed sequences ready for LSTM training
        """
        if stopes is None:
            stopes = list(Stope.objects.filter(is_active=True))
        
        logger.info(f"Processing {len(stopes)} stopes for LSTM training data")
        
        # Extract static features
        static_features = self.extract_static_features(stopes)
        
        # Generate time series sequences for all stopes
        all_sequences = []
        
        # Define time range for sequence generation (use available data)
        end_time = timezone.now()
        start_time = end_time - timedelta(days=self.config.sequence_length * 2)  # Extra buffer
        
        for i, stope in enumerate(stopes):
            try:
                stope_sequences = self.generate_time_series_sequences(stope, start_time, end_time)
                
                for seq in stope_sequences:
                    # Add static features for this stope
                    seq['static_features'] = static_features[i]
                    all_sequences.append(seq)
                    
            except Exception as e:
                logger.warning(f"Failed to process stope {stope.stope_name}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid sequences generated. Check data availability and configuration.")
        
        # Convert to arrays
        static_array = np.array([seq['static_features'] for seq in all_sequences])
        dynamic_array = np.array([seq['sequence_features'] for seq in all_sequences])
        risk_labels = np.array([seq['risk_label'] for seq in all_sequences])
        timestamps = np.array([seq['sequence_start'] for seq in all_sequences])
        stope_ids = np.array([seq['stope_id'] for seq in all_sequences])
        
        # Normalize dynamic features
        original_shape = dynamic_array.shape
        dynamic_reshaped = dynamic_array.reshape(-1, dynamic_array.shape[-1])
        dynamic_normalized = self.dynamic_scaler.fit_transform(dynamic_reshaped)
        dynamic_array = dynamic_normalized.reshape(original_shape)
        
        # Create metadata
        metadata = {
            'total_sequences': len(all_sequences),
            'sequence_length': self.config.sequence_length,
            'feature_dimension': dynamic_array.shape[-1],
            'static_feature_dimension': static_array.shape[-1],
            'num_classes': 4,  # Stable, Elevated, High Risk, Critical
            'class_distribution': np.bincount(risk_labels),
            'processing_timestamp': timezone.now(),
            'stope_count': len(set(stope_ids)),
            'config': self.config
        }
        
        logger.info(f"Processed {len(all_sequences)} sequences from {len(set(stope_ids))} stopes")
        logger.info(f"Dynamic features shape: {dynamic_array.shape}")
        logger.info(f"Static features shape: {static_array.shape}")
        logger.info(f"Class distribution: {metadata['class_distribution']}")
        
        return SequenceData(
            static_features=static_array,
            dynamic_features=dynamic_array,
            risk_labels=risk_labels,
            timestamps=timestamps,
            stope_ids=stope_ids,
            metadata=metadata
        )
    
    def create_train_val_test_splits(self, sequence_data: SequenceData) -> Tuple[Dict, Dict, Dict]:
        """
        Create training, validation, and test splits with temporal considerations
        
        Uses temporal split to prevent data leakage:
        - Training: Earliest sequences
        - Validation: Middle sequences  
        - Test: Latest sequences
        """
        total_sequences = len(sequence_data.timestamps)
        
        # Sort by timestamp to ensure temporal order
        sort_indices = np.argsort(sequence_data.timestamps)
        
        # Calculate split indices
        train_size = int(total_sequences * (1 - self.config.validation_split - self.config.test_split))
        val_size = int(total_sequences * self.config.validation_split)
        
        train_indices = sort_indices[:train_size]
        val_indices = sort_indices[train_size:train_size + val_size]
        test_indices = sort_indices[train_size + val_size:]
        
        def create_split(indices):
            return {
                'static_features': sequence_data.static_features[indices],
                'dynamic_features': sequence_data.dynamic_features[indices],
                'risk_labels': sequence_data.risk_labels[indices],
                'timestamps': sequence_data.timestamps[indices],
                'stope_ids': sequence_data.stope_ids[indices],
                'indices': indices
            }
        
        train_data = create_split(train_indices)
        val_data = create_split(val_indices)
        test_data = create_split(test_indices)
        
        logger.info(f"Created data splits:")
        logger.info(f"  Training: {len(train_indices)} sequences")
        logger.info(f"  Validation: {len(val_indices)} sequences")
        logger.info(f"  Test: {len(test_indices)} sequences")
        
        return train_data, val_data, test_data
    
    def validate_data_quality(self, sequence_data: SequenceData) -> Dict:
        """
        Validate data quality and identify potential issues
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        # Check for missing values
        static_missing = np.isnan(sequence_data.static_features).sum()
        dynamic_missing = np.isnan(sequence_data.dynamic_features).sum()
        
        if static_missing > 0:
            validation_results['issues'].append(f"Missing values in static features: {static_missing}")
            validation_results['valid'] = False
        
        if dynamic_missing > 0:
            validation_results['issues'].append(f"Missing values in dynamic features: {dynamic_missing}")
            validation_results['valid'] = False
        
        # Check class imbalance
        class_counts = np.bincount(sequence_data.risk_labels)
        min_class_ratio = np.min(class_counts) / np.max(class_counts)
        
        if min_class_ratio < 0.1:  # Severe imbalance
            validation_results['issues'].append(f"Severe class imbalance: {class_counts}")
            validation_results['recommendations'].append("Consider data augmentation for minority classes")
        
        # Check sequence length consistency
        if len(set(seq.shape[0] for seq in sequence_data.dynamic_features)) > 1:
            validation_results['issues'].append("Inconsistent sequence lengths")
            validation_results['valid'] = False
        
        # Feature statistics
        validation_results['statistics'] = {
            'total_sequences': len(sequence_data.risk_labels),
            'static_feature_stats': {
                'mean': np.mean(sequence_data.static_features, axis=0).tolist(),
                'std': np.std(sequence_data.static_features, axis=0).tolist(),
                'min': np.min(sequence_data.static_features, axis=0).tolist(),
                'max': np.max(sequence_data.static_features, axis=0).tolist()
            },
            'dynamic_feature_stats': {
                'mean': np.mean(sequence_data.dynamic_features).item(),
                'std': np.std(sequence_data.dynamic_features).item(),
                'min': np.min(sequence_data.dynamic_features).item(),
                'max': np.max(sequence_data.dynamic_features).item()
            },
            'class_distribution': class_counts.tolist(),
            'class_percentages': (class_counts / len(sequence_data.risk_labels) * 100).tolist()
        }
        
        return validation_results
