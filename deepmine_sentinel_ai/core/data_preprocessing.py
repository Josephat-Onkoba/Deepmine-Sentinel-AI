"""
Data Preprocessing Pipeline for LSTM Training

Provides comprehensive data preprocessing, feature engineering, and validation
for time series data used in LSTM-based stability prediction.
"""

import numpy as np
import pandas as pd
from django.db import transaction
from django.utils import timezone
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import json

from core.models import (
    Stope, MonitoringData, OperationalEvent, TimeSeriesData, 
    FeatureEngineeringConfig, DataQualityMetrics, ImpactScore
)

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Main preprocessing pipeline for converting raw monitoring data
    into LSTM-ready time series sequences.
    """
    
    def __init__(self, config_name: str = 'default'):
        """Initialize preprocessor with feature engineering configuration"""
        try:
            self.config = FeatureEngineeringConfig.objects.get(
                config_name=config_name, is_active=True
            )
        except FeatureEngineeringConfig.DoesNotExist:
            # Create default configuration if none exists
            self.config = self._create_default_config(config_name)
        
        self.scaler = None
        self._setup_scaler()
    
    def _create_default_config(self, config_name: str) -> FeatureEngineeringConfig:
        """Create default feature engineering configuration"""
        config = FeatureEngineeringConfig.objects.create(
            config_name=config_name,
            description="Default LSTM preprocessing configuration",
            enabled_sensor_types=[
                'vibration', 'deformation', 'stress', 'acoustic',
                'strain', 'displacement'
            ],
            enabled_feature_types=[
                'raw', 'statistical', 'temporal', 'operational'
            ],
            window_sizes=[1, 4, 12, 24],  # 1h, 4h, 12h, 24h windows
            aggregation_functions=[
                'mean', 'std', 'min', 'max', 'percentile_25', 'percentile_75'
            ],
            include_event_features=True,
            normalization_method='zscore',
            outlier_detection_method='iqr',
            outlier_threshold=3.0
        )
        logger.info(f"Created default configuration: {config_name}")
        return config
    
    def _setup_scaler(self):
        """Setup data scaler based on configuration"""
        if self.config.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config.normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:  # default to zscore
            self.scaler = StandardScaler()
    
    def create_sequences(
        self, 
        stope: Stope,
        start_time: datetime,
        end_time: datetime,
        sequence_length: int = 24,
        sequence_type: str = 'training',
        overlap: float = 0.5
    ) -> List[TimeSeriesData]:
        """
        Create LSTM training sequences from monitoring data
        
        Args:
            stope: Stope to process
            start_time: Start time for data extraction
            end_time: End time for data extraction
            sequence_length: Number of time steps per sequence
            sequence_type: Type of sequence (training, validation, test)
            overlap: Overlap between consecutive sequences (0.0-1.0)
            
        Returns:
            List of TimeSeriesData objects ready for LSTM training
        """
        logger.info(f"Creating sequences for {stope.stope_name} from {start_time} to {end_time}")
        
        # Extract and process raw data
        raw_data = self._extract_raw_data(stope, start_time, end_time)
        if raw_data.empty:
            logger.warning(f"No monitoring data found for {stope.stope_name}")
            return []
        
        # Generate features
        features_df = self._generate_features(stope, raw_data, start_time, end_time)
        if features_df.empty:
            logger.warning(f"No features generated for {stope.stope_name}")
            return []
        
        # Get impact score targets
        targets_df = self._extract_targets(stope, start_time, end_time)
        
        # Create sequences with sliding window
        sequences = []
        step_size = max(1, int(sequence_length * (1 - overlap)))
        
        for start_idx in range(0, len(features_df) - sequence_length + 1, step_size):
            end_idx = start_idx + sequence_length
            
            # Extract sequence data
            seq_features = features_df.iloc[start_idx:end_idx]
            seq_targets = targets_df.iloc[start_idx:end_idx] if not targets_df.empty else None
            
            # Create TimeSeriesData object
            sequence = self._create_sequence_object(
                stope=stope,
                features_df=seq_features,
                targets_df=seq_targets,
                sequence_type=sequence_type,
                start_idx=start_idx
            )
            
            if sequence and sequence.validate_sequence():
                sequences.append(sequence)
        
        logger.info(f"Created {len(sequences)} sequences for {stope.stope_name}")
        return sequences
    
    def _extract_raw_data(
        self, 
        stope: Stope, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """Extract raw monitoring data for the specified time period"""
        
        # Get monitoring data
        monitoring_data = MonitoringData.objects.filter(
            stope=stope,
            timestamp__gte=start_time,
            timestamp__lte=end_time,
            sensor_type__in=self.config.enabled_sensor_types
        ).order_by('timestamp')
        
        if not monitoring_data.exists():
            return pd.DataFrame()
        
        # Convert to DataFrame
        data_list = []
        for record in monitoring_data:
            data_list.append({
                'timestamp': record.timestamp,
                'sensor_type': record.sensor_type,
                'value': record.value,
                'unit': record.unit,
                'sensor_id': record.sensor_id,
                'confidence': record.confidence,
                'is_anomaly': record.is_anomaly
            })
        
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def _generate_features(
        self, 
        stope: Stope, 
        raw_data: pd.DataFrame,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate engineered features from raw data"""
        
        if raw_data.empty:
            return pd.DataFrame()
        
        # Create time index with regular intervals
        time_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq='1H'  # 1-hour intervals
        )
        
        features_list = []
        
        for timestamp in time_index:
            feature_vector = {}
            feature_vector['timestamp'] = timestamp
            
            # Get data window around this timestamp
            window_start = timestamp - timedelta(hours=1)
            window_end = timestamp
            window_data = raw_data[
                (raw_data.index >= window_start) & 
                (raw_data.index <= window_end)
            ]
            
            # Generate features by sensor type
            for sensor_type in self.config.enabled_sensor_types:
                sensor_data = window_data[window_data['sensor_type'] == sensor_type]
                
                if not sensor_data.empty:
                    # Raw features
                    if 'raw' in self.config.enabled_feature_types:
                        feature_vector[f'{sensor_type}_latest'] = sensor_data['value'].iloc[-1]
                        feature_vector[f'{sensor_type}_confidence'] = sensor_data['confidence'].mean()
                    
                    # Statistical features
                    if 'statistical' in self.config.enabled_feature_types:
                        for agg_func in self.config.aggregation_functions:
                            if agg_func == 'mean':
                                feature_vector[f'{sensor_type}_mean'] = sensor_data['value'].mean()
                            elif agg_func == 'std':
                                feature_vector[f'{sensor_type}_std'] = sensor_data['value'].std()
                            elif agg_func == 'min':
                                feature_vector[f'{sensor_type}_min'] = sensor_data['value'].min()
                            elif agg_func == 'max':
                                feature_vector[f'{sensor_type}_max'] = sensor_data['value'].max()
                            elif agg_func == 'range':
                                feature_vector[f'{sensor_type}_range'] = (
                                    sensor_data['value'].max() - sensor_data['value'].min()
                                )
                            elif agg_func == 'percentile_25':
                                feature_vector[f'{sensor_type}_p25'] = sensor_data['value'].quantile(0.25)
                            elif agg_func == 'percentile_75':
                                feature_vector[f'{sensor_type}_p75'] = sensor_data['value'].quantile(0.75)
                else:
                    # Fill missing sensor data with appropriate defaults
                    for agg_func in self.config.aggregation_functions:
                        feature_vector[f'{sensor_type}_{agg_func}'] = 0.0
                    if 'raw' in self.config.enabled_feature_types:
                        feature_vector[f'{sensor_type}_latest'] = 0.0
                        feature_vector[f'{sensor_type}_confidence'] = 0.0
            
            # Operational event features
            if 'operational' in self.config.enabled_feature_types:
                event_features = self._generate_event_features(stope, timestamp)
                feature_vector.update(event_features)
            
            features_list.append(feature_vector)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        features_df.set_index('timestamp', inplace=True)
        
        # Handle missing values
        features_df.fillna(0.0, inplace=True)
        
        return features_df
    
    def _generate_event_features(self, stope: Stope, timestamp: datetime) -> Dict[str, float]:
        """Generate features based on operational events"""
        event_features = {}
        
        # Look for events in the past 24 hours
        lookback_time = timestamp - timedelta(hours=24)
        
        events = OperationalEvent.objects.filter(
            stope=stope,
            timestamp__gte=lookback_time,
            timestamp__lte=timestamp
        )
        
        # Event type counts
        event_type_counts = {}
        for event in events:
            event_type = event.event_type
            if event_type not in event_type_counts:
                event_type_counts[event_type] = 0
            event_type_counts[event_type] += 1
        
        # Add event counts as features
        event_types = [
            'blasting', 'heavy_equipment', 'excavation', 'drilling',
            'water_exposure', 'support_installation'
        ]
        
        for event_type in event_types:
            event_features[f'event_{event_type}_count_24h'] = event_type_counts.get(event_type, 0)
        
        # Recent event impact (with decay)
        recent_impact = 0.0
        for event in events:
            hours_ago = (timestamp - event.timestamp).total_seconds() / 3600
            decay_factor = self.config.event_decay_factor ** hours_ago
            
            # Get severity multiplier
            severity_multiplier = {
                'low': 1.0,
                'medium': 2.0,
                'high': 3.0,
                'critical': 5.0
            }.get(event.severity, 1.0)
            
            recent_impact += severity_multiplier * decay_factor
        
        event_features['recent_event_impact'] = recent_impact
        
        return event_features
    
    def _extract_targets(
        self, 
        stope: Stope, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame:
        """Extract impact scores as target values"""
        
        # Get impact scores for the time period
        impact_scores = ImpactScore.objects.filter(
            stope=stope,
            timestamp__gte=start_time,
            timestamp__lte=end_time
        ).order_by('timestamp')
        
        if not impact_scores.exists():
            # Create empty DataFrame with proper time index
            time_index = pd.date_range(start=start_time, end=end_time, freq='1H')
            return pd.DataFrame(
                index=time_index,
                data={'impact_score': 0.0, 'risk_level': 'stable'}
            )
        
        # Convert to DataFrame
        target_data = []
        for score in impact_scores:
            target_data.append({
                'timestamp': score.timestamp,
                'impact_score': score.current_score,
                'risk_level': score.risk_level
            })
        
        targets_df = pd.DataFrame(target_data)
        targets_df['timestamp'] = pd.to_datetime(targets_df['timestamp'])
        targets_df.set_index('timestamp', inplace=True)
        
        # Resample to hourly intervals
        targets_df = targets_df.resample('1H').last().fillna(method='ffill')
        
        return targets_df
    
    def _create_sequence_object(
        self,
        stope: Stope,
        features_df: pd.DataFrame,
        targets_df: Optional[pd.DataFrame],
        sequence_type: str,
        start_idx: int
    ) -> Optional[TimeSeriesData]:
        """Create TimeSeriesData object from processed data"""
        
        if features_df.empty:
            return None
        
        try:
            # Prepare feature data
            feature_names = list(features_df.columns)
            raw_features = features_df.values.tolist()
            
            # Normalize features
            normalized_features = self.scaler.fit_transform(features_df.values).tolist()
            
            # Prepare target data
            impact_scores = []
            risk_levels = []
            
            if targets_df is not None and not targets_df.empty:
                # Align targets with features by index
                aligned_targets = targets_df.reindex(features_df.index, method='ffill')
                impact_scores = aligned_targets['impact_score'].fillna(0.0).tolist()
                risk_levels = aligned_targets['risk_level'].fillna('stable').tolist()
            else:
                # Fill with default values
                impact_scores = [0.0] * len(features_df)
                risk_levels = ['stable'] * len(features_df)
            
            # Generate unique sequence ID
            sequence_id = f"{stope.stope_name}_{sequence_type}_{start_idx}_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create TimeSeriesData object
            time_series = TimeSeriesData(
                stope=stope,
                sequence_id=sequence_id,
                sequence_type=sequence_type,
                start_timestamp=features_df.index[0],
                end_timestamp=features_df.index[-1],
                sequence_length=len(features_df),
                feature_set=self.config.config_name,
                feature_count=len(feature_names),
                raw_features=raw_features,
                normalized_features=normalized_features,
                feature_names=feature_names,
                impact_score_sequence=impact_scores,
                risk_level_sequence=risk_levels,
                preprocessing_version=self.config.version
            )
            
            return time_series
            
        except Exception as e:
            logger.error(f"Error creating sequence object: {e}")
            return None
    
    def process_stope_data(
        self,
        stope: Stope,
        start_time: datetime,
        end_time: datetime,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, List[TimeSeriesData]]:
        """
        Process complete stope data and split into train/validation/test sets
        
        Returns:
            Dictionary with 'train', 'validation', and 'test' sequences
        """
        
        # Calculate split timestamps
        total_duration = end_time - start_time
        train_end = start_time + timedelta(seconds=total_duration.total_seconds() * train_split)
        val_end = train_end + timedelta(seconds=total_duration.total_seconds() * val_split)
        
        # Create sequences for each split
        train_sequences = self.create_sequences(
            stope, start_time, train_end, sequence_type='training'
        )
        
        val_sequences = self.create_sequences(
            stope, train_end, val_end, sequence_type='validation'
        )
        
        test_sequences = self.create_sequences(
            stope, val_end, end_time, sequence_type='test'
        )
        
        # Save all sequences to database
        with transaction.atomic():
            for seq_list in [train_sequences, val_sequences, test_sequences]:
                for sequence in seq_list:
                    sequence.save()
                    # Create quality metrics
                    self._create_quality_metrics(sequence)
        
        return {
            'train': train_sequences,
            'validation': val_sequences,
            'test': test_sequences
        }
    
    def _create_quality_metrics(self, time_series: TimeSeriesData):
        """Create quality metrics for a time series sequence"""
        
        # Calculate basic quality scores
        completeness = 1.0 - (time_series.missing_data_percentage / 100.0)
        consistency = 1.0 - min(1.0, time_series.anomaly_count / time_series.sequence_length)
        validity = 1.0 if time_series.is_valid else 0.5
        temporal = 1.0  # Assume good temporal resolution for now
        
        quality_metrics = DataQualityMetrics(
            time_series_data=time_series,
            completeness_score=completeness,
            consistency_score=consistency,
            validity_score=validity,
            temporal_resolution_score=temporal,
            outlier_count=time_series.anomaly_count,
            outlier_percentage=(time_series.anomaly_count / time_series.sequence_length) * 100,
            invalid_readings_count=0,  # Will be calculated in detailed analysis
            timestamp_irregularities=0,
            analysis_version=self.config.version
        )
        
        quality_metrics.calculate_overall_quality()
        quality_metrics.save()


class DataValidator:
    """Comprehensive data validation for LSTM training data"""
    
    @staticmethod
    def validate_time_series_data(time_series: TimeSeriesData) -> Dict[str, Any]:
        """
        Comprehensive validation of time series data
        
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'quality_score': 0.0
        }
        
        # Check basic structure
        if not time_series.normalized_features:
            validation_results['errors'].append("No normalized features found")
            validation_results['is_valid'] = False
        
        # Check sequence length consistency
        if len(time_series.normalized_features) != time_series.sequence_length:
            validation_results['errors'].append(
                f"Feature length {len(time_series.normalized_features)} != sequence_length {time_series.sequence_length}"
            )
            validation_results['is_valid'] = False
        
        # Check for NaN or infinite values
        try:
            features_array = np.array(time_series.normalized_features)
            if np.any(np.isnan(features_array)):
                validation_results['errors'].append("Features contain NaN values")
                validation_results['is_valid'] = False
            
            if np.any(np.isinf(features_array)):
                validation_results['errors'].append("Features contain infinite values")
                validation_results['is_valid'] = False
        except Exception as e:
            validation_results['errors'].append(f"Error checking feature values: {e}")
            validation_results['is_valid'] = False
        
        # Check target consistency
        if len(time_series.impact_score_sequence) != time_series.sequence_length:
            validation_results['warnings'].append("Target sequence length mismatch")
        
        # Calculate quality score
        if validation_results['is_valid']:
            validation_results['quality_score'] = time_series.data_quality_score
        
        return validation_results


# ===== DATA PREPROCESSING PIPELINE COMPLETE =====
# Comprehensive preprocessing, feature engineering, and validation for LSTM training
