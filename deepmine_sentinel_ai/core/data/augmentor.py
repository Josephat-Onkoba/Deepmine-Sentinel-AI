"""
Data Augmentation for Rare Event Scenarios

Implements sophisticated data augmentation techniques to handle class imbalance
and generate synthetic training examples for rare but critical mining scenarios:

- SMOTE-based sequence augmentation for minority classes
- Gaussian noise injection with domain constraints
- Time warping for temporal pattern variations
- Event pattern synthesis for critical scenarios
- Geological condition variations

Based on PROJECT_FINAL_REPORT.md requirements for handling rare critical events
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
import random

from core.data.preprocessor import SequenceData

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation operations"""
    target_samples_per_class: int = 1000  # Target number of samples per class
    smote_k_neighbors: int = 5             # K-neighbors for SMOTE
    noise_factor: float = 0.05             # Gaussian noise factor (5% of std)
    time_warp_factor: float = 0.1          # Time warping strength (10%)
    critical_oversample_factor: float = 3.0  # Extra oversampling for critical class
    preserve_geological_constraints: bool = True  # Maintain realistic geological bounds
    max_augmentation_ratio: float = 5.0    # Maximum ratio of augmented to original data
    

class DataAugmentor:
    """
    Advanced data augmentation for LSTM training with mining domain constraints
    
    Focuses on generating realistic synthetic examples for rare but critical
    mining scenarios while preserving domain-specific constraints and relationships
    """
    
    def __init__(self, config: AugmentationConfig = None):
        """Initialize augmentor with configuration"""
        self.config = config or AugmentationConfig()
        
        # Geological constraints for realistic augmentation
        self.geological_bounds = {
            'rqd': (60.0, 95.0),           # Rock Quality Designation bounds
            'depth': (50.0, 800.0),       # Depth bounds in meters
            'dip': (15.0, 85.0),          # Dip angle bounds in degrees
            'hr': (1.5, 4.0),             # Height/width ratio bounds
            'undercut_width': (10.0, 40.0), # Undercut width bounds
            'support_density': (0.3, 1.5),  # Support density bounds
            'impact_score': (0.0, 15.0),    # Impact score bounds
        }
        
        # Event intensity constraints
        self.event_bounds = {
            'blasting': (0.0, 10.0),
            'heavy_equipment': (0.0, 5.0),
            'water_exposure': (0.0, 8.0),
            'drilling': (0.0, 3.0),
            'mucking': (0.0, 2.0),
            'support_installation': (0.0, 1.0)
        }
        
        logger.info("Data Augmentor initialized")
        logger.info(f"Target samples per class: {self.config.target_samples_per_class}")
        logger.info(f"Critical oversample factor: {self.config.critical_oversample_factor}")
    
    def analyze_class_distribution(self, sequence_data: SequenceData) -> Dict:
        """
        Analyze class distribution and determine augmentation requirements
        """
        class_counts = np.bincount(sequence_data.risk_labels)
        total_samples = len(sequence_data.risk_labels)
        
        analysis = {
            'class_counts': class_counts.tolist(),
            'class_percentages': (class_counts / total_samples * 100).tolist(),
            'total_samples': total_samples,
            'imbalance_ratio': np.max(class_counts) / np.min(class_counts[class_counts > 0]),
            'needs_augmentation': {}
        }
        
        # Determine augmentation needs for each class
        for class_idx, count in enumerate(class_counts):
            if count == 0:
                continue
                
            target_count = self.config.target_samples_per_class
            if class_idx == 3:  # Critical class gets extra oversampling
                target_count = int(target_count * self.config.critical_oversample_factor)
            
            if count < target_count:
                needed_samples = target_count - count
                max_allowed = int(count * self.config.max_augmentation_ratio)
                actual_target = min(needed_samples, max_allowed)
                
                analysis['needs_augmentation'][class_idx] = {
                    'current_count': count,
                    'target_count': target_count,
                    'samples_needed': needed_samples,
                    'samples_to_generate': actual_target,
                    'augmentation_ratio': actual_target / count if count > 0 else 0
                }
        
        logger.info(f"Class distribution analysis:")
        logger.info(f"  Current counts: {class_counts}")
        logger.info(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
        logger.info(f"  Classes needing augmentation: {list(analysis['needs_augmentation'].keys())}")
        
        return analysis
    
    def smote_sequence_augmentation(self, sequences: np.ndarray, 
                                  static_features: np.ndarray,
                                  class_labels: np.ndarray,
                                  target_class: int,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        SMOTE-based augmentation for sequence data
        
        Generates synthetic sequences by interpolating between similar sequences
        of the same class, maintaining temporal structure
        """
        # Find samples of target class
        class_mask = class_labels == target_class
        class_sequences = sequences[class_mask]
        class_static = static_features[class_mask]
        
        if len(class_sequences) < self.config.smote_k_neighbors:
            logger.warning(f"Not enough samples for SMOTE in class {target_class}")
            return np.array([]), np.array([])
        
        # Flatten sequences for nearest neighbor search
        # Use mean pooling to create feature vector for each sequence
        sequence_vectors = np.mean(class_sequences, axis=1)  # (n_samples, n_features)
        
        # Combine with static features for similarity calculation
        # Ensure dimensions match before concatenation
        if sequence_vectors.shape[0] != class_static.shape[0]:
            logger.warning(f"Dimension mismatch: sequences {sequence_vectors.shape[0]} vs static {class_static.shape[0]}")
            # Take minimum to align arrays
            min_samples = min(sequence_vectors.shape[0], class_static.shape[0])
            sequence_vectors = sequence_vectors[:min_samples]
            class_static = class_static[:min_samples]
        
        combined_vectors = np.concatenate([sequence_vectors, class_static], axis=1)
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(self.config.smote_k_neighbors, len(combined_vectors)))
        nbrs.fit(combined_vectors)
        
        synthetic_sequences = []
        synthetic_static = []
        
        for _ in range(n_samples):
            # Randomly select a sample from the minority class
            idx = np.random.randint(0, len(class_sequences))
            sample_vector = combined_vectors[idx:idx+1]
            
            # Find its k-nearest neighbors
            _, neighbor_indices = nbrs.kneighbors(sample_vector)
            neighbor_indices = neighbor_indices[0]
            
            # Randomly select one neighbor (excluding the sample itself if present)
            neighbor_idx = np.random.choice([n for n in neighbor_indices if n != idx])
            
            # Generate synthetic sample by linear interpolation
            alpha = np.random.random()  # Random interpolation factor
            
            # Interpolate sequence data
            base_sequence = class_sequences[idx]
            neighbor_sequence = class_sequences[neighbor_idx]
            synthetic_sequence = base_sequence + alpha * (neighbor_sequence - base_sequence)
            
            # Interpolate static features
            base_static = class_static[idx]
            neighbor_static = class_static[neighbor_idx]
            synthetic_static_sample = base_static + alpha * (neighbor_static - base_static)
            
            # Apply domain constraints to synthetic data
            synthetic_sequence = self._apply_sequence_constraints(synthetic_sequence)
            synthetic_static_sample = self._apply_static_constraints(synthetic_static_sample)
            
            synthetic_sequences.append(synthetic_sequence)
            synthetic_static.append(synthetic_static_sample)
        
        return np.array(synthetic_sequences), np.array(synthetic_static)
    
    def gaussian_noise_augmentation(self, sequences: np.ndarray,
                                  static_features: np.ndarray,
                                  class_labels: np.ndarray,
                                  target_class: int,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gaussian noise-based augmentation with domain constraints
        
        Adds controlled Gaussian noise to existing samples while maintaining
        realistic operational and geological constraints
        """
        class_mask = class_labels == target_class
        class_sequences = sequences[class_mask]
        class_static = static_features[class_mask]
        
        if len(class_sequences) == 0:
            return np.array([]), np.array([])
        
        synthetic_sequences = []
        synthetic_static = []
        
        # Calculate noise parameters based on data statistics
        sequence_std = np.std(class_sequences, axis=0)
        static_std = np.std(class_static, axis=0)
        
        for _ in range(n_samples):
            # Randomly select a base sample
            base_idx = np.random.randint(0, len(class_sequences))
            base_sequence = class_sequences[base_idx].copy()
            base_static = class_static[base_idx].copy()
            
            # Add Gaussian noise to sequence data
            sequence_noise = np.random.normal(0, self.config.noise_factor * sequence_std, 
                                            base_sequence.shape)
            noisy_sequence = base_sequence + sequence_noise
            
            # Add Gaussian noise to static features
            static_noise = np.random.normal(0, self.config.noise_factor * static_std,
                                          base_static.shape)
            noisy_static = base_static + static_noise
            
            # Apply domain constraints
            noisy_sequence = self._apply_sequence_constraints(noisy_sequence)
            noisy_static = self._apply_static_constraints(noisy_static)
            
            synthetic_sequences.append(noisy_sequence)
            synthetic_static.append(noisy_static)
        
        return np.array(synthetic_sequences), np.array(synthetic_static)
    
    def time_warping_augmentation(self, sequences: np.ndarray,
                                class_labels: np.ndarray,
                                target_class: int,
                                n_samples: int) -> np.ndarray:
        """
        Time warping augmentation for temporal pattern variation
        
        Applies non-linear time warping to create variations in temporal patterns
        while preserving overall sequence characteristics
        """
        class_mask = class_labels == target_class
        class_sequences = sequences[class_mask]
        
        if len(class_sequences) == 0:
            return np.array([])
        
        synthetic_sequences = []
        sequence_length = class_sequences.shape[1]
        
        for _ in range(n_samples):
            # Randomly select a base sequence
            base_idx = np.random.randint(0, len(class_sequences))
            base_sequence = class_sequences[base_idx]
            
            # Create warping function
            # Generate random control points for warping
            n_control_points = max(3, sequence_length // 20)  # At least 3 control points
            control_indices = np.linspace(0, sequence_length - 1, n_control_points)
            
            # Add random perturbations to control points
            warp_strength = self.config.time_warp_factor * sequence_length
            warp_offsets = np.random.normal(0, warp_strength, n_control_points)
            warped_indices = np.clip(control_indices + warp_offsets, 0, sequence_length - 1)
            
            # Ensure monotonic ordering
            warped_indices = np.sort(warped_indices)
            
            # Create interpolation function
            original_indices = np.arange(sequence_length)
            
            synthetic_sequence = np.zeros_like(base_sequence)
            
            # Apply warping to each feature dimension
            for feature_idx in range(base_sequence.shape[1]):
                feature_values = base_sequence[:, feature_idx]
                
                # Create interpolation function
                interp_func = interp1d(control_indices, 
                                     feature_values[control_indices.astype(int)], 
                                     kind='linear', 
                                     bounds_error=False, 
                                     fill_value='extrapolate')
                
                # Apply warping
                warped_feature = interp_func(warped_indices)
                
                # Interpolate back to original timeline
                reinterp_func = interp1d(warped_indices, warped_feature, 
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value='extrapolate')
                
                synthetic_sequence[:, feature_idx] = reinterp_func(original_indices)
            
            # Apply constraints
            synthetic_sequence = self._apply_sequence_constraints(synthetic_sequence)
            synthetic_sequences.append(synthetic_sequence)
        
        return np.array(synthetic_sequences)
    
    def critical_scenario_synthesis(self, sequences: np.ndarray,
                                  static_features: np.ndarray,
                                  class_labels: np.ndarray,
                                  n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize critical scenarios based on domain knowledge
        
        Creates realistic critical mining scenarios by combining:
        - High-risk geological conditions
        - Intensive operational patterns
        - Realistic progression to critical states
        """
        synthetic_sequences = []
        synthetic_static = []
        
        # Get examples of different risk levels for pattern mixing
        stable_mask = class_labels == 0
        elevated_mask = class_labels == 1
        high_risk_mask = class_labels == 2
        critical_mask = class_labels == 3
        
        for _ in range(n_samples):
            # Create high-risk static conditions
            synthetic_static_features = self._generate_high_risk_static_features()
            
            # Create critical operational sequence
            if np.any(critical_mask):
                # Use existing critical pattern as base
                critical_sequences = sequences[critical_mask]
                base_critical = critical_sequences[np.random.randint(0, len(critical_sequences))]
                synthetic_sequence = self._intensify_operational_pattern(base_critical)
            else:
                # Create from scratch using domain knowledge
                synthetic_sequence = self._create_critical_progression_sequence(sequences.shape[1])
            
            # Apply constraints
            synthetic_sequence = self._apply_sequence_constraints(synthetic_sequence)
            synthetic_static_features = self._apply_static_constraints(synthetic_static_features)
            
            synthetic_sequences.append(synthetic_sequence)
            synthetic_static.append(synthetic_static_features)
        
        return np.array(synthetic_sequences), np.array(synthetic_static)
    
    def _apply_sequence_constraints(self, sequence: np.ndarray) -> np.ndarray:
        """Apply domain constraints to sequence data"""
        constrained_sequence = sequence.copy()
        
        # Constrain impact scores (first column)
        constrained_sequence[:, 0] = np.clip(constrained_sequence[:, 0], 
                                           self.geological_bounds['impact_score'][0],
                                           self.geological_bounds['impact_score'][1])
        
        # Constrain event intensities (columns 3 onwards, assuming 6 event types)
        if sequence.shape[1] > 3:
            event_types = ['blasting', 'heavy_equipment', 'water_exposure', 'drilling', 'mucking', 'support_installation']
            for i, event_type in enumerate(event_types):
                if i + 3 < sequence.shape[1]:
                    bounds = self.event_bounds.get(event_type, (0.0, 10.0))
                    constrained_sequence[:, i + 3] = np.clip(constrained_sequence[:, i + 3],
                                                           bounds[0], bounds[1])
        
        # Ensure non-negative values for event counts/intensities
        if sequence.shape[1] > 3:
            constrained_sequence[:, 3:] = np.maximum(constrained_sequence[:, 3:], 0.0)
        
        return constrained_sequence
    
    def _apply_static_constraints(self, static_features: np.ndarray) -> np.ndarray:
        """Apply geological constraints to static features"""
        constrained_features = static_features.copy()
        
        # Feature order: RQD, depth, dip, rock_type_encoded, mining_method_encoded, 
        #                undercut_width, support_density, hr
        constraints = [
            self.geological_bounds['rqd'],
            self.geological_bounds['depth'],
            self.geological_bounds['dip'],
            (0, 4),  # Rock type encoding (0-4)
            (0, 3),  # Mining method encoding (0-3)
            self.geological_bounds['undercut_width'],
            self.geological_bounds['support_density'],
            self.geological_bounds['hr']
        ]
        
        for i, (min_val, max_val) in enumerate(constraints):
            if i < len(constrained_features):
                constrained_features[i] = np.clip(constrained_features[i], min_val, max_val)
        
        return constrained_features
    
    def _generate_high_risk_static_features(self) -> np.ndarray:
        """Generate static features representing high-risk geological conditions"""
        # Create conditions that predispose to instability
        
        # Lower RQD (poorer rock quality)
        rqd = np.random.uniform(60, 75)  # Lower end of acceptable range
        
        # Greater depth (higher stress)
        depth = np.random.uniform(400, 800)  # Deeper range
        
        # Steeper dip (less stable orientation)
        dip = np.random.uniform(60, 85)  # Steeper angles
        
        # Rock type biased toward less stable types
        rock_type = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])  # Favor schist, quartzite
        
        # Mining method appropriate for depth
        mining_method = 0 if depth > 300 else np.random.choice([0, 1])  # Sublevel stoping for deep
        
        # Larger undercut (more challenging)
        undercut_width = np.random.uniform(25, 40)
        
        # Lower support density (less reinforcement)
        support_density = np.random.uniform(0.3, 0.6)
        
        # Higher HR ratio (more challenging geometry)
        hr = np.random.uniform(3.0, 4.0)
        
        return np.array([rqd, depth, dip, rock_type, mining_method, undercut_width, support_density, hr])
    
    def _intensify_operational_pattern(self, base_sequence: np.ndarray) -> np.ndarray:
        """Intensify operational patterns to create more critical scenarios"""
        intensified = base_sequence.copy()
        
        # Increase overall impact scores
        intensified[:, 0] *= np.random.uniform(1.2, 1.8)  # 20-80% increase
        
        # Increase blasting intensity (if present)
        if base_sequence.shape[1] > 3:
            intensified[:, 3] *= np.random.uniform(1.5, 2.5)  # Increase blasting
            intensified[:, 4] *= np.random.uniform(1.2, 1.8)  # Increase equipment use
            intensified[:, 5] *= np.random.uniform(1.3, 2.0)  # Increase water exposure
        
        return intensified
    
    def _create_critical_progression_sequence(self, sequence_length: int) -> np.ndarray:
        """Create a realistic progression to critical state"""
        sequence = np.zeros((sequence_length, 9))  # Assuming 9 features
        
        # Create progressive impact increase
        base_impact = 1.0
        impact_growth_rate = 0.05  # 5% growth per hour on average
        
        for t in range(sequence_length):
            # Progressive impact increase with some randomness
            impact_increase = np.random.normal(impact_growth_rate, 0.02)
            base_impact *= (1 + impact_increase)
            
            # Add occasional spikes from intensive operations
            if np.random.random() < 0.1:  # 10% chance of operational spike
                base_impact += np.random.uniform(0.5, 2.0)
            
            sequence[t, 0] = base_impact  # Current impact
            sequence[t, 1] = base_impact * 0.9  # Previous impact (slightly lower)
            sequence[t, 2] = base_impact - sequence[t, 1]  # Impact change
            
            # Add realistic operational events
            # More frequent events as situation deteriorates
            event_intensity = min(1.0, base_impact / 10.0)
            
            sequence[t, 3] = np.random.poisson(event_intensity * 2)  # Blasting
            sequence[t, 4] = np.random.poisson(event_intensity * 3)  # Equipment
            sequence[t, 5] = np.random.poisson(event_intensity * 1)  # Water
            sequence[t, 6] = np.random.poisson(event_intensity * 2)  # Drilling
            sequence[t, 7] = np.random.poisson(event_intensity * 1)  # Mucking
            sequence[t, 8] = np.random.poisson(event_intensity * 0.5)  # Support (decreases as crisis develops)
        
        return sequence
    
    def augment_data(self, sequence_data: SequenceData) -> SequenceData:
        """
        Apply complete data augmentation pipeline
        
        Returns augmented SequenceData with balanced class distribution
        """
        logger.info("Starting data augmentation pipeline")
        
        # Analyze current distribution
        distribution_analysis = self.analyze_class_distribution(sequence_data)
        
        if not distribution_analysis['needs_augmentation']:
            logger.info("No augmentation needed - classes are already balanced")
            return sequence_data
        
        # Collect augmented samples
        augmented_sequences = [sequence_data.dynamic_features]
        augmented_static = [sequence_data.static_features]
        augmented_labels = [sequence_data.risk_labels]
        augmented_timestamps = [sequence_data.timestamps]
        augmented_stope_ids = [sequence_data.stope_ids]
        
        total_generated = 0
        
        for class_idx, aug_info in distribution_analysis['needs_augmentation'].items():
            samples_needed = aug_info['samples_to_generate']
            
            logger.info(f"Augmenting class {class_idx}: generating {samples_needed} samples")
            
            # Split augmentation across different techniques
            smote_samples = samples_needed // 3
            noise_samples = samples_needed // 3
            warp_samples = samples_needed // 3
            remaining_samples = samples_needed - (smote_samples + noise_samples + warp_samples)
            
            # SMOTE augmentation
            if smote_samples > 0:
                smote_seq, smote_static = self.smote_sequence_augmentation(
                    sequence_data.dynamic_features,
                    sequence_data.static_features,
                    sequence_data.risk_labels,
                    class_idx,
                    smote_samples
                )
                if len(smote_seq) > 0:
                    augmented_sequences.append(smote_seq)
                    augmented_static.append(smote_static)
                    augmented_labels.append(np.full(len(smote_seq), class_idx))
                    # Create synthetic timestamps
                    if sequence_data.timestamps.size > 0:
                        base_time = np.max(sequence_data.timestamps)
                    else:
                        base_time = pd.Timestamp.now()
                    synthetic_times = [base_time + pd.Timedelta(hours=i) for i in range(len(smote_seq))]
                    augmented_timestamps.append(np.array(synthetic_times))
                    augmented_stope_ids.append(np.full(len(smote_seq), -1))  # Mark as synthetic
            
            # Gaussian noise augmentation
            if noise_samples > 0:
                noise_seq, noise_static = self.gaussian_noise_augmentation(
                    sequence_data.dynamic_features,
                    sequence_data.static_features,
                    sequence_data.risk_labels,
                    class_idx,
                    noise_samples
                )
                if len(noise_seq) > 0:
                    augmented_sequences.append(noise_seq)
                    augmented_static.append(noise_static)
                    augmented_labels.append(np.full(len(noise_seq), class_idx))
                    if len(augmented_timestamps) > 1:
                        base_time = np.max(augmented_timestamps[-1]) if augmented_timestamps[-1].size > 0 else pd.Timestamp.now()
                    else:
                        base_time = np.max(sequence_data.timestamps) if sequence_data.timestamps.size > 0 else pd.Timestamp.now()
                    synthetic_times = [base_time + pd.Timedelta(hours=i) for i in range(len(noise_seq))]
                    augmented_timestamps.append(np.array(synthetic_times))
                    augmented_stope_ids.append(np.full(len(noise_seq), -2))  # Different marker
            
            # Time warping augmentation
            if warp_samples > 0:
                warp_seq = self.time_warping_augmentation(
                    sequence_data.dynamic_features,
                    sequence_data.risk_labels,
                    class_idx,
                    warp_samples
                )
                if len(warp_seq) > 0:
                    # Use static features from original samples of same class
                    class_mask = sequence_data.risk_labels == class_idx
                    class_static = sequence_data.static_features[class_mask]
                    
                    if len(class_static) > 0:
                        # Repeat static features to match number of warped sequences
                        n_repeats = (len(warp_seq) // len(class_static)) + 1
                        warp_static = np.tile(class_static, (n_repeats, 1))[:len(warp_seq)]
                        
                        # Ensure dimensions match
                        if warp_static.shape[0] != len(warp_seq):
                            logger.warning(f"Static feature dimension mismatch: {warp_static.shape[0]} vs {len(warp_seq)}")
                            # Create matching static features by replicating the first static feature
                            warp_static = np.tile(class_static[0], (len(warp_seq), 1))
                    else:
                        # Create zero static features if no class samples available
                        warp_static = np.zeros((len(warp_seq), sequence_data.static_features.shape[1]))
                    
                    augmented_sequences.append(warp_seq)
                    augmented_static.append(warp_static)
                    augmented_labels.append(np.full(len(warp_seq), class_idx))
                    if len(augmented_timestamps) > 1:
                        base_time = np.max(augmented_timestamps[-1]) if augmented_timestamps[-1].size > 0 else pd.Timestamp.now()
                    else:
                        base_time = np.max(sequence_data.timestamps) if sequence_data.timestamps.size > 0 else pd.Timestamp.now()
                    synthetic_times = [base_time + pd.Timedelta(hours=i) for i in range(len(warp_seq))]
                    augmented_timestamps.append(np.array(synthetic_times))
                    augmented_stope_ids.append(np.full(len(warp_seq), -3))  # Different marker
            
            # Critical scenario synthesis for critical class
            if class_idx == 3 and remaining_samples > 0:
                crit_seq, crit_static = self.critical_scenario_synthesis(
                    sequence_data.dynamic_features,
                    sequence_data.static_features,
                    sequence_data.risk_labels,
                    remaining_samples
                )
                if len(crit_seq) > 0:
                    augmented_sequences.append(crit_seq)
                    augmented_static.append(crit_static)
                    augmented_labels.append(np.full(len(crit_seq), class_idx))
                    if len(augmented_timestamps) > 1:
                        base_time = np.max(augmented_timestamps[-1]) if augmented_timestamps[-1].size > 0 else pd.Timestamp.now()
                    else:
                        base_time = np.max(sequence_data.timestamps) if sequence_data.timestamps.size > 0 else pd.Timestamp.now()
                    synthetic_times = [base_time + pd.Timedelta(hours=i) for i in range(len(crit_seq))]
                    augmented_timestamps.append(np.array(synthetic_times))
                    augmented_stope_ids.append(np.full(len(crit_seq), -4))  # Critical synthetic marker
            
            total_generated += samples_needed
        
        # Combine all data
        if len(augmented_sequences) > 1:
            # Ensure all sequences have the same feature dimension
            target_shape = augmented_sequences[0].shape
            consistent_sequences = []
            consistent_static = []
            consistent_labels = []
            consistent_timestamps = []
            consistent_stope_ids = []
            
            for i, seq_batch in enumerate(augmented_sequences):
                if seq_batch.size > 0:
                    # Check if dimensions match
                    if len(seq_batch.shape) == 3 and seq_batch.shape[1:] == target_shape[1:]:
                        consistent_sequences.append(seq_batch)
                        consistent_static.append(augmented_static[i])
                        consistent_labels.append(augmented_labels[i])
                        consistent_timestamps.append(augmented_timestamps[i])
                        consistent_stope_ids.append(augmented_stope_ids[i])
                    else:
                        logger.warning(f"Skipping batch {i} due to dimension mismatch: {seq_batch.shape} vs {target_shape}")
            
            if len(consistent_sequences) > 0:
                try:
                    final_sequences = np.vstack(consistent_sequences)
                    final_static = np.vstack(consistent_static)
                    final_labels = np.concatenate(consistent_labels)
                    
                    # Handle timestamps concatenation with error checking
                    try:
                        final_timestamps = np.concatenate(consistent_timestamps)
                    except ValueError as e:
                        logger.warning(f"Timestamp concatenation failed: {e}, using original timestamps")
                        final_timestamps = sequence_data.timestamps
                    
                    # Handle stope_ids concatenation with error checking
                    try:
                        final_stope_ids = np.concatenate(consistent_stope_ids)
                    except ValueError as e:
                        logger.warning(f"Stope ID concatenation failed: {e}, using original stope IDs")
                        final_stope_ids = sequence_data.stope_ids
                except ValueError as e:
                    logger.error(f"Array concatenation failed: {e}")
                    logger.warning("Returning original data due to concatenation errors")
                    final_sequences = sequence_data.dynamic_features
                    final_static = sequence_data.static_features
                    final_labels = sequence_data.risk_labels
                    final_timestamps = sequence_data.timestamps
                    final_stope_ids = sequence_data.stope_ids
            else:
                logger.warning("No consistent sequences found, returning original data")
                final_sequences = sequence_data.dynamic_features
                final_static = sequence_data.static_features
                final_labels = sequence_data.risk_labels
                final_timestamps = sequence_data.timestamps
                final_stope_ids = sequence_data.stope_ids
        else:
            # No augmentation was applied
            final_sequences = sequence_data.dynamic_features
            final_static = sequence_data.static_features
            final_labels = sequence_data.risk_labels
            final_timestamps = sequence_data.timestamps
            final_stope_ids = sequence_data.stope_ids
        
        # Update metadata
        augmented_metadata = sequence_data.metadata.copy()
        augmented_metadata.update({
            'augmentation_applied': True,
            'original_samples': len(sequence_data.risk_labels),
            'augmented_samples': len(final_labels),
            'generated_samples': total_generated,
            'augmentation_ratio': total_generated / len(sequence_data.risk_labels),
            'final_class_distribution': np.bincount(final_labels).tolist(),
            'augmentation_config': self.config
        })
        
        logger.info(f"Data augmentation completed:")
        logger.info(f"  Original samples: {len(sequence_data.risk_labels)}")
        logger.info(f"  Generated samples: {total_generated}")
        logger.info(f"  Final samples: {len(final_labels)}")
        logger.info(f"  Final class distribution: {np.bincount(final_labels)}")
        
        return SequenceData(
            static_features=final_static,
            dynamic_features=final_sequences,
            risk_labels=final_labels,
            timestamps=final_timestamps,
            stope_ids=final_stope_ids,
            metadata=augmented_metadata
        )
    
    def augment_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper method for backward compatibility with demo commands
        
        Converts numpy arrays to SequenceData, augments, and returns numpy arrays
        """
        # Create minimal SequenceData from numpy arrays
        dummy_static = np.zeros((X.shape[0], 1))  # Minimal static features
        dummy_timestamps = [[] for _ in range(X.shape[0])]
        
        sequence_data = SequenceData(
            dynamic_features=X,
            static_features=dummy_static,
            risk_labels=y,
            timestamps=dummy_timestamps,
            num_samples=X.shape[0],
            sequence_length=X.shape[1],
            feature_dim=X.shape[2]
        )
        
        # Apply augmentation
        augmented_data = self.augment_data(sequence_data)
        
        # Return as numpy arrays
        return augmented_data.dynamic_features, augmented_data.risk_labels
