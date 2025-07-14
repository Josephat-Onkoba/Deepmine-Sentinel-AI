#!/usr/bin/env python3
"""
Test Dataset Generator for Mine Stope Stability Prediction
Generates realistic test data with similar patterns to training data
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import random

class TestDatasetGenerator:
    """Generate realistic test dataset for mine stope stability prediction"""
    
    def __init__(self, seed: int = 2025):
        """Initialize with seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        
        # Test dataset configuration - smaller than training
        self.num_stopes = 15  # Smaller test set
        self.days_per_stope = 60  # 2 months of data per stope
        self.start_date = datetime(2024, 4, 1)  # Different time period from training
        
        # Feature ranges based on mining engineering knowledge
        self.feature_ranges = {
            'rqd': (25, 95),  # Rock Quality Designation
            'hr': (2, 15),    # Hazard rating
            'depth': (150, 1000),  # Depth in meters
            'dip': (35, 85),  # Dip angle in degrees
            'undercut_wdt': (3, 8),  # Undercut width in meters
            'support_density': (0.1, 1.0),  # Support density ratio
            'vibration_velocity': (0.1, 3.0),  # mm/s
            'deformation_rate': (0.01, 1.2),  # mm/day
            'stress': (15, 80),  # MPa
            'temperature': (18, 35),  # Celsius
            'humidity': (40, 95)  # Percentage
        }
        
        # Categorical options
        self.rock_types = ['Granite', 'Limestone', 'Quartzite', 'Basalt', 'Sandstone', 'Schist', 'Gneiss']
        self.support_types = ['Rock Bolts', 'Mesh', 'Shotcrete', 'Cable Bolts', 'Steel Sets']
        self.directions = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
        
        # Stability distribution - similar to training but different exact numbers
        # Training: 37 stable, 13 unstable -> Test: 11 stable, 4 unstable
        self.stable_count = 11
        self.unstable_count = 4
        
    def generate_static_features(self) -> pd.DataFrame:
        """Generate static features for test stopes"""
        print("📊 Generating test static features...")
        
        static_data = []
        stability_labels = [1] * self.stable_count + [0] * self.unstable_count
        np.random.shuffle(stability_labels)
        
        for i in range(self.num_stopes):
            stope_id = f"Test_Stope_{i+1:03d}"
            stability = stability_labels[i]
            
            # Generate features with realistic correlations
            features = self._generate_correlated_features(stability)
            
            static_data.append({
                'stope_name': stope_id,
                'rqd': features['rqd'],
                'hr': features['hr'],
                'depth': features['depth'],
                'dip': features['dip'],
                'direction': np.random.choice(self.directions),
                'undercut_wdt': features['undercut_wdt'],
                'rock_type': features['rock_type'],
                'support_type': features['support_type'],
                'support_density': features['support_density'],
                'support_installed': features['support_installed'],
                'stability': stability
            })
        
        df = pd.DataFrame(static_data)
        
        # Print statistics
        stable_count = sum(df['stability'])
        unstable_count = len(df) - stable_count
        print(f"   Test static features: {len(df)} stopes")
        print(f"   Class distribution: Stable={stable_count}, Unstable={unstable_count}")
        print(f"   Balance ratio: {stable_count/len(df):.2f} stable")
        
        return df
    
    def _generate_correlated_features(self, stability: int) -> Dict:
        """Generate correlated features based on stability"""
        
        if stability == 1:  # Stable stope
            # Higher RQD, lower hazard rating, better support
            rqd = np.random.normal(75, 12)
            hr = np.random.gamma(2, 1.5) + 2
            depth = np.random.exponential(200) + 150
            support_installed = np.random.choice([0, 1], p=[0.15, 0.85])  # 85% have support
            support_density = np.random.beta(4, 2) * 0.9 + 0.1
            
        else:  # Unstable stope
            # Lower RQD, higher hazard rating, potentially poor support
            rqd = np.random.normal(45, 15)
            hr = np.random.gamma(3, 2) + 5
            depth = np.random.exponential(300) + 200
            support_installed = np.random.choice([0, 1], p=[0.4, 0.6])  # 60% have support
            support_density = np.random.beta(2, 4) * 0.8 + 0.1
        
        # Clip values to realistic ranges
        rqd = np.clip(rqd, *self.feature_ranges['rqd'])
        hr = np.clip(hr, *self.feature_ranges['hr'])
        depth = np.clip(depth, *self.feature_ranges['depth'])
        support_density = np.clip(support_density, *self.feature_ranges['support_density'])
        
        # Other features
        dip = np.random.uniform(*self.feature_ranges['dip'])
        undercut_wdt = np.random.uniform(*self.feature_ranges['undercut_wdt'])
        
        # Select rock and support types with some bias based on stability
        if stability == 1:
            rock_type = np.random.choice(self.rock_types, 
                                       p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.15, 0.1])  # Favor granite, quartzite
            support_type = np.random.choice(self.support_types,
                                          p=[0.3, 0.15, 0.25, 0.2, 0.1])  # Favor rock bolts, shotcrete
        else:
            rock_type = np.random.choice(self.rock_types,
                                       p=[0.1, 0.2, 0.1, 0.15, 0.25, 0.1, 0.1])  # Favor softer rocks
            support_type = np.random.choice(self.support_types,
                                          p=[0.15, 0.3, 0.15, 0.1, 0.3])  # More mesh, steel sets
        
        return {
            'rqd': round(rqd, 2),
            'hr': round(hr, 2),
            'depth': round(depth, 2),
            'dip': round(dip, 2),
            'undercut_wdt': round(undercut_wdt, 2),
            'support_density': round(support_density, 2),
            'support_installed': support_installed,
            'rock_type': rock_type,
            'support_type': support_type
        }
    
    def generate_timeseries_data(self, static_df: pd.DataFrame) -> pd.DataFrame:
        """Generate timeseries data for test stopes"""
        print("📈 Generating test timeseries data...")
        
        timeseries_data = []
        
        for _, stope in static_df.iterrows():
            stope_name = stope['stope_name']
            stability = stope['stability']
            
            # Generate time series for this stope
            dates = [self.start_date + timedelta(days=i) for i in range(self.days_per_stope)]
            
            # Initialize base parameters based on stope characteristics
            base_params = self._get_base_timeseries_params(stope)
            
            for i, date in enumerate(dates):
                # Generate correlated sensor readings
                readings = self._generate_sensor_readings(base_params, stability, i, self.days_per_stope)
                
                timeseries_data.append({
                    'stope_name': stope_name,
                    'timestamp': date.strftime('%Y-%m-%d'),
                    'vibration_velocity': readings['vibration_velocity'],
                    'deformation_rate': readings['deformation_rate'],
                    'stress': readings['stress'],
                    'temperature': readings['temperature'],
                    'humidity': readings['humidity']
                })
        
        df = pd.DataFrame(timeseries_data)
        
        print(f"   Test timeseries data: {len(df)} records")
        print(f"   Time range: {self.start_date.strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        
        return df
    
    def _get_base_timeseries_params(self, stope: pd.Series) -> Dict:
        """Get base parameters for timeseries generation based on stope characteristics"""
        
        # Correlate with static features
        depth_factor = (stope['depth'] - 150) / (1000 - 150)  # 0-1 normalized
        rqd_factor = stope['rqd'] / 100
        hr_factor = stope['hr'] / 15
        
        return {
            'base_vibration': 0.5 + hr_factor * 0.8 + depth_factor * 0.3,
            'base_deformation': 0.1 + hr_factor * 0.4 + (1 - rqd_factor) * 0.3,
            'base_stress': 25 + depth_factor * 30 + hr_factor * 15,
            'base_temperature': 20 + depth_factor * 12,  # Temperature increases with depth
            'base_humidity': 50 + np.random.normal(0, 10)
        }
    
    def _generate_sensor_readings(self, base_params: Dict, stability: int, 
                                day_index: int, total_days: int) -> Dict:
        """Generate realistic sensor readings with temporal patterns"""
        
        # Add temporal trends and noise
        time_factor = day_index / total_days
        
        # Instability tends to increase over time for unstable stopes
        if stability == 0:  # Unstable
            instability_trend = time_factor * 0.5
            noise_factor = 1.3  # More noise for unstable stopes
        else:  # Stable
            instability_trend = 0
            noise_factor = 0.8  # Less noise for stable stopes
        
        # Generate readings with correlations and realistic noise
        vibration = (base_params['base_vibration'] + instability_trend + 
                    np.random.normal(0, 0.3 * noise_factor))
        
        deformation = (base_params['base_deformation'] + instability_trend * 0.3 + 
                      np.random.normal(0, 0.15 * noise_factor))
        
        stress = (base_params['base_stress'] + instability_trend * 10 + 
                 vibration * 5 + np.random.normal(0, 5 * noise_factor))
        
        # Temperature has seasonal variation and depth correlation
        seasonal_temp = 3 * np.sin(2 * np.pi * day_index / 365)
        temperature = (base_params['base_temperature'] + seasonal_temp + 
                      np.random.normal(0, 1.5))
        
        # Humidity has inverse correlation with temperature and some randomness
        humidity = (base_params['base_humidity'] - (temperature - 22) * 1.5 + 
                   np.random.normal(0, 8))
        
        # Clip to realistic ranges
        return {
            'vibration_velocity': round(np.clip(vibration, *self.feature_ranges['vibration_velocity']), 2),
            'deformation_rate': round(np.clip(deformation, *self.feature_ranges['deformation_rate']), 2),
            'stress': round(np.clip(stress, *self.feature_ranges['stress']), 2),
            'temperature': round(np.clip(temperature, *self.feature_ranges['temperature']), 2),
            'humidity': round(np.clip(humidity, *self.feature_ranges['humidity']), 2)
        }
    
    def generate_dataset(self, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete test dataset"""
        
        print("🧪 Generating test dataset for mine stope stability prediction...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate datasets
        static_df = self.generate_static_features()
        timeseries_df = self.generate_timeseries_data(static_df)
        
        # Save datasets
        static_path = os.path.join(output_dir, 'test_stope_static_features.csv')
        timeseries_path = os.path.join(output_dir, 'test_stope_timeseries_data.csv')
        
        print("\n💾 Saving test dataset...")
        static_df.to_csv(static_path, index=False)
        timeseries_df.to_csv(timeseries_path, index=False)
        
        print(f"✅ Test static features saved to: {static_path}")
        print(f"✅ Test timeseries data saved to: {timeseries_path}")
        
        # Print final statistics
        self._print_dataset_statistics(static_df, timeseries_df)
        
        return static_df, timeseries_df
    
    def _print_dataset_statistics(self, static_df: pd.DataFrame, timeseries_df: pd.DataFrame):
        """Print comprehensive dataset statistics"""
        
        print("\n📊 Test Dataset Statistics:")
        print("\nStatic Features:")
        print(f"  Rock types: {', '.join(static_df['rock_type'].unique())}")
        print(f"  Support types: {', '.join(static_df['support_type'].unique())}")
        print(f"  RQD range: {static_df['rqd'].min():.1f} - {static_df['rqd'].max():.1f}")
        print(f"  Depth range: {static_df['depth'].min():.1f} - {static_df['depth'].max():.1f} m")
        print(f"  Support installed: {static_df['support_installed'].sum()}/{len(static_df)} stopes")
        
        print("\nTimeseries Features:")
        for col in ['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']:
            print(f"  {col}: {timeseries_df[col].min():.2f} - {timeseries_df[col].max():.2f}")
        
        print(f"\n🎯 Test dataset generation completed!")
        print(f"   Total test stopes: {len(static_df)}")
        print(f"   Total test timeseries records: {len(timeseries_df)}")
        print(f"   Average days per stope: {len(timeseries_df) / len(static_df):.1f}")


def main():
    """Main function to generate test dataset"""
    
    # Initialize generator
    generator = TestDatasetGenerator(seed=2025)
    
    # Generate test dataset
    output_dir = "/home/jose/Desktop/Deepmine-Sentinel-AI/deepmine_sentinel_ai/data/test_data"
    static_df, timeseries_df = generator.generate_dataset(output_dir)
    
    print(f"\n🧪 Test dataset ready for evaluation!")
    print(f"📁 Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
