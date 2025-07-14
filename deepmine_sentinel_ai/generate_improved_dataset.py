#!/usr/bin/env python3
"""
Improved Dataset Generator for Mine Stope Stability Prediction
Creates realistic, well-distributed training data with proper geological patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ImprovedDatasetGenerator:
    def __init__(self):
        # Define realistic geological parameters
        self.rock_types = ['Granite', 'Basalt', 'Schist', 'Quartzite', 'Gneiss', 'Sandstone', 'Limestone']
        self.support_types = ['Rock Bolts', 'Cable Bolts', 'Mesh', 'Shotcrete', 'Steel Sets']
        self.directions = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
        
        # Rock quality correlations (realistic geological relationships)
        self.rock_stability_factors = {
            'Granite': {'rqd_range': (60, 95), 'base_stability': 0.7},
            'Quartzite': {'rqd_range': (55, 90), 'base_stability': 0.65},
            'Gneiss': {'rqd_range': (45, 85), 'base_stability': 0.6},
            'Basalt': {'rqd_range': (40, 80), 'base_stability': 0.55},
            'Sandstone': {'rqd_range': (35, 75), 'base_stability': 0.5},
            'Limestone': {'rqd_range': (30, 70), 'base_stability': 0.45},
            'Schist': {'rqd_range': (25, 65), 'base_stability': 0.4}
        }
        
        # Support effectiveness
        self.support_effectiveness = {
            'Cable Bolts': 0.85,
            'Rock Bolts': 0.75,
            'Steel Sets': 0.7,
            'Shotcrete': 0.65,
            'Mesh': 0.5
        }
        
    def generate_static_features(self, num_stopes: int = 50) -> pd.DataFrame:
        """Generate realistic static features with proper geological correlations"""
        
        static_data = []
        
        for i in range(1, num_stopes + 1):
            # Select rock type
            rock_type = np.random.choice(self.rock_types)
            rock_params = self.rock_stability_factors[rock_type]
            
            # Generate correlated RQD based on rock type
            rqd = np.random.uniform(rock_params['rqd_range'][0], rock_params['rqd_range'][1])
            
            # Generate depth with realistic distribution (more shallow stopes)
            depth = np.random.exponential(300) + 150  # exponential distribution starting at 150m
            depth = min(depth, 1000)  # cap at 1000m
            
            # Hydraulic radius correlated with depth and rock quality
            hr_base = 4 + (depth / 200) + np.random.normal(0, 1.5)
            hr = max(2, min(20, hr_base))  # constrain between 2-20m
            
            # Dip angle - realistic geological distribution
            dip = np.random.beta(2, 2) * 85 + 15  # beta distribution for realistic dip angles
            
            # Direction
            direction = np.random.choice(self.directions)
            
            # Undercut width correlated with hydraulic radius
            undercut_wdt = hr * np.random.uniform(0.7, 1.3)
            undercut_wdt = max(2, min(10, undercut_wdt))
            
            # Support type selection (better support for worse rock)
            if rqd < 40:
                support_type = np.random.choice(['Steel Sets', 'Cable Bolts'], p=[0.6, 0.4])
            elif rqd < 60:
                support_type = np.random.choice(['Rock Bolts', 'Cable Bolts', 'Shotcrete'], p=[0.4, 0.3, 0.3])
            else:
                support_type = np.random.choice(['Rock Bolts', 'Mesh', 'Shotcrete'], p=[0.5, 0.3, 0.2])
            
            # Support density based on rock quality and depth
            base_density = 1.0 - (rqd / 100) + (depth / 2000)
            support_density = np.random.uniform(base_density * 0.7, base_density * 1.3)
            support_density = max(0.1, min(1.0, support_density))
            
            # Support installation (70% chance installed)
            support_installed = np.random.choice([0, 1], p=[0.3, 0.7])
            
            # Calculate stability based on realistic factors
            stability_score = self._calculate_stability(
                rqd, hr, depth, dip, rock_type, support_type, 
                support_density, support_installed
            )
            
            # Convert to binary (0/1) with some uncertainty around threshold
            threshold = 0.5 + np.random.normal(0, 0.05)  # varying threshold
            stability = 1 if stability_score > threshold else 0
            
            static_data.append({
                'stope_name': f'Stope_{i:03d}',
                'rqd': round(rqd, 2),
                'hr': round(hr, 2),
                'depth': round(depth, 2),
                'dip': round(dip, 2),
                'direction': direction,
                'undercut_wdt': round(undercut_wdt, 2),
                'rock_type': rock_type,
                'support_type': support_type,
                'support_density': round(support_density, 2),
                'support_installed': support_installed,
                'stability': stability
            })
        
        df = pd.DataFrame(static_data)
        
        # Ensure balanced classes (adjust some marginal cases)
        self._balance_stability_classes(df)
        
        return df
    
    def _calculate_stability(self, rqd: float, hr: float, depth: float, dip: float, 
                           rock_type: str, support_type: str, support_density: float, 
                           support_installed: int) -> float:
        """Calculate stability score based on geological and engineering factors"""
        
        # Base stability from rock type
        base_stability = self.rock_stability_factors[rock_type]['base_stability']
        
        # RQD effect (higher RQD = more stable)
        rqd_effect = (rqd - 20) / 80  # normalize to 0-1
        
        # Depth effect (deeper = less stable)
        depth_effect = max(0, 1 - depth / 1000)
        
        # Hydraulic radius effect (larger spans = less stable)
        hr_effect = max(0, 1 - hr / 20)
        
        # Dip effect (steeper dips can be less stable)
        dip_effect = 1 - abs(dip - 45) / 90  # optimal around 45 degrees
        
        # Support effect
        support_effect = 0
        if support_installed:
            support_base = self.support_effectiveness[support_type]
            support_effect = support_base * support_density
        
        # Combine factors with weights
        stability = (
            base_stability * 0.3 +
            rqd_effect * 0.25 +
            depth_effect * 0.15 +
            hr_effect * 0.15 +
            dip_effect * 0.05 +
            support_effect * 0.1
        )
        
        # Add some noise
        stability += np.random.normal(0, 0.05)
        
        return max(0, min(1, stability))
    
    def _balance_stability_classes(self, df: pd.DataFrame) -> None:
        """Ensure realistic class distribution with more stable stopes"""
        total = len(df)
        
        # Realistic distribution: 74% stable (37/50), 26% unstable (13/50)
        target_stable = 37  # Odd number as requested
        target_unstable = 13  # Odd number as requested
        
        stable_count = (df['stability'] == 1).sum()
        unstable_count = (df['stability'] == 0).sum()
        
        if stable_count > target_stable:
            # Convert some stable to unstable
            excess = stable_count - target_stable
            stable_indices = df[df['stability'] == 1].index
            to_flip = np.random.choice(stable_indices, size=excess, replace=False)
            df.loc[to_flip, 'stability'] = 0
        elif stable_count < target_stable:
            # Convert some unstable to stable
            needed = target_stable - stable_count
            unstable_indices = df[df['stability'] == 0].index
            if len(unstable_indices) >= needed:
                to_flip = np.random.choice(unstable_indices, size=needed, replace=False)
                df.loc[to_flip, 'stability'] = 1
    
    def generate_timeseries_data(self, static_df: pd.DataFrame, 
                                days_per_stope: int = 90) -> pd.DataFrame:
        """Generate realistic timeseries data with trends and patterns"""
        
        timeseries_data = []
        start_date = datetime(2024, 1, 1)
        
        for _, stope in static_df.iterrows():
            stope_name = stope['stope_name']
            stability = stope['stability']
            
            # Generate parameters for this stope
            params = self._get_stope_timeseries_params(stope)
            
            for day in range(days_per_stope):
                current_date = start_date + timedelta(days=day)
                
                # Generate realistic sensor readings
                sensors = self._generate_sensor_readings(day, days_per_stope, stability, params)
                
                timeseries_data.append({
                    'stope_name': stope_name,
                    'timestamp': current_date.strftime('%Y-%m-%d'),
                    'vibration_velocity': round(sensors['vibration'], 2),
                    'deformation_rate': round(sensors['deformation'], 2),
                    'stress': round(sensors['stress'], 2),
                    'temperature': round(sensors['temperature'], 2),
                    'humidity': round(sensors['humidity'], 2)
                })
        
        return pd.DataFrame(timeseries_data)
    
    def _get_stope_timeseries_params(self, stope: pd.Series) -> Dict:
        """Get stope-specific parameters for timeseries generation"""
        
        rock_type = stope['rock_type']
        depth = stope['depth']
        stability = stope['stability']
        
        # Base temperature increases with depth
        base_temp = 20 + (depth / 100)  # roughly 1°C per 100m
        
        # Base humidity depends on rock type and depth
        base_humidity = {
            'Limestone': 75, 'Sandstone': 70, 'Schist': 65,
            'Basalt': 60, 'Gneiss': 55, 'Quartzite': 50, 'Granite': 45
        }[rock_type] + (depth / 50)
        
        # Instability trends
        if stability == 0:
            # Unstable stopes show degrading trends
            vibration_trend = np.random.uniform(0.002, 0.008)
            deformation_trend = np.random.uniform(0.001, 0.005)
            stress_trend = np.random.uniform(0.1, 0.3)
        else:
            # Stable stopes show minimal or no trends
            vibration_trend = np.random.uniform(-0.001, 0.002)
            deformation_trend = np.random.uniform(-0.001, 0.001)
            stress_trend = np.random.uniform(-0.05, 0.1)
        
        return {
            'base_temp': base_temp,
            'base_humidity': base_humidity,
            'vibration_trend': vibration_trend,
            'deformation_trend': deformation_trend,
            'stress_trend': stress_trend,
            'rock_type': rock_type,
            'depth': depth
        }
    
    def _generate_sensor_readings(self, day: int, total_days: int, 
                                stability: int, params: Dict) -> Dict:
        """Generate realistic sensor readings for a specific day"""
        
        # Time-based factors
        time_factor = day / total_days
        
        # Seasonal effects (simplified)
        seasonal = np.sin(2 * np.pi * day / 365) * 0.1
        
        # Generate vibration velocity (mm/s)
        base_vibration = 0.5 + np.random.uniform(0, 1.5)
        vibration_trend = params['vibration_trend'] * day
        vibration_noise = np.random.normal(0, 0.2)
        vibration = max(0.1, base_vibration + vibration_trend + vibration_noise + seasonal)
        
        # Generate deformation rate (mm/day)
        base_deformation = 0.1 + np.random.uniform(0, 0.5)
        deformation_trend = params['deformation_trend'] * day
        deformation_noise = np.random.normal(0, 0.1)
        deformation = max(0.01, base_deformation + deformation_trend + deformation_noise)
        
        # Generate stress (MPa) - correlated with vibration
        base_stress = 15 + vibration * 10 + np.random.uniform(0, 15)
        stress_trend = params['stress_trend'] * day
        stress_noise = np.random.normal(0, 2)
        stress = max(5, base_stress + stress_trend + stress_noise)
        
        # Generate temperature (°C) - depth-dependent with seasonal variation
        temp_seasonal = seasonal * 3  # 3°C seasonal variation
        temp_noise = np.random.normal(0, 1)
        temperature = params['base_temp'] + temp_seasonal + temp_noise
        
        # Generate humidity (%) - inversely related to temperature
        humidity_seasonal = -seasonal * 5  # opposite to temperature
        humidity_noise = np.random.normal(0, 3)
        humidity = max(30, min(95, params['base_humidity'] + humidity_seasonal + humidity_noise))
        
        return {
            'vibration': vibration,
            'deformation': deformation,
            'stress': stress,
            'temperature': temperature,
            'humidity': humidity
        }

def main():
    """Generate improved dataset"""
    print("🏗️ Generating improved mine stope dataset...")
    
    generator = ImprovedDatasetGenerator()
    
    # Generate static features (50 stopes for better distribution)
    print("📊 Generating static features...")
    static_df = generator.generate_static_features(num_stopes=50)
    
    # Check balance
    stable_count = (static_df['stability'] == 1).sum()
    unstable_count = (static_df['stability'] == 0).sum()
    print(f"   Static features: {len(static_df)} stopes")
    print(f"   Class distribution: Stable={stable_count}, Unstable={unstable_count}")
    print(f"   Balance ratio: {stable_count/len(static_df):.2f} stable")
    
    # Generate timeseries data
    print("📈 Generating timeseries data...")
    timeseries_df = generator.generate_timeseries_data(static_df, days_per_stope=90)
    print(f"   Timeseries data: {len(timeseries_df)} records")
    print(f"   Time range: {timeseries_df['timestamp'].min()} to {timeseries_df['timestamp'].max()}")
    
    # Save to files
    static_path = 'data/stope_static_features_aligned.csv'
    timeseries_path = 'data/stope_timeseries_data_aligned.csv'
    
    print("\n💾 Saving improved dataset...")
    static_df.to_csv(static_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)
    
    print(f"✅ Static features saved to: {static_path}")
    print(f"✅ Timeseries data saved to: {timeseries_path}")
    
    # Display sample statistics
    print("\n📊 Dataset Statistics:")
    print("\nStatic Features:")
    print(f"  Rock types: {', '.join(static_df['rock_type'].value_counts().index.tolist())}")
    print(f"  Support types: {', '.join(static_df['support_type'].value_counts().index.tolist())}")
    print(f"  RQD range: {static_df['rqd'].min():.1f} - {static_df['rqd'].max():.1f}")
    print(f"  Depth range: {static_df['depth'].min():.1f} - {static_df['depth'].max():.1f} m")
    print(f"  Support installed: {(static_df['support_installed'] == 1).sum()}/{len(static_df)} stopes")
    
    print("\nTimeseries Features:")
    for col in ['vibration_velocity', 'deformation_rate', 'stress', 'temperature', 'humidity']:
        print(f"  {col}: {timeseries_df[col].min():.2f} - {timeseries_df[col].max():.2f}")
    
    print(f"\n🎯 Dataset generation completed!")
    print(f"   Total stopes: {len(static_df)}")
    print(f"   Total timeseries records: {len(timeseries_df)}")
    print(f"   Average days per stope: {len(timeseries_df)/len(static_df):.1f}")

if __name__ == "__main__":
    main()
