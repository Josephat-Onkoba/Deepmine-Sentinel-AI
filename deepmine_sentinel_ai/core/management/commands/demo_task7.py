"""
Task 7 Demonstration: LSTM Training Data Preparation

Demonstrates working Task 7 implementation with:
- Synthetic dataset generation  
- Data preprocessing pipeline
- Feature engineering
- Data augmentation
- LSTM training pipeline
"""

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.data.synthetic_generator import SyntheticDataGenerator
from core.data.preprocessor import DataPreprocessor, PreprocessingConfig, SequenceData
from core.data.feature_engineer import FeatureEngineer, FeatureEngineeringConfig  
from core.data.augmentor import DataAugmentor
from core.data.lstm_pipeline import LSTMDataPipeline
from core.models import Stope, OperationalEvent


class Command(BaseCommand):
    help = 'Demonstrate Task 7: LSTM Training Data Preparation functionality'

    def add_arguments(self, parser):
        parser.add_argument(
            '--quick',
            action='store_true',
            help='Run abbreviated demo for quick testing',
        )

    def handle(self, *args, **options):
        """Run complete Task 7 demonstration"""
        try:
            self.stdout.write('🚀 Task 7: LSTM Training Data Preparation Demo')
            self.stdout.write('=' * 70)
            
            self.demo_synthetic_generation()
            self.demo_data_preprocessing()
            self.demo_feature_engineering()
            self.demo_data_augmentation()
            self.demo_lstm_pipeline()
            
            self.stdout.write('\n✅ Task 7 Demo completed successfully!')
            
        except Exception as e:
            self.stdout.write(f'\n❌ Demo failed: {str(e)}')
            import traceback
            traceback.print_exc()

    def demo_synthetic_generation(self):
        """Demonstrate synthetic data generation"""
        self.stdout.write('\n🎲 Demo 1: Synthetic Data Generation')
        self.stdout.write('-' * 50)
        
        generator = SyntheticDataGenerator()
        
        # Generate complete synthetic dataset
        success = generator.generate_complete_dataset()
        
        if success:
            stope_count = Stope.objects.count()
            event_count = OperationalEvent.objects.count()
            
            self.stdout.write(f'  ✅ Generated {stope_count} synthetic stopes')
            self.stdout.write(f'  ✅ Generated {event_count} operational events')
            
            # Show rock type distribution
            rock_types = Stope.objects.values_list('rock_type', flat=True).distinct().count()
            mining_methods = Stope.objects.values_list('mining_method', flat=True).distinct().count()
            
            self.stdout.write(f'  📊 Rock types: {rock_types} different types')
            self.stdout.write(f'  ⛏️  Mining methods: {mining_methods} different methods')
            self.stdout.write('  ✅ Synthetic generation: WORKING')

    def demo_data_preprocessing(self):
        """Demonstrate data preprocessing pipeline"""
        self.stdout.write('\n🔄 Demo 2: Data Preprocessing Components')
        self.stdout.write('-' * 50)
        
        config = PreprocessingConfig(
            sequence_length=24,  # 24 hours
            prediction_horizon=12,  # 12 hours ahead
            sampling_interval=1,  # 1 hour intervals
            min_events_per_sequence=1,
            validation_split=0.2,
            test_split=0.2
        )
        
        preprocessor = DataPreprocessor(config)
        
        self.stdout.write('  ✅ Preprocessor initialized')
        self.stdout.write(f'  📏 Sequence length: {config.sequence_length} hours')
        self.stdout.write(f'  🔮 Prediction horizon: {config.prediction_horizon} hours')
        
        # Test with a small subset
        test_stopes = Stope.objects.all()[:5]
        self.stdout.write(f'  📊 Testing with {len(test_stopes)} stopes')
        
        try:
            sequence_data = preprocessor.prepare_sequences(test_stopes)
            self.stdout.write(f'  ✅ Static features shape: {sequence_data.static_features.shape}')
            self.stdout.write('  ✅ Data preprocessing: WORKING')
        except Exception as e:
            self.stdout.write(f'  ⚠️  Preprocessing: {str(e)}')

    def demo_feature_engineering(self):
        """Demonstrate feature engineering capabilities"""
        self.stdout.write('\n🔧 Demo 3: Feature Engineering')
        self.stdout.write('-' * 50)
        
        config = FeatureEngineeringConfig()
        engineer = FeatureEngineer(config)
        
        # Create dummy SequenceData object
        dynamic_features = np.random.random((10, 24, 8))  # 10 samples, 24 timesteps, 8 features
        static_features = np.random.random((10, 8))  # 10 samples, 8 static features
        timestamps = np.array([pd.Timestamp.now() + pd.Timedelta(hours=i) for i in range(10)])
        risk_labels = np.random.randint(0, 4, 10)  # 10 samples, 4 risk classes
        stope_ids = np.arange(10)
        
        dummy_data = SequenceData(
            static_features=static_features,
            dynamic_features=dynamic_features,
            risk_labels=risk_labels,
            timestamps=timestamps,
            stope_ids=stope_ids,
            metadata={'test': True}
        )
        
        self.stdout.write(f'  📊 Input data shape: {dummy_data.dynamic_features.shape}')
        
        # Test feature engineering
        engineered_features = engineer.engineer_features(dummy_data)
        
        self.stdout.write(f'  ✅ Engineered features shape: {engineered_features.dynamic_features.shape}')
        self.stdout.write(f'  📈 Feature expansion: {dummy_data.dynamic_features.shape[2]} → {engineered_features.dynamic_features.shape[2]}')
        
        self.stdout.write('  ✅ Feature engineering: WORKING')

    def demo_data_augmentation(self):
        """Demonstrate data augmentation"""
        self.stdout.write('\n📈 Demo 4: Data Augmentation')
        self.stdout.write('-' * 50)
        
        augmentor = DataAugmentor()
        
        # Create balanced dummy data
        dynamic_features = np.random.random((20, 12, 8))  # 20 samples, 12 timesteps, 8 features
        static_features = np.random.random((20, 8))  # 20 samples, 8 static features  
        timestamps = np.array([pd.Timestamp.now() + pd.Timedelta(hours=i) for i in range(20)])
        risk_labels = np.random.randint(0, 4, 20)  # Balanced across 4 classes
        stope_ids = np.arange(20)
        
        dummy_data = SequenceData(
            static_features=static_features,
            dynamic_features=dynamic_features,
            risk_labels=risk_labels,
            timestamps=timestamps,
            stope_ids=stope_ids,
            metadata={'test': True}
        )
        
        self.stdout.write(f'  📊 Input data shape: {dummy_data.dynamic_features.shape}')
        self.stdout.write(f'  📊 Risk distribution: {np.bincount(dummy_data.risk_labels)}')
        
        # Test augmentation
        augmented_data = augmentor.augment_data(dummy_data)
        
        self.stdout.write(f'  ✅ Augmented data shape: {augmented_data.dynamic_features.shape}')
        self.stdout.write(f'  📈 Sample increase: {len(dummy_data.risk_labels)} → {len(augmented_data.risk_labels)}')
        self.stdout.write(f'  ⚖️  New distribution: {np.bincount(augmented_data.risk_labels)}')
        
        self.stdout.write('  ✅ Data augmentation: WORKING')

    def demo_lstm_pipeline(self):
        """Demonstrate end-to-end LSTM pipeline"""
        self.stdout.write('\n🧠 Demo 5: Complete LSTM Pipeline')
        self.stdout.write('-' * 50)
        
        pipeline = LSTMDataPipeline()
        
        self.stdout.write('  ✅ Pipeline initialized')
        
        # Run complete pipeline (generates own data)
        self.stdout.write(f'  🔄 Running complete pipeline')
        
        try:
            results = pipeline.run_complete_pipeline()
            
            if results['success']:
                self.stdout.write(f'  ✅ Pipeline completed successfully')
                self.stdout.write(f'  ⏱️  Duration: {results["pipeline_duration"]:.1f} seconds')
                self.stdout.write(f'  📊 Stages: {", ".join(results["stages_completed"])}')
                
                # Show statistics if available
                stats = results.get('statistics', {})
                if 'train_samples' in stats:
                    self.stdout.write(f'  � Training samples: {stats["train_samples"]}')
                if 'final_class_distribution' in stats:
                    self.stdout.write(f'  ⚖️  Final distribution: {stats["final_class_distribution"]}')
                
                self.stdout.write('  ✅ LSTM pipeline: WORKING')
            else:
                self.stdout.write(f'  ❌ Pipeline failed: {results.get("error", "Unknown error")}')
            
        except Exception as e:
            self.stdout.write(f'  ⚠️  Pipeline: {str(e)}')

    def display_summary(self):
        """Display final summary statistics"""
        self.stdout.write('\n📈 Demo Summary')
        self.stdout.write('-' * 50)
        
        # Database statistics
        stope_count = Stope.objects.count()
        event_count = OperationalEvent.objects.count()
        
        self.stdout.write(f'  📊 Total stopes: {stope_count}')
        self.stdout.write(f'  ⚡ Total events: {event_count}')
        
        if event_count > 0:
            # Event type distribution
            event_types = OperationalEvent.objects.values_list('event_type', flat=True)
            event_dist = pd.Series(event_types).value_counts()
            
            self.stdout.write('  📋 Event distribution:')
            for event_type, count in event_dist.items():
                self.stdout.write(f'    - {event_type}: {count}')
