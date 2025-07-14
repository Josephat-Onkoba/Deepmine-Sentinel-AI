"""
Django management command to train the dual-branch stability prediction model.
Usage: python manage.py train_model
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys
import json
from datetime import datetime
import tensorflow as tf
from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
from core.utils import get_stope_profile_summary


class Command(BaseCommand):
    help = 'Train the dual-branch stability prediction model'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs (default: 50)'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Training batch size (default: 32)'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate (default: 0.001)'
        )
        parser.add_argument(
            '--validation-split',
            type=float,
            default=0.2,
            help='Validation split ratio (default: 0.2)'
        )
        parser.add_argument(
            '--test-split',
            type=float,
            default=0.1,
            help='Test split ratio (default: 0.1)'
        )
        parser.add_argument(
            '--save-plots',
            action='store_true',
            help='Save training plots'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS("🚀 Starting Deepmine Sentinel AI Model Training...")
        )
        
        try:
            # Configure TensorFlow
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                self.stdout.write(
                    self.style.SUCCESS(f"✅ GPU available: {physical_devices[0].name}")
                )
            else:
                self.stdout.write(
                    self.style.WARNING("⚠️  No GPU available, using CPU")
                )

            # Setup paths for datasets
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go to deepmine_sentinel_ai
            static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
            timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
            
            # Verify datasets exist
            if not os.path.exists(static_features_path):
                raise CommandError(f"❌ Static features dataset not found: {static_features_path}")
            if not os.path.exists(timeseries_path):
                raise CommandError(f"❌ Timeseries dataset not found: {timeseries_path}")

            # Initialize enhanced model
            self.stdout.write("🧠 Initializing enhanced dual-branch neural network...")
            model = EnhancedDualBranchStabilityPredictor(
                static_features_path=static_features_path,
                timeseries_path=timeseries_path
            )

            # Train enhanced model with temporal prediction capabilities
            self.stdout.write("🎯 Starting enhanced model training with future predictions...")
            history = model.train_enhanced_model(
                epochs=options['epochs'],
                batch_size=options['batch_size'],
                validation_split=options['validation_split']
            )

            # Print enhanced training results
            if hasattr(history.history, 'current_stability_accuracy'):
                current_accuracy = history.history['val_current_stability_accuracy'][-1]
                current_loss = history.history['val_current_stability_loss'][-1]
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\n🎉 Enhanced training completed successfully!\n"
                        f"📊 Current Stability Prediction Results:\n"
                        f"   - Validation Accuracy: {current_accuracy:.4f}\n"
                        f"   - Validation Loss: {current_loss:.4f}"
                    )
                )
                
                # Future prediction results
                future_acc_keys = [k for k in history.history.keys() 
                                 if 'future_risk' in k and 'accuracy' in k and 'val_' in k]
                if future_acc_keys:
                    self.stdout.write("📈 Future Prediction Results:")
                    for key in future_acc_keys:
                        horizon = key.split('_')[3].replace('d', '')
                        accuracy = history.history[key][-1]
                        self.stdout.write(f"   - {horizon} days ahead: {accuracy:.4f} accuracy")
            else:
                # Fallback for legacy metrics
                final_accuracy = history.history.get('val_accuracy', [0])[-1]
                final_loss = history.history.get('val_loss', [0])[-1]
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\n🎉 Training completed successfully!\n"
                        f"📊 Final Results:\n"
                        f"   - Validation Accuracy: {final_accuracy:.4f}\n"
                        f"   - Validation Loss: {final_loss:.4f}"
                    )
                )

            # Save the trained model (CRITICAL FIX)
            self.stdout.write("💾 Saving trained model and components...")
            model_save_path = os.path.join(settings.BASE_DIR, 'models', 'enhanced_dual_branch_model.keras')
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            
            try:
                model.save_enhanced_model(model_save_path)
                self.stdout.write(self.style.SUCCESS(f"✅ Model saved to: {model_save_path}"))
                self.stdout.write(self.style.SUCCESS(f"✅ Components saved to: {model_save_path.replace('.keras', '_components.pkl')}"))
                
                # Save training metadata
                metadata = {
                    'training_date': datetime.now().isoformat(),
                    'epochs': options['epochs'],
                    'batch_size': options['batch_size'],
                    'validation_split': options['validation_split'],
                    'learning_rate': options['learning_rate'],
                    'final_accuracy': history.history.get('val_accuracy', [0])[-1] if 'val_accuracy' in history.history else 0,
                    'final_loss': history.history.get('val_loss', [0])[-1] if 'val_loss' in history.history else 0,
                    'total_epochs_trained': len(history.history['loss']),
                    'model_parameters': model.combined_model.count_params() if hasattr(model, 'combined_model') and model.combined_model else 0,
                    'training_completed': True
                }
                
                metadata_path = model_save_path.replace('.keras', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                self.stdout.write(self.style.SUCCESS(f"✅ Training metadata saved to: {metadata_path}"))
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Failed to save model: {str(e)}"))
                raise CommandError(f"Model saving failed: {str(e)}")

            # Save enhanced plots if requested
            if options['save_plots']:
                self.stdout.write("📈 Saving enhanced training plots...")
                plot_path = model.plot_enhanced_training_history(history)
                self.stdout.write(
                    self.style.SUCCESS(f"✅ Enhanced training plots saved to: {plot_path}")
                )

            # Test enhanced predictions on sample data
            self.stdout.write("🧪 Testing enhanced model predictions...")
            try:
                from core.models import Stope
                sample_stopes = Stope.objects.all()[:3]  # Test fewer stopes due to complexity
                
                for stope in sample_stopes:
                    try:
                        # Test comprehensive prediction
                        result = model.predict_comprehensive_stability(stope.name)
                        
                        if result:
                            current = result['current_stability']
                            future_risks = result['future_predictions']
                            
                            self.stdout.write(
                                f"   📍 {stope.name}:"
                            )
                            self.stdout.write(
                                f"      Current: {current['risk_level']} risk "
                                f"(prob: {current['instability_probability']:.3f})"
                            )
                            
                            # Show future predictions
                            for pred in future_risks[:3]:  # Show first 3 horizons
                                self.stdout.write(
                                    f"      {pred['horizon_days']}d: {pred['predicted_risk_level']} "
                                    f"(conf: {pred['confidence']:.3f})"
                                )
                            
                            # Show trend
                            self.stdout.write(f"      Trend: {result['risk_trend']}")
                            
                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f"   ⚠️  Failed to predict for {stope.name}: {str(e)}"
                            )
                        )
                        
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f"⚠️  Could not test predictions: {str(e)}")
                )

            # Save the trained model
            self.stdout.write("💾 Saving the trained model...")
            model_save_path = os.path.join(project_root, 'models', 'enhanced_dual_branch_model.h5')
            model.save(model_save_path)
            self.stdout.write(
                self.style.SUCCESS(f"✅ Trained model saved to: {model_save_path}")
            )

            self.stdout.write(
                self.style.SUCCESS(
                    "\n✅ Enhanced model training completed successfully!\n"
                    "🎯 The model now provides:\n"
                    "   • Current stability prediction\n"
                    "   • Multi-horizon future risk forecasting\n"
                    "   • Risk trend analysis\n"
                    "   • Intelligent explanations and recommendations\n"
                    "🚀 Model type: Dual-Branch Neural Network"
                )
            )

        except Exception as e:
            raise CommandError(f"❌ Training failed: {str(e)}")
