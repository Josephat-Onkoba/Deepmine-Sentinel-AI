"""
Django management command to make stability predictions using the trained model.
Usage: python manage.py predict_stability [stope_name]
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys
from core.ml.models.dual_branch_stability_predictor import DualBranchStopeStabilityPredictor
from core.models import Stope
from core.utils import get_stope_profile_summary


class Command(BaseCommand):
    help = 'Make stability predictions using the trained dual-branch model'

    def add_arguments(self, parser):
        parser.add_argument(
            'stope_names',
            nargs='*',
            type=str,
            help='Names of stopes to predict (leave empty for all stopes)'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.5,
            help='Risk threshold for classification (default: 0.5)'
        )
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed predictions with confidence scores'
        )
        parser.add_argument(
            '--export',
            type=str,
            help='Export results to CSV file'
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS("🔮 Starting Deepmine Sentinel AI Predictions...")
        )
        
        try:
            # Setup paths for datasets with robust path resolution
            project_root = getattr(settings, 'BASE_DIR', None)
            if project_root is None:
                # Fallback to relative path calculation
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            
            static_features_path = os.path.join(project_root, 'data', 'stope_static_features_aligned.csv')
            timeseries_path = os.path.join(project_root, 'data', 'stope_timeseries_data_aligned.csv')
            
            # Verify that data files exist
            if not os.path.exists(static_features_path):
                raise CommandError(f"❌ Static features file not found at {static_features_path}")
            if not os.path.exists(timeseries_path):
                raise CommandError(f"❌ Timeseries data file not found at {timeseries_path}")
            
            # Check if trained model exists
            model_path = os.path.join(project_root, 'core', 'ml', 'models', 'saved', 'dual_branch_stability_model.h5')
            if not os.path.exists(model_path):
                raise CommandError(
                    f"❌ Trained model not found at {model_path}. "
                    "Please run 'python manage.py train_model' first."
                )
            
            # Initialize model (it will load the existing trained model)
            self.stdout.write("🧠 Loading trained model...")
            model = DualBranchStopeStabilityPredictor(
                static_features_path=static_features_path,
                timeseries_path=timeseries_path
            )
            
            # Load the saved model
            model.load_model(model_path)
            self.stdout.write(self.style.SUCCESS("✅ Model loaded successfully!"))

            # Determine which stopes to predict
            if options['stope_names']:
                # Predict for specific stopes from the dataset
                stope_names = options['stope_names']
            else:
                # Get all unique stope names from the static features dataset
                static_df = model.static_df
                stope_names = static_df['stope_name'].unique().tolist()[:10]  # Limit to 10 for demo
                
            if not stope_names:
                raise CommandError("❌ No stopes found to predict")

            self.stdout.write(f"🎯 Making predictions for {len(stope_names)} stopes...")
            
            predictions = []
            for stope_name in stope_names:
                try:
                    # Make prediction
                    prediction_result = model.predict_stability(stope_name)
                    
                    if prediction_result is not None:
                        instability_prob = prediction_result['instability_probability']
                        risk_level = prediction_result['risk_level']
                        is_stable = prediction_result['stable']
                        
                        stability_class = "Stable" if is_stable else "Unstable"
                        confidence = (1 - instability_prob) if is_stable else instability_prob
                        
                        predictions.append({
                            'stope_name': stope_name,
                            'instability_probability': instability_prob,
                            'stability_class': stability_class,
                            'risk_level': risk_level,
                            'confidence': confidence
                        })
                        
                        risk_emoji = "🟢" if is_stable else "🔴"
                        self.stdout.write(
                            f"   {risk_emoji} {stope_name}: {stability_class} "
                            f"(risk: {risk_level}, prob: {instability_prob:.3f})"
                        )
                    else:
                        self.stdout.write(
                            self.style.WARNING(f"   ⚠️  Unable to predict for {stope_name}")
                        )
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f"   ❌ Error predicting {stope_name}: {str(e)}")
                    )

            # Summary
            successful_predictions = [p for p in predictions if 'instability_probability' in p]
            high_risk_count = len([p for p in successful_predictions if p['stability_class'] == 'Unstable'])
            stable_count = len([p for p in successful_predictions if p['stability_class'] == 'Stable'])
            
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n📊 Prediction Summary:\n"
                    f"   🔴 Unstable: {high_risk_count}\n"
                    f"   🟢 Stable: {stable_count}\n"
                    f"   📍 Total: {len(successful_predictions)}"
                )
            )

            # Export results if requested
            if options['export']:
                self.stdout.write(f"💾 Exporting results to {options['export']}...")
                try:
                    import pandas as pd
                    df = pd.DataFrame(predictions)
                    df.to_csv(options['export'], index=False)
                    self.stdout.write(self.style.SUCCESS("✅ Results exported successfully!"))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"❌ Export failed: {str(e)}"))

        except Exception as e:
            raise CommandError(f"❌ Prediction failed: {str(e)}")
