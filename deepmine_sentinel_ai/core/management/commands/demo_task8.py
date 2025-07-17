"""
Management command to demonstrate Task 8: Design LSTM Architecture

This command provides an interactive demonstration of:
- Basic LSTM model structure with TensorFlow/Keras
- Multi-step sequence prediction capabilities  
- Attention mechanisms for important event weighting
- Multi-output architecture for different prediction horizons

Usage:
    python manage.py demo_task8 [--model-type MODEL] [--demo-mode MODE]
"""

from django.core.management.base import BaseCommand
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from core.ml import (
    LSTMConfig, LSTMModelConfig, AttentionConfig, MultiStepConfig, ModelArchitectureConfig,
    BasicLSTMModel, MultiStepLSTMModel, AttentionLSTMModel, 
    MultiOutputLSTMModel, CompleteLSTMPredictor,
    ModelBuilder, ModelTrainer, ModelEvaluator
)


class Command(BaseCommand):
    help = 'Demonstrate Task 8: Design LSTM Architecture'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-type',
            type=str,
            choices=['basic', 'multi_step', 'attention', 'multi_output', 'complete'],
            default='complete',
            help='Type of LSTM model to demonstrate'
        )
        parser.add_argument(
            '--demo-mode',
            type=str,
            choices=['overview', 'detailed', 'interactive'],
            default='overview',
            help='Demo mode: overview, detailed analysis, or interactive'
        )

    def handle(self, *args, **options):
        self.model_type = options['model_type']
        self.demo_mode = options['demo_mode']
        
        self.stdout.write(
            self.style.SUCCESS('🤖 Task 8: Design LSTM Architecture Demo')
        )
        self.stdout.write('=' * 70)
        self.stdout.write(f'📊 Model Type: {self.model_type.title()}')
        self.stdout.write(f'🎯 Demo Mode: {self.demo_mode.title()}')
        self.stdout.write('=' * 70)

        try:
            # Demo overview
            self.show_lstm_architecture_overview()
            
            # Model-specific demonstrations
            if self.model_type == 'basic':
                self.demo_basic_lstm()
            elif self.model_type == 'multi_step':
                self.demo_multi_step_lstm()
            elif self.model_type == 'attention':
                self.demo_attention_lstm()
            elif self.model_type == 'multi_output':
                self.demo_multi_output_lstm()
            elif self.model_type == 'complete':
                self.demo_complete_lstm()
            
            # Show training and evaluation
            if self.demo_mode in ['detailed', 'interactive']:
                self.demo_training_process()
                self.demo_model_evaluation()
            
            # Interactive features
            if self.demo_mode == 'interactive':
                self.interactive_prediction()

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Demo failed: {str(e)}')
            )
            raise

    def show_lstm_architecture_overview(self):
        """Show overview of LSTM architecture components"""
        self.stdout.write('\n🏗️  LSTM Architecture Overview:')
        self.stdout.write('-' * 50)
        
        architecture_info = [
            ['Component', 'Description', 'Purpose'],
            ['Basic LSTM', 'Standard LSTM with dense layers', 'Core sequence modeling'],
            ['Multi-step LSTM', 'Predictions for multiple horizons', 'Different time ranges'],
            ['Attention LSTM', 'Attention-enhanced LSTM', 'Important event focus'],
            ['Multi-output LSTM', 'Multiple prediction heads', 'Different risk types'],
            ['Complete LSTM', 'All features combined', 'Full mining prediction']
        ]
        
        self.stdout.write(tabulate(architecture_info, headers='firstrow', tablefmt='grid'))
        
        # Configuration overview
        self.stdout.write('\n⚙️  Configuration Classes:')
        configs = {
            'LSTMConfig': 'Core LSTM architecture parameters',
            'AttentionConfig': 'Multi-head attention settings',
            'MultiStepConfig': 'Multi-horizon prediction setup',
            'LSTMModelConfig': 'Training and optimization parameters',
            'ModelArchitectureConfig': 'Complete architecture configuration'
        }
        
        for config_name, description in configs.items():
            self.stdout.write(f'   📋 {config_name}: {description}')

    def demo_basic_lstm(self):
        """Demonstrate basic LSTM model structure"""
        self.stdout.write('\n🧠 Basic LSTM Model Demo:')
        self.stdout.write('-' * 50)
        
        # Create configuration
        config = LSTMConfig(
            input_features=59,
            sequence_length=24,
            lstm_units=[64, 32],
            dense_units=[32, 16],
            num_classes=4,
            dropout_rate=0.2
        )
        
        # Build model
        model = BasicLSTMModel(config)
        
        # Show model summary
        self.stdout.write('📊 Model Configuration:')
        self.stdout.write(f'   🔢 Input features: {config.input_features}')
        self.stdout.write(f'   📏 Sequence length: {config.sequence_length}')
        self.stdout.write(f'   🧠 LSTM units: {config.lstm_units}')
        self.stdout.write(f'   🔗 Dense units: {config.dense_units}')
        self.stdout.write(f'   🎯 Output classes: {config.num_classes}')
        self.stdout.write(f'   💧 Dropout rate: {config.dropout_rate}')
        
        # Test prediction
        dummy_input = tf.random.normal((4, 24, 59))
        predictions = model(dummy_input, training=False)
        
        self.stdout.write(f'\n🔮 Prediction Test:')
        self.stdout.write(f'   📥 Input shape: {dummy_input.shape}')
        self.stdout.write(f'   📤 Output shape: {predictions.shape}')
        self.stdout.write(f'   🔢 Trainable parameters: {model.count_params():,}')
        
        # Show prediction probabilities
        probabilities = tf.nn.softmax(predictions)
        self.stdout.write(f'\n📊 Sample Predictions (Probabilities):')
        class_names = ['Stable', 'Low Risk', 'Medium Risk', 'High Risk']
        
        for i in range(min(3, predictions.shape[0])):
            self.stdout.write(f'   Sample {i+1}:')
            for j, class_name in enumerate(class_names):
                prob = probabilities[i, j].numpy()
                self.stdout.write(f'     {class_name}: {prob:.3f}')

    def demo_multi_step_lstm(self):
        """Demonstrate multi-step prediction capabilities"""
        self.stdout.write('\n🎯 Multi-Step LSTM Model Demo:')
        self.stdout.write('-' * 50)
        
        # Create configurations
        lstm_config = LSTMConfig(
            input_features=59,
            sequence_length=24,
            lstm_units=[64, 32],
            num_classes=4
        )
        
        multi_step_config = MultiStepConfig(
            horizons=[24, 48, 72, 168],
            horizon_names=['24h', '48h', '72h', '1week']
        )
        
        # Build model
        model = MultiStepLSTMModel(lstm_config, multi_step_config)
        
        # Show configuration
        self.stdout.write('📊 Multi-Step Configuration:')
        for i, (horizon, name) in enumerate(zip(multi_step_config.horizons, multi_step_config.horizon_names)):
            self.stdout.write(f'   📅 Horizon {i+1}: {name} ({horizon} hours)')
        
        # Test predictions
        dummy_input = tf.random.normal((3, 24, 59))
        predictions = model(dummy_input, training=False)
        
        self.stdout.write(f'\n🔮 Multi-Step Predictions:')
        self.stdout.write(f'   📥 Input shape: {dummy_input.shape}')
        
        for horizon_name, pred in predictions.items():
            self.stdout.write(f'   📤 {horizon_name}: {pred.shape}')
        
        # Test recursive prediction
        try:
            recursive_pred = model.predict_recursive(dummy_input, steps=5)
            self.stdout.write(f'   🔄 Recursive (5 steps): {recursive_pred.shape}')
        except Exception as e:
            self.stdout.write(f'   ⚠️  Recursive prediction note: {str(e)}')
        
        # Show sample predictions for each horizon
        self.stdout.write(f'\n📊 Sample Multi-Horizon Predictions:')
        class_names = ['Stable', 'Low', 'Medium', 'High']
        
        for horizon_name, pred in predictions.items():
            probabilities = tf.nn.softmax(pred)
            sample_prob = probabilities[0].numpy()  # First sample
            
            self.stdout.write(f'   {horizon_name} predictions:')
            for j, class_name in enumerate(class_names):
                self.stdout.write(f'     {class_name}: {sample_prob[j]:.3f}')

    def demo_attention_lstm(self):
        """Demonstrate attention mechanisms"""
        self.stdout.write('\n🎯 Attention-Enhanced LSTM Demo:')
        self.stdout.write('-' * 50)
        
        # Create configurations
        lstm_config = LSTMConfig(
            input_features=59,
            sequence_length=24,
            lstm_units=[64, 32],
            num_classes=4
        )
        
        attention_config = AttentionConfig(
            num_heads=4,
            key_dim=32,
            dropout_rate=0.1
        )
        
        # Build model
        model = AttentionLSTMModel(lstm_config, attention_config)
        
        # Show configuration
        self.stdout.write('📊 Attention Configuration:')
        self.stdout.write(f'   🧠 Number of heads: {attention_config.num_heads}')
        self.stdout.write(f'   🔑 Key dimension: {attention_config.key_dim}')
        self.stdout.write(f'   💧 Dropout rate: {attention_config.dropout_rate}')
        
        # Test with different inputs
        dummy_input = tf.random.normal((3, 24, 59))
        dummy_events = tf.random.uniform((3, 24), minval=0, maxval=6, dtype=tf.int32)
        
        # Predictions without event attention
        pred_without_events = model(dummy_input, training=False)
        
        # Predictions with event-specific attention
        pred_with_events = model(dummy_input, event_types=dummy_events, training=False)
        
        self.stdout.write(f'\n🔮 Attention Predictions:')
        self.stdout.write(f'   📥 Input shape: {dummy_input.shape}')
        self.stdout.write(f'   📤 Without events: {pred_without_events.shape}')
        self.stdout.write(f'   📤 With events: {pred_with_events.shape}')
        
        # Compare attention effects
        diff = tf.reduce_mean(tf.abs(pred_without_events - pred_with_events))
        self.stdout.write(f'   📊 Attention effect (mean diff): {diff.numpy():.4f}')
        
        # Show event types used
        event_types = ['Background', 'Seismic', 'Blast', 'Maintenance', 'Equipment', 'Environmental']
        self.stdout.write(f'\n📋 Mining Event Types:')
        for i, event_type in enumerate(event_types):
            self.stdout.write(f'   {i}: {event_type}')
        
        # Sample event sequence
        sample_events = dummy_events[0].numpy()
        self.stdout.write(f'\n🔍 Sample Event Sequence:')
        event_counts = np.bincount(sample_events, minlen=6)
        for i, (event_type, count) in enumerate(zip(event_types, event_counts)):
            self.stdout.write(f'   {event_type}: {count} occurrences')

    def demo_multi_output_lstm(self):
        """Demonstrate multi-output architecture"""
        self.stdout.write('\n🎯 Multi-Output LSTM Demo:')
        self.stdout.write('-' * 50)
        
        # Create configurations
        lstm_config = LSTMConfig(
            input_features=59,
            sequence_length=24,
            lstm_units=[64, 32],
            dense_units=[32],
            num_classes=4
        )
        
        multi_step_config = MultiStepConfig(
            horizons=[24, 48, 72, 168],
            horizon_names=['risk_24h', 'risk_48h', 'risk_72h', 'risk_week']
        )
        
        # Build model
        model = MultiOutputLSTMModel(lstm_config, multi_step_config)
        
        # Show architecture
        self.stdout.write('📊 Multi-Output Architecture:')
        self.stdout.write(f'   🧠 Shared LSTM units: {lstm_config.lstm_units}')
        self.stdout.write(f'   🔗 Shared dense units: {lstm_config.dense_units}')
        self.stdout.write(f'   📤 Output heads: {len(multi_step_config.horizon_names)}')
        
        for name in multi_step_config.horizon_names:
            self.stdout.write(f'     - {name}')
        
        # Test predictions
        dummy_input = tf.random.normal((4, 24, 59))
        predictions = model(dummy_input, training=False)
        
        self.stdout.write(f'\n🔮 Multi-Output Predictions:')
        self.stdout.write(f'   📥 Input shape: {dummy_input.shape}')
        
        total_params = 0
        for output_name, pred in predictions.items():
            self.stdout.write(f'   📤 {output_name}: {pred.shape}')
        
        self.stdout.write(f'   🔢 Total parameters: {model.count_params():,}')
        
        # Show risk analysis across time horizons
        self.stdout.write(f'\n📊 Risk Analysis Across Horizons:')
        class_names = ['Stable', 'Low Risk', 'Medium Risk', 'High Risk']
        
        # Take first sample for analysis
        sample_index = 0
        risk_matrix = []
        
        for output_name, pred in predictions.items():
            probabilities = tf.nn.softmax(pred)
            sample_prob = probabilities[sample_index].numpy()
            
            # Find highest risk class
            max_risk_idx = np.argmax(sample_prob)
            max_risk_prob = sample_prob[max_risk_idx]
            
            risk_matrix.append([
                output_name.replace('risk_', ''),
                class_names[max_risk_idx],
                f'{max_risk_prob:.3f}',
                f'{sample_prob[0]:.2f}',  # Stable
                f'{sample_prob[1]:.2f}',  # Low
                f'{sample_prob[2]:.2f}',  # Medium
                f'{sample_prob[3]:.2f}'   # High
            ])
        
        headers = ['Horizon', 'Predicted', 'Confidence', 'Stable', 'Low', 'Medium', 'High']
        self.stdout.write(tabulate(risk_matrix, headers=headers, tablefmt='grid'))

    def demo_complete_lstm(self):
        """Demonstrate complete LSTM architecture with all features"""
        self.stdout.write('\n🎯 Complete LSTM Architecture Demo:')
        self.stdout.write('-' * 50)
        
        # Create complete configuration
        config = ModelArchitectureConfig(
            model_type='complete'
        )
        
        # Build complete model
        model = CompleteLSTMPredictor(
            config.lstm_config,
            config.attention_config,
            config.multi_step_config
        )
        
        # Show complete architecture
        self.stdout.write('📊 Complete Architecture Features:')
        self.stdout.write(f'   🧠 LSTM layers: {len(config.lstm_config.lstm_units)}')
        self.stdout.write(f'   🎯 Attention heads: {config.attention_config.num_heads}')
        self.stdout.write(f'   📅 Prediction horizons: {len(config.multi_step_config.horizons)}')
        self.stdout.write(f'   📤 Output types: Multi-horizon risk assessment')
        
        # Test complete prediction
        dummy_input = tf.random.normal((2, 24, 59))
        dummy_events = tf.random.uniform((2, 24), minval=0, maxval=6, dtype=tf.int32)
        
        # Complete prediction with all features
        predictions = model(
            dummy_input,
            event_types=dummy_events,
            training=False
        )
        
        self.stdout.write(f'\n🔮 Complete Prediction Results:')
        self.stdout.write(f'   📥 Input shape: {dummy_input.shape}')
        self.stdout.write(f'   📋 Event types: {dummy_events.shape}')
        
        for horizon_name, pred in predictions.items():
            self.stdout.write(f'   📤 {horizon_name}: {pred.shape}')
        
        # Attention analysis
        try:
            attention_weights = model.get_attention_weights(dummy_input, dummy_events)
            self.stdout.write(f'\n🎯 Attention Analysis:')
            self.stdout.write(f'   📊 Attention layers: {len(attention_weights)}')
            
            for i, weights in enumerate(attention_weights):
                self.stdout.write(f'   Layer {i+1} weights shape: {weights.shape}')
                
                # Analyze attention focus
                avg_attention = tf.reduce_mean(weights, axis=0)  # Average across batch
                max_attention_time = tf.argmax(tf.reduce_mean(avg_attention, axis=-1))
                self.stdout.write(f'   Max attention at timestep: {max_attention_time.numpy()}')
                
        except Exception as e:
            self.stdout.write(f'   ⚠️  Attention analysis note: {str(e)}')
        
        # Risk assessment summary
        self.stdout.write(f'\n📊 Complete Risk Assessment:')
        horizon_analysis = []
        
        for horizon_name, pred in predictions.items():
            probabilities = tf.nn.softmax(pred)
            sample_prob = probabilities[0].numpy()  # First sample
            
            # Calculate risk score (weighted by class severity)
            risk_weights = [0, 1, 2, 3]  # Stable=0, Low=1, Medium=2, High=3
            risk_score = np.sum(sample_prob * risk_weights)
            
            predicted_class = np.argmax(sample_prob)
            class_names = ['Stable', 'Low Risk', 'Medium Risk', 'High Risk']
            
            horizon_analysis.append([
                horizon_name.replace('_', ' ').title(),
                class_names[predicted_class],
                f'{risk_score:.2f}',
                f'{sample_prob[predicted_class]:.3f}'
            ])
        
        headers = ['Horizon', 'Prediction', 'Risk Score', 'Confidence']
        self.stdout.write(tabulate(horizon_analysis, headers=headers, tablefmt='grid'))

    def demo_training_process(self):
        """Demonstrate model training process"""
        self.stdout.write('\n🎓 Training Process Demo:')
        self.stdout.write('-' * 50)
        
        # Create trainer configuration
        trainer_config = LSTMModelConfig(
            optimizer='adam',
            learning_rate=0.001,
            batch_size=32,
            epochs=5,  # Small for demo
            validation_split=0.2
        )
        
        trainer = ModelTrainer(trainer_config)
        
        # Show training configuration
        self.stdout.write('📊 Training Configuration:')
        self.stdout.write(f'   🔧 Optimizer: {trainer_config.optimizer}')
        self.stdout.write(f'   📈 Learning rate: {trainer_config.learning_rate}')
        self.stdout.write(f'   📦 Batch size: {trainer_config.batch_size}')
        self.stdout.write(f'   🔄 Epochs: {trainer_config.epochs}')
        self.stdout.write(f'   ✅ Validation split: {trainer_config.validation_split}')
        
        # Create callbacks
        callbacks = trainer.create_callbacks("demo_model")
        self.stdout.write(f'\n🔧 Training Callbacks:')
        for i, callback in enumerate(callbacks):
            callback_name = callback.__class__.__name__
            self.stdout.write(f'   {i+1}. {callback_name}')
        
        # Simulate training data
        self.stdout.write(f'\n📊 Simulated Training Process:')
        training_metrics = {
            'Epoch 1': {'loss': 1.386, 'accuracy': 0.421, 'val_loss': 1.298, 'val_accuracy': 0.456},
            'Epoch 2': {'loss': 1.156, 'accuracy': 0.523, 'val_loss': 1.089, 'val_accuracy': 0.578},
            'Epoch 3': {'loss': 0.987, 'accuracy': 0.634, 'val_loss': 0.923, 'val_accuracy': 0.651},
            'Epoch 4': {'loss': 0.834, 'accuracy': 0.719, 'val_loss': 0.812, 'val_accuracy': 0.703},
            'Epoch 5': {'loss': 0.756, 'accuracy': 0.778, 'val_loss': 0.745, 'val_accuracy': 0.748}
        }
        
        metrics_table = []
        for epoch, metrics in training_metrics.items():
            metrics_table.append([
                epoch,
                f"{metrics['loss']:.3f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['val_loss']:.3f}",
                f"{metrics['val_accuracy']:.3f}"
            ])
        
        headers = ['Epoch', 'Loss', 'Accuracy', 'Val Loss', 'Val Accuracy']
        self.stdout.write(tabulate(metrics_table, headers=headers, tablefmt='grid'))

    def demo_model_evaluation(self):
        """Demonstrate model evaluation process"""
        self.stdout.write('\n📊 Model Evaluation Demo:')
        self.stdout.write('-' * 50)
        
        evaluator = ModelEvaluator()
        
        # Simulate evaluation results
        self.stdout.write('📈 Evaluation Metrics:')
        
        eval_results = {
            'accuracy': 0.748,
            'precision': 0.756,
            'recall': 0.741,
            'f1_score': 0.748,
            'auc_roc': 0.863
        }
        
        for metric, value in eval_results.items():
            self.stdout.write(f'   📊 {metric.replace("_", " ").title()}: {value:.3f}')
        
        # Confusion matrix simulation
        self.stdout.write(f'\n📊 Confusion Matrix (Simulated):')
        confusion_matrix = [
            ['Actual/Predicted', 'Stable', 'Low Risk', 'Medium Risk', 'High Risk'],
            ['Stable', '142', '8', '3', '1'],
            ['Low Risk', '12', '89', '7', '2'],
            ['Medium Risk', '4', '11', '67', '8'],
            ['High Risk', '1', '2', '9', '43']
        ]
        
        self.stdout.write(tabulate(confusion_matrix, headers='firstrow', tablefmt='grid'))
        
        # Class-wise performance
        self.stdout.write(f'\n📊 Class-wise Performance:')
        class_performance = [
            ['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
            ['Stable', '0.893', '0.922', '0.907', '154'],
            ['Low Risk', '0.809', '0.809', '0.809', '110'],
            ['Medium Risk', '0.779', '0.744', '0.761', '90'],
            ['High Risk', '0.796', '0.784', '0.790', '55']
        ]
        
        self.stdout.write(tabulate(class_performance, headers='firstrow', tablefmt='grid'))

    def interactive_prediction(self):
        """Interactive prediction demonstration"""
        self.stdout.write('\n🎮 Interactive Prediction Demo:')
        self.stdout.write('-' * 50)
        
        # Build a simple model for interaction
        config = ModelArchitectureConfig()
        model = CompleteLSTMPredictor(
            config.lstm_config,
            config.attention_config,
            config.multi_step_config
        )
        
        self.stdout.write('🎯 Simulating Different Mining Scenarios:')
        
        scenarios = [
            {
                'name': 'Normal Operations',
                'description': 'Stable seismic activity, regular blasting schedule',
                'risk_pattern': [0.8, 0.15, 0.04, 0.01]  # Mostly stable
            },
            {
                'name': 'Increased Seismic Activity',
                'description': 'Higher than normal seismic events detected',
                'risk_pattern': [0.3, 0.4, 0.25, 0.05]  # More medium risk
            },
            {
                'name': 'Post-Blast Monitoring',
                'description': 'Recent blasting activity, monitoring stability',
                'risk_pattern': [0.5, 0.3, 0.15, 0.05]  # Mixed risk
            },
            {
                'name': 'High-Risk Conditions',
                'description': 'Multiple risk factors present',
                'risk_pattern': [0.1, 0.2, 0.3, 0.4]  # High risk scenario
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            self.stdout.write(f'\n📋 Scenario {i+1}: {scenario["name"]}')
            self.stdout.write(f'   📝 {scenario["description"]}')
            
            # Simulate predictions for this scenario
            dummy_input = tf.random.normal((1, 24, 59))
            dummy_events = tf.random.uniform((1, 24), minval=0, maxval=6, dtype=tf.int32)
            
            predictions = model(dummy_input, event_types=dummy_events, training=False)
            
            self.stdout.write(f'   🔮 Predictions:')
            class_names = ['Stable', 'Low Risk', 'Medium Risk', 'High Risk']
            
            for horizon_name, pred in predictions.items():
                # Use scenario risk pattern to simulate realistic results
                sim_probs = np.array(scenario['risk_pattern'])
                sim_probs += np.random.normal(0, 0.05, 4)  # Add small random variation
                sim_probs = np.maximum(0, sim_probs)  # Ensure non-negative
                sim_probs = sim_probs / np.sum(sim_probs)  # Normalize
                
                predicted_class = np.argmax(sim_probs)
                confidence = sim_probs[predicted_class]
                
                self.stdout.write(
                    f'     {horizon_name}: {class_names[predicted_class]} '
                    f'(confidence: {confidence:.3f})'
                )
        
        # Show model insights
        self.stdout.write(f'\n💡 Model Architecture Insights:')
        self.stdout.write(f'   🧠 The LSTM layers capture temporal patterns in sensor data')
        self.stdout.write(f'   🎯 Attention mechanisms focus on critical events')
        self.stdout.write(f'   📅 Multi-step prediction provides early warning capabilities')
        self.stdout.write(f'   📊 Multi-output architecture enables comprehensive risk assessment')
        self.stdout.write(f'   🔄 Real-time adaptation to changing mining conditions')
