"""
Django management command to train the dual-branch stability prediction model.
Usage: python manage.py train_model
"""

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
import os
import sys
import json
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor
from core.utils import get_stope_profile_summary


class TrainingVisualizer:
    """
    Comprehensive visualization suite for model training process.
    """
    
    def __init__(self, save_dir='training_visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def save_training_history_plots(self, history, model_horizons):
        """Generate comprehensive training history visualizations."""
        print("📈 Generating training history visualizations...")
        
        # Create main training history plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced Dual-Branch Model Training History', fontsize=16, fontweight='bold')
        
        # Overall loss
        self._plot_loss(history, axes[0, 0])
        
        # Current stability accuracy
        self._plot_current_stability_metrics(history, axes[0, 1])
        
        # Future predictions accuracy (average)
        self._plot_future_predictions_average(history, model_horizons, axes[0, 2])
        
        # Learning rate schedule
        self._plot_learning_rate(history, axes[1, 0])
        
        # Individual horizon accuracies
        self._plot_individual_horizons(history, model_horizons, axes[1, 1])
        
        # Loss components breakdown
        self._plot_loss_components(history, axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate separate detailed plots
        self._generate_detailed_plots(history, model_horizons)
        
    def _plot_loss(self, history, ax):
        """Plot overall training and validation loss."""
        epochs = range(1, len(history.history['loss']) + 1)
        
        ax.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_title('Model Loss Over Time', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Highlight best epoch
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_loss = min(history.history['val_loss'])
        ax.annotate(f'Best: Epoch {best_epoch}\\nLoss: {best_loss:.4f}',
                   xy=(best_epoch, best_loss), xytext=(best_epoch + 5, best_loss + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def _plot_current_stability_metrics(self, history, ax):
        """Plot current stability prediction metrics."""
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Try to find current stability metrics
        acc_key = None
        val_acc_key = None
        
        for key in history.history.keys():
            if 'current_stability' in key and 'accuracy' in key and 'val' not in key:
                acc_key = key
            elif 'current_stability' in key and 'accuracy' in key and 'val' in key:
                val_acc_key = key
        
        if acc_key and val_acc_key:
            ax.plot(epochs, history.history[acc_key], 'g-', label='Training Accuracy', linewidth=2)
            ax.plot(epochs, history.history[val_acc_key], 'orange', label='Validation Accuracy', linewidth=2)
            
            final_acc = history.history[val_acc_key][-1]
            ax.set_title(f'Current Stability Accuracy\\n(Final: {final_acc:.3f})', fontweight='bold')
        else:
            # Fallback to general accuracy
            if 'accuracy' in history.history:
                ax.plot(epochs, history.history['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
            if 'val_accuracy' in history.history:
                ax.plot(epochs, history.history['val_accuracy'], 'orange', label='Validation Accuracy', linewidth=2)
                final_acc = history.history['val_accuracy'][-1]
                ax.set_title(f'Model Accuracy\\n(Final: {final_acc:.3f})', fontweight='bold')
            else:
                ax.set_title('Current Stability Accuracy\\n(No data available)', fontweight='bold')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    def _plot_future_predictions_average(self, history, horizons, ax):
        """Plot average future predictions accuracy."""
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Find future prediction accuracy keys
        future_keys = [k for k in history.history.keys() 
                      if 'future_risk' in k and 'accuracy' in k and 'val' not in k]
        future_val_keys = [k for k in history.history.keys() 
                          if 'future_risk' in k and 'accuracy' in k and 'val' in k]
        
        if future_keys and future_val_keys:
            # Calculate average accuracy across horizons
            train_avg = np.mean([history.history[k] for k in future_keys], axis=0)
            val_avg = np.mean([history.history[k] for k in future_val_keys], axis=0)
            
            ax.plot(epochs, train_avg, 'purple', label='Training Avg', linewidth=2)
            ax.plot(epochs, val_avg, 'cyan', label='Validation Avg', linewidth=2)
            
            final_avg = val_avg[-1]
            ax.set_title(f'Future Predictions Accuracy (Avg)\\n(Final: {final_avg:.3f})', fontweight='bold')
        else:
            ax.set_title('Future Predictions Accuracy\\n(No data available)', fontweight='bold')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    def _plot_learning_rate(self, history, ax):
        """Plot learning rate schedule."""
        if 'lr' in history.history:
            epochs = range(1, len(history.history['lr']) + 1)
            ax.plot(epochs, history.history['lr'], 'red', linewidth=2)
            ax.set_title('Learning Rate Schedule', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Learning Rate\\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Rate Schedule', fontweight='bold')
    
    def _plot_individual_horizons(self, history, horizons, ax):
        """Plot individual horizon accuracies."""
        epochs = range(1, len(history.history['loss']) + 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(horizons)))
        
        plotted = False
        for i, (horizon, color) in enumerate(zip(horizons, colors)):
            key = f'val_future_risk_{horizon}d_accuracy'
            if key in history.history:
                ax.plot(epochs, history.history[key], color=color, 
                       label=f'{horizon}d', linewidth=2, alpha=0.8)
                plotted = True
        
        if plotted:
            ax.set_title('Future Prediction Accuracy by Horizon', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, 'Horizon Data\\nNot Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Future Prediction Accuracy by Horizon', fontweight='bold')
    
    def _plot_loss_components(self, history, ax):
        """Plot individual loss components."""
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Find loss component keys
        loss_keys = [k for k in history.history.keys() 
                    if 'loss' in k and 'val' not in k and k != 'loss']
        
        if loss_keys:
            colors = plt.cm.Set3(np.linspace(0, 1, len(loss_keys)))
            for key, color in zip(loss_keys, colors):
                label = key.replace('_', ' ').title()
                ax.plot(epochs, history.history[key], color=color, 
                       label=label, linewidth=2, alpha=0.8)
            
            ax.set_title('Individual Loss Components', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Loss Components\\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Individual Loss Components', fontweight='bold')
    
    def _generate_detailed_plots(self, history, horizons):
        """Generate additional detailed plots."""
        print("📊 Generating detailed analysis plots...")
        
        # Performance summary plot
        self._create_performance_summary(history, horizons)
        
        # Training progression heatmap
        self._create_training_heatmap(history, horizons)
        
        # Metrics comparison chart
        self._create_metrics_comparison(history)
    
    def _create_performance_summary(self, history, horizons):
        """Create performance summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
        
        # Final accuracies bar chart
        final_accuracies = []
        horizon_labels = ['Current']
        
        # Current stability accuracy
        current_acc_key = None
        for key in history.history.keys():
            if 'current_stability' in key and 'accuracy' in key and 'val' in key:
                current_acc_key = key
                break
        
        if current_acc_key:
            final_accuracies.append(history.history[current_acc_key][-1])
        else:
            final_accuracies.append(history.history.get('val_accuracy', [0])[-1])
        
        # Future prediction accuracies
        for horizon in horizons:
            key = f'val_future_risk_{horizon}d_accuracy'
            if key in history.history:
                final_accuracies.append(history.history[key][-1])
                horizon_labels.append(f'{horizon}d')
        
        bars = ax1.bar(horizon_labels, final_accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(final_accuracies))))
        ax1.set_title('Final Validation Accuracies', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Loss progression
        epochs = range(1, len(history.history['loss']) + 1)
        ax2.plot(epochs, history.history['loss'], label='Training', linewidth=2)
        ax2.plot(epochs, history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Loss Progression', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training stability (loss variance)
        window_size = max(1, len(epochs) // 10)
        train_variance = []
        val_variance = []
        
        for i in range(window_size, len(epochs)):
            train_window = history.history['loss'][i-window_size:i]
            val_window = history.history['val_loss'][i-window_size:i]
            train_variance.append(np.var(train_window))
            val_variance.append(np.var(val_window))
        
        var_epochs = epochs[window_size:]
        ax3.plot(var_epochs, train_variance, label='Training Variance', linewidth=2)
        ax3.plot(var_epochs, val_variance, label='Validation Variance', linewidth=2)
        ax3.set_title('Training Stability (Loss Variance)', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Overfitting analysis
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        overfitting_metric = [v - t for t, v in zip(train_loss, val_loss)]
        
        ax4.plot(epochs, overfitting_metric, color='red', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Analysis (Val Loss - Train Loss)', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.grid(True, alpha=0.3)
        
        # Color regions
        ax4.fill_between(epochs, overfitting_metric, 0, where=np.array(overfitting_metric) > 0, 
                        color='red', alpha=0.3, label='Overfitting')
        ax4.fill_between(epochs, overfitting_metric, 0, where=np.array(overfitting_metric) <= 0, 
                        color='green', alpha=0.3, label='Good Fit')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_training_heatmap(self, history, horizons):
        """Create training progress heatmap."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Collect metrics data
        metrics_data = []
        metric_names = []
        
        # Current stability metrics
        for metric in ['accuracy', 'precision', 'recall']:
            key = f'val_current_stability_{metric}'
            if key in history.history:
                metrics_data.append(history.history[key])
                metric_names.append(f'Current {metric.title()}')
        
        # Future prediction metrics
        for horizon in horizons:
            key = f'val_future_risk_{horizon}d_accuracy'
            if key in history.history:
                metrics_data.append(history.history[key])
                metric_names.append(f'{horizon}d Accuracy')
        
        if metrics_data:
            # Create heatmap data
            heatmap_data = np.array(metrics_data)
            
            # Normalize data for better visualization
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            heatmap_normalized = scaler.fit_transform(heatmap_data.T).T
            
            sns.heatmap(heatmap_normalized, 
                       xticklabels=range(1, len(history.history['loss']) + 1),
                       yticklabels=metric_names,
                       cmap='RdYlGn', 
                       ax=ax,
                       cbar_kws={'label': 'Normalized Performance'})
            
            ax.set_title('Training Progress Heatmap (Normalized Metrics)', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Metrics')
            
            # Only show every 5th epoch on x-axis for readability
            epoch_ticks = range(0, len(history.history['loss']), max(1, len(history.history['loss']) // 10))
            ax.set_xticks(epoch_ticks)
            ax.set_xticklabels([str(i+1) for i in epoch_ticks])
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\\nfor Heatmap', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Training Progress Heatmap', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_comparison(self, history):
        """Create metrics comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training vs Validation Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Find all accuracy metrics
        train_acc_keys = [k for k in history.history.keys() if 'accuracy' in k and 'val' not in k]
        val_acc_keys = [k for k in history.history.keys() if 'accuracy' in k and 'val' in k]
        
        if train_acc_keys and val_acc_keys:
            epochs = range(1, len(history.history['loss']) + 1)
            
            # Plot accuracy metrics
            ax = axes[0, 0]
            for i, (train_key, val_key) in enumerate(zip(train_acc_keys, val_acc_keys)):
                color = plt.cm.tab10(i)
                metric_name = train_key.replace('_accuracy', '').replace('_', ' ').title()
                ax.plot(epochs, history.history[train_key], '--', color=color, 
                       label=f'{metric_name} (Train)', alpha=0.7)
                ax.plot(epochs, history.history[val_key], '-', color=color, 
                       label=f'{metric_name} (Val)', linewidth=2)
            
            ax.set_title('Accuracy Metrics', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Loss comparison
        ax = axes[0, 1]
        ax.plot(epochs, history.history['loss'], label='Training Loss', linewidth=2)
        ax.plot(epochs, history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_title('Loss Comparison', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convergence analysis
        ax = axes[1, 0]
        train_loss_smooth = self._smooth_curve(history.history['loss'])
        val_loss_smooth = self._smooth_curve(history.history['val_loss'])
        ax.plot(epochs, train_loss_smooth, label='Training (Smoothed)', linewidth=2)
        ax.plot(epochs, val_loss_smooth, label='Validation (Smoothed)', linewidth=2)
        ax.set_title('Loss Convergence (Smoothed)', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final metrics bar chart
        ax = axes[1, 1]
        final_metrics = {}
        
        # Collect final validation metrics
        for key in history.history.keys():
            if 'val_' in key and 'accuracy' in key:
                metric_name = key.replace('val_', '').replace('_accuracy', '').replace('_', ' ').title()
                final_metrics[metric_name] = history.history[key][-1]
        
        if final_metrics:
            bars = ax.bar(final_metrics.keys(), final_metrics.values(), 
                         color=plt.cm.viridis(np.linspace(0, 1, len(final_metrics))))
            ax.set_title('Final Validation Metrics', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, final_metrics.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_curve(self, points, factor=0.9):
        """Apply exponential smoothing to a curve."""
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    
    def save_model_architecture_plot(self, model):
        """Save model architecture visualization."""
        print("🏗️ Generating model architecture visualization...")
        
        try:
            # Save model summary to file
            with open(os.path.join(self.save_dir, 'model_architecture.txt'), 'w') as f:
                model.combined_model.summary(print_fn=lambda x: f.write(x + '\\n'))
            
            # Create architecture visualization plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Model statistics
            total_params = model.combined_model.count_params()
            trainable_params = total_params  # Assuming all are trainable
            
            # Create visual representation
            layers_info = []
            for layer in model.combined_model.layers:
                layer_info = {
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'params': layer.count_params()
                }
                layers_info.append(layer_info)
            
            # Plot layer parameters
            layer_names = [info['name'][:20] + '...' if len(info['name']) > 20 else info['name'] 
                          for info in layers_info]
            layer_params = [info['params'] for info in layers_info]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
            bars = ax.bar(range(len(layer_names)), layer_params, color=colors)
            
            ax.set_title(f'Model Architecture Overview\\nTotal Parameters: {total_params:,}', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Layers')
            ax.set_ylabel('Number of Parameters')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Add value labels on significant bars
            max_params = max(layer_params) if layer_params else 0
            for i, (bar, params) in enumerate(zip(bars, layer_params)):
                if params > max_params * 0.05:  # Only label bars with >5% of max params
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_params * 0.01,
                           f'{params:,}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'model_architecture.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate architecture plot: {e}")
    
    def generate_training_report(self, history, model, training_time, options):
        """Generate comprehensive training report."""
        print("📋 Generating comprehensive training report...")
        
        report_path = os.path.join(self.save_dir, 'training_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Dual-Branch Stability Predictor - Training Report\\n")
            f.write("=" * 70 + "\\n\\n")
            f.write(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Training Duration**: {training_time:.2f} seconds\\n\\n")
            
            # Training configuration
            f.write("## 🛠️ Training Configuration\\n\\n")
            f.write(f"- **Epochs**: {options['epochs']}\\n")
            f.write(f"- **Batch Size**: {options['batch_size']}\\n")
            f.write(f"- **Learning Rate**: {options['learning_rate']}\\n")
            f.write(f"- **Validation Split**: {options['validation_split']}\\n")
            f.write(f"- **Total Epochs Trained**: {len(history.history['loss'])}\\n\\n")
            
            # Model architecture
            f.write("## 🏗️ Model Architecture\\n\\n")
            f.write(f"- **Model Type**: Enhanced Dual-Branch Neural Network\\n")
            f.write(f"- **Total Parameters**: {model.combined_model.count_params():,}\\n")
            f.write(f"- **Prediction Horizons**: {model.prediction_horizons} days\\n")
            f.write(f"- **Risk Classes**: {len(model.risk_levels)} classes\\n\\n")
            
            # Final performance
            f.write("## 📊 Final Performance\\n\\n")
            
            # Current stability performance
            current_acc_key = None
            for key in history.history.keys():
                if 'current_stability' in key and 'accuracy' in key and 'val' in key:
                    current_acc_key = key
                    break
            
            if current_acc_key:
                current_acc = history.history[current_acc_key][-1]
                f.write(f"### Current Stability Prediction\\n")
                f.write(f"- **Validation Accuracy**: {current_acc:.4f}\\n\\n")
            
            # Future predictions performance
            f.write(f"### Future Risk Predictions\\n")
            for horizon in model.prediction_horizons:
                key = f'val_future_risk_{horizon}d_accuracy'
                if key in history.history:
                    acc = history.history[key][-1]
                    f.write(f"- **{horizon}-day Accuracy**: {acc:.4f}\\n")
            f.write("\\n")
            
            # Training metrics
            f.write("## 📈 Training Metrics\\n\\n")
            f.write(f"- **Final Training Loss**: {history.history['loss'][-1]:.6f}\\n")
            f.write(f"- **Final Validation Loss**: {history.history['val_loss'][-1]:.6f}\\n")
            f.write(f"- **Best Validation Loss**: {min(history.history['val_loss']):.6f}\\n")
            f.write(f"- **Best Epoch**: {np.argmin(history.history['val_loss']) + 1}\\n\\n")
            
            # Training stability
            overfitting_metric = [v - t for t, v in zip(history.history['loss'], history.history['val_loss'])]
            avg_overfitting = np.mean(overfitting_metric[-10:])  # Last 10 epochs
            
            f.write("## 🎯 Training Analysis\\n\\n")
            f.write(f"- **Average Overfitting (Last 10 epochs)**: {avg_overfitting:.6f}\\n")
            if avg_overfitting < 0.01:
                f.write("- **Assessment**: Well-fitted model ✅\\n")
            elif avg_overfitting < 0.05:
                f.write("- **Assessment**: Slight overfitting ⚠️\\n")
            else:
                f.write("- **Assessment**: Significant overfitting ❌\\n")
            f.write("\\n")
            
            # Generated files
            f.write("## 📁 Generated Visualizations\\n\\n")
            f.write("- `training_history_comprehensive.png` - Complete training history\\n")
            f.write("- `performance_summary.png` - Performance summary and analysis\\n")
            f.write("- `training_heatmap.png` - Training progress heatmap\\n")
            f.write("- `metrics_comparison.png` - Training vs validation comparison\\n")
            f.write("- `model_architecture.png` - Model architecture visualization\\n")
            f.write("- `model_architecture.txt` - Detailed model summary\\n\\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\\n\\n")
            
            final_acc = history.history.get('val_accuracy', history.history.get(current_acc_key, [0]))[-1]
            if final_acc < 0.85:
                f.write("- Consider increasing model complexity or training duration\\n")
                f.write("- Review data quality and feature engineering\\n")
            elif final_acc > 0.95:
                f.write("- Excellent performance achieved! ✅\\n")
                f.write("- Monitor for overfitting in production\\n")
            else:
                f.write("- Good performance achieved ✅\\n")
                f.write("- Consider fine-tuning for production deployment\\n")
            
            if avg_overfitting > 0.05:
                f.write("- Implement stronger regularization (dropout, L2)\\n")
                f.write("- Consider reducing model complexity\\n")
                f.write("- Increase training data if possible\\n")
            
            f.write("\\n---\\n")
            f.write("*Report generated by Deepmine Sentinel AI Training System*\\n")
        
        print(f"✅ Training report saved to: {report_path}")
        return report_path


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
            default=True,
            help='Save comprehensive training plots and visualizations (default: True)'
        )
        parser.add_argument(
            '--plots-dir',
            type=str,
            default='training_visualizations',
            help='Directory to save plots (default: training_visualizations)'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Verbose output'
        )
        parser.add_argument(
            '--generate-report',
            action='store_true',
            default=True,
            help='Generate comprehensive training report (default: True)'
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
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Go to deepmine_sentinel_ai
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

            # Save comprehensive visualizations if requested
            if options['save_plots']:
                start_viz_time = time.time()
                self.stdout.write("📈 Generating comprehensive training visualizations...")
                
                # Initialize visualizer
                visualizer = TrainingVisualizer(save_dir=options['plots_dir'])
                
                # Generate all training history plots
                visualizer.save_training_history_plots(history, model.prediction_horizons)
                
                # Generate model architecture visualization
                visualizer.save_model_architecture_plot(model)
                
                # Generate comprehensive training report
                if options['generate_report']:
                    training_time = time.time() - start_viz_time
                    report_path = visualizer.generate_training_report(
                        history, model, training_time, options
                    )
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Comprehensive training report generated: {report_path}")
                    )
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"✅ All training visualizations saved to: {options['plots_dir']}/\n"
                        f"   📊 Generated files:\n"
                        f"   • training_history_comprehensive.png - Complete training curves\n"
                        f"   • performance_summary.png - Performance analysis\n"
                        f"   • training_heatmap.png - Progress heatmap\n"
                        f"   • metrics_comparison.png - Training vs validation\n"
                        f"   • model_architecture.png - Architecture visualization\n"
                        f"   • model_architecture.txt - Detailed model summary\n"
                        f"   • training_report.md - Comprehensive training report"
                    )
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

            self.stdout.write(
                self.style.SUCCESS(
                    "\n✅ Enhanced model training completed successfully!\n"
                    "🎯 The model now provides:\n"
                    "   • Current stability prediction\n"
                    "   • Multi-horizon future risk forecasting\n"
                    "   • Risk trend analysis\n"
                    "   • Intelligent explanations and recommendations\n"
                    "🚀 Model type: Enhanced Dual-Branch Neural Network\n"
                    f"📁 Visualizations saved to: {options['plots_dir']}/"
                )
            )

        except Exception as e:
            raise CommandError(f"❌ Training failed: {str(e)}")
