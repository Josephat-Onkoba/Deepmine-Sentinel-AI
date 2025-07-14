#!/usr/bin/env python3
"""
Enhanced Model Evaluation Script
===============================

Comprehensive evaluation script for the Enhanced Dual-Branch Stability Predictor.
Implements a full suite of evaluation metrics including safety, operational, 
and technical performance indicators.

Features:
- Multi-task model evaluation (current + future predictions)
- Safety-critical metrics for mining applications
- Temporal analysis and horizon-specific performance
- Model interpretability and uncertainty quantification
- Comprehensive reporting and visualization

Author: Deepmine Sentinel AI Team
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve, log_loss, brier_score_loss,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Setup path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.ml.models.dual_branch_stability_predictor import EnhancedDualBranchStabilityPredictor

class ModelEvaluator:
    """
    Comprehensive evaluation framework for the Enhanced Dual-Branch Stability Predictor.
    """
    
    def __init__(self, model, test_data=None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained EnhancedDualBranchStabilityPredictor instance
            test_data: Optional test dataset, if None will use model's validation data
        """
        self.model = model
        self.test_data = test_data
        self.results = {}
        self.risk_levels = ['stable', 'slight_elevated', 'elevated', 'high', 'critical', 'unstable']
        
    def evaluate_comprehensive(self, save_dir='evaluation_results'):
        """
        Run comprehensive evaluation suite.
        
        Args:
            save_dir: Directory to save evaluation results and plots
        """
        print("🔍 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        # Create results directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare test data
        X_static, X_temporal, y_current, y_future = self._prepare_test_data()
        
        # Generate predictions
        print("🎯 Generating predictions...")
        current_pred, future_pred, pred_probs = self._generate_predictions(X_static, X_temporal)
        
        # 1. Current Stability Evaluation
        print("📊 Evaluating current stability predictions...")
        current_metrics = self._evaluate_current_stability(y_current, current_pred, pred_probs, save_dir)
        
        # 2. Future Risk Evaluation
        print("🔮 Evaluating future risk predictions...")
        future_metrics = self._evaluate_future_predictions(y_future, future_pred, save_dir)
        
        # 3. Safety-Critical Analysis
        print("🛡️ Performing safety-critical analysis...")
        safety_metrics = self._evaluate_safety_metrics(y_current, current_pred, pred_probs, save_dir)
        
        # 4. Temporal Analysis
        print("⏰ Performing temporal analysis...")
        temporal_metrics = self._evaluate_temporal_patterns(y_future, future_pred, save_dir)
        
        # 5. Model Interpretability
        print("🔍 Analyzing model interpretability...")
        interpretability_metrics = self._evaluate_interpretability(X_static, X_temporal, save_dir)
        
        # 6. Robustness Analysis
        print("💪 Performing robustness analysis...")
        robustness_metrics = self._evaluate_robustness(X_static, X_temporal, y_current, save_dir)
        
        # Compile comprehensive results
        self.results = {
            'current_stability': current_metrics,
            'future_predictions': future_metrics,
            'safety_analysis': safety_metrics,
            'temporal_analysis': temporal_metrics,
            'interpretability': interpretability_metrics,
            'robustness': robustness_metrics
        }
        
        # Generate comprehensive report
        self._generate_comprehensive_report(save_dir)
        
        print(f"✅ Evaluation completed! Results saved to: {save_dir}")
        return self.results
    
    def _prepare_test_data(self):
        """Prepare test data for evaluation."""
        if self.test_data is not None:
            return self.test_data
        else:
            # Use model's prepared data
            data = self.model.prepare_enhanced_training_data()
            # Use last 20% as test data
            test_size = int(0.2 * len(data['X_static']))
            return (
                data['X_static'][-test_size:],
                data['X_timeseries'][-test_size:], 
                data['y_current'][-test_size:],
                data['y_future'][-test_size:]
            )
    
    def _generate_predictions(self, X_static, X_temporal):
        """Generate model predictions for evaluation."""
        # Get predictions from the combined model
        predictions = self.model.combined_model.predict([X_static, X_temporal], verbose=0)
        
        # Extract current stability predictions
        current_pred_probs = predictions[0]
        current_pred = (current_pred_probs > 0.5).astype(int).flatten()
        
        # Extract future predictions
        future_pred = []
        for i in range(len(self.model.prediction_horizons)):
            future_pred.append(np.argmax(predictions[i+1], axis=1))
        future_pred = np.array(future_pred).T
        
        return current_pred, future_pred, current_pred_probs.flatten()
    
    def _evaluate_current_stability(self, y_true, y_pred, y_probs, save_dir):
        """Evaluate current stability prediction performance."""
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        f2 = f1_score(y_true, y_pred, average='binary', beta=2.0, zero_division=0)
        
        # Advanced metrics
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
            logloss = log_loss(y_true, y_probs)
            brier = brier_score_loss(y_true, y_probs)
        except:
            roc_auc = logloss = brier = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create visualizations
        self._plot_confusion_matrix(cm, ['Unstable', 'Stable'], save_dir, 'current_stability_cm.png')
        self._plot_roc_curve(y_true, y_probs, save_dir, 'current_stability_roc.png')
        self._plot_precision_recall_curve(y_true, y_probs, save_dir, 'current_stability_pr.png')
        self._plot_calibration_curve(y_true, y_probs, save_dir, 'current_stability_calibration.png')
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'f2_score': f2,
            'balanced_accuracy': balanced_acc,
            'matthews_corrcoef': mcc,
            'cohen_kappa': kappa,
            'roc_auc': roc_auc,
            'log_loss': logloss,
            'brier_score': brier,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def _evaluate_future_predictions(self, y_true, y_pred, save_dir):
        """Evaluate future risk prediction performance."""
        horizon_metrics = {}
        
        for i, horizon in enumerate(self.model.prediction_horizons):
            y_true_horizon = y_true[:, i]
            y_pred_horizon = y_pred[:, i]
            
            # Multi-class metrics
            accuracy = accuracy_score(y_true_horizon, y_pred_horizon)
            precision_macro = precision_score(y_true_horizon, y_pred_horizon, average='macro', zero_division=0)
            recall_macro = recall_score(y_true_horizon, y_pred_horizon, average='macro', zero_division=0)
            f1_macro = f1_score(y_true_horizon, y_pred_horizon, average='macro', zero_division=0)
            
            precision_weighted = precision_score(y_true_horizon, y_pred_horizon, average='weighted', zero_division=0)
            recall_weighted = recall_score(y_true_horizon, y_pred_horizon, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true_horizon, y_pred_horizon, average='weighted', zero_division=0)
            
            # Per-class metrics
            per_class_report = classification_report(y_true_horizon, y_pred_horizon, 
                                                   target_names=self.risk_levels[:max(y_true_horizon)+1],
                                                   output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_horizon, y_pred_horizon)
            
            # Plot confusion matrix for this horizon
            self._plot_confusion_matrix(cm, self.risk_levels[:cm.shape[0]], save_dir, 
                                      f'future_risk_{horizon}d_cm.png')
            
            horizon_metrics[f'{horizon}d'] = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted,
                'confusion_matrix': cm.tolist(),
                'per_class_metrics': per_class_report
            }
        
        # Plot horizon performance decay
        self._plot_horizon_performance(horizon_metrics, save_dir)
        
        return horizon_metrics
    
    def _evaluate_safety_metrics(self, y_true, y_pred, y_probs, save_dir):
        """Evaluate safety-critical metrics for mining applications."""
        
        # False negative rate (critical for safety)
        fn_rate = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0
        
        # True positive rate for unstable conditions (safety recall)
        safety_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        # False alarm rate
        false_alarm_rate = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        
        # Critical risk detection (assuming 0 = unstable, high probability = high risk)
        critical_threshold = 0.2  # Probability threshold for critical risk
        critical_predictions = y_probs <= critical_threshold
        critical_actual = y_true == 0
        
        if np.sum(critical_actual) > 0:
            critical_detection_rate = np.sum(critical_predictions & critical_actual) / np.sum(critical_actual)
        else:
            critical_detection_rate = 0.0
        
        # Safety score (weighted combination emphasizing safety)
        safety_score = 0.4 * safety_recall + 0.3 * (1 - fn_rate) + 0.2 * (1 - false_alarm_rate) + 0.1 * critical_detection_rate
        
        return {
            'false_negative_rate': fn_rate,
            'safety_recall': safety_recall,
            'false_alarm_rate': false_alarm_rate,
            'critical_detection_rate': critical_detection_rate,
            'safety_score': safety_score,
        }
    
    def _evaluate_temporal_patterns(self, y_future, pred_future, save_dir):
        """Evaluate temporal prediction patterns and consistency."""
        
        # Temporal consistency (how often predictions are consistent across horizons)
        consistency_scores = []
        for sample in range(len(pred_future)):
            sample_pred = pred_future[sample]
            # Check if predictions are monotonic or consistent in direction
            differences = np.diff(sample_pred)
            consistency = 1.0 - (np.sum(np.abs(differences) > 1) / len(differences))
            consistency_scores.append(consistency)
        
        temporal_consistency = np.mean(consistency_scores)
        
        # Accuracy decay over horizons
        horizon_accuracies = []
        for i, horizon in enumerate(self.model.prediction_horizons):
            acc = accuracy_score(y_future[:, i], pred_future[:, i])
            horizon_accuracies.append(acc)
        
        # Trend analysis (risk escalation/de-escalation prediction)
        trend_accuracy = self._calculate_trend_accuracy(y_future, pred_future)
        
        return {
            'temporal_consistency': temporal_consistency,
            'horizon_accuracies': horizon_accuracies,
            'accuracy_decay_rate': self._calculate_decay_rate(horizon_accuracies),
            'trend_accuracy': trend_accuracy
        }
    
    def _evaluate_interpretability(self, X_static, X_temporal, save_dir):
        """Evaluate model interpretability and feature importance."""
        
        # Feature importance analysis (simplified)
        # Note: For full interpretability, would need SHAP or LIME analysis
        
        feature_names = [
            'RQD', 'HR', 'Depth', 'Dip', 'Direction', 'Undercut_Width',
            'Rock_Type', 'Support_Type', 'Support_Density', 'Support_Installed'
        ]
        
        # Simple correlation-based importance
        static_importance = {}
        for i, name in enumerate(feature_names):
            if i < X_static.shape[1]:
                importance = np.abs(np.corrcoef(X_static[:, i], 
                                              np.mean(X_temporal, axis=(1,2)))[0, 1])
                static_importance[name] = importance if not np.isnan(importance) else 0.0
        
        # Temporal feature importance (average absolute values)
        temporal_features = ['Vibration', 'Deformation', 'Stress', 'Temperature', 'Humidity']
        temporal_importance = {}
        for i, name in enumerate(temporal_features):
            if i < X_temporal.shape[2]:
                importance = np.mean(np.abs(X_temporal[:, :, i]))
                temporal_importance[name] = importance
        
        # Plot feature importance
        self._plot_feature_importance(static_importance, temporal_importance, save_dir)
        
        return {
            'static_feature_importance': static_importance,
            'temporal_feature_importance': temporal_importance
        }
    
    def _evaluate_robustness(self, X_static, X_temporal, y_true, save_dir):
        """Evaluate model robustness through perturbation analysis."""
        
        # Noise robustness test
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        robustness_scores = []
        
        # Get baseline predictions
        baseline_pred, _, _ = self._generate_predictions(X_static, X_temporal)
        baseline_accuracy = accuracy_score(y_true, baseline_pred)
        
        for noise_level in noise_levels:
            # Add Gaussian noise to inputs
            X_static_noisy = X_static + np.random.normal(0, noise_level, X_static.shape)
            X_temporal_noisy = X_temporal + np.random.normal(0, noise_level, X_temporal.shape)
            
            # Generate predictions with noisy inputs
            noisy_pred, _, _ = self._generate_predictions(X_static_noisy, X_temporal_noisy)
            noisy_accuracy = accuracy_score(y_true, noisy_pred)
            
            # Calculate robustness score (how much accuracy drops)
            robustness_score = noisy_accuracy / baseline_accuracy
            robustness_scores.append(robustness_score)
        
        # Plot robustness analysis
        self._plot_robustness_analysis(noise_levels, robustness_scores, save_dir)
        
        return {
            'baseline_accuracy': baseline_accuracy,
            'noise_robustness': dict(zip(noise_levels, robustness_scores)),
            'average_robustness': np.mean(robustness_scores)
        }
    
    def _plot_confusion_matrix(self, cm, labels, save_dir, filename):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true, y_scores, save_dir, filename):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, y_true, y_scores, save_dir, filename):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, y_true, y_probs, save_dir, filename):
        """Plot calibration curve."""
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_horizon_performance(self, horizon_metrics, save_dir):
        """Plot performance vs prediction horizon."""
        horizons = [int(h[:-1]) for h in horizon_metrics.keys()]
        accuracies = [metrics['accuracy'] for metrics in horizon_metrics.values()]
        f1_scores = [metrics['f1_macro'] for metrics in horizon_metrics.values()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, accuracies, 'o-', label='Accuracy', linewidth=2)
        plt.plot(horizons, f1_scores, 's-', label='F1-Score (Macro)', linewidth=2)
        plt.xlabel('Prediction Horizon (days)')
        plt.ylabel('Performance Score')
        plt.title('Model Performance vs Prediction Horizon')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'horizon_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, static_importance, temporal_importance, save_dir):
        """Plot feature importance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Static features
        features = list(static_importance.keys())
        importances = list(static_importance.values())
        ax1.barh(features, importances)
        ax1.set_title('Static Feature Importance')
        ax1.set_xlabel('Importance Score')
        
        # Temporal features
        features = list(temporal_importance.keys())
        importances = list(temporal_importance.values())
        ax2.barh(features, importances)
        ax2.set_title('Temporal Feature Importance')
        ax2.set_xlabel('Average Magnitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_robustness_analysis(self, noise_levels, robustness_scores, save_dir):
        """Plot robustness analysis."""
        plt.figure(figsize=(8, 6))
        plt.plot(noise_levels, robustness_scores, 'o-', linewidth=2)
        plt.xlabel('Noise Level')
        plt.ylabel('Relative Accuracy')
        plt.title('Model Robustness to Input Noise')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_trend_accuracy(self, y_true, y_pred):
        """Calculate accuracy of risk trend predictions."""
        trend_accuracies = []
        
        for sample in range(len(y_true)):
            true_trend = np.diff(y_true[sample])
            pred_trend = np.diff(y_pred[sample])
            
            # Check if trends match in direction
            trend_match = np.sum(np.sign(true_trend) == np.sign(pred_trend))
            accuracy = trend_match / len(true_trend) if len(true_trend) > 0 else 0
            trend_accuracies.append(accuracy)
        
        return np.mean(trend_accuracies)
    
    def _calculate_decay_rate(self, accuracies):
        """Calculate performance decay rate over horizons."""
        if len(accuracies) < 2:
            return 0.0
        
        # Fit linear regression to calculate slope
        x = np.arange(len(accuracies))
        slope = np.polyfit(x, accuracies, 1)[0]
        return -slope  # Negative slope indicates decay
    
    def _generate_comprehensive_report(self, save_dir):
        """Generate comprehensive evaluation report."""
        report_path = os.path.join(save_dir, 'evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Dual-Branch Stability Predictor - Evaluation Report\\n")
            f.write("=" * 70 + "\\n\\n")
            f.write(f"**Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # Current Stability Performance
            f.write("## 📊 Current Stability Prediction Performance\\n\\n")
            current = self.results['current_stability']
            f.write(f"- **Accuracy**: {current['accuracy']:.3f}\\n")
            f.write(f"- **Precision**: {current['precision']:.3f}\\n")
            f.write(f"- **Recall**: {current['recall']:.3f}\\n")
            f.write(f"- **F1-Score**: {current['f1_score']:.3f}\\n")
            f.write(f"- **ROC-AUC**: {current['roc_auc']:.3f}\\n")
            f.write(f"- **Balanced Accuracy**: {current['balanced_accuracy']:.3f}\\n\\n")
            
            # Safety Metrics
            f.write("## 🛡️ Safety-Critical Metrics\\n\\n")
            safety = self.results['safety_analysis']
            f.write(f"- **Safety Recall**: {safety['safety_recall']:.3f}\\n")
            f.write(f"- **False Negative Rate**: {safety['false_negative_rate']:.3f}\\n")
            f.write(f"- **False Alarm Rate**: {safety['false_alarm_rate']:.3f}\\n")
            f.write(f"- **Critical Detection Rate**: {safety['critical_detection_rate']:.3f}\\n")
            f.write(f"- **Overall Safety Score**: {safety['safety_score']:.3f}\\n\\n")
            
            # Future Predictions
            f.write("## 🔮 Future Risk Prediction Performance\\n\\n")
            future = self.results['future_predictions']
            for horizon, metrics in future.items():
                f.write(f"### {horizon} Prediction\\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.3f}\\n")
                f.write(f"- F1-Score (Macro): {metrics['f1_macro']:.3f}\\n")
                f.write(f"- F1-Score (Weighted): {metrics['f1_weighted']:.3f}\\n\\n")
            
            # Temporal Analysis
            f.write("## ⏰ Temporal Analysis\\n\\n")
            temporal = self.results['temporal_analysis']
            f.write(f"- **Temporal Consistency**: {temporal['temporal_consistency']:.3f}\\n")
            f.write(f"- **Accuracy Decay Rate**: {temporal['accuracy_decay_rate']:.4f}/horizon\\n")
            f.write(f"- **Trend Accuracy**: {temporal['trend_accuracy']:.3f}\\n\\n")
            
            # Robustness
            f.write("## 💪 Model Robustness\\n\\n")
            robustness = self.results['robustness']
            f.write(f"- **Baseline Accuracy**: {robustness['baseline_accuracy']:.3f}\\n")
            f.write(f"- **Average Robustness**: {robustness['average_robustness']:.3f}\\n\\n")
            
            f.write("## 📁 Generated Visualizations\\n\\n")
            f.write("- Current stability confusion matrix\\n")
            f.write("- ROC and Precision-Recall curves\\n")
            f.write("- Calibration curves\\n")
            f.write("- Horizon performance analysis\\n")
            f.write("- Feature importance analysis\\n")
            f.write("- Robustness analysis\\n")

if __name__ == "__main__":
    # Example usage
    print("Enhanced Model Evaluation Script")
    print("================================")
    print()
    print("To use this script:")
    print("1. Ensure you have a trained Enhanced Dual-Branch model")
    print("2. Import the ModelEvaluator class")
    print("3. Run comprehensive evaluation:")
    print()
    print("   from model_evaluation import ModelEvaluator")
    print("   evaluator = ModelEvaluator(trained_model)")
    print("   results = evaluator.evaluate_comprehensive()")
    print()
    print("This will generate a comprehensive evaluation report")
    print("with all safety, operational, and technical metrics.")
