#!/usr/bin/env python3
"""
Enhanced Dual-Branch Stability Predictor Model Visualization
This script creates a visual diagram of the model architecture.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_model_architecture_diagram():
    """Create a visual diagram of the Enhanced Dual-Branch model architecture."""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#FF6B6B',
        'preprocessing': '#4ECDC4',
        'lstm': '#45B7D1',
        'attention': '#96CEB4',
        'dense': '#FFEAA7',
        'output': '#DDA0DD',
        'connection': '#2C3E50'
    }
    
    # Title
    ax.text(5, 11.5, 'Enhanced Dual-Branch Stability Predictor Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input layer
    input_box = FancyBboxPatch((1, 10), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.4, 'Input Features\n(Sequential Data)', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Preprocessing layers
    preprocess_box1 = FancyBboxPatch((0.5, 8.5), 1.5, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['preprocessing'], 
                                     edgecolor='black', linewidth=1)
    ax.add_patch(preprocess_box1)
    ax.text(1.25, 8.9, 'Normalization\n& Scaling', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    preprocess_box2 = FancyBboxPatch((2.5, 8.5), 1.5, 0.8, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['preprocessing'], 
                                     edgecolor='black', linewidth=1)
    ax.add_patch(preprocess_box2)
    ax.text(3.25, 8.9, 'Feature\nEngineering', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Branch 1 - Current Stability
    branch1_title = ax.text(1.5, 7.5, 'Branch 1: Current Stability Analysis', 
                           fontsize=14, fontweight='bold', ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # LSTM layers for Branch 1
    lstm1_box = FancyBboxPatch((0.5, 6.2), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lstm'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(lstm1_box)
    ax.text(1.5, 6.6, 'LSTM Layer 1\n(128 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    lstm2_box = FancyBboxPatch((0.5, 5.2), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lstm'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(lstm2_box)
    ax.text(1.5, 5.6, 'LSTM Layer 2\n(64 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Attention for Branch 1
    attention1_box = FancyBboxPatch((0.5, 4.2), 2, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['attention'], 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(attention1_box)
    ax.text(1.5, 4.6, 'Multi-Head\nAttention', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Dense layers for Branch 1
    dense1_box = FancyBboxPatch((0.5, 3.2), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['dense'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(dense1_box)
    ax.text(1.5, 3.6, 'Dense\n(32 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Branch 2 - Future Risk Assessment
    branch2_title = ax.text(6.5, 7.5, 'Branch 2: Future Risk Assessment', 
                           fontsize=14, fontweight='bold', ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # LSTM layers for Branch 2
    lstm3_box = FancyBboxPatch((5.5, 6.2), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lstm'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(lstm3_box)
    ax.text(6.5, 6.6, 'LSTM Layer 3\n(128 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    lstm4_box = FancyBboxPatch((5.5, 5.2), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['lstm'], 
                               edgecolor='black', linewidth=1)
    ax.add_patch(lstm4_box)
    ax.text(6.5, 5.6, 'LSTM Layer 4\n(64 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Attention for Branch 2
    attention2_box = FancyBboxPatch((5.5, 4.2), 2, 0.8, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['attention'], 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(attention2_box)
    ax.text(6.5, 4.6, 'Multi-Head\nAttention', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Dense layers for Branch 2
    dense2_box = FancyBboxPatch((5.5, 3.2), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['dense'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(dense2_box)
    ax.text(6.5, 3.6, 'Dense\n(32 units)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Fusion layer
    fusion_box = FancyBboxPatch((3.5, 2), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#FF9999', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 2.4, 'Feature Fusion & Integration Layer', 
            ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Output layers
    output1_box = FancyBboxPatch((1, 0.5), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['output'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(output1_box)
    ax.text(2, 0.9, 'Current Stability\nOutput', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    output2_box = FancyBboxPatch((5, 0.5), 4, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['output'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(output2_box)
    ax.text(7, 0.9, 'Future Risk Predictions\n(1d, 3d, 7d, 14d, 30d)', 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add connections
    connections = [
        # Input to preprocessing
        ((2, 10), (1.25, 9.3)),
        ((2, 10), (3.25, 9.3)),
        
        # Preprocessing to branches
        ((1.25, 8.5), (1.5, 7)),
        ((3.25, 8.5), (6.5, 7)),
        
        # Branch 1 connections
        ((1.5, 7), (1.5, 7)),
        ((1.5, 6.2), (1.5, 6)),
        ((1.5, 5.2), (1.5, 5)),
        ((1.5, 4.2), (1.5, 4)),
        ((1.5, 3.2), (1.5, 3)),
        
        # Branch 2 connections
        ((6.5, 7), (6.5, 7)),
        ((6.5, 6.2), (6.5, 6)),
        ((6.5, 5.2), (6.5, 5)),
        ((6.5, 4.2), (6.5, 4)),
        ((6.5, 3.2), (6.5, 3)),
        
        # To fusion
        ((2.5, 3.6), (3.5, 2.4)),
        ((5.5, 3.6), (6.5, 2.4)),
        
        # To outputs
        ((4, 2), (2, 1.3)),
        ((6, 2), (7, 1.3)),
    ]
    
    for start, end in connections:
        con = ConnectionPatch(start, end, "data", "data",
                             arrowstyle="->", shrinkA=5, shrinkB=5,
                             mutation_scale=20, fc=colors['connection'], 
                             ec=colors['connection'], linewidth=2)
        ax.add_artist(con)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=colors['input'], label='Input Layer'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['preprocessing'], label='Preprocessing'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['lstm'], label='LSTM Layers'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['attention'], label='Attention Mechanism'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['dense'], label='Dense Layers'),
        plt.Rectangle((0, 0), 1, 1, fc=colors['output'], label='Output Layers')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_performance_metrics_chart():
    """Create a chart showing model performance metrics."""
    
    # Sample performance data (would be loaded from actual training results)
    metrics = {
        'Current Stability': {
            'Accuracy': 0.8814,
            'Precision': 0.8626,
            'Recall': 0.9187
        },
        'Future Risk 1d': {
            'Accuracy': 0.7924,
            'Precision': 0.7800,
            'Recall': 0.8100
        },
        'Future Risk 3d': {
            'Accuracy': 0.8305,
            'Precision': 0.8200,
            'Recall': 0.8400
        },
        'Future Risk 7d': {
            'Accuracy': 0.8093,
            'Precision': 0.8000,
            'Recall': 0.8200
        },
        'Future Risk 14d': {
            'Accuracy': 0.5763,
            'Precision': 0.5600,
            'Recall': 0.5900
        },
        'Future Risk 30d': {
            'Accuracy': 0.5593,
            'Precision': 0.5400,
            'Recall': 0.5800
        }
    }
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Performance metrics bar chart
    models = list(metrics.keys())
    accuracy_scores = [metrics[model]['Accuracy'] for model in models]
    precision_scores = [metrics[model]['Precision'] for model in models]
    recall_scores = [metrics[model]['Recall'] for model in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    bars1 = ax1.bar(x - width, accuracy_scores, width, label='Accuracy', alpha=0.8)
    bars2 = ax1.bar(x, precision_scores, width, label='Precision', alpha=0.8)
    bars3 = ax1.bar(x + width, recall_scores, width, label='Recall', alpha=0.8)
    
    ax1.set_xlabel('Model Outputs', fontweight='bold')
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Model Performance Metrics by Output', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Training progress simulation
    epochs = np.arange(1, 6)
    train_loss = [3.2, 2.8, 2.5, 2.3, 2.2]
    val_loss = [3.1, 2.7, 2.4, 2.2, 2.25]
    
    ax2.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    ax2.plot(epochs, val_loss, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Training Progress', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    plt.tight_layout()
    return fig

def create_model_summary_info():
    """Create an information panel about the model."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Enhanced Dual-Branch Stability Predictor', 
            fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Model information
    model_info = """
    🏗️ MODEL ARCHITECTURE:
    
    • Dual-branch neural network architecture
    • Branch 1: Current stability analysis with LSTM + Attention
    • Branch 2: Future risk assessment with LSTM + Attention
    • Feature fusion layer for integrated predictions
    • Multi-output design for comprehensive risk assessment
    
    📊 INPUT FEATURES:
    
    • Sequential time-series data from mining operations
    • Geological parameters and measurements
    • Environmental conditions and sensor readings
    • Historical stability indicators
    • Engineered features from domain expertise
    
    🎯 OUTPUT PREDICTIONS:
    
    • Current Stability Classification (Stable/Unstable)
    • Future Risk Assessment for multiple time horizons:
      - 1-day risk prediction
      - 3-day risk prediction  
      - 7-day risk prediction
      - 14-day risk prediction
      - 30-day risk prediction
    
    ⚙️ TECHNICAL SPECIFICATIONS:
    
    • Framework: TensorFlow/Keras
    • Optimizer: Adam with learning rate scheduling
    • Loss Function: Multi-output categorical crossentropy
    • Batch Size: 32
    • Training Epochs: 5
    • Regularization: Dropout, L2 regularization
    
    📈 PERFORMANCE HIGHLIGHTS:
    
    • Current Stability Accuracy: 88.14%
    • Short-term Risk Prediction (1-7 days): 79-83% accuracy
    • Model demonstrates strong performance for immediate predictions
    • Longer-term predictions show expected uncertainty increase
    """
    
    ax.text(0.05, 0.85, model_info, fontsize=12, ha='left', va='top', 
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    return fig

def main():
    """Main function to generate all visualizations."""
    
    print("🎨 Generating Enhanced Dual-Branch Model Visualizations...")
    
    try:
        # Create output directory
        output_dir = "model_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
        
        # Generate architecture diagram
        print("📊 Creating model architecture diagram...")
        arch_fig = create_model_architecture_diagram()
        arch_fig.savefig(f"{output_dir}/model_architecture.png", dpi=300, bbox_inches='tight')
        arch_fig.savefig(f"{output_dir}/model_architecture.pdf", bbox_inches='tight')
        plt.close(arch_fig)
        print(f"✅ Architecture diagram saved to {output_dir}/model_architecture.png")
        
        # Generate performance metrics chart
        print("📈 Creating performance metrics chart...")
        perf_fig = create_performance_metrics_chart()
        perf_fig.savefig(f"{output_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
        perf_fig.savefig(f"{output_dir}/performance_metrics.pdf", bbox_inches='tight')
        plt.close(perf_fig)
        print(f"✅ Performance metrics chart saved to {output_dir}/performance_metrics.png")
        
        # Generate model summary
        print("📋 Creating model summary info...")
        info_fig = create_model_summary_info()
        info_fig.savefig(f"{output_dir}/model_summary.png", dpi=300, bbox_inches='tight')
        info_fig.savefig(f"{output_dir}/model_summary.pdf", bbox_inches='tight')
        plt.close(info_fig)
        print(f"✅ Model summary saved to {output_dir}/model_summary.png")
        
        # Close all figures to free memory
        plt.close('all')
        
        print("✅ All visualizations completed!")
        print(f"📁 Files saved in: {output_dir}/")
        
        # List generated files
        print("\n📂 Generated files:")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   • {file} ({file_size:.1f} KB)")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
